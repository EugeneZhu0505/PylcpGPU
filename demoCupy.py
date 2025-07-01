
import pylcp
import cupy as cp
import scipy.constants as cts
import pylcp.obeCupy
import time
import h5py

from config import coolingArgs
from scipy.spatial.transform import Rotation


class MOT2DBeams(pylcp.fields.laserBeams):

    def __init__(self, ki=1, delta=0,i1=15,i2=2, pol=1, rotation_angles=[0., 0., 0.],
                 rotation_spec='XYZ', beam_type=pylcp.fields.laserBeam,pol_coord='spherical', **kwargs):
        super().__init__()
        rot_mat = cp.asarray(Rotation.from_euler(rotation_spec, cp.asnumpy(rotation_angles)).as_matrix())
        kvecs = [cp.array([ 1.,  0.,  0.]), cp.array([-1.,  0.,  0.]),
                 cp.array([ 0.,  1.,  0.]), cp.array([ 0., -1.,  0.]),
                 cp.array([ 0.,  0.,  1.])]
        ss=[i1, i1, i1, i1, i2]
        deltas=[delta, delta, delta, delta, delta]
        pols = [-pol, -pol, pol, pol, -pol]
        for kvec, pol,delta,s in zip(kvecs, pols, deltas, ss):
            self.add_laser(beam_type(kvec=rot_mat @ (ki * kvec), pol=pol, delta=delta, s=s, pol_coord='spherical', **kwargs))





class CoolingModule:
    def __init__(self):

        self.args = {}

        pass

    def prepareArgs(self, coolingArgs):
            
        atom = pylcp.atom(coolingArgs['atom']) # 原子类

        # 固定数值的参数
        k = 2 * cp.pi # 波矢
        x0 = 1 / k # 长度单位换算因子
        gamma = atom.state[2].gammaHz # 原子自然线宽
        t0 = 1 / gamma # 时间单位换算因子
        Isat = 1.6

        # 预处理
        numAtom = coolingArgs['num_atom'] # 原子个数
        po2D = cp.array(coolingArgs['po_2d']) / x0
        roffset2D = cp.array(coolingArgs['roffset_2d']) / x0
        # 加载初始化数据
        r0, v0, rho0 = [], [], []
        with h5py.File('E:\BaiduSyncdisk\Master\Project\Code\AtomEvolve\ZGD\Rb\inti_solsobe612.h5', 'r') as f:
            for key in f.keys():
                group = f[key]
                # t = np.array(group['t'])
                r0.append(cp.array(group['r']))
                v0.append(cp.array(group['v']))
                rho0.append(cp.array(group['rho']).flatten())
        r0 = cp.array(r0).T
        v0 = cp.array(v0).T
        rho0 = cp.array(rho0).T
        det2D = coolingArgs['det_2d']
        wb2D = coolingArgs['wb_2d']
        tmax2D = coolingArgs['tmax_2d']
        numPoints = coolingArgs['num_points']
        maxScatterProbability = coolingArgs['max_scatter_probability']
        g = cp.array([0., -9.8, 0.]) * t0 ** 2 / (x0 * 1e-2)
        randomRecoilFlag = coolingArgs['random_recoil_flag']
        rotationAngles2D = cp.array(coolingArgs['rotation_angles_2d'])
        Ige2D = coolingArgs['Ige_2d'] / Isat
        Ire2D = coolingArgs['Ire_2d'] / Isat
        alpha2D = (3/2) * cts.value('Bohr magneton in Hz/T') * 1e-4 * 8 * x0 / gamma * 2
        mass = 86.9 * cts.value('atomic mass constant') * (x0 * 1e-2)**2 / cts.hbar / t0

        self.args.update(
            {
                "atom": atom,
                "r0": r0,
                "v0": v0,
                "rho0": rho0,
                "x0": x0,
                "g": g,
                "numAtom": numAtom,
                "alpha2D": alpha2D,
                "mass": mass,
                "Ige2D": Ige2D,
                "Ire2D": Ire2D,
                "det2D": det2D,
                "wb2D": wb2D,
                "rotationAngles2D": rotationAngles2D,
                "tmax2D": tmax2D,
                "numPoints": numPoints,
                "po2D": po2D,
                "roffset2D": roffset2D,
                "randomRecoilFlag": randomRecoilFlag,
                "maxScatterProbability": maxScatterProbability,
            }
        )



        pass
        


    def buildField(self):

        atom = self.args['atom']
        H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
            atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
            atom.state[0].Ahfs / atom.state[2].gammaHz,
            Bhfs=0, Chfs=0, muB=1)
        H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
            atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
            Ahfs=atom.state[2].Ahfs / atom.state[2].gammaHz,
            Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz,
            Chfs=0, muB=1)
        dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
            atom.state[0].J, atom.state[2].J, atom.I)
        E_e_D2 = cp.unique(cp.diagonal(H_e_D2))
        E_g_D2 = cp.unique(cp.diagonal(0.05 * H_g_D2))
        self.hamiltonian2D = pylcp.hamiltonian(0.05 * H_g_D2, H_e_D2,
                                          mu_q_g_D2, mu_q_e_D2, dijq_D2,
                                          mass=self.args['mass'])
        self.hamiltonian2D.mass = self.args['mass']
        laserBeamsCooling = MOT2DBeams(delta=E_e_D2[-1]-E_g_D2[1]+self.args['det2D'],
                                        rotation_angles=cp.array(self.args['rotationAngles2D']), pol=1,
                                        beam_type=pylcp.fields.gaussianBeam,
                                        wb=self.args['wb2D'], i1=self.args['Ire2D'], i2=self.args['Ige2D'])
        laserBeamsRepumping = MOT2DBeams(delta=E_e_D2[1]-E_g_D2[0],
                                          rotation_angles=cp.array(self.args['rotationAngles2D']),
                                          beam_type=pylcp.fields.gaussianBeam,
                                          wb=self.args['wb2D'], i1=self.args['Ire2D'], i2=self.args['Ige2D'])
        self.laserBeams2D = laserBeamsCooling + laserBeamsRepumping
        self.magField2D = pylcp.fields.MOT2DMagneticField(self.args['alpha2D'])

        pass

    def evolve2D(self):

        self.args['r0'] -= self.args['po2D'][:, cp.newaxis]
        tSpan = cp.array([0., self.args['tmax2D']])
        tEval = cp.linspace(tSpan[0], tSpan[1], self.args['numPoints'])
        
        print("initializing OBE")
        startTime = time.time()
        self.args['r0'] -= self.args['po2D'][:, cp.newaxis]
        obe = pylcp.obeCupy.Obe(
            laserBeams=self.laserBeams2D,
            magField=self.magField2D,
            hamiltonian=self.hamiltonian2D,
            a=self.args['g'],
            r0=self.args['r0'],
            v0=self.args['v0'],
            rho0=self.args['rho0'],
        )
        print("OBE initialized in {} s".format(time.time() - startTime))

        print("evolving OBE")
        sol = obe.evolve_motion(
            tSpan=tSpan,
            # random_recoil=self.args['randomRecoilFlag'],
            # max_scatter_probability=self.args['maxScatterProbability'],
            tEval=tEval,
            progressBarFlag=True
        )

        with h5py.File("./sol2D.h5", 'w') as f:
            group = f.create_group('sol')
            for key in sol.keys():
                group.create_dataset(name=key, data=sol[key])



        pass


if __name__ == "__main__":

    # 初始化冷却模块
    coolingModule = CoolingModule()
    # 预处理（参数和数据）
    print("prepare args")
    coolingModule.prepareArgs(coolingArgs)
    # 构建场
    print("build field")
    coolingModule.buildField()
    # 进行演化
    print("evolve 2D")
    coolingModule.evolve2D()