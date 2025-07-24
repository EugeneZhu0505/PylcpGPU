#%%
import numpy as np
import pylcp
import pylcp.fields as fields
from scipy.special import ellipk, ellipe
from scipy.spatial.transform import Rotation
# %%
class generate_atom():
    def __init__(self, J, A, B, C, T,kb,x0,t0,int_mass):
        self.natoms = 4
        v0 = np.sqrt(3 * kb * T / int_mass)
        v_0 = v0/x0*t0*100
        vx1_list = []
        vy1_list = []
        vz1_list = []
        for _ in range(self.natoms):
            numbers = np.random.rand(3)
            numbers /= numbers.sum()  # 归一化使其和为1
            numbers *= v_0  # 缩放使其和为总和
            vx1, vy1, vz1 = numbers
            vx1_list.append(vx1)
            vy1_list.append(vy1)
            vz1_list.append(vz1)
        vx = np.random.uniform(-1, 1, self.natoms) * np.array(vx1_list)
        vy = np.random.uniform(-1, 1, self.natoms) * np.array(vy1_list)
        vz = np.random.uniform(-1, 1, self.natoms) * np.array(vz1_list)
        self.v = np.array([vx, vy, vz]).T  # 转置以确保是 (natoms, 3) 的形状
        r0 =np.array([0.0, 0.0, -5.0])/x0
        rx = r0[0] + np.random.randn(self.natoms) * 0.5 
        ry = r0[1] + np.random.randn(self.natoms) * 0.5 
        rz = r0[2] + np.random.randn(self.natoms) * 0.5 
        self.r = np.array([rx, ry, rz]).T  # 转置以确保是 (natoms, 3) 的形状
        probabilities = np.zeros(24)
        probabilities[0:4] = 0.1
        probabilities[4] = 0.2
        probabilities[5:7] = 0.1
        probabilities[7] = 0.2 
        if len(probabilities) != 24:
            raise ValueError("概率列表的长度应为24。")
        N1 = []
        rho1 = []
        for _ in range(self.natoms):
            rand_num = np.random.rand()
            selected_position = np.argmax(np.cumsum(probabilities) > rand_num)
            N0 = np.zeros(24)
            N0[selected_position] = 1
            rho0=np.diag(N0)
            N1.append(N0)
            rho1.append(rho0)
        self.N = np.array(N1) 
        self.rho = np.array(rho1) 
             
#%%
class MOT_2DBeams(pylcp.fields.laserBeams):

    def __init__(self, ki=1, delta=0,i1=15,i2=2, pol=1, rotation_angles=[0., 0., 0.],
                 rotation_spec='XYZ', beam_type=pylcp.fields.laserBeam,pol_coord='spherical', **kwargs):
        super().__init__()
        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()
        kvecs = [np.array([ 1.,  0.,  0.]), np.array([-1.,  0.,  0.]),
                 np.array([ 0.,  1.,  0.]), np.array([ 0., -1.,  0.]),
                 np.array([ 0.,  0.,  1.])]
        ss=[i1,i1,i1,i1,i2]
        deltas=[delta,delta,delta,delta,delta]
        pols = [-pol,-pol,pol,pol,-pol]
        for kvec, pol,delta,s in zip(kvecs, pols,deltas,ss):
            self.add_laser(beam_type(kvec=rot_mat @ (ki*kvec), pol=pol, delta=delta,s=s,pol_coord='spherical',**kwargs))
 
class MOT2D_module():
    def __init__(self, t0_2D,atom,alpha_2D,mass_2D,g_2D,det_2D,po_2D,rotation_angles_2D,wb_2D,Ige_2D,Ire_2D,roffset_2D,voffset_2D,rscale_2D,vscale_2D,tmax_2D,sols_i_2D):
        self.alpha_D2 =alpha_2D
        self.atom=atom
        # The unitless mass parameter:
        self.mass=mass_2D
        # Gravity
        self.g=g_2D
        self.laserBeams = {}
        self.po=po_2D
        self.D2_tmax=tmax_2D
        self.roffset=roffset_2D
        self.voffset=voffset_2D
        self.rscale=rscale_2D
        self.vscale=vscale_2D
        self.t0=t0_2D
        magField = pylcp.fields.MOT2DMagneticField(alpha_2D)
        # Define the hamiltonians:
        H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
        muB=1)
        H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz,
        Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
        muB=1)
        dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
        atom.state[0].J, atom.state[2].J, atom.I)
        E_e_D2 = np.unique(np.diagonal(H_e_D2))
        E_g_D2 = np.unique(np.diagonal(0.05*H_g_D2))
        hamiltonian = pylcp.hamiltonian(0.05*H_g_D2,H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2,mass=self.mass)
        hamiltonian.mass=mass_2D
        laserBeams = {}
        laserBeams_cooling=MOT_2DBeams(delta=E_e_D2[-1]-E_g_D2[1]+det_2D,rotation_angles=rotation_angles_2D, pol=1, 
                                             beam_type=pylcp.fields.gaussianBeam,wb=wb_2D,i1=Ire_2D,i2=Ige_2D)
        laserBeams_repumping=MOT_2DBeams(delta=E_e_D2[1]-E_g_D2[0],rotation_angles=rotation_angles_2D, 
                                             beam_type=pylcp.fields.gaussianBeam,wb=wb_2D,i1=Ire_2D,i2=Ige_2D)
        laserBeams=laserBeams_cooling+laserBeams_repumping
        self.obe = pylcp.obe(laserBeams, magField, hamiltonian,self.g,transform_into_re_im=True)
        self.eqn = pylcp.rateeq(laserBeams, magField, hamiltonian, self.g)
        self.sols=sols_i_2D
        
    def generate_random_solution_2D(self, args, complete=True):
        if complete:
            return self.generate_random_solution_2D_eqn(args)
        else:
            return self.generate_random_solution_2D_obe(args)
    
       
    def generate_random_solution_2D_eqn(self,args):
        eqn,additional_param= args
        import numpy as np  
        args = ([0, self.D2_tmax], )
        kwargs = {'t_eval':np.linspace(0, self.D2_tmax, 1001),
                  'random_recoil':False,
                  'progress_bar':True,
                  'max_scatter_probability': 0.5,
                  'record_force':False}
        eqn.v0=self.sols[additional_param].v
        eqn.r0=self.sols[additional_param].r+5e4*np.random.randn(3)-self.po    
        eqn.N0=self.sols[additional_param].N 
        eqn.evolve_motion(*args, **kwargs)
        eqn.sol.r=eqn.sol.r+self.roffset
        return eqn.sol   
              
    def generate_random_solution_2D_obe(self,args):
        obe,additional_param= args
        import numpy as np  
        args = ([0, self.D2_tmax], )
        kwargs = {'t_eval':np.linspace(0, self.D2_tmax, 1001),
                  'random_recoil':False,
                  'progress_bar':True,
                  'max_scatter_probability': 0.5,
                  'record_force':False}
        obe.v0=self.sols[additional_param].v
        obe.r0=self.sols[additional_param].r+5e3*np.random.randn(3)-self.po    
        obe.rho0=(self.sols[additional_param].rho).flatten() 
        #obe.set_initial_rho_from_rateeq()
        obe.evolve_motion(*args, **kwargs)
        obe.sol.r=obe.sol.r+self.roffset
        return obe.sol        
    
                
class Relmagnetic():
    def __init__(self, II, R, O, n):
        self.II = II
        self.R = R
        self.O = O
        self.n = n
        self.defout = 500000  
        self.p = self.generate_p()
        
    def generate_points(self):
        beta = 5
        z = np.arange(-beta, beta + 0.25, 0.25) * 50000
        y = np.arange(-beta, beta + 0.25, 0.25) * 50000
        x = np.arange(-beta, beta + 0.25, 0.25) * 50000
        points = (x, y, z)
        return points
    
    def generate_p(self):
        beta = 5
        z = np.arange(-beta, beta + 0.25, 0.25) * 50000
        y = np.arange(-beta, beta + 0.25, 0.25) * 50000
        x = np.arange(-beta, beta + 0.25, 0.25) * 50000
        XX, YY, ZZ = np.meshgrid(x, y, z)
        xxx = XX.flatten()
        yyy = YY.flatten()
        zzz = ZZ.flatten()
        points = (x, y, z)
        p = np.column_stack((xxx, yyy, zzz))
        return p

    def magnetic_circle_combined(self, II, R, O, n):
        mu0 = 4 * np.pi * 1E-7
        pO = self.p - O
        x = np.dot(pO, n.T)
        r_v = pO - x.reshape(np.size(x), 1) * n
        r = np.linalg.norm(r_v, axis=1).reshape(np.size(x), 1)
        zero_indices = np.where(r == 0)[0]
        epsilon = 1e-19
        r[zero_indices] = epsilon
        alpha = (r / R).reshape(np.size(x), 1)
        beta = x / R
        Q = ((1 + alpha)**2) + beta**2
        Qa = Q - 4 * alpha
        M = 4 * alpha / Q
        K, E = ellipk(M**2), ellipe(M**2)    
        Bx = II * mu0 / (2 * R * np.pi * np.sqrt(Q)) * (E * (1 - alpha**2 - beta**2) / Qa + K)
        denominator = 2 * R * np.pi * np.sqrt(Q)
        if denominator.any() != 0:
            Br = II * mu0 / denominator * (x / r) * (E * (1 + alpha**2 + beta**2) / Qa - K)
        else:
            Br = II * mu0 / epsilon * (x / r) * (E * (1 + alpha**2 + beta**2) / Qa - K)
        B = Br / r   
        B_total = Bx * n + B * r_v
        indices = np.where(r < epsilon)[0]
        B_total[indices, :] += Bx[indices] * n
        indices = np.where(r < 0.000001)[0]
        B_total[indices, :] += Bx[indices] * n
        return B_total

    def calculate_combined_magnetic_field(self):
        # 计算两个环的磁场并相加#magnetic_circle_combined(II,R0,O0,n0,p)
        
        mp1 = self.magnetic_circle_combined(self.II * 3e2 * self.defout**2 / 1e-5,self.R * self.defout,self.O * self.defout,
                                           self.n.reshape(1, 3))

        mp2 = self.magnetic_circle_combined(self.II * 3e2 * self.defout**2 / 1e-5,self.R * self.defout,self.O *(-1)* self.defout,
                                           -1*self.n.reshape(1, 3))
        mp = mp1 + mp2
        return mp
    
class MOT3D_module():
    def __init__(self, alpha_3D,mass_3D,det_3D,Ige_3D, Ire_3D, atom,g_3D, rotation_angles_3D,wb_3D,sols_i_3,tmax_3D):
        self.atom = atom     
        self.Ige=Ige_3D
        self.Ire=Ire_3D
        self.wb=wb_3D
        self.alpha =alpha_3D
        self.tmax_3D=tmax_3D
        self.mass =mass_3D
        self.D3_rotation_angles=rotation_angles_3D
        self.det=det_3D
        self.g=g_3D
        self.sols_i_3D=sols_i_3
        hamiltonian,laserBeams = self.cooling_module(Complete=True)
        magField = pylcp.quadrupoleMagneticField(self.alpha)
        self.obe = pylcp.obe(laserBeams, magField, hamiltonian,self.g,
                        transform_into_re_im=True)
        self.eqn = pylcp.rateeq(laserBeams, magField, hamiltonian,self.g)
        
    def cooling_module(self,Complete=True):
        atom = self.atom 
        H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
        muB=1)
        H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz,
        Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
        muB=1)
        #H_e_D2=np.diag(np.hstack((np.diagonal(H_e_D2)[1:4],np.diagonal(H_e_D2)[9:16])))
        dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
        atom.state[0].J, atom.state[2].J, atom.I)
        E_e_D2 = np.unique(np.diagonal(H_e_D2))
        E_g_D2 = np.unique(np.diagonal(0.05*H_g_D2))
        hamiltonian = pylcp.hamiltonian(0.05*H_g_D2,H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2,mass=self.mass)
        
        laserBeams_cooling_D2 = pylcp.conventional3DMOTBeams(
            s=self.Ire, delta=E_e_D2[-1]-E_g_D2[1]+self.det, beam_type=pylcp.gaussianBeam,wb=self.wb,rotation_angles=self.D3_rotation_angles)
        laserBeams_repump_D2 = pylcp.conventional3DMOTBeams(
            s=self.Ige, delta=E_e_D2[1]-E_g_D2[0],beam_type=pylcp.gaussianBeam,wb=self.wb,rotation_angles=self.D3_rotation_angles)
        laserBeams=laserBeams_cooling_D2+laserBeams_repump_D2
        return hamiltonian,laserBeams

    def generate_random_solution_3D_obe(self,args):
        import numpy as np  # Add this import
        obe, additional_param = args
        sols_i=self.sols_i_3D
        args = ([0, self.tmax_3D], )
        kwargs2 = {'t_eval': np.linspace(0, self.tmax_3D, 1001),
                'random_recoil': True,
                'progress_bar': True,
                'max_scatter_probability': 0.5,
                'record_force': True}
        obe.v0=sols_i[additional_param].v[:,-1]
        obe.r0=sols_i[additional_param].r[:,-1]
        obe.rho0= (sols_i[additional_param].rho[:,:,-1]).flatten()   
        obe.evolve_motion(*args, **kwargs2)  
        return obe.sol 
    
    
    def generate_random_solution_3D_eqn(self,args):
        import numpy as np  # Add this import
        eqn, additional_param = args
        sols_i=self.sols_i_3D
        args = ([0, self.tmax_3D], )
        kwargs2 = {'t_eval': np.linspace(0, self.tmax_3D, 1001),
                'random_recoil': True,
                'progress_bar': True,
                'max_scatter_probability': 0.5,
                'record_force': True}
        eqn.v0=sols_i[additional_param].v[:,-1]
        eqn.r0=sols_i[additional_param].r[:,-1] 
        eqn.N0=sols_i[additional_param].N[:,-1]
        eqn.evolve_motion(*args, **kwargs2)  
        return eqn.sol 