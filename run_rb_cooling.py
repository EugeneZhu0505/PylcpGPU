import numpy as np
import pylcp
from pylcp.common import progressBar
import scipy.constants as cts
import pathos
from pylcp.integration_tools import RandomOdeResult
from functools import partial
import h5py
from MOT_ALL import MOT2D_module
from MOT_ALL import generate_atom



def prepare_args():
    atom = pylcp.atom("87Rb")               #原子的类型 87Rb 

    #固定值
    k = 2*np.pi/780E-7                      #波矢 单位cm^{-1} 
    x0 = 1/k                                #长度单位换算因子 换算后单位是cm
    gamma=atom.state[2].gammaHz             #原子自然线宽 单位Hz 
    t0 = 1/gamma                            #时间单位换算因子 换算后单位是s
    kb = 1.3806503E-23
    int_mass = 86.9*cts.value('atomic mass constant')

    #变量
    J = 1                                   #原子的密度  
    A = 4                                   #原子初始运动定义中的长度 
    B = 2                                   #原子初始运动定义中的宽度 
    C = 2                                   #原子初始运动定义中的高度 
    T = 300                                 #原子温度 

    #附值
    Initial_data = generate_atom(J, A, B, C, T,kb,x0,t0,int_mass)
    sols_r = Initial_data.r
    sols_v = Initial_data.v*0.01
    sols_N = Initial_data.N
    sols_rho = Initial_data.rho
    sols_t=np.zeros(Initial_data.natoms)

    # 创建一个sols0eqn.h5文件，用于存储读取的sols0(2D)数据
    t_list = sols_t
    r_list = sols_r
    v_list = sols_v
    N_list = sols_N
    rho_list = sols_rho

    with h5py.File('inti_solseqn.h5', 'w') as f:
        for i, (t, r, v, N) in enumerate(zip(t_list, r_list, v_list, N_list)):
            group = f.create_group(f'sol_{i}')
            group.create_dataset('t', data=t)
            group.create_dataset('r', data=r)
            group.create_dataset('v', data=v)
            group.create_dataset('N', data=N)

    #读取sols0的数据
    sols_0 = []
    with h5py.File('inti_solseqn.h5', 'r') as f:
        for key in f.keys():
            group = f[key]
            t = np.array(group['t'])
            r = np.array(group['r'])
            v = np.array(group['v'])
            N = np.array(group['N'])       
            sol = RandomOdeResult(t=t, r=r, v=v, N=N)
            sols_0.append(sol)

    with h5py.File('inti_solsobe.h5', 'w') as f:
        for i, (t, r, v, rho) in enumerate(zip(t_list, r_list, v_list, rho_list)):
            group = f.create_group(f'sol_{i}')
            group.create_dataset('t', data=t)
            group.create_dataset('r', data=r)
            group.create_dataset('v', data=v)
            group.create_dataset('rho', data=rho)

    sols_1 = []
    with h5py.File('inti_solsobe612.h5', 'r') as f:
        for key in f.keys():
            group = f[key]
            t = np.array(group['t'])
            r = np.array(group['r'])
            v = np.array(group['v'])
            rho = np.array(group['rho'])       
            sol = RandomOdeResult(t=t, r=r, v=v, rho=rho)
            sols_1.append(sol)




    #内置参数
    I_sat=1.6                               #饱和光强 单位mw/cm^2  
    #输入参数（2D）
    #变量
    atom = pylcp.atom("87Rb")               #原子的类型 87Rb 
    det_2D=-2                               #失谐 单位Hz 取值范围：[-100,0]
    wb_2D=5/x0                                  #光斑尺寸、光束直径     取值范围：[0,10]
    po_2D=np.array([0.,0.,-5.])/x0             #光阱中心位置 取值范围：[-50,0] 在这里我们是以Z轴为原子推送轴 
    roffset_2D = (np.array([0.0, 0.0, -10.0])/x0)[:, np.newaxis]   #原子初始位置补偿 取值范围：[-50,50]  
    voffset_2D = np.array([0.0, 0.0, 0.0])     #原子初始速度补偿 取值范围：[-5,5]
    rscale_2D = np.array([0.2, 0.2, 0.2]) /x0  #原子初始位置添加随机   取值范围：[-1,1]
    vscale_2D = np.array([0.1, 0.1, 0.1])      #原子初始速度添加随机   取值范围：[-1,1]
    t0_2D=0
    tmax_2D=0.000003/t0                         #2DMOT演化时间 取值范围：[0,1] 
    g_2D=-np.array([0.,9.8,0.])*t0**2/(x0*1e-2)                #重力加速度 单位m/s^2 在这里我们是以Y轴为重力轴
    Natoms_2D =4                             #原子个数  取值范围：[2,N]   
    chunksize_2D = 4                          #用于运算的核的个数  取值范围：[2,N] 
    rotation_angles_2D=[0., 0., 0.]         #2D光场旋转角度  取值范围：[0,2*np.pi] 
    Ige_2D=2/I_sat                                  #泵浦光的光强 单位mW/cm²  取值范围：[0,1]
    Ire_2D=15/I_sat                                  #冷却光的光强 单位mW/cm²  取值范围：[0,16]
    #内置参数
    I_sat=1.6                               #饱和光强 单位mw/cm^2  
    alpha_2D =(3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/gamma*2  #磁场参数 
    mass_2D = 86.9*cts.value('atomic mass constant')*(x0*1e-2)**2/cts.hbar/t0 #原子质量:86.9*cts.value('atomic mass constant')
    
    # 返回包含所有参数的字典
    args = {
        't0_2D': t0_2D,
        'atom': atom,
        'alpha_2D': alpha_2D,
        'mass_2D': mass_2D,
        'g_2D': g_2D,
        'det_2D': det_2D,
        'po_2D': po_2D,
        'rotation_angles_2D': rotation_angles_2D,
        'wb_2D': wb_2D,
        'Ige_2D': Ige_2D,
        'Ire_2D': Ire_2D,
        'roffset_2D': roffset_2D,
        'voffset_2D': voffset_2D,
        'rscale_2D': rscale_2D,
        'vscale_2D': vscale_2D,
        'tmax_2D': tmax_2D,
        'sols_i_2D': sols_i_2D,
        'Natoms_2D': Natoms_2D,
        'chunksize_2D': chunksize_2D,
        'sols_0': sols_0,
        'sols_1': sols_1,
    }
    
    return args




if __name__ == "__main__":

    args = prepare_args()

    MOT2D_test=MOT2D_module(args['t0_2D'],
                            args['atom'],
                            args['alpha_2D'],
                            args['mass_2D'],
                            args['g_2D'],
                            args['det_2D'],
                            args['po_2D'],
                            args['rotation_angles_2D'],
                            args['wb_2D'],
                            args['Ige_2D'],
                            args['Ire_2D'],
                            args['roffset_2D'],
                            args['voffset_2D'],
                            args['rscale_2D'],
                            args['vscale_2D'],
                            args['tmax_2D'],
                            args['sols_1'])  
    sol_range = np.arange(args['Natoms_2D']).reshape((int(args['Natoms_2D']/args['chunksize_2D']), args['chunksize_2D']))
    progress = progressBar()

    sols = []
    for jj in range(int(args['Natoms_2D']/args['chunksize_2D'])):
        with pathos.pools.ProcessPool(nodes=4) as pool:
            arg_list = [(MOT2D_test.obe, idx) for idx in sol_range[jj,:]]
            partial_function =  partial(MOT2D_test.generate_random_solution_2D_obe)
            sols += pool.map(partial_function, arg_list)
        progress.update((jj+1)/int(args['Natoms_2D']/args['chunksize_2D']))


        #     arg_list = [(MOT2D_test.eqn,idx) for idx in sol_range[jj,:]]
        #     partial_function =  partial(MOT2D_test.generate_random_solution_2D_eqn)
        #     sols0 += pool.map(partial_function, arg_list)
        # progress.update((jj+1)/int(Natoms_2D/chunksize_2D))
    

    with h5py.File('sol2Dobe.h5', 'w') as f:
        for i, sol in enumerate(sols):
            group = f.create_group(f'sol_{i}')
            group.create_dataset('t', data=sol[i].t)
            group.create_dataset('r', data=sol[i].r)
            group.create_dataset('v', data=sol[i].v)
            group.create_dataset('rho', data=sol[i].rho)