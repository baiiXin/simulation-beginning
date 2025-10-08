import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, bicgstab
import time

# 直接法
#x = spsolve(A_sparse, b)
# 共轭梯度法（适合对称正定矩阵）
#x_cg, info = cg(A_sparse, b)
# BiCGSTAB（适合非对称或不规则矩阵）
#x_bi, info = bicgstab(A_sparse, b)

class Spring:
    def __init__(self, num=None, ele=None, stiff_k=None, rest_len=None):
        self.num = num # 弹簧数量；1
        self.ele = ele # 弹簧连接的质点编号；[[0, 1], [1, 2], [2, 3], [3, 4]]
        self.rest_len = rest_len # 弹簧的初始长度；[1.0, 1.0, 1.0, 1.0]
        self.stiff_k = stiff_k  # 弹簧的刚度；1


class Mass:
    def __init__(self, num=int, 
                 pos=None, vel=None, pos_hat=None, vel_hat=None,
                 ele=None, mass=None, 
                 force=None, Hessian=None, Mass_k=None,
                 dump=None, gravity=None, Spring=Spring, dt=None, 
                 tolerance_newton=None):
        self.num = num # 质点数量；1
        self.ele = ele # 三角元；[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        self.pos = pos # 质点位置；[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        self.vel = vel # 质点速度；[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.pos_hat = pos_hat # 预测质点位置
        self.vel_hat = vel_hat # 预测质点速度
        self.force = force # 力向量--牛顿迭代--线性方程组--b
        self.Hessian = Hessian # 矩阵--牛顿迭代--线性方程组--A
        self.mass = mass # 质点质量；1
        self.dump = dump # 阻尼系数；1
        self.gravity = gravity # 重力加速度；9.8
        self.Spring = Spring # 弹簧
        self.dt = dt # 时间步长
        self.tolerance_newton = tolerance_newton # 牛顿迭代的容差

        # fixed points
        # 初始化
        self.fixed_idx = [36, 44] #[72, 80] #[0, 8] #[36, 44]
        self.all_idx = np.arange(self.num)
        self.free_idx = np.setdiff1d(self.all_idx, self.fixed_idx)
        free_idx = np.array(self.free_idx)
        dof_matrix = 3 * free_idx[:, np.newaxis] + np.arange(3)
        self.free_dof = dof_matrix.flatten()

    def Single_Newton_Method(self, Spring: Spring, fixed_num, ite_num, space_dim=3):
        '''
        功能:  牛顿迭代, 一个时间步
        input: 质点, 弹簧, 迭代步, 固定点数量
        output: 
            1. 直接修改mss类的位置和速度
            2. 绘制收敛曲线的参数:
                Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm
        '''
        # Newton Method (Implicit Euler)
        # 计时
        times_ms = []
        # 残差曲线
        Newton_step = []
        Error_dx_norm = []
        Residual_norm = []
        Energy_norm = []
        
        # 迭代初值
        self.pos_hat = self.pos.copy()

        # 迭代
        for times in range(ite_num):
            # 计时
            start_time = time.time()

            # 组装
            self.assemebel_HF(Spring)

            # 注意：只求解自由部分
            A_free = self.Hessian[np.ix_(self.free_dof, self.free_dof)]
            b_free = self.force[self.free_dof]
            
            # 矩阵稀疏化
            A_sparse = csr_matrix(A_free)
            
            # 计算dX
            dX_free, info = cg(A_sparse, b_free)
            # print('cg:',info)

            # 更新位置（只更新非固定点）
            pos_new = self.pos_hat.copy()
            # 将 x_free 重新写回到 pos_new 对应的自由点
            for i, p in enumerate(self.free_idx):
                pos_new[p] += dX_free[3*i : 3*i+3]
            self.pos_hat = pos_new.copy()

            # 组装时间
            end_time = time.time()  # 结束时间
            times_ms.append((end_time - start_time) * 1000)  # 计算并存储运行时间（毫秒）

            # 计算误差
            error_dx_norm = np.linalg.norm(dX_free)

            # 计算能量
            Energy = self.Energy_compute(Spring)
            
            # 计算残差
            residual_norm = self.Residual_compute(Spring)

            # 记录残差
            Newton_step.append(times)
            Error_dx_norm.append(error_dx_norm)
            Residual_norm.append(residual_norm)
            Energy_norm.append(Energy)

            # 打印残差
            print('\nNewton iterative step: ', times)
            print('times_ms = ', (end_time - start_time) * 1000)
            print('error_dx_norm =', error_dx_norm)
            print('residual_norm = ', residual_norm, residual_norm.dtype)
            print('Enegy = ', Energy)
            
            # 如果误差足够小，提前结束迭代
            if error_dx_norm < self.tolerance_newton:
                break
        
        # 更新位置和速度
        self.vel = (self.pos_hat - self.pos) / self.dt * self.dump
        self.pos = self.pos_hat.copy()

        return Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm
        
    def assemebel_HF(self, Spring: Spring):
        '''
        input: 质点＋弹簧
        output: 直接修改 self.Hessian 和 self.force; 
                不是实际的force和hessian; 对应牛顿迭代的A和b
        '''
        # 获取质点数量和空间维度
        Nm = self.num  # 质点数量
        space_dim = 3  # 空间维度，应该是3
        NS = Spring.num  # 弹簧数量
        
        # 初始化矩阵和向量
        all_points = Nm * space_dim
        F = np.zeros(all_points)  # 力向量
        
        I = np.zeros((all_points, all_points))  # 质量矩阵
        H = np.zeros((all_points, all_points))  # 海森矩阵
        
        f = np.zeros(all_points)  # 弹簧力
        g = np.zeros(all_points)  # 重力
        
        # 循环计算弹簧力和海森矩阵
        for i in range(NS): 
            a = Spring.ele[i][0]  # 弹簧连接的第一个质点
            b = Spring.ele[i][1]  # 弹簧连接的第二个质点
            x_ab = self.pos_hat[a] - self.pos_hat[b]  # 两点之间的向量
            x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
            
            # 计算弹簧力
            f_spring = -Spring.stiff_k * (x_ab/x_ab_norm) * (x_ab_norm - Spring.rest_len[i])
            
            # 计算海森矩阵
            h_spring = Spring.stiff_k * np.outer(x_ab, x_ab) / (x_ab_norm**2) + \
                    Spring.stiff_k * (1 - Spring.rest_len[i]/x_ab_norm) * \
                    (np.eye(space_dim) - np.outer(x_ab, x_ab)/(x_ab_norm**2))
            
            # 更新力向量
            f[(a*space_dim):(a*space_dim+space_dim)] += f_spring
            f[(b*space_dim):(b*space_dim+space_dim)] -= f_spring
            
            # 更新海森矩阵
            H[(a*space_dim):(a*space_dim+space_dim), (a*space_dim):(a*space_dim+space_dim)] += h_spring
            H[(a*space_dim):(a*space_dim+space_dim), (b*space_dim):(b*space_dim+space_dim)] -= h_spring
            H[(b*space_dim):(b*space_dim+space_dim), (a*space_dim):(a*space_dim+space_dim)] -= h_spring
            H[(b*space_dim):(b*space_dim+space_dim), (b*space_dim):(b*space_dim+space_dim)] += h_spring
        
        # 计算质点的重力和质量矩阵
        for j in range(Nm):
            g_vec = np.zeros(space_dim)
            g_vec[space_dim-1] = -self.mass * self.gravity  # 在z方向上施加重力
            g[(j*space_dim):(j*space_dim+space_dim)] = g_vec
            
            # 质量/单位矩阵
            I_eyes = np.eye(space_dim)
            I[(j*space_dim):(j*space_dim+space_dim), (j*space_dim):(j*space_dim+space_dim)] = I_eyes
            
            # 计算F向量
            F[(j*space_dim):(j*space_dim+space_dim)] = self.pos_hat[j] - self.pos[j] - self.dt * self.vel[j]
        
        # 返回计算结果
        self.force = -1/self.dt/self.dt *self.mass * F + f + g  # 计算b向量
        self.Hessian = 1/self.dt/self.dt *self.mass * I + H  # 计算A矩阵

    def Energy_compute(self, Spring: Spring):   
        '''
        功能:  计算能量
        input: 质点, 弹簧
        output: Energy
        '''          
        # 获取质点数量和空间维度
        Nm = self.num  # 质点数量
        space_dim = 3  # 空间维度，应该是3
        NS = Spring.num  # 弹簧数量
        
        # 初始化矩阵和向量
        all_points = Nm * space_dim
        norm_F_vec = np.zeros(all_points)  
        Energy_E = 0.0
        Energy_F = 0.0
        Energy_G = 0.0

        # 0重力势能
        # 计算质点的重力和质量矩阵
        for j in range(Nm):
            # 计算能量
            Energy_G  +=  self.mass * self.gravity * self.pos_hat[j][2]
        
            # 计算F向量
            norm_F_vec[(j*space_dim):(j*space_dim+space_dim)] = self.pos_hat[j] - self.pos[j] - self.dt * self.vel[j]
        # 计算能量
        Energy_F +=  (1.0/ (2.0 * self.dt * self.dt)) * self.mass * np.linalg.norm(norm_F_vec) * np.linalg.norm(norm_F_vec)

        # 循环计算弹簧力和海森矩阵
        for i in range(NS): 
            a = Spring.ele[i][0]  # 弹簧连接的第一个质点
            b = Spring.ele[i][1]  # 弹簧连接的第二个质点
            x_ab = self.pos_hat[a] - self.pos_hat[b]  # 两点之间的向量
            x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
            # 计算弹簧力
            Energy_E += 0.5 * Spring.stiff_k * (x_ab_norm - Spring.rest_len[i]) * (x_ab_norm - Spring.rest_len[i])
        # 计算能量
        Energy = Energy_E + Energy_F + Energy_G
        return Energy

    def Residual_compute(self, Spring: Spring):   
        '''
        input: 质点＋弹簧
        output: 直接修改 self.Hessian 和 self.force; 
                不是实际的force和hessian; 对应牛顿迭代的A和b
        '''
        # 获取质点数量和空间维度
        Nm = self.num  # 质点数量
        space_dim = 3  # 空间维度，应该是3
        NS = Spring.num  # 弹簧数量
        
        # 初始化矩阵和向量
        all_points = Nm * space_dim
        F = np.zeros(all_points)  # 力向量
        
        I = np.zeros((all_points, all_points))  # 质量矩阵
        
        f = np.zeros(all_points)  # 弹簧力
        g = np.zeros(all_points)  # 重力
        
        # 循环计算弹簧力和海森矩阵
        for i in range(NS): 
            a = Spring.ele[i][0]  # 弹簧连接的第一个质点
            b = Spring.ele[i][1]  # 弹簧连接的第二个质点
            x_ab = self.pos_hat[a] - self.pos_hat[b]  # 两点之间的向量
            x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
            
            # 计算弹簧力
            f_spring = -Spring.stiff_k * (x_ab/x_ab_norm) * (x_ab_norm - Spring.rest_len[i])
            
            # 更新力向量
            f[(a*space_dim):(a*space_dim+space_dim)] += f_spring
            f[(b*space_dim):(b*space_dim+space_dim)] -= f_spring
        
        # 计算质点的重力和质量矩阵
        for j in range(Nm):
            g_vec = np.zeros(space_dim)
            g_vec[space_dim-1] = -self.mass * self.gravity  # 在z方向上施加重力
            g[(j*space_dim):(j*space_dim+space_dim)] = g_vec
            
            # 质量/单位矩阵
            I_eyes = np.eye(space_dim)
            I[(j*space_dim):(j*space_dim+space_dim), (j*space_dim):(j*space_dim+space_dim)] = I_eyes
            
            # 计算F向量
            F[(j*space_dim):(j*space_dim+space_dim)] = self.pos_hat[j] - self.pos[j] - self.dt * self.vel[j]
        
        # 返回计算结果
        residual = F - self.dt * self.dt * (1/self.mass) * (f+g)  # 计算b向量
        residual_norm = np.linalg.norm(residual[self.free_dof])
        return residual_norm
