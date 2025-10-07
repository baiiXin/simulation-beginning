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

            # 计算能量
            Energy = self.Energy_compute(Spring)
            # 组装
            self.assemebel_HF(Spring)

            # 注意：只求解自由部分
            A_free = -self.Hessian[np.ix_(self.free_dof, self.free_dof)]
            b_free = self.force[self.free_dof]
        
            # fixed_num
            #A, b = self.boundary_solve(fixed_num, space_dim)
            
            # 矩阵稀疏化
            A_sparse = csr_matrix(A_free)
            
            # 计算dX
            dX_free, info = cg(A_sparse, b_free)
            # print('cg:',info)
            '''
            # 更新位置（只更新非固定点）
            for j in range(fixed_num, self.num):
                idx = (j-fixed_num)*space_dim
                self.pos_hat[j] += dX[idx:idx+space_dim]
            self.vel_hat = (self.pos_hat - self.pos) / self.dt * self.dump
            '''
            pos_new = self.pos_hat.copy()
            # 将 x_free 重新写回到 pos_new 对应的自由点
            for i, p in enumerate(self.free_idx):
                pos_new[p] += dX_free[3*i : 3*i+3]
            self.pos_hat = pos_new.copy()
            '''
            # 更新位置（只更新非固定点）
            for j in range(self.num):
                self.pos_hat[j] += dX[j:j+space_dim]
            self.vel_hat = (self.pos_hat - self.pos) / self.dt * self.dump
            print('dX', dX.shape, '\n', dX)
            '''
            # 组装时间
            end_time = time.time()  # 结束时间
            times_ms.append((end_time - start_time) * 1000)  # 计算并存储运行时间（毫秒）

            # 计算误差
            error_dx_norm = np.linalg.norm(dX_free)
            
            # 计算残差
            residual = b_free  
            residual_norm = np.linalg.norm(residual)

            # 记录残差
            Newton_step.append(times)
            Error_dx_norm.append(error_dx_norm)
            Residual_norm.append(residual_norm)
            Energy_norm.append(Energy)

            # 打印残差
            print('Newton Method step: ', times)
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

        # 碰撞检测和响应
        self.Sphere_Collision(ball_c=np.array([0.5, -0.5, 6]), ball_r=2.50, u_N=0.1, u_T=0.45, method=2)

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
            h_spring = -Spring.stiff_k * np.outer(x_ab, x_ab) / (x_ab_norm**2) - \
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
        self.force = F - self.dt * self.dt * (1/self.mass) * (f+g)  # 计算b向量
        self.Hessian = I - self.dt * self.dt * (1/self.mass) *H  # 计算A矩阵

    def Sphere_Collision(self, ball_c, ball_r, u_N, u_T, method):
        '''
        功能:  碰撞检测和响应
        input: 质点, 碰撞球的位置和半径, 碰撞系数, 碰撞检测方法
        output: 直接修改mss类的位置和速度
        '''
        if method == 1:
            # 碰撞检测
            for j in range(self.num):
                # 计算球心到质点的单位向量
                r_vec = self.pos[j] - ball_c
                r_norm = np.linalg.norm(r_vec)

                # 检查是否发生碰撞
                if r_norm < ball_r:
                    # 计算碰撞后的速度
                    dvel = 1/self.dt *(ball_c + ball_r * r_vec / r_norm - self.pos[j])
                    self.vel[j] += dvel 

                    # 计算碰撞后的位置
                    self.pos[j] = ball_c + ball_r * r_vec / r_norm

        elif method == 2:
            # 碰撞检测        
            for j in range(self.num):
                # 计算球心到质点的单位向量
                r_vec = self.pos[j] - ball_c
                r_norm = np.linalg.norm(r_vec)
                r_vec_n = r_vec / r_norm
                phi_x = r_norm - ball_r

                # 检查是否发生碰撞
                if phi_x < 0:
                    # 计算碰撞后的位置
                    self.pos[j] += np.linalg.norm(phi_x) * r_vec_n

                    # 速度分解
                    vel_norm = self.vel[j] @ r_vec_n
                    vel_N = vel_norm * r_vec_n
                    vel_T = self.vel[j] - vel_N
                    # 碰撞系数
                    alpha1 = 1.0 - u_T * ( 1.0 + u_N ) * np.linalg.norm(vel_N) / np.linalg.norm(vel_T)
                    alpha = max (0, alpha1)
                    # 碰撞后的速度
                    vel_N *= -u_N 
                    vel_T *= alpha 
                    self.vel[j] = vel_N + vel_T

                elif self.pos[j][2] < 0: 
                    # 计算地面到质点的单位向量
                    r_norm = np.linalg.norm(self.pos[j][2])
                    r_vec_n = np.array([0, 0, 1])

                    # 计算碰撞后的位置
                    self.pos[j] += r_norm * r_vec_n

                    # 速度分解
                    vel_norm = self.vel[j] @ r_vec_n
                    vel_N = vel_norm * r_vec_n
                    vel_T = self.vel[j] - vel_N
                    # 碰撞系数
                    alpha1 = 1.0 - u_T * ( 1.0 + u_N ) * np.linalg.norm(vel_N) / np.linalg.norm(vel_T)
                    alpha = max (0, alpha1)
                    # 碰撞后的速度
                    vel_N *= -u_N 
                    vel_T *= alpha 
                    self.vel[j] = vel_N + vel_T

    def Energy_compute(self, Spring: Spring, method=0):   

        if method == 0:
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

        elif method == 1:
            '''
            功能:  计算能量
            input: 质点, 弹簧
            output: Energy
            '''          
            # 获取质点数量和空间维度
            Nm = self.num  # 质点数量
            NS = Spring.num  # 弹簧数量
            
            # 初始化矩阵和向量
            Energy_E = 0.0
            Energy_F = 0.0
            
            # 计算质点的重力和质量矩阵
            for j in range(Nm):
                # 计算F向量
                norm_F_vec = self.pos_hat[j] - self.pos[j] - self.dt * self.vel[j]
                # 计算能量
                Energy_F +=  (1.0/ 2.0) * np.linalg.norm(norm_F_vec) * np.linalg.norm(norm_F_vec)

            # 循环计算弹簧力和海森矩阵
            for i in range(NS): 
                a = Spring.ele[i][0]  # 弹簧连接的第一个质点
                b = Spring.ele[i][1]  # 弹簧连接的第二个质点
                x_ab = self.pos_hat[a] - self.pos_hat[b]  # 两点之间的向量
                x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
                
                # 计算弹簧力
                Energy_E += self.dt * self.dt * (1/self.mass) * 0.5 * Spring.stiff_k * (x_ab_norm - Spring.rest_len[i]) * (x_ab_norm - Spring.rest_len[i])
            
            Energy = Energy_E + Energy_F

            return Energy

        elif method == 2:
            '''
            功能:  计算能量
            input: 质点, 弹簧
            output: Energy
            '''          
            # 获取质点数量和空间维度
            Nm = self.num  # 质点数量
            NS = Spring.num  # 弹簧数量
            
            # 初始化矩阵和向量
            Energy_G0 = 0.0
            Energy_G = 0.0
            Energy_V = 0.0
            Energy_E = 0.0

            # 0重力势能
            for j in range(Nm):
                # 计算能量
                Energy_G0 +=  self.mass * self.gravity * self.pos[j][2]
                Energy_G  +=  self.mass * self.gravity * self.pos_hat[j][2]
                Energy_V  +=  0.5 * self.mass * np.linalg.norm(self.vel_hat[j]) * np.linalg.norm(self.vel_hat[j])

            # 循环计算弹簧力和海森矩阵
            for i in range(NS): 
                a = Spring.ele[i][0]  # 弹簧连接的第一个质点
                b = Spring.ele[i][1]  # 弹簧连接的第二个质点
                x_ab = self.pos_hat[a] - self.pos_hat[b]  # 两点之间的向量
                x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
                
                # 计算弹簧力
                Energy_E += 0.5 * Spring.stiff_k * (x_ab_norm - Spring.rest_len[i]) * (x_ab_norm - Spring.rest_len[i])
            
            Energy = Energy_E + Energy_V + Energy_G - Energy_G0
            return Energy

    def boundary_solve(self, fixed_num, space_dim):
        """
        Python translation of MATLAB function treat_Dirichlet.

        Parameters
        ----------
        coe_fun : callable
            Function coe_fun(x, y) returning boundary value.
        boundarynodes : (2, nbe) ndarray
            boundarynodes[0, k] : 标记（-1 表示 Dirichlet 边界）
            boundarynodes[1, k] : 节点索引 (0-based in Python).
        P : (2, n_points) ndarray
            P[0, i], P[1, i] give coordinates of node i.
        A : (n_points, n_points) ndarray
            System matrix to modify in-place.
        b : (n_points,) ndarray
            RHS vector to modify in-place.

        Returns
        -------
        A, b : modified arrays
        """
        A = -self.Hessian
        b = self.force

        for i in fixed_num:
            k = i*space_dim
            A[k, :] = 0.0                     # zero out row i
            A[k+1, :] = 0.0
            A[k+2, :] = 0.0
            A[k, k] = 1.0                     # set diagonal to 1
            A[k+1, k+1] = 1.0
            A[k+2, k+2] = 1.0

            b[k] = 0 #self.pos[i,0]  # set boundary value
            b[k+1] = 0 #self.pos[i,1]  # set boundary value
            b[k+2] = 0 #self.pos[i,2]  # set boundary value
        return A, b
