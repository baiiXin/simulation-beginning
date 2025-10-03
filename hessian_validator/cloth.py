### truncate the displacement
from platform import java_ver
import warp as wp
import numpy as np
import trimesh

# cpmpute bounds
import newton
from newton._src.solvers.zcy_vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.zcy_vbd.zcy_solver_vbd import zcy_SolverVBD, get_vertex_num_adjacent_edges, get_vertex_adjacent_edge_id_order, get_vertex_num_adjacent_faces, get_vertex_adjacent_face_id_order, ForceElementAdjacencyInfo

# TODO: compute the conservative bounds
@wp.kernel
def iterate_vertex_neighbor_primitives(
    adjacency: ForceElementAdjacencyInfo,
    min_dis_v_t: wp.array(dtype=float),
    min_dis_e_e: wp.array(dtype=float),
    min_dis_t_v: wp.array(dtype=float),
    gama_p: float,
    # output
    bounds: wp.array(dtype=float),
):
    particle_idx = wp.tid()
    bounds[particle_idx] = min_dis_v_t[particle_idx]

    # iterating over neighbor faces
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_idx)
    for face_counter in range(num_adj_faces):
        adj_face_idx, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_idx, face_counter)
        bounds[particle_idx] = wp.min(bounds[particle_idx], min_dis_t_v[adj_face_idx])

    # iterating over neighbor edges
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_idx)
    for edge_counter in range(num_adj_edges):
        edge_idx, v_order = get_vertex_adjacent_edge_id_order(adjacency, particle_idx, edge_counter)
        bounds[particle_idx] = wp.min(bounds[particle_idx], min_dis_e_e[edge_idx])
    bounds[particle_idx] = gama_p * bounds[particle_idx]

# TODO: truncate the displacement
@wp.kernel
def truncate_displacement(
    vertices: wp.array(dtype=wp.vec3),
    displacements: wp.array(dtype=wp.vec3),
    new_pos: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=float)
):
    tid = wp.tid()  # 获取当前线程的索引
    if wp.length(displacements[tid]) > bounds[tid]:
        new_pos[tid] = vertices[tid] + displacements[tid] * (bounds[tid] / wp.length(displacements[tid]))
    else:
        new_pos[tid] = vertices[tid] + displacements[tid]

### cloth sim
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
                 pos_cur=None, vel_cur=None, pos_prev=None, vel_prev=None,
                 ele=None, mass=None, 
                 force=None, Hessian=None, Mass_k=None,
                 dump=None, gravity=None, Spring=Spring, dt=None, 
                 tolerance_newton=None):
        self.num = num # 质点数量；1
        self.ele = ele # 三角元；[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        self.pos_cur = pos_cur # 质点位置；[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        self.vel_cur = vel_cur # 质点速度；[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.pos_prev = pos_prev # 预测质点位置
        self.vel_prev = vel_prev # 预测质点速度
        self.force = force # 力向量--牛顿迭代--线性方程组--b
        self.Hessian = Hessian # 矩阵--牛顿迭代--线性方程组--A
        self.mass = mass # 质点质量；1
        self.dump = dump # 阻尼系数；1
        self.gravity = gravity # 重力加速度；9.8
        self.Spring = Spring # 弹簧
        self.dt = dt # 时间步长
        self.tolerance_newton = tolerance_newton # 牛顿迭代的容差
        self.iterations = 10
        # self.fixed_num = 9
        self.space_dim = 3

        # fixed points
        # 初始化
        self.fixed_idx = [36, 44] #[72, 80] #[0, 8] #[36, 44]
        self.all_idx = np.arange(self.num)
        self.free_idx = np.setdiff1d(self.all_idx, self.fixed_idx)
        free_idx = np.array(self.free_idx)
        dof_matrix = 3 * free_idx[:, np.newaxis] + np.arange(3)
        self.free_dof = dof_matrix.flatten()
        # print('self.free_dof', self.free_dof)

        # 初始值
        self.pos_prev = self.pos_cur.copy()
        self.vel_prev = self.vel_cur.copy()

        # warp_vbd_self_collison_init
        self.device = wp.get_device('cpu')
        self.pos_warp = [wp.vec3(self.pos_cur[i,:]) for i in range(self.num)]

        self.builder = newton.ModelBuilder()
        self.builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vertices=self.pos_warp,
                    indices=self.ele.reshape(-1),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=0.02,
                    tri_ke=1.0e5,
                    tri_ka=1.0e5,
                    tri_kd=2.0e-6,
                    edge_ke=10,
        )
        self.builder.add_ground_plane()
        self.builder.color(include_bending=True)
        self.model = self.builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.vbd_integrator = zcy_SolverVBD(model=self.model, iterations=self.iterations, handle_self_contact=True)

        # transform
        #self.pos_warp = wp.array(self.pos_warp, dtype=wp.vec3)
        #self.pos_prev_warp = wp.array(self.pos_prev, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)
        self.pos_warp = self.state_1.particle_q
        self.pos_prev_warp = self.state_0.particle_q
        

        # 检查
        print('model.tri_indices', self.model.tri_indices.shape)
        print('model.edge_indices', self.model.edge_indices.shape)

        # to access ForceElementAdjacencyInfo, you need to construct a VBDIntegrator (you dont need to understand what it is)
        self.collision_detector = TriMeshCollisionDetector(self.model)
        self.collision_detector.vertex_triangle_collision_detection(0.2)
        self.collision_detector.edge_edge_collision_detection(0.2)

        # bounds_init
        self.gama_p = 0.4
        self.bounds = wp.empty(shape=self.model.particle_count, dtype=float)
        
    def Single_Newton_Method(self, Spring: Spring, fixed_num, ite_num, space_dim=3):
        '''
        功能:  牛顿迭代, 一个时间步
        input: 质点, 弹簧, 迭代步, 固定点数量
        output: 
            1. 直接修改mss类的位置和速度
            2. 绘制收敛曲线的参数:
                Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm
        '''
        print('\n---bounds and forward step---')
        self.contacts = self.model.collide(self.state_0)

        # must include this after you update the mesh position, otherwise the collision detection results are not precise
        self.vbd_integrator.trimesh_collision_detector.refit(self.pos_prev_warp)
        self.vbd_integrator.trimesh_collision_detector.triangle_triangle_intersection_detection()

        # ===== 打印三角形相交检测结果 =====
        print("===== Triangle-Triangle Intersection Results =====")
        # 2. 每个三角形的相交数量
        counts = self.vbd_integrator.trimesh_collision_detector.triangle_intersecting_triangles_count.numpy()
        # 额外：总相交数量
        print("Total intersections:", counts.sum())

        #print('max(pos_cur-pos_prev):', np.max(self.pos_cur-self.pos_prev), np.max(self.pos_prev-self.pos_cur))
        # compute bounds and forward
        # self.compute_bounds()
        #print('pos_warp', id(self.pos_warp))
        #print('pos_prev_warp', id(self.pos_prev_warp))
        self.vbd_integrator.zcy_forward_step_penetration_free(self.pos_warp, self.pos_prev_warp, self.vel_warp, self.dt)
        pos_cur = self.pos_warp.numpy()
        vel_cur = self.vel_warp.numpy()
        # self.truncate_displacement(self.bounds, self.pos_cur, self.pos_cur-self.pos_prev)

        # free point update
        for i, p in enumerate(self.free_idx):
                self.pos_cur[p] = pos_cur[p]
                self.vel_cur[p] = vel_cur[p]
        
        #print('max(pos_cur-pos_prev)0:', np.max(pos_cur-self.pos_prev), np.max(self.pos_prev-pos_cur))
        #print('max(pos_cur-pos_prev)1:', np.max(self.pos_cur-self.pos_prev), np.max(self.pos_prev-self.pos_cur))
        self.vbd_integrator.zcy_collision_detection_penetration_free(self.pos_warp)
        
        # Newton Method (Implicit Euler)
        # 计时
        times_ms = []
        # 残差曲线
        Newton_step = []
        Error_dx_norm = []
        Residual_norm = []
        Energy_norm = []

        # 迭代初值
        # self.pos_hat = self.pos.copy()

        print('\n---iteration---')
        # 迭代
        for times in range(ite_num):
            print('---iteration step start---')
            # 计时
            start_time = time.time()
            print('---flag---')
            # 计算能量
            Energy = self.Energy_compute(Spring)
            
            # 组装
            self.assemebel_HF(Spring)
            
            # fixed_num
            # 注意：只求解自由部分
            A = -self.Hessian[np.ix_(self.free_dof, self.free_dof)]
            b =  self.force[self.free_dof]

            # 矩阵稀疏化
            A_sparse = csr_matrix(A)
            
            # 计算dX
            dX, info = cg(A_sparse, b)
            # print('cg:',info)
            #print('\ndX:', dX)
            
            # 更新位置（只更新非固定点）
            pos_new = self.pos_cur.copy()
            # 将 x_free 重新写回到 pos_new 对应的自由点
            for i, p in enumerate(self.free_idx):
                pos_new[p] += dX[3*i : 3*i+3]
            self.pos_cur = pos_new.copy()
            self.pos_warp = wp.array(self.pos_cur, dtype=wp.vec3)

            # 截断
            # self.truncate_displacement(self.bounds, self.pos_cur, self.pos_cur-self.pos_prev)
            # print('self.pos_warp', id(self.pos_warp))
            self.vbd_integrator.zcy_truncation_by_conservative_bound(self.pos_warp)
            self.pos_cur = self.pos_warp.numpy()
            # print('self.pos_warp', id(self.pos_warp))

            # 组装时间
            end_time = time.time()  # 结束时间
            times_ms.append((end_time - start_time) * 1000)  # 计算并存储运行时间（毫秒）

            # 计算误差
            error_dx_norm = np.linalg.norm(dX)
            
            # 计算残差
            residual = b  
            residual_norm = np.linalg.norm(residual)

            # 记录残差
            Newton_step.append(times)
            Error_dx_norm.append(error_dx_norm)
            Residual_norm.append(residual_norm)
            Energy_norm.append(Energy)

            # 打印残差
            print('\nNewton Method iteration step: ', times)
            print('times_ms = ', (end_time - start_time) * 1000)
            print('error_dx_norm =', error_dx_norm)
            print('residual_norm = ', residual_norm, residual_norm.dtype)
            print('Enegy = ', Energy)
            '''
            with open("debug.log", "a") as f:
                f.write("checkpoint\n")
                f.flush()
            '''

            print('---iteration step end---\n')
            # 如果误差足够小，提前结束迭代
            if error_dx_norm < self.tolerance_newton:
                break
 
        # 更新位置和速度
        print('\n---update---')
        self.vel_cur = (self.pos_cur - self.pos_prev) / self.dt * self.dump
        self.pos_cur, self.pos_prev = self.pos_prev, self.pos_cur

        # transform
        self.pos_warp = wp.array(self.pos_cur, dtype=wp.vec3)
        self.pos_prev_warp = wp.array(self.pos_prev, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)

        return Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm

    def assemebel_HF(self, Spring: Spring):
        '''
        input: 质点＋弹簧
        output: 直接修改 self.Hessian 和 self.force; 
                不是实际的force和hessian; 对应牛顿迭代的A和b
        '''
        # self_collision_force_and_hessian
        self.vbd_integrator.zcy_compute_hessian_force(self.pos_warp, self.pos_prev_warp, self.dt, self.state_0, self.state_1, self.control, self.contacts)
        self.particle_forces, self.particle_hessians = self.vbd_integrator.particle_forces.numpy(), self.vbd_integrator.particle_hessians.numpy()

        # 测试 
        print('self.particle_forces', self.particle_forces.shape, type(self.particle_forces))
        #print('self.particle_hessians', self.particle_hessians.shape, type(self.particle_hessians))
        #print(self.particle_forces[0])
        #print(self.particle_hessians[0])
        #print(np.isnan(self.particle_forces).all())
        #print(np.isnan(self.particle_hessians).all())
        print('max(self.particle_forces):', np.max(np.abs(self.particle_forces)))
        print('max(self.particle_hessians):', np.max(np.abs(self.particle_hessians)))
        #print(np.argwhere(np.isnan(self.particle_forces)))
        #print(np.argwhere(np.isnan(self.particle_hessians)))
        
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
            x_ab = self.pos_cur[a] - self.pos_cur[b]  # 两点之间的向量
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

            # 接触力
            f[(j*space_dim):(j*space_dim+space_dim)] += self.particle_forces[j]
            for i in range(Nm):
                H[(j*space_dim):(j*space_dim+space_dim), (i*space_dim):(i*space_dim+space_dim)] += self.particle_hessians[j*Nm+i]
            
            # 质量/单位矩阵
            I_eyes = np.eye(space_dim)
            I[(j*space_dim):(j*space_dim+space_dim), (j*space_dim):(j*space_dim+space_dim)] = I_eyes
            
            # 计算F向量
            F[(j*space_dim):(j*space_dim+space_dim)] = self.pos_cur[j] - self.pos_prev[j] - self.dt * self.vel_cur[j]
        
        # 返回计算结果
        self.force = F - self.dt * self.dt * (1/self.mass) * (f+g)  # 计算b向量
        self.Hessian = I - self.dt * self.dt * (1/self.mass) *H  # 计算A矩阵

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
            Energy_G  +=  self.mass * self.gravity * self.pos_cur[j][2]
        
            # 计算F向量
            norm_F_vec[(j*space_dim):(j*space_dim+space_dim)] = self.pos_cur[j] - self.pos_prev[j] - self.dt * self.vel_cur[j]
        # 计算能量
        Energy_F +=  (1.0/ (2.0 * self.dt * self.dt)) * self.mass * np.linalg.norm(norm_F_vec) * np.linalg.norm(norm_F_vec)

        # 循环计算弹簧力和海森矩阵
        for i in range(NS): 
            a = Spring.ele[i][0]  # 弹簧连接的第一个质点
            b = Spring.ele[i][1]  # 弹簧连接的第二个质点
            x_ab = self.pos_cur[a] - self.pos_cur[b]  # 两点之间的向量
            x_ab_norm = np.linalg.norm(x_ab)  # 向量的范数（长度）
            # 计算弹簧力
            Energy_E += 0.5 * Spring.stiff_k * (x_ab_norm - Spring.rest_len[i]) * (x_ab_norm - Spring.rest_len[i])
        # 计算能量
        Energy = Energy_E + Energy_F + Energy_G
        return Energy


'''
    def compute_bounds(self):

        self.collision_detector.refit(self.pos_warp)
        self.collision_detector.vertex_triangle_collision_detection(0.3)
        self.collision_detector.edge_edge_collision_detection(0.3)
        self.vbd_integrator.pos_prev_collision_detection.assign(self.pos_warp)

        wp.launch(
            kernel=iterate_vertex_neighbor_primitives,
            inputs=[
                self.vbd_integrator.adjacency, 
                self.collision_detector.vertex_colliding_triangles_min_dist, 
                self.collision_detector.edge_colliding_edges_min_dist, 
                self.collision_detector.triangle_colliding_vertices_min_dist, 
                self.gama_p],
            outputs=[
                self.bounds],
            dim=self.model.particle_count,
            device=self.device
        )

    def truncate_displacement(self, bounds, vertices, displacement):

        # 将vertices列表转换为warp数组
        vertices_array = wp.array(vertices, dtype=wp.vec3, device=self.device)
        # 将displacement_init列表转换为warp数组
        displacement_array = wp.array(displacement, dtype=wp.vec3, device=self.device)

        # replace this with actual truncation
        new_pos = wp.array(shape=len(displacement), dtype=wp.vec3, device=self.device)

        wp.launch(
            truncate_displacement,
            dim=self.num,
            inputs=[vertices_array, displacement_array, new_pos, bounds],
            device=self.device
        )

        self.pos_cur = new_pos.numpy()
'''

