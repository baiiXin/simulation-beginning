### truncate the displacement
from platform import java_ver
import warp as wp
import numpy as np
import trimesh
import os

# cpmpute bounds
import newton
from newton._src.solvers.zcy_vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.zcy_vbd.zcy_solver_vbd import zcy_SolverVBD

### cloth sim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, bicgstab
import time


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
                 damp=None, gravity=None, Spring=Spring, dt=None, 
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
        self.damp = damp # 阻尼系数；1
        self.gravity = gravity # 重力加速度；9.8
        self.Spring = Spring # 弹簧
        self.dt = dt # 时间步长
        self.tolerance_newton = tolerance_newton # 牛顿迭代的容差
        self.iterations = 10
        # self.fixed_num = 9
        self.space_dim = 3
        self.load = False

        # load vertexs
        self._load_cloth_data(self.load)

        # fixed points
        # 初始化
        self.fixed_idx = [36, 44]#[360, 440] #[0, 4] #[10, 14] #[72, 80] #[0, 8] #[36, 44]
        self._compute_fixed_information()

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

        # spring information
        self.spring_indices = [x for row in self.Spring.ele for x in row]
        self.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
        self.spring_rest_length = wp.array(self.Spring.rest_len, dtype=wp.float32)
        self.spring_stiffness = [self.Spring.stiff_k for i in range(len(self.spring_rest_length))]
        self.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32)

        print('spring_indices', type(self.spring_indices))

        self.vbd_integrator = zcy_SolverVBD(
            model=self.model,
            # self parameters
            dt = self.dt,
            mass = self.mass,
            # fixed particle information
            fixed_particle_num = self.fixed_particle_num,
            free_particle_offset = self.free_particle_offset,
            all_particle_flag = self.all_particle_flag,
            # other
            iterations=self.iterations, 
            # before
            handle_self_contact=True,
            self_contact_radius=0.08,
            self_contact_margin=0.08,
            spring_indices = self.spring_indices, 
            spring_rest_length = self.spring_rest_length, 
            spring_stiffness = self.spring_stiffness
        )

        # transform
        self.pos_warp = wp.array(self.pos_warp, dtype=wp.vec3)
        self.pos_prev_warp = wp.array(self.pos_prev, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)
        self.vel_prev_warp = wp.array(self.vel_prev, dtype=wp.vec3)
        
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
        
        # Newton Method (Implicit Euler)
        # 计时
        times_ms = []
        # 残差曲线
        Newton_step = []
        Error_dx_norm = []
        Residual_norm = []
        Energy_norm = []

        #
        print('\n---cloth---', self.iterations)

        self.vbd_integrator.zcy_simulate_one_step(
            pos_warp = self.pos_warp,
            pos_prev_warp = self.pos_prev_warp,
            vel_warp = self.vel_warp,
            dt = self.dt, 
            mass = self.mass, 
            damping = self.damp, 
            num_iter = ite_num,
            tolerance = self.tolerance_newton,
        )

        self.pos_cur = self.pos_warp.numpy()
        self.vel_cur = self.vel_warp.numpy()

        self.pos_warp, self.pos_prev_warp = self.pos_prev_warp, self.pos_warp

        return Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm

    def _compute_fixed_information(self):
        self.fixed_particle_num = len(self.fixed_idx)

        self.all_particle_flag = []
        self.free_particle_offset = []
        flag = 0
        for i in range(self.num):
            if i in self.fixed_idx:
                self.all_particle_flag.append(-1)
                flag += 1
            else:
                self.all_particle_flag.append(flag)
                self.free_particle_offset.append(flag)

        self.all_particle_flag = wp.array(self.all_particle_flag, dtype=wp.int32)
        self.free_particle_offset = wp.array(self.free_particle_offset, dtype=wp.int32)

    def _load_cloth_data(self, load=False):
        if not load:
            return
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼出 data 目录的绝对路径
        file_path = os.path.join(script_dir, "data", "verts_last_frame.npy")
        file_path_vel = os.path.join(script_dir, "data", "vel_last_frame.npy")
        
        # 判断文件是否存在
        if not os.path.exists(file_path):
            print(f"未找到文件: {file_path}，跳过加载。")
            return

        # 加载数据
        verts = np.load(file_path)
        vel = np.load(file_path_vel)
        self.pos_cur = verts
        self.vel_cur = vel
        print('\n---Finish loading cloth data---\n')
        #print('pos_cur', self.pos_cur)
        #print('vel_cur', self.vel_cur)

