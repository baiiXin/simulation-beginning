### truncate the displacement
import warp as wp
import numpy as np
#import trimesh
import os

# cpmpute bounds
import newton
from newton._src.solvers.zcy_vbd.zcy_solver_vbd import zcy_SolverVBD

@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt



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
                 tolerance_newton=None, cloth_size=None, DeBUG=None):
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
        self.cloth_size = cloth_size
        self.DeBUG = DeBUG

        # load vertexs
        self._load_cloth_data(self.load)

        # fixed points
        # rotation fixed points
        #cloth_size = 9
        left_side = [ i for i in range(cloth_size)]
        right_side = [cloth_size * (cloth_size-1) + i for i in range(cloth_size)]
        rot_point_indices = left_side + right_side
        # 初始化
        self.fixed_idx = [0, 1, 2] #[360, 440] #[0, 4] #[10, 14] #[72, 80] #[0, 8] #[36, 44]
        self._compute_fixed_information()

        # 缩放
        self.scale=1.0

        # contact parameters
        self.contact_radius=0.02
        self.contact_margin=0.03
        self.contact_ke=1.0e3
        self.tri_ke=1.0e3
        self.bend_ke=1.0e3

        # 初始值
        #self.pos_cur[:, [1, 2]] = self.pos_cur[:, [2, 1]]
        self.pos_prev = self.pos_cur.copy()*self.scale
        self.vel_prev = self.vel_cur.copy()*self.scale

        # warp_vbd_self_collison_init
        self.device = wp.get_device('cpu')
        self.pos_warp = [wp.vec3(self.pos_cur[i,:]) for i in range(self.num)]

        self.builder = newton.ModelBuilder()
        self.builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=self.scale,
                    vertices=self.pos_warp,
                    indices=self.ele.reshape(-1),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=0.2,
                    tri_ke=self.tri_ke,
                    tri_kd=self.tri_ke,
                    tri_ka=self.tri_ke,
                    edge_ke=self.bend_ke,
                    edge_kd=self.bend_ke,
        )
        self.builder.add_ground_plane()
        self.builder.color(include_bending=True)
        self.model = self.builder.finalize()

        # contact parameters
        self.model.contact_ke = self.contact_ke

        # model.gravity
        self.model.gravity = wp.vec3(0.0, 0.0, -self.gravity)
        print('self.model.g', self.model.gravity)

        # spring information
        self.spring_indices = [x for row in self.Spring.ele for x in row]
        self.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
        self.spring_rest_length = wp.array(self.Spring.rest_len, dtype=wp.float32)
        self.spring_stiffness = [self.Spring.stiff_k for i in range(len(self.spring_rest_length))]
        self.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32)

        print('spring_indices', type(self.spring_indices))

        self.vbd_integrator = zcy_SolverVBD(
                    model=self.model,
                    # DeBUG
                    DeBUG = self.DeBUG,
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
                    self_contact_radius=self.contact_radius,
                    self_contact_margin=self.contact_margin,
                    spring_indices = self.spring_indices, 
                    spring_rest_length = self.spring_rest_length, 
                    spring_stiffness = self.spring_stiffness
        )

        # state
        #self.state_0 = self.model.state()
        #self.state_1 = self.model.state()
        # transform
        self.pos_cur *= self.scale
        self.vel_cur *= self.scale
        self.pos_warp = wp.array(self.pos_cur, dtype=wp.vec3)
        self.pos_prev_warp = wp.array(self.pos_prev, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)
        self.vel_prev_warp = wp.array(self.vel_prev, dtype=wp.vec3)

        # 检查
        print('model.tri_indices', self.model.tri_indices.shape)
        print('model.edge_indices', self.model.edge_indices.shape)

        # rotation
        rot_axes = [[0, 1, 0]] * len(right_side) + [[0, -1, 0]] * len(left_side)

        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros((1,), dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)

        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        self.rot_angular_velocity = np.pi / 3
        self.rot_end_time = 10

        self._init_rotation()

    def time_step(self, Spring: Spring, fixed_num, ite_num, space_dim=3, time_step=0, rotation=False):
        self._apply_rotation(rotation=rotation)

        Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm = self.Single_Newton_Method(Spring, fixed_num, ite_num, space_dim, time_step)

        return Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm
        
    def Single_Newton_Method(self, Spring: Spring, fixed_num, ite_num, space_dim=3, time_step=0):
        
        # Newton Method (Implicit Euler)
        # 计时
        times_ms = []
        # 残差曲线
        Newton_step = []
        Error_dx_norm = []
        Residual_norm = []
        Energy_norm = []

        # iteration
        print('\n---cloth---', self.iterations)

        # step
        self.vbd_integrator.zcy_simulate_one_step(
            pos_warp = self.pos_warp,
            pos_prev_warp = self.pos_prev_warp,
            vel_warp = self.vel_warp,
            dt = self.dt, 
            mass = self.mass, 
            damping = self.damp, 
            num_iter = ite_num,
            tolerance = self.tolerance_newton,
            time_step = time_step,
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

    def _init_cloth_data(self):
        self.pos_warp = wp.array(self.pos_cur, dtype=wp.vec3)
        self.pos_prev_warp = wp.array(self.pos_cur, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)
        self.vel_prev_warp = wp.array(self.vel_cur, dtype=wp.vec3)

    def _init_rotation(self):

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.pos_warp,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[
                self.roots,
                self.roots_to_ps,
            ],
        )

    def _apply_rotation(self, rotation=False):

        if not rotation:
            return

        wp.launch(
            kernel=apply_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.rot_axes,
                self.roots,
                self.roots_to_ps,
                self.t,
                self.rot_angular_velocity,
                self.dt,
                self.rot_end_time,
            ],
            outputs=[
                self.pos_prev_warp,
                self.pos_warp,
            ],
        )