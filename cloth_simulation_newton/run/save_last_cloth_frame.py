import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sim import generate_cloth



# =================== 数据构造部分 ===================
# 你的数据（假设已加载）
# cloth_data: List of (Nm, 3) arrays
# triangles: (Ne, 3) array of vertex indices
# ===== 仿真参数 =====
# 网格生成
a = -2.0
b = 2.0 
c = -2.0
d = 2.0
h1 = 1.0
h2 = 1.0
#fixed_num0 = (int((b-a)/h1)+1)*2 + 1
#fixed_num1 = (int((b-a)/h1)+1)*3 + 1 
#fixed_num = [fixed_num0, fixed_num1]
fixed_num = 0 # int((b-a)/h1+1)

# 材料参数
z=8.0
mass_m = 1
stiff_k = 8000

# 阻尼参数
dump = 0.98
gravity = 9.8

# simulation
# 初始参数
dt = 0.03
N = 11
ite_num = 4
tolerance_newton = 1e-6

# 球体参数
sphere_center = np.array([0, -0.5, 6.0])  # 球心坐标
sphere_radius = 0.99                      # 球半径

# 仿真计算
[triangles, cloth_data, Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm, cloth_vel] = generate_cloth(a, b, c, d, h1, h2, z, mass_m, stiff_k, dump, gravity, dt, N, ite_num, tolerance_newton, fixed_num)

# =================== 保存最后一帧 ===================
frame = -1
verts = cloth_data[frame]
vel = cloth_vel[frame]


# =================== 保存 verts ===================
save_dir = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(save_dir, exist_ok=True)  # 若 data 文件夹不存在则创建
save_path = os.path.join(save_dir, "verts_last_frame.npy")
save_path_vel = os.path.join(save_dir, "vel_last_frame.npy")

np.save(save_path, verts)
print(f"已保存最后一帧顶点到: {save_path}")
np.save(save_path_vel, vel)
print(f"已保存最后一帧速度到: {save_path_vel}")



# =================== 之后导入使用 ===================
# 查看当前工作目录
print("当前工作目录:", os.getcwd())

# 获取当前脚本文件所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼出 data 目录的绝对路径
file_path = os.path.join(script_dir, "..", "data", "verts_last_frame.npy")
file_path_vel = os.path.join(script_dir, "..", "data", "vel_last_frame.npy")

print("脚本目录:", script_dir)
print("最终加载路径:", os.path.abspath(file_path))
print("最终加载路径:", os.path.abspath(file_path_vel))

print(f"尝试加载文件: {file_path}")
verts = np.load(file_path)
vel = np.load(file_path_vel)

print(verts.shape)

