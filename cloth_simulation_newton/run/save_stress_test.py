import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from generate_fun import generate_spring, generate_mass
from cloth import Mass, Spring

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
h1 = 0.5
h2 = 0.5
#fixed_num0 = (int((b-a)/h1)+1)*2 + 1
#fixed_num1 = (int((b-a)/h1)+1)*3 + 1 
#fixed_num = [fixed_num0, fixed_num1]
fixed_num = 0 # int((b-a)/h1+1)

cloth_size = int((b-a)/h1+1)

# 材料参数
z=0.0
mass_m = 1
stiff_k = 800

# 阻尼参数
dump = 0.995
gravity = 0.0

# simulation
# 初始参数
dt = 0.003
N = 1000
ite_num = 100
tolerance_newton = 1e-4


# 仿真计算
# Mass
[Mass_num, Mass_X, Mass_E, Mass_V, Mass_m] = generate_mass(a, b, c, d, h1, h2, z, mass_m)
print("Mass_num", Mass_num)
# Spring
[Spring_num, Spring_ele, Spring_len, Spring_stiff_k] = generate_spring(Mass_X, Mass_E, stiff_k)
print("Spring_num", Spring_num)

# 创建弹簧
mySpring = Spring(
    num=Spring_num,
    ele=Spring_ele,
    rest_len=Spring_len,
    stiff_k=Spring_stiff_k
)
#print(Mass_X)
# 创建质点
myMass = Mass(
    num=Mass_num,
    ele=Mass_E,
    pos_cur=Mass_X.copy(),
    vel_cur=Mass_V.copy(),
    pos_prev=Mass_X.copy(),
    vel_prev=Mass_V.copy(),
    mass=Mass_m,
    damp=dump,
    gravity=gravity,
    Spring=mySpring,
    dt=dt,
    tolerance_newton=tolerance_newton,
    cloth_size=cloth_size
)

# 储存结果
cloth_data = [myMass.pos_cur.copy()]
cloth_vel = [myMass.vel_cur.copy()]

# 计算
for i in range(N):
    print("\n\n=====Time step: ", i, "=====")
    [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.time_step(mySpring, fixed_num, ite_num, time_step=i)
    cloth_data.append(myMass.pos_cur.copy())
    cloth_vel.append(myMass.vel_cur.copy())


# =================== 保存最后一帧 ===================
frame = -1
verts = cloth_data[frame]
vel = cloth_vel[frame]


# =================== 保存 verts ===================
save_dir = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(save_dir, exist_ok=True)  # 若 data 文件夹不存在则创建
save_path = os.path.join(save_dir, "verts_last_frame.npy")
save_path_vel = os.path.join(save_dir, "vel_last_frame.npy")
save_path_cloth = os.path.join(save_dir, "cloth_data.npy")

np.save(save_path, verts)
print(f"已保存最后一帧顶点到: {save_path}")
np.save(save_path_vel, vel)
print(f"已保存最后一帧速度到: {save_path_vel}")

# 动画
np.save(save_path_cloth, np.array(cloth_data))
print(f"已保存所有帧到: {save_path_cloth}")



# =================== 之后导入使用 ===================
# 查看当前工作目录
print("当前工作目录:", os.getcwd())

# 获取当前脚本文件所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼出 data 目录的绝对路径
file_path = os.path.join(script_dir, "..", "data", "verts_last_frame.npy")
file_path_vel = os.path.join(script_dir, "..", "data", "vel_last_frame.npy")
file_path_cloth = os.path.join(script_dir, "..", "data", "cloth_data.npy")

print("脚本目录:", script_dir)
print("最终加载路径:", os.path.abspath(file_path))
print("最终加载路径:", os.path.abspath(file_path_vel))

print(f"尝试加载文件: {file_path}")
verts = np.load(file_path)
vel = np.load(file_path_vel)
cloth_data = np.load(file_path_cloth)

print(verts.shape)
print(vel.shape)
print(cloth_data.shape)

