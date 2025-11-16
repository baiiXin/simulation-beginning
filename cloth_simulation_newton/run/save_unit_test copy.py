import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =================== 数据构造部分 ===================
fixed_num = 0 # int((b-a)/h1+1)

# 材料参数
z=8.0
mass_m = 1
stiff_k = 8000

# 阻尼参数
dump = 0.999
gravity = 9.8

# simulation
# 初始参数
dt = 0.003
N = 3000
ite_num = 100
tolerance_newton = 1e-4

# 仿真计算
from cloth import Mass, Spring

Mass_X = np.array([[0.0,0.0,0.0], [3.0,0.0,0.0], [0.0,4.0,0.0], [1.0,1.0,1.0], [2.0,1.0,1.5], [1.0,2.0,1.5]])

Mass_V = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])

Mass_E = np.array([[0,1,2], [3,4,5]])

Spring_ele = np.array([[0,1], [0,2], [1,2], [3,4], [3,5], [4,5]])

Spring_len = np.array([np.linalg.norm(Mass_X[Spring_ele[i,0]] - Mass_X[Spring_ele[i,1]]) for i in range(Spring_ele.shape[0])])

# 创建弹簧
mySpring = Spring(
    num=Spring_ele.shape[0],
    ele=Spring_ele,
    rest_len=Spring_len,
    stiff_k=stiff_k
)
#print(Mass_X)
# 创建质点
myMass = Mass(
    num=Mass_X.shape[0],
    ele=Mass_E,
    pos_cur=Mass_X.copy(),
    vel_cur=Mass_V.copy(),
    pos_prev=Mass_X.copy(),
    vel_prev=Mass_V.copy(),
    mass=mass_m,
    damp=dump,
    gravity=gravity,
    Spring=mySpring,
    dt=dt,
    tolerance_newton=tolerance_newton
)

# 储存结果
cloth_data = [myMass.pos_cur.copy()]
cloth_vel = [myMass.vel_cur.copy()]

# 计算
for i in range(N):
    print("\n\n=====Time step: ", i, "=====")
    [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.Single_Newton_Method(mySpring, fixed_num, ite_num, time_step=i)
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
save_path_cloth = os.path.join(save_dir, "cloth_all_frame.npy")
save_path_tri = os.path.join(save_dir, "triangles.npy")
save_path_cloth_data = os.path.join(save_dir, "cloth_data.npy")

np.save(save_path, verts)
print(f"已保存最后一帧顶点到: {save_path}")
np.save(save_path_vel, vel)
print(f"已保存最后一帧速度到: {save_path_vel}")
np.save(save_path_tri, Mass_E.astype(np.int32))
print(f"已保存三角面到: {save_path_tri}")

# 动画
np.save(save_path_cloth, np.array(cloth_data))
print(f"已保存所有帧到: {save_path_cloth}")
np.save(save_path_cloth_data, np.array(cloth_data))
print(f"已保存所有帧到: {save_path_cloth_data}")



# =================== 之后导入使用 ===================
# 查看当前工作目录
print("当前工作目录:", os.getcwd())

# 获取当前脚本文件所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼出 data 目录的绝对路径
file_path = os.path.join(script_dir, "..", "data", "verts_last_frame.npy")
file_path_vel = os.path.join(script_dir, "..", "data", "vel_last_frame.npy")
file_path_cloth = os.path.join(script_dir, "..", "data", "cloth_all_frame.npy")

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

