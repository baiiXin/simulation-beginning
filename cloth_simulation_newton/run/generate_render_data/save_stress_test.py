import numpy as np

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

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
mass_m = 0.2
stiff_k = 8000

# 阻尼参数
dump = 0.995
gravity = 0.0

# simulation
# 初始参数
dt = 0.003
N = 1200
ite_num = 10
tolerance_newton = 1e-4


# 仿真计算
# Mass
[Mass_num, Mass_X, Mass_E, Mass_V, Mass_m] = generate_mass(a, b, c, d, h1, h2, z, mass_m)
Mass_X = Mass_X.astype(np.float64, copy=False)
Mass_V = Mass_V.astype(np.float64, copy=False)
Mass_E = Mass_E.astype(np.int32, copy=False)
Mass_m = np.asarray(Mass_m, dtype=np.float64)
print("Mass_num", Mass_num)
# Spring
[Spring_num, Spring_ele, Spring_len, Spring_stiff_k] = generate_spring(Mass_X, Mass_E, stiff_k)
Spring_ele = Spring_ele.astype(np.int32, copy=False)
Spring_len = Spring_len.astype(np.float64, copy=False)
Spring_stiff_k = np.asarray(Spring_stiff_k, dtype=np.float64)
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
cloth_data = [myMass.pos_cur.astype(np.float64).copy()]
cloth_vel = [myMass.vel_cur.astype(np.float64).copy()]

# 计算
for i in range(N):
    print("\n\n=====Time step: ", i, "=====")
    [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.time_step(mySpring, fixed_num, ite_num, time_step=i, rotation=True)
    cloth_data.append(myMass.pos_cur.astype(np.float64).copy())
    cloth_vel.append(myMass.vel_cur.astype(np.float64).copy())


# =================== 保存最后一帧 ===================
frame = -1
verts = cloth_data[frame].astype(np.float64, copy=False)
vel = cloth_vel[frame].astype(np.float64, copy=False)


# =================== 保存 verts ===================
save_dir = os.path.join(os.path.dirname(__file__), "../render/input")
os.makedirs(save_dir, exist_ok=True)
run_id = os.path.splitext(os.path.basename(__file__))[0].replace("save_", "")
path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
path_topy = os.path.join(save_dir, f"cloth_topy_{run_id}.npy")
np.save(path_data, np.array(cloth_data))
np.save(path_topy, Mass_E.astype(np.int32))



pass

