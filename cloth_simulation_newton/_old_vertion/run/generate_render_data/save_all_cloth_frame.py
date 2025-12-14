import numpy as np

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
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
h1 = 0.5
h2 = 0.5
#fixed_num0 = (int((b-a)/h1)+1)*2 + 1
#fixed_num1 = (int((b-a)/h1)+1)*3 + 1 
#fixed_num = [fixed_num0, fixed_num1]
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
dt = 0.001
N = 3000
ite_num = 1000
tolerance_newton = 1e-2

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
save_dir = os.path.join(os.path.dirname(__file__), "../render/input")
os.makedirs(save_dir, exist_ok=True)
run_id = os.path.splitext(os.path.basename(__file__))[0].replace("save_", "")
path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
path_topy = os.path.join(save_dir, f"cloth_topy_{run_id}.npy")
np.save(path_data, np.array(cloth_data))
np.save(path_topy, triangles.astype(np.int32))



pass

