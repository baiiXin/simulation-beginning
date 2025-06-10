import matplotlib.pyplot as plt
from sim import generate_cloth

# 示例数据
newton_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Newton 迭代步数
residuals = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]  # 每一步的残差
# =================== 数据构造部分 ===================
# 你的数据（假设已加载）
# cloth_data: List of (Nm, 3) arrays
# triangles: (Ne, 3) array of vertex indices
# ===== 仿真参数 =====
# 网格生成
a = -2
b = 2.0 
c = 0.0
d = 4.0
h1 = 0.5
h2 = 0.5
fixed_num = 0 #int((b-a)/h1+1) 

# 材料参数
z=1
mass_m = 1
stiff_k = 8000

# 阻尼参数
dump = 1.0
gravity = 9.8

# simulation
# 初始参数
dt = 0.02
N = 1
ite_num = 12
tolerance_newton = 1e-16

# 仿真计算
[triangles, cloth_data, Newton_steps, Times_ms, Error_dx_norm, Residual_norm, Energy_norm] = generate_cloth(a, b, c, d, h1, h2, z, mass_m, stiff_k, dump, gravity, dt, N, ite_num, tolerance_newton, fixed_num)

# 计算原始 Times_ms
original_Times_ms = [sum(Times_ms[:i+1]) for i in range(len(Times_ms))]

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制收敛曲线 (Newton_steps 作为底部 X 轴)
ax1.plot(Newton_steps, Energy_norm, marker='o', linestyle='-', color='b', label='Energy vs. Newton Steps')
ax1.set_xlabel('Newton Steps', fontsize=14)
ax1.set_ylabel('Energy', fontsize=14)
#ax1.set_yscale('log')  # 设置纵坐标为对数刻度
ax1.set_ylim(min(Energy_norm), max(Energy_norm))  # 设置纵坐标范围

# 创建第二个坐标轴 (original_Times_ms 作为顶部 X 轴)
ax2 = ax1.twiny()  # 与 ax1 共享 y 轴
ax2.plot(original_Times_ms, Energy_norm, marker='x', linestyle='--', color='r', label='Energy vs. Time (ms)')
ax2.set_xlabel('Time (ms)', fontsize=14)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Convergence Curve', fontsize=16)
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.show()