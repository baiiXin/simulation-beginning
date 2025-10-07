import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource  # 导入光源类

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sim import generate_cloth

import imageio_ffmpeg
# 获取 ffmpeg 的路径
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
# 设置 matplotlib 使用的 ffmpeg 路径
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

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
fixed_num = int((b-a)/h1+1) 

# 材料参数
z=10
mass_m = 1
stiff_k = 8000

# 阻尼参数
dump = 1.0
gravity = 9.8

# simulation
# 初始参数
dt = 0.015
N = 300
ite_num = 10
tolerance_newton = 1e-6

# 球体参数
sphere_center = np.array([0.5, -0.5, 6.0])  # 球心坐标
sphere_radius = 2.49                        # 球半径

# 仿真计算
[triangles, cloth_data, Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = generate_cloth(a, b, c, d, h1, h2, z, mass_m, stiff_k, dump, gravity, dt, N, ite_num, tolerance_newton, fixed_num)

# =================== 动画部分 ===================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 设置视角 - 仰角30度，方位角45度
ax.view_init(elev=15, azim=145)  # elev=0表示水平视角，azim=90表示从x轴方向看过去

# 创建光源 - 从y轴负方向照射，平行xy平面
ls = LightSource(azdeg=0, altdeg=0)  # azdeg=180表示南方向（y轴负方向），altdeg=0表示平行于xy平面

def update(frame):
    ax.cla()  # 清除上一帧

    # 设置坐标轴范围
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # -------- 绘制固定球体 --------
    # 绘制固定球体（根据球心和半径）
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = sphere_radius * np.cos(u) * np.sin(v) + sphere_center[0]
    y_sphere = sphere_radius * np.sin(u) * np.sin(v) + sphere_center[1]
    z_sphere = sphere_radius * np.cos(v) + sphere_center[2]
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=1.0, linewidth=0)


    # -------- 绘制布料表面 --------
    verts = cloth_data[frame]  # 当前帧的 (Nm, 3) 坐标
    faces = verts[triangles]  # shape = (Ne, 3, 3)

    # 计算每个面的法向量，用于光照效果
    face_colors = np.ones((len(faces), 4))  # RGBA颜色数组
    face_colors[:, 0:3] = [0.5, 0.7, 1.0]  # 设置基础颜色为天蓝色
    
    # 应用光照效果
    # 提取每个面的法向量
    normals = []
    for face in faces:
        # 计算两个边向量
        v1 = face[1] - face[0]
        v2 = face[2] - face[0]
        # 计算法向量（叉乘）
        normal = np.cross(v1, v2)
        # 归一化
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        normals.append(normal)
    
    normals = np.array(normals)
    # 应用光照效果 - 根据法向量与光源方向的点积计算明暗
    light_dir = np.array([0, -1, 0])  # 光源方向：y轴负方向，平行xy平面
    intensities = np.abs(np.dot(normals, light_dir))  # 点积的绝对值作为光照强度
    
    # 调整颜色亮度
    for i, intensity in enumerate(intensities):
        face_colors[i, 0:3] = face_colors[i, 0:3] * (0.5 + 0.5 * intensity)  # 调整亮度范围
    
    # 绘制布料表面（带光照效果）
    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly)

ani = FuncAnimation(fig, update, frames=len(cloth_data), interval=20)
plt.show()

ani.save('cloth_simulation.mp4', writer='ffmpeg', fps=60)
