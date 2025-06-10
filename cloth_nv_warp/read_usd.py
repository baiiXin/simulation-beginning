import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pxr import Usd, UsdGeom, Sdf

# -------- 设置参数 --------
usd_file = "example_cloth.usd"  # 你的 USD 文件路径
prim_path = "/root/surface"     # 布料几何体路径
frame_start = 0
frame_end = 299
fps = 60
output_video = "cloth_simulation.mp4"

# -------- 打开 USD 文件 --------
# 打开 USD
stage = Usd.Stage.Open(usd_file)
mesh_prim = stage.GetPrimAtPath(prim_path)
mesh = UsdGeom.Mesh(mesh_prim)

# -------- 创建绘图窗口 --------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# -------- 渲染每一帧 --------
frames = []
for t in range(frame_start, frame_end + 1):
    time_code = Sdf.TimeCode(t)
    points = mesh.GetPointsAttr().Get(time_code)
    
    if points is None:
        continue

    points_np = np.array(points)
    
    ax.clear()
    ax.set_title(f"Frame {t}")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 5])
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=2, c='b')

    # 保存当前帧图像
    fig.canvas.draw()
    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_image)

# -------- 导出为视频 --------
print("正在导出视频，请稍候...")
imageio.mimsave(output_video, frames, fps=fps)
print(f"✅ 视频保存成功：{output_video}")
