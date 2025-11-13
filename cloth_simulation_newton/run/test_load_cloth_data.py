import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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