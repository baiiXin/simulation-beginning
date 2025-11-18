import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# =================== 之后导入使用 ===================
# 查看当前工作目录
print("当前工作目录:", os.getcwd())

# 获取当前脚本文件所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "render", "input")
files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
if not files:
    raise FileNotFoundError("no cloth_data_*.npy in render/input")
data_file = os.path.join(input_dir, files[0])
suffix = files[0][len("cloth_data_"):-4]
topy_file = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
cloth_data = np.load(data_file)
triangles = np.load(topy_file) if os.path.exists(topy_file) else None
print(cloth_data.shape)
print(triangles.shape if triangles is not None else None)