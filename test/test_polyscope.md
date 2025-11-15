# test_polyscope.py 使用说明（布料动画渲染）

## 功能概述
- 从 `cloth_data` 多帧顶点数据（形状 `N×M×3`）渲染布料动画。
- 优先使用三角网格 `triangles (F×3)` 注册为表面网格；缺失时回退为点云。
- 自动截图并导出视频 `cloth_polyscope.mp4`（需安装 `opencv-python`），否则导出逐帧图片到 `frames_polyscope/`。

## 数据要求与默认路径
- `cloth_data`：`(N, M, 3)`，N 为时间步数，M 为质点数量。
- `triangles`：`(F, 3)` 的整型索引；顶点索引从 0 到 `M-1`。
- 默认读取位置（按优先级）：
  - `../cloth_simulation_newton/data/cloth_data.npy`
  - `../cloth_simulation_newton/data/verts_frames.npy`
  - 若仍未找到，则从单帧 `../cloth_simulation_newton/data/verts_last_frame.npy` 读取并自动扩展为 1 帧动画。
- 三角网格默认读取位置：
  - `../cloth_simulation_newton/data/triangles.npy`
  - 若缺失，且 `M` 为完全平方数（如 `25=5×5`），将按规则网格自动推断三角面；否则回退点云渲染。

代码位置：
- 数据加载：`test/test_polyscope.py:11`、`test/test_polyscope.py:46`
- 网格推断：`test/test_polyscope.py:31`

## 安装依赖
- 必备：`pip install polyscope`
- 视频导出（可选）：`pip install opencv-python`

## 运行
- 在项目根目录执行：
  - `python test/test_polyscope.py`
- 输出：
  - 视频：`test/cloth_polyscope.mp4`
  - 或图片帧：`test/frames_polyscope/frame_00000.png` 等
  - 同时会弹出 Polyscope 交互窗口用于预览（`ps.show()`）。

## 参数与自定义
- 替换/添加你的数据路径：
  - `cloth_data` 路径候选：`test/test_polyscope.py:13`（列表中按需添加你的 `.npy` 绝对路径即可）
  - `triangles` 路径候选：`test/test_polyscope.py:48`
- 视频参数：
  - 帧率：`fps` 在 `test/test_polyscope.py:74`
  - 分辨率：`frame_size` 在 `test/test_polyscope.py:75`
- 相机参数：
  - 轨道半径 `R`：`test/test_polyscope.py:83`
  - 相机高度 `height`：`test/test_polyscope.py:84`
- 渲染注册：
  - 表面网格：`test/test_polyscope.py:69`
  - 点云回退：`test/test_polyscope.py:72`

## 快速示例：保存你的数据
```python
import numpy as np
# cloth_data: 形状 (N, M, 3)
# triangles: 形状 (F, 3)
np.save(r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\cloth_data.npy", cloth_data)
np.save(r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\triangles.npy", triangles)
```

## 故障排查
- “ModuleNotFoundError: polyscope”：未安装 Polyscope，执行 `pip install polyscope`。
- 没有生成 mp4：未安装 `opencv-python`，已回退保存到 `frames_polyscope/`。安装后自动输出 mp4。
- 报错“未找到cloth_data.npy或verts_frames.npy”：请将你的数据保存到上述默认路径之一，或在候选列表中添加你的绝对路径（参考 `test/test_polyscope.py:13`）。
- 顶点数量不是完全平方数且无 `triangles.npy`：无法推断网格，脚本会改为点云动画；如需布料表面，请提供 `triangles.npy`。

## 设计说明
- 截图与视频写出：`test/test_polyscope.py:94`、`test/test_polyscope.py:95`、`test/test_polyscope.py:101-103`、`test/test_polyscope.py:111-115`
- 相机动态绕圈与逐帧更新：`test/test_polyscope.py:85-93`