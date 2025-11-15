# 文件用途：使用 Polyscope 对布料仿真数据进行可视化并导出视频/帧
# 注释分块：导入、数据加载、网格构建、三角面加载、可视化主流程

# 导入依赖
import os
import math
import numpy as np

# 在候选路径列表中查找首个存在的文件路径
def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# 加载布料顶点数据（优先完整帧序列，退路为单帧）
def load_cloth_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 多路径候选，兼容相对路径与绝对路径
    candidates = [
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "cloth_data.npy"),
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "verts_frames.npy"),
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "cloth_all_frame.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\cloth_data.npy",
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\verts_frames.npy",
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\cloth_all_frame.npy",
    ]
    # 优先加载完整帧序列 (N, M, 3)
    p = _find_file(candidates)
    if p is not None:
        return np.load(p)
    # 退路：只有最后一帧时，将其扩展为 (1, M, 3)
    alt = _find_file([
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "verts_last_frame.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\verts_last_frame.npy",
    ])
    if alt is not None:
        v = np.load(alt)
        return np.expand_dims(v, 0)
    # 均找不到时抛出异常（中文信息）
    raise FileNotFoundError("未找到cloth_data.npy或verts_frames.npy")

# 根据顶点数 nm 构造 n×n 网格的两三角剖分
def build_grid_triangles(nm):
    n = int(round(math.sqrt(nm)))
    if n * n != nm:
        # 顶点数量无法组成正方形网格时返回 None
        return None
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            # 每个网格单元拆分成两个三角形
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int32)

# 加载三角面索引（磁盘优先，否则基于网格构建）
def load_triangles(nm):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "triangles.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\triangles.npy",
    ]
    p = _find_file(candidates)
    if p is not None:
        return np.load(p)
    # 未找到预生成索引时，按网格顶点数构建默认三角面
    return build_grid_triangles(nm)

# 主流程：初始化 Polyscope、注册对象、环绕拍摄并导出
def main():
    # 1) 加载数据与三角面
    cloth_data = load_cloth_data()
    N, M, _ = cloth_data.shape
    triangles = load_triangles(M)
    try:
        import cv2
    except Exception:
        cv2 = None
    import polyscope as ps
    # 3) 初始化 Polyscope 渲染环境
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    # 4) 注册对象：优先表面网格，否则点云
    if triangles is not None:
        mesh = ps.register_surface_mesh("cloth", cloth_data[0], triangles, color=(0.5, 0.7, 1.0), smooth_shade=True)
    else:
        mesh = ps.register_point_cloud("cloth_points", cloth_data[0])
        try:
            mesh.set_radius(0.02, relative=True)
        except Exception:
            pass
    # 5) 输出配置：视频或帧序列
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloth_polyscope.mp4")
    fps = 240
    frame_size = (1280, 720)
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_tmp.png")
    if cv2 is not None:
        # 使用 OpenCV 写出 mp4 视频
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    else:
        writer = None
        frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_polyscope")
        os.makedirs(frames_dir, exist_ok=True)
    # 6) 视角与包围盒：计算中心、半径、相机轨迹
    all_min = np.min(cloth_data.reshape(-1, 3), axis=0)
    all_max = np.max(cloth_data.reshape(-1, 3), axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    R = max(1.5, 1.2 * radius)
    height = R
    # 7) 帧循环：更新顶点、绕圈拍摄、写出视频/帧
    for frame in range(N):
        verts = cloth_data[frame]
        if hasattr(mesh, "update_vertex_positions"):
            mesh.update_vertex_positions(verts)
        else:
            mesh.update_point_positions(verts)
        theta = 2 * math.pi * frame / max(N, 1)
        # 环绕相机位置（水平圆轨迹 + 固定高度）
        cam_pos = center + np.array([R * math.cos(theta), R * math.sin(theta), height])
        ps.look_at(cam_pos, center.tolist())
        # 截屏到临时文件
        ps.screenshot(tmp_path)
        if writer is not None:
            img = None
            try:
                img = __import__("cv2").imread(tmp_path)
            except Exception:
                img = None
            if img is not None:
                img = __import__("cv2").resize(img, frame_size)
                writer.write(img)
        else:
            name = os.path.join(frames_dir, f"frame_{frame:05d}.png")
            try:
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, name)
            except Exception:
                pass
    # 8) 资源清理与展示
    if writer is not None:
        writer.release()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
    ps.show()

if __name__ == "__main__":
    main()