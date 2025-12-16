# 文件用途：使用 Polyscope 对布料仿真数据进行可视化并导出视频/帧
# 注释分块：导入、数据加载、网格构建、三角面加载、可视化主流程

# 导入依赖
import os
import math
import numpy as np
import argparse

# 在候选路径列表中查找首个存在的文件路径
def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# 加载布料顶点数据（优先完整帧序列，退路为单帧）
def list_input_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    return files

def choose_file_interactive(files, input_dir):
    print("可选数据文件:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    print("请输入索引或文件名:")
    s = input().strip()
    if os.path.isabs(s) or s.endswith(".npy"):
        p = s if os.path.isabs(s) else os.path.join(input_dir, s)
        if os.path.exists(p):
            return p
    if s.isdigit():
        i = int(s)
        i = max(0, min(i, len(files) - 1))
        return os.path.join(input_dir, files[i])
    return os.path.join(input_dir, files[0])

def load_cloth_data_and_topy_for(data_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    suffix = os.path.splitext(os.path.basename(data_file))[0][len("cloth_data_"):]
    topy_file = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    data = np.load(data_file)
    topy = np.load(topy_file) if os.path.exists(topy_file) else None
    # 打印加载信息
    print(f"loaded {data_file} with shape {data.shape}")
    if topy is not None:
        print(f"loaded {topy_file} with shape {topy.shape}")
    return data, topy

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
def load_triangles(nm, topy):
    if topy is not None:
        return topy
    return build_grid_triangles(nm)

# 主流程：初始化 Polyscope、注册对象、环绕拍摄并导出
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    files = list_input_files(input_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    if args.list:
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        return
    if not files and args.file is None:
        raise FileNotFoundError("no cloth_data_*.npy in input")
    data_file = args.file
    if data_file is None or args.choose:
        data_file = choose_file_interactive(files, input_dir)
    cloth_data, topy = load_cloth_data_and_topy_for(data_file)
    N, M, _ = cloth_data.shape
    triangles = load_triangles(M, topy)
    try:
        import cv2
    except Exception:
        cv2 = None
    import polyscope as ps
    # 3) 初始化 Polyscope 渲染环境
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cloth_polyscope.mp4")
    fps = 240
    frame_size = (1920, 1080)
    tmp_path = os.path.join(output_dir, "frame_tmp.png")
    try:
        import imageio
    except Exception:
        imageio = None
    writer_cv2 = None
    writer_imageio = None
    if cv2 is not None:
        writer_cv2 = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    elif imageio is not None:
        writer_imageio = imageio.get_writer(output_path, fps=float(fps))
    else:
        raise RuntimeError("opencv 和 imageio 不可用，无法写出 mp4")
    # 6) 视角与包围盒：计算中心、半径、相机轨迹
    frame0 = cloth_data[0]
    all_min = np.min(frame0, axis=0)
    all_max = np.max(frame0, axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    R = 1.2 * radius
    height = 0.8 * R
    cam_pos = center + np.array([R, 0.0, height])
    ps.look_at(cam_pos, center.tolist())
    for frame in range(N):
        verts = cloth_data[frame]
        if hasattr(mesh, "update_vertex_positions"):
            mesh.update_vertex_positions(verts)
        else:
            mesh.update_point_positions(verts)
        
        # 截屏到临时文件
        ps.screenshot(tmp_path)
        if writer_cv2 is not None:
            img = cv2.imread(tmp_path)
            if img is not None:
                h, w = img.shape[:2]
                if (w, h) != frame_size:
                    img = cv2.resize(img, frame_size)
                writer_cv2.write(img)
        elif writer_imageio is not None:
            im = imageio.imread(tmp_path)
            writer_imageio.append_data(im)
    # 8) 资源清理与展示
    if writer_cv2 is not None:
        writer_cv2.release()
    if writer_imageio is not None:
        writer_imageio.close()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
    if args.show:
        try:
            import polyscope as ps
            # ps.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()