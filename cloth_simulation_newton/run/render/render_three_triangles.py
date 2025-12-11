import os
import math
import numpy as np
import argparse

# --- 辅助函数：文件与路径处理 (非 Polyscope 核心) ---
def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def list_input_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
    # 按修改时间排序，方便找到最新的数据
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    return files

def choose_file_interactive(files, input_dir):
    """交互式命令行菜单，让用户选择要渲染的文件"""
    print("可选数据文件:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    print("请输入索引或文件名:")
    s = input().strip()
    # 支持输入绝对路径或文件名
    if os.path.isabs(s) or s.endswith(".npy"):
        p = s if os.path.isabs(s) else os.path.join(input_dir, s)
        if os.path.exists(p):
            return p
    # 支持输入数字索引
    if s.isdigit():
        i = int(s)
        i = max(0, min(i, len(files) - 1))
        return os.path.join(input_dir, files[i])
    return os.path.join(input_dir, files[0])

def load_cloth_data_and_topy_for(data_file):
    """
    加载数据。
    data: (帧数, 顶点数, 3) 的坐标数组。
    topy: (面数, 3) 的三角形索引数组 (拓扑结构)。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    suffix = os.path.splitext(os.path.basename(data_file))[0][len("cloth_data_"):]
    topy_file = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    
    data = np.load(data_file)
    topy = np.load(topy_file) if os.path.exists(topy_file) else None
    return data, topy

# --- 几何处理部分：构建网格与分离组件 ---

def build_grid_triangles(nm):
    """如果缺少拓扑文件，尝试根据顶点数量自动生成一个正方形网格的拓扑"""
    n = int(round(math.sqrt(nm)))
    if n * n != nm:
        return None
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            # 一个方格拆成两个三角形
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int32)

def load_triangles(nm, topy):
    """决定使用哪个拓扑结构"""
    if topy is not None:
        return topy
    if nm >= 7:
        # 硬编码的测试用例（通常不会走到这）
        return np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]], dtype=np.int32)
    return build_grid_triangles(nm)

def components_from_triangles(tris):
    """
    [算法重点] 使用并查集 (Union-Find) 将分离的网格拆解为独立的组件。
    目的：Polyscope 渲染时，如果把几块独立的布料混在一起渲染，视觉效果不好。
         拆分后，可以给每一块布料赋予不同的颜色。
    """
    # 1. 找出所有用到的顶点
    used = np.unique(tris.reshape(-1))
    parent = {}

    # 并查集标准实现：查找
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x]) # 路径压缩
        return parent[x]

    # 并查集标准实现：合并
    def union(a, b):
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    # 2. 遍历所有三角形，将同一个三角形的三个顶点连通
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        union(a, b)
        union(b, c)
        union(a, c)

    # 3. 按根节点分组，收集每个组件的三角形
    groups = {}
    for t in tris:
        r = find(int(t[0]))
        groups.setdefault(r, []).append(t.tolist())

    # 4. 重构数据结构：生成 (顶点索引列表, 局部三角形索引) 的元组
    comps = []
    for _, tri_list in groups.items():
        tri_arr = np.array(tri_list, dtype=np.int32)
        verts_unique = np.unique(tri_arr.reshape(-1))
        # 建立 全局索引 -> 组件局部索引 的映射
        mapping = {int(g): i for i, g in enumerate(verts_unique.tolist())}
        local = np.empty_like(tri_arr)
        for i in range(tri_arr.shape[0]):
            for j in range(3):
                local[i, j] = mapping[int(tri_arr[i, j])]
        comps.append((verts_unique, local))
    return comps

# --- 主程序：Polyscope 核心逻辑 ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    files = list_input_files(input_dir)

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", action="store_true", default=True) # 默认显示窗口
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()

    # 处理文件选择逻辑
    if args.list:
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        return
    if not files and args.file is None:
        raise FileNotFoundError("no cloth_data_*.npy in input")
    data_file = args.file
    if data_file is None or args.choose:
        data_file = choose_file_interactive(files, input_dir)

    # 加载数据：N=帧数, M=总顶点数
    cloth_data, topy = load_cloth_data_and_topy_for(data_file)
    N, M, _ = cloth_data.shape
    triangles = load_triangles(M, topy)

    # 导入 Polyscope
    import polyscope as ps
    # 尝试导入 cv2 或 imageio 用于视频保存
    try:
        import cv2
    except Exception:
        cv2 = None

    # [Polyscope 1] 初始化
    # 必须最先调用。它会创建 OpenGL 上下文和窗口后端。
    ps.init()

    # [Polyscope 2] 全局设置
    # set_up_dir: 告诉渲染器哪个轴是朝上的。物理仿真通常 Z 轴朝上，而图形学常用 Y 轴。
    # 这里设为 "z_up"，Polyscope 会自动调整相机操作习惯。
    ps.set_up_dir("z_up")
    
    # set_ground_plane_mode: "none" 表示隐藏默认的方格地面。
    # 默认是 "tile_reflection" (带倒影的地面)。
    ps.set_ground_plane_mode("none")

    meshes = [] # 用于存储注册好的网格对象引用

    # [Polyscope 3] 注册几何体 (Registering Structures)
    if triangles is not None:
        # 将整个大网格拆分成独立组件（如：衣服、裤子分开）
        comps = components_from_triangles(triangles)
        
        # 定义颜色盘，给不同组件上色
        palette = [
            (1.0, 0.2, 0.2), # 红
            (0.2, 0.6, 1.0), # 蓝
            (0.6, 1.0, 0.4), # 绿
            (1.0, 0.6, 0.0), # 橙
            (0.7, 0.3, 0.9), # 紫
            (0.2, 1.0, 0.8), # 青
        ]

        for k, (verts_unique, local_tris) in enumerate(comps):
            color = palette[k % len(palette)]
            
            # === 核心 API 调用 ===
            # ps.register_surface_mesh(name, vertices, faces, ...)
            # 参数:
            #   name: 字符串，UI里显示的唯一ID。
            #   vertices: (V, 3) 顶点坐标，这里取第0帧的数据作为初始状态。
            #   faces: (F, 3) 三角形面片索引。
            #   color: (R, G, B) 基础颜色。
            #   smooth_shade: False 表示使用平直着色(Flat Shading)，能看清每个三角形的棱角。
            mesh = ps.register_surface_mesh(
                f"component_{k}",              # 名字：component_0, component_1...
                cloth_data[0, verts_unique],   # 顶点：只取该组件用到的点
                local_tris,                    # 拓扑：局部三角形索引
                color=color,
                smooth_shade=False,
            )
            
            # 将 mesh 对象存起来，后面动画更新时要用
            meshes.append((mesh, verts_unique))
    else:
        # 如果没有三角形数据，退化为渲染点云 (Point Cloud)
        # ps.register_point_cloud(name, points, ...)
        pc = ps.register_point_cloud("cloth_points", cloth_data[0])
        try:
            # 设置点的显示半径，relative=True 表示半径是相对于场景大小的比例
            pc.set_radius(0.02, relative=True)
        except Exception:
            pass

    # [Polyscope 4] 自动计算相机位置 (Camera setup)
    # 计算所有帧的包围盒，确保不管物体怎么动，都在视野内
    all_min = np.min(cloth_data.reshape(-1, 3), axis=0)
    all_max = np.max(cloth_data.reshape(-1, 3), axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6: radius = 1.0 # 防止除零
    
    # 根据球坐标系计算相机位置
    R = 1.1 * radius      # 距离中心的半径
    az_deg = 150.0        # 方位角 (Azimuth)
    el_deg = 10.0         # 仰角 (Elevation)
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    
    # 球坐标转笛卡尔坐标
    x = R * math.cos(el) * math.cos(az)
    y = R * math.cos(el) * math.sin(az)
    z = R * math.sin(el)
    cam_pos = center + np.array([x, y, z])

    # ps.look_at(camera_position, target_position)
    # 强制将相机摆放在 cam_pos，并看向 center
    ps.look_at(cam_pos, center.tolist())

    # --- 视频录制准备 (OpenCV / ImageIO) ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "three_triangles.mp4")
    fps = 120
    frame_size = (1280, 720)
    tmp_path = os.path.join(output_dir, "frame_tmp.png") # 临时截图路径

    try:
        import imageio
    except Exception:
        imageio = None
    writer_cv2 = None
    writer_imageio = None

    if cv2 is not None:
        # 初始化 OpenCV 视频写入器
        writer_cv2 = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    elif imageio is not None:
        writer_imageio = imageio.get_writer(output_path, fps=float(fps))
    else:
        raise RuntimeError("opencv 和 imageio 不可用，无法写出 mp4")

    # [Polyscope 5] 渲染循环 (Animation Loop)
    print(f"开始渲染 {N} 帧...")
    for frame in range(N):
        # 步骤 A: 更新几何体数据
        for mesh, verts_idx in meshes:
            # 取出当前帧该组件的顶点位置
            verts = cloth_data[frame, verts_idx]
            
            # mesh.update_vertex_positions(new_positions)
            # 这是 Polyscope 动态渲染的关键。
            # 我们不需要重新 register (那样很慢)，只需要告诉 GPU 更新顶点位置 buffer。
            mesh.update_vertex_positions(verts)
        
        # 步骤 B: 锁定相机
        # 每一帧都重置相机位置，防止用户在脚本运行时鼠标乱动导致视频抖动
        # 如果你想做“跟随相机”，可以动态修改 cam_pos 或 center
        ps.look_at(cam_pos, center.tolist())
        
        # 步骤 C: 截图
        # ps.screenshot(path) 将当前视口保存为图片
        ps.screenshot(tmp_path)
        
        # 步骤 D: 将图片写入视频 (标准 OpenCV 操作)
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
            
        if frame % 20 == 0:
            print(f"渲染进度: {frame}/{N}")

    # 清理视频写入资源
    if writer_cv2 is not None:
        writer_cv2.release()
    if writer_imageio is not None:
        writer_imageio.close()
    
    # 删除临时截图
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    print(f"视频已保存至: {output_path}")

    # [Polyscope 6] 交互式展示
    # 视频录制完了，程序还没结束。
    # ps.show() 会弹出一个窗口，阻塞主线程。
    # 此时你可以用鼠标旋转、缩放、点击模型查看数据，直到手动关闭窗口。
    if args.show:
        try:
            import polyscope as ps
            print("打开交互式窗口... (关闭窗口以结束程序)")
            ps.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()