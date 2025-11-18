import os
import math
import numpy as np
import argparse

def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

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
    input_dir = os.path.join(script_dir, "input")
    suffix = os.path.splitext(os.path.basename(data_file))[0][len("cloth_data_"):]
    topy_file = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    data = np.load(data_file)
    topy = np.load(topy_file) if os.path.exists(topy_file) else None
    return data, topy

def build_grid_triangles(nm):
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
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int32)

def load_triangles(nm, topy):
    if topy is not None:
        return topy
    if nm >= 7:
        return np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]], dtype=np.int32)
    return build_grid_triangles(nm)

def components_from_triangles(tris):
    used = np.unique(tris.reshape(-1))
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        union(a, b)
        union(b, c)
        union(a, c)
    groups = {}
    for t in tris:
        r = find(int(t[0]))
        groups.setdefault(r, []).append(t.tolist())
    comps = []
    for _, tri_list in groups.items():
        tri_arr = np.array(tri_list, dtype=np.int32)
        verts_unique = np.unique(tri_arr.reshape(-1))
        mapping = {int(g): i for i, g in enumerate(verts_unique.tolist())}
        local = np.empty_like(tri_arr)
        for i in range(tri_arr.shape[0]):
            for j in range(3):
                local[i, j] = mapping[int(tri_arr[i, j])]
        comps.append((verts_unique, local))
    return comps

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
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
    import polyscope as ps
    try:
        import cv2
    except Exception:
        cv2 = None
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    meshes = []
    if triangles is not None:
        comps = components_from_triangles(triangles)
        palette = [
            (1.0, 0.2, 0.2),
            (0.2, 0.6, 1.0),
            (0.6, 1.0, 0.4),
            (1.0, 0.6, 0.0),
            (0.7, 0.3, 0.9),
            (0.2, 1.0, 0.8),
        ]
        for k, (verts_unique, local_tris) in enumerate(comps):
            color = palette[k % len(palette)]
            mesh = ps.register_surface_mesh(
                f"component_{k}",
                cloth_data[0, verts_unique],
                local_tris,
                color=color,
                smooth_shade=False,
            )
            meshes.append((mesh, verts_unique))
    else:
        pc = ps.register_point_cloud("cloth_points", cloth_data[0])
        try:
            pc.set_radius(0.02, relative=True)
        except Exception:
            pass
    all_min = np.min(cloth_data.reshape(-1, 3), axis=0)
    all_max = np.max(cloth_data.reshape(-1, 3), axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    R = max(1.5, 1.1 * radius)
    az_deg = 150.0
    el_deg = 10.0
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = R * math.cos(el) * math.cos(az)
    y = R * math.cos(el) * math.sin(az)
    z = R * math.sin(el)
    cam_pos = center + np.array([x, y, z])
    ps.look_at(cam_pos, center.tolist())
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "three_triangles.mp4")
    fps = 240
    frame_size = (1280, 720)
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
    for frame in range(N):
        for mesh, verts_idx in meshes:
            verts = cloth_data[frame, verts_idx]
            mesh.update_vertex_positions(verts)
        ps.look_at(cam_pos, center.tolist())
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
            ps.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()