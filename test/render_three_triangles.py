import os
import math
import numpy as np

def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_cloth_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "cloth_data.npy"),
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "cloth_all_frame.npy"),
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "verts_frames.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\cloth_data.npy",
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\cloth_all_frame.npy",
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\verts_frames.npy",
    ]
    p = _find_file(candidates)
    if p is not None:
        return np.load(p)
    alt = _find_file([
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "verts_last_frame.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\verts_last_frame.npy",
    ])
    if alt is not None:
        v = np.load(alt)
        return np.expand_dims(v, 0)
    raise FileNotFoundError("no cloth data")

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

def load_triangles(nm):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "cloth_simulation_newton", "data", "triangles.npy"),
        r"D:\Z-Documents\simulation\repo_sim_cloth\cloth_simulation_newton\data\triangles.npy",
    ]
    p = _find_file(candidates)
    if p is not None:
        return np.load(p)
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
    cloth_data = load_cloth_data()
    N, M, _ = cloth_data.shape
    triangles = load_triangles(M)
    import polyscope as ps
    try:
        import cv2
    except Exception:
        cv2 = None
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
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
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_triangles.mp4")
    fps = 240
    frame_size = (1280, 720)
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_tmp.png")
    if cv2 is not None:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    else:
        writer = None
        frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_three_triangles")
        os.makedirs(frames_dir, exist_ok=True)
    for frame in range(N):
        for mesh, verts_idx in meshes:
            verts = cloth_data[frame, verts_idx]
            mesh.update_vertex_positions(verts)
        ps.look_at(cam_pos, center.tolist())
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