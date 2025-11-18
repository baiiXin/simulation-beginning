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

def main():
    cloth_data = load_cloth_data()
    N, M, _ = cloth_data.shape
    idxA = np.array([0, 1, 2], dtype=np.int32)
    idxB = np.array([3, 4, 5], dtype=np.int32)
    import polyscope as ps
    try:
        import cv2
    except Exception:
        cv2 = None
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    meshA = ps.register_surface_mesh(
        "large_triangle",
        cloth_data[0, idxA],
        np.array([[0, 1, 2]], dtype=np.int32),
        color=(1.0, 0.2, 0.2),
        smooth_shade=False,
    )
    meshB = ps.register_surface_mesh(
        "small_triangle",
        cloth_data[0, idxB],
        np.array([[0, 1, 2]], dtype=np.int32),
        color=(0.2, 0.6, 1.0),
        smooth_shade=False,
    )
    all_min = np.min(cloth_data.reshape(-1, 3), axis=0)
    all_max = np.max(cloth_data.reshape(-1, 3), axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    R = max(1.5, 1.2 * radius)
    cam_pos = center + np.array([R, 0.0, 0.0])
    ps.look_at(cam_pos, center.tolist())
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_triangles.mp4")
    fps = 240
    frame_size = (1280, 720)
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_tmp.png")
    if cv2 is not None:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    else:
        writer = None
        frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_two_triangles")
        os.makedirs(frames_dir, exist_ok=True)
    for frame in range(N):
        vertsA = cloth_data[frame, idxA]
        vertsB = cloth_data[frame, idxB]
        meshA.update_vertex_positions(vertsA)
        meshB.update_vertex_positions(vertsB)
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