import warp as wp
import numpy as np
import os
import argparse
from warp.sim.collide import TriMeshCollisionDetector
import time

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

def load_pair(data_path, input_dir):
    base = os.path.splitext(os.path.basename(data_path))[0]
    if not base.startswith("cloth_data_"):
        raise ValueError("数据文件名需为 cloth_data_*.npy")
    suffix = base[len("cloth_data_"):]
    topy_path = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    if not os.path.exists(topy_path):
        raise FileNotFoundError("未找到对应的 cloth_topy_*.npy")
    data = np.load(data_path)
    tris = np.load(topy_path).astype(np.int32)
    return data, tris

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "render", "input")
    
    # Ensure input dir exists
    if not os.path.exists(input_dir):
         print(f"Warning: Input directory {input_dir} does not exist.")
         # Fallback to current directory or user provided
         input_dir = script_dir

    files = list_input_files(input_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stop-on-first", action="store_true", help="Stop after finding the first intersecting frame")
    args = parser.parse_args()

    if not files and args.file is None:
        print("input 目录无 cloth_data_*.npy")
        return

    data_file = args.file
    if data_file is None or args.choose:
        if not files:
             print("No files found.")
             return
        data_file = choose_file_interactive(files, input_dir)
    else:
        if not os.path.exists(data_file):
            possible_path = os.path.join(input_dir, data_file)
            if os.path.exists(possible_path):
                data_file = possible_path
    
    print(f"Loading {data_file}...")
    data, triangles = load_pair(data_file, input_dir)
    N, M, _ = data.shape
    print(f"Data loaded. Frames: {N}, Vertices: {M}")

    device = wp.get_device(args.device)
    
    # Initialize model with frame 0
    verts_0 = data[0].astype(np.float32)
    vertices = [wp.vec3(float(verts_0[i, 0]), float(verts_0[i, 1]), float(verts_0[i, 2])) for i in range(M)]
    
    builder = wp.sim.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=vertices,
        indices=triangles.reshape(-1),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=2.0e-6,
        edge_ke=10,
    )
    
    try:
        builder.color()
    except Exception:
        pass
        
    model = builder.finalize(device=device)
    collision_detector = TriMeshCollisionDetector(model)
    
    intersecting_frames = []
    
    print("-" * 30)
    start_time = time.time()
    
    for f in range(N):
        verts_np = data[f].astype(np.float32)
        
        # Update model vertices
        collision_detector.refit(verts_np)
        
        # Detect intersections
        collision_detector.triangle_triangle_intersection_detection()
        
        counts = collision_detector.triangle_intersecting_triangles_count.numpy()
        total_intersections = int(counts.sum())
        
        if total_intersections > 0:
            print(f"Frame {f}: Total intersections: {total_intersections}")
            intersecting_frames.append(f)
            if args.stop_on_first:
                print("Stopping on first intersection.")
                break
        
        if f % 10 == 0:
            print(f"Processed frame {f}/{N}...", end="\r")

    print(f"\nProcessing complete in {time.time() - start_time:.2f}s")
    print("-" * 30)
    
    if intersecting_frames:
        print(f"Frames with intersections ({len(intersecting_frames)}):")
        print(intersecting_frames)
        print(f"Initial penetration frame: {intersecting_frames[0]}")
    else:
        print("No intersections found in any frame.")

if __name__ == "__main__":
    main()
