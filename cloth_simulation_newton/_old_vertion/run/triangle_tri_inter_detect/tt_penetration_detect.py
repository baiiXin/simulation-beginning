import numpy as np
import os
import argparse
import time
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# File I/O Helpers (Borrowed from existing scripts)
# -----------------------------------------------------------------------------

def list_input_files(input_dir):
    if not os.path.exists(input_dir):
        return []
    files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    return files

def choose_file_interactive(files, input_dir):
    print("可选数据文件:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    print("请输入索引或文件名:")
    try:
        s = input().strip()
    except EOFError:
        s = "0"
        
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
        raise FileNotFoundError(f"未找到对应的 cloth_topy_*.npy: {topy_path}")
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    tris = np.load(topy_path).astype(np.int32)
    return data, tris

def choose_triangles_interactive():
    print("是否需要打印特定三角形的坐标？(y/n) [默认: n]")
    s = input().strip().lower()
    if s != 'y':
        return []
    
    print("请输入三角形索引，以空格分隔:")
    try:
        line = input().strip()
        parts = line.replace(',', ' ').split()
        indices = [int(x) for x in parts]
        return indices
    except ValueError:
        print("输入无效，不打印三角形。")
        return []

# -----------------------------------------------------------------------------
# Geometric Primitives (Double Precision)
# -----------------------------------------------------------------------------

def cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float64)

def dot(a, b):
    return np.sum(a * b)

# -----------------------------------------------------------------------------
# Distance Calculation Helpers
# -----------------------------------------------------------------------------

def sub(a, b):
    return a - b

def add(a, b):
    return a + b

def mul(a, s):
    return a * s

def mag2(a):
    return dot(a, a)

def dist_segment_segment(p1, p2, q1, q2):
    # Shortest distance between segment p1-p2 and q1-q2
    EPSILON = 1e-12
    d1 = sub(p2, p1)
    d2 = sub(q2, q1)
    r = sub(p1, q1)
    a = dot(d1, d1)
    e = dot(d2, d2)
    f = dot(d2, r)
    
    if a <= EPSILON and e <= EPSILON:
        # Both segments are points
        return np.sqrt(dot(r, r))
    
    if a <= EPSILON:
        # First segment is a point
        s = 0.0
        t = f / e
        t = np.clip(t, 0.0, 1.0)
    elif e <= EPSILON:
        # Second segment is a point
        t = 0.0
        s = -dot(d1, r) / a
        s = np.clip(s, 0.0, 1.0)
    else:
        # General case
        c = dot(d1, r)
        b = dot(d1, d2)
        denom = a * e - b * b
        
        if denom != 0.0:
            s = (b * f - c * e) / denom
        else:
            s = 0.0 # Parallel
            
        t = (b * s + f) / e
        
        if s < 0.0:
            s = 0.0
            t = f / e
            t = np.clip(t, 0.0, 1.0)
        elif s > 1.0:
            s = 1.0
            t = (b + f) / e
            t = np.clip(t, 0.0, 1.0)
            
        t2 = (b * s + f) / e
        if t2 < 0.0:
            t = 0.0
            s = -c / a
            s = np.clip(s, 0.0, 1.0)
        elif t2 > 1.0:
            t = 1.0
            s = (b - c) / a
            s = np.clip(s, 0.0, 1.0)
            
    c1 = add(p1, mul(d1, s))
    c2 = add(q1, mul(d2, t))
    return np.sqrt(mag2(sub(c1, c2)))

def dist_point_triangle_face(p, a, b, c):
    # Distance from p to triangle abc, ONLY considering the face interior.
    # If projection falls outside, returns infinity (handled by edge checks elsewhere).
    ab = sub(b, a)
    ac = sub(c, a)
    ap = sub(p, a)
    
    n = cross(ab, ac)
    n_len2 = dot(n, n)
    if n_len2 < 1e-12:
        return float('inf') # Degenerate
        
    # Projected point
    dist_plane = dot(ap, n) / np.sqrt(n_len2)
    proj = sub(p, mul(n, dist_plane / np.sqrt(n_len2)))
    
    # Barycentric check
    # v0 = ab, v1 = ac, v2 = proj - a
    v0 = ab
    v1 = ac
    v2 = sub(proj, a)
    
    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return float('inf')
        
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    if v >= 0 and w >= 0 and u >= 0:
        return abs(dist_plane)
    return float('inf')

def compute_triangle_distance(t1, t2):
    # t1, t2: (3, 3) arrays
    min_dist = float('inf')
    
    # 1. Edge-Edge distances (9 pairs)
    for i in range(3):
        edge1_p1 = t1[i]
        edge1_p2 = t1[(i+1)%3]
        for j in range(3):
            edge2_p1 = t2[j]
            edge2_p2 = t2[(j+1)%3]
            d = dist_segment_segment(edge1_p1, edge1_p2, edge2_p1, edge2_p2)
            if d < min_dist:
                min_dist = d
                
    # 2. Vertex-Face distances (6 pairs)
    # Vertices of T1 against Face T2
    for i in range(3):
        d = dist_point_triangle_face(t1[i], t2[0], t2[1], t2[2])
        if d < min_dist:
            min_dist = d
            
    # Vertices of T2 against Face T1
    for i in range(3):
        d = dist_point_triangle_face(t2[i], t1[0], t1[1], t1[2])
        if d < min_dist:
            min_dist = d
            
    return min_dist

def intersect_triangle_triangle(t1, t2):
    """
    Möller-Trumbore (1997) Triangle-Triangle Intersection Test
    t1, t2: (3, 3) numpy arrays of float64
    """
    EPSILON = 1e-12
    
    p1, p2, p3 = t1[0], t1[1], t1[2]
    q1, q2, q3 = t2[0], t2[1], t2[2]
    
    # 1. Compute Plane equation of T2
    v2_0 = q2 - q1
    v2_1 = q3 - q1
    N2 = cross(v2_0, v2_1)
    d2 = -dot(N2, q1)
    
    # Signed distances of T1 vertices to Plane T2
    dv1 = dot(N2, p1) + d2
    dv2 = dot(N2, p2) + d2
    dv3 = dot(N2, p3) + d2
    
    # Check if T1 is completely on one side of Plane T2
    if (dv1 > EPSILON and dv2 > EPSILON and dv3 > EPSILON) or \
       (dv1 < -EPSILON and dv2 < -EPSILON and dv3 < -EPSILON):
        return False
        
    # 2. Compute Plane equation of T1
    v1_0 = p2 - p1
    v1_1 = p3 - p1
    N1 = cross(v1_0, v1_1)
    d1 = -dot(N1, p1)
    
    # Signed distances of T2 vertices to Plane T1
    dq1 = dot(N1, q1) + d1
    dq2 = dot(N1, q2) + d1
    dq3 = dot(N1, q3) + d1
    
    # Check if T2 is completely on one side of Plane T1
    if (dq1 > EPSILON and dq2 > EPSILON and dq3 > EPSILON) or \
       (dq1 < -EPSILON and dq2 < -EPSILON and dq3 < -EPSILON):
        return False
        
    # 3. Compute direction of intersection line
    D = cross(N1, N2)
    
    # Check for coplanarity
    len_D_sq = dot(D, D)
    if len_D_sq < EPSILON:
        # Coplanar triangles are not considered intersecting for this check
        # (unless we wanted to handle 2D overlap)
        return False 
        
    # 4. Compute intervals on the line of intersection L = P + t * D
    
    # Projection of vertices onto D (This is sufficient for ordering)
    proj_p = np.array([dot(D, p1), dot(D, p2), dot(D, p3)])
    proj_q = np.array([dot(D, q1), dot(D, q2), dot(D, q3)])
    
    dv = np.array([dv1, dv2, dv3])
    dq = np.array([dq1, dq2, dq3])
    
    def compute_interval(dists, projs):
        # Find intersection segment of triangle with the other plane on line D
        inter_points = []
        
        # Check edges
        for i in range(3):
            j = (i + 1) % 3
            d_i = dists[i]
            d_j = dists[j]
            
            if (d_i > EPSILON and d_j < -EPSILON) or (d_i < -EPSILON and d_j > EPSILON):
                # Edge crosses plane
                t = d_i / (d_i - d_j)
                pt = projs[i] + t * (projs[j] - projs[i])
                inter_points.append(pt)
            elif abs(d_i) <= EPSILON:
                # Vertex on plane
                inter_points.append(projs[i])
        
        if len(inter_points) < 2:
            # Point contact or no intersection found (should have been caught)
            if len(inter_points) == 1:
                return inter_points[0], inter_points[0]
            return 0.0, 0.0
            
        return min(inter_points), max(inter_points)

    t1_min, t1_max = compute_interval(dv, proj_p)
    t2_min, t2_max = compute_interval(dq, proj_q)
    
    # Check interval overlap
    # [t1_min, t1_max] vs [t2_min, t2_max]
    if t1_max < t2_min - EPSILON or t2_max < t1_min - EPSILON:
        return False
        
    return True

# -----------------------------------------------------------------------------
# Broad Phase (Spatial Hashing)
# -----------------------------------------------------------------------------

class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
        
    def add(self, tri_idx, aabb_min, aabb_max):
        # Determine cell range
        min_idx = np.floor(aabb_min / self.cell_size).astype(int)
        max_idx = np.floor(aabb_max / self.cell_size).astype(int)
        
        # Clamp range to avoid huge loops if something blows up
        if np.any(max_idx - min_idx > 100):
             # Fallback for huge triangles: just center
             center = (aabb_min + aabb_max) * 0.5
             idx = np.floor(center / self.cell_size).astype(int)
             min_idx = idx
             max_idx = idx
        
        for i in range(min_idx[0], max_idx[0] + 1):
            for j in range(min_idx[1], max_idx[1] + 1):
                for k in range(min_idx[2], max_idx[2] + 1):
                    key = (i, j, k)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(tri_idx)
                    
    def get_candidates(self):
        # Return list of unique pairs
        candidates = set()
        for key, indices in self.grid.items():
            if len(indices) > 1:
                # Generate pairs
                n = len(indices)
                for i in range(n):
                    for j in range(i + 1, n):
                        idx1 = indices[i]
                        idx2 = indices[j]
                        if idx1 < idx2:
                            candidates.add((idx1, idx2))
                        else:
                            candidates.add((idx2, idx1))
        return list(candidates)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Borrowed path logic
    input_dir = os.path.join(script_dir, "..", "render", "input")
    
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--stop-on-first", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--print-tris", type=int, nargs='+', help="Indices of triangles to print coordinates for")
    args = parser.parse_args()
    
    files = list_input_files(input_dir)
    if not files and args.file is None:
        print(f"No cloth_data_*.npy files found in {input_dir}")
        return
        
    data_file = args.file
    if data_file is None or args.choose:
        if not files:
             print("No files available.")
             return
        data_file = choose_file_interactive(files, input_dir)
    else:
        if not os.path.exists(data_file):
            p = os.path.join(input_dir, data_file)
            if os.path.exists(p):
                data_file = p
                
    data, triangles = load_pair(data_file, input_dir)
    N_frames, M_verts, _ = data.shape
    print(f"Loaded. Frames: {N_frames}, Vertices: {M_verts}, Triangles: {len(triangles)}")
    
    triangles_to_print = args.print_tris
    if triangles_to_print is None:
        triangles_to_print = choose_triangles_interactive()

    intersecting_frames = []
    start_f = args.start_frame
    end_f = args.end_frame if args.end_frame >= 0 else N_frames
    
    print("-" * 40)
    print("Starting Double Precision Penetration Detection...")
    t0_total = time.time()
    
    for f in range(start_f, end_f):
        verts = data[f].astype(np.float64) # Ensure double precision
        
        if triangles_to_print:
            print(f"\n--- Visualizing Frame {f} Selected Triangles (Scaled x100) ---")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            all_points = []
            
            for t_idx in triangles_to_print:
                if 0 <= t_idx < len(triangles):
                    t_indices = triangles[t_idx]
                    # Scale x100
                    t_coords = verts[t_indices] * 100.0
                    
                    # 0->1->2->0
                    xs = [t_coords[0,0], t_coords[1,0], t_coords[2,0], t_coords[0,0]]
                    ys = [t_coords[0,1], t_coords[1,1], t_coords[2,1], t_coords[0,1]]
                    zs = [t_coords[0,2], t_coords[1,2], t_coords[2,2], t_coords[0,2]]
                    
                    ax.plot(xs, ys, zs, label=f'Tri {t_idx}')
                    
                    all_points.extend(t_coords)
                else:
                    print(f"Triangle {t_idx}: Index out of bounds")
            
            if len(triangles_to_print) == 2:
                # Calculate and print distance
                idx1, idx2 = triangles_to_print[0], triangles_to_print[1]
                if 0 <= idx1 < len(triangles) and 0 <= idx2 < len(triangles):
                    tri1 = verts[triangles[idx1]]
                    tri2 = verts[triangles[idx2]]
                    dist = compute_triangle_distance(tri1, tri2)
                    print(f"*** Distance between Triangle {idx1} and {idx2}: {dist:.6e} ***")

            if len(triangles_to_print) == 2:
                # Calculate and print distance
                idx1, idx2 = triangles_to_print[0], triangles_to_print[1]
                if 0 <= idx1 < len(triangles) and 0 <= idx2 < len(triangles):
                    tri1 = verts[triangles[idx1]]
                    tri2 = verts[triangles[idx2]]
                    dist = compute_triangle_distance(tri1, tri2)
                    print(f"*** Distance between Triangle {idx1} and {idx2}: {dist:.6e} ***")

            if len(triangles_to_print) == 2:
                # Calculate and print distance
                idx1, idx2 = triangles_to_print[0], triangles_to_print[1]
                if 0 <= idx1 < len(triangles) and 0 <= idx2 < len(triangles):
                    tri1 = verts[triangles[idx1]]
                    tri2 = verts[triangles[idx2]]
                    dist = compute_triangle_distance(tri1, tri2)
                    print(f"*** Distance between Triangle {idx1} and {idx2}: {dist:.6e} ***")

            if all_points:
                all_points = np.array(all_points)
                # Equal aspect ratio hack for 3D
                # Create a cubic bounding box around the data
                max_range = (np.max(all_points, axis=0) - np.min(all_points, axis=0)).max() / 2.0
                mid_vals = (np.max(all_points, axis=0) + np.min(all_points, axis=0)) * 0.5
                
                ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
                ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
                ax.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f'Frame {f} (Scale x100)')
            plt.legend()
            plt.show()

        # --- Broad Phase ---
        tri_verts = verts[triangles] # (NumTri, 3, 3)
        min_vals = np.min(tri_verts, axis=1)
        max_vals = np.max(tri_verts, axis=1)
        
        # Determine cell size based on average bounding box size
        if f == start_f:
             extent = max_vals - min_vals
             avg_dim = np.mean(extent)
             cell_size = max(avg_dim * 2.0, 1e-5)
             print(f"Using Spatial Hash Cell Size: {cell_size:.6f}")
             
        spatial = SpatialHash(cell_size)
        for t_i in range(len(triangles)):
            spatial.add(t_i, min_vals[t_i], max_vals[t_i])
            
        candidates = spatial.get_candidates()
        
        # --- Narrow Phase ---
        intersections = 0
        for (idx1, idx2) in candidates:
            # Adjacency Check (Ignore shared vertices)
            t1_idx = triangles[idx1]
            t2_idx = triangles[idx2]
            
            # Quick check if they share any vertex index
            if (t1_idx[0] in t2_idx) or (t1_idx[1] in t2_idx) or (t1_idx[2] in t2_idx):
                continue
                
            # AABB Overlap Refinement
            if not (min_vals[idx1][0] > max_vals[idx2][0] or max_vals[idx1][0] < min_vals[idx2][0] or
                    min_vals[idx1][1] > max_vals[idx2][1] or max_vals[idx1][1] < min_vals[idx2][1] or
                    min_vals[idx1][2] > max_vals[idx2][2] or max_vals[idx1][2] < min_vals[idx2][2]):
                
                # Precise Geometric Test
                t1 = tri_verts[idx1]
                t2 = tri_verts[idx2]
                
                if intersect_triangle_triangle(t1, t2):
                    intersections += 1
        
        if intersections > 0:
            print(f"Frame {f}: {intersections} intersections found.")
            intersecting_frames.append(f)
            if args.stop_on_first:
                print("Stopping on first intersection.")
                break
        
        if f % 10 == 0 or f == end_f - 1:
             print(f"Processed frame {f}/{end_f} (Found {len(intersecting_frames)} frames so far)...", end="\r")
             
    print(f"\nProcessing complete in {time.time() - t0_total:.2f}s")
    if intersecting_frames:
        print(f"Frames with intersections: {intersecting_frames}")
        print(f"Initial penetration frame: {intersecting_frames[0]}")
    else:
        print("No intersections found.")

if __name__ == "__main__":
    main()