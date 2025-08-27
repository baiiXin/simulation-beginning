import warp as wp
import numpy as np
import trimesh
from warp.context import kernel
from warp.sim.collide import TriMeshCollisionDetector
import os

# device = wp.get_device('cuda')
# if you don't have a CUDA-compatible GPU try switching to CPU
device = wp.get_device('cpu')
print('=====分割线=====')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the PLY file
mesh_path = os.path.join(current_dir, 'Cube_subdivided.ply')

mesh = trimesh.load(mesh_path)

builder = wp.sim.ModelBuilder()

vertices = [wp.vec3(mesh.vertices[i,:]) for i in range(mesh.vertices.shape[0])]
print('vertices length:', len(vertices))

builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=mesh.faces.reshape(-1),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=2.0e-6,
            edge_ke=10,
)
builder.color()
model = builder.finalize()
print('model partical:', model.particle_count)
print('model triangle:', model.tri_count)
print('model edge:', model.edge_count)

# to access ForceElementAdjacencyInfo, you need to construct a VBDIntegrator (you dont need to understand what it is)
vbd_integrator = wp.sim.VBDIntegrator(model)

collision_detector = TriMeshCollisionDetector(model)

collision_detector.vertex_triangle_collision_detection(5.0)
collision_detector.edge_edge_collision_detection(5.0)

# d^v_{min}
print(collision_detector.vertex_colliding_triangles_min_dist)
# d^E_{min}
print(collision_detector.edge_colliding_edges_min_dist)
# d^T_{min}
print(collision_detector.triangle_colliding_vertices_min_dist)
print('d^v_{min}:', len(collision_detector.vertex_colliding_triangles_min_dist))
print('d^E_{min}:', len(collision_detector.edge_colliding_edges_min_dist))
print('d^T_{min}:', len(collision_detector.triangle_colliding_vertices_min_dist))

print('=====分割线=====')
from warp.sim.integrator_vbd import get_vertex_num_adjacent_edges, get_vertex_adjacent_edge_id_order, get_vertex_num_adjacent_faces, get_vertex_adjacent_face_id_order, ForceElementAdjacencyInfo
# how to iterate over neighbor elements
@wp.kernel
def iterate_vertex_neighbor_primitives(
    adjacency: ForceElementAdjacencyInfo,
    min_dis_v_t: wp.array(dtype=float),
    min_dis_e_e: wp.array(dtype=float),
    min_dis_t_v: wp.array(dtype=float),
    gama_p: float,
    # output
    bounds: wp.array(dtype=float),
):
    particle_idx = wp.tid()

    bounds[particle_idx] = min_dis_v_t[particle_idx]

    # iterating over neighbor faces
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_idx)
    for face_counter in range(num_adj_faces):
        adj_face_idx, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_idx, face_counter)
        bounds[particle_idx] = wp.min(bounds[particle_idx], min_dis_t_v[adj_face_idx])

    # iterating over neighbor edges
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_idx)
    for edge_counter in range(num_adj_edges):
        edge_idx, v_order = get_vertex_adjacent_edge_id_order(adjacency, particle_idx, edge_counter)
        bounds[particle_idx] = wp.min(bounds[particle_idx], min_dis_e_e[edge_idx])

    bounds[particle_idx] = gama_p * bounds[particle_idx]

gama_p = 0.4
bounds = wp.empty(shape=model.particle_count, dtype=float, device=device)

wp.launch(
    kernel=iterate_vertex_neighbor_primitives,
    inputs=[
        vbd_integrator.adjacency, 
        collision_detector.vertex_colliding_triangles_min_dist, 
        collision_detector.edge_colliding_edges_min_dist, 
        collision_detector.triangle_colliding_vertices_min_dist, 
        gama_p],
    outputs=[
        bounds],
    dim=model.particle_count,
    device=device
)
print(bounds)
print('bounds:', len(bounds))

# your turn: you need to:
# Implement conservative bounds computation using the instructions provided above
# It must be implemented using @warp.kernel to maximize efficiency


# cpmpute bounds
@wp.kernel
def compute_bounds(
    d_v: wp.array(dtype=float),
    d_e: wp.array(dtype=float),
    d_t: wp.array(dtype=float),
    b_v: wp.array(dtype=float)
):
    tid = wp.tid()  # 获取当前线程的索引
    b_v[tid] = wp.min(wp.min(d_v[tid], d_e[tid]), d_t[tid])  # 计算三个数组对应位置的最大值

b_v = wp.empty(shape=len(collision_detector.vertex_colliding_triangles_min_dist), dtype=float, device=device)

wp.launch(
    compute_bounds,
    dim=model.particle_count,
    inputs=[collision_detector.vertex_colliding_triangles_min_dist, 
        collision_detector.edge_colliding_edges_min_dist, 
        collision_detector.triangle_colliding_vertices_min_dist],
    outputs=[b_v],
    device=device
)
print(b_v)
print('b_v:', len(b_v))





# 将 Warp 数组转换为 NumPy 数组
vertex_min_dist_np = collision_detector.vertex_colliding_triangles_min_dist.numpy()
edge_min_dist_np = collision_detector.edge_colliding_edges_min_dist.numpy()
triangle_min_dist_np = collision_detector.triangle_colliding_vertices_min_dist.numpy()

# 打印数组长度
print('d^v_{min} length:', len(vertex_min_dist_np))
print('d^E_{min} length:', len(edge_min_dist_np))
print('d^T_{min} length:', len(triangle_min_dist_np))

# 求最小值并打印
if len(vertex_min_dist_np) > 0:
    d_v_min = np.min(vertex_min_dist_np)
    print(f'd^v_{{min}} value: {d_v_min}')
else:
    print('d^v_{min} array is empty')

if len(edge_min_dist_np) > 0:
    d_E_min = np.min(edge_min_dist_np)
    print(f'd^E_{{min}} value: {d_E_min}')
else:
    print('d^E_{min} array is empty')

if len(triangle_min_dist_np) > 0:
    d_T_min = np.min(triangle_min_dist_np)
    print(f'd^T_{{min}} value: {d_T_min}')
else:
    print('d^T_{min} array is empty')