import warp as wp
import numpy as np
import trimesh
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

# to access ForceElementAdjacencyInfo, you need to construct a VBDIntegrator (you dont need to understand what it is)
vbd_integrator = wp.sim.VBDIntegrator(model)

colision_detector = TriMeshCollisionDetector(model)

colision_detector.vertex_triangle_collision_detection(5.0)
colision_detector.edge_edge_collision_detection(5.0)

# d^v_{min}
print(colision_detector.vertex_colliding_triangles_min_dist)
# d^E_{min}
print(colision_detector.edge_colliding_edges_min_dist)
# d^T_{min}
print(colision_detector.triangle_colliding_vertices_min_dist)
print('d^T_{min}:', len(colision_detector.vertex_colliding_triangles_min_dist))

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

b_v = wp.empty(shape=len(colision_detector.vertex_colliding_triangles_min_dist), dtype=float, device=device)

wp.launch(
    compute_bounds,
    dim=model.particle_count,
    inputs=[colision_detector.vertex_colliding_triangles_min_dist, colision_detector.edge_colliding_edges_min_dist, colision_detector.triangle_colliding_vertices_min_dist, b_v],
    device=device
)
print(b_v)
print('b_v:', len(b_v))

from warp.sim.integrator_vbd import get_vertex_num_adjacent_edges, get_vertex_adjacent_edge_id_order, get_vertex_num_adjacent_faces, get_vertex_adjacent_face_id_order, ForceElementAdjacencyInfo
# how to iterate over neighbor elements
@wp.kernel
def iterate_vertex_neighbor_primitives(
    adjacency: ForceElementAdjacencyInfo
):
    particle_idx = wp.tid()

    # iterating over neighbor faces
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_idx)
    for face_counter in range(num_adj_faces):
        adj_face_idx, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_idx, face_counter)
    # iterating over neighbor edges
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_idx)
    for edge_counter in range(num_adj_edges):
        edge_idx, v_order = get_vertex_adjacent_edge_id_order(adjacency, particle_idx, edge_counter)

wp.launch(
    iterate_vertex_neighbor_primitives,
    dim=model.particle_count,
    inputs=[vbd_integrator.adjacency],
    device=device
)


# your turn: you need to:
# Implement conservative bounds computation using the instructions provided above
# It must be implemented using @warp.kernel to maximize efficiency