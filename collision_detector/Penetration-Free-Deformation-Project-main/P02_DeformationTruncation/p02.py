import warp as wp
import numpy as np
import trimesh

from warp.sim.collide import TriMeshCollisionDetector

import glob
import os
from os.path import join
import numpy as np

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Correct path to the Data directory
data_path = os.path.join(current_dir, 'Data')

shape_pre_initialization_files = sorted(glob.glob(join(data_path, "step_*_pre_initialization_shape.ply")))
displacement_initialization_files = sorted(glob.glob(join(data_path, "step_*_initialization_displacement.npy")))

shape_pre_iteration_files = sorted(glob.glob(join(data_path, "step_*_iter_*_shape.ply")))
displacement_iteration_files = sorted(glob.glob(join(data_path, "step_*_iter_*_displacement.npy")))

# device = wp.get_device('cuda')
# if you don't have a CUDA-compatible GPU try switching to CPU
device = wp.get_device('cpu')

print('=====分割线=====')
print('shape_pre_initialization_files', type(shape_pre_initialization_files),len(shape_pre_initialization_files))
print('displacement_initialization_files', type(displacement_initialization_files),len(displacement_initialization_files))
print('shape_pre_iteration_files', type(shape_pre_iteration_files),len(shape_pre_iteration_files))
print('displacement_iteration_files', type(displacement_iteration_files),len(displacement_iteration_files))

print('=====分割线=====')

print('shape_pre_initialization_files[0]', shape_pre_initialization_files[0])

print('=====分割线=====')

mesh_pre_init = trimesh.load(shape_pre_initialization_files[0])
vertices = [wp.vec3(mesh_pre_init.vertices[i,:]) for i in range(mesh_pre_init.vertices.shape[0])]
print('vertices length:', len(vertices))

displacement_init_np = np.load(displacement_initialization_files[0])
displacement_init = [wp.vec3(displacement_init_np[i,:]) for i in range(displacement_init_np.shape[0])]
print('displacement_init:', len(displacement_init))

builder = wp.sim.ModelBuilder()
builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=mesh_pre_init.faces.reshape(-1),
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
collision_detector = TriMeshCollisionDetector(model)

collision_detector.vertex_triangle_collision_detection(5.0)
collision_detector.edge_edge_collision_detection(5.0)

# 检查
print('model.tri_indices', model.tri_indices.shape)
print('model.edge_indices', model.edge_indices.shape)

# TODO: compute the conservative bounds
# cpmpute bounds
print('=====TODO: compute the conservative bounds=====')
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


# TODO: truncate the displacement
# replace this with actual truncation
new_pos = wp.array(shape=len(displacement_init), dtype=wp.vec3, device=device)
# cpmpute bounds
print('=====TODO: truncate the displacement=====')
@wp.kernel
def truncate_displacement(
    vertices: wp.array(dtype=wp.vec3),
    displacements: wp.array(dtype=wp.vec3),
    new_pos: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=float)
):
    tid = wp.tid()  # 获取当前线程的索引
    if wp.length(displacements[tid]) > bounds[tid]:
        new_pos[tid] = vertices[tid] + displacements[tid] * (bounds[tid] / wp.length(displacements[tid]))
    else:
        new_pos[tid] = vertices[tid] + displacements[tid]

# 将vertices列表转换为warp数组
vertices_array = wp.array(vertices, dtype=wp.vec3, device=device)
# 将displacement_init列表转换为warp数组
displacement_init_array = wp.array(displacement_init, dtype=wp.vec3, device=device)

wp.launch(
    truncate_displacement,
    dim=model.particle_count,
    inputs=[vertices_array, displacement_init_array, new_pos, bounds],
    device=device
)
print(new_pos)
print('new_pos:', len(new_pos))


# must include this after you update the mesh position, otherwise the collision detection results are not precise
collision_detector.refit(new_pos)
collision_detector.triangle_triangle_intersection_detection()

# TODO: analysis the results to see if there is an intersection
# NOTE that the triangle-triangle intersection test applied in triangle_triangle_intersection_detection() is single precision
# You need to implement a double-precision version yourself to see if it actually intersects. It can be a python code or warp kernel, both are okay
# ===== 打印三角形相交检测结果 =====
print("===== Triangle-Triangle Intersection Results =====")

# 1. 所有相交三角形索引（压缩存储）
print("triangle_intersecting_triangles:")
print(collision_detector.triangle_intersecting_triangles.numpy())

# 2. 每个三角形的相交数量
counts = collision_detector.triangle_intersecting_triangles_count.numpy()
print("triangle_intersecting_triangles_count:")
print(counts)

# 3. 缓冲区溢出标志
print("resize_flags:")
print(collision_detector.resize_flags.numpy())

# 额外：总相交数量
print("Total intersections:", counts.sum())



# TODO: Make this automated and process through all the files


# Bonus: implement double precision tri-tri intersection narrow phase test
# in line 2145 of the warp/sim/collide.py, there is a call of the built-in tri-tri intersection implemented in single precision
#         if wp.intersect_tri_tri(v1, v2, v3, u1, u2, u3):
# You can replace it with a double-precision version, to make your intersection more robust
# you can use @wp.func to implement a subsitute
