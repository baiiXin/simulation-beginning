from platform import java_ver
from traceback import print_exception
import warp as wp
import numpy as np

print()
wp.init()
print()

from torch_collision_detector import TorchCollisionDetector


# analytical function

# energy computation loop through contact pairs


# force computation loop through contact pairs and assemble


# hessian computation loop through contact pairs and assemble



# contact pairs gather

# compute the closest point



# computation force by finite differnce method 


# computation hessian by finite differnce method 


# computation force and hessian by automatic differentiation



# input_mesh
vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.5, 0.0), wp.vec3(0.5, 0.0, 0.0),
             wp.vec3(0.0, 0.0, 0.1), wp.vec3(0.0, 0.5, 0.1), wp.vec3(0.5, 0.0, 0.1),
             wp.vec3(0.0, 0.0, 0.2), wp.vec3(0.0, 0.5, 0.2), wp.vec3(0.5, 0.0, 0.2)]

mesh_indices = [0,1,2,3,4,5,6,7,8]

contact_radius = 0.2
contact_margin = 0.3

# collision detector
collision_detector = TorchCollisionDetector(vertices, mesh_indices, contact_radius, contact_margin)

warp_vertices = wp.array(vertices, dtype=wp.vec3, device="cpu")
torch_vertices = wp.to_torch(warp_vertices)

tri_indices, edge_indices = collision_detector()
print('\nvertices:', torch_vertices)
print('\nmodel.tri_indices:', tri_indices)
print('\nmodel.edge_indices:', edge_indices)

# detect
vertex_colliding_triangles, edge_colliding_edges = collision_detector.detect(warp_vertices)


'''
# edge_idx
@wp.kernel
def comupute_edge_hessian(
    vertices_indices: wp.array(dtype = wp.vec3),
    edges_indices: wp.array(dtype=wp.vec4),
    trimesh_collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    edge_force: wp.array(dtype=wp.vec3),
    edge_hessian: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()

    collision_info = trimesh_collision_info_array[0]

    # process edge-edge collisions
    if tid * 2 < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = collision_info.edge_colliding_edges[2 * tid]
        e2_idx = collision_info.edge_colliding_edges[2 * tid + 1]

        if e1_idx != -1 and e2_idx != -1:
            e1_v1 = edges_indices[e1_idx, 2]
            e1_v2 = edges_indices[e1_idx, 3]
            e2_v1 = edges_indices[e2_idx, 2]
            e2_v2 = edges_indices[e2_idx, 3]

            e1_v1_pos = vertices_indices[e1_v1]
            e1_v2_pos = vertices_indices[e1_v2]
            e2_v1_pos = vertices_indices[e2_v1]
            e2_v2_pos = vertices_indices[e2_v2]

            st = closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
            s = st[0]
            t = st[1]
            e1_vec = e1_v2_pos - e1_v1_pos
            e2_vec = e2_v2_pos - e2_v1_pos
            c1 = e1_v1_pos + e1_vec * s
            c2 = e2_v1_pos + e2_vec * t
            diff = c1 - c2
            dis = st[2]
            collision_normal = diff / dis






@wp.func  
def closest_point_edge_edge(
    p1: wp.vec3,
    q1: wp.vec3,
    p2: wp.vec3,
    q2: wp.vec3,
    epsilon: float
) -> wp.vec3:
    """
    Computes closest points between two edges (p1-q1 and p2-q2).
    Returns vec3(s, t, d) where s, t are barycentric weights on edges, d is distance.
    """
    
    # Edge directions
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2

    a = wp.dot(u, u)
    b = wp.dot(u, v)
    c = wp.dot(v, v)
    d = wp.dot(u, w0)
    e = wp.dot(v, w0)

    denom = a*c - b*b

    s = 0.0
    t = 0.0

    if denom > epsilon:
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
    else:
        # Edges are nearly parallel, pick s=0
        s = 0.0
        t = e / c if c > epsilon else 0.0

    # Clamp s and t to [0,1]
    s = wp.clamp(s, 0.0, 1.0)
    t = wp.clamp(t, 0.0, 1.0)

    # Closest points on edges
    cp1 = p1 + u * s
    cp2 = p2 + v * t

    # Distance
    dist = wp.length(cp1 - cp2)

    return wp.vec3(s, t, dist)

@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """
    feature_type type:
        TRI_CONTACT_FEATURE_VERTEX_A
        TRI_CONTACT_FEATURE_VERTEX_B
        TRI_CONTACT_FEATURE_VERTEX_C
        TRI_CONTACT_FEATURE_EDGE_AB      : at edge A-B
        TRI_CONTACT_FEATURE_EDGE_AC      : at edge A-C
        TRI_CONTACT_FEATURE_EDGE_BC      : at edge B-C
        TRI_CONTACT_FEATURE_FACE_INTERIOR
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary, feature_type

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary, feature_type

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary, feature_type

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary, feature_type

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary, feature_type

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary, feature_type

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type


'''