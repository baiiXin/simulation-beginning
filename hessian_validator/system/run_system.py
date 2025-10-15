import warp as wp
import numpy as np
import torch
from torch.func import hessian

torch.set_default_dtype(torch.float64)
# 设置打印精度为 10 位小数
torch.set_printoptions(precision=30)

print()
wp.init()
print()

from torch_collision_detector import TorchCollisionDetector

### Finish:
## 1.hessian computation
# analytical function in computation of hessian
# energy computation loop through contact pairs
# force computation loop through contact pairs and assemble
# hessian computation loop through contact pairs and assemble
# compute the closest point

## 2.collision detector
# contact pairs gather

## 3.finite difference method
# computation force by finite differnce method 
# computation hessian by finite differnce method 

## 4.automatic differentiation
# computation force and hessian by automatic differentiation

### TODO:
# 5.DeBUG


# -----------------------------
# edge-edge closest points
# -----------------------------
def closest_point_edge_edge(p1: torch.Tensor, q1: torch.Tensor,
                            p2: torch.Tensor, q2: torch.Tensor,
                            epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes closest points between two edges (p1-q1 and p2-q2).
    Returns tensor [s, t, dist] where s, t are barycentric weights on edges, dist is distance.
    p1, q1, p2, q2: torch.Tensor shape (3,)
    """
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2

    a = torch.dot(u, u)
    b = torch.dot(u, v)
    c = torch.dot(v, v)
    d = torch.dot(u, w0)
    e = torch.dot(v, w0)

    denom = a * c - b * b
    s = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
    t = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)

    if denom > epsilon:
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
    else:
        s = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
        t = e / c if c > epsilon else torch.tensor(0.0, dtype=p1.dtype, device=p1.device)

    # Clamp s and t to [0,1]
    s = torch.clamp(s, 0.0, 1.0)
    t = torch.clamp(t, 0.0, 1.0)

    cp1 = p1 + u * s
    cp2 = p2 + v * t
    dist = torch.norm(cp1 - cp2)

    return torch.tensor([s, t, dist], dtype=p1.dtype, device=p1.device)


# -----------------------------
# triangle closest point
# -----------------------------
def triangle_closest_point(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: torch.Tensor):
    """
    Returns (closest_point, barycentric_coords, feature_type)
    """
    TRI_CONTACT_FEATURE_VERTEX_A = 0
    TRI_CONTACT_FEATURE_VERTEX_B = 1
    TRI_CONTACT_FEATURE_VERTEX_C = 2
    TRI_CONTACT_FEATURE_EDGE_AB = 3
    TRI_CONTACT_FEATURE_EDGE_AC = 4
    TRI_CONTACT_FEATURE_EDGE_BC = 5
    TRI_CONTACT_FEATURE_FACE_INTERIOR = 6

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = torch.dot(ab, ap)
    d2 = torch.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = torch.tensor([1.0, 0.0, 0.0], dtype=p.dtype, device=p.device)
        return a, bary, feature_type

    bp = p - b
    d3 = torch.dot(ab, bp)
    d4 = torch.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = torch.tensor([0.0, 1.0, 0.0], dtype=p.dtype, device=p.device)
        return b, bary, feature_type

    cp = p - c
    d5 = torch.dot(ab, cp)
    d6 = torch.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = torch.tensor([0.0, 0.0, 1.0], dtype=p.dtype, device=p.device)
        return c, bary, feature_type

    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = torch.tensor([1.0 - v, v, 0.0], dtype=p.dtype, device=p.device)
        return a + v * ab, bary, feature_type

    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = torch.tensor([1.0 - v, 0.0, v], dtype=p.dtype, device=p.device)
        return a + v * ac, bary, feature_type

    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = torch.tensor([0.0, 1.0 - v, v], dtype=p.dtype, device=p.device)
        return b + v * (c - b), bary, feature_type

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = torch.tensor([1.0 - v - w, v, w], dtype=p.dtype, device=p.device)
    return a + v*ab + w*ac, bary, feature_type


# -----------------------------
# edge hessian kernel (simplified for PyTorch)
# -----------------------------
def compute_edge_hessian(
    vertices: torch.Tensor, 
    vertices_no_grad: torch.Tensor,
    edges_indices: torch.Tensor,
    edge_collision_pairs: torch.Tensor,
    edge_edge_parallel_epsilon=1e-6,
    collision_radius = 0.2,
    collision_stiffness = 1000.0
):
    """
    vertices: (V,3)
    edges_indices: (E,4)  # (v1_idx, v2_idx, v1_idx, v2_idx)
    edge_collision_pairs: (N,2) list of edge indices that collide
    returns: edge_energy: scalar tensor
             edge_force: (V,3)
             edge_hessian: (V,V,3,3)
    """
    N = vertices_no_grad.shape[0]
    edge_hessian = torch.zeros((N, N, 3, 3), dtype=vertices_no_grad.dtype, device=vertices_no_grad.device)
    edge_force = torch.zeros((N, 3), dtype=vertices_no_grad.dtype, device=vertices_no_grad.device)
    edge_energy = torch.tensor(0)

    for tid in range(edge_collision_pairs.shape[0]//2):
        e1_idx = edge_collision_pairs[2*tid]
        e2_idx = edge_collision_pairs[2*tid+1]

        if e1_idx == -1 or e2_idx == -1:
            continue

        e1_v1, e1_v2 = edges_indices[e1_idx, 2], edges_indices[e1_idx, 3]
        e2_v1, e2_v2 = edges_indices[e2_idx, 2], edges_indices[e2_idx, 3]

        e1_v1_pos = vertices[e1_v1]
        e1_v2_pos = vertices[e1_v2]
        e2_v1_pos = vertices[e2_v1]
        e2_v2_pos = vertices[e2_v2]
        e1_v1_pos_no_grad = vertices_no_grad[e1_v1]
        e1_v2_pos_no_grad = vertices_no_grad[e1_v2]
        e2_v1_pos_no_grad = vertices_no_grad[e2_v1]
        e2_v2_pos_no_grad = vertices_no_grad[e2_v2]
        
        st = closest_point_edge_edge(e1_v1_pos_no_grad, e1_v2_pos_no_grad, e2_v1_pos_no_grad, e2_v2_pos_no_grad, edge_edge_parallel_epsilon)
        s, t, dis = st[0], st[1], st[2]

        if dis >= collision_radius:
            continue

        e1_vec_no_grad = e1_v2_pos_no_grad - e1_v1_pos_no_grad
        e2_vec_no_grad = e2_v2_pos_no_grad - e2_v1_pos_no_grad
        c1_no_grad = e1_v1_pos_no_grad + e1_vec_no_grad * s
        c2_no_grad = e2_v1_pos_no_grad + e2_vec_no_grad * t
        diff_no_grad = c1_no_grad - c2_no_grad
        collision_normal = diff_no_grad / dis

        e1_vec = e1_v2_pos - e1_v1_pos
        e2_vec = e2_v2_pos - e2_v1_pos
        c1 = e1_v1_pos + e1_vec * s
        c2 = e2_v1_pos + e2_vec * t
        diff = c1 - c2

        energy = 0.5 * 0.5 * collision_stiffness * (collision_radius - torch.dot(diff, collision_normal)) **2

        force = 0.5 * collision_stiffness * (collision_radius - torch.dot(diff, collision_normal)) * collision_normal 

        hessian = 0.5 * collision_stiffness * torch.outer(collision_normal, collision_normal)

        # edge_force/edge_hessian can be filled here depending on your force model
        # energy
        edge_energy = edge_energy + energy

        # force
        edge_force[e1_v1] = edge_force[e1_v1] + force * (1.0 - s)
        edge_force[e1_v2] = edge_force[e1_v2] + force * s
        edge_force[e2_v1] = edge_force[e2_v1] -force * (1.0 - t)
        edge_force[e2_v2] = edge_force[e2_v2] -force * t

        # hessian
        # e1
        edge_hessian[e1_v1, e1_v1] = edge_hessian[e1_v1, e1_v1] + hessian * (1.0 - s) * (1.0 - s)
        edge_hessian[e1_v2, e1_v2] = edge_hessian[e1_v2, e1_v2] + hessian * s * s
        edge_hessian[e1_v1, e1_v2] = edge_hessian[e1_v1, e1_v2] + hessian * (1.0 - s) * s
        edge_hessian[e1_v2, e1_v1] = edge_hessian[e1_v2, e1_v1] + hessian * s * (1.0 - s)
        # e2
        edge_hessian[e2_v1, e2_v1] = edge_hessian[e2_v1, e2_v1] + hessian * (1.0 - t) * (1.0 - t)
        edge_hessian[e2_v2, e2_v2] = edge_hessian[e2_v2, e2_v2] + hessian * t * t
        edge_hessian[e2_v1, e2_v2] = edge_hessian[e2_v1, e2_v2] + hessian * (1.0 - t) * t
        edge_hessian[e2_v2, e2_v1] = edge_hessian[e2_v2, e2_v1] + hessian * t * (1.0 - t)
        # e1-e2
        edge_hessian[e1_v1, e2_v1] = edge_hessian[e1_v1, e2_v1] - hessian * (1.0 - s) * (1.0 - t)
        edge_hessian[e1_v1, e2_v2] = edge_hessian[e1_v1, e2_v2] - hessian * (1.0 - s) * t
        edge_hessian[e1_v2, e2_v1] = edge_hessian[e1_v2, e2_v1] - hessian * s * (1.0 - t)
        edge_hessian[e1_v2, e2_v2] = edge_hessian[e1_v2, e2_v2] - hessian * s * t
        # e2-e1
        edge_hessian[e2_v1, e1_v1] = edge_hessian[e2_v1, e1_v1] - hessian * (1.0 - t) * (1.0 - s)
        edge_hessian[e2_v1, e1_v2] = edge_hessian[e2_v1, e1_v2] - hessian * (1.0 - t) * s
        edge_hessian[e2_v2, e1_v1] = edge_hessian[e2_v2, e1_v1] - hessian * t * (1.0 - s)
        edge_hessian[e2_v2, e1_v2] = edge_hessian[e2_v2, e1_v2] - hessian * t * s 

    return edge_energy, edge_force, edge_hessian


# -----------------------------
# point-triangle hessian kernel (PyTorch)
# -----------------------------
def compute_point_triangle_hessian(
    vertices: torch.Tensor,
    vertices_no_grad: torch.Tensor,
    tri_indices: torch.Tensor,
    pt_collision_pairs: torch.Tensor,
    collision_radius = 0.2,
    collision_stiffness = 1000.0
):
    """
    vertices: (V,3)
    tri_indices: (T,4)  # (v0_idx, v1_idx, v2_idx, p_idx)
    pt_collision_pairs: (N,) list of triangle indices that have point collisions
    returns: pt_energy: scalar tensor
             pt_force: (V,3)
             pt_hessian: (V,V,3,3)
    """
    N = vertices_no_grad.shape[0]
    pt_hessian = torch.zeros((N, N, 3, 3), dtype=vertices_no_grad.dtype, device=vertices_no_grad.device)
    pt_force = torch.zeros((N, 3), dtype=vertices_no_grad.dtype, device=vertices_no_grad.device)
    pt_energy = torch.tensor(0)

    for tid in range(pt_collision_pairs.shape[0]//2):
        p_idx = pt_collision_pairs[2*tid]
        t_idx = pt_collision_pairs[2*tid+1]
        if p_idx == -1 or t_idx == -1:
            continue

        v0_idx, v1_idx, v2_idx = tri_indices[t_idx]
        v0, v1, v2, p = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx], vertices[p_idx]
        v0_no_grad, v1_no_grad, v2_no_grad, p_no_grad = vertices_no_grad[v0_idx], vertices_no_grad[v1_idx], vertices_no_grad[v2_idx], vertices_no_grad[p_idx]

        # 使用你的 triangle_closest_point 函数
        closest_no_grad, bary_no_grad, _ = triangle_closest_point(v0_no_grad, v1_no_grad, v2_no_grad, p_no_grad)
        u, v, w = bary_no_grad
        diff_no_grad = closest_no_grad - p_no_grad
        dis = torch.norm(diff_no_grad) 
        collision_normal = diff_no_grad / dis
        if dis >= collision_radius:
            continue
        
        closest = u * v0 + v * v1 + w * v2
        diff = closest - p

        energy = 0.5 * collision_stiffness * (collision_radius - torch.dot(diff, collision_normal))**2
        force = collision_stiffness * (collision_radius - torch.dot(diff, collision_normal)) * collision_normal
        hessian = collision_stiffness * torch.outer(collision_normal, collision_normal)

        # 能量
        pt_energy = pt_energy + energy

        # 力
        pt_force[v0_idx] = pt_force[v0_idx] + force * u
        pt_force[v1_idx] = pt_force[v1_idx] + force * v
        pt_force[v2_idx] = pt_force[v2_idx] + force * w
        pt_force[p_idx]   = pt_force[p_idx]   - force

        # Hessian
        # tri-tri
        pt_hessian[v0_idx,v0_idx] = pt_hessian[v0_idx,v0_idx] + hessian * u * u
        pt_hessian[v1_idx,v1_idx] = pt_hessian[v1_idx,v1_idx] + hessian * v * v
        pt_hessian[v2_idx,v2_idx] = pt_hessian[v2_idx,v2_idx] + hessian * w * w

        pt_hessian[v0_idx,v1_idx] = pt_hessian[v0_idx,v1_idx] + hessian * u * v
        pt_hessian[v0_idx,v2_idx] = pt_hessian[v0_idx,v2_idx] + hessian * u * w
        pt_hessian[v1_idx,v0_idx] = pt_hessian[v1_idx,v0_idx] + hessian * v * u
        pt_hessian[v1_idx,v2_idx] = pt_hessian[v1_idx,v2_idx] + hessian * v * w
        pt_hessian[v2_idx,v0_idx] = pt_hessian[v2_idx,v0_idx] + hessian * w * u
        pt_hessian[v2_idx,v1_idx] = pt_hessian[v2_idx,v1_idx] + hessian * w * v

        # tri-point
        pt_hessian[v0_idx,p_idx] = pt_hessian[v0_idx,p_idx] - hessian * u
        pt_hessian[v1_idx,p_idx] = pt_hessian[v1_idx,p_idx] - hessian * v
        pt_hessian[v2_idx,p_idx] = pt_hessian[v2_idx,p_idx] - hessian * w

        pt_hessian[p_idx,v0_idx] = pt_hessian[p_idx,v0_idx] - hessian * u
        pt_hessian[p_idx,v1_idx] = pt_hessian[p_idx,v1_idx] - hessian * v
        pt_hessian[p_idx,v2_idx] = pt_hessian[p_idx,v2_idx] - hessian * w

        # point-point
        pt_hessian[p_idx,p_idx] = pt_hessian[p_idx,p_idx] + hessian

    return pt_energy, pt_force, pt_hessian


# -----------------------------
# all hessian kernel (PyTorch)
# -----------------------------
def compute_full_hessian(
    vertices: torch.Tensor, 
    vertices_no_grad: torch.Tensor,
    edge_indices, edge_colliding_edges, 
    tri_indices, vertex_colliding_triangles, 
    contact_radius, contact_stiffness
):
    edge_energy, edge_force, edge_hessian = compute_edge_hessian(
        vertices = vertices, 
        vertices_no_grad = vertices_no_grad, 
        edges_indices = edge_indices, 
        edge_collision_pairs = edge_colliding_edges, 
        collision_radius = contact_radius, 
        collision_stiffness = contact_stiffness)

    pt_energy, pt_force, pt_hessian = compute_point_triangle_hessian(
        vertices = vertices, 
        vertices_no_grad = vertices_no_grad, 
        tri_indices = tri_indices, 
        pt_collision_pairs = vertex_colliding_triangles, 
        collision_radius = contact_radius, 
        collision_stiffness = contact_stiffness)

    return edge_energy + pt_energy, edge_force + pt_force, edge_hessian + pt_hessian


# =============================
# Finite difference (PyTorch)
# -----------------------------
def compute_finite_difference_results(
    vertices: torch.Tensor, 
    vertices_no_grad: torch.Tensor,
    edge_indices, edge_colliding_edges, 
    tri_indices, vertex_colliding_triangles, 
    contact_radius, contact_stiffness
):
        """
        中心差分计算Hessian - PyTorch版本
        """
        N = vertices_no_grad.shape[0]
        h = 1e-3
        
        # force
        # 展平索引，逐个 perturb
        grad = torch.zeros_like(vertices_no_grad)
        for idx in range(N):
            for space_dx in range(3):
                perturb = torch.zeros_like(vertices_no_grad)
                perturb[idx,space_dx] = h

                f_plus, _, _ = compute_full_hessian(
                    vertices + perturb,
                    vertices_no_grad + perturb,
                    edge_indices, edge_colliding_edges, 
                    tri_indices, vertex_colliding_triangles, 
                    contact_radius, contact_stiffness)
                f_minus, _, _ = compute_full_hessian(
                    vertices - perturb, 
                    vertices_no_grad - perturb,
                    edge_indices, edge_colliding_edges, 
                    tri_indices, vertex_colliding_triangles, 
                    contact_radius, contact_stiffness)

                # print(f'f_plus = {f_plus}, f_minus = {f_minus}')

                grad[idx,space_dx] = (f_plus - f_minus) / (2 * h)

        # hessian
        Jacobi = torch.zeros((N, N, 3, 3), dtype=vertices_no_grad.dtype, device=vertices_no_grad.device)
        for jdx in range(N):
            for space_jdx in range(3):
                perturb = torch.zeros_like(vertices_no_grad)
                perturb[jdx,space_jdx] = h
                
                _, f_plus, _ = compute_full_hessian(
                    vertices + perturb, 
                    vertices_no_grad + perturb,
                    edge_indices, edge_colliding_edges, 
                    tri_indices, vertex_colliding_triangles, 
                    contact_radius, contact_stiffness)
                _, f_minus, _ = compute_full_hessian(
                    vertices - perturb, 
                    vertices_no_grad - perturb,
                    edge_indices, edge_colliding_edges, 
                    tri_indices, vertex_colliding_triangles, 
                    contact_radius, contact_stiffness)
                Jacobi[:, jdx, :, space_jdx] = (f_plus - f_minus) / (2 * h)
        
        return -grad, -Jacobi


# =============================
# Auto difference (PyTorch)
# -----------------------------
def compute_automatic_diff_results(
    vertices: torch.Tensor, 
    vertices_no_grad: torch.Tensor,
    edge_indices, edge_colliding_edges, 
    tri_indices, vertex_colliding_triangles, 
    contact_radius, contact_stiffness
):
        """
        使用PyTorch自动微分计算Hessian
        """
        # 确保point需要梯度
        vertices.requires_grad_(True)
        
        # 计算能量
        energy, _, _ = compute_full_hessian(
            vertices, vertices_no_grad,
            edge_indices, edge_colliding_edges, 
            tri_indices, vertex_colliding_triangles, 
            contact_radius, contact_stiffness)

        # 计算力（保持梯度图）
        force = torch.autograd.grad(energy, vertices, create_graph=True)[0]

        # 初始化 Hessian
        N = vertices.shape[0]
        hessian_auto = torch.zeros((N, N, 3, 3), device=vertices.device)

        # 逐分量求二阶导数
        for i in range(N):
            for a in range(3):
                # force[i, a] 仍在计算图中，因此可以再次求导
                grad2 = torch.autograd.grad(
                    force[i, a], vertices, retain_graph=True, create_graph=False
                )[0]
                hessian_auto[i, :, :, a] = grad2

        return -force, hessian_auto




if __name__ == "__main__":
    # input_mesh
    vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.5, 0.0), wp.vec3(0.0, 0.0, 1.0),
                wp.vec3(0.01, 0.01, 0.01), wp.vec3(0.01, 0.51, 0.01), wp.vec3(0.01, 0.01, 1.01)]

    mesh_indices = [0,1,2,3,4,5]

    contact_radius = 0.2
    contact_margin = 0.3
    contact_stiffness = 1000.0

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
    
    energy, force, Hessian = compute_full_hessian(
        torch_vertices, torch_vertices, 
        edge_indices, edge_colliding_edges, 
        tri_indices, vertex_colliding_triangles, 
        contact_radius, contact_stiffness)

    print('\nenergy:', energy)
    print('\nforce:', force)
    print('\nHessian:', Hessian.shape)
    print('\nHessian (max min):', Hessian.max(), Hessian.min())

    force_fd, hessian_fd = compute_finite_difference_results(
        torch_vertices, torch_vertices, 
        edge_indices, edge_colliding_edges, 
        tri_indices, vertex_colliding_triangles, 
        contact_radius, contact_stiffness)

    print('\nforce_fd:', force_fd)
    print('\nhessian_fd:', hessian_fd.shape)
    print('\nhessian_fd (max min):', hessian_fd.max(), hessian_fd.min())
    
    print('\nHessian vs finite difference Hessian:', torch.norm(Hessian-hessian_fd, p=float('inf')))

    # 计算自动微分结果
    torch_vertices.requires_grad_(True)
    torch_vertices_no_grad = torch_vertices.detach().clone()

    force_auto, hessian_auto = compute_automatic_diff_results(
        torch_vertices, torch_vertices_no_grad,
        edge_indices, edge_colliding_edges, 
        tri_indices, vertex_colliding_triangles, 
        contact_radius, contact_stiffness)
    
    print('\nforce_auto:', force_auto)
    print('\nhessian_auto:', hessian_auto.shape)
    print('\nhessian_auto (max min):', hessian_auto.max(), hessian_auto.min())
    print('\nHessian vs auto Hessian:', torch.norm(Hessian-hessian_auto, p=float('inf')), 
        torch.max(torch.abs(Hessian-hessian_auto)) == 0)

    print('\n---fd hessian and hessian')
    print('\nhessian', hessian_fd)
    print('\nhessian_auto', hessian_auto)