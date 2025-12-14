# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from tkinter import N
import warnings

import numpy as np
import warp as wp
from warp.types import float32, matrix

from warp.sparse import bsr_from_triplets
from warp.optim.linear import cg
from warp.sparse import bsr_zeros, bsr_identity, bsr_diag
from warp.sparse import bsr_mm, bsr_mv, bsr_axpy, bsr_scale

import numpy as np
from scipy.sparse import bsr_matrix
from pypardiso import spsolve


from ...core.types import override
from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16


class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ForceElementAdjacencyInfo,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    face_indices: wp.array(dtype=wp.int32, ndim=2),
):
    wp.printf("num vertices: %d\n", adjacency.v_adj_edges_offsets.shape[0] - 1)
    for vertex in range(adjacency.v_adj_edges_offsets.shape[0] - 1):
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, vertex)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(adjacency, vertex, i_bd)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, vertex)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(
                adjacency,
                vertex,
                i_face,
            )

            if face_indices[face, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_faces: %d\n", vertex, num_adj_faces)
                wp.printf("--i_face: %d | face id: %d | v_order: %d\n", i_face, face, v_order)
                wp.printf(
                    "--face: %d %d %d\n",
                    face_indices[face, 0],
                    face_indices[face, 1],
                    face_indices[face, 2],
                )


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # StVK energy density: psi = mu * ||G||_F^2 + 0.5 * lambda * (trace(G))^2

    # Deformation gradient F = [f0, f1] (3x2 matrix as two 3D column vectors)
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    # Cache tri_pose elements
    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Compute F columns directly: F = [x01, x02] * tri_pose = [f0, f1]
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Green strain tensor: G = 0.5(F^T F - I) = [[G00, G01], [G01, G11]] (symmetric 2x2)
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0_dot_f0 - 1.0)
    G11 = 0.5 * (f1_dot_f1 - 1.0)
    G01 = 0.5 * f0_dot_f1

    # Frobenius norm squared of Green strain: ||G||_F^2 = G00^2 + G11^2 + 2 * G01^2
    G_frobenius_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_frobenius_sq < 1.0e-20:
        return wp.vec3(0.0), wp.mat33(0.0)

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress tensor (StVK model)
    # PK1 = 2*mu*F*G + lambda*trace(G)*F = [PK1_col0, PK1_col1] (3x2)
    lambda_trace_G = lmbd * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lambda_trace_G) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lambda_trace_G)

    # Vertex selection using masks to avoid branching
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    # Deformation gradient derivatives w.r.t. current vertex position
    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force via chain rule: force = -(dpsi/dF) : (dF/dx)
    dpsi_dx = PK1_col0 * df0_dx + PK1_col1 * df1_dx
    force = -dpsi_dx

    # Hessian computation using Cauchy-Green invariants
    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx
    df0_df1_cross = df0_dx * df1_dx

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd
    I33 = wp.identity(n=3, dtype=float)

    f0_outer_f0 = wp.outer(f0, f0)
    f1_outer_f1 = wp.outer(f1, f1)
    f0_outer_f1 = wp.outer(f0, f1)
    f1_outer_f0 = wp.outer(f1, f0)

    H_IIc00_scaled = mu * (f0_dot_f0 * I33 + 2.0 * f0_outer_f0 + f1_outer_f1)
    H_IIc11_scaled = mu * (f1_dot_f1 * I33 + 2.0 * f1_outer_f1 + f0_outer_f0)
    H_IIc01_scaled = mu * (f0_dot_f1 * I33 + f1_outer_f0)

    # d2(psi)/dF^2 components
    d2E_dF2_00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I33 + H_IIc00_scaled
    d2E_dF2_01 = lmbd * f0_outer_f1 + H_IIc01_scaled
    d2E_dF2_11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I33 + H_IIc11_scaled

    # Chain rule: H = (dF/dx)^T * (d2(psi)/dF^2) * (dF/dx)
    hessian = df0_dx_sq * d2E_dF2_00 + df1_dx_sq * d2E_dF2_11 + df0_df1_cross * (d2E_dF2_01 + wp.transpose(d2E_dF2_01))

    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Previous deformation gradient for velocity
        x0_prev = pos_prev[v0]
        x01_prev = pos_prev[v1] - x0_prev
        x02_prev = pos_prev[v2] - x0_prev

        vel_x01 = (x01 - x01_prev) * inv_dt
        vel_x02 = (x02 - x02_prev) * inv_dt

        df0_dt = vel_x01 * DmInv00 + vel_x02 * DmInv10
        df1_dt = vel_x01 * DmInv01 + vel_x02 * DmInv11

        # First constraint: Cmu = ||G||_F (Frobenius norm of Green strain)
        Cmu = wp.sqrt(G_frobenius_sq)

        G00_normalized = G00 / Cmu
        G01_normalized = G01 / Cmu
        G11_normalized = G11 / Cmu

        # Time derivative of Green strain: dG/dt = 0.5 * (F^T * dF/dt + (dF/dt)^T * F)
        dG_dt_00 = wp.dot(f0, df0_dt)  # dG00/dt
        dG_dt_11 = wp.dot(f1, df1_dt)  # dG11/dt
        dG_dt_01 = 0.5 * (wp.dot(f0, df1_dt) + wp.dot(f1, df0_dt))  # dG01/dt

        # Time derivative of first constraint: dCmu/dt = (1/||G||_F) * (G : dG/dt)
        dCmu_dt = G00_normalized * dG_dt_00 + G11_normalized * dG_dt_11 + 2.0 * G01_normalized * dG_dt_01

        # Gradient of first constraint w.r.t. deformation gradient: dCmu/dF = (G/||G||_F) * F
        dCmu_dF_col0 = G00_normalized * f0 + G01_normalized * f1  # dCmu/df0
        dCmu_dF_col1 = G01_normalized * f0 + G11_normalized * f1  # dCmu/df1

        # Gradient of constraint w.r.t. vertex position: dCmu/dx = (dCmu/dF) : (dF/dx)
        dCmu_dx = df0_dx * dCmu_dF_col0 + df1_dx * dCmu_dF_col1

        # Damping force from first constraint: -mu * damping * (dCmu/dt) * (dCmu/dx)
        kd_mu = mu * damping
        force += -kd_mu * dCmu_dt * dCmu_dx

        # Damping Hessian: mu * damping * (1/dt) * (dCmu/dx) x (dCmu/dx)
        hessian += kd_mu * inv_dt * wp.outer(dCmu_dx, dCmu_dx)

        # Second constraint: Clmbd = trace(G) = G00 + G11 (trace of Green strain)
        # Time derivative of second constraint: dClmbd/dt = trace(dG/dt)
        dClmbd_dt = dG_dt_00 + dG_dt_11

        # Gradient of second constraint w.r.t. deformation gradient: dClmbd/dF = F
        dClmbd_dF_col0 = f0  # dClmbd/df0
        dClmbd_dF_col1 = f1  # dClmbd/df1

        # Gradient of Clmbd w.r.t. vertex position: dClmbd/dx = (dClmbd/dF) : (dF/dx)
        dClmbd_dx = df0_dx * dClmbd_dF_col0 + df1_dx * dClmbd_dF_col1

        # Damping force from second constraint: -lambda * damping * (dClmbd/dt) * (dClmbd/dx)
        kd_lmbd = lmbd * damping
        force += -kd_lmbd * dClmbd_dt * dClmbd_dx

        # Damping Hessian from second constraint: lambda * damping * (1/dt) * (dClmbd/dx) x (dClmbd/dx)
        hessian += kd_lmbd * inv_dt * wp.outer(dClmbd_dx, dClmbd_dx)

    # Apply area scaling
    force *= area
    hessian *= area

    return force, hessian


@wp.func
def compute_normalized_vector_derivative(
    unnormalized_vec_length: float, normalized_vec: wp.vec3, unnormalized_vec_derivative: wp.mat33
) -> wp.mat33:
    projection_matrix = wp.identity(n=3, dtype=float) - wp.outer(normalized_vec, normalized_vec)

    # d(normalized_vec)/dx = (1/|unnormalized_vec|) * (I - normalized_vec * normalized_vec^T) * d(unnormalized_vec)/dx
    return (1.0 / unnormalized_vec_length) * projection_matrix * unnormalized_vec_derivative


@wp.func
def compute_angle_derivative(
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
    dn1hat_dx: wp.mat33,
    dn2hat_dx: wp.mat33,
    sin_theta: float,
    cos_theta: float,
    skew_n1: wp.mat33,
    skew_n2: wp.mat33,
) -> wp.vec3:
    dsin_dx = wp.transpose(skew_n1 * dn2hat_dx - skew_n2 * dn1hat_dx) * e_hat
    dcos_dx = wp.transpose(dn1hat_dx) * n2_hat + wp.transpose(dn2hat_dx) * n1_hat

    # dtheta/dx = dsin/dx * cos - dcos/dx * sin
    return dsin_dx * cos_theta - dcos_dx * sin_theta


@wp.func
def evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[bending_index, 0] == -1 or edge_indices[bending_index, 1] == -1:
        return wp.vec3(0.0), wp.mat33(0.0)

    eps = 1.0e-6

    vi0 = edge_indices[bending_index, 0]
    vi1 = edge_indices[bending_index, 1]
    vi2 = edge_indices[bending_index, 2]
    vi3 = edge_indices[bending_index, 3]

    x0 = pos[vi0]  # opposite 0
    x1 = pos[vi1]  # opposite 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    # Early exit for degenerate cases
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return wp.vec3(0.0), wp.mat33(0.0)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    k = stiffness * edge_rest_length[bending_index]
    dE_dtheta = k * (theta - edge_rest_angle[bending_index])

    # Pre-compute skew matrices (shared across all angle derivative computations)
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1_hat)
    skew_n2 = wp.skew(n2_hat)

    # Compute the derivatives of unit normals with respect to each vertex; required for computing angle derivatives
    dn1hat_dx0 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_e)
    dn2hat_dx0 = wp.mat33(0.0)

    dn1hat_dx1 = wp.mat33(0.0)
    dn2hat_dx1 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_e)

    dn1hat_dx2 = compute_normalized_vector_derivative(n1_norm, n1_hat, -skew_x03)
    dn2hat_dx2 = compute_normalized_vector_derivative(n2_norm, n2_hat, skew_x13)

    dn1hat_dx3 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_x02)
    dn2hat_dx3 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_x12)

    # Compute all angle derivatives (required for damping)
    dtheta_dx0 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx0, dn2hat_dx0, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx1 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx1, dn2hat_dx1, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx2 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx2, dn2hat_dx2, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx3 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx3, dn2hat_dx3, sin_theta, cos_theta, skew_n1, skew_n2
    )

    # Use float masks for branch-free selection
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)
    mask3 = float(v_order == 3)

    # Select the derivative for the current vertex without branching
    dtheta_dx = dtheta_dx0 * mask0 + dtheta_dx1 * mask1 + dtheta_dx2 * mask2 + dtheta_dx3 * mask3

    # Compute elastic force and hessian
    bending_force = -dE_dtheta * dtheta_dx
    bending_hessian = k * wp.outer(dtheta_dx, dtheta_dx)

    if damping > 0.0:
        inv_dt = 1.0 / dt
        x_prev0 = pos_prev[vi0]
        x_prev1 = pos_prev[vi1]
        x_prev2 = pos_prev[vi2]
        x_prev3 = pos_prev[vi3]

        # Compute displacement vectors
        dx0 = x0 - x_prev0
        dx1 = x1 - x_prev1
        dx2 = x2 - x_prev2
        dx3 = x3 - x_prev3

        # Compute angular velocity using all derivatives
        dtheta_dt = (
            wp.dot(dtheta_dx0, dx0) + wp.dot(dtheta_dx1, dx1) + wp.dot(dtheta_dx2, dx2) + wp.dot(dtheta_dx3, dx3)
        ) * inv_dt

        damping_coeff = damping * k  # damping coefficients following the VBD convention
        damping_force = -damping_coeff * dtheta_dt * dtheta_dx
        damping_hessian = damping_coeff * inv_dt * wp.outer(dtheta_dx, dtheta_dx)

        bending_force = bending_force + damping_force
        bending_hessian = bending_hessian + damping_hessian

    return bending_force, bending_hessian


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0:
        body_contact_force_norm = penetration_depth * soft_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = soft_contact_ke * wp.outer(n, n)

        mu = shape_material_mu[shape_index]

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0:
            damping_hessian = (soft_contact_kd / dt) * body_contact_hessian
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # friction
        e0, e1 = build_orthonormal_basis(n)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * relative_translation
        eps_u = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(mu, body_contact_force_norm, T, u, eps_u)
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def damp_collision(
    displacement: wp.vec3,
    collision_normal: wp.vec3,
    collision_hessian: wp.mat33,
    collision_damping: float,
    dt: float,
):
    if wp.dot(displacement, collision_normal) > 0:
        damping_hessian = (collision_damping / dt) * collision_hessian
        damping_force = damping_hessian * displacement
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        pos_prev,
        edge_indices
        collision_radius
        collision_stiffness
        dt
        edge_edge_parallel_epsilon: threshold to determine whether 2 edges are parallel
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)
        friction_force = friction_force * v_bary
        friction_hessian = friction_hessian * v_bary * v_bary

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        if v_order == 0:
            displacement = pos_prev[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_prev[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_prev[e2_v1] - e2_v1_pos
        else:
            displacement = pos_prev[e2_v2] - e2_v2_pos

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + friction_force
        collision_hessian = collision_hessian + friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        displacement_0 = pos_prev[e1_v1] - e1_v1_pos
        displacement_1 = pos_prev[e1_v2] - e1_v2_pos

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        return True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_hessian, collision_hessian

# zcy
@wp.func
def zcy_evaluate_stvk_force_hessian(
    face: int,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
):
    # StVK energy：输出3个force与9个Hessian块（保持原参数签名，不引入包装）

    # 1) 组装 F 的两列
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # 2) Green 应变与阈值
    f0f0 = wp.dot(f0, f0)
    f1f1 = wp.dot(f1, f1)
    f0f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1

    G_norm_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_norm_sq < 1.0e-20:
        z = wp.vec3(0.0, 0.0, 0.0)
        Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return z, z, z, Z, Z, Z, Z, Z, Z, Z, Z, Z

    # 3) PK1 两列
    t = G00 + G11
    two_mu = 2.0 * mu
    lt = lmbd * t
    PK1_col0 = f0 * (two_mu * G00 + lt) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lt)

    # 4) f 空间 Hessian 块 A00/A01/A11
    Ic = f0f0 + f1f1
    c = -mu + (0.5 * Ic - 1.0) * lmbd
    I3 = wp.identity(n=3, dtype=float)

    f0of0 = wp.outer(f0, f0)
    f1of1 = wp.outer(f1, f1)
    f0of1 = wp.outer(f0, f1)
    f1of0 = wp.outer(f1, f0)

    A00 = lmbd * f0of0 + c * I3 + mu * (f0f0 * I3 + 2.0 * f0of0 + f1of1)
    A01 = lmbd * f0of1 + mu * (f0f1 * I3 + f1of0)
    A11 = lmbd * f1of1 + c * I3 + mu * (f1f1 * I3 + 2.0 * f1of1 + f0of0)

    # 5) 位置空间系数 alpha
    a00 = -(DmInv00 + DmInv10)
    a01 = DmInv00
    a02 = DmInv10
    b00 = -(DmInv01 + DmInv11)
    b01 = DmInv01
    b02 = DmInv11

    # 6) 三个顶点力（忽略 v_order，直接全输出）
    f_a0 = -(PK1_col0 * a00 + PK1_col1 * b00) * area
    f_a1 = -(PK1_col0 * a01 + PK1_col1 * b01) * area
    f_a2 = -(PK1_col0 * a02 + PK1_col1 * b02) * area

    # 7) 九个海森块（Hab 公式）
    A01T = wp.transpose(A01)

    H00 = (a00 * a00) * A00 + (b00 * b00) * A11 + (a00 * b00) * A01 + (b00 * a00) * A01T
    H01 = (a00 * a01) * A00 + (b00 * b01) * A11 + (a00 * b01) * A01 + (b00 * a01) * A01T
    H02 = (a00 * a02) * A00 + (b00 * b02) * A11 + (a00 * b02) * A01 + (b00 * a02) * A01T

    H10 = (a01 * a00) * A00 + (b01 * b00) * A11 + (a01 * b00) * A01 + (b01 * a00) * A01T
    H11 = (a01 * a01) * A00 + (b01 * b01) * A11 + (a01 * b01) * A01 + (b01 * a01) * A01T
    H12 = (a01 * a02) * A00 + (b01 * b02) * A11 + (a01 * b02) * A01 + (b01 * a02) * A01T

    H20 = (a02 * a00) * A00 + (b02 * b00) * A11 + (a02 * b00) * A01 + (b02 * a00) * A01T
    H21 = (a02 * a01) * A00 + (b02 * b01) * A11 + (a02 * b01) * A01 + (b02 * a01) * A01T
    H22 = (a02 * a02) * A00 + (b02 * b02) * A11 + (a02 * b02) * A01 + (b02 * a02) * A01T

    # 面积缩放（力已乘 area；海森也乘 area）
    H00 *= area; H01 *= area; H02 *= area
    H10 *= area; H11 *= area; H12 *= area
    H20 *= area; H21 *= area; H22 *= area

    return f_a0, f_a1, f_a2, H00, H01, H02, H10, H11, H12, H20, H21, H22

@wp.func
def zcy_evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        ### edge1
        collision_force_10 = collision_force * bs[0]
        collision_force_11 = collision_force * bs[1]

        collision_hessian_10 = collision_hessian * bs[0] * bs[0]
        collision_hessian_11 = collision_hessian * bs[1] * bs[1]
        collision_hessian_101 = collision_hessian * bs[0] * bs[1] #+ bs[0] * bs[1] * friction_hessian
        collision_hessian_110 = collision_hessian * bs[1] * bs[0] #+ bs[1] * bs[0] * friction_hessian
        # collision_normal_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)

        ### edge2
        collision_force_20 = collision_force * bs[2]
        collision_force_21 = collision_force * bs[3]

        collision_hessian_20 = collision_hessian * bs[2] * bs[2]
        collision_hessian_21 = collision_hessian * bs[3] * bs[3]
        collision_hessian_201 = collision_hessian * bs[2] * bs[3] #+ bs[2] * bs[3] * friction_hessian
        collision_hessian_210 = collision_hessian * bs[3] * bs[2] #+ bs[3] * bs[2] * friction_hessian
        # collision_normal_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)

        # edge1 to edge2
        collision_hessian_1200 = bs[0] * bs[2] * collision_hessian  #+ bs[0] * bs[2] * friction_hessian
        collision_hessian_1201 = bs[0] * bs[3] * collision_hessian  #+ bs[0] * bs[3] * friction_hessian
        collision_hessian_1210 = bs[1] * bs[2] * collision_hessian  #+ bs[1] * bs[2] * friction_hessian
        collision_hessian_1211 = bs[1] * bs[3] * collision_hessian  #+ bs[1] * bs[3] * friction_hessian

        # edge2 to edge1
        collision_hessian_2100 = bs[2] * bs[0] * collision_hessian  #+ bs[2] * bs[0] * friction_hessian
        collision_hessian_2101 = bs[2] * bs[1] * collision_hessian  #+ bs[2] * bs[1] * friction_hessian
        collision_hessian_2110 = bs[3] * bs[0] * collision_hessian  #+ bs[3] * bs[0] * friction_hessian
        collision_hessian_2111 = bs[3] * bs[1] * collision_hessian  #+ bs[3] * bs[1] * friction_hessian

        return True, collision_force_10, collision_force_11, collision_hessian_10, collision_hessian_11, \
                    collision_force_20, collision_force_21, collision_hessian_20, collision_hessian_21, \
                    collision_hessian_101, collision_hessian_110, collision_hessian_201, collision_hessian_210, \
                    collision_hessian_1200, collision_hessian_1201, collision_hessian_1210, collision_hessian_1211, \
                    collision_hessian_2100, collision_hessian_2101, collision_hessian_2110, collision_hessian_2111
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_hessian, collision_hessian,\
                    collision_force, collision_force, collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian

@wp.func
def zcy_evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]
        # collision_normal_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)

        # vertex to triangle
        collision_hessian_01 = bs[0] * bs[1] * collision_hessian #+ bs[0] * bs[1] * friction_hessian
        collision_hessian_02 = bs[0] * bs[2] * collision_hessian #+ bs[0] * bs[2] * friction_hessian
        collision_hessian_03 = bs[0] * bs[3] * collision_hessian #+ bs[0] * bs[3] * friction_hessian
        collision_hessian_10 = bs[1] * bs[0] * collision_hessian #+ bs[1] * bs[0] * friction_hessian
        collision_hessian_12 = bs[1] * bs[2] * collision_hessian #+ bs[1] * bs[2] * friction_hessian
        collision_hessian_13 = bs[1] * bs[3] * collision_hessian #+ bs[1] * bs[3] * friction_hessian
        collision_hessian_20 = bs[2] * bs[0] * collision_hessian #+ bs[2] * bs[0] * friction_hessian
        collision_hessian_21 = bs[2] * bs[1] * collision_hessian #+ bs[2] * bs[1] * friction_hessian
        collision_hessian_23 = bs[2] * bs[3] * collision_hessian #+ bs[2] * bs[3] * friction_hessian
        collision_hessian_30 = bs[3] * bs[0] * collision_hessian #+ bs[3] * bs[0] * friction_hessian
        collision_hessian_31 = bs[3] * bs[1] * collision_hessian #+ bs[3] * bs[1] * friction_hessian
        collision_hessian_32 = bs[3] * bs[2] * collision_hessian #+ bs[3] * bs[2] * friction_hessian


        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
            collision_hessian_01,
            collision_hessian_02,
            collision_hessian_03,
            collision_hessian_10,
            collision_hessian_12,
            collision_hessian_13,
            collision_hessian_20,
            collision_hessian_21,
            collision_hessian_23,
            collision_hessian_30,
            collision_hessian_31,
            collision_hessian_32,
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )

@wp.func
def zcy_evaluate_spring_force_and_hessian(
    v0: wp.vec3, 
    v1: wp.vec3,
    l0: float,
    k: float,
):
    # 计算向量与长度
    diff = v0 - v1
    l = wp.length(diff)

    # 防止除以零
    if l < 1.0e-6:
        return wp.vec3(0.0, 0.0, 0.0), wp.mat33(0.0)

    # 方向
    n = diff / l

    # -------------------------------
    # 力：F = -k (l - l0) * n
    # -------------------------------
    f = - k * (l - l0) * n

    # -------------------------------
    # 正定 Hessian (近似几何刚度项)
    # -------------------------------
    # 这里用标准弹簧 Hessian 推导形式
    # H = k [ n n^T + ((l - l0)/l) * (I - n n^T) ]
    I = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        )
    H = k * (wp.outer(n, n) + ((l - l0) / l) * (I - wp.outer(n, n)))

    return f, H

# zcy


@wp.func
def evaluate_vertex_triangle_collision_force_hessian(
    v: int,
    v_order: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1], friction_force[2],
            )
        # fmt: on

        if v_order == 0:
            displacement = pos_prev[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_prev[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_prev[tri_indices[tri, 2]] - c
        else:
            displacement = pos_prev[v] - p

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + v_bary * friction_force
        collision_hessian = collision_hessian + v_bary * v_bary * friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1],
                friction_force[2],
            )
        # fmt: on

        displacement_0 = pos_prev[tri_indices[tri, 0]] - a
        displacement_1 = pos_prev[tri_indices[tri, 1]] - b
        displacement_2 = pos_prev[tri_indices[tri, 2]] - c
        displacement_3 = pos_prev[v] - p

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_2,
            collision_normal * collision_normal_sign[2],
            collision_hessian_2,
            collision_damping,
            dt,
        )
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_3,
            collision_normal * collision_normal_sign[3],
            collision_hessian_3,
            collision_damping,
            dt,
        )
        collision_force_3 += damping_force + bs[3] * friction_force
        collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian
        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )


@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix).
        u: 2D displacement vector.
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.vec3,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE:
        inertia[particle] = pos_prev[particle]
        return
    vel_new = vel[particle] + (gravity + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]


@wp.kernel
def forward_step_penetration_free(
    dt: float,
    gravity: wp.vec3,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()

    pos_prev[particle_index] = pos[particle_index]
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        inertia[particle_index] = pos_prev[particle_index]
        return
    vel_new = vel[particle_index] + (gravity + external_force[particle_index] * inv_mass[particle_index]) * dt
    pos_inertia = pos[particle_index] + vel_new * dt
    inertia[particle_index] = pos_inertia

    pos[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_inertia, pos_prev_collision_detection, particle_conservative_bounds
    )


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def validate_conservative_bound(
    pos: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    v_index = wp.tid()

    displacement = wp.length(pos[v_index] - pos_prev_collision_detection[v_index])

    if displacement > particle_conservative_bounds[v_index] * 1.01 and displacement > 1e-5:
        # wp.expect_eq(displacement <= particle_conservative_bounds[v_index] * 1.01, True)
        wp.printf(
            "Vertex %d has moved by %f exceeded the limit of %f\n",
            v_index,
            displacement,
            particle_conservative_bounds[v_index],
        )


@wp.func
def apply_conservative_bound_truncation(
    v_index: wp.int32,
    pos_new: wp.vec3,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_pos_prev_collision_detection = pos_prev_collision_detection[v_index]
    accumulated_displacement = pos_new - particle_pos_prev_collision_detection
    conservative_bound = particle_conservative_bounds[v_index]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > 1e-5:
        accumulated_displacement_norm_truncated = conservative_bound
        accumulated_displacement = accumulated_displacement * (
            accumulated_displacement_norm_truncated / accumulated_displacement_norm
        )

        return particle_pos_prev_collision_detection + accumulated_displacement
    else:
        return pos_new


@wp.kernel
def solve_trimesh_no_self_contact_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # # inertia force and hessian
    # f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    # h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )
        # compute damping

        f += f_tri
        h += h_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                thread_idx,
                vertex_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
            # fmt: on

    #
    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )

            pos_new[particle_index] = particle_pos + h_inv * f_total


@wp.kernel
def solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_index = particle_ids_in_color[tid]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_id, 0],
                tri_indices[tri_id, 1],
                tri_indices[tri_id, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_id,
            particle_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_id],
            tri_areas[tri_id],
            tri_materials[tri_id, 0],
            tri_materials[tri_id, 1],
            tri_materials[tri_id, 2],
            dt,
        )

        f = f + f_tri
        h = h + h_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_adj_tri,
                particle_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
        # fmt: on

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    h += particle_hessians[particle_index]
    f += particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


@wp.kernel
def copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = particle_ids_in_color[tid]

    pos[particle] = pos_new[particle]


@wp.kernel
def update_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
):
    contact_index = wp.tid()
    count = min(contact_max, contact_count[0])
    if contact_index >= count:
        return

    particle_index = soft_contact_particle[contact_index]
    offset = particle_index * body_particle_contact_buffer_pre_alloc

    contact_counter = wp.atomic_add(body_particle_contact_count, particle_index, 1)
    if contact_counter < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_buffer[offset + contact_counter] = contact_index


@wp.kernel
def accumulate_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # body-particle contact
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_prev,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_prev,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                    )

                    if has_contact:
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


# zcy
@wp.kernel
def zcy_forward_step_penetration_free(
    dt: float,
    gravity: wp.vec3,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    all_particle_flag: wp.array(dtype=wp.int32),
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    vel_new = vel[particle_index] + gravity * dt
    pos_inertia = prev_pos[particle_index] + vel_new * dt
    inertia[particle_index] = pos_inertia

    pos[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_inertia, pos_prev_collision_detection, particle_conservative_bounds
    )

@wp.kernel
def zcy_truncation_by_conservative_bounds(
    pos_new: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    pos_cur_truncation: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()

    pos_cur_truncation[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_new[particle_index], pos_prev_collision_detection, particle_conservative_bounds
    )

@wp.kernel
def zcy_VBD_accumulate_contact_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    # edge_contact
    edge_contact_forces: wp.array(dtype=wp.vec3),
    edge_contact_hessian_values: wp.array(dtype=wp.mat33),
    edge_contact_hessian_rows: wp.array(dtype=int),
    edge_contact_hessian_cols: wp.array(dtype=int),
    # vertex-triangle_contact
    vt_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_hessian_values: wp.array(dtype=wp.mat33),
    vt_contact_hessian_rows: wp.array(dtype=int),
    vt_contact_hessian_cols: wp.array(dtype=int),
):
    
    t_id = wp.tid()
    collision_info = collision_info_array[0]
    
    # process edge-edge collisions
    if t_id * 2 < collision_info.edge_colliding_edges.shape[0]:
        e1_idx = collision_info.edge_colliding_edges[2 * t_id]
        e2_idx = collision_info.edge_colliding_edges[2 * t_id + 1]

        if e1_idx != -1 and e2_idx != -1:
            e1_v1 = edge_indices[e1_idx, 2]
            e1_v2 = edge_indices[e1_idx, 3]
            e2_v1 = edge_indices[e2_idx, 2]
            e2_v2 = edge_indices[e2_idx, 3]

            has_contact, collision_force_10, collision_force_11, collision_hessian_10, collision_hessian_11, \
                collision_force_20, collision_force_21, collision_hessian_20, collision_hessian_21, \
                collision_hessian_101, collision_hessian_110, collision_hessian_201, collision_hessian_210, \
                collision_hessian_1200, collision_hessian_1201, collision_hessian_1210, collision_hessian_1211, \
                collision_hessian_2100, collision_hessian_2101, collision_hessian_2110, collision_hessian_2111 = \
            (
                zcy_evaluate_edge_edge_contact_2_vertices(
                    e1_idx,
                    e2_idx,
                    pos,
                    edge_indices,
                    collision_radius,
                    soft_contact_ke,
                    soft_contact_kd,
                    friction_mu,
                    friction_epsilon,
                    edge_edge_parallel_epsilon,
                )
            )

            #加两遍，除2
            if has_contact:
                # edge1
                # force
                wp.atomic_add(edge_contact_forces, e1_v1, collision_force_10*0.5 )
                wp.atomic_add(edge_contact_forces, e1_v2, collision_force_11*0.5 )

                # edge2
                # force
                wp.atomic_add(edge_contact_forces, e2_v1, collision_force_20*0.5 )
                wp.atomic_add(edge_contact_forces, e2_v2, collision_force_21*0.5 )

                # 假设每个 contact 预分配 16 个条目
                # contact_base: 每个 contact 的起始索引
                contact_index = t_id
                contact_base = contact_index * 16  # contact_index 需根据循环传入

                # --- edge1 ---
                # (e1_v1, e1_v1)
                edge_contact_hessian_rows[contact_base + 0] = e1_v1
                edge_contact_hessian_cols[contact_base + 0] = e1_v1
                edge_contact_hessian_values[contact_base + 0] = collision_hessian_10*0.5
                # (e1_v1, e1_v2)
                edge_contact_hessian_rows[contact_base + 1] = e1_v1
                edge_contact_hessian_cols[contact_base + 1] = e1_v2
                edge_contact_hessian_values[contact_base + 1] = collision_hessian_101*0.5
                # (e1_v2, e1_v1)
                edge_contact_hessian_rows[contact_base + 2] = e1_v2
                edge_contact_hessian_cols[contact_base + 2] = e1_v1
                edge_contact_hessian_values[contact_base + 2] = collision_hessian_110*0.5
                # (e1_v2, e1_v2)
                edge_contact_hessian_rows[contact_base + 3] = e1_v2
                edge_contact_hessian_cols[contact_base + 3] = e1_v2
                edge_contact_hessian_values[contact_base + 3] = collision_hessian_11*0.5

                # --- edge2 ---
                edge_contact_hessian_rows[contact_base + 4] = e2_v1
                edge_contact_hessian_cols[contact_base + 4] = e2_v1
                edge_contact_hessian_values[contact_base + 4] = collision_hessian_20*0.5

                edge_contact_hessian_rows[contact_base + 5] = e2_v1
                edge_contact_hessian_cols[contact_base + 5] = e2_v2
                edge_contact_hessian_values[contact_base + 5] = collision_hessian_201*0.5

                edge_contact_hessian_rows[contact_base + 6] = e2_v2
                edge_contact_hessian_cols[contact_base + 6] = e2_v1
                edge_contact_hessian_values[contact_base + 6] = collision_hessian_210*0.5

                edge_contact_hessian_rows[contact_base + 7] = e2_v2
                edge_contact_hessian_cols[contact_base + 7] = e2_v2
                edge_contact_hessian_values[contact_base + 7] = collision_hessian_21*0.5

                # --- edge1 <-> edge2 cross blocks ---
                edge_contact_hessian_rows[contact_base + 8] = e1_v1
                edge_contact_hessian_cols[contact_base + 8] = e2_v1
                edge_contact_hessian_values[contact_base + 8] = collision_hessian_1200*0.5

                edge_contact_hessian_rows[contact_base + 9] = e1_v1
                edge_contact_hessian_cols[contact_base + 9] = e2_v2
                edge_contact_hessian_values[contact_base + 9] = collision_hessian_1201*0.5

                edge_contact_hessian_rows[contact_base + 10] = e1_v2
                edge_contact_hessian_cols[contact_base + 10] = e2_v1
                edge_contact_hessian_values[contact_base + 10] = collision_hessian_1210*0.5

                edge_contact_hessian_rows[contact_base + 11] = e1_v2
                edge_contact_hessian_cols[contact_base + 11] = e2_v2
                edge_contact_hessian_values[contact_base + 11] = collision_hessian_1211*0.5

                # --- edge2 <-> edge1 cross blocks ---
                edge_contact_hessian_rows[contact_base + 12] = e2_v1
                edge_contact_hessian_cols[contact_base + 12] = e1_v1
                edge_contact_hessian_values[contact_base + 12] = collision_hessian_2100*0.5

                edge_contact_hessian_rows[contact_base + 13] = e2_v1
                edge_contact_hessian_cols[contact_base + 13] = e1_v2
                edge_contact_hessian_values[contact_base + 13] = collision_hessian_2101*0.5

                edge_contact_hessian_rows[contact_base + 14] = e2_v2
                edge_contact_hessian_cols[contact_base + 14] = e1_v1
                edge_contact_hessian_values[contact_base + 14] = collision_hessian_2110*0.5

                edge_contact_hessian_rows[contact_base + 15] = e2_v2
                edge_contact_hessian_cols[contact_base + 15] = e1_v2
                edge_contact_hessian_values[contact_base + 15] = collision_hessian_2111*0.5


    # process vertex-triangle collisions
    if t_id * 2 < collision_info.vertex_colliding_triangles.shape[0]:
        particle_idx = collision_info.vertex_colliding_triangles[2 * t_id]
        tri_idx = collision_info.vertex_colliding_triangles[2 * t_id + 1]

        if particle_idx != -1 and tri_idx != -1:
            tri_a = tri_indices[tri_idx, 0]
            tri_b = tri_indices[tri_idx, 1]
            tri_c = tri_indices[tri_idx, 2]
            (
                has_contact,
                collision_force_0,
                collision_force_1,
                collision_force_2,
                collision_force_3,
                collision_hessian_0,
                collision_hessian_1,
                collision_hessian_2,
                collision_hessian_3,
                collision_hessian_01,
                collision_hessian_02,
                collision_hessian_03,
                collision_hessian_10,
                collision_hessian_12,
                collision_hessian_13,
                collision_hessian_20,
                collision_hessian_21,
                collision_hessian_23,
                collision_hessian_30,
                collision_hessian_31,
                collision_hessian_32,
            ) = zcy_evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                particle_idx,
                tri_idx,
                pos,
                tri_indices,
                collision_radius,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
            )
            if has_contact:
                contact_index = t_id
                contact_base = contact_index * 16  # 每个 particle-tri contact 占16个条目

                # --- 力累加 ---
                wp.atomic_add(vt_contact_forces, particle_idx, collision_force_3)
                wp.atomic_add(vt_contact_forces, tri_a, collision_force_0)
                wp.atomic_add(vt_contact_forces, tri_b, collision_force_1)
                wp.atomic_add(vt_contact_forces, tri_c, collision_force_2)

                # --- 对角块 ---
                # particle
                vt_contact_hessian_rows[contact_base + 0] = particle_idx
                vt_contact_hessian_cols[contact_base + 0] = particle_idx
                vt_contact_hessian_values[contact_base + 0] = collision_hessian_3

                # tri_a
                vt_contact_hessian_rows[contact_base + 1] = tri_a
                vt_contact_hessian_cols[contact_base + 1] = tri_a
                vt_contact_hessian_values[contact_base + 1] = collision_hessian_0

                # tri_b
                vt_contact_hessian_rows[contact_base + 2] = tri_b
                vt_contact_hessian_cols[contact_base + 2] = tri_b
                vt_contact_hessian_values[contact_base + 2] = collision_hessian_1

                # tri_c
                vt_contact_hessian_rows[contact_base + 3] = tri_c
                vt_contact_hessian_cols[contact_base + 3] = tri_c
                vt_contact_hessian_values[contact_base + 3] = collision_hessian_2

                # --- cross blocks ---
                # a0
                vt_contact_hessian_rows[contact_base + 4] = tri_a
                vt_contact_hessian_cols[contact_base + 4] = tri_b
                vt_contact_hessian_values[contact_base + 4] = collision_hessian_01

                vt_contact_hessian_rows[contact_base + 5] = tri_a
                vt_contact_hessian_cols[contact_base + 5] = tri_c
                vt_contact_hessian_values[contact_base + 5] = collision_hessian_02

                vt_contact_hessian_rows[contact_base + 6] = tri_a
                vt_contact_hessian_cols[contact_base + 6] = particle_idx
                vt_contact_hessian_values[contact_base + 6] = collision_hessian_03

                # b1
                vt_contact_hessian_rows[contact_base + 7] = tri_b
                vt_contact_hessian_cols[contact_base + 7] = tri_a
                vt_contact_hessian_values[contact_base + 7] = collision_hessian_10

                vt_contact_hessian_rows[contact_base + 8] = tri_b
                vt_contact_hessian_cols[contact_base + 8] = tri_c
                vt_contact_hessian_values[contact_base + 8] = collision_hessian_12

                vt_contact_hessian_rows[contact_base + 9] = tri_b
                vt_contact_hessian_cols[contact_base + 9] = particle_idx
                vt_contact_hessian_values[contact_base + 9] = collision_hessian_13

                # c2
                vt_contact_hessian_rows[contact_base + 10] = tri_c
                vt_contact_hessian_cols[contact_base + 10] = tri_a
                vt_contact_hessian_values[contact_base + 10] = collision_hessian_20

                vt_contact_hessian_rows[contact_base + 11] = tri_c
                vt_contact_hessian_cols[contact_base + 11] = tri_b
                vt_contact_hessian_values[contact_base + 11] = collision_hessian_21

                vt_contact_hessian_rows[contact_base + 12] = tri_c
                vt_contact_hessian_cols[contact_base + 12] = particle_idx
                vt_contact_hessian_values[contact_base + 12] = collision_hessian_23

                # p3
                vt_contact_hessian_rows[contact_base + 13] = particle_idx
                vt_contact_hessian_cols[contact_base + 13] = tri_a
                vt_contact_hessian_values[contact_base + 13] = collision_hessian_30

                vt_contact_hessian_rows[contact_base + 14] = particle_idx
                vt_contact_hessian_cols[contact_base + 14] = tri_b
                vt_contact_hessian_values[contact_base + 14] = collision_hessian_31

                vt_contact_hessian_rows[contact_base + 15] = particle_idx
                vt_contact_hessian_cols[contact_base + 15] = tri_c
                vt_contact_hessian_values[contact_base + 15] = collision_hessian_32


@wp.kernel
def zcy_accumulate_spring_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    # outputs: particle force and hessian
    spring_forces: wp.array(dtype=wp.vec3),
    spring_hessian_values: wp.array(dtype=wp.mat33),
    spring_hessian_rows: wp.array(dtype=int),
    spring_hessian_cols: wp.array(dtype=int)
):
    spring_index = wp.tid()
    # 获取两个端点
    i = spring_indices[spring_index * 2]
    j = spring_indices[spring_index * 2 + 1]
    v0 = pos[i]
    v1 = pos[j]

    f_ij, H = zcy_evaluate_spring_force_and_hessian(
        v0, v1,
        spring_rest_length[spring_index],
        spring_stiffness[spring_index],
    )

    # --- 累加到端点 ---
    # i: 受到 +f_ij 力
    # j: 受到 -f_ij 力
    wp.atomic_add(spring_forces, i, f_ij)
    wp.atomic_add(spring_forces, j, -f_ij)

    # 记录4个对称块
    # 每个spring_index 生成4个条目：base + [0,1,2,3]
    base = spring_index * 4

    # (i,i): +H
    spring_hessian_rows[base + 0] = i
    spring_hessian_cols[base + 0] = i
    spring_hessian_values[base + 0] = H

    # (i,j): -H
    spring_hessian_rows[base + 1] = i
    spring_hessian_cols[base + 1] = j
    spring_hessian_values[base + 1] = -H

    # (j,i): -H
    spring_hessian_rows[base + 2] = j
    spring_hessian_cols[base + 2] = i
    spring_hessian_values[base + 2] = -H

    # (j,j): +H
    spring_hessian_rows[base + 3] = j
    spring_hessian_cols[base + 3] = j
    spring_hessian_values[base + 3] = H

@wp.kernel
def zcy_assemble_inertia_and_gravity_add_force(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # force
    spring_forces: wp.array(dtype=wp.vec3),
    edge_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_forces: wp.array(dtype=wp.vec3),
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    b: wp.array(dtype=wp.vec3)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # wp.printf("Thread %d: free_particle = %d\n", tid, free_particle)

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    b[tid] = -1.0/dt/dt * mass * inertia + spring_forces[free_particle] + edge_contact_forces[free_particle] + vt_contact_forces[free_particle] + gravity

@wp.kernel
def zcy_residual_computation(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # force
    spring_forces: wp.array(dtype=wp.vec3),
    edge_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_forces: wp.array(dtype=wp.vec3),
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    residual: wp.array(dtype=wp.vec3)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # wp.printf("Thread %d: free_particle = %d\n", tid, free_particle)

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    residual[tid] = inertia - 1.0/mass *dt*dt * (spring_forces[free_particle] + edge_contact_forces[free_particle] + vt_contact_forces[free_particle] + gravity)


@wp.kernel
def zcy_update_velocity(
    dt: float, damping: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), 
    vel: wp.array(dtype=wp.vec3), all_particle_flag: wp.array(dtype=wp.int32)
):
    particle = wp.tid()

    if all_particle_flag[particle] == -1:
        return

    vel[particle] = damping * (pos[particle] - pos_prev[particle]) / dt

@wp.kernel
def zcy_update_position(
    pos: wp.array(dtype=wp.vec3), dx: wp.array(dtype=wp.vec3), all_particle_flag: wp.array(dtype=wp.int32)
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    offset = all_particle_flag[particle_index]
    pos[particle_index] += dx[particle_index-offset]



def warp_coo_deduplicate(rows, cols, vals):
    """
    去重 COO 格式，vals 为 3x3 矩阵块，只做 sum 聚合
    """
    rows_np = rows.numpy()
    cols_np = cols.numpy()
    vals_np = vals.numpy()  # shape (nnz, 3, 3)
    
    max_col = np.max(cols_np) if len(cols_np) > 0 else 0
    idx = rows_np * (max_col + 1) + cols_np
    
    unique_idx, inv = np.unique(idx, return_inverse=True)
    n_unique = len(unique_idx)
    
    out_rows_np = unique_idx // (max_col + 1)
    out_cols_np = unique_idx % (max_col + 1)
    
    # 向量化累加 3x3 块
    vals_flat = vals_np.reshape(vals_np.shape[0], -1)   # (nnz, 9)
    out_vals_flat = np.zeros((n_unique, 9), dtype=vals_np.dtype)
    np.add.at(out_vals_flat, inv, vals_flat)
    out_vals_np = out_vals_flat.reshape(n_unique, 3, 3)

    return (
        wp.array(out_rows_np, dtype=int),
        wp.array(out_cols_np, dtype=int),
        wp.array(out_vals_np, dtype=vals.dtype)
    )

def remove_fixed_blocks(rows, cols, vals, flag_all_particle):
    """
    从 COO (rows, cols, vals) 中删除涉及 fixed_points 的块，
    并根据 flag_all_particle 进行行列偏移。
    
    参数:
        rows_np, cols_np : np.ndarray, shape (nnz,)
        vals_np          : np.ndarray, shape (nnz, 3, 3)
        flag_all_particle : np.ndarray, shape (n_points,)
            -1 表示固定点
             其他值表示偏移量（即删除的固定点数）
    返回:
        过滤并重编号后的 (rows_np, cols_np, vals_np)
    """
    rows_np = rows.numpy()
    cols_np = cols.numpy()
    vals_np = vals.numpy() 
    flag_all_particle = flag_all_particle.numpy()

    nnz = len(rows_np)
    keep_mask = np.ones(nnz, dtype=bool)

    # 1️⃣ 找出涉及固定点的条目
    for i in range(nnz):
        if flag_all_particle[rows_np[i]] == -1 or flag_all_particle[cols_np[i]] == -1:
            keep_mask[i] = False

    # 2️⃣ 保留非固定条目
    rows_np = rows_np[keep_mask]
    cols_np = cols_np[keep_mask]
    vals_np = vals_np[keep_mask]

    # 3️⃣ 应用偏移：新索引 = 原索引 - flag_all_particle[原索引]
    rows_np = rows_np - flag_all_particle[rows_np]
    cols_np = cols_np - flag_all_particle[cols_np]

    return (
        wp.array(rows_np, dtype=int),
        wp.array(cols_np, dtype=int),
        wp.array(vals_np, dtype=wp.mat33)
    )

def build_bsr_from_block_coo(blocks_data: np.ndarray,
                              row: np.ndarray,
                              col: np.ndarray,
                              nb: int,
                              blocksize: tuple[int, int] = (3, 3),
                              sort_blocks: bool = True) -> bsr_matrix:
    """
    用 3×3×N 的数值块数组 + 块级坐标 (row, col) 构造 BSR 矩阵。

    参数:
      - blocks_data: 形状为 (nnz_blocks, br, bc) 的数值数组，例如 (N, 3, 3)。
      - row, col: 每个块的块行与块列索引，长度为 nnz_blocks，范围在 [0, nb)。
      - nb: 块行/块列的总数（最终矩阵尺寸为 (nb*br, nb*bc)）。
      - blocksize: 每个块的尺寸 (br, bc)，默认 (3, 3)。
      - sort_blocks: 是否按 (row, col) 排序以生成规范的 indptr/indices。

    返回:
      - scipy.sparse.bsr_matrix，形状为 (nb*br, nb*bc)。
    """
    blocks_data = np.asarray(blocks_data, dtype=np.float64)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)

    if blocks_data.ndim != 3:
        raise ValueError("blocks_data 必须是三维数组，形状为 (nnz_blocks, br, bc)")
    nnz_blocks, br, bc = blocks_data.shape
    if (br, bc) != tuple(blocksize):
        raise ValueError(f"块尺寸不匹配: blocks_data 为 {(br, bc)}, 期望 {blocksize}")
    if row.shape != (nnz_blocks,) or col.shape != (nnz_blocks,):
        raise ValueError("row/col 长度必须与块数 nnz_blocks 相同")
    if np.any(row < 0) or np.any(row >= nb) or np.any(col < 0) or np.any(col >= nb):
        raise ValueError("row/col 索引越界: 必须在 [0, nb) 范围内")

    # 规范顺序：按 (row, col) 排序，便于构建 indptr/indices
    if sort_blocks:
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        blocks_data = blocks_data[order]

    # 构造 BSR 压缩格式需要的 indptr/indices
    counts = np.bincount(row, minlength=nb)
    indptr = np.empty(nb + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = col

    A_bsr = bsr_matrix((blocks_data, indices, indptr), shape=(nb * br, nb * bc))
    return A_bsr


@wp.kernel
def zcy_accumulate_stvk_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    # stvk force and hessian
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    # outputs: particle force and hessian
    stvk_forces: wp.array(dtype=wp.vec3),
    stvk_hessian_values: wp.array(dtype=wp.mat33),
    stvk_hessian_rows: wp.array(dtype=int),
    stvk_hessian_cols: wp.array(dtype=int)
):
    tri_index = wp.tid()
    
    # 获取当前三角形的索引和顶点顺序
    a = tri_indices[tri_index, 0]
    b = tri_indices[tri_index, 1]
    c = tri_indices[tri_index, 2]

    # elastic force and hessian
    f_a, f_b, f_c, h_aa, h_ab, h_ac, h_ba, h_bb, h_bc, h_ca, h_cb, h_cc = zcy_evaluate_stvk_force_hessian(
        tri_index,
        pos,
        tri_indices,
        tri_poses[tri_index],
        tri_areas[tri_index],
        tri_materials[tri_index, 0],
        tri_materials[tri_index, 1],
    )

    # --- 累加到端点 ---
    wp.atomic_add(stvk_forces, a, f_a)
    wp.atomic_add(stvk_forces, b, f_b)
    wp.atomic_add(stvk_forces, c, f_c)

    # 记录9个对称块
    # 每个spring_index 生成9个条目：base + [0,1,2,3,4,5,6,7,8]
    base = tri_index * 9

    # (a,a):
    stvk_hessian_rows[base + 0] = a
    stvk_hessian_cols[base + 0] = a
    stvk_hessian_values[base + 0] = h_aa

    # (a,b):
    stvk_hessian_rows[base + 1] = a
    stvk_hessian_cols[base + 1] = b
    stvk_hessian_values[base + 1] = h_ab

    # (a,c):
    stvk_hessian_rows[base + 2] = a
    stvk_hessian_cols[base + 2] = c
    stvk_hessian_values[base + 2] = h_ac

    # (b,a): 
    stvk_hessian_rows[base + 3] = b
    stvk_hessian_cols[base + 3] = a
    stvk_hessian_values[base + 3] = h_ba

    # (b,b): 
    stvk_hessian_rows[base + 4] = b
    stvk_hessian_cols[base + 4] = b
    stvk_hessian_values[base + 4] = h_bb

    # (b,c): 
    stvk_hessian_rows[base + 5] = b
    stvk_hessian_cols[base + 5] = c
    stvk_hessian_values[base + 5] = h_bc

    # (c,a): 
    stvk_hessian_rows[base + 6] = c
    stvk_hessian_cols[base + 6] = a
    stvk_hessian_values[base + 6] = h_ca

    # (c,b): 
    stvk_hessian_rows[base + 7] = c
    stvk_hessian_cols[base + 7] = b
    stvk_hessian_values[base + 7] = h_cb

    # (c,c): 
    stvk_hessian_rows[base + 8] = c
    stvk_hessian_cols[base + 8] = c
    stvk_hessian_values[base + 8] = h_cc


# zcy



@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    l = wp.length(diff)
    l0 = spring_rest_length[spring_idx]

    force_sign = 1.0 if particle_idx == v0 else -1.0

    spring_force = force_sign * spring_stiffness[spring_idx] * (l0 - l) / l * diff
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float) - (l0 / l) * (wp.identity(3, float) - wp.outer(diff, diff) / (l * l))
    )

    # compute damping
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    f_d = h_d * (pos_prev[particle_idx] - pos[particle_idx])

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.kernel
def accumulate_spring_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_ids_in_color: wp.array(dtype=int),
    adjacency: ForceElementAdjacencyInfo,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    num_adj_springs = get_vertex_num_adjacent_springs(adjacency, particle_index)
    for spring_counter in range(num_adj_springs):
        spring_index = get_vertex_adjacent_spring_id(adjacency, particle_index, spring_counter)
        spring_force, spring_hessian = evaluate_spring_force_and_hessian(
            particle_index,
            spring_index,
            dt,
            pos,
            pos_prev,
            spring_indices,
            spring_rest_length,
            spring_stiffness,
            spring_damping,
        )

        particle_forces[particle_index] = particle_forces[particle_index] + spring_force
        particle_hessians[particle_index] = particle_hessians[particle_index] + spring_hessian


@wp.kernel
def accumulate_contact_force_and_hessian_no_self_contact(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]
    particle_pos = pos[particle_index]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = particle_pos
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )

        f = f + f_tri
        h = h + h_tri


    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        # vertex is on the edge; otherwise it only effects the bending energy n
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_prev, edge_indices, edge_rest_angles, edge_rest_length,
                edge_bending_properties[nei_edge_index, 0], edge_bending_properties[nei_edge_index, 1], dt
            )

            f = f + f_edge
            h = h + h_edge

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        h_inv = wp.inverse(h)
        particle_pos_new = pos[particle_index] + h_inv * f

        pos_new[particle_index] = apply_conservative_bound_truncation(
            particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
        )


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )

        f += f_tri
        h += h_tri

    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_pos_new = particle_pos + h_inv * f_total

            pos_new[particle_index] = apply_conservative_bound_truncation(
                particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
            )


class zcy_SolverVBD(SolverBase):

    def __init__(
        self,
        # self parameters
        dt: float,
        mass: float,
        # fixed particle mask
        fixed_particle_num: int,
        free_particle_offset: wp.array(dtype=wp.int32),
        all_particle_flag: wp.array(dtype=wp.int32),
        # spring information
        spring_indices: wp.array(dtype=wp.int32), 
        spring_rest_length: wp.array(dtype=wp.float), 
        spring_stiffness: wp.array(dtype=wp.float),
        # defult
        model: Model,
        iterations: int = 10,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.02,
        self_contact_margin: float = 0.02,
        integrate_with_external_rigid_solver: bool = False,
        penetration_free_conservative_bound_relaxation: float = 0.42,
        friction_epsilon: float = 1e-2,
        vertex_collision_buffer_pre_alloc: int = 32,
        edge_collision_buffer_pre_alloc: int = 64,
        collision_detection_interval: int = 0,
        edge_edge_parallel_epsilon: float = 1e-5,
        use_tile_solve: bool = True
    ):
        # region: before
        super().__init__(model)
        self.iterations = iterations
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self.collision_detection_interval = collision_detection_interval

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact
        self.self_contact_radius = self_contact_radius
        self.self_contact_margin = self_contact_margin


        soft_contact_max = model.shape_count * model.particle_count
        if handle_self_contact:
            if self_contact_margin < self_contact_radius:
                raise ValueError(
                    "self_contact_margin is smaller than self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make self_contact_margin 1.5-2 times larger than self_contact_radius."
                )

            self.conservative_bound_relaxation = penetration_free_conservative_bound_relaxation
            self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
            self.particle_conservative_bounds = wp.full((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            self.collision_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                soft_contact_max,
            )
        else:
            self.collision_evaluation_kernel_launch_size = soft_contact_max
    # endregion: before

        # region: my

        # particle information
        self.num_particle = self.model.particle_count

        # spring information
        self.spring_indices = spring_indices
        self.spring_rest_length = spring_rest_length
        self.spring_stiffness = spring_stiffness
        print('\n', self.spring_indices.shape, self.spring_rest_length.shape, self.spring_stiffness.shape)
        print(self.model.tri_indices.shape, self.model.tri_poses.shape, self.model.tri_materials.shape, self.model.tri_areas.shape)

        # fixed particle
        self.fixed_particle_num = fixed_particle_num
        self.free_particle_num = self.num_particle - fixed_particle_num
        self.free_particle_offset = free_particle_offset
        self.all_particle_flag = all_particle_flag

        # sparse hessian
        self.spring = 0
        self.num_spring = self.spring_rest_length.shape[0]
        self.num_triangles = self.model.tri_indices.shape[0]
        self.num_contact = self.collision_evaluation_kernel_launch_size

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.free_particle_num, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.free_particle_num, dtype=wp.mat33, device=self.device)

        self.friction_epsilon = friction_epsilon
        
        # spring
        if self.spring:
            self.spring_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
            self.spring_hessian_values = wp.zeros(self.num_spring*4, dtype=wp.mat33, device=self.device)
            self.spring_hessian_rows = wp.zeros(self.num_spring*4, dtype=int, device=self.device)
            self.spring_hessian_cols = wp.zeros(self.num_spring*4, dtype=int, device=self.device)
        else:
            self.spring_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
            self.spring_hessian_values = wp.zeros(self.num_triangles*9, dtype=wp.mat33, device=self.device)
            self.spring_hessian_rows = wp.zeros(self.num_triangles*9, dtype=int, device=self.device)
            self.spring_hessian_cols = wp.zeros(self.num_triangles*9, dtype=int, device=self.device)

        # edge_contact
        self.edge_contact_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
        self.edge_contact_hessian_values = wp.zeros(self.num_contact*16, dtype=wp.mat33, device=self.device)
        self.edge_contact_hessian_rows = wp.zeros(self.num_contact*16, dtype=int, device=self.device)
        self.edge_contact_hessian_cols = wp.zeros(self.num_contact*16, dtype=int, device=self.device)
        # vertex-triangle_contact
        self.vt_contact_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
        self.vt_contact_hessian_values = wp.zeros(self.num_contact*16, dtype=wp.mat33, device=self.device)
        self.vt_contact_hessian_rows = wp.zeros(self.num_contact*16, dtype=int, device=self.device)
        self.vt_contact_hessian_cols = wp.zeros(self.num_contact*16, dtype=int, device=self.device)

        # static matrix
        self.zcy_compute_static_matrix(dt, mass)

        # endregion: my
        
    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")
        spring_array = model.spring_indices.to("cpu")
        face_indices = model.tri_indices.to("cpu")

        with wp.ScopedDevice("cpu"):
            if edges_array.size:
                # build vertex-edge adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent edges has been filled to each vertex
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if face_indices.size:
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # preallocate memory based on counting results
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if spring_array.size:
                # build vertex-springs adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent springs has been filled to each vertex
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

# zcy
    def zcy_simulate_one_step(
        self,  pos_warp, pos_prev_warp, vel_warp, dt: float, mass: float, damping: float, num_iter: int, tolerance: float
    ):
        # 
        #print('tri_indices', self.model.tri_indices)
        #print('edge_indices', self.model.edge_indices)

        # collision detection before initialization to compute conservative bounds for initialization
        self.zcy_collision_detection_penetration_free(pos_prev_warp)
        
        # forward
        self.zcy_forward_step_penetration_free(pos_warp, pos_prev_warp, vel_warp, dt)

        # after initialization, we need new collision detection to update the bounds
        # collision detection
        self.zcy_collision_detection_penetration_free(pos_warp)

        #print('\ncollision_info', self.trimesh_collision_detector.collision_info)
        #np.savetxt("debug_contact.txt", self.trimesh_collision_detector.vertex_colliding_triangles, fmt="%d")
        '''
        # 写入 JSON 文件
        import json
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(self.trimesh_collision_detector.collision_info, f)
        '''

        for _iter in range(num_iter):
            # collision detection
            self.zcy_collision_detection_penetration_free(pos_warp)

            # assemble matrix and vector
            A = self.zcy_assemble_matrix(pos_warp)
            b = self.zcy_assemble_vector(pos_warp, pos_prev_warp, vel_warp, dt, mass)

            '''
            # preallocate memory for dx
            dx = wp.zeros_like(b)
            # solve
            result = cg(A, b, dx, tol=1e-6, maxiter=100)
            print(f'\n --- iter:{_iter} ---')
            print('cg_result:', result)
            '''
            dx = spsolve(A.tocsr(), b.numpy().reshape(self.free_particle_num*3).astype(np.float64))
            dx = wp.array(dx.reshape(self.free_particle_num,3), dtype=wp.vec3)

            # update position
            wp.launch(
                kernel=zcy_update_position,
                inputs=[pos_warp, dx, self.all_particle_flag],
                dim=self.num_particle,
                device=self.device,
            )

            # truncation
            self.zcy_truncation_by_conservative_bound(pos_warp)

            # compute residual
            residual_norm = self.zcy_compute_residual(pos_warp, pos_prev_warp, vel_warp, dt, mass)
            print('residual_norm:', residual_norm)
            if residual_norm < tolerance:
                break
            if _iter == num_iter - 1:
                raise RuntimeError("\n--- warning: reach max iter ---\n")

        wp.launch(
            kernel=zcy_update_velocity,
            inputs=[dt, damping, pos_prev_warp, pos_warp, vel_warp, self.all_particle_flag],
            dim=self.model.particle_count,
            device=self.device,
        )

    def zcy_assemble_vector(self, pos_warp, pos_prev_warp, vel_warp, dt, mass):
        
        # inertia and gravity
        b = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)
        gravity = wp.vec3(0.0, 0.0, -9.81)
        wp.launch(
            kernel=zcy_assemble_inertia_and_gravity_add_force,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass,
                gravity,
                # force
                self.spring_forces,
                self.edge_contact_forces,
                self.vt_contact_forces,
                # fixed particle
                self.free_particle_offset,
                # outputs: 
                b
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        return b

    def zcy_assemble_matrix(self, pos_warp):
        # contact hessian
        edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values, vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values = self.zcy_compute_contact_hessian_force(pos_warp)
        # spring hessian
        spring_hessian_rows, spring_hessian_cols, spring_hessian_values = self.zcy_compute_spring_hessian_force(pos_warp)
        
        A_rows = np.concatenate((self.A_rows, spring_hessian_rows.numpy(), edge_contact_hessian_rows.numpy(), vt_contact_hessian_rows.numpy()), axis=0)
        A_cols = np.concatenate((self.A_cols, spring_hessian_cols.numpy(), edge_contact_hessian_cols.numpy(), vt_contact_hessian_cols.numpy()), axis=0)
        A_values = np.concatenate((self.A_values, spring_hessian_values.numpy(), edge_contact_hessian_values.numpy(), vt_contact_hessian_values.numpy()), axis=0)

        A_rows, A_cols, A_values = warp_coo_deduplicate(wp.array(A_rows, dtype=int), wp.array(A_cols, dtype=int), wp.array(A_values, dtype=wp.mat33))

        A_rows, A_cols, A_values = remove_fixed_blocks(A_rows, A_cols, A_values, self.all_particle_flag)

        A = build_bsr_from_block_coo(
            A_values.numpy(), A_rows.numpy(), A_cols.numpy(), 
            nb=self.free_particle_num, blocksize=(3, 3)
        )
        '''
        A = bsr_from_triplets(
                rows_of_blocks=self.free_particle_num,      # 行块数
                cols_of_blocks=self.free_particle_num,      # 列块数
                rows=A_rows,             # 块行索引
                columns=A_cols,          # 块列索引
                values=A_values            # 块数据
        )
        '''
        return A

    def zcy_compute_residual(self, pos_warp, pos_prev_warp, vel_warp, dt, mass):
        
        # inertia and gravity
        residual = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)
        gravity = wp.vec3(0.0, 0.0, -9.81)
        wp.launch(
            kernel=zcy_residual_computation,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass,
                gravity,
                # force
                self.spring_forces,
                self.edge_contact_forces,
                self.vt_contact_forces,
                # fixed particle
                self.free_particle_offset,
                # outputs: 
                residual
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        residual_norm = np.linalg.norm(np.linalg.norm(residual.numpy(), axis=1))
        return residual_norm


    def zcy_compute_static_matrix(self, dt, mass):
        # inertia and gravity
        self.A_rows = np.array([i for i in range(self.num_particle)])
        self.A_cols = np.array([i for i in range(self.num_particle)])
        self.A_values = np.array([np.eye(3) * mass / dt**2 for _ in range(self.num_particle)])
        
    def zcy_compute_contact_hessian_force(
        self, pos_warp
    ):
        # edge_contact
        self.edge_contact_forces.zero_()
        self.edge_contact_hessian_values.zero_()
        self.edge_contact_hessian_rows.zero_()
        self.edge_contact_hessian_cols.zero_()
        # vertex-triangle_contact
        self.vt_contact_forces.zero_()
        self.vt_contact_hessian_values.zero_()
        self.vt_contact_hessian_rows.zero_()
        self.vt_contact_hessian_cols.zero_()

        # dim
        wp.launch(
            kernel=zcy_VBD_accumulate_contact_force_and_hessian,
                # inputs
            dim=self.num_contact,
            inputs=[
                pos_warp,
                self.model.tri_indices,
                self.model.edge_indices,
                # self-contact
                self.trimesh_collision_info,
                self.self_contact_radius,
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
            ],
            outputs=[
                # edge_contact
                self.edge_contact_forces,
                self.edge_contact_hessian_values,
                self.edge_contact_hessian_rows,
                self.edge_contact_hessian_cols,
                # vertex-triangle_contact
                self.vt_contact_forces,
                self.vt_contact_hessian_values,
                self.vt_contact_hessian_rows,
                self.vt_contact_hessian_cols,
            ],
            device=self.device,
        )

        # edge
        #print('\n---edge---')
        #print(f"\nedge_contact_hessian_rows={self.edge_contact_hessian_rows}, edge_contact_hessian_rows.shape={self.edge_contact_hessian_rows.shape}")
        #print(f"\nedge_contact_hessian_cols={self.edge_contact_hessian_cols}, edge_contact_hessian_cols.shape={self.edge_contact_hessian_cols.shape}")
        edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values = warp_coo_deduplicate(
            self.edge_contact_hessian_rows, self.edge_contact_hessian_cols, self.edge_contact_hessian_values)
        
        # vt
        #print('\n---vt---')
        #print(f"\nvt_contact_hessian_rows={self.vt_contact_hessian_rows}, vt_contact_hessian_rows.shape={self.vt_contact_hessian_rows.shape}")
        #print(f"\nvt_contact_hessian_cols={self.vt_contact_hessian_cols}, vt_contact_hessian_cols.shape={self.vt_contact_hessian_cols.shape}")
        #np.savetxt("debug_rows.txt", self.vt_contact_hessian_rows, fmt="%d")
        #np.savetxt("debug_cols.txt", self.vt_contact_hessian_cols, fmt="%d")
        vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values = warp_coo_deduplicate(
            self.vt_contact_hessian_rows, self.vt_contact_hessian_cols, self.vt_contact_hessian_values)
        #print(f"\nvt_contact_hessian_rows={vt_contact_hessian_rows}, vt_contact_hessian_rows.shape={vt_contact_hessian_rows.shape}")
        #print(f"\nvt_contact_hessian_cols={vt_contact_hessian_cols}, vt_contact_hessian_cols.shape={vt_contact_hessian_cols.shape}")


        return edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values, vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values

    def zcy_compute_spring_hessian_force(
        self, pos_warp,
    ):
        # choose energy
        spring = 0

        # spring
        self.spring_forces.zero_()
        self.spring_hessian_values.zero_()
        self.spring_hessian_rows.zero_()
        self.spring_hessian_cols.zero_()

        # dim
        if spring :
            wp.launch(
                kernel=zcy_accumulate_spring_force_and_hessian,
                inputs=[
                    pos_warp,
                    # spring constraints
                    self.spring_indices,
                    self.spring_rest_length,
                    self.spring_stiffness,
                    # outputs: particle force and hessian
                    self.spring_forces,
                    self.spring_hessian_values,
                    self.spring_hessian_rows,
                    self.spring_hessian_cols
                ],
                dim=self.num_spring,
                device=self.device,
            )
        else :
            wp.launch(
                kernel=zcy_accumulate_stvk_force_and_hessian,
                inputs=[
                    pos_warp,
                    # stvk force and hessian
                    self.model.tri_indices,
                    self.model.tri_poses,
                    self.model.tri_materials,
                    self.model.tri_areas,
                    # outputs: particle force and hessian
                    self.spring_forces,
                    self.spring_hessian_values,
                    self.spring_hessian_rows,
                    self.spring_hessian_cols
                ],
                dim=self.num_triangles,
                device=self.device,
            )
        
        # spring
        #print('\n---spring---')
        spring_hessian_rows, spring_hessian_cols, spring_hessian_values = warp_coo_deduplicate(
            self.spring_hessian_rows, self.spring_hessian_cols, self.spring_hessian_values)

        return spring_hessian_rows, spring_hessian_cols, spring_hessian_values


    def zcy_forward_step_penetration_free(
        self, pos_warp, pos_prev_warp, vel_warp, dt: float
    ):
        model=self.model

        # give the gravity to the model
        model.gravity = wp.vec3(0.0, 0.0, -9.81)
        print(model.gravity)    

        # pos_prev_warp give information to update pos_warp
        wp.launch(
            kernel=zcy_forward_step_penetration_free,
            inputs=[
                dt,
                model.gravity,
                pos_prev_warp,
                pos_warp,
                vel_warp,
                self.pos_prev_collision_detection,
                self.particle_conservative_bounds,
                self.inertia,
                self.all_particle_flag,
            ],
            dim=model.particle_count,
            device=self.device,
        )


    def zcy_collision_detection_penetration_free(self, pos_prev_warp):
        self.trimesh_collision_detector.refit(pos_prev_warp)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)

        self.pos_prev_collision_detection.assign(pos_prev_warp)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.conservative_bound_relaxation,
                self.self_contact_margin,
                self.adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    
    def zcy_truncation_by_conservative_bound(self, pos_new):

        pos_old = wp.clone(pos_new)

        wp.launch(
            kernel=zcy_truncation_by_conservative_bounds,
            inputs=[
                pos_old,
                self.pos_prev_collision_detection,
                self.particle_conservative_bounds,
            ],
            outputs=[
                pos_new,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )
# zcy


    def collision_detection_penetration_free(self, current_state: State, dt: float):
        self.trimesh_collision_detector.refit(current_state.particle_q)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.conservative_bound_relaxation,
                self.self_contact_margin,
                self.adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.handle_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1

    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1

    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1

    @wp.kernel
    def count_num_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
            num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1

    @wp.kernel
    def fill_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_springs: wp.array(dtype=wp.int32),
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
            vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
            vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
            vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
            vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1
