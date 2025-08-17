import numpy as np
from scipy.spatial import cKDTree

class ClothCollisionHandler:
    def __init__(self, faces, edges, mass=1.0, thickness=1e-3, mu=0.4):
        self.faces = faces
        self.edges = edges
        self.mass = mass
        self.h = thickness
        self.mu = mu

    def step(self, x, v, dt):
        x_bar = x + v * dt
        v_bar_half = (x_bar - x) / dt

        self.build_bvh(x)

        v_tilde_half = self.apply_repulsion_impulses(x, v_bar_half)
        v_half = self.resolve_geometric_collisions(x, v_tilde_half, dt)

        x_next = x + v_half * dt
        v_next = v_half
        return x_next, v_next

    # ----------------------------------------
    # BVH acceleration
    def build_bvh(self, x):
        centers = np.mean(x[self.faces], axis=1)
        self.face_tree = cKDTree(centers)
        edge_centers = np.mean(x[self.edges], axis=1)
        self.edge_tree = cKDTree(edge_centers)

    def find_close_faces(self, x, i):
        return self.face_tree.query_ball_point(x[i], r=self.h)

    def find_close_edges(self, x, edge):
        center = np.mean(x[list(edge)], axis=0)
        return self.edge_tree.query_ball_point(center, r=self.h)

    # ----------------------------------------
    # Repulsion impulses
    def apply_repulsion_impulses(self, x, v_half):
        updated_v = v_half.copy()
        for i4, face in self.find_close_point_triangle_pairs(x):
            i1, i2, i3 = face
            w1, w2, w3, _, n_hat = self.compute_projection_and_normal(x, face, i4)
            v_tri = w1 * v_half[i1] + w2 * v_half[i2] + w3 * v_half[i3]
            v_rel = v_half[i4] - v_tri
            v_N = np.dot(v_rel, n_hat)

            if v_N < 0:
                Ic = 0.5 * self.mass * v_N
                v_T = v_rel - v_N * n_hat
                d_vT = -np.minimum(self.mu * abs(v_N), np.linalg.norm(v_T)) * (v_T / (np.linalg.norm(v_T) + 1e-8))

                for idx, w in zip([i1, i2, i3], [w1, w2, w3]):
                    updated_v[idx] += (w * Ic / self.mass) * n_hat + (w * d_vT / self.mass)
                updated_v[i4] -= (Ic / self.mass) * n_hat + (d_vT / self.mass)

        for (i1, i2), (j1, j2) in self.find_close_edge_edge_pairs(x):
            xA1, xA2 = x[i1], x[i2]
            xB1, xB2 = x[j1], x[j2]
            a, b, closest_A, closest_B = self.closest_edge_points(xA1, xA2, xB1, xB2)
            n_hat = closest_A - closest_B
            dist = np.linalg.norm(n_hat)
            if dist > 1e-10:
                n_hat /= dist
            if dist < self.h:
                vA = (1 - a) * v_half[i1] + a * v_half[i2]
                vB = (1 - b) * v_half[j1] + b * v_half[j2]
                v_rel = vA - vB
                v_N = np.dot(v_rel, n_hat)
                if v_N < 0:
                    Ic = 0.5 * self.mass * v_N
                    dIc = self.mu * abs(Ic)
                    v_T = v_rel - v_N * n_hat
                    d_vT = -np.minimum(dIc, np.linalg.norm(v_T)) * (v_T / (np.linalg.norm(v_T) + 1e-8))

                    for idx, w in zip([i1, i2], [1 - a, a]):
                        updated_v[idx] += (w * Ic / self.mass) * n_hat + (w * d_vT / self.mass)
                    for idx, w in zip([j1, j2], [1 - b, b]):
                        updated_v[idx] -= (w * Ic / self.mass) * n_hat + (w * d_vT / self.mass)

        return updated_v

    # ----------------------------------------
    # Geometric CCD response
    def resolve_geometric_collisions(self, x, v_half, dt):
        max_iters = 10
        for _ in range(max_iters):
            converged = True
            for i4, face in self.find_colliding_point_triangle_pairs(x, v_half, dt):
                i1, i2, i3 = face
                w1, w2, w3, _, n_hat = self.compute_projection_and_normal(x, face, i4)
                v_tri = w1 * v_half[i1] + w2 * v_half[i2] + w3 * v_half[i3]
                v_rel = v_half[i4] - v_tri
                v_N = np.dot(v_rel, n_hat)
                if v_N < 0:
                    Ic = 0.5 * self.mass * v_N
                    for idx, w in zip([i1, i2, i3], [w1, w2, w3]):
                        v_half[idx] += (w * Ic / self.mass) * n_hat
                    v_half[i4] -= (Ic / self.mass) * n_hat
                    converged = False

            for e1, e2 in self.find_colliding_edge_edge_pairs(x, v_half, dt):
                iA0, iA1 = e1
                iB0, iB1 = e2
                A0, A1 = x[iA0], x[iA1]
                B0, B1 = x[iB0], x[iB1]
                a, b, pA, pB = self.closest_edge_points(A0, A1, B0, B1)
                n_hat = pA - pB
                norm = np.linalg.norm(n_hat)
                if norm < 1e-10:
                    continue
                n_hat /= norm
                vA = (1 - a) * v_half[iA0] + a * v_half[iA1]
                vB = (1 - b) * v_half[iB0] + b * v_half[iB1]
                v_rel = vA - vB
                v_N = np.dot(v_rel, n_hat)
                if v_N < 0:
                    Ic = 0.5 * self.mass * v_N
                    for idx, w in zip([iA0, iA1], [1 - a, a]):
                        v_half[idx] += (w * Ic / self.mass) * n_hat
                    for idx, w in zip([iB0, iB1], [1 - b, b]):
                        v_half[idx] -= (w * Ic / self.mass) * n_hat
                    converged = False
            if converged:
                break
        return v_half

    # ----------------------------------------
    # Utility
    def compute_projection_and_normal(self, x, face, i4):
        i1, i2, i3 = face
        x1, x2, x3 = x[i1], x[i2], x[i3]
        x4 = x[i4]
        n_hat = np.cross(x2 - x1, x3 - x1)
        n_hat /= np.linalg.norm(n_hat) + 1e-10
        d = np.dot(x4 - x1, n_hat)
        x_proj = x4 - d * n_hat
        A = np.column_stack([x1 - x3, x2 - x3])
        b = x_proj - x3
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        w1, w2 = w
        w3 = 1 - w1 - w2
        return w1, w2, w3, x_proj, n_hat

    def closest_edge_points(self, A0, A1, B0, B1):
        u = A1 - A0
        v = B1 - B0
        w0 = A0 - B0
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)
        denom = a * c - b * b
        if denom == 0: return 0.5, 0.5, (A0 + A1) / 2, (B0 + B1) / 2
        s = np.clip((b * e - c * d) / denom, 0, 1)
        t = np.clip((a * e - b * d) / denom, 0, 1)
        pA = (1 - s) * A0 + s * A1
        pB = (1 - t) * B0 + t * B1
        return s, t, pA, pB

    def find_close_point_triangle_pairs(self, x):
        for i4 in range(len(x)):
            for idx in self.find_close_faces(x, i4):
                face = self.faces[idx]
                if i4 in face:
                    continue
                x1, x2, x3, x4 = x[face[0]], x[face[1]], x[face[2]], x[i4]
                n = np.cross(x2 - x1, x3 - x1)
                n = n / (np.linalg.norm(n) + 1e-10)
                dist = abs(np.dot(x4 - x1, n))
                if dist < self.h:
                    yield i4, face

    def find_close_edge_edge_pairs(self, x):
        for i, e1 in enumerate(self.edges):
            neighbors = self.find_close_edges(x, e1)
            for j in neighbors:
                if i == j: continue
                e2 = self.edges[j]
                if len(set(e1) & set(e2)) > 0:
                    continue
                xA0, xA1 = x[e1[0]], x[e1[1]]
                xB0, xB1 = x[e2[0]], x[e2[1]]
                _, _, pA, pB = self.closest_edge_points(xA0, xA1, xB0, xB1)
                if np.linalg.norm(pA - pB) < self.h:
                    yield e1, e2

    def find_colliding_point_triangle_pairs(self, x, v, dt):
        for i4 in range(len(x)):
            for face in self.faces:
                if i4 in face: continue
                i1, i2, i3 = face
                x1_0, x2_0, x3_0, x4_0 = x[i1], x[i2], x[i3], x[i4]
                x1_1, x2_1, x3_1, x4_1 = x1_0 + dt*v[i1], x2_0 + dt*v[i2], x3_0 + dt*v[i3], x4_0 + dt*v[i4]
                def f(t):
                    x1_t = (1-t)*x1_0 + t*x1_1
                    x2_t = (1-t)*x2_0 + t*x2_1
                    x3_t = (1-t)*x3_0 + t*x3_1
                    x4_t = (1-t)*x4_0 + t*x4_1
                    n = np.cross(x2_t - x1_t, x3_t - x1_t)
                    return np.dot(x4_t - x1_t, n)
                t0, t1, f0, f1 = 0.0, 1.0, f(0), f(1)
                if f0 * f1 > 0: continue
                for _ in range(20):
                    tm = 0.5 * (t0 + t1)
                    fm = f(tm)
                    if abs(fm) < 1e-6: break
                    if f0 * fm < 0:
                        t1, f1 = tm, fm
                    else:
                        t0, f0 = tm, fm
                t_impact = 0.5 * (t0 + t1)
                x1 = (1-t_impact)*x1_0 + t_impact*x1_1
                x2 = (1-t_impact)*x2_0 + t_impact*x2_1
                x3 = (1-t_impact)*x3_0 + t_impact*x3_1
                x4 = (1-t_impact)*x4_0 + t_impact*x4_1
                n = np.cross(x2 - x1, x3 - x1)
                n /= np.linalg.norm(n) + 1e-10
                proj = x4 - np.dot(x4 - x1, n) * n
                A = np.column_stack([x1 - x3, x2 - x3])
                b = proj - x3
                w = np.linalg.lstsq(A, b, rcond=None)[0]
                w1, w2 = w
                w3 = 1 - w1 - w2
                if -0.01 <= w1 <= 1.01 and -0.01 <= w2 <= 1.01 and -0.01 <= w3 <= 1.01:
                    yield i4, face

    def find_colliding_edge_edge_pairs(self, x, v, dt):
        for e1 in self.edges:
            for e2 in self.edges:
                if len(set(e1) & set(e2)) > 0: continue
                A0_0, A1_0 = x[e1[0]], x[e1[1]]
                B0_0, B1_0 = x[e2[0]], x[e2[1]]
                A0_1 = A0_0 + dt * v[e1[0]]
                A1_1 = A1_0 + dt * v[e1[1]]
                B0_1 = B0_0 + dt * v[e2[0]]
                B1_1 = B1_0 + dt * v[e2[1]]
                def dist_sq(t):
                    A0_t = (1 - t) * A0_0 + t * A0_1
                    A1_t = (1 - t) * A1_0 + t * A1_1
                    B0_t = (1 - t) * B0_0 + t * B0_1
                    B1_t = (1 - t) * B1_0 + t * B1_1
                    _, _, pA, pB = self.closest_edge_points(A0_t, A1_t, B0_t, B1_t)
                    return np.sum((pA - pB) ** 2)
                t0, t1 = 0.0, 1.0
                d0, d1 = dist_sq(t0), dist_sq(t1)
                if d0 > self.h**2 and d1 > self.h**2: continue
                for _ in range(20):
                    tm = 0.5 * (t0 + t1)
                    dm = dist_sq(tm)
                    if dm < self.h**2:
                        yield e1, e2
                        break
                    if d0 < d1:
                        t1, d1 = tm, dm
                    else:
                        t0, d0 = tm, dm
