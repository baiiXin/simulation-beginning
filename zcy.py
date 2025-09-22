

# process edge-edge collisions
    if t_id * 2 < collision_info.edge_colliding_edges.shape[0]:
        e1_idx = collision_info.edge_colliding_edges[2 * t_id]
        e2_idx = collision_info.edge_colliding_edges[2 * t_id + 1]

        if e1_idx != -1 and e2_idx != -1:
            e1_v1 = edge_indices[e1_idx, 2]
            e1_v2 = edge_indices[e1_idx, 3]
            if particle_colors[e1_v1] == current_color or particle_colors[e1_v2] == current_color:
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
                    if particle_colors[e1_v1] == current_color:
                        wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                        wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                    if particle_colors[e1_v2] == current_color:
                        wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                        wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)

    # process vertex-triangle collisions
    if t_id * 2 < collision_info.vertex_colliding_triangles.shape[0]:
        particle_idx = collision_info.vertex_colliding_triangles[2 * t_id]
        tri_idx = collision_info.vertex_colliding_triangles[2 * t_id + 1]

        if particle_idx != -1 and tri_idx != -1:
            tri_a = tri_indices[tri_idx, 0]
            tri_b = tri_indices[tri_idx, 1]
            tri_c = tri_indices[tri_idx, 2]
            if (
                particle_colors[particle_idx] == current_color
                or particle_colors[tri_a] == current_color
                or particle_colors[tri_b] == current_color
                or particle_colors[tri_c] == current_color
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
                    if particle_colors[particle_idx] == current_color:
                        wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                        wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                    # tri_a
                    if particle_colors[tri_a] == current_color:
                        wp.atomic_add(particle_forces, tri_a, collision_force_0)
                        wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                    # tri_b
                    if particle_colors[tri_b] == current_color:
                        wp.atomic_add(particle_forces, tri_b, collision_force_1)
                        wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                    # tri_c
                    if particle_colors[tri_c] == current_color:
                        wp.atomic_add(particle_forces, tri_c, collision_force_2)
                        wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)

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
                shape_materials,
                shape_body,
                body_q,
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
