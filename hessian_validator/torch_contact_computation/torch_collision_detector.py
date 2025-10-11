import torch
import warp as wp
import numpy as np
from typing import Any


# cpmpute bounds
import newton
from newton._src.solvers.zcy_vbd.tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo

class TorchCollisionDetector():
    def __init__(self, vertices, mesh_indices, contact_radius=0.2, contact_margin=0.3, device="cpu"):
        """
        参数:
            vertices: torch.Tensor, [N, 3], 网格顶点坐标
            mesh_indices: torch.Tensor, [M, 3], 三角形索引
            device: 运行设备
        """
        self.device = device
        self.contact_radius = contact_radius
        self.contact_margin = contact_margin

        # self.model
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=5.0e1,
            tri_ka=5.0e1,
            tri_kd=1.0e-1,
            edge_ke=1.0e1,
            edge_kd=1.0e0,
        )
        self.model = builder.finalize()

        # collision detector
        self.collision_detector = TriMeshCollisionDetector(self.model)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''
        return:
            tri_indices: torch.Tensor [M, 3], 碰撞的三角形索引
            edge_indices: torch.Tensor [E, 2], 碰撞的边索引
        '''
        tri_indices = wp.to_torch(self.model.tri_indices)
        edge_indices = wp.to_torch(self.model.edge_indices)
        return tri_indices, edge_indices

    def detect(self, vertices: torch.Tensor):
        """
        对输入点集（PyTorch 张量）执行 Warp 碰撞检测。
        返回结果仍是 torch 张量。

        参数:
            vertices: torch.Tensor [N, 3]
        返回:
            contacts: dict 包含 {dist, normal, face_index, closest_point}
        """

        self.collision_detector.refit(vertices)
        self.collision_detector.vertex_triangle_collision_detection(self.contact_margin)
        self.collision_detector.edge_edge_collision_detection(self.contact_margin)
        self.collision_info = self.collision_detector.collision_info

        vertex_colliding_triangles = wp.to_torch(self.collision_info.vertex_colliding_triangles)
        edge_colliding_edges = wp.to_torch(self.collision_info.edge_colliding_edges)

        return vertex_colliding_triangles, edge_colliding_edges
