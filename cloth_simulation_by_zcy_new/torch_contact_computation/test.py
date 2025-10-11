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

from torch_contact_computation import TorchContactComputation


# input_mesh
vertices = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
    [0.01, 0.01, 0.01],
    [0.01, 0.51, 0.01],
    [0.01, 0.01, 1.01]
], dtype=np.float64)

mesh_indices = np.array([0,1,2,3,4,5], dtype=np.int32)

contact_radius = 0.2
contact_margin = 0.3
contact_stiffness = 1000.0

# contact computation
contact_computation = TorchContactComputation(
    vertices, 
    mesh_indices, 
    contact_radius, 
    contact_margin, 
    contact_stiffness,
    device="cpu")
# collision detector
contact_computation.collision_detect(vertices)
# computation
energy, force, Hessian = contact_computation.compute_contact_energy_force_Hessian(vertices)
print('\nenergy:', energy)
print('\nforce:', force)
print('\nHessian:', Hessian.shape)
print('\nHessian (max min):', Hessian.max(), Hessian.min())