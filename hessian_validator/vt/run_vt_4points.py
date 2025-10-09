import torch
from hessian_vt_validator import HessianVTValidator

torch.set_default_dtype(torch.float64)
# 设置打印精度为 10 位小数
torch.set_printoptions(precision=30)

import math
import matplotlib.pyplot as plt

# 示例：用户自定义的能量函数类
class ContactEnergy:
    def __init__(self, stiffness=1.0e3, collision_radius=0.2, points=None):
        self.stiffness = stiffness
        self.collision_radius = collision_radius
        x_t1, x_t2, x_t3 = points[0], points[1], points[2]
        x_p = points[3]
        self.u = 1/3
        self.v = 1/3
        self.w = 1/3
        x_a = self.u*x_t1 + self.v*x_t2 + self.w*x_t3
        x_b = x_p
        self.hat_normal = (x_a - x_b) / torch.norm(x_a - x_b)
        self.hat_normal = self.hat_normal.detach()
    
    def __call__(self, points):
        """能量值计算"""
        x_a = self.u*points[0] + self.v*points[1] + self.w*points[2]
        x_b = points[3]
        return 0.5 * self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b))**2

    def analytical_force(self, points):
        """解析力：对于弹簧能量就是力"""
        x_a = self.u*points[0] + self.v*points[1] + self.w*points[2]
        x_b = points[3]
        force = self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b)) * self.hat_normal
        force0 = force * self.u
        force1 = force * self.v
        force2 = force * self.w
        force3 = -force
        return torch.stack([force0, force1, force2, force3], dim=0)
    
    def analytical_hessian(self, points):
        """解析Hessian：对于弹簧能量就是刚度矩阵"""
        hessian = self.stiffness * torch.outer(self.hat_normal,self.hat_normal)
        hessian00 = hessian * self.u * self.u
        hessian01 = hessian * self.u * self.v
        hessian02 = hessian * self.u * self.w
        hessian03 = -hessian * self.u

        hessian10 = hessian * self.v * self.u
        hessian11 = hessian * self.v * self.v
        hessian12 = hessian * self.v * self.w
        hessian13 = -hessian * self.v

        hessian20 = hessian * self.w * self.u
        hessian21 = hessian * self.w * self.v
        hessian22 = hessian * self.w * self.w
        hessian23 = -hessian * self.w

        hessian30 = -hessian * self.u
        hessian31 = -hessian * self.v
        hessian32 = -hessian * self.w
        hessian33 = hessian 

        hessian0 = torch.stack([hessian00, hessian10, hessian20, hessian30], dim=0)
        hessian1 = torch.stack([hessian01, hessian11, hessian21, hessian31], dim=0)
        hessian2 = torch.stack([hessian02, hessian12, hessian22, hessian32], dim=0)
        hessian3 = torch.stack([hessian03, hessian13, hessian23, hessian33], dim=0)

        return torch.stack([hessian0, hessian1, hessian2, hessian3], dim=0)

# 测试点
test_point=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3)/2, 0.0], [1/3, math.sqrt(3)/6, 0.01]], device='cpu')

# 使用
energy = ContactEnergy(stiffness=1.0e3, collision_radius=0.2, points=test_point)
print('\nceshi:', energy(test_point),'\n')

# vt验证
vt_validator = HessianVTValidator(tolerance = 1e-8, device='cpu')
result = vt_validator.validate(test_point, energy)


print('error_fd_type:', type(result['finite_diff_hessian']))
error_fd = result['finite_diff_hessian'] - result['analytical_hessian']
norm_fd = torch.max(torch.abs(error_fd))
print('\nnorm_fd:', norm_fd, '\n')

print('\nresult:', type(result))
