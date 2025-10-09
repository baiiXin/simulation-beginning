import torch
from hessian_ee_validator import HessianEEValidator

torch.set_default_dtype(torch.float64)
# 设置打印精度为 10 位小数
torch.set_printoptions(precision=30)

import matplotlib.pyplot as plt

# 示例：用户自定义的能量函数类
class ContactEnergy:
    def __init__(self, stiffness=1.0e3, collision_radius=0.2, points=None):
        self.stiffness = stiffness
        self.collision_radius = collision_radius
        x_a1, x_a2 = points[0], points[1]
        x_b1, x_b2 = points[2], points[3]
        self.s = 0.5
        self.t = 0.5
        x_a = (1-self.s)*x_a1 + self.s*x_a2
        x_b = (1-self.t)*x_b1 + self.t*x_b2
        self.hat_normal = (x_a - x_b) / torch.norm(x_a - x_b)
        self.hat_normal = self.hat_normal.detach()
    
    def __call__(self, points):
        """能量值计算"""
        x_a = (1-self.s)*points[0] + self.s*points[1]
        x_b = (1-self.t)*points[2] + self.t*points[3]
        return 0.5 * self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b))**2

    def analytical_force(self, points):
        """解析力：对于弹簧能量就是力"""
        x_a = (1-self.s)*points[0] + self.s*points[1]
        x_b = (1-self.t)*points[2] + self.t*points[3]
        force = self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b)) * self.hat_normal
        force0 = force * (1-self.s)
        force1 = force * self.s
        force2 = -force * (1-self.t)
        force3 = -force * self.t
        return torch.stack([force0, force1, force2, force3], dim=0)
    
    def analytical_hessian(self, points):
        """解析Hessian：对于弹簧能量就是刚度矩阵"""
        hessian = self.stiffness * torch.outer(self.hat_normal,self.hat_normal)
        hessian00 = hessian * (1-self.s) * (1-self.s)
        hessian01 = hessian * (1-self.s) * self.s
        hessian02 = -hessian * (1-self.s) * (1-self.t)
        hessian03 = -hessian * (1-self.s) * self.t

        hessian10 = hessian * self.s * (1-self.s)
        hessian11 = hessian * self.s * self.s
        hessian12 = -hessian * self.s * (1-self.t)
        hessian13 = -hessian * self.s * self.t

        hessian20 = -hessian * (1-self.t) * (1-self.s)
        hessian21 = -hessian * (1-self.t) * self.s
        hessian22 = hessian * (1-self.t) * (1-self.t)
        hessian23 = hessian * (1-self.t) * self.t

        hessian30 = -hessian * self.t * (1-self.s)
        hessian31 = -hessian * self.t * self.s
        hessian32 = hessian * self.t * (1-self.t)
        hessian33 = hessian * self.t * self.t

        hessian0 = torch.stack([hessian00, hessian10, hessian20, hessian30], dim=0)
        hessian1 = torch.stack([hessian01, hessian11, hessian21, hessian31], dim=0)
        hessian2 = torch.stack([hessian02, hessian12, hessian22, hessian32], dim=0)
        hessian3 = torch.stack([hessian03, hessian13, hessian23, hessian33], dim=0)

        return torch.stack([hessian0, hessian1, hessian2, hessian3], dim=0)


# 测试点
test_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.01, 0.01, 0.01], [0.01, 0.01, 1.01]], device='cpu')

# 使用
energy = ContactEnergy(stiffness=1.0e3, collision_radius=0.2, points=test_point)
print('\nceshi:', energy(test_point),'\n')

# ee验证
ee_validator = HessianEEValidator(tolerance = 1e-8, device='cpu')
result = ee_validator.validate(test_point, energy)


print('error_fd_type:', type(result['finite_diff_hessian']))
error_fd = result['finite_diff_hessian'] - result['analytical_hessian']
norm_fd = torch.max(torch.abs(error_fd))
print('\nnorm_fd:', norm_fd, '\n')

print('\nresult:', type(result))
