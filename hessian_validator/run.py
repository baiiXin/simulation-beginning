import torch
from hessian_point_validator import HessianPointValidator
torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt

# 示例：用户自定义的能量函数类
class ContactEnergy:
    def __init__(self, stiffness=1.0e3, collision_radius=0.2, point=None):
        self.stiffness = stiffness
        self.collision_radius = collision_radius
        x_a, x_b = point[0], point[1]
        self.hat_normal = (x_a - x_b) / torch.norm(x_a - x_b)
    
    def __call__(self, x_a, x_b):
        """能量值计算"""
        return torch.max(torch.tensor(0.0), 0.5 * self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b))**2)

    def analytical_force(self, x_a, x_b):
        """解析力：对于弹簧能量就是力"""
        return -self.stiffness * (self.collision_radius - torch.dot(self.hat_normal, x_a - x_b)) * self.hat_normal
    
    def analytical_hessian(self, x_a, x_b):
        """解析Hessian：对于弹簧能量就是刚度矩阵"""
        return self.stiffness * torch.outer(self.hat_normal,self.hat_normal)

# 测试点
test_point=torch.tensor([[0.01, 0.01, 0.01], [0.0, 0.0, 0.0]], device='cpu')

# 使用
energy = ContactEnergy(point=test_point)
print('\nceshi:', energy(test_point[0], test_point[1]),'\n')

# 点验证
point_validator = HessianPointValidator(tolerance = 1e-3, device='cpu')
result = point_validator.validate(test_point, energy)

# 系统验证
#system_validator = HessianSystemValidator(device='cuda')
#system_validator.set_energy_function(example_energy_function)
#system_result = system_validator.validate(sampling_strategy='random', num_samples=5)


print('\nresult:', type(result))

tolerance = [0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

error_analytical_vs_finite_diff = []
error_analytical_vs_auto_diff = []

for t in tolerance:
    point_validator = HessianPointValidator(tolerance = t, device='cpu')
    result = point_validator.validate(test_point, energy)
    error_analytical_vs_finite_diff.append(result['error_analytical_vs_finite_diff'])
    error_analytical_vs_auto_diff.append(result['error_analytical_vs_auto_diff'])

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制两个数据系列在同一个x轴上
line1 = ax1.plot(tolerance, error_analytical_vs_finite_diff, marker='o', linestyle='-', color='b', label='Error analytical vs. finite diff')
line2 = ax1.plot(tolerance, error_analytical_vs_auto_diff, marker='x', linestyle='--', color='r', label='Error analytical vs. auto diff')

ax1.set_xlabel('Perturbation', fontsize=14)
ax1.set_xscale('log')  # 设置横坐标为对数刻度

ax1.set_ylabel('Error', fontsize=14)
ax1.set_yscale('log')  # 设置纵坐标为对数刻度
ax1.set_ylim(1e-16, 1e0)  # 设置纵坐标范围

# 添加图例
ax1.legend(loc='best')

plt.title('Error analytical vs. finite diff vs. auto diff', fontsize=16)
plt.grid(True, which="major", ls="--", c='0.7', alpha=0.7)  # 只显示主刻度网格线
plt.tight_layout()
plt.show()