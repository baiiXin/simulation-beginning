import torch
import warp as wp

hessian = torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]], )

print(hessian[1, :])

def multivariable_hessian_example():
    """多变量函数的Hessian矩阵计算"""
    print("\n=== 多变量函数Hessian示例 ===")
    
    # 创建2D点
    point = torch.tensor([1.0, 2.0], requires_grad=True)
    print(f"输入点 = {point}")
    
    # 定义函数：f(x,y) = x^2 + 2y^2
    def energy_func(p):
        return p[0]**2 + 2 * p[1]**2
    
    # 计算能量
    energy = energy_func(point)
    print(f"能量值 f(x,y) = {energy.item()}")
    
    # 计算梯度
    grad = torch.autograd.grad(energy, point, create_graph=True)[0]
    print(f"梯度 = {grad}")
    
    # 计算Hessian矩阵
    hessian = torch.zeros(2, 2)
    for i in range(2):
        grad_i = grad[i]
        grad2 = torch.autograd.grad(grad_i, point, retain_graph=True)[0]
        hessian[i, :] = grad2
        print(f"第{i}行的二阶导数: {grad2}")
    
    print(f"Hessian矩阵:\n{hessian}")
    # 理论值应该是: [[2, 0], [0, 4]]

# 运行多变量示例
multivariable_hessian_example()


point = torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0]], )
x_a, x_b = point[0], point[1]

print(x_a, x_b, point.shape)

print('torch_dtype:', torch.get_default_dtype())

# 全局设置当前项目使用双精度
torch.set_default_dtype(torch.float64)

# 验证
print("Default dtype:", torch.get_default_dtype())  # 输出: torch.float64


vec = torch.rand((3,3,3), dtype=torch.float64)
print(vec)
print(vec[0])


print('-----')
matrix = torch.zeros((3,3,3,3), dtype=torch.float64)
print(f"matrix = {matrix}, shape = {matrix.shape}, shape[0] = {matrix.shape[0]}")

# input_mesh
vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.5, 0.0), wp.vec3(0.5, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.1), wp.vec3(0.0, 0.5, 0.1), wp.vec3(0.5, 0.0, 0.1),
            wp.vec3(0.0, 0.0, 0.2), wp.vec3(0.0, 0.5, 0.2), wp.vec3(0.5, 0.0, 0.2)]

warp_vertices = wp.array(vertices, dtype=wp.vec3, device="cpu")
torch_vertices = wp.to_torch(warp_vertices, requires_grad=True)

print(f"torch_vertices = {torch_vertices}, shape = {torch_vertices.shape}, shape[0] = {torch_vertices.shape[0]}")


# 梯度切断

x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = x[0]
y_no_grad = y.detach().clone()

# 对 y_no_grad 做操作
z = y_no_grad * 2  # 不会跟踪梯度

# 对原 tensor 做操作
w = y * 3
w.sum().backward()

print()
print(x.grad)  # tensor([3., 0., 0.]) -> 原 tensor 的梯度仍然存在
