import warp as wp
import numpy as np
from warp.optim.linear import cg

# 初始化 Warp
wp.init()
device = wp.get_preferred_device()
print("Device:", device)

# 数据类型
dtype = wp.float32

# 弹簧质点系统参数
n = 10  # 质点数量
k = 1.0  # 弹簧刚度

# 构造稀疏矩阵 A
row_indices = []
col_indices = []
data = []
for i in range(n):
    row_indices.append(i)
    col_indices.append(i)
    data.append(2.0 * k)
    if i < n - 1:
        row_indices.append(i)
        col_indices.append(i + 1)
        data.append(-k)
        row_indices.append(i + 1)
        col_indices.append(i)
        data.append(-k)

row_indices = wp.array(row_indices, dtype=wp.int32, device=device)
col_indices = wp.array(col_indices, dtype=wp.int32, device=device)
data = wp.array(data, dtype=dtype, device=device)

A = wp.sparse.bsr_matrix(
    rows=n,
    cols=n,
    dtype=dtype,
    block_size=1,
    row_inds=row_indices,
    col_inds=col_indices,
    values=data,
    device=device
)

# 定义右端项和初始解
b = wp.zeros(n, dtype=dtype, device=device)
b[0] = 1.0  # 第一个质点受力
x = wp.zeros_like(b)

# 求解 Ax = b
cg(A=A, b=b, x=x, tol=1e-6, max_iters=100, device=device)

# 计算残差
@wp.kernel
def compute_residual(Ax: wp.array(dtype=dtype), b: wp.array(dtype=dtype), residual: wp.array(dtype=dtype)):
    i = wp.tid()
    residual[i] = Ax[i] - b[i]

Ax = wp.zeros_like(b)
wp.sparse.bsr_mv(A, x, Ax, alpha=1.0, beta=0.0, device=device)
residual = wp.zeros_like(b)
wp.launch(compute_residual, dim=n, inputs=[Ax, b, residual], device=device)
residual_norm = wp.norm(residual)

# 同步并打印结果
wp.synchronize_device()
print("Solution x:", x.numpy())
print("Residual norm:", residual_norm.numpy())