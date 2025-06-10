import warp as wp
import numpy as np
from warp.optim.linear import cg

# 初始化 Warp（自动选择设备）
wp.init()
device = wp.get_preferred_device()
print("Device:", device)

# 数据类型
dtype = wp.float32

# 稀疏矩阵参数
num_rows = 10
num_cols = 10
block_size = 1  # scalar block

# 定义稀疏矩阵的三元组数据
row_indices = wp.array([0, 0, 1], dtype=wp.int32, device=device)
col_indices = wp.array([0, 1, 0], dtype=wp.int32, device=device)
data = wp.array([1.0, 2.0, 3.0], dtype=dtype, device=device)

# 构造稀疏矩阵 A
A = wp.sparse.bsr_matrix(
    rows=num_rows,
    cols=num_cols,
    dtype=dtype,
    block_size=block_size,
    row_inds=row_indices,
    col_inds=col_indices,
    values=data
)

# 定义右端项 b 和初始猜测 x
b = wp.array([1.0, 2.0, 3.0] + [0.0]*7, dtype=dtype, device=device)
x = wp.zeros_like(b)

# 调用共轭梯度求解器
cg(A=A, b=b, x=x, tol=1e-6, max_iters=100)

# 计算残差：r = Ax - b
Ax = wp.zeros_like(b)
wp.sparse.mv(A, x, Ax)
residual = Ax - b
residual_norm = wp.norm(residual)
print("Residual norm:", residual_norm)
