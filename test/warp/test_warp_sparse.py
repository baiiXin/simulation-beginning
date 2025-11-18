import warp as wp
from warp.sparse import bsr_zeros

wp.init()

# ------------------------
# 系统维度
# ------------------------
N = 1000  # 大规模矩阵，N 个自由度块
# 对应总自由度 = N * block_size

# ------------------------
# 1. 3x3 块的 BSR 矩阵
# ------------------------
block_size_3x3 = 3
A_3x3 = bsr_zeros(
    rows_of_blocks=N,
    cols_of_blocks=N,
    block_type=wp.mat33  # 3x3 块
)
print("\n3x3 BSR matrix:")
print("Shape of blocks:", A_3x3.block_shape)
print("Matrix shape:", A_3x3.shape)


print('\n')


from warp.sparse import bsr_from_triplets

# --------------------------
# 定义 COO triplets
# --------------------------
# 假设我们要构建一个 4x4 的矩阵，由 2x2 块组成
# 总共有 2x2 = 4 个块
rows = wp.array([0, 0, 1, 1], dtype=int)  # 块的行索引
cols = wp.array([0, 1, 0, 1], dtype=int)  # 块的列索引

# 每个块是 2x2 矩阵
vals = wp.array([
    [[1.0, 0.5],
     [0.5, 2.0]],   # 块 (0,0)
    [[0.0, 1.0],
     [1.0, 0.0]],   # 块 (0,1)
    [[2.0, 0.0],
     [0.0, 2.0]],   # 块 (1,0)
    [[1.0, -1.0],
     [-1.0, 1.0]]   # 块 (1,1)
], dtype=wp.float32)

# --------------------------
# 创建 BSR 矩阵
# --------------------------
A = bsr_from_triplets(
    rows_of_blocks=2,      # 行块数
    cols_of_blocks=2,      # 列块数
    rows=rows,             # 块行索引
    columns=cols,          # 块列索引
    values=vals            # 块数据
)

# --------------------------
# 查看矩阵信息
# --------------------------
print("BSR matrix shape (blocks):", A.nrow, "x", A.ncol)
print("Block shape:", A.block_shape)
print("Values array:\n", A.values)

# 可以获取压缩行索引和列索引
nnz = A.nnz_sync()
print("Exact number of non-zero blocks:", nnz)
print("Row indices of blocks:", A.uncompress_rows()[:nnz])
print("Column indices of blocks:", A.columns[:nnz])
print(A)

print('\n')

# --------------------------
# 定义 COO triplet，只一个非零块
# --------------------------
rows = wp.array([0], dtype=int)  # 块的行索引
cols = wp.array([0], dtype=int)  # 块的列索引

# 块大小 2x2
vals = wp.array([
    [[1.0, 0.0],
     [0.0, 1.0]]   # 块 (0,0)
], dtype=wp.float32)

# --------------------------
# 创建 BSR 矩阵
# --------------------------
A = bsr_from_triplets(
    rows_of_blocks=2,  # 2x2 块矩阵 → 总矩阵 4x4
    cols_of_blocks=2,
    rows=rows,
    columns=cols,
    values=vals
)

# --------------------------
# 查看矩阵信息
# --------------------------
nnz = A.nnz_sync()
print("Exact number of non-zero blocks:", nnz)
print("Row indices of blocks:", A.uncompress_rows()[:nnz])
print("Column indices of blocks:", A.columns[:nnz])
print("Values array:\n", A.values)
print(A)



print('\n')

# --------------------------
# COO triplets，包含重复元素
# --------------------------
rows = wp.array([0, 0, 1], dtype=int)   # 行索引
cols = wp.array([1, 1, 2], dtype=int)   # 列索引
vals = wp.array([2.0, 3.0, 4.0], dtype=float)  # 对应值

# --------------------------
# 创建 BSR 矩阵，1x1 块（CSR）
# --------------------------
A = bsr_from_triplets(
    rows_of_blocks=2,  # 行块数 → 总矩阵 2x2 块，每块 1x1 → 总矩阵 2x2
    cols_of_blocks=3,  # 列块数 → 总矩阵 2x3
    rows=rows,
    columns=cols,
    values=vals
)

print(A)