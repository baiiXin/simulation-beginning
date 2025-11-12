import numpy as np
from scipy.sparse import bmat, bsr_matrix
from pypardiso import spsolve


def build_block_matrix():
    bsz = 3
    I = np.eye(bsz)
    A00 = 10.0 * I
    A11 = 10.0 * I
    A22 = 10.0 * I
    A01 = 1.0 * I
    A10 = A01.T  # 保持对称
    A12 = 1.0 * I
    A21 = A12.T  # 保持对称

    blocks = [
        [A00, A01, None],
        [A10, A11, A12],
        [None, A21, A22],
    ]
    # 使用 bmat 直接按块构造稀疏矩阵，输出为 CSR 便于求解
    A = bmat(blocks, format="csr")
    return A


def build_bsr_from_block_coo(blocks_data: np.ndarray,
                              row: np.ndarray,
                              col: np.ndarray,
                              nb: int,
                              blocksize: tuple[int, int] = (3, 3),
                              sort_blocks: bool = True) -> bsr_matrix:
    """
    用 3×3×N 的数值块数组 + 块级坐标 (row, col) 构造 BSR 矩阵。

    参数:
      - blocks_data: 形状为 (nnz_blocks, br, bc) 的数值数组，例如 (N, 3, 3)。
      - row, col: 每个块的块行与块列索引，长度为 nnz_blocks，范围在 [0, nb)。
      - nb: 块行/块列的总数（最终矩阵尺寸为 (nb*br, nb*bc)）。
      - blocksize: 每个块的尺寸 (br, bc)，默认 (3, 3)。
      - sort_blocks: 是否按 (row, col) 排序以生成规范的 indptr/indices。

    返回:
      - scipy.sparse.bsr_matrix，形状为 (nb*br, nb*bc)。
    """
    blocks_data = np.asarray(blocks_data, dtype=np.float64)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)

    if blocks_data.ndim != 3:
        raise ValueError("blocks_data 必须是三维数组，形状为 (nnz_blocks, br, bc)")
    nnz_blocks, br, bc = blocks_data.shape
    if (br, bc) != tuple(blocksize):
        raise ValueError(f"块尺寸不匹配: blocks_data 为 {(br, bc)}, 期望 {blocksize}")
    if row.shape != (nnz_blocks,) or col.shape != (nnz_blocks,):
        raise ValueError("row/col 长度必须与块数 nnz_blocks 相同")
    if np.any(row < 0) or np.any(row >= nb) or np.any(col < 0) or np.any(col >= nb):
        raise ValueError("row/col 索引越界: 必须在 [0, nb) 范围内")

    # 规范顺序：按 (row, col) 排序，便于构建 indptr/indices
    if sort_blocks:
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        blocks_data = blocks_data[order]

    # 构造 BSR 压缩格式需要的 indptr/indices
    counts = np.bincount(row, minlength=nb)
    indptr = np.empty(nb + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = col

    A_bsr = bsr_matrix((blocks_data, indices, indptr), shape=(nb * br, nb * bc))
    return A_bsr


def main():
    np.random.seed(0)
    A_csr = build_block_matrix()
    # 用 3x3xN 数组 + 块坐标再构造一次 BSR，并转换为 CSR 以对比
    bsz = 3
    I = np.eye(bsz)
    A00 = 10.0 * I
    A11 = 10.0 * I
    A22 = 10.0 * I
    A01 = 1.0 * I
    A10 = A01.T
    A12 = 1.0 * I
    A21 = A12.T
    blocks_data = np.stack([A00, A01, A10, A11, A12, A21, A22], axis=0)
    row = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    col = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
    A_bsr_from_data = build_bsr_from_block_coo(blocks_data, row, col, nb=3, blocksize=(3, 3))
    A_csr_from_data = A_bsr_from_data.tocsr()
    diff_nnz = (A_csr - A_csr_from_data).nnz
    # 可选：展示 BSR 视图（3x3 blocksize），求解仍使用 CSR
    A_bsr = A_csr.tobsr(blocksize=(3, 3))
    A_csr = A_bsr_from_data.tocsr()
    n = A_csr.shape[0]
    x_true = np.random.randn(n)
    b = A_csr @ x_true
    x = spsolve(A_csr, b)
    resid = np.linalg.norm(A_csr @ x - b)
    err = np.linalg.norm(x - x_true)
    print(f"Matrix shape: {A_csr.shape}, nnz: {A_csr.nnz}")
    print(f"BSR blocksize: {A_bsr.blocksize}, nnz: {A_bsr.nnz}")
    print(f"BSR(3x3xN数据)→CSR 与 bmat→CSR 的差异 nnz: {diff_nnz}")
    print(f"Residual norm ||Ax-b||: {resid:.3e}")
    print(f"Solution error ||x-x_true||: {err:.3e}")


if __name__ == "__main__":
    main()
