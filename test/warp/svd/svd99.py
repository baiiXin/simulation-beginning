import warp as wp
import numpy as np
import time

wp.init()

# ==============================================================================
# 1. 辅助函数 (9x9)
# ==============================================================================

@wp.func
def idx_9(r: int, c: int):
    return r * 9 + c

@wp.func
def load_block_3x3_9(
    row_block: int, col_block: int, 
    base_offset: int, 
    temp_buffer: wp.array(dtype=float), 
    m: wp.mat33
):
    """ 将 mat33 写入 flatten array """
    row_start = row_block * 3
    col_start = col_block * 3
    
    # Row 0
    temp_buffer[base_offset + idx_9(row_start+0, col_start+0)] = m[0, 0]
    temp_buffer[base_offset + idx_9(row_start+0, col_start+1)] = m[0, 1]
    temp_buffer[base_offset + idx_9(row_start+0, col_start+2)] = m[0, 2]
    # Row 1
    temp_buffer[base_offset + idx_9(row_start+1, col_start+0)] = m[1, 0]
    temp_buffer[base_offset + idx_9(row_start+1, col_start+1)] = m[1, 1]
    temp_buffer[base_offset + idx_9(row_start+1, col_start+2)] = m[1, 2]
    # Row 2
    temp_buffer[base_offset + idx_9(row_start+2, col_start+0)] = m[2, 0]
    temp_buffer[base_offset + idx_9(row_start+2, col_start+1)] = m[2, 1]
    temp_buffer[base_offset + idx_9(row_start+2, col_start+2)] = m[2, 2]

@wp.func
def reconstruct_block_3x3_9(
    row_block: int, col_block: int, 
    base_offset: int, offset_V: int, 
    temp_buffer: wp.array(dtype=float)
):
    res = wp.mat33(0.0)
    row_start = row_block * 3
    col_start = col_block * 3
    
    for i in range(3):
        for j in range(3):
            global_r = row_start + i
            global_c = col_start + j
            
            sum_val = float(0.0) # Explicit float
            
            # While loop to prevent unrolling
            k = int(0)
            while k < 9:
                lam = temp_buffer[base_offset + idx_9(k, k)]
                v_rk = temp_buffer[offset_V + idx_9(global_r, k)]
                v_ck = temp_buffer[offset_V + idx_9(global_c, k)]
                sum_val += v_rk * lam * v_ck
                k += 1
            
            res[i, j] = sum_val
    return res

# ==============================================================================
# 2. 核心设备函数 (9x9 Optimized)
# ==============================================================================

@wp.func
def filter_hessian_9x9_device(
    h_aa: wp.mat33, h_ab: wp.mat33, h_ac: wp.mat33,
    h_ba: wp.mat33, h_bb: wp.mat33, h_bc: wp.mat33,
    h_ca: wp.mat33, h_cb: wp.mat33, h_cc: wp.mat33,
    temp_buffer: wp.array(dtype=float),
    tid: int
):
    # 9x9 needs 81(Mat) + 81(Vec) = 162 floats
    base_offset = tid * 162
    offset_V = base_offset + 81
    
    # --- 1. Load Data ---
    load_block_3x3_9(0, 0, base_offset, temp_buffer, h_aa)
    load_block_3x3_9(0, 1, base_offset, temp_buffer, h_ab)
    load_block_3x3_9(0, 2, base_offset, temp_buffer, h_ac)
    
    load_block_3x3_9(1, 0, base_offset, temp_buffer, h_ba)
    load_block_3x3_9(1, 1, base_offset, temp_buffer, h_bb)
    load_block_3x3_9(1, 2, base_offset, temp_buffer, h_bc)
    
    load_block_3x3_9(2, 0, base_offset, temp_buffer, h_ca)
    load_block_3x3_9(2, 1, base_offset, temp_buffer, h_cb)
    load_block_3x3_9(2, 2, base_offset, temp_buffer, h_cc)

    # --- 2. Jacobi SVD ---
    
    # Init V
    i = int(0)
    while i < 9:
        j = int(0)
        while j < 9:
            temp_buffer[offset_V + idx_9(i, j)] = 1.0 if i == j else 0.0
            j += 1
        i += 1

    # 15 sweeps
    iter_count = int(0)
    while iter_count < 15:
        p = int(0)
        while p < 8:
            q = p + 1
            while q < 9:
                idx_pq = base_offset + idx_9(p, q)
                a_pq = temp_buffer[idx_pq]
                
                if wp.abs(a_pq) >= 1e-6:
                    idx_pp = base_offset + idx_9(p, p)
                    idx_qq = base_offset + idx_9(q, q)
                    a_pp = temp_buffer[idx_pp]
                    a_qq = temp_buffer[idx_qq]

                    tau = (a_qq - a_pp) / (2.0 * a_pq)
                    t = float(0.0)
                    if tau >= 0.0: t = 1.0 / (tau + wp.sqrt(1.0 + tau*tau))
                    else:          t = -1.0 / (-tau + wp.sqrt(1.0 + tau*tau))
                    c = 1.0 / wp.sqrt(1.0 + t*t)
                    s = t * c

                    # Rotate A
                    temp_buffer[idx_pp] = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq
                    temp_buffer[idx_qq] = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq
                    temp_buffer[idx_pq] = 0.0
                    temp_buffer[base_offset + idx_9(q, p)] = 0.0

                    # Rotate Rows/Cols
                    k = int(0)
                    while k < 9:
                        if k != p and k != q:
                            idx_ip = base_offset + idx_9(k, p)
                            idx_iq = base_offset + idx_9(k, q)
                            a_ip = temp_buffer[idx_ip]
                            a_iq = temp_buffer[idx_iq]
                            
                            a_ip_n = c * a_ip - s * a_iq
                            a_iq_n = s * a_ip + c * a_iq
                            
                            temp_buffer[idx_ip] = a_ip_n
                            temp_buffer[base_offset + idx_9(p, k)] = a_ip_n
                            temp_buffer[idx_iq] = a_iq_n
                            temp_buffer[base_offset + idx_9(q, k)] = a_iq_n
                        k += 1
                    
                    # Rotate V
                    k = int(0)
                    while k < 9:
                        idx_ip = offset_V + idx_9(k, p)
                        idx_iq = offset_V + idx_9(k, q)
                        v_ip = temp_buffer[idx_ip]
                        v_iq = temp_buffer[idx_iq]
                        
                        temp_buffer[idx_ip] = c * v_ip - s * v_iq
                        temp_buffer[idx_iq] = s * v_ip + c * v_iq
                        k += 1
                q += 1
            p += 1
        iter_count += 1

    # --- 3. Filter Eigenvalues ---
    k = int(0)
    while k < 9:
        idx = base_offset + idx_9(k, k)
        val = temp_buffer[idx]
        temp_buffer[idx] = wp.max(val, 0.0)
        k += 1

    # --- 4. Reconstruct ---
    out_aa = reconstruct_block_3x3_9(0, 0, base_offset, offset_V, temp_buffer)
    out_ab = reconstruct_block_3x3_9(0, 1, base_offset, offset_V, temp_buffer)
    out_ac = reconstruct_block_3x3_9(0, 2, base_offset, offset_V, temp_buffer)
    
    out_ba = reconstruct_block_3x3_9(1, 0, base_offset, offset_V, temp_buffer)
    out_bb = reconstruct_block_3x3_9(1, 1, base_offset, offset_V, temp_buffer)
    out_bc = reconstruct_block_3x3_9(1, 2, base_offset, offset_V, temp_buffer)
    
    out_ca = reconstruct_block_3x3_9(2, 0, base_offset, offset_V, temp_buffer)
    out_cb = reconstruct_block_3x3_9(2, 1, base_offset, offset_V, temp_buffer)
    out_cc = reconstruct_block_3x3_9(2, 2, base_offset, offset_V, temp_buffer)

    return out_aa, out_ab, out_ac, \
           out_ba, out_bb, out_bc, \
           out_ca, out_cb, out_cc

# ==============================================================================
# 3. Kernel Wrapper
# ==============================================================================

@wp.kernel
def test_filter_kernel_9x9(
    in_aa: wp.array(dtype=wp.mat33), in_ab: wp.array(dtype=wp.mat33), in_ac: wp.array(dtype=wp.mat33),
    in_ba: wp.array(dtype=wp.mat33), in_bb: wp.array(dtype=wp.mat33), in_bc: wp.array(dtype=wp.mat33),
    in_ca: wp.array(dtype=wp.mat33), in_cb: wp.array(dtype=wp.mat33), in_cc: wp.array(dtype=wp.mat33),
    
    out_aa: wp.array(dtype=wp.mat33), out_ab: wp.array(dtype=wp.mat33), out_ac: wp.array(dtype=wp.mat33),
    out_ba: wp.array(dtype=wp.mat33), out_bb: wp.array(dtype=wp.mat33), out_bc: wp.array(dtype=wp.mat33),
    out_ca: wp.array(dtype=wp.mat33), out_cb: wp.array(dtype=wp.mat33), out_cc: wp.array(dtype=wp.mat33),
    
    temp_mem: wp.array(dtype=float)
):
    tid = wp.tid()
    
    res_aa, res_ab, res_ac, \
    res_ba, res_bb, res_bc, \
    res_ca, res_cb, res_cc = filter_hessian_9x9_device(
        in_aa[tid], in_ab[tid], in_ac[tid],
        in_ba[tid], in_bb[tid], in_bc[tid],
        in_ca[tid], in_cb[tid], in_cc[tid],
        temp_mem, tid
    )
    
    out_aa[tid] = res_aa; out_ab[tid] = res_ab; out_ac[tid] = res_ac
    out_ba[tid] = res_ba; out_bb[tid] = res_bb; out_bc[tid] = res_bc
    out_ca[tid] = res_ca; out_cb[tid] = res_cb; out_cc[tid] = res_cc

# ==============================================================================
# 4. Main Program
# ==============================================================================

def solve_numpy_reference(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clamped = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

if __name__ == "__main__":
    count = 1
    device = "cuda:0"

    print(f"--- 1. Constructing a BAD (Indefinite) 9x9 Hessian ---")
    np.random.seed(42)
    H_bad = np.random.randn(9, 9)
    H_bad = H_bad + H_bad.T
    H_bad[0, 0] = -50.0 
    H_bad[4, 4] = -20.0
    H_bad[8, 8] = 100.0

    print("Original Matrix (Top Left 3x3):\n", H_bad[:3, :3])
    orig_vals, _ = np.linalg.eigh(H_bad)
    print(f"Original Eigenvalues: {np.round(orig_vals, 2)}")

    inputs_cpu = [np.zeros((count, 3, 3), dtype=np.float32) for _ in range(9)]
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            inputs_cpu[idx][0] = H_bad[r*3:(r+1)*3, c*3:(c+1)*3]

    in_arrays = [wp.array(arr, dtype=wp.mat33, device=device) for arr in inputs_cpu]
    out_arrays = [wp.zeros(count, dtype=wp.mat33, device=device) for _ in range(9)]
    temp_mem = wp.zeros(count * 162, dtype=float, device=device)

    print(f"\n--- 2. Running Warp Kernel (9x9) ---")
    print("Warmup (Compiling)...")
    wp.launch(
        kernel=test_filter_kernel_9x9,
        dim=count,
        inputs=[*in_arrays, *out_arrays, temp_mem], 
        device=device
    )
    wp.synchronize()
    print("Compilation Done.")

    t0 = time.time()
    for _ in range(100): 
        wp.launch(
            kernel=test_filter_kernel_9x9,
            dim=count,
            inputs=[*in_arrays, *out_arrays, temp_mem],
            device=device
        )
    wp.synchronize()
    print(f"Average Execution Time: {(time.time() - t0)/100 * 1000:.4f} ms")

    # Verify
    res_warp_blocks = [arr.numpy()[0] for arr in out_arrays]
    H_warp = np.zeros((9, 9))
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            H_warp[r*3:(r+1)*3, c*3:(c+1)*3] = res_warp_blocks[idx]

    H_numpy = solve_numpy_reference(H_bad)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS (9x9)")
    print("="*60)

    warp_vals, _ = np.linalg.eigh(H_warp)
    numpy_vals, _ = np.linalg.eigh(H_numpy)

    print(f"Warp Filtered Eigenvalues:  {np.round(warp_vals, 3)}")
    print(f"NumPy Filtered Eigenvalues: {np.round(numpy_vals, 3)}")
    
    diff = np.abs(H_warp - H_numpy).max()
    print(f"\nMax Element Difference: {diff:.6e}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: Warp implementation matches NumPy ground truth!")
    else:
        print("❌ WARNING: Significant difference detected.")