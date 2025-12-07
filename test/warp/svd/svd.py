import warp as wp
import numpy as np

# 初始化 Warp
wp.init()

# ==========================================
#  核心算法实现 (Device Functions)
# ==========================================

# --- 辅助函数：Jacobi 迭代求特征值 ---
@wp.func
def eigen_decomposition_3x3(A: wp.mat33):
    Q = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D = A
    
    # 5次扫描通常足够收敛
    for _ in range(5):
        # 处理 (0,1)
        if wp.abs(D[0, 1]) > 1e-6:
            q = D[0, 0] - D[1, 1]
            p = D[0, 1] * 2.0
            theta = 0.5 * wp.atan2(p, q)
            c = wp.cos(theta)
            s = wp.sin(theta)
            
            d00 = D[0, 0]
            d11 = D[1, 1]
            d01 = D[0, 1]
            d02 = D[0, 2]
            d12 = D[1, 2]
            
            D[0, 0] = c*c*d00 + s*s*d11 + 2.0*c*s*d01
            D[1, 1] = s*s*d00 + c*c*d11 - 2.0*c*s*d01
            D[0, 1] = 0.0
            D[1, 0] = 0.0
            D[0, 2] = c*d02 + s*d12
            D[2, 0] = D[0, 2]
            D[1, 2] = -s*d02 + c*d12
            D[2, 1] = D[1, 2]
            
            for k in range(3):
                q0 = Q[k, 0]
                q1 = Q[k, 1]
                Q[k, 0] = c*q0 + s*q1
                Q[k, 1] = -s*q0 + c*q1

        # 处理 (0,2)
        if wp.abs(D[0, 2]) > 1e-6:
            q = D[0, 0] - D[2, 2]
            p = D[0, 2] * 2.0
            theta = 0.5 * wp.atan2(p, q)
            c = wp.cos(theta)
            s = wp.sin(theta)
            
            d00 = D[0, 0]
            d22 = D[2, 2]
            d02 = D[0, 2]
            d01 = D[0, 1]
            d12 = D[1, 2]
            
            D[0, 0] = c*c*d00 + s*s*d22 + 2.0*c*s*d02
            D[2, 2] = s*s*d00 + c*c*d22 - 2.0*c*s*d02
            D[0, 2] = 0.0
            D[2, 0] = 0.0
            D[0, 1] = c*d01 + s*d12
            D[1, 0] = D[0, 1]
            D[1, 2] = -s*d01 + c*d12
            D[2, 1] = D[1, 2]
            
            for k in range(3):
                q0 = Q[k, 0]
                q2 = Q[k, 2]
                Q[k, 0] = c*q0 + s*q2
                Q[k, 2] = -s*q0 + c*q2

        # 处理 (1,2)
        if wp.abs(D[1, 2]) > 1e-6:
            q = D[1, 1] - D[2, 2]
            p = D[1, 2] * 2.0
            theta = 0.5 * wp.atan2(p, q)
            c = wp.cos(theta)
            s = wp.sin(theta)
            
            d11 = D[1, 1]
            d22 = D[2, 2]
            d12 = D[1, 2]
            d01 = D[0, 1]
            d02 = D[0, 2]
            
            D[1, 1] = c*c*d11 + s*s*d22 + 2.0*c*s*d12
            D[2, 2] = s*s*d11 + c*c*d22 - 2.0*c*s*d12
            D[1, 2] = 0.0
            D[2, 1] = 0.0
            D[0, 1] = c*d01 + s*d02
            D[1, 0] = D[0, 1]
            D[0, 2] = -s*d01 + c*d02
            D[2, 0] = D[0, 2]
            
            for k in range(3):
                q1 = Q[k, 1]
                q2 = Q[k, 2]
                Q[k, 1] = c*q1 + s*q2
                Q[k, 2] = -s*q1 + c*q2

    evals = wp.vec3(D[0, 0], D[1, 1], D[2, 2])
    return evals, Q

# --- 方法 1: Clamp (截断负数) ---
@wp.func
def method1_clamp(H: wp.mat33):
    evals, E = eigen_decomposition_3x3(H)
    
    # 将负数直接变 0
    val_x = wp.max(evals[0], 0.0)
    val_y = wp.max(evals[1], 0.0)
    val_z = wp.max(evals[2], 0.0)
    
    col0 = wp.vec3(E[0,0], E[1,0], E[2,0])
    col1 = wp.vec3(E[0,1], E[1,1], E[2,1])
    col2 = wp.vec3(E[0,2], E[1,2], E[2,2])
    
    return val_x * wp.outer(col0, col0) + \
           val_y * wp.outer(col1, col1) + \
           val_z * wp.outer(col2, col2)

# --- 方法 2: Abs/SVD (取绝对值) ---
@wp.func
def method2_svd_abs(H: wp.mat33):
    U, Sigma, V = wp.svd3(H)
    
    col0 = wp.vec3(U[0,0], U[1,0], U[2,0])
    col1 = wp.vec3(U[0,1], U[1,1], U[2,1])
    col2 = wp.vec3(U[0,2], U[1,2], U[2,2])
    
    # === 修正点：加上 wp.abs() ===
    # Warp 的 svd3 可能会返回负的奇异值(Signed SVD)，必须强转绝对值
    s0 = wp.abs(Sigma[0])
    s1 = wp.abs(Sigma[1])
    s2 = wp.abs(Sigma[2])
    
    return s0 * wp.outer(col0, col0) + \
           s1 * wp.outer(col1, col1) + \
           s2 * wp.outer(col2, col2)

# ==========================================
#  Kernel 定义
# ==========================================

@wp.kernel
def compare_filtering_methods(
    inputs: wp.array(dtype=wp.mat33),
    out_clamp: wp.array(dtype=wp.mat33),
    out_svd: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()
    H = inputs[tid]
    
    # 运行方法 1
    out_clamp[tid] = method1_clamp(H)
    
    # 运行方法 2
    out_svd[tid] = method2_svd_abs(H)

# ==========================================
#  主程序
# ==========================================

# 1. 构造测试数据
# 这是一个经典的 "对角线看似没问题，但其实是不定矩阵" 的例子
# 特征值大概是: 10.4, -5.4, 2.0 (中间有个负数)
H_data = np.array([
    [ 10.0,  2.0,  0.0],
    [  2.0, -5.0,  1.0], 
    [  0.0,  1.0,  2.0]
], dtype=np.float32)

# 封装到 Warp 数组
count = 1
input_gpu = wp.array(np.array([H_data]), dtype=wp.mat33, device="cuda:0")
output_clamp_gpu = wp.zeros(count, dtype=wp.mat33, device="cuda:0")
output_svd_gpu = wp.zeros(count, dtype=wp.mat33, device="cuda:0")

# 2. 运行 Kernel
wp.launch(
    kernel=compare_filtering_methods,
    dim=count,
    inputs=[input_gpu, output_clamp_gpu, output_svd_gpu],
    device="cuda:0"
)

# 3. 获取结果并打印
res_clamp = output_clamp_gpu.numpy()[0]
res_svd = output_svd_gpu.numpy()[0]

print("-" * 50)
print("原始 Hessian (存在负特征值):")
print(H_data)
vals, _ = np.linalg.eigh(H_data)
print(f"原始特征值 (Numpy算出): {vals}")

print("\n" + "-" * 50)
print("方法 1: Clamp (负数变0)")
print("解释: 丢失了负方向的能量，矩阵变得更'软'。")
print(res_clamp)
vals_c, _ = np.linalg.eigh(res_clamp)
print(f"特征值: {vals_c}")

print("\n" + "-" * 50)
print("方法 2: SVD/Abs (负数变正)")
print("解释: 负方向反转为正方向，保持了刚度，矩阵比较'硬'。")
print(res_svd)
vals_s, _ = np.linalg.eigh(res_svd)
print(f"特征值: {vals_s}")
print("-" * 50)