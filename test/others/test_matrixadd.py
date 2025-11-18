import numpy as np
import warp as wp
import time


#wp.init()  # 初始化 Warp（会选择 CPU 或 CUDA 设备，取决于环境与安装）

N = 8000  # 可以改成任意正整数
# 用 numpy 生成随机矩阵，再转换为 warp 数组
a_np = np.random.rand(N, N).astype(np.float32)
b_np = np.random.rand(N, N).astype(np.float32)

# 转换为 warp 数组（会在默认设备上分配）
#a = wp.array(a_np)
#b = wp.array(b_np)
#c = wp.array(np.zeros_like(a_np))

#print('warp 加 法 测试')
start_time_np = time.time()
# 直接使用 + 做逐元素相加（warp 支持 numpy 风格的算术运算）
# c = a + b
c_np = a_np + b_np
end_time_np = time.time()
print(f"NumPy 加 法 耗 时： {end_time_np - start_time_np:.10f} 秒, {c_np.shape}")