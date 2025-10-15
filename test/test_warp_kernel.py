import warp as wp

# 初始化 Warp（加载编译环境）
wp.init()

# 定义一个 kernel：对输入数组的每个元素乘以 scale，写入输出数组
@wp.kernel
def scale_and_store(
    x_in: wp.array(dtype=wp.float32),   # 输入数组
    scale: float,                       # 关键信息（缩放因子）
    x_out: wp.array(dtype=wp.float32)   # 输出数组
):
    tid = wp.tid()                      # 当前线程索引
    x_out[tid] = x_in[tid] * scale      # 写入输出结果


# === 创建数据 ===
n = 5
x_in = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=wp.float32, device="cpu")
x_out = wp.zeros(n, dtype=wp.float32, device="cpu")  # 输出数组初始化为 0
scale = 2.5

# === 启动 kernel ===
wp.launch(
    kernel=scale_and_store,
    dim=n,                     # 启动 n 个线程
    inputs=[x_in, scale, x_out],
    device="cpu"
)

# === 拷回 CPU 并打印 ===
print("Input :", x_in.numpy())
print("Output:", x_out.numpy())


import warp as wp
import torch

# 例子：创建 warp 数组（在 GPU 上）
arr_wp = wp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]],
                   dtype=wp.vec3, device="cpu")

# 直接转为 torch tensor（零拷贝）
arr_torch = wp.to_torch(arr_wp)

print(type(arr_torch))     # <class 'torch.Tensor'>
print(arr_torch.device)    # cuda:0
print(arr_torch.shape)     # torch.Size([2, 3])

# 修改其中一个，也会影响另一个（共享内存）
arr_torch[0, 0] = 999.0
print(arr_wp)   # Warp array 里第一项也会变


t = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]],
                   device="cpu", dtype=torch.float32)

a = wp.from_torch(t, dtype=wp.vec3)

print('\nwp.from_torch(t):',a ,type(a) )     # (10,)

print(type(x_in))
