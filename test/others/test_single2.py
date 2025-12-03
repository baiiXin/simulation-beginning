import warp as wp
import numpy as np
wp.init()

result = np.float32(1e-8) + np.float32(1e-8)
print(result)


# float32 constants (Warp 内部一般是这种值)
a = np.float32(-9.8)          # 加速度
dt = np.float32(0.003)        # 时间步
damping = np.float32(0.99)    # 阻尼

v0 = np.float32(0.0)          # 初始速度

# step 1: integrate acceleration
v = v0 + a * dt

# step 2: apply damping
v = v * damping

print("v =", v)
print("repr:", np.format_float_scientific(v, precision=10))



a = np.float32(-9.8)
dt = np.float32(0.003)
damping = np.float32(0.99)
v0 = np.float32(0.0)

v = v0 + a * dt      # 单独乘法，再加法
v = v * damping      # 阻尼

print("NumPy result:", v)




@wp.kernel
def compute_velocity(a: float, dt: float, damping: float, v0: float, out: wp.array(dtype=float)):
    v = v0 + a * dt     # CUDA 会把这个编译成 fma(a, dt, v0)
    v = v * damping
    out[0] = v
    wp.print(v)

# float32 values (same as numpy)
a = -9.8
dt = 0.003
damping = 0.99
v0 = 0.0

out = wp.zeros(1, dtype=wp.float32)

wp.launch(
    kernel=compute_velocity,
    dim=1,
    inputs=[a, dt, damping, v0, out],
    device="cpu"   # 或 "cpu"，不过 GPU 才有 FMA
)

print("Warp result:", out.numpy()[0])
