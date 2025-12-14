import warp as wp

wp.init()

if wp.is_cuda_available():
    device = "cuda"
else:
    device = "cpu"

device = wp.get_device(device)

print('device:', device)