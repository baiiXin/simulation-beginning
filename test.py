import numpy as np
import warp as wp

pos = np.array([[-2.0, -2.0, 8.0],
                [-1.5, -2.0, 8.0],
                [-1.0, -2.0, 8.0]])

print('\npos:', pos, pos.shape, type(pos))

vetexis = [wp.vec3(pos[i,:]) for i in range(pos.shape[0])]

print('\nvetexis:', vetexis)

pos_warp = wp.array(vetexis, dtype=wp.vec3)

vel_warp = wp.zeros_like(pos_warp)

print('\npos_warp:', pos_warp, pos_warp.shape, type(pos_warp))
print('\nvel_warp:', vel_warp, vel_warp.shape, type(vel_warp))

pos_cur = pos_warp.numpy()
vel_cur = vel_warp.numpy()

print('\npos_cur:', pos_cur, pos_cur.shape, type(pos_cur))
print('\nvel_cur:', vel_cur, vel_cur.shape, type(vel_cur))

pos_cur_array = wp.array(pos_cur, dtype=wp.vec3)
vel_cur_array = wp.array(vel_cur, dtype=wp.vec3)

print('\npos_cur_array:', pos_cur_array, pos_cur_array.shape, type(pos_cur_array), id(pos_cur_array))
print('\nvel_cur_array:', vel_cur_array, vel_cur_array.shape, type(vel_cur_array), id(vel_cur_array))


#print('\npos_cur[0]:', pos_cur[0], pos_cur[0].shape, type(pos_cur[0]))
#pos_cur_flatten = pos_cur.flatten()
#print('\npos_cur_flatten:',pos_cur_flatten, pos_cur_flatten[0:3])


pos_cur_array, vel_cur_array = vel_cur_array, pos_cur_array
print('\npos_cur_array:', pos_cur_array, pos_cur_array.shape, type(pos_cur_array), id(pos_cur_array))
print('\nvel_cur_array:', vel_cur_array, vel_cur_array.shape, type(vel_cur_array), id(vel_cur_array))


arr = np.random.rand(5, 3)   # 举例，一个 5x3 的数组
norms = np.linalg.norm(arr, axis=1)  # 结果是长度为 5 的 1D array
print('\narr:', arr)
print('\nnorms:',norms)



pos_new = wp.clone(pos_warp)
print('\npos_warp:',pos_warp, id(pos_warp))
print('\npos_new:', pos_new, id(pos_new))


print('\n---.fill()&.assign()---')
# 创建两个 Warp 数组
a = wp.array([0, 0, 0, 0], dtype=wp.int32, device="cpu")
b = wp.array([1, 2, 3, 4], dtype=wp.int32, device="cpu")

# fill 把所有元素变成 -1
a.fill_(-1)
print(a)   # [-1 -1 -1 -1]

# assign 把 b 的内容拷贝到 a
a.assign(b)
print(a)   # [1 2 3 4]

print('\n---.full()---')
# 在 CPU 上创建一个长度为 5 的数组，所有元素都是 -1
a = wp.full(5, -1, dtype=wp.int32, device="cpu")
print(a)   # [-1 -1 -1 -1 -1]

# 创建一个 2x3 的浮点数组，值全是 3.14
b = wp.full((2, ), dtype=wp.float32, device="cpu")
print(b)
b = wp.full((2, 3), 3.14, dtype=wp.float32, device="cpu")
print(b)

# 生成器
up_vector = (0, 1, 0)
gravity = 9.81
gen = (g * gravity for g in up_vector)

print(gen)
print(list(gen))


up_vector = (0, 0, 1)
gravity = 9.81

mgravity = wp.array([wp.vec3(*(g * gravity for g in up_vector))], dtype=wp.vec3)
print(mgravity)