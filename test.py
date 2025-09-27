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
