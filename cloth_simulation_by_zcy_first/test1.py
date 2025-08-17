import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cloth_collision_handler import ClothCollisionHandler

# 生成一个平面布料网格
N = 10
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)
x, y = np.meshgrid(x_vals, y_vals)
z = np.zeros_like(x)
verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

faces = []
edges = set()
for i in range(N - 1):
    for j in range(N - 1):
        idx = lambda ii, jj: ii * N + jj
        a = idx(i, j)
        b = idx(i + 1, j)
        c = idx(i + 1, j + 1)
        d = idx(i, j + 1)
        faces.append([a, b, d])
        faces.append([b, c, d])
        edges.update([(a, b), (b, c), (c, d), (d, a), (a, c), (b, d)])

faces = np.array(faces)
edges = np.array(list(edges))

# 初始速度为零，只在中间部分下压一部分节点
vel = np.zeros_like(verts)
for i in range(len(verts)):
    if 0.4 < verts[i, 0] < 0.6 and 0.4 < verts[i, 1] < 0.6:
        vel[i, 2] = -0.1

handler = ClothCollisionHandler(faces, edges, thickness=0.02)

# 时间推进
positions = [verts.copy()]
for _ in range(10):
    verts, vel = handler.step(verts, vel, dt=0.05)
    positions.append(verts.copy())

# 可视化结果
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
for step in [0, 3, 6, 9]:
    mesh = positions[step]
    tris = [[mesh[i] for i in face] for face in faces]
    poly = Poly3DCollection(tris, alpha=0.3, facecolor='cyan')
    ax.add_collection3d(poly)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-0.2, 0.2)
plt.title("Cloth self-collision evolution")
plt.show()