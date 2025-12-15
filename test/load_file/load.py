import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

data = np.load("/data/zhoucy/sim/test/load_file/cloth_data_cloth_fall_sphere_unit.npy")
print(data.shape)

xs_all = data[:, :, 0]
ys_all = data[:, :, 1]
zs_all = data[:, :, 2]

x_min, x_max = xs_all.min(), xs_all.max()
y_min, y_max = ys_all.min(), ys_all.max()
z_min, z_max = zs_all.min(), zs_all.max()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

frame0 = data[0]
scat = ax.scatter(frame0[:, 0], frame0[:, 1], frame0[:, 2], s=5)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Cloth positions")


def update(frame_index):
    frame = data[frame_index]
    xs = frame[:, 0]
    ys = frame[:, 1]
    zs = frame[:, 2]
    scat._offsets3d = (xs, ys, zs)
    ax.set_title(f"Cloth positions at time index {frame_index}")
    return scat


anim = animation.FuncAnimation(
    fig, update, frames=data.shape[0], interval=50, blit=False
)

writer = animation.PillowWriter(fps=20)
anim.save("cloth_animation.gif", writer=writer)
