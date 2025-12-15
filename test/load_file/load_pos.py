import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

data = np.load("/data/zhoucy/sim/test/load_file/cloth_data_cloth_fall_sphere_unit.npy")
print(data.shape)

print(data[50,:,:])