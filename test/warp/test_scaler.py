import numpy as np

mass = 0.000083
dt = 0.01
num_particle = 3

# inertia and gravity
A_rows = np.array([i for i in range(num_particle)])
A_cols = np.array([i for i in range(num_particle)])
A_values = np.array([np.eye(3) * mass / dt**2 for _ in range(num_particle)])


print(A_rows)
print(A_cols)
print(A_values[0:2])