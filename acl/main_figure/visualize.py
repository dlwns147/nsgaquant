from matplotlib import pyplot as plt
import numpy as np


# Generate random walk data
n_points = 100
data = np.cumsum(np.random.normal(0, 1, n_points))
data_1 = np.cumsum(np.random.normal(0, 1, n_points))

# Create the plot
plt.figure(figsize=(10, 6))
data[10] = 100
data[26] = 60
plt.plot(data, c = 'red', linewidth = 5, linestyle = '--')
plt.plot(data_1, c = 'green', linewidth = 5, linestyle = '-.')
plt.plot(data_1[::-1], c = 'blue', linewidth = 5, linestyle = ':')
# plt.show()
plt.savefig('random_walk.png')