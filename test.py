import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Parameters for the sigmoid beta schedule
beta_start = 0.1
beta_end = 0.9
s_limit = 6
T = 100

# Calculate beta_t for t from 1 to 100 using the sigmoid beta schedule
beta_t_sigmoid = [beta_start + sigmoid(-s_limit + 2 * t / T * s_limit) * (beta_end - beta_start) for t in range(1, 101)]
print(np.array(beta_t_sigmoid).shape)

plt.plot(beta_t_sigmoid)
plt.show()