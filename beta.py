import numpy as np
import matplotlib.pyplot as plt


# Define beta schedules
def linear_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def cosine_schedule(timesteps, s=0.008):
    step = timesteps + 1
    x = np.linspace(0, timesteps, step)
    f_t = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alpha_t_bar = f_t / f_t[0]
    beta_t = 1 - (alpha_t_bar[1:] / alpha_t_bar[:-1])
    return np.clip(beta_t, 0.0001, 0.9999)


def sigmoid_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    s_limit = 6
    x = np.linspace(0, timesteps, timesteps+1)
    beta_t = beta_start + 1 / (1 + np.exp(-s_limit + 2 * x * s_limit / timesteps)) * (beta_end - beta_start)
    return np.clip(beta_t, 0.0001, 0.9999)

def alpha_cumprod_schedule(beta_schedule):
    alpha_schedule = 1 - beta_schedule
    alpha_cumprod = np.cumprod(alpha_schedule)
    return alpha_cumprod
# Plot the beta schedules
timesteps = 300
# Linear schedule
beta_t_linear = linear_schedule(timesteps)
alpha_t_cumprod_linear = alpha_cumprod_schedule(beta_t_linear)
# Cosine schedule
beta_t_cosine = cosine_schedule(timesteps)
alpha_t_cumprod_cosine = alpha_cumprod_schedule(beta_t_cosine)
# Sigmoid schedule
beta_t_sigmoid = sigmoid_schedule(timesteps)
alpha_t_cumprod_sigmoid = alpha_cumprod_schedule(beta_t_sigmoid)

plt.plot(alpha_t_cumprod_linear, label="Linear")
plt.plot(alpha_t_cumprod_cosine, label="Cosine")
plt.plot(alpha_t_cumprod_sigmoid, label="Sigmoid")
plt.title("Alpha Cumulative Product")
plt.xlabel("Timesteps")
plt.ylabel("alpha_t_cumprod")
plt.legend()
plt.show()
