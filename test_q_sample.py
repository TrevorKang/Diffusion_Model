from ex02_diffusion_origin import Diffusion, linear_beta_schedule
from extractCIFAR10 import pick_up_single_image, visualize_image
import torch
import matplotlib.pyplot as plt
import numpy as np

scheduler = lambda x: linear_beta_schedule(0.001, 0.02, x)
diffusor = Diffusion(timesteps=100, get_noise_schedule=scheduler, img_size=32)

# pick up an image from the cifar train set
image = pick_up_single_image(index=5)
# visualize the image
# visualize_image(image)

# convert the image to a torch tensor
x_0 = torch.tensor(image, dtype=torch.float32).cuda()
# reshape the image to the correct shape
x_0 = x_0.permute(2, 0, 1).unsqueeze(0)
# normalize the image
x_0 = x_0 / 255.0
# run the diffusion model
noise_step = [5, 10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90]
outputs = []
for t in noise_step:
    t = torch.tensor([t]).cuda()
    output = diffusor.q_sample(x_zero=x_0, t=t, noise=None)
    output = output.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    outputs.append(output)

print(len(outputs))

for i in range(len(outputs)):
    plt.subplot(3, 4, i+1)
    plt.axis('off')
    plt.imshow(outputs[i])
plt.show()
