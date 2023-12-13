import torch
import numpy as np
from ex02_main import visualize_samples
from ex02_diffusion_origin import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from ex02_model import Unet
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
device = "cuda" if torch.cuda.is_available() else "cpu"
my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
timesteps = 1000
image_size = 32
diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
save_path_schedule = "./results/scheduler/linear_scheduler"
# save_path_schedule = "./results/schedule/cosine_scheduler"
# save_path_schedule = "./results/schedule/sigmoid_scheduler"
model = torch.load("./models/model_5_ckpt.pt")
visualize_samples(diffusor, model, device, save_path_schedule, reverse_transform)