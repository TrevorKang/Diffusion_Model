import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    step = timesteps + 1
    x = torch.linspace(0, timesteps, step)
    f_t = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_t_bar = f_t / f_t[0]
    beta_t = 1 - (alpha_t_bar[1:] / alpha_t_bar[:-1])
    return torch.clip(beta_t, 0.0001, 0.9999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    beta_start = 1e-4
    beta_end = 0.02
    s_limit = 6
    x = torch.linspace(0, timesteps, timesteps+1)
    sigmoid = -s_limit + 2 * x * s_limit / timesteps
    beta_t = beta_start + torch.sigmoid(sigmoid) * (beta_end - beta_start)
    return torch.clip(beta_t, 0.0001, 0.9999)


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps).to(self.device)
        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas
        # TODO
        self.alphas = torch.tensor(1 - self.betas, device=device)
        self.alpha_cumsum = torch.cumprod(self.alphas, dim=0) # FOR EQUATION 5 - Product across all alpha
        # pre calculations 
        self.alpha_root = torch.sqrt(self.alphas)
        self.alpha_sqrt_cumsum = torch.sqrt(1-self.alpha_cumsum)
        self.beta_root = torch.sqrt(self.betas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        
        self.q_x_t = torch.zeros((img_size,img_size),device=device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.q_x_t_1 = torch.zeros((img_size,img_size),device=device) 

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO: 
        predicted_noise = model(x, t)
        alpha = self.alphas[t][:, None, None, None]
        alpha_hat = self.alpha_cumsum[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]
        if t_index > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        self.q_x_t_1 = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return self.q_x_t_1

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        x_zero = torch.randn(batch_size, channels, self.img_size, self.img_size, device=self.device)  # random gaussian noise
        # t_iter = torch.arange(0, self.timesteps, device=self.device).long()
        model.eval()
        for t_index in tqdm(reversed(range(1, self.timesteps)), position=0):
            t_index_tensor = torch.tensor(t_index, device=self.device)
            t = torch.full((batch_size,), t_index_tensor, device=self.device).long()
            x_zero = self.p_sample(model, x_zero, t, t_index_tensor)
      
        # TODO (2.2): Return the generated images
        return x_zero

    # forward diffusion (using the nice property)    
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn(self.img_size, self.img_size, device=self.device)  # random gaussian noise
        x_zero = x_zero.to(self.device)
        t = t.to(self.device)
        alpha_root = self.alpha_root[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Add singleton dimensions
        alpha_sqrt_cumsum = self.alpha_sqrt_cumsum[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Add singleton dimensions
        self.q_x_t = alpha_root * x_zero + alpha_sqrt_cumsum * noise
        return self.q_x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        noise_image = self.q_sample(x_zero, t, noise)
        denoised = denoise_model(noise_image, t)
        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = torch.sum(torch.abs(denoised - noise_image))
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = torch.sqrt(torch.sum(torch.square(denoised - noise_image)))
        else:
            raise NotImplementedError()

        return loss
