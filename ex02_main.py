import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, reverse_transform):
    # TODO: Implement - adapt code and method signature as needed
    noise = diffusor.sample(model)
    # save images in store_path
    for i in range(n_images):
        image = reverse_transform(noise[i].cpu())
        image.save(os.path.join(store_path, f"test_{i}.png"))


def val(model, valloader, diffusor, device, epoch, args):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for metrics, e.g., total_loss, accuracy, etc.
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        # Iterate over the test dataset
        for step, (images, labels) in enumerate(valloader):
            images = images.to(device)

            # Sample timesteps for each image
            timesteps = torch.randint(0, diffusor.timesteps, (len(images),), device=device).long()

            # Compute the loss (and any other metrics)
            # Note: You need to define a suitable loss function for your task
            loss = diffusor.p_losses(model, images, timesteps, loss_type="l2")

            # Accumulate the metrics
            total_loss += loss.item()
            num_samples += len(images)

        # Calculate average loss
        average_loss = total_loss / num_samples

        # Print or log the metrics
        print(f'Val Epoch: {epoch}, Average Loss: {average_loss}')

    # Set the model back to training mode
    model.train()


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        # loss = diffusor.p_losses(model, images, t, loss_type="l2")
        loss = diffusor.p_losses(model, images, t, loss_type="l2")
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test(args):
    # TODO (2.2): implement testing functionality, including generation of stored images.
    pass


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10("./dataset/trainset", download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10("./dataset/testset", download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        val(model, valloader, diffusor, device, epoch, args)

    # test(model, testloader, diffusor, device, args)

    save_path = "./results"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path, reverse_transform)
    torch.save(model.state_dict(), os.path.join("./models", f"model_{epoch}_ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)