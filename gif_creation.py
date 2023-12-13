from pathlib import Path
import imageio, os


def create_gif(image_folder="D:\\adl_ws23\\Diffusion_Model\\results\\scheduler\\linear_scheduler", gif_name="my_gif.gif"):
    images = []
    filenames = os.listdir(image_folder)
    for filename in filenames:
        t = imageio.v2.imread(os.path.join(image_folder, filename))
        images.append(t)
    imageio.mimsave(gif_name, images, duration=0.1)


if __name__ == '__main__':
    create_gif()