import torch
from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def pick_up_single_image(index):
    f = open('./dataset/trainset/cifar-10-batches-py/data_batch_1', 'rb')
    tupled_data= pickle.load(f, encoding='bytes')
    f.close()
    img = tupled_data[b'data']
    single_img = np.array(img[index])
    single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    return single_img_reshaped


def visualize_image(img):
    plt.imshow(img)
    plt.show()


def pick_up_a_batch(batch_size=16):
    f = open('./dataset/trainset/cifar-10-batches-py/data_batch_2', 'rb')
    tupled_data= pickle.load(f, encoding='bytes')
    f.close()
    img = torch.Tensor(tupled_data[b'data'][:batch_size]).reshape(batch_size, 3, 32, 32)/255
    # reshape into [b, c, h, w]

    return img


def visualize_a_batch(img):
    """

    :param img:torch.tensor, shape = [b, c, h, w]
    :return:
    """
    img = (img.clamp(-1, 1) + 1) / 2
    img = img.permute(0, 2, 3, 1) * 255
    img = img.cpu().numpy().astype(np.uint8)
    for i in range(0, img.shape[0]):
        plt.subplot(4, img.shape[0]//4, i+1)
        plt.imshow(img[i])
    plt.show()


if __name__ == '__main__':
    batch = pick_up_a_batch(16)
    visualize_a_batch(batch)
    # print(batch.shape)
