import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def show_image_grid(imgs):
    grd = make_grid(imgs)
    npimg = grd.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def cifar2rgb(img_tensor):
    """
    Get numpy img, transpose tensor rep in, shuffle channels, and return
    :param img: Tensor Image from CIFAR10
    :return: None
    """
    img_tensor_denormalized = img_tensor / 2 + 0.5  # denormalize
    npimg_gbr = img_tensor_denormalized.numpy() # tensor to Numpy
    npimp_rgb = np.transpose(npimg_gbr, (1, 2, 0)) # 3, 32*32 to 32*32, 3
    return npimp_rgb

def rgbshow(img):
    """
    Get tensor image, convert to RGB and display
    :param img: Tensor Image from CIFAR10
    :return: None
    """
    rgb = cifar2rgb(img)
    plt.imshow(rgb)
    plt.show()

def imadd(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def imshow(img):
    imadd(img)
    plt.show()

def select_n_random(data, labels, n=100):
    """
    Selects n random datapoints and their corresponding labels from a 3D dataset
    """
    assert len(data) == len(labels)

    # TODO: sort this out for 3D data
    # p1 = torch.randperm(len(data))
    # sample_labels = labels[p1][:n]
    # sample_data = data[p1][:n]
    return data[:n], labels[:n]

def samples_path():
    dir = os.path.dirname(os.path.abspath(__file__))
    samples = os.path.join(dir, 'samples')
    return samples

def three_sample_images():
    samples = samples_path()
    _truck = np.array(Image.open(os.path.join(samples, "truck.png")))
    _deer = np.array(Image.open(os.path.join(samples, "deer.png")))
    _frog = np.array(Image.open(os.path.join(samples, "frog.png")))
    truck = transforms.ToTensor()(_truck)
    deer = transforms.ToTensor()(_deer)
    frog = transforms.ToTensor()(_frog)
    return torch.stack([truck, deer, frog])

def three_different_np_images():
    rgb1 = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb1[..., 0] = 192
    rgb1[..., 1] = 0
    rgb1[..., 2] = 0
    # img1 = Image.fromarray(rgb1)

    rgb2 = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb2[..., 0] = 0
    rgb2[..., 1] = 192
    rgb2[..., 2] = 0
    # img2 = Image.fromarray(rgb2)

    rgb3 = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb3[..., 0] = 0
    rgb3[..., 1] = 0
    rgb3[..., 2] = 192
    # img3 = Image.fromarray(rgb3)

    return (rgb1, rgb2, rgb3)


def three_different_tensor_images():
    rgb1, rgb2, rgb3 = three_different_np_images()

    timg1 = torch.tensor(rgb1).reshape(32*32, 3).transpose(1,0).reshape(3,32,32)
    timg2 = torch.tensor(rgb2).reshape(32*32, 3).transpose(1,0).reshape(3,32,32)
    timg3 = torch.tensor(rgb3).reshape(32*32, 3).transpose(1,0).reshape(3,32,32)
    return (timg1, timg2, timg3)
