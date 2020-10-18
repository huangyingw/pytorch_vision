# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random


# %matplotlib inline
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


import scipy.misc

ascent = scipy.misc.ascent()
plt.gray()
plt.imshow(ascent, interpolation='nearest')
cropped_ascent = ascent[:100, 300:]
plt.imshow(cropped_ascent, interpolation='nearest')
print(cropped_ascent.shape)
print(cropped_ascent[90, 90])
print(cropped_ascent.dtype)

img = torch.from_numpy(cropped_ascent.astype(float))
print(img.size())
print(img[90, 90])
img = img.clone().view(1, 100, 212)
print(img[:, 90, 90])
print(img.size())
img = torch.cat((img, img, img), 0).float()
show(img)
print(img[:, 90, 90])
img.div_(255)
print(img.size())

show(transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])(img))

img2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(256),
    transforms.ToTensor(),
])(img)
print(img2.size())
show(img2)

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
cifar = dset.CIFAR10(root="abc/def/ghi", download=True)

trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import torchvision.utils as tutils

transformed_images = []
for i in range(20):
    transformed_images += [trans(cifar[i][0])]
    print(transformed_images[i].mean(), transformed_images[i].std(),
          transformed_images[i].min(), transformed_images[i].max())
show(tutils.make_grid(transformed_images))

for i in range(20):
    transformed_images[i] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transformed_images[i])
    print(transformed_images[i].mean(), transformed_images[i].std(),
          transformed_images[i].min(), transformed_images[i].max())
show(tutils.make_grid(transformed_images))


# Random Affine transform

from PIL import Image

img = scipy.misc.ascent()
pil_img = Image.fromarray(img.astype(np.uint8))
pil_img

transformed_images = [None] * 5
to_tensor = transforms.ToTensor()
for i in range(5):
    t = transforms.RandomAffine(degrees=(-45, 45), fillcolor=128)
    transformed_images[i] = to_tensor(t(pil_img))
plt.figure(figsize=(16, 16))
show(tutils.make_grid(transformed_images))

transformed_images = [None] * 5
to_tensor = transforms.ToTensor()
for i in range(5):
    t = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=255)
    transformed_images[i] = to_tensor(t(pil_img))
plt.figure(figsize=(16, 16))
show(tutils.make_grid(transformed_images))

transformed_images = [None] * 5
to_tensor = transforms.ToTensor()
for i in range(5):
    t = transforms.RandomAffine(degrees=0, scale=(0.5, 1.5), fillcolor=255)
    transformed_images[i] = to_tensor(t(pil_img))
plt.figure(figsize=(16, 16))
show(tutils.make_grid(transformed_images))

transformed_images = [None] * 5
to_tensor = transforms.ToTensor()
for i in range(5):
    t = transforms.RandomAffine(degrees=0, shear=10, fillcolor=255)
    transformed_images[i] = to_tensor(t(pil_img))
plt.figure(figsize=(16, 16))
show(tutils.make_grid(transformed_images))

transformed_images = [None] * 5
to_tensor = transforms.ToTensor()
for i in range(5):
    t = transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.7, 1.2), shear=10, fillcolor=255)
    transformed_images[i] = to_tensor(t(pil_img))
plt.figure(figsize=(16, 16))
show(tutils.make_grid(transformed_images))
