#!/usr/bin/env python
# coding: utf-8

# # GAN mix
# 
# Combining ideas from:
# - [dogs-starter-24jul](https://www.kaggle.com/phoenix9032/gan-dogs-starter-24-jul-custom-layers)
# - [Pytorch RaLS-C-SAGAN](https://www.kaggle.com/mpalermo/pytorch-rals-c-sagan)

# In[ ]:


import gzip
import os
import pathlib
import pickle
import random
import shutil
import time
import urllib
import warnings
import xml.etree.ElementTree as ET
import zipfile
from time import time

import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from scipy import linalg
from scipy.stats import truncnorm
from torch import nn as nn
from torch import optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dset
from torchvision import transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

print(os.listdir("../input"))


# In[ ]:


# utilities
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def show_generated_img_all():
    gen_z = torch.randn(32, nz, 1, 1, device=device)
    gen_images = netG(gen_z).to("cpu").clone().detach()
    gen_images = gen_images.numpy().transpose(0, 2, 3, 1)
    gen_images = (gen_images + 1.0) / 2.0
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_images):
        ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
    # plt.savefig(filename)


### This is to show one sample image for iteration of chosing
def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = (gen_image + 1.0) / 2.0
    plt.imshow(gen_image)
    plt.show()


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def show_generated_img(n_images=5, nz=128):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        dog_label = torch.randint(0, len(encoded_dog_labels), (1,), device=device)
        gen_image = netG((noise, dog_label)).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)

    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
    for index, axis in enumerate(axes):
        axis.axis("off")
        image_array = (sample[index] + 1.0) / 2.0
        axis.imshow(image_array)
    plt.show()


def analyse_generated_by_class(n_images=5):
    good_breeds = []
    for l in range(len(decoded_dog_labels)):
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            dog_label = torch.full((1,), l, device=device, dtype=torch.long)
            gen_image = netG((noise, dog_label)).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)

        d = np.round(
            np.sum([mse(sample[k], sample[k + 1]) for k in range(len(sample) - 1)])
            / n_images,
            1,
        )
        if d < 1.0:
            continue  # had mode colapse(discard)

        print(f"Generated breed({d}): ", decoded_dog_labels[l])
        figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
        for index, axis in enumerate(axes):
            axis.axis("off")
            image_array = (sample[index] + 1.0) / 2.0
            axis.imshow(image_array)
        plt.show()

        good_breeds.append(l)
    return good_breeds


def create_submit(good_breeds):
    print("Creating submit")
    os.makedirs("../output_images", exist_ok=True)
    im_batch_size = 100
    n_images = 10000

    all_dog_labels = np.random.choice(good_breeds, size=n_images, replace=True)
    for i_batch in range(0, n_images, im_batch_size):
        noise = torch.randn(im_batch_size, nz, 1, 1, device=device)
        dog_labels = torch.from_numpy(
            all_dog_labels[i_batch : (i_batch + im_batch_size)]
        ).to(device)
        gen_images = netG((noise, dog_labels))
        gen_images = (gen_images.to("cpu").clone().detach() + 1) / 2
        for ii, img in enumerate(gen_images):
            save_image(
                gen_images[ii, :, :, :],
                os.path.join("../output_images", f"image_{i_batch + ii:05d}.png"),
            )

    import shutil

    shutil.make_archive("images", "zip", "../output_images")


# In[ ]:


class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf, crop_dogs=True):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples
        self.samples, self.labels = self.load_dogs_data(directory, crop_dogs)

    def load_dogs_data(self, directory, crop_dogs):
        required_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(64), torchvision.transforms.CenterCrop(64)]
        )

        imgs = []
        labels = []
        paths = []
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames)[: min(self.n_samples, 1e7)]:
                path = os.path.join(root, fname)
                paths.append(path)
        if LOCAL:
            ROOT = "../input/Annotation/Annotation/"
        else:
            ROOT = "../input/annotation/Annotation/"

        for path in paths:
            # Load image
            try:
                img = dset.folder.default_loader(path)
            except:
                continue

            # Get bounding boxes
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(
                dirname
                for dirname in os.listdir(ROOT)
                if dirname.startswith(annotation_basename.split("_")[0])
            )

            if crop_dogs:
                tree = ET.parse(
                    os.path.join(
                        ROOT,
                        annotation_dirname,
                        annotation_basename,
                    )
                )
                root = tree.getroot()
                objects = root.findall("object")
                for o in objects:
                    bndbox = o.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)
                    object_img = required_transforms(img.crop((xmin, ymin, xmax, ymax)))
                    imgs.append(object_img)
                    labels.append(annotation_dirname.split("-")[1].lower())

            else:
                object_img = required_transforms(img)
                imgs.append(object_img)
                labels.append(annotation_dirname.split("-")[1].lower())

        return imgs, labels

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)
        return np.asarray(sample), label

    def __len__(self):
        return len(self.samples)


# In[ ]:


# model utilities
# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.0).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)
        # return the computed values:
        return y


def snconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
):
    return spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(
        nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    )


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.snconv1x1_phi = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.snconv1x1_g = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.snconv1x1_attn = snconv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma * attn_g
        return out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.0)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, inputs):
        x, y = inputs

        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


# In[ ]:


class UpConvBlock(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        num_classes,
        k_size=4,
        stride=2,
        padding=0,
        bias=False,
        dropout_p=0.0,
        norm=None,
    ):
        super(UpConvBlock, self).__init__()
        self.norm = norm
        self.dropout_p = dropout_p
        self.upconv = spectral_norm(
            nn.ConvTranspose2d(
                n_input,
                n_output,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )
        if norm == "cbn":
            self.norm = ConditionalBatchNorm2d(n_output, num_classes)
        elif norm == "pixnorm":
            self.norm = PixelwiseNorm()
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(n_output)
        elif norm == None:
            self.norm = None
        self.activ = nn.LeakyReLU(0.05, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, inputs):
        x0, labels = inputs

        x = self.upconv(x0)
        if self.norm is not None:
            if self.norm == "cbn":
                x = self.activ(self.norm((x, labels)))
            else:
                x = self.activ(self.norm(x))
        if self.dropout_p > 0.0:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz=128, num_classes=120, channels=3, nfilt=64):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.channels = channels

        self.label_emb = nn.Embedding(num_classes, nz)
        self.pixnorm = PixelwiseNorm()

        self.upconv1 = UpConvBlock(
            2 * nz,
            nfilt * 16,
            num_classes,
            k_size=4,
            stride=1,
            padding=0,
            dropout_p=0.1,
        )
        self.upconv2 = UpConvBlock(
            nfilt * 16,
            nfilt * 8,
            num_classes,
            k_size=4,
            stride=2,
            padding=1,
            dropout_p=0.1,
            norm="pixnorm",
        )
        self.upconv3 = UpConvBlock(
            nfilt * 8,
            nfilt * 4,
            num_classes,
            k_size=4,
            stride=2,
            padding=1,
            dropout_p=0.1,
            norm="pixnorm",
        )
        self.upconv4 = UpConvBlock(
            nfilt * 4,
            nfilt * 2,
            num_classes,
            k_size=4,
            stride=2,
            padding=1,
            dropout_p=0.1,
            norm="pixnorm",
        )
        self.upconv5 = UpConvBlock(
            nfilt * 2,
            nfilt,
            num_classes,
            k_size=4,
            stride=2,
            padding=1,
            dropout_p=0.1,
            norm="pixnorm",
        )
        self.self_attn = Self_Attn(nfilt)
        self.upconv6 = UpConvBlock(nfilt, 3, num_classes, k_size=3, stride=1, padding=1)
        self.out_conv = spectral_norm(nn.Conv2d(3, 3, 3, 1, 1, bias=False))
        self.out_activ = nn.Tanh()

    def forward(self, inputs):
        z, labels = inputs

        enc = self.label_emb(labels).view((-1, self.nz, 1, 1))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((z, enc), 1)

        x = self.upconv1((x, labels))
        x = self.upconv2((x, labels))
        x = self.upconv3((x, labels))
        x = self.upconv4((x, labels))
        x = self.upconv5((x, labels))
        x = self.self_attn(x)
        x = self.upconv6((x, labels))
        x = self.out_conv(x)
        img = self.out_activ(x)
        
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes=120, channels=3, nfilt=64, emb_dim=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        def down_convlayer(
            n_input, n_output, k_size=4, stride=2, padding=0, dropout_p=0.0
        ):
            block = [
                spectral_norm(
                    nn.Conv2d(
                        n_input,
                        n_output,
                        kernel_size=k_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(n_output),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if dropout_p > 0.0:
                block.append(nn.Dropout(p=dropout_p))
            return block

        self.label_emb = nn.Embedding(num_classes, self.emb_dim * self.emb_dim)
        self.model = nn.Sequential(
            *down_convlayer(self.channels + 1, nfilt, 4, 2, 1),
            Self_Attn(nfilt),
            *down_convlayer(nfilt, nfilt * 2, 4, 2, 1, dropout_p=0.10),
            *down_convlayer(nfilt * 2, nfilt * 4, 4, 2, 1, dropout_p=0.15),
            *down_convlayer(nfilt * 4, nfilt * 8, 4, 2, 1, dropout_p=0.25),
            spectral_norm(nn.Conv2d(nfilt * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, inputs):
        imgs, labels = inputs

        enc = self.label_emb(labels).view((-1, 1, self.emb_dim, self.emb_dim))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((imgs, enc), 1)
        out = self.model(x)
        out = torch.sigmoid(out)
        
        return out.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# model definitions
class Generator(nn.Module):
    def __init__(self, nz=128, num_classes=120, nfeats=64, nchannels=3):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, nz)
        # input is Z, going into a convolution
        self.conv1 = spectral_norm(
            nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)
        )
        # self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4

        self.conv2 = spectral_norm(
            nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)
        )
        # self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8

        self.conv3 = spectral_norm(
            nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        )
        # self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16

        self.conv4 = spectral_norm(
            nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        )
        # self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32

        self.conv5 = spectral_norm(
            nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)
        )
        # self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64

        self.conv6 = spectral_norm(
            nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False)
        )
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()

    def forward(self, inputs):

        z, labels = inputs

        enc = self.label_emb(labels).view((-1, self.nz, 1, 1))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((z, enc), 1)
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, nchannels=3, nfeats=64):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32

        self.conv2 = spectral_norm(nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16

        self.conv3 = spectral_norm(
            nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        )
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8

        self.conv4 = spectral_norm(
            nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        )
        self.bn4 = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8 + 1, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        # x = self.pixnorm(x)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        # x= self.conv5(x)
        return x.view(-1, 1)
# In[ ]:


LOCAL = False
BATCH_SIZE = 32

start = time()
seed_everything()

# dataset initialization
if LOCAL:
    database = "../input/all-dogs/"
else:
    database = "../input/all-dogs/all-dogs/"

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = DataGenerator(database, transform=transform, n_samples=25000)
train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=4
)

decoded_dog_labels = {
    i: breed for i, breed in enumerate(sorted(set(train_data.labels)))
}
encoded_dog_labels = {
    breed: i for i, breed in enumerate(sorted(set(train_data.labels)))
}
train_data.labels = [encoded_dog_labels[l] for l in train_data.labels]


# In[ ]:


# inspect training samples & labels, first batch
for i, (x, y) in enumerate(train_loader):
    imgs_, labels_ = x, y
    break
imgs_ = np.swapaxes(imgs_.numpy(), 1, -1)
labels_ = labels_.numpy()

N_COLS, N_ROWS = 4, 4
fig, ax = plt.subplots(N_COLS, N_ROWS, figsize=(20, 20))
idx = 0
for i in range(N_COLS):
    for j in range(N_ROWS):
        ax[i, j].imshow(imgs_[idx])
        ax[i, j].set_title(labels_[idx])
        idx += 1


# In[ ]:


# training parameters initialization
LR_DISC = 0.0003
LR_GEN = 0.0001
NOISE_DIM = 128

beta1 = 0.5
epochs = 301
real_label = 0.7
fake_label = 0.0


criterion = nn.BCELoss()
# criterion = nn.MSELoss()

netG = Generator(nz=NOISE_DIM, num_classes=120).to(device)
netD = Discriminator(120, 3, 64).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=LR_DISC, betas=(beta1, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=LR_GEN, betas=(beta1, 0.99))

lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizerG, T_0=epochs // 200, eta_min=0.00005
)
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizerD, T_0=epochs // 200, eta_min=0.00005
)

nz = NOISE_DIM
fixed_noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
batch_size = train_loader.batch_size


# In[ ]:


### training here
step = 0
for epoch in range(epochs):
    for ii, (real_images, class_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        end = time()
        if (end - start) > 25000:
            break
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_images = real_images.to(device)
        class_labels = torch.tensor(class_labels, device=device)
        batch_size = real_images.size(0)
        labels = torch.full(
            (batch_size, 1), real_label, device=device
        ) + np.random.uniform(-0.1, 0.1)

        output = netD((real_images, class_labels))
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG((noise, class_labels))
        labels.fill_(fake_label) + np.random.uniform(0, 0.2)
        output = netD((fake.detach(), class_labels))
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD((fake, class_labels))
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if step % 500 == 0:
            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (
                    epoch + 1,
                    epochs,
                    ii,
                    len(train_loader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

            valid_image = netG((fixed_noise, class_labels))
        step += 1
        lr_schedulerG.step(epoch)
        lr_schedulerD.step(epoch)

    if epoch % 10 == 0:
        show_generated_img()

# torch.save(netG.state_dict(), 'generator.pth')
# torch.save(netD.state_dict(), 'discriminator.pth')


# In[ ]:


good_breeds = analyse_generated_by_class(6)
create_submit(good_breeds)

