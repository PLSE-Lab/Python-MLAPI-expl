#!/usr/bin/env python
# coding: utf-8

# In[ ]:


EPOCHS = 574
BATCH = 32
SIZE = 64

LRG = 0.0004
LRD = 0.0008

NZ = 64
FEATURES_G = 80
FEATURES_D = 112

LRFACTOR_G = 0.9
LRFACTOR_D = 0.9

REAL_LABEL = [0.9, 0.9]
FAKE_LABEL = [0.00, 0.00]
REAL_FAKE_LABEL = 0.9

SEED = 1234
RATIO = 1.50

PATH_ANNOTATIONS = '../input/annotation/Annotation'
PATH_ALL_DOGS = '../input/all-dogs/all-dogs'
PATH_DOGS = '../cropped-dogs'


# In[ ]:


import os
import sys
import time
import math
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn, optim
import torch.optim as op

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.set_random_seed(seed)

def R4(f): return F'{f:0.4}'.ljust(6, '0').ljust(8, ' ')
def R6(f): return F'{f:0.6}'.ljust(8, '0').ljust(8, ' ')
    
random_seed(SEED) 


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport xml.etree.ElementTree as ET\n\n'''\n    Cropped Images\n'''\ndef enumerate_images(path):\n    for fn in os.listdir(path):\n        with Image.open(os.path.join(path, fn)) as image:\n            yield image.copy()\n\ndef enumerate_annotation_images(path_annotation, path_dogs):\n    breed_id = 0\n    breeds = os.listdir(path_annotation)\n    for breed in breeds:\n        dog_id = 0\n        breed_id += 1\n        for dog in os.listdir(os.path.join(path_annotation, F'{breed}')):\n            try:\n                image = Image.open(os.path.join(path_dogs, F'{dog}.jpg'))\n            except :\n                print(dog, 'not found!')\n                continue\n\n            tree = ET.parse(os.path.join(path_annotation, F'{breed}', F'{dog}'))\n            root = tree.getroot()\n            objects = root.findall('object')\n            for o in objects:\n                dog_id += 1\n                bndbox = o.find('bndbox')\n                xmin = int(bndbox.find('xmin').text)\n                ymin = int(bndbox.find('ymin').text)\n                xmax = int(bndbox.find('xmax').text)\n                ymax = int(bndbox.find('ymax').text)\n                img = image.crop((xmin, ymin, xmax, ymax))\n                yield img, breed_id, dog_id\n\ndef save_cropped_images(path_annotation, path_dogs, path_target, size=64):\n    if not os.path.exists(path_target):\n        os.mkdir(path_target)\n    n = 0\n    current_id = 0\n    with tqdm(enumerate_annotation_images(path_annotation, path_dogs)) as t:\n        for image, breed_id, dog_id in t:\n            if current_id != breed_id:\n                current_id = breed_id\n                t.set_postfix({'id': breed_id})\n            w, h = image.size\n            if w < size or h < size: continue\n            ratio = w / h\n            if h > w: ratio = h / w\n            sz = math.ceil(size * ratio)\n            if ratio < RATIO:\n                image.thumbnail((sz, sz), Image.ANTIALIAS)\n                image.save(os.path.join(path_target, F'{breed_id:03}_{dog_id:03}.png'))\n                n += 1\n    return n\n\nsave_cropped_images(PATH_ANNOTATIONS, PATH_ALL_DOGS, PATH_DOGS)")


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    GENERATOR
'''
class Generator(nn.Module):
    def __init__(self, nz, nfeats):
        super(Generator, self).__init__()

        self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats * 8)
        self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 8)
        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 2)
        self.conv5 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats)
        self.conv6 = nn.ConvTranspose2d(nfeats, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = torch.tanh(self.conv6(x))
        return x

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, path):
        torch.save(self.state_dict(), path)

'''
    DISCRIMINATOR
'''
class Discriminator(nn.Module):
    def __init__(self, nfeats):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, nfeats, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        self.conv5 = nn.Conv2d(nfeats * 8, 1, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = torch.sigmoid(self.conv5(x))
        x = x.view(-1, 1)
        return x

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, path):
        torch.save(self.state_dict(), path)


# In[ ]:


import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset

import cv2

'''
    DATASET
'''
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        width = max(height, int(w * r))
    else:
        r = width / float(w)
        height = max(width, int(h * r))
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def random_crop(arr_image, width, height):
    x = random.randint(0, arr_image.shape[1] - width)
    y = random.randint(0, arr_image.shape[0] - height)
    arr_image = arr_image[y:y + height, x:x + width]
    return arr_image

def random_fliplr(arr_image):
    r = random.randint(0, 2)
    if r == 1:
        return np.fliplr(arr_image)
    return arr_image

class ArrayDataset(Dataset):
    def __init__(self, path, size=64, margin=0, mean=None, std=None, count=None):
        self.size = size
        self.margin = margin
        files = os.listdir(path)
        if count is not None:
            files = files[:count]
        self.arrange(path, files)
        if mean is None or std is None:
            m, s = self.calc_norm()
            if mean is None: mean = m
            if std is None: std = s
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def arrange(self, path, files):
        self.images = []
        self.labels = []
        sizem = self.size + self.margin
        for fn in files:
            image = cv2.imread(os.path.join(path, fn))
            if image.shape[0] > image.shape[1]:
                image = image_resize(image, width=sizem)
            else:
                image = image_resize(image, height=sizem)
            image = image[..., ::-1].copy()
            self.images.append(image)
            lbl = fn[:3]
            if str.isdigit(lbl):
                self.labels.append(int(lbl))
            else:
                self.labels.append(0)

    def calc_norm(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        for x in self.images:
            x = np.transpose(x, (2, 0, 1))
            x = x.reshape(3, -1)
            mean += x.mean(1)
            std += x.std(1)
        m = (mean / len(self.images)) / 255.0
        s = (std / len(self.images)) / 255.0
        return m, s

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = random_crop(x, self.size, self.size)
        x = random_fliplr(x).copy()
        x = np.transpose(x, (2, 0, 1))
        x = x / 255.0
        x = x - self.mean
        x = x / self.std
        x = x.astype(np.float32)
        return x, self.labels[idx]


# In[ ]:


import numpy as np
import torch

from tqdm import tqdm as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    TRAINER
'''
class Trainer(object):
    def __init__(self, netG, netD, optimizerG, optimizerD, criterion, mean, std):
        self.netG = netG
        self.netD = netD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.mean = mean
        self.std = std

    def next_step(self, data_loader, noise_gen, label_gen):
        for step, (real_images, real_labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            labels_true = label_gen(batch_size, 'real').to(DEVICE)
            labels_false = label_gen(batch_size, 'fake').to(DEVICE)
            labels_true_false = label_gen(batch_size, 'real-fake').to(DEVICE)

            # train with real
            self.netD.zero_grad()
            real_images = real_images.to(DEVICE)
            output = self.netD(real_images)
            errD_real = self.criterion(output, labels_true)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = noise_gen(batch_size).to(DEVICE)
            fake = self.netG(noise)
            fake = fake / 2 + 0.5
            fake = fake - self.mean
            fake = fake / self.std

            output = self.netD(fake.detach())
            errD_fake = self.criterion(output, labels_false)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizerD.step()
        
            # Update G network
            self.netG.zero_grad()
            output = self.netD(fake)
            errG = self.criterion(output, labels_true_false)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()

        return errD.item(), errG.item(), D_x, D_G_z1, D_G_z2

    def stabilize(self, count, batch_size, noise_gen):
        self.netG.train()
        self.netD.train()
        with torch.no_grad():
            for n in range(count // batch_size):
                gen_z = noise_gen(batch_size).to(DEVICE)
                _ = self.netG(gen_z)

    def generate(self, count, batch_size, noise_gen):
        self.netG.eval()
        images_set = []
        with torch.no_grad():
            for n in range(count // batch_size):
                gen_z = noise_gen(batch_size).to(DEVICE)
                gen_images = self.netG(gen_z)
                images = gen_images.detach().cpu()
                images_set.append(images)
        images = torch.cat(images_set)
        self.netG.train()
        return images

    def generate_numpy(self, count, batch_size, noise_gen):
        self.netG.eval()
        images_set = []
        with torch.no_grad():
            for n in range(count // batch_size):
                gen_z = noise_gen(batch_size).to(DEVICE)
                gen_images = self.netG(gen_z)
                images = gen_images.detach().cpu()
                images = images.numpy().transpose(0, 2, 3, 1)
                images = (images + 1.0) * 0.5 * 255
                images = images.astype(int)
                images_set.append(images)
        images = np.concatenate(images_set)
        self.netG.train()
        return images


# In[ ]:


get_ipython().run_cell_magic('time', '', "'''\n    PREPARE\n'''\ntrain_data = ArrayDataset(PATH_DOGS, size=SIZE, margin=0)\ntrain_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH, num_workers=4, drop_last=True)\n\nnetG = Generator(NZ, FEATURES_G).to(DEVICE)\nnetD = Discriminator(FEATURES_D).to(DEVICE)\n\ncriterion = nn.BCELoss()\n\noptimizerG = optim.Adam(netG.parameters(), lr=LRG, betas=(0.5, 0.999))\noptimizerD = optim.Adam(netD.parameters(), lr=LRD, betas=(0.5, 0.999))\n\nschedulerG = op.lr_scheduler.LambdaLR(optimizerG, lambda epoch: LRFACTOR_G ** epoch)\nschedulerD = op.lr_scheduler.LambdaLR(optimizerD, lambda epoch: LRFACTOR_D ** epoch)\n\ndef noise_generator(batch_size):\n    return torch.randn(batch_size, NZ, 1, 1)\n\ndef label_generator(batch_size, mode):\n    if mode == 'real':\n        label = np.random.uniform(*REAL_LABEL, batch_size).reshape(batch_size, 1)\n        return torch.tensor(label, dtype=torch.float32)\n    if mode == 'fake':\n        label = np.random.uniform(*FAKE_LABEL, batch_size).reshape(batch_size, 1)\n        return torch.tensor(label, dtype=torch.float32)\n    if mode == 'real-fake':\n        return torch.full((batch_size, 1), REAL_FAKE_LABEL)\n    raise Exception('Invalid parameter mode')\n    ")


# In[ ]:


def needs_scheduler(epoch):
    if epoch > 0:
        if epoch % 50 == 0:
            print('Scheduler update')
            return True
    return False


# In[ ]:


get_ipython().run_cell_magic('time', '', "'''\n    TRAIN\n'''\nhist_metrics = []\nlast_metrics = 0\n\nmean = train_data.mean\nmean = np.expand_dims(mean, axis=0)\nmean = np.repeat(mean, BATCH, 0)\nmean = torch.tensor(mean).float().to(DEVICE)\n\nstd = train_data.std\nstd = np.expand_dims(std, axis=0)\nstd = np.repeat(std, BATCH, 0)\nstd = torch.tensor(std).float().to(DEVICE)\n\ntrainer = Trainer(netG, netD, optimizerG, optimizerD, criterion, mean, std)\n\nmetrics = 0.0\nt0 = time.time()\nprint('Training...')\nfor epoch in range(EPOCHS):        \n    random_seed(SEED + epoch)\n    errD, errG, dx, dgz1, dgz2 = trainer.next_step(train_loader, noise_generator, label_generator)\n    \n    if needs_scheduler(epoch):\n        schedulerG.step()\n        schedulerD.step()")


# In[ ]:


torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')


# In[ ]:


random_seed(1234)
images = trainer.generate(32, 32, noise_generator)
image = (images + 1.0) * 0.5

save_image(images, 'samples.jpg', normalize=True)
img = Image.open('samples.jpg')
img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.ANTIALIAS)
img.save('samples.jpg')

plt.figure(figsize=(16, 16))
plt.imshow(np.transpose(make_grid(images, padding=2, normalize=True), (1, 2, 0)))
plt.show()


# In[ ]:


SEEDS = [
1525, 491, 1117, 1509, 1538, 1478, 16, 1251, 1126, 81, 254, 704, 1740, 2081, 79, 980, 1770, 795, 1317, 1471, 2198, 153, 1488, 1456, 454, 573, 690, 2289, 1753, 2075, 772, 301, 1240, 83, 686, 932, 826, 27, 24, 1744, 880, 2029, 1421, 1490, 667, 1387, 1948, 2130, 15, 133, 2151, 602, 76, 2090, 1277, 717, 2175, 2168, 1313, 1562, 1526, 1092, 800, 1804, 1605, 1120, 2320, 2372, 2080, 460, 126, 2302, 1940, 983, 668, 773, 1788, 565, 1540, 2337, 140, 2173, 2010, 1784, 1248, 1106, 1722, 208, 1757, 2371, 449, 481, 1914, 2217, 437, 1161, 169, 693, 807, 1839, 217, 1520, 1102, 30, 1199, 239, 603, 730, 569, 2309, 1434, 1679, 3, 313, 532, 35, 825, 824, 388, 255, 1758, 763, 87, 751, 2073, 585, 407, 1510, 365, 739, 417, 463, 1412, 1636, 1237, 1439, 278, 227, 1916, 2261, 154, 1622, 1833, 2213, 1738, 74, 892, 1203, 2001, 1227, 815, 872, 1769, 699, 2374, 1050, 1073, 620, 202, 1420, 2211, 840, 568, 1715, 1337, 1651, 1591, 878, 1445, 1100, 1702, 551, 72, 1917, 1256, 1479, 152, 2192, 559, 2031, 1920, 23, 2304, 2153, 1848, 2032, 1976, 578, 650, 1464, 1706, 820, 2232, 762, 33, 1627, 1141, 2262, 781, 174,
]


# In[ ]:


state_dict = torch.load('generator.pth')

def load_model(model):
    model.load_state_dict(state_dict)

def generate_numpy_images(netG, count, batch_size, noise_gen):
    netG.eval()
    images_set = []
    with torch.no_grad():
        for n in range(count // batch_size):
            gen_z = noise_gen(batch_size).to(DEVICE)
            gen_images = netG(gen_z)
            aimages = gen_images.detach().cpu()
            aimages = (aimages + 1.0) * 0.5 * 255
            images = aimages.numpy().transpose(0, 2, 3, 1)
            images = images.astype(int)
            images_set.append(images)
    images = np.concatenate(images_set)
    return images

def generate_tensors(netG, count, batch_size, noise_gen):
    netG.eval()
    images_set = []
    with torch.no_grad():
        for n in range(count // batch_size):
            gen_z = noise_gen(batch_size).to(DEVICE)
            gen_images = netG(gen_z)
            images = gen_images.detach().cpu()
            images_set.append(images)
    images = torch.cat(images_set)
    netG.train()
    return images

def generate_images(count, batch, seed):
    netG = Generator(NZ, FEATURES_G).to(DEVICE)
    load_model(netG)
    random_seed(seed)
    images = generate_tensors(netG, count, batch, noise_generator)
    return images


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nPATH_GENERATED = '../output_images'\n\nif not os.path.exists(PATH_GENERATED):\n    os.mkdir(PATH_GENERATED)\n\nn = 0\nfor seed in SEEDS:\n    images = generate_images(50, 50, seed)\n    images = (images + 1.0) * 0.5\n    for ix in range(50):\n        image = images[ix]\n        save_image(image, os.path.join(PATH_GENERATED, f'image_{n:05d}.png'), normalize=False)\n        n += 1\n\nprint(n)\n\nimport shutil\nshutil.make_archive('images', 'zip', PATH_GENERATED)    ")

