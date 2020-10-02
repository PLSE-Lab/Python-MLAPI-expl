#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import copy
from itertools import chain
import io
import math
from multiprocessing import cpu_count
from pathlib import Path
from pdb import set_trace
import time
from threading import Thread
from xml.etree import ElementTree
import zipfile

# General utils
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from imageio import imread
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from scipy.stats import truncnorm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image


# ### Kernel global variables initialization
# 
# We initialize random generator seed, override `print` function to duplicate its output into kernel logging stream, and setup a "watchdog" that tracks how many time we've spent to run the kernel.

# In[ ]:


HOUR = 3600

class Watchdog:
    def __init__(self, max_seconds=8.5 * HOUR):
        self.start = time.time()
        self.deadline = max_seconds
    
    @property
    def timeout(self):
        return self.elapsed >= self.deadline
       
    @property
    def elapsed(self):
        return time.time() - self.start

wd = Watchdog()

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
ANNOTS = Path.cwd().parent/'input'/'annotation'/'Annotation'
IMAGES = Path.cwd().parent/'input'/'all-dogs'/'all-dogs'

try:
    # make sure we patch printing function only once
    patched
except NameError:
    patched = True
    __print__ = print
    def print(message):
        import os
        from datetime import datetime
        log_message = datetime.now().strftime(f'[Kernel][%Y-%m-%d %H:%M:%S] {message}')
        os.system(f'echo \"{log_message}\"')
        __print__(message)
        
class VisualStyle:
    """Convenience wrapper on top of matplotlib config."""

    def __init__(self, config, default=None):
        if default is None:
            default = plt.rcParams
        self.default = default.copy()
        self.config = config

    def replace(self):
        plt.rcParams = self.config

    def override(self, extra=None):
        plt.rcParams.update(self.config)
        if extra is not None:
            plt.rcParams.update(extra)

    def restore(self):
        plt.rcParams = self.default

    def __enter__(self):
        self.override()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


class NotebookStyle(VisualStyle):
    def __init__(self):
        super().__init__({
            'figure.figsize': (11, 8),
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'font.size': 16
        })

NotebookStyle().override()
        
print(f'Annotations: {ANNOTS}')
print(f'Images: {IMAGES}')


# ### Functions to prepare the dataset for training
# 
# We need to convert the dataset into format ready for training. For this purpose, we read and crop images, and concatenate them into a single tensor of (B x C x W x H) format suitable for PyTorch.

# In[ ]:


def parse_annotation(path):
    root = ElementTree.parse(path).getroot()
    size = [int(root.find(f'size/{leaf}').text) 
            for leaf in ('width', 'height')] 
    bbox = [int(root.find(f'object/bndbox/{leaf}').text) 
            for leaf in ('xmin', 'ymin', 'xmax', 'ymax')]
    breed = path.parent.name.split('-')[-1]
    return {'path': str(path), 'name': path.name, 
            'breed': breed, 'size': size, 'bbox': bbox}

def enrich_with_image_paths(annotations, images_directory):
    image_files = {x.stem: x for x in images_directory.iterdir()}
    enriched_data = []
    for annot in annotations:
        image_path = image_files.get(annot['name'], None)
        if image_path is None:
            print('Warning: image not found for annotation entry: %s.' % annot['path'])
            continue
        annot['image'] = str(image_path)
        enriched_data.append(annot)
    return enriched_data

def load_annotations():
    return enrich_with_image_paths([
        parse_annotation(path) 
        for directory in ANNOTS.iterdir() 
        for path in directory.iterdir()
    ], IMAGES)

def dog(annot):
    img = imread(annot['image'])
    xmin, ymin, xmax, ymax = annot['bbox']
    cropped = img[ymin:ymax, xmin:xmax]    
    return cropped

def chunks(seq, chunk_size=10):
    n = len(seq)
    n_chunks = n // chunk_size + int((n % chunk_size) != 0)
    for i in range(n_chunks):
        yield seq[i*chunk_size:(i+1)*chunk_size]
        
def resize(image, new_size):
    return np.array(PIL.Image.fromarray(image).resize(new_size))

def parallel(func, sequence, func_args=None, n_jobs=None):
    with Parallel(n_jobs=n_jobs or cpu_count()) as p:
        func_args = func_args or {}
        results = p(delayed(func)(item, **func_args) for item in sequence)
    return results

def load_single_image(annot, size):
    cropped = dog(annot)
    resized = resize(cropped, size)
    return resized

def load_dogs_images(annots, size=(64, 64)):
    return np.stack(parallel(load_single_image, annots, func_args={'size': size}))

def as_pil_list(dataset):
    return [PIL.Image.fromarray(image, 'RGB') for image in dataset]

class Normalizer:
    def __init__(self, method='tanh', params=None):
        assert method in ('tanh', 'stats')
        self.method = method
        self.params = params or {}
    
    def transform(self, dataset):
        dataset = dataset.float()
        if self.method == 'tanh':
            return (dataset - 127.5)/127.5
        if self.method == 'stats':
            mean = self.params.get('mean', (0.5, 0.5, 0.5))
            std = self.params.get('std', (0.5, 0.5, 0.5))
            return functional.normalize(dataset, mean, std)
    
    def inv_transform(self, dataset):
        if self.method == 'tanh':
            inv_dataset = (dataset + 1)*127.5
        if self.method == 'stats':
            mean = params.get('mean', (0.5, 0.5, 0.5))
            std = params.get('std', (0.5, 0.5, 0.5))
            mean, std = [torch.as_tensor(
                x, dtype=torch.float32, device=dataset.device)]
            dataset.mul_(std[:, None, None]).add_(mean[:, None, None])
            inv_dataset = dataset
        return inv_dataset.long()


# ### Reading the data
# 
# We use the functions defined above to read the data and prepare it for training.

# In[ ]:


print('Reading dogs images and annotations.')
annots = load_annotations()
print(f'Total number of examples: {len(annots)}.')
dogs = load_dogs_images(annots, (128, 128))
assert len(dogs) == len(annots)
print(f'Dogs dataset shape: {dogs.shape}.')
pils = as_pil_list(dogs)
print(f'Numbers of PIL images: {len(pils)}')
del dogs, annots


# In[ ]:


def show_pil(img, *imgs, n_rows=4):
    imgs = [img] + list(imgs)
    n_cols = len(imgs) // n_rows
    f, axes = plt.subplots(n_rows, n_cols)
    for img, ax in zip(imgs, axes.flat): 
        ax.imshow(img)
        ax.axis('off')
    f.subplots_adjust(wspace=0, hspace=0)


# In[ ]:


# show_pil(*pils[:16])


# In[ ]:


class PILDataset:
    def __init__(self, pil_images, transform=None):
        self.pil_images = pil_images
        self.tr = transform or (lambda x: x)
    def __getitem__(self, i):
        if isinstance(i, int): return self.tr(self.pil_images[i])
        elif isinstance(i, (list, np.ndarray)): return [self.tr(self.pil_images[ii]) for ii in i]
        elif isinstance(i, slice): return [self.tr(img) for img in self.pil_images[i]]
        raise TypeError(f'unknown index type: {type(i)}')
    def __len__(self):
        return len(self.pil_images)

class RandomCropOfFive:
    def __init__(self, size):
        self.five_crop = transforms.FiveCrop(size)
    def __call__(self, x):
        [idx] = np.random.randint(0, 4, 1)
        cropped = self.five_crop(x)[idx]
        return cropped
    
def show_tensor(t, n_rows=4, denorm=False):
    if denorm: t = (255 * (t + 1)/2)
    canvas = make_grid(t).numpy().transpose(1, 2, 0).astype(np.uint8)
    f, ax = plt.subplots(1, 1)
    ax.imshow(canvas)
    ax.axis('off')


# In[ ]:


# show_tensor(torch.stack(dataset[np.random.randint(0, len(dataset), 64)]), denorm=True)


# In[ ]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class SpectralNorm(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)
    def forward(self, x):
        return self.module(x)

class PixelwiseNorm(nn.Module):
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.alpha).sqrt()
        y = x / y
        return y
    
class MinibatchStdDev(nn.Module):
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = y.pow(2.).mean(dim=0, keepdim=False).add(self.alpha).sqrt()
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        y = torch.cat([x, y], 1)
        return y
    
class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    def __call__(self, b1, b2): 
        assert b1.size(0) == b2.size(0)
        lam = np.random.beta(self.alpha, self.alpha, size=b1.size(0))
        lam = torch.from_numpy(lam).float().to(b1.device)
        lam = lam.view(-1, 1, 1, 1)
        return lam*b1 + (1 - lam)*b2


# In[ ]:


class DCGAN_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            MinibatchStdDev(),
            SpectralNorm(nn.Conv2d(1024 + 1, 1, 4, 2, 1, bias=False)),
            nn.Sigmoid()
        )
        self.main.apply(weights_init)
    def forward(self, x):
        return self.main(x)


# In[ ]:


class DCGAN_G(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(nz, 1024, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            PixelwiseNorm(),
            SpectralNorm(nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False)),
            nn.Tanh()
        )
        self.main.apply(weights_init)
    def forward(self, x):
        return self.main(x)


# In[ ]:


bs = 16
nz = 128
lr_d = 0.0005
lr_g = 0.0005
beta_1 = 0.5
use_adam = True
mixup = Mixup(0.2)

dataset = PILDataset(pils, transform=transforms.Compose([
    transforms.Resize(70),
    RandomCropOfFive(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

netD = DCGAN_D().cuda()
netG = DCGAN_G(nz).cuda()
if use_adam:
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta_1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta_1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr_d)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr_g)
schedD = CosineWithRestarts(optimizerD, eta_min=lr_d*0.1, t_initial=1000, t_mul=math.sqrt(2))
schedG = CosineWithRestarts(optimizerG, eta_min=lr_g*0.1, t_initial=1000, t_mul=math.sqrt(2))


# In[ ]:


def truncated_normal(size, threshold=1):
    return truncnorm.rvs(-threshold, threshold, size=size)

def sample(dataset, batch_size):
    idx = np.random.randint(0, len(dataset), batch_size)
    return torch.stack(dataset[idx]).cuda()

def smooth_positive(labels):
    jitter = torch.from_numpy(np.random.uniform(0.05, 0.1, len(labels))).float().to(labels.device)
    jitter = jitter.view(labels.size())
    return (labels - jitter)

def exp_mov_avg(acc_net, curr_net, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(acc_net.parameters(), curr_net.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# In[ ]:


# from functools import reduce

# def ema(new_value, acc, alpha=0.999):
#     return alpha*acc + new_value*(1 - alpha)

# xs = np.random.randn(1000).cumsum()
# acc = xs[0]
# ys = [acc]
# for x in xs[1:]:
#     ys.append(ema(x, acc, 0.99))
#     acc = ys[-1]

# f, ax = plt.subplots(1, 1)
# ax.plot(xs, label='value')
# ax.plot(ys, label='EMA')
# ax.legend()


# In[ ]:


def ema(avg_net, curr_net, alpha=0.999):
    for avg_param, curr_param in zip(avg_net.parameters(), curr_net.parameters()):
        avg_param.data.mul_(alpha).add_(1 - alpha, curr_param.data)


# In[ ]:


def train(bs=32):
    print('Starting training loop...')
    epoch = 0
    real_label = 1
    fake_label = 0
    loss_fn = nn.BCELoss()
    n = len(dataset)
    n_batches = n // bs 
    avg_model = type(netG)(netG.nz).cuda()
    avg_model.load_state_dict(netG.state_dict())
    
    while True:
        idx1 = np.random.permutation(n)
        idx2 = np.random.permutation(n)
        
        for i in range(n_batches):

            if wd.timeout: return avg_model
            
            epoch += 1
            
            batch1 = torch.stack(dataset[idx1[i*bs:(i+1)*bs]]).float().cuda()
            batch2 = torch.stack(dataset[idx2[i*bs:(i+1)*bs]]).float().cuda()
            mixed = mixup(batch1, batch2)
            
            netD.zero_grad()
            x_real = mixed.cuda()
            batch_size = x_real.size(0)
            labels = torch.full((batch_size, 1), real_label).cuda()
            labels = smooth_positive(labels) 
            output = netD(x_real).view(-1, 1)
            errD_real = loss_fn(output, labels)
            errD_real.backward()
            d_x = output.mean().item()

            noise = torch.from_numpy(truncated_normal((batch_size, nz, 1, 1))).float().cuda()
            x_fake = netG(noise)
            labels.fill_(fake_label)
            output = netD(x_fake.detach()).view(-1, 1)
            errD_fake = loss_fn(output, labels)
            errD_fake.backward()
            d_g_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            labels.fill_(real_label)
            output = netD(x_fake).view(-1, 1)
            errG = loss_fn(output, labels)
            errG.backward()
            d_g_z2 = output.mean().item()
            optimizerG.step()
            
            if epoch % 150 == 0:
                print(f'[{epoch:06d}][{i+1:03d}] '
                      f'lr_d: {schedD.get_values()[0]:.6f}, '
                      f'lr_g: {schedG.get_values()[0]:.6f} | '
                      f'loss_d: {errD.item():.4f}, '
                      f'loss_g: {errG.item():.4f} | '
                      f'D(x): {d_x:.4f}, D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}')
        
            schedD.step()
            schedG.step()
            ema(avg_model, netG)
            
    return avg_model


# In[ ]:


avgG = train(bs=16)


# In[ ]:


print('Final model images generation.')
print('Creating archive to write the images.')
arch = zipfile.ZipFile('images.zip', 'w')
img_no = 0
for batch in range(100):
    t_noise = torch.from_numpy(truncated_normal((100, nz, 1, 1))).float().cuda()
    # images = netG(t_noise).detach().cpu()
    images = avgG(t_noise).detach().cpu()
    images = images.mul(0.5).add(0.5)
    images = (255 * images.numpy()).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    for image in images:
        buf = io.BytesIO()
        PIL.Image.fromarray(image).save(buf, format='png')
        buf.seek(0)
        arch.writestr(f'{img_no}.png', buf.getvalue())
        img_no += 1
arch.close()
print('Saving is done!')


# In[ ]:




