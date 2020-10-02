#!/usr/bin/env python
# coding: utf-8

# ## SAGAN with residual connections
# 
# code is based on https://github.com/brain-research/self-attention-gan

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xml.etree.ElementTree as ET 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
from pathlib import Path
import enum
import functools

# Any results you write to the current directory are saved as output.


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt

Image.open("../input/all-dogs/all-dogs/n02110627_13662.jpg")


# In[ ]:


from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.utils.data

import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image

from tqdm import tqdm_notebook as tqdm


# In[ ]:


EPOCHS = 270
BATCH_SIZE = 64
NORMALIZE_Z = True

Z_SIZE = 128
N_FEATURES = 32


CONDITIONAL = True

SELF_ATTENTION = True
    
SPECTRAL_IN_G = True
SPECTRAL_IN_D = True
BN_IN_D = False
RESIDUAL_IN_D = True

GENERATOR_ACTIVATION = F.relu  # functools.partial(F.leaky_relu, negative_slope=0.02)

## Discriminator
DISCRIMINATOR_ACTIVATION = F.relu  # functools.partial(F.leaky_relu, negative_slope=0.02)


class LossType(enum.Enum):
    LS = 1
    RaLS = 2
    Hinge = 3
    RaHinge = 4
    
    def requires_real_for_generator(self) -> bool:
        return self == LossType.RaLS


LOSS = LossType.Hinge
BETA1 = 0.0
BETA2 = 0.999
LR_GENERATOR = 0.0001
LR_DISCRIMINATOR = 0.0004

CROP_PROB = 1.0


## Debugging
SHOW_FREQ = None
SHOW_FREQ_EPOCH = EPOCHS // 10


# In[ ]:


def sample(size):
    z = np.random.normal(loc=0, scale=1, size=(size, Z_SIZE)).astype(np.float32)
    if NORMALIZE_Z:
        z /= np.sqrt(np.sum(z ** 2, axis=1, keepdims=True))
    return torch.as_tensor(z)

torch.manual_seed(42)
FIXED_NOISES = sample(16)


# In[ ]:


def extract_class(path: str) -> str:
    name = Path(path).name
    idx = name.index("_")
    return name[:idx]


def gather_image_and_annots(annot_path: str = "../input/annotation/Annotation/", image_path: str = "../input/all-dogs/all-dogs/") -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    breeds = os.listdir(annot_path)
    for breed in breeds:
        for dog in os.listdir(os.path.join(annot_path, breed)):
            annot = os.path.join(annot_path, breed, dog)
            image = os.path.join(image_path, dog + ".jpg")
            if os.path.exists(image):
                results.append((image, annot))
    return results
        

def get_classes(path: str):
    classes = set()
    for filepath in Path(path).glob("*.jpg"):
        classes.add(extract_class(filepath))
    return sorted(classes)


def filter_small_boxes(boxes: List[List[int]]) -> List[List[int]]:
    return [x for x in boxes if min(x[2] - x[0], x[3] - x[1]) >= 64]


def get_bounding_boxes(path: str, filter_small: bool = True) -> List[List[int]]:
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall('object')
    boxes: List[List[int]] = []
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    if filter_small:
        boxes = filter_small_boxes(boxes)
    return boxes

Boxes = List[List[int]]

def append_bounding_boxes(pairs: List[Tuple[str, str]], q: float = 0.99) -> List[Tuple[str, Boxes]]:
    n = len(pairs)
    sizes = []
    for img, _ in tqdm(pairs):
        sizes.append(os.stat(img).st_size)
    pairs = np.array(pairs)
    sizes = np.array(sizes)
    indices = sizes < np.quantile(sizes, q)
    pairs = pairs[indices]
    results = []
    for i, (img_path, annot) in enumerate(tqdm(pairs)):
        img = Image.open(img_path)
        if min(*img.size) < 64:
            continue
        boxes = get_bounding_boxes(annot, filter_small=True)
        if not boxes:
            continue
        results.append((img_path, boxes))
    print("Valid sized images: {} / {}".format(len(results), n))
    return results

CLASSES = get_classes("../input/all-dogs/all-dogs/")
PAIRS = append_bounding_boxes(gather_image_and_annots())
IMAGES, BOXES = [p[0] for p in PAIRS], [p[1] for p in PAIRS]
len(CLASSES), len(PAIRS)


# In[ ]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_files, boxes, transform, crop_prob: float = 0.5, jitter_ratio: float = 0.05):
        super().__init__()
        self.transform = transform
        self.image_files = image_files
        self.boxes = boxes
        self.crop_prob = crop_prob
        self.jitter_ratio = jitter_ratio
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filepath = self.image_files[idx]
        image = Image.open(str(filepath)).convert("RGB")
        if self.crop_prob >= 1.0 or np.random.random() < self.crop_prob:
            boxes = self.boxes[idx]
            image = self._crop_bounding_boxes(image, boxes)
        if CONDITIONAL:
            class_id = CLASSES.index(extract_class(filepath))
        else:
            class_id = 0  # dummy
        image_tensor = self.transform(image)
        return image_tensor, torch.as_tensor(class_id)
    
    def _crop_bounding_boxes(self, image: Image.Image, boxes: List[List[int]]) -> Image.Image:
        i = np.random.randint(0, len(boxes))
        xmin, ymin, xmax, ymax = boxes[i]
        if min(xmax - xmin, ymax - ymin) < 64:
            return image
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2
        size = max(xmax - xmin, ymax - ymin)
        if int(size * self.jitter_ratio):
            x_jitter = np.random.randint(int(-size * self.jitter_ratio), int(size * self.jitter_ratio))
            x_center += x_jitter
            y_jitter = np.random.randint(int(-size * self.jitter_ratio), int(size * self.jitter_ratio))
            y_center += y_jitter
        size_jitter = np.random.randint(0, int(size * self.jitter_ratio))
        xmin = max(0, x_center - (size + size_jitter) // 2)
        ymin = max(0, y_center - (size + size_jitter) // 2)
        xmax = min(image.size[0], x_center + (size + size_jitter) // 2)
        ymax = min(image.size[1], y_center + (size + size_jitter) // 2)
        return image.crop((xmin, ymin, xmax, ymax))


# In[ ]:


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = Dataset(IMAGES, BOXES, transform, crop_prob=CROP_PROB)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
print(len(dataset), len(dataloader))
_ = next(iter(dataloader))


# ### ops

# In[ ]:


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def spectral_init(module, use_norm: bool, gain=1):
    init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    if use_norm:
        return spectral_norm(module)
    else:
        return module


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    def __init__(self, in_channel: int, use_norm: bool, gain=1):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   use_norm=use_norm,
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 use_norm=use_norm,
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   use_norm=use_norm,
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(n_class, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out

def snconv2d(
    in_features: int, 
    out_features: int, 
    kernel_size: int, 
    stride: int, 
    padding: int,
    use_norm: bool,
) -> nn.Module:
    conv = nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding, bias=True)
    init.xavier_uniform_(conv.weight)
    conv.bias.data.zero_()
    if use_norm:
        return spectral_norm(conv)
    else:
        return conv


def snlinear(in_features: int, out_features: int, use_norm: bool) -> nn.Module:
    linear = nn.Linear(in_features, out_features, bias=True)
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()
    if use_norm:
        return spectral_norm(linear)
    else:
        return linear


def sn_embedding(num_classes: int, embedding_dim: int, use_norm: bool) -> nn.Module:
    emb = nn.Embedding(num_classes, embedding_dim)
    init.xavier_uniform_(emb.weight)
    if use_norm:
        return spectral_norm(emb)
    else:
        return emb


# ### Generator

# In[ ]:


def upsample_conv(in_features: int, out_features: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        snconv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, use_norm=SPECTRAL_IN_G),
    )


class GeneratorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_classes: int) -> None:
        super().__init__()
        self.in_features = in_features
        if CONDITIONAL:
            self.bn0 = ConditionalNorm(in_features, num_classes)
            self.bn1 = ConditionalNorm(out_features, num_classes)
        else:
            self.bn0 = nn.BatchNorm2d(in_features)
            self.bn1 = nn.BatchNorm2d(out_features)
        self.upconv = upsample_conv(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.conv1 = snconv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_G)
        self.skip_upconv = upsample_conv(in_features, out_features, kernel_size=1, stride=1, padding=0)
        self.activation = GENERATOR_ACTIVATION
        
    def forward(self, x, class_id):
        x0 = x
        if CONDITIONAL:
            x = self.activation(self.bn0(x, class_id))
        else:
            x = self.activation(self.bn0(x))
        x = self.upconv(x)
        if CONDITIONAL:
            x = self.activation(self.bn1(x, class_id))
        else:
            x = self.activation(self.bn1(x))
        x = self.conv1(x)
        
        x0 = self.skip_upconv(x0)
        return x + x0
    

class Generator(nn.Module):
    def __init__(self, z_size: int = Z_SIZE, num_classes: int = len(CLASSES), features: int = N_FEATURES) -> None:
        super().__init__()
        self.linear0 = snlinear(Z_SIZE, features * 4 * 4 * 8, use_norm=SPECTRAL_IN_G)
        # 8C x 4 x 4
        self.block0 = GeneratorBlock(features * 8, features * 8, num_classes=num_classes) 
        # 8C x 8 x 8
        self.block1 = GeneratorBlock(features * 8, features * 4, num_classes=num_classes) 
        # 4C x 16 x 16
        self.block2 = GeneratorBlock(features * 4, features * 2, num_classes=num_classes)
        # 2C x 32 x 32
        if SELF_ATTENTION:
            self.attn = SelfAttention(features * 2, use_norm=SPECTRAL_IN_G)
        # 2C x 32 x 32
        self.block3 = GeneratorBlock(features * 2, features * 1, num_classes=num_classes)
        # C x 64 x 64
        self.bn = nn.BatchNorm2d(features * 1)
        # 3 x 64 x 64
        self.last_conv = snconv2d(features, 3, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_G)
        self.activation = GENERATOR_ACTIVATION
        
    def forward(self, z, class_id):
        x = self.linear0(z)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.block0(x, class_id)
        x = self.block1(x, class_id)
        x = self.block2(x, class_id)
        if SELF_ATTENTION:
            x = self.attn(x)
        x = self.block3(x, class_id)
        x = self.activation(self.bn(x))
        x = self.last_conv(x)
        x = torch.tanh(x)
        return x
    
    
generator = Generator()
print(generator(torch.randn(1, Z_SIZE), torch.zeros(1, dtype=torch.long)).shape)


# ### Discriminator

# In[ ]:


def downsample(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, down=True, activation=DISCRIMINATOR_ACTIVATION):
        super().__init__()
        self.down = down
        self.activation = activation
        if BN_IN_D:
            self.bn0 = nn.BatchNorm2d(in_features)
            self.bn1 = nn.BatchNorm2d(out_features)
        self.conv0 = snconv2d(in_features,  out_features, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_D)
        self.conv1 = snconv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_D)
        if RESIDUAL_IN_D and (down or in_features != out_features):
            self.skip_conv = snconv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, use_norm=SPECTRAL_IN_D)
        else:
            self.skip_conv = None
            
    def forward(self, x):
        x0 = x
        if BN_IN_D:
            x = self.bn0(x)
        x = self.activation(x)
        x = self.conv0(x)
        if BN_IN_D:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.down:
            x = downsample(x)
        if self.skip_conv is not None:
            x0 = self.skip_conv(x0)
            if self.down:
                x0 = downsample(x0)
        if RESIDUAL_IN_D:
            x = x0 + x
        return x


class DiscriminatorDownOptimizedBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=DISCRIMINATOR_ACTIVATION) -> None:
        super().__init__()
        self.conv1 = snconv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_D)
        self.activation = activation
        self.conv2 = snconv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, use_norm=SPECTRAL_IN_D)
        if RESIDUAL_IN_D:
            self.skip_conv = snconv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, use_norm=SPECTRAL_IN_D)
        
    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = downsample(x)
        if RESIDUAL_IN_D:
            x0 = downsample(x0)
            x0 = self.skip_conv(x0)
            x = x0 + x
        return x

    
class Discriminator(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), features: int = N_FEATURES, activation=DISCRIMINATOR_ACTIVATION) -> None:
        super().__init__()
        self.activation = activation
        self.block0 = DiscriminatorDownOptimizedBlock(3, features * 1)
        if SELF_ATTENTION:
            self.attn = SelfAttention(features * 1, use_norm=SPECTRAL_IN_D)
        self.block1 = DiscriminatorBlock(features * 1, features * 2)
        self.block2 = DiscriminatorBlock(features * 2, features * 4)
        self.block3 = DiscriminatorBlock(features * 4, features * 8)
        self.block4 = DiscriminatorBlock(features * 8, features * 8, down=False)
        self.linear = snlinear(features * 8, 1, use_norm=SPECTRAL_IN_D)
        if CONDITIONAL:
            self.emb = sn_embedding(num_classes, features * 8, use_norm=SPECTRAL_IN_D)
        
    def forward(self, x, class_id):
        x = self.block0(x)
        if SELF_ATTENTION:
            x = self.attn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        h = self.activation(x)
        h = h.sum(3).sum(2)
        output = self.linear(h).squeeze(1)
        if CONDITIONAL:
            h_labels = self.emb(class_id)
            output = output + (h * h_labels).sum(1)
        return output
    
Discriminator()(torch.randn(4, 3, 64, 64), torch.zeros(4, dtype=torch.long)).shape


# In[ ]:


@torch.no_grad()
def inverse_transformation(tensor):
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    for i in range(3):
        m = mean[i]
        s = std[i]
        tensor[:, i].mul_(s).add_(m)
    return tensor

@torch.no_grad()
def generate_images(generator, noises, labels=None):
    if labels is None:
        labels = torch.multinomial(torch.ones(len(CLASSES), dtype=torch.float32) / len(CLASSES), noises.size(0)).to(noises.device)
    images = generator(noises, labels)
    return inverse_transformation(images)


def show_images(tensor):
    grid = torchvision.utils.make_grid(tensor)
    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_numpy)
    plt.show()


# In[ ]:


class DiscriminatorCriterion(nn.Module):
    def forward(self, fake_images_output, real_images_output):
        if LOSS == LossType.RaLS:
            return torch.mean((real_images_output - fake_images_output.mean() - 1) ** 2) +                    torch.mean((fake_images_output - real_images_output.mean() + 1) ** 2)
        elif LOSS == LossType.LS:
            return torch.mean(real_images_output ** 2) + torch.mean((fake_images_output - 1) ** 2)
        elif LOSS == LossType.Hinge:
            return F.relu(1.0 - real_images_output).mean() + F.relu(1 + fake_images_output).mean()
        else:
            raise Exception("unimplemented: {}".format(LOSS))
    
    
class GeneratorCriterion(nn.Module):
    def forward(self, fake_images_output, real_images_output):
        if LOSS == LossType.RaLS:
            return torch.mean((real_images_output - fake_images_output.mean() + 1) ** 2) +                    torch.mean((fake_images_output - real_images_output.mean() - 1) ** 2)
        elif LOSS == LossType.LS:
            return torch.mean(fake_images_output ** 2)
        elif LOSS == LossType.Hinge:
            return -fake_images_output.mean()
        else:
            raise Exception("unimplemented: {}".format(LOSS))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    

def train_one_epoch(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, device, show_freq=None):
    generator.train()
    discriminator.train()
    disc_criterion = DiscriminatorCriterion()
    gen_criterion = GeneratorCriterion()
    
    lr_scheduler_gen = None
    lr_scheduler_disc = None
    if False and epoch == 0:
        lr_scheduler_gen = warmup_lr_scheduler(generator_optimizer, len(dataloader), warmup_factor=1. / 1000)
        lr_scheduler_disc = warmup_lr_scheduler(discriminator_optimizer, len(dataloader), warmup_factor=1. / 1000)
    
    loss_discriminators = []
    loss_generators = []
    for step, (real_images, real_labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        # 1. Update D
        # 1.1. train with real images
        discriminator.zero_grad()
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_images.size(0)
        
        real_images_output = discriminator(real_images, real_labels)

        # 1.2. train with fake images
        noise = sample(batch_size).to(device)
        # fake_labels = torch.multinomial(torch.ones(len(CLASSES), dtype=torch.float32) / len(CLASSES), batch_size).to(device)
        fake_images = generator(noise, real_labels)
        fake_images_output = discriminator(fake_images.detach(), real_labels)

        loss_discriminator = disc_criterion(fake_images_output, real_images_output)
        loss_discriminator.backward()
        discriminator_optimizer.step()
        loss_discriminators.append(loss_discriminator.detach().item())
        
        # 2. Update G
        generator.zero_grad()
        noise = sample(batch_size).to(device)
        fake_images = generator(noise, real_labels)
        fake_images_output = discriminator(fake_images, real_labels)
        if LOSS.requires_real_for_generator():
            real_images_output = discriminator(real_images, real_labels).detach()
        else:
            real_images_output = None
        loss_generator = gen_criterion(fake_images_output, real_images_output)
        loss_generator.backward()
        generator_optimizer.step()
        loss_generators.append(loss_generator.detach().item())
        
        if show_freq and step % show_freq == 0:
            generator.eval()
            fake_images = generate_images(generator, FIXED_NOISES.to(device))
            show_images(fake_images)
            generator.train()
            
        if lr_scheduler_gen is not None:
            lr_scheduler_gen.step()
        if lr_scheduler_disc is not None:
            lr_scheduler_disc.step()
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'
      % (epoch + 1, EPOCHS, 
         np.mean(loss_discriminators), np.mean(loss_generators)))


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()

import datetime
print(datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y/%m/%d %H:%M:%S"))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR, betas=(BETA1, BETA2))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR_DISCRIMINATOR, betas=(BETA1, BETA2))

if not os.path.exists("./tmp"):
    os.mkdir("./tmp")


# In[ ]:



for epoch in range(EPOCHS):
    print(f"[Epoch {epoch}]")
    train_one_epoch(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, device, show_freq=SHOW_FREQ)
    generator.eval()
    if (epoch + 1) % SHOW_FREQ_EPOCH == 0:
        fake_images = generate_images(generator, FIXED_NOISES.to(device))
        show_images(fake_images)
        if CONDITIONAL:
            fake_images = generate_images(generator, FIXED_NOISES.to(device))
            show_images(fake_images)
        save_image(fake_images, "./tmp/epoch{}.png".format(epoch + 1))


# In[ ]:


print(datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y/%m/%d %H:%M:%S"))


# In[ ]:


from torchvision.utils import save_image

if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 1
n_images = 10000
generator.eval()
for i_batch in tqdm(range(0, n_images), total=n_images):
    gen_z = sample(im_batch_size).to(device)
    gen_images = generate_images(generator, gen_z)
    images = gen_images.cpu().clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




