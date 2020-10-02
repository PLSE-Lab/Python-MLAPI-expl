#!/usr/bin/env python
# coding: utf-8

# ### reference: 

# In[ ]:


from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image
from torch.nn.utils import spectral_norm
from tqdm import tqdm
from torch.autograd import Variable
import sys


# In[ ]:


start_time = time.time()

batch_size = 32
epochs = 210
seed = 1029

TRAIN_DIR = Path('../input/all-dogs/')
ANNOTATION_DIR = Path('../input/annotation/Annotation/')
DOG_DIR = Path('../dogs/dogs/')
OUT_DIR = Path('../output_images/')
DOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

device = torch.device('cuda')

lr = 0.0005
beta1 = 0.5
nz = 256

real_label = 0.95
fake_label = 0

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[ ]:


mylist = os.listdir(ANNOTATION_DIR)


# In[ ]:


label_map = {}
for listname in mylist:
    label_map[listname.split('-')[0]]=listname.split('-')[1]


# In[ ]:


label_number={}
for index, label in enumerate(label_map):
    label_number[label]=index


# In[ ]:


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
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


# In[ ]:


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias
    
class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        torch.nn.init.ones_(self.weights.weight.data)
        torch.nn.init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


# In[ ]:


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        #nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        #nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            if in_channels != out_channels:
                self.bypass = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
                )
            else:
                self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        #print (x.shape)
        #print (self.model(x).shape)
        #print (self.bypass(x).shape)
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            
            #if in_channels == out_channels:
            #    self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            #else:
            #    self.bypass = nn.Sequential(
            #        spectral_norm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #        nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)


# In[ ]:


class Generator(nn.Module):
    
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.channels = channels
        
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                spectral_norm(nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)),
                nn.ReLU(inplace=True),
            ]
            return block

        self.layer1 = nn.Sequential(
            *convlayer(self.nz, 64, 4, 1, 0),
            #*convlayer(512, 64, 4, 2, 1)
        )
        
        #self.CBN1 = CategoricalConditionalBatchNorm2d(120,512)
        #self.layer2 = nn.Sequential(*convlayer(512, 256, 4, 2, 1))
        self.layer2 = nn.Sequential(ResBlockGenerator(64, 64, 2))
        self.CBN2 = CategoricalConditionalBatchNorm2d(120,64)
        self.layer3 = nn.Sequential(ResBlockGenerator(64, 64, 2))
        self.CBN3 = CategoricalConditionalBatchNorm2d(120,64)
        self.layer4 = nn.Sequential(ResBlockGenerator(64, 64, 2))
        self.CBN4 = CategoricalConditionalBatchNorm2d(120,64)
        self.layer5 = nn.Sequential(ResBlockGenerator(64, 64, 2))
        self.CBN5 = CategoricalConditionalBatchNorm2d(120,64)
        self.outlay = spectral_norm(nn.Conv2d(64, 3, 3, 1, 1))
        self.Tanh = nn.Tanh()
        

    def forward(self, z, y):
        #label = label.view(-1, 120,1,1).type(torch.cuda.FloatTensor)
        #label = self.label_conv1(label)
        z = z.view(-1, self.nz, 1, 1)
        #z = self.first_layer(z)
        z = self.layer1(z)
        #z = self.CBN1(z,y)
        z = self.layer2(z)
        z = self.CBN2(z,y)
        z = self.layer3(z)
        z = self.CBN3(z,y)
        z = self.layer4(z)
        z = self.CBN4(z,y)
        z = self.layer5(z)
        z = self.CBN5(z,y)
        z = self.outlay(z)
        img = self.Tanh(z)
        #print (img.shape)
        return img


class Discriminator(nn.Module):
    
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block =[]
            if bn:
                block.append(spectral_norm(nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)))
                block.append(nn.BatchNorm2d(n_output))
            else:
                block.append(nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False))
            #if bn:
            #    block.append(spectral_norm())
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        #self.label_conv2 = nn.Conv2d(120, 128, 4, 2, 1, bias=False)
        #self.first_layer2 = nn.Sequential(*convlayer(self.channels, 128, 4, 2, 1))

        self.model = nn.Sequential(
            *convlayer(self.channels, 64, 4, 2, 1),
            ResBlockDiscriminator(64,128,2),
            ResBlockDiscriminator(128,256,2),
            ResBlockDiscriminator(256,728,2),
            ResBlockDiscriminator(728,728,1),
            
            #*convlayer(128, 256, 4, 2, 1),
            #*convlayer(256, 512, 4, 2, 1, bn=True),
            #*convlayer(512, 1024, 4, 2, 1, bn=True),
        )
        self.logic_out = nn.Conv2d(728, 1, 4, 1, 0, bias=False)
        self.aux_out = nn.Conv2d(728, 120, 4, 1, 0, bias=False)

    def forward(self, imgs, label):
        #label = self.label_conv2(label)
        #imgs = self.first_layer2(imgs)
        out = self.model(imgs)
        logic_output = self.logic_out(out)
        #print (logic_output.shape)
        aux_output = self.aux_out(out)
        #print (aux_output.shape)
        return logic_output.view(-1, 1),aux_output.view(-1,120)


# In[ ]:


class DogsDataset(Dataset):
    
    def __init__(self, root, annotation_root, transform=None,
                 target_transform=None, loader=default_loader, n_process=4):
        self.root = Path(root)
        self.annotation_root = Path(annotation_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs= self.cut_out_dogs(n_process)

    def _get_annotation_path(self, img_path):
        dog = Path(img_path).stem
        breed = dog.split('_')[0]
        breed_dir = next(self.annotation_root.glob(f'{breed}-*'))
        return breed_dir / dog
    
    @staticmethod
    def _get_dog_box(annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        objects = root.findall('object')
        for o in objects:
            bndbox = o.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            yield (xmin, ymin, xmax, ymax)
            
    def crop_dog(self, path):
        imgs = []
        annotation_path = self._get_annotation_path(path)
        dog = Path(path).stem
        label = label_number[dog.split('_')[0]]
        #label = np.eye(120)[label]
        for bndbox in self._get_dog_box(annotation_path):
            img = self.loader(path)
            img_ = img.crop(bndbox)
            if np.sum(img_) != 0:
                img = img_
            imgs.append([img,label])
        return imgs
    
    def label_dog(self, path):
        label = []
        dog = Path(path).stem
        label.append(label_number[dog.split('_')[0]])
        return label
                
    def cut_out_dogs(self, n_process):
        with Pool(n_process) as p:
            imgs = p.map(self.crop_dog, self.root.iterdir())
            #labels = p.map(self.label_dog, self.root.iterdir())
        return imgs
    
    def __getitem__(self, index):
        #samples = random.choice(self.imgs[index])
        if self.transform is not None:
            samples = self.transform(self.imgs[index][0][0])
        lables = self.imgs[index][0][1]
        return samples,lables
    
    def __len__(self):
        return len(self.imgs)


# In[ ]:


class ParamScheduler(object):
    
    def __init__(self, optimizer, scale_fn, step_size):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
        self.optimizer = optimizer
        self.scale_fn = scale_fn
        self.step_size = step_size
        self.last_batch_iteration = 0
        
    def batch_step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scale_fn(self.last_batch_iteration / self.step_size)
        
        self.last_batch_iteration += 1


def combine_scale_functions(scale_fns, phases=None):
    if phases is None:
        phases = [1. / len(scale_fns)] * len(scale_fns)
    phases = [phase / sum(phases) for phase in phases]
    phases = torch.tensor([0] + phases)
    phases = torch.cumsum(phases, 0)
    
    def _inner(x):
        idx = (x >= phases).nonzero().max()
        actual_x = (x - phases[idx]) / (phases[idx + 1] - phases[idx])
        return scale_fns[idx](actual_x)
        
    return _inner


def scale_cos(start, end, x):
    return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2


# In[ ]:


random_transforms = [transforms.RandomRotation(degrees=5)]
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = DogsDataset(TRAIN_DIR / 'all-dogs/', ANNOTATION_DIR, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4)

#imgs = next(iter(train_loader))
#imgs = imgs.numpy().transpose(0, 2, 3, 1)


# In[ ]:


imgs,labels = next(iter(train_loader))


# In[ ]:


net_g = Generator(nz).to(device)
net_d = Discriminator().to(device)

criterion = nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
scale_fn = combine_scale_functions(
    [partial(scale_cos, 1e-5, 5e-4), partial(scale_cos, 5e-4, 1e-4)], [0.2, 0.8])

optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

#step_size = 2000
#base_lr, max_lr = lr, lr*10
scheduler_g = ParamScheduler(optimizer_g, scale_fn, epochs * len(train_loader))
scheduler_d = ParamScheduler(optimizer_d, scale_fn, epochs * len(train_loader))


# In[ ]:


for epoch in range(epochs):
    for i, (real_images,label) in enumerate(train_loader):
        # --------------------------------------
        # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        # --------------------------------------
        net_d.zero_grad()
        real_images = real_images.to(device)
        label = label.type(torch.LongTensor).to(device)
        #label_g = torch.eye(120)[label.tolist()].to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        
        #label_d = fill[label.tolist()].to(device)
        
        scheduler_d.batch_step()
        output_real,aux_real = net_d(real_images,label)
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        #y_noise = (torch.rand(batch_size, 1) * 120).type(torch.LongTensor).squeeze()
        #label_g_fake = torch.eye(120)[y_noise.tolist()].to(device)
        #label_d_fake = fill[y_noise.tolist()].to(device)
        
        fake = net_g(noise,label)
        
        output_fake,aux_fake = net_d(fake.detach(),label)
        
        loss_aux_real = auxiliary_loss(aux_real,label)
        loss_aux_fake = auxiliary_loss(aux_fake,label)
        
        err_d = (torch.mean((output_real - torch.mean(output_fake) - labels) ** 2) + 
                 torch.mean((output_fake - torch.mean(output_real) + labels) ** 2)) / 2
        
        err_d = err_d*0.7 + loss_aux_real*0.15 + loss_aux_fake*0.15
        err_d.backward(retain_graph=True)
        optimizer_d.step()
        
        # --------------------------------------
        # Update Generator network: maximize log(D(G(z)))
        # --------------------------------------
        net_g.zero_grad()
        scheduler_g.batch_step()
        output_fake, aux_fake = net_d(fake,label)
        
        loss_aux_fake_g = auxiliary_loss(aux_fake,label)
        err_g = (torch.mean((output_real - torch.mean(output_fake) + labels) ** 2) +
                 torch.mean((output_fake - torch.mean(output_real) - labels) ** 2)) / 2
        err_g = err_g*0.85 + loss_aux_fake_g*0.15
        err_g.backward()
        optimizer_g.step()
        
    print(f'[{epoch + 1}/{epochs}] Loss_d: {err_d.item():.4f} Loss_g: {err_g.item():.4f}')


# In[ ]:


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


# In[ ]:


im_batch_size = 50
n_images = 10000

for i_batch in range(0, n_images, im_batch_size):
    z = truncated_normal((im_batch_size, nz, 1, 1), threshold=1)
    gen_z = torch.from_numpy(z).float().to(device)
    gen_labels = Variable(torch.LongTensor(np.random.randint(0, 120, im_batch_size))).to(device)
    #y_noise = (torch.rand(im_batch_size, 1) * 120).type(torch.LongTensor).squeeze()
    #label_g_fake = torch.eye(120)[y_noise.tolist()].to(device)
    gen_images = (net_g(gen_z,gen_labels) + 1) / 2
    images = gen_images.to('cpu').clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], OUT_DIR / f'image_{i_batch + i_image:05d}.png')


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i, j in enumerate(images[:32]):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    plt.imshow(j)


# In[ ]:


shutil.make_archive('images', 'zip', OUT_DIR)


# In[ ]:


elapsed_time = time.time() - start_time
print(f'All process done in {int(elapsed_time // 3600)} hours {int(elapsed_time % 3600 // 60)} min.')

