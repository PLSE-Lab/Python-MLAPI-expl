#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from PIL import Image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math
from torch.nn.utils import spectral_norm

import glob

import xml.etree.ElementTree as ET

from tqdm import tqdm_notebook as tqdm

import tensorboardX as tbx
writer = tbx.SummaryWriter(".")


# In[ ]:


start_time = time.time()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# In[ ]:


if not os.path.exists('../class_images'):
    os.mkdir('../class_images')
if not os.path.exists('../class_images/train'):
    os.mkdir('../class_images/train')


# In[ ]:


# This loader will use the underlying loader plus crop the image based on the annotation
def doggo_loader(path):
    img = torchvision.datasets.folder.default_loader(path) # default loader
    
    # Get bounding box
    annotation_basename = os.path.splitext(os.path.basename(path))[0]
    annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))
    annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)
    tree = ET.parse(annotation_filename)
    root = tree.getroot()
    objects = root.findall('object')
    for idx, o in enumerate(objects):
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        l = max([w,h])

        if w > h:
            ymin = ymin - int((l-h)/2)
            #ymin = max(ymin - int((l-h)/2), 0)
            #ymax = min(ymin - int((l-h)/2) + l, img.height)
        else:
            xmin = xmin - int((l-w)/2)
            #xmin = max(xmin - int((l-w)/2), 0)
            #xmax = min(xmin - int((l-w)/2) + l, img.width)
        ##bbox = (xmin, ymin , xmax, ymax)
        bbox = (xmin, ymin , xmin+l, ymin+l)

        c_img = img.crop(bbox).copy()
        #img = img.resize((100, 100))
        c_img.thumbnail((100, 100), Image.ANTIALIAS)

        c_img.save('../class_images/train/image_{}_{}.png'.format(annotation_basename, idx))

    return img


# The dataset (example)
dataset = torchvision.datasets.ImageFolder(
    '../input/all-dogs/',
    loader=doggo_loader, # THE CUSTOM LOADER
    transform=torchvision.transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),

        torchvision.transforms.ToTensor(),
    ]) # some transformations, add your data preprocessing here
)


# In[ ]:


if not os.path.exists('../class_images'):
    os.mkdir('../class_images')
if not os.path.exists('../class_images/train'):
    os.mkdir('../class_images/train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=50,
                                         shuffle=False, num_workers=8)
for i, data in enumerate(tqdm(dataloader)):
    pass


# In[ ]:


# Initial_setting
workers = 8
batch_size=32
nz = 128
nch_g = 64
nch_d = 64
n_epoch = 20000000   
lr = 0.0002
beta1 = 0.5
outf = './result_lsgan'
display_interval = 100
save_fake_image_interval = 1500
plt.rcParams['figure.figsize'] = 10, 6
 
try:
    os.makedirs(outf, exist_ok=True)
except OSError as error: 
    print(error)
    pass
 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:    
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[ ]:


'''
class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=64, nch=3):
        super(Generator, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0),     
                nn.BatchNorm2d(nch_g * 8),
                nn.ReLU()                                       
            ),  # (100, 1, 1) -> (512, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU(),
                #nn.Dropout(p=0.5, inplace=False)
            ),  # (512, 4, 4) -> (256, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU(),
                #nn.Dropout(p=0.5, inplace=False)
            ),  # (256, 8, 8) -> (128, 16, 16)
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),  # (128, 16, 16) -> (64, 32, 32)
            # 'layer3-5': nn.Sequential(
            #    nn.ConvTranspose2d(nch_g, nch_g, 4, 2, 1),
            #    nn.BatchNorm2d(nch_g),
            #    nn.ReLU()
            #),  # (64, 32, 32) -> (64, 64, 64)
            'layer4': nn.Sequential(
                #nn.ConvTranspose2d(nch_g, nch, 3, 1, 0),
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            )   # (64, 32, 32) -> (3, 64, 64)
        })
 
    def forward(self, z):
        for layer in self.layers.values():  
            z = layer(z)
        return z
    
    def denorm(self, z):
        for layer in self.layers.values():  
            z = layer(z)   
        z = z * 0.5 + 0.5
        return z
 
 
class Discriminator(nn.Module):
    def __init__(self, nch=3, nch_d=64):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),     
                nn.LeakyReLU(negative_slope=0.2)    
            ),  # (3, 64, 64) -> (64, 32, 32)
            'layer1': nn.Sequential(
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (64, 32, 32) -> (128, 16, 16)
            'layer2': nn.Sequential(
                nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (128, 16, 16) -> (256, 8, 8)
            'layer3': nn.Sequential(
                nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 8),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (256, 8, 8) -> (512, 4, 4)
            'layer4': nn.Conv2d(nch_d * 8, 1, 4, 1, 0)
            # (512, 4, 4) -> (1, 1, 1)
        })
 
    def forward(self, x):
        for layer in self.layers.values():  
            x = layer(x)
        return x.squeeze()  
'''


# In[ ]:


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
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
    
class MinibatchStdDev(torch.nn.Module):
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
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size,1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y

class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        #self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        #self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = spectral_norm(nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        #self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = spectral_norm(nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))
        #self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = spectral_norm(nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()
    def forward(self, x):
        #x = F.leaky_relu(self.bn1(self.conv1(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)))
        #x = F.leaky_relu(self.bn3(self.conv3(x)))
        #x = F.leaky_relu(self.bn4(self.conv4(x)))
        #x = F.leaky_relu(self.bn5(self.conv5(x)))
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
    def denorm(self, z):
        #for layer in self.layers.values():  
        z = self.forward(z)   
        z = z * 0.5 + 0.5
        return z



class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        
        self.conv2 = spectral_norm(nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = spectral_norm(nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = spectral_norm(nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8 +1, 1, 2, 1, 0, bias=False))
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
        #x= self.conv5(x)
        return x.view(-1, 1)


# In[ ]:


class DogDataset(Dataset):
    """Dog Dataset."""
    
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_list = sorted(glob.glob(data_path, recursive=True))
        self.transform = transform
        #self.data_name_list = sorted(glob.glob(data_path, recursive=True))
        #self.transform = transform
        #self.data_list = [transforms.ToTensor()(Image.open(img_name)) for img_name in self.data_name_list]
        
    def __len__(self):
        return len(self.data_list)
        #return len(self.data_name_list)
    
    def __getitem__(self, idx):
        image = Image.open(self.data_list[idx])
        #image = transforms.ToPILImage()(self.data_list[idx])
        
        if self.transform:
            image = self.transform(image)
        return image


# In[ ]:


dataset = DogDataset(data_path='../class_images/train/*.png',
                              transform=transforms.Compose([
                              transforms.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(1., 1.)),
                              transforms.RandomHorizontalFlip(),
                              #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))


# In[ ]:


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))


# In[ ]:



x = next(iter(dataloader))

fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(x[:32]):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


# In[ ]:


#netG = Generator(nz=nz, nch_g=nch_g).to(device)
netG = Generator(nz=nz, nfeats=nch_g, nchannels=3).to(device)
#netG.apply(weights_init)    
print(netG)
    
#netD = Discriminator(nch_d=nch_d).to(device)
netD = Discriminator(nchannels=3, nfeats=nch_d).to(device)
#netD = nn.Sequential(netD, nn.Sigmoid()) # For DRAGAN
#netD.apply(weights_init)
print(netD)
    
criterion = nn.MSELoss()
#criterion = nn.BCELoss() # For DRAGAN
 
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  

#optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))  # For DRAGAN 
#optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))  # For DRAGAN
    
    
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)  # for save_fake_image
Loss_D_list, Loss_G_list = [], []
    
save_fake_image_count = 1


# In[ ]:


def save_fig(epoch):
    if not os.path.exists('../output_images_{}'.format(epoch)):
        os.mkdir('../output_images_{}'.format(epoch))
    im_batch_size = 50
    n_images=10000

    for i_batch in range(0, n_images, im_batch_size):
        gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
        gen_images = netG.denorm(gen_z)
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join('../output_images_{}'.format(epoch), f'image_{i_batch+i_image:05d}.png'))


    import shutil
    shutil.make_archive('images_{}'.format(epoch), 'zip', '../output_images_{}'.format(epoch))


# In[ ]:


def cosine_annealing(optimizer, start_lr, cur_steps, num_cycle):
    t_cur = cur_steps % num_cycle
    lr = 0.5 * start_lr * (math.cos(math.pi * t_cur / num_cycle) + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# In[ ]:


# For DRAGAN
def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Loss weight for gradient penalty
    lambda_gp = 10
    # Random weight term for interpolation
    alpha = torch.Tensor(np.random.random(size=X.shape)).cuda()

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).cuda()))
    interpolates = Variable(interpolates, requires_grad=True).cuda()

    d_interpolates = D(interpolates)

    fake = Variable(torch.Tensor(X.shape[0], ).fill_(1.0), requires_grad=False).cuda()

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# In[ ]:


rap_time = start_time
for epoch in range(n_epoch):
    #random.seed(0)
    #np.random.seed(0)
    #torch.manual_seed(0)
    for itr, data in tqdm(enumerate(dataloader), total = len(dataloader)):
        real_image = data.to(device)   # Real Images
        sample_size = real_image.size(0)  # The number of images
        noise = torch.randn(sample_size, nz, 1, 1, device=device)   # generate input noise 

        real_target = torch.full((sample_size,), 1., device=device)   # real target
        #real_target = torch.full((sample_size,), 0.7, device=device) + torch.rand(sample_size, device=device)/2   # real target
        fake_target = torch.full((sample_size,), 0., device=device)   # fake target
        #fake_target = torch.full((sample_size,), 0., device=device) + torch.rand(sample_size, device=device)/3.3  # fake target
        #--------  Update Discriminator  ---------
        netD.zero_grad()    # initialize gradient
 
        output = netD(real_image)   # Discriminator output for real image
        errD_real = criterion(output, real_target)  # MSELoss
        D_x = output.mean().item()  # for logging
 
        fake_image = netG(noise)    # fake images
        
        output = netD(fake_image.detach())  # Discriminator output for fake image
        errD_fake = criterion(output, fake_target)  # MSELoss
        D_G_z1 = output.mean().item()  # for logging
        
        #gradient_penalty = compute_gradient_penalty(netD, real_image.data) # For DRAGAN
 
        errD = errD_real + errD_fake    # Discriminator Loss
        #errD = errD + gradient_penalty   # For DRAGAN
        
        errD.backward()    # backward
        optimizerD.step()   # Updata Discriminator params
 
        #---------  Update Generator   ----------
        netG.zero_grad()    # initialize gradient      
        output = netD(fake_image)   # Discriminator output for fake image
        errG = criterion(output, real_target)   # MSELoss
        errG.backward()     # backward
        D_G_z2 = output.mean().item()  # for logging
 
        optimizerG.step()   # Updata Generator params
        writer.add_scalar("errD", errD.item(), epoch*len(dataloader)+itr)
        writer.add_scalar("errG", errG.item(), epoch*len(dataloader)+itr)
        writer.add_scalar("D_G_z1", D_G_z1, epoch*len(dataloader)+itr)
        writer.add_scalar("D_G_z2", D_G_z2, epoch*len(dataloader)+itr)
    writer.add_images('fake_images', fake_image[:10] * 0.5 + 0.5, global_step=epoch*len(dataloader)+itr, walltime=None, dataformats='NCHW')

    cosine_annealing(optimizerG, 0.0008, time.time() - start_time, 60*60*10)
    cosine_annealing(optimizerD, 0.0002, time.time() - start_time, 60*60*10)

    if time.time() - start_time > 60*60*8.95:
        break


# In[ ]:


netG.eval()
#netG.train()


# In[ ]:


from scipy.stats import truncnorm

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


# In[ ]:


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return torch.from_numpy(values).float()


# In[ ]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000

for i_batch in range(0, n_images, im_batch_size):
    gen_z = truncated_normal((im_batch_size, nz, 1, 1), threshold=0.5).to(device)
    gen_images = netG.denorm(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[ ]:


samples = truncated_normal((32, nz, 1, 1), threshold=0.5).to(device)
samples = netG.denorm(samples).detach().cpu().numpy().transpose(0, 2, 3, 1)

fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(samples):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    plt.imshow(img)


# In[ ]:


torch.save(netG.state_dict(), 'netG.pth')
torch.save(netD.state_dict(), 'netD.pth')

