#!/usr/bin/env python
# coding: utf-8

# # CGAN with historical weight averaging
# The historical averaging trick from Ian Goodfellow's paper - https://arxiv.org/pdf/1606.03498.pdf did the trick for me. I couldn't go below 47 though. 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os
from time import time
from PIL import Image
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import random
from torch.nn.utils import spectral_norm
from scipy.stats import truncnorm
from tqdm import tqdm_notebook as tqdm

batch_size = 32
start = time()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[ ]:


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def mse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err
    
def make_answer():
    good_breeds = analyse_generated_by_class()
    create_submit(good_breeds)
    
def analyse_generated_by_class(n_images=5):
    good_breeds = []
    for l in range(len(decoded_dog_labels)):
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            dog_label = torch.full((1,) , l, device=device, dtype=torch.long)
            gen_image = netG(noise, dog_label).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)
        
        d = np.round(np.sum([mse(sample[k], sample[k+1]) for k in range(len(sample)-1)])/n_images, 1)
        if d < 1.0: continue  # had mode colapse(discard)
            
        print(f"Generated breed({d}): ", decoded_dog_labels[l])
        figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = (sample[index] + 1.) / 2.
            axis.imshow(image_array)
        plt.show()
        
        good_breeds.append(l)
    return good_breeds


def create_submit(good_breeds):
    print("Creating submit")
    os.makedirs('../output_images', exist_ok=True)
    im_batch_size = 100
    n_images = 10000
    
    all_dog_labels = np.random.choice(good_breeds, size=n_images, replace=True)
    for i_batch in range(0, n_images, im_batch_size):
        noise = torch.randn(im_batch_size, nz, 1, 1, device=device)
        dog_labels = torch.from_numpy(all_dog_labels[i_batch: (i_batch+im_batch_size)]).to(device)
        gen_images = netG(noise, dog_labels)
        gen_images = (gen_images.to("cpu").clone().detach() + 1.) / 2.
        for ii, img in enumerate(gen_images):
            save_image(gen_images[ii, :, :, :], os.path.join('../output_images', f'image_{i_batch + ii:05d}.png'))
            
    import shutil
    shutil.make_archive('images', 'zip', '../output_images')  


# In[ ]:


class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf, crop_dogs=True):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples        
        self.samples, self.labels = self.load_dogs_data(directory, crop_dogs)

    def load_dogs_data(self, directory, crop_dogs):
        required_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(64),
                torchvision.transforms.CenterCrop(64),
        ])

        imgs = []
        labels = []
        paths = []
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            # Load image
            try: img = dset.folder.default_loader(path)
            except: continue
            
            # Get bounding boxes
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(
                    dirname for dirname in os.listdir('../input/annotation/Annotation/') if
                    dirname.startswith(annotation_basename.split('_')[0]))
                
            if crop_dogs:
                tree = ET.parse(os.path.join('../input/annotation/Annotation/',
                                             annotation_dirname, annotation_basename))
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    object_img = required_transforms(img.crop((xmin, ymin, xmax, ymax)))
                    imgs.append(object_img)
                    labels.append(annotation_dirname.split('-')[1].lower())

            else:
                object_img = required_transforms(img)
                imgs.append(object_img)
                labels.append(annotation_dirname.split('-')[1].lower())
            
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


get_ipython().run_cell_magic('time', '', "database = '../input/all-dogs/all-dogs/'\n\nrandom_transforms = [transforms.ColorJitter(brightness=(1,1.3), contrast=(1,1.3), saturation=0, hue=0), transforms.RandomRotation(degrees=5)]\ntransform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),\n                                transforms.RandomApply(random_transforms, p=0.5),\n                                transforms.ToTensor(),\n                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n\ntrain_data = DataGenerator(database, transform=transform, n_samples=25000, crop_dogs=True)\n\ndecoded_dog_labels = {i:breed for i, breed in enumerate(sorted(set(train_data.labels)))}\nencoded_dog_labels = {breed:i for i, breed in enumerate(sorted(set(train_data.labels)))}\ntrain_data.labels = [encoded_dog_labels[l] for l in train_data.labels] # encode dog labels in the data generator\n\n\n\ntrain_loader = torch.utils.data.DataLoader(train_data, shuffle=True,\n                                           batch_size=batch_size, num_workers=4)")


# # Gan Helpers

# In[ ]:


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
    
def show_generated_img_all(n_images=5, nz=128):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        dog_label = torch.randint(0, len(encoded_dog_labels), (1, ), device=device)
        gen_image = netG(noise, dog_label).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)
        
    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = (sample[index] + 1.) / 2.
        axis.imshow(image_array)
    plt.show()
        
# def show_generated_img():
#     noise = torch.randn(1, nz, 1, 1, device=device)
#     gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
#     gen_image = gen_image.numpy().transpose(1, 2, 0)
#     gen_image = ((gen_image+1.0)/2.0)
#     plt.imshow(gen_image)
#     plt.show()  
    
def show_generated_img(n_images=5, nz=128):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        dog_label = torch.randint(0, len(encoded_dog_labels), (1, ), device=device)
        gen_image = netG(noise, dog_label).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)
        
    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = (sample[index] + 1.) / 2.
        axis.imshow(image_array)
    plt.show()    
    
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
    def forward(self, x, alpha=1e-8):
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size,1, height, width)
        y = torch.cat([x, y], 1)
        return y    
    
    
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))  

def sndeconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))  

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))

class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# In[ ]:


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out    
    
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out        


# # Generator and Discriminator

# In[ ]:


class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels, num_classes):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        self.block1 = GenBlock(nfeats * 8, nfeats * 8, num_classes)
        # state size. (nfeats*8) x 8 x 8
        self.block2 = GenBlock(nfeats * 8, nfeats * 4, num_classes)
        # state size. (nfeats*4) x 16 x 16
        self.block3 = GenBlock(nfeats * 4, nfeats * 2, num_classes)        
        # state size. (nfeats*2) x 32 x 32
        self.block4 = GenBlock(nfeats * 2, nfeats, num_classes)
        self.bn = nn.BatchNorm2d(nfeats, eps=1e-5, momentum=0.0001, affine=True)
        # state size. nfeats x 64 x 64
        self.conv6 = spectral_norm(nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. nfeats x 64 x 64
        self.pixnorm = PixelwiseNorm()
        self.relu = nn.ReLU()
        
    def forward(self, x, labels):
        #x = F.leaky_relu(self.bn1(self.conv1(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)))
        #x = F.leaky_relu(self.bn3(self.conv3(x)))
        #x = F.leaky_relu(self.bn4(self.conv4(x)))
        #x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.conv1(x)
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        x = self.block3(x, labels)
        x = self.block4(x, labels)
        x = self.bn(x)
        x = self.relu(x)
#         x = self.pixnorm(x)
        x = self.conv6(x)
        x = torch.tanh(x)
        return x



    


# class Discriminator(nn.Module):
#     def __init__(self, nchannels, nfeats, num_classes):
#         super(Discriminator, self).__init__()

#         # input is (nchannels) x 64 x 64
#         self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
#         # state size. (nfeats) x 32 x 32
#         self.block1 = DiscBlock(nfeats, nfeats*2)
#         self.bn1 = nn.BatchNorm2d(nfeats * 2)
#         # state size. (nfeats*2) x 16 x 16
#         self.block2 = DiscBlock(nfeats*2, nfeats*4)
# #         self.bn2 = nn.BatchNorm2d(nfeats * 4)
#         # state size. (nfeats*4) x 8 x 8
#         self.block3 = DiscBlock(nfeats*4, nfeats*8)
#         # state size. (nfeats*8) x 4 x 4
#         self.block4 = DiscBlock(nfeats*8, nfeats*8)
#         # state size. (nfeats*8) x 4 x 4 
#         self.snlinear1 = snlinear(in_features=nfeats*8, out_features=1)
#         self.sn_embedding1 = sn_embedding(num_classes, nfeats*8)
        
#     def forward(self, x, labels):
#         x = self.conv1(x)
#         x = self.block1(x)
#         x = self.bn1(x)
#         x = self.block2(x)
# #         x = self.bn2(x)
#         x = self.block3(x)
#         x = self.block4(x, downsample=False)
#         x = F.relu(x)
#         x = torch.sum(x, dim=[2,3]) # n x (nfeats*8)
#         output1 = torch.squeeze(self.snlinear1(x)) # n x 1
        
#         # Projection
#         h_labels = self.sn_embedding1(labels)
#         proj = torch.mul(x, h_labels) 
#         output2 = torch.sum(proj, dim=[1])
        
#         output = output1 + output2
# #         return output, torch.sigmoid(output), 
#         return torch.sigmoid(output).view(-1, 1)

class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = sn_embedding(num_classes, 64*64)
        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels+1, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        self.block1 = DiscBlock(nfeats, nfeats*2)
        self.bn1 = nn.BatchNorm2d(nfeats * 2)
#         self.self_attn = Self_Attn(nfeats*2)
        # state size. (nfeats*2) x 16 x 16
        self.block2 = DiscBlock(nfeats*2, nfeats*4)
#         self.bn2 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
        self.block3 = DiscBlock(nfeats*4, nfeats*8)
        # state size. (nfeats*8) x 4 x 4
        self.block4 = DiscBlock(nfeats*8, nfeats*8)
#         self.downscale = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 2 x 2
#         self.batch_discriminator = MinibatchStdDev()
        # state size. (nfeats*8+1) x 2 x 2
        
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1
        
    def forward(self, imgs, labels):
        enc = self.label_emb(labels).view((-1, 1, 64, 64))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((imgs, enc), 1)
        
        
        x = self.conv1(x)
        x = self.block1(x)
        x = self.bn1(x)
#         x = self.self_attn(x)
        x = self.block2(x)
#         x = self.bn2(x)
        x = self.block3(x)
        x = self.block4(x)
#         x = self.downscale(x)
        x = F.leaky_relu(x, 0.2)
#         x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, 1)
     


# # Hyperparameters

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTUR
lr_d = 0.0006
lr_g = 0.0003
beta1 = 0.5
epochs = 300
num_classes = len(encoded_dog_labels)
netG = Generator(128, 32, 3, num_classes).to(device)
netD = Discriminator(3, 48, num_classes).to(device)

criterion = nn.BCELoss()
criterionH = nn.MSELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
# lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
#                                                                      T_0=epochs//200, eta_min=0.00005)
# lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
#                                                                      T_0=epochs//200, eta_min=0.00005)

nz = 128
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

real_label = 0.7
fake_label = 0.0
batch_size = train_loader.batch_size

def get_model_weights(net):
    average = {}
    params = dict(net.named_parameters())
    for p in params:
        average[p] = params[p].detach()    
    return average    

print(sum(p.numel() for p in netG.parameters()))
print(sum(p.numel() for p in netD.parameters()))

averageD = False
averageG = False
hist_average_cutoff = 96 #Emperically found this good
# hist_average_cutoff = -1


# # Train

# In[ ]:


fids = {}
step = 1
for epoch in range(epochs):
    if (time() - start) > 310 : #Change to 31000  
        break
   
        
    for ii, (real_images, dog_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if (time() - start) > 310 : #Change to 31000
            break
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        
#         dog_labels = torch.tensor(dog_labels, device=device)
        dog_labels = dog_labels.to(device)
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device) +  np.random.uniform(-0.1, 0.1)

        output = netD(real_images, dog_labels)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()
        
        # Historical averaging weights
        err_hD = 0
        if epoch > hist_average_cutoff:
            if not averageD:
                print("Starting historical weight averaging for discriminator")
                averageD = get_model_weights(netD)
            paramsD = dict(netD.named_parameters())
            for p in paramsD:
                err_hD += criterionH(paramsD[p], averageD[p])
                averageD[p] = (averageD[p] * (step-1) + paramsD[p].detach())/step
            err_hD.backward()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)

#         print(noise.shape)
        fake = netG(noise, dog_labels)
        labels.fill_(fake_label) + np.random.uniform(0, 0.2)
        output = netD(fake.detach(), dog_labels)
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD_final = errD_real + errD_fake + err_hD
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################   
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake, dog_labels)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
                
        err_hG = 0
        if epoch > hist_average_cutoff:
            if not averageG:
                print("Starting historical weight averaging for generator")
                averageG = get_model_weights(netG)
            paramsG = dict(netG.named_parameters())
            for p in paramsG:
                err_hG += criterionH(paramsG[p], averageG[p])
                averageG[p] = (averageG[p] * (step-1) + paramsG[p].detach())/step
            err_hG.backward()
            step += 1
        
        errG_final = errG + err_hG
        
        optimizerG.step()
        
        if ii % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader),
                     errD_final.item(), errG_final.item(), D_x, D_G_z1, D_G_z2))
            
#         lr_schedulerG.step(epoch)
#         lr_schedulerD.step(epoch)

#     if epoch % 5 == 0:
#         show_generated_img(6)


# In[ ]:


# show_generated_img_all()
make_answer()


# In[ ]:




