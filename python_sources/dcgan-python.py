# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/img_align_celeba"))
#Root directory for dataset
base_dir = "../input/img_align_celeba/img_align_celeba"

#from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils

#from PIL import Image

import glob
import cv2

#set random seed
seed = 999
print("Random Seed: ",seed)
random.seed(seed)
torch.manual_seed(seed)

workers = 2     #number of workers for dataloader
batch_size = 128
image_size = 64
nc = 3      #number of channels in the training images.
nz = 100    #size of generator input, size of z latent vector
ngf = 64    #size of feature maps in generator
ndf = 64    #size of feature maps in discriminator
num_epochs = 5  #number of training epochs
lr = 0.0002
beta1 = 0.5 #beta1 hyperparam for Adam optimizers
ngpu = 1    #number of GPUs available. Use 0 for CPU mode

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

class CellsDataset(Dataset):
    def __init__(self,data_root,transforms):
        data_image = glob.glob(data_root+'/*.jpg')
        self.data_image = data_image
        self.transforms = transforms
        
    def __getitem__(self,index):
        data_image_path = self.data_image[index]
        img = cv2.imread(data_image_path,-1)
        b,g,r = cv2.split(img)
        image_data = cv2.merge([r,g,b])
        if self.transforms:
            image_data = self.transforms(image_data)
        return image_data
    
    def __len__(self):
        return len(self.data_image)

dataset = CellsDataset(base_dir,transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

#plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64],padding=2,normalize=True).cpu(),(1,2,0)))
plt.savefig("training_image.jpg")

#custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

#generator
class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is Z,going into a convolution
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
            nn.Tanh()
        )
    def forward(self,input):
        return self.main(input)

#create the generator
netG = Generator(ngpu).to(device)

#handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
            )
    def forward(self,input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu>1):
    netD = nn.DataParallel(netD,list(range(ngpu)))
    
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()
#create batch of latent vectors that we will use to visualize
#the progression of the generator
fixed_noise = torch.randn(64, nz,1,1,device=device)

#Establish convention for real and fake labels during training
real_label = 1
fake_label = 0
#Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))

#Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
#For each epoch
for epoch in range(num_epochs):
    #For each batch in the dataloader
    for i , data in enumerate(dataloader,0):
        ################
        #updta D network:maximize log(D(x) + log(1-D(G(z))))
        ################
        ##train with all-real batch
        netD.zero_grad()
        #format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label,device=device)
        #forward pass real batch through D
        output = netD(real_cpu).view(-1)
        #Calculate loss on all-real batch
        errD_real = criterion(output,label)
        #calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        
        ##train with all-fake batch
        #generate batch of latent vectors
        noise = torch.randn(b_size,nz,1,1,device=device)
        #generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        #classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ##########
        #update G network: maximize log(D(G(z)))
        ##########
        netG.zero_grad()
        label.fill_(real_label)
        #since we just updated D,perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        errG = criterion(output,label)
        #calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        #update G
        optimizerG.step()
        
        #output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)):%.4f / %.4f'
                    %(epoch, num_epochs, i, len(dataloader),errD.item(),errG.item(),D_x,D_G_z1,D_G_z2))
        #save losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        #check how the generator is doing by saving G's output on fixed_noise
        if (iters%500==0) or ((epoch == num_epochs-1) and (i==len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
        iters += 1
        
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("loss.jpg")

real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
plt.savefig("fake_images.jpg")