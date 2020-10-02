#!/usr/bin/env python
# coding: utf-8

# ### Reference
# * https://arxiv.org/abs/1610.09585  
# * https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py  
# * https://github.com/gitlimlab/ACGAN-PyTorch
# * https://github.com/kimhc6028/acgan-pytorch  
# * https://www.kaggle.com/paulorzp/show-annotations-and-breeds/notebook  
# * https://www.kaggle.com/mpalermo/pytorch-rals-c-sagan

# In[ ]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import PIL
import glob
import xml.etree.ElementTree as ET
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
from scipy.stats import truncnorm
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['image.interpolation'] = 'nearest'


# In[ ]:


print(os.listdir("../input/"))


# In[ ]:


print(os.listdir("../input/annotation/"))
print(os.listdir("../input/all-dogs/"))


# In[ ]:


anno_dir = "../input/annotation/Annotation/"
data_dir = "../input/all-dogs/all-dogs/"


# In[ ]:


# create a folder for saving extracted dogs' images
classed_input = "../classed_input/"
os.makedirs(classed_input, exist_ok=True)
print("Folder: " + classed_input + " created")


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()


# In[ ]:


# visualization random breeds
breeds = os.listdir(anno_dir)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15,15))
for indx, axis in enumerate(axes.flatten()):
    breed = np.random.choice(breeds)
    dog = np.random.choice(os.listdir(anno_dir + breed))
    img = PIL.Image.open(data_dir + dog + '.jpg') 
    tree = ET.parse(anno_dir + breed + '/' + dog)
    root = tree.getroot()
    objects = root.findall('object')
    axis.set_axis_off() 
    imgplot = axis.imshow(img)
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        axis.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # show box
        axis.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # show breed

plt.tight_layout(pad=0, w_pad=0, h_pad=0)


# In[ ]:


# visualization a breed in random
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15,15))
breed = np.random.choice(breeds)  
for indx, axis in enumerate(axes.flatten()):
    dog = np.random.choice(os.listdir(anno_dir + breed)) 
    img = PIL.Image.open(data_dir + dog + '.jpg') 
    tree = ET.parse(anno_dir + breed + '/' + dog)
    root = tree.getroot()
    objects = root.findall('object')
    axis.set_axis_off() 
    imgplot = axis.imshow(img)
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        axis.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # show box
        axis.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # show breed

plt.tight_layout(pad=0, w_pad=0, h_pad=0)


# In[ ]:


def get_all_objects(file_path):
    bbxs = []
    root = ET.parse(file_path).getroot()
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        bbxs.append([int(it.text) for it in bndbox])
    return bbxs

def get_subdir(dir):
    return sorted([name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))])

def get_files(parent_dir, sub_dir):
    return os.listdir(os.path.join(parent_dir, sub_dir))

anno_folder = get_subdir(anno_dir)
print("n_folder(n_class): {}".format(len(anno_folder)))


# In[ ]:


exceptions = {}

for subdir in tqdm(anno_folder):
    # print("Processing Directory: " + subdir)
    new_folder = os.path.join(classed_input, subdir)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    files = get_files(anno_dir, subdir)
    for f in files:
        basename = os.path.splitext(f)[0] 
        try:
            objects = get_all_objects(os.path.join(anno_dir, subdir, basename))
            for i, obj in enumerate(objects):
                xmin, ymin, xmax, ymax = obj
                image = PIL.Image.open(data_dir + basename + ".jpg")
                save_path = os.path.join(classed_input, subdir, "cropped_" + basename + str(i) + ".jpg")
                cropped = image.crop((xmin, ymin, xmax, ymax)).save(save_path, "JPEG")
                
        except Exception as e:
            exceptions[str(e)] = os.path.join(anno_dir, subdir, basename)


# In[ ]:


exceptions
# This image: n02105855_2933.jpg in "../input/all-dogs/all-dogs/" is missed. 
# But the corresponding annotation exist.


# In[ ]:


classed_dir = get_subdir(classed_input)
print(len(classed_dir)) # Good: equal to the number of annoattions folders


# In[ ]:


lengths = []
for subdir in classed_dir:
    lengths.append(len(os.listdir(os.path.join(classed_input, subdir))))
    
breeds = [classed_dir[i].split("-")[1] for i in range(len(classed_dir))] 
fig, ax = plt.subplots(figsize=(15,20))
y_pos = np.arange(len(breeds))
ax.barh(y_pos, lengths, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(breeds)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of Images')
ax.set_title("Number of Dogs' Images  by breed")
plt.tight_layout()
plt.show()


# In[ ]:


# visualization all dogs
fig = plt.figure(figsize=(20,40))
for i, dir in enumerate(classed_dir):
    ax = fig.add_subplot(20,6,i+1)
    imgs = os.listdir(os.path.join(classed_input, dir))
    # print(dir)
    img = PIL.Image.open(classed_input + "/" + dir + "/" + imgs[0])
    ax.axis('off')
    ax.set_title(breeds[i])
    ax.imshow(img, cmap="gray")
plt.tight_layout()
plt.show()    


# In[ ]:


# Setting parameters
dataroot = "../classed_input/"
workers = 4
batch_size = 32
image_size = 64
nc = 3
nz = 128
ngf = 64
ndf = 64
num_epochs = 300
lr = 0.001
beta1 = 0.5
ngpu = 1
num_show = 6
n_class = 120
outf = '../output_result'
if not os.path.exists(outf):
    os.mkdir(outf)


# In[ ]:


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


# In[ ]:


# visualization batch image
real_batch = iter(dataloader).next()
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Training Images")
image = np.transpose(vutils.make_grid(real_batch[0].to(device), normalize=True).cpu(),axes=(1,2,0))
plt.imshow(image)


# In[ ]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[ ]:


class Generator(nn.Module):

    def __init__(self, ngpu, nz=nz, ngf=ngf, nc=nc, n_class=n_class):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz+n_class, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

        self.apply(weights_init)


    def forward(self, input):

        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)

        x = self.conv5(x)
        output = self.Tanh(x)
        return output
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)


# In[ ]:


class Discriminator(nn.Module):

    def __init__(self, ngpu, ndf=ndf, nc=nc, n_class=n_class):

        super(Discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False)
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, n_class)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        c = self.softmax(c)
        s = self.disc_linear(x)
        s = self.sigmoid(s)
        return s,c

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)


# In[ ]:


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[ ]:


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def show_generated_img_all():
    dog_label = torch.randint(n_class, (64,), dtype=torch.long, device=device)
    noise = torch.randn(64, nz, 1, 1, device=device)
    gen_image = concat_noise_label(noise, dog_label, device)  
    gen_image = netG(gen_image).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
    # gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = (gen_image + 1.0) / 2.0
    
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_image):
        ax = fig.add_subplot(8, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
        

def show_generated_img(num_show):
    gen_images = []
    for _ in range(num_show):
        noise = torch.randn(1, nz, 1, 1, device=device)
        dog_label = torch.randint(0, n_class, (1, ), device=device)
        gen_image = concat_noise_label(noise, dog_label, device)
        gen_image = netG(gen_image).to("cpu").clone().detach().squeeze(0)
        # gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        gen_images.append(gen_image)
        
    fig = plt.figure(figsize=(10, 5))
    for i, gen_image in enumerate(gen_images):
        ax = fig.add_subplot(1, num_show, i + 1, xticks=[], yticks=[])
        plt.imshow(gen_image + 1 / 2)
    plt.show()
    
def show_loss(ylim): 
    sns.set_style("white")
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(G_losses,label="G",c="b")
    ax.plot(D_losses,label="D",c="r")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    if ylim == True:
        ax.set_ylim(0,4)


# In[ ]:


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def analyse_generated_by_class(n_images):
    good_breeds = []
    for l in range(n_class):
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            dog_label = l
            noise_label = concat_noise_label(noise, dog_label, device)
            gen_image = netG(noise_label).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)

        d = np.round(np.sum([mse(sample[k], sample[k + 1]) for k in range(len(sample) - 1)]) / n_images, 1,)
        if d < 1.0:
            continue  # had mode colapse(discard)
            
        print(f"Generated breed({d}): ", breeds[l])    
        good_breeds.append(l)
    return good_breeds


def create_submit(good_breeds):
    print("Creating submit")
    os.makedirs("../output_images", exist_ok=True)
    im_batch_size = 100
    n_images = 10000

    for i_batch in range(0, n_images, im_batch_size):
        z = truncated_normal((im_batch_size, nz, 1, 1), threshold=1)
        noise = torch.from_numpy(z).float().to(device)
        
        dog_label = np.random.choice(good_breeds, size=im_batch_size, replace=True) 
        dog_label = torch.from_numpy(dog_label).to(device).clone().detach().squeeze(0)
        noise_label = concat_noise_label(noise, dog_label, device)
    
        gen_images = (netG(noise_label) + 1) / 2
        
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))

    import shutil
    shutil.make_archive("images", "zip", "../output_images")


# In[ ]:


def onehot_encode(label, device, n_class=n_class):  
    eye = torch.eye(n_class, device=device) 
    return eye[label].view(-1, n_class, 1, 1)   
 
def concat_image_label(image, label, device, n_class=n_class):
    B, C, H, W = image.shape   
    oh_label = onehot_encode(label, device=device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)
 
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device=device)
    return torch.cat((noise, oh_label), dim=1)


# In[ ]:


# Loss functions
s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

r_label = 0.7
f_label = 0

input = torch.tensor([batch_size, nc, image_size, image_size], device=device)
noise = torch.tensor([batch_size, nz, 1, 1], device=device)

fixed_noise = torch.randn(1, nz, 1, 1, device=device)
fixed_label = torch.randint(0, n_class, (1, ), device=device)
fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)


# In[ ]:


# Training Loop

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(tqdm(dataloader)):
        # prepare real image and label
        real_label = data[1].to(device)
        real_image = data[0].to(device)
        b_size = real_label.size(0)      
        
        # prepare fake image and label
        fake_label = torch.randint(n_class, (b_size,), dtype=torch.long, device=device)
        noise = torch.randn(b_size, nz, 1, 1, device=device).squeeze(0)
        noise = concat_noise_label(noise, real_label, device)  
        fake_image = netG(noise)
        
        # target
        real_target = torch.full((b_size,), r_label, device=device)
        fake_target = torch.full((b_size,), f_label, device=device)
        
        #-----------------------
        # Update Discriminator
        #-----------------------
        netD.zero_grad()
        
        # train with real
        s_output, c_output = netD(real_image)
        s_errD_real = s_criterion(s_output, real_target)  # realfake
        c_errD_real = c_criterion(c_output, real_label)  # class
        errD_real = s_errD_real + c_errD_real
        errD_real.backward()
        D_x = s_output.data.mean()

        # train with fake
        s_output,c_output = netD(fake_image.detach())
        s_errD_fake = s_criterion(s_output, fake_target)  # realfake
        c_errD_fake = c_criterion(c_output, real_label)  # class
        errD_fake = s_errD_fake + c_errD_fake
        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        
        errD = s_errD_real + s_errD_fake
        optimizerD.step()        

        #-----------------------
        # Update Generator
        #-----------------------
        netG.zero_grad()
        
        s_output,c_output = netD(fake_image)
        s_errG = s_criterion(s_output, real_target)  # realfake
        c_errG = c_criterion(c_output, real_label)  # class
        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.data.mean()
        
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        iters += 1

    # scheduler.step(errD.item())
    
    print('[%d/%d][%d/%d]\nLoss_D: %.4f\tLoss_G: %.4f\nD(x): %.4f\tD(G(z)): %.4f / %.4f'
          % (epoch+1, num_epochs, i+1, len(dataloader),
             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
    show_generated_img(num_show)
    
#     # --------- save fake image  ----------
#     fake_image = netG(fixed_noise_label)   
#     vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
#                     normalize=True, nrow=5)
 
#     # ---------  save model  ----------
#     if (epoch + 1) % 10 == 0:  
#         torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))
#         torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))


# In[ ]:


show_loss(ylim=False)


# In[ ]:


show_loss(ylim=True)


# In[ ]:


good_breeds = analyse_generated_by_class(6)
create_submit(good_breeds)


# In[ ]:


len(good_breeds)


# In[ ]:


show_generated_img_all()


# In[ ]:


# visualization generate image of all breeds 
fig = plt.figure(figsize=(20,40))
for i in range(n_class):
    ax = fig.add_subplot(20,6,i+1)
    noise = torch.randn(1, nz, 1, 1, device=device)
    dog_label = i
    noise_label = concat_noise_label(noise, dog_label, device)
    gen_image = netG(noise_label).to("cpu").clone().detach().squeeze(0)
    # gen_image = gen_image.numpy().transpose(0, 2, 3, 1)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = (gen_image + 1.0) / 2.0
    ax.axis('off')
    ax.set_title(breeds[i])
    ax.imshow(gen_image, cmap="gray")
plt.tight_layout()
plt.show()    

