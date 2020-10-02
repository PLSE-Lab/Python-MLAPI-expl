#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from shutil import copyfile
copyfile(src = "../usr/lib/loggerscript/loggerscript.py", dst = "../working/utils.py")

from utils import Logger


# In[ ]:


import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch #top-level package
import torch.nn as nn #used to build NN models
import torch.optim as optim #optimization algorithms (SGD, Adam)
from torch.autograd.variable import Variable
from IPython import display

import torchvision.transforms.functional as F


# In[ ]:


objects = []
with (open("../input/taskdataset/image_features_grayscale.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
            
print(objects)


# In[ ]:


print(type(objects))
print(len(objects))


# In[ ]:


item = objects[0]
print(type(item))
print(len(item))


# In[ ]:


max_val = 0
min_val = 20
for x in item:
    tmp_max = np.amax(x)
    tmp_min = np.amin(x)
    if (tmp_max > max_val): max_val = tmp_max
    if (tmp_min < min_val): min_val = tmp_min

print('max: ', max_val)
print('min: ', min_val)


# In[ ]:


new_item = []
for img_element in item:
    img_element = ((img_element-min_val)/(max_val-min_val) - 0.5)*2
    new_item.append(img_element)

new_item.append(new_item[0])
new_item.append(new_item[1])


# In[ ]:


print(type(new_item))
print(len(new_item))


# In[ ]:


img = new_item[0]
print(type(img))
print(img.shape)


# In[ ]:


print(img)


# In[ ]:


plt.imshow(img)


# In[ ]:


img2 = new_item[1]

print(type(img2))
print(img2.shape)


# In[ ]:


plt.imshow(img2)


# In[ ]:


dataLoader = []
batch = torch.zeros(105, 55, 300)
all_batches = []
i = 0
for element in new_item:
    tensor = torch.from_numpy(element) #torch tensor containing the image
    batch[i, :, :] = tensor
    i += 1
    if(i == 105):
        all_batches.append(batch)
        batch = torch.zeros(105, 55, 300)
        i = 0


# In[ ]:


print(len(all_batches))


# In[ ]:


x = all_batches[1]
print(type(x))

print(len(x))
print(x.shape)


# In[ ]:


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)


# In[ ]:


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        # input => [105, 55, 300], we use 55 as number of channels
        self.in_layer = nn.Sequential(
                    nn.Conv1d(55, 128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.25))
        #output => [105, 128, 150]
        self.conv1 = nn.Sequential(
                    nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.25),
                    nn.BatchNorm1d(256))
        #output => [105, 256, 75]
        self.conv2 = nn.Sequential(
                    nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.25),
                    nn.BatchNorm1d(512))
        #output => [105, 512, 37]
        self.conv3 = nn.Sequential(
                    nn.Conv1d(512, 1024, kernel_size=7, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.25),
                    nn.BatchNorm1d(1024))
        #output => [105, 1024, 17]
        self.conv4 = nn.Sequential(
                    nn.Conv1d(1024, 1024, kernel_size=10, stride=2, padding=0),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.25),
                    nn.BatchNorm1d(1024))
        #output => [105, 1024, 4]
        self.out_layer = nn.Sequential(
                    nn.Conv1d(1024, 1, kernel_size=4, stride=2, padding=0),
                    nn.Sigmoid())
        #output => [105, 1, 1]
        
    def forward(self, x):
        x = self.in_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_layer(x)

        return x


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        # input => [105, 1, 55, 300], we use 55 as number of channels
        self.in_layer = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=(3, 10), stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25))
        #output => [105, 64, 28, 147]
        self.conv1 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=(3, 20), stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm2d(128))
        #output => [105, 128, 14, 65]
        self.conv2 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=(3, 20), stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm2d(256))
        #output => [105, 256, 7, 24]
        self.conv3 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=(3, 20), stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm2d(512))
        #output => [105, 512, 4, 4]
        self.out_layer = nn.Sequential(
                    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
                    nn.Sigmoid())
        
    def forward(self, x):
        x = self.in_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out_layer(x)

        return x


# In[ ]:


ngpu = 1
# Create the model
netD = Discriminator(ngpu=1).to(device)

netD.apply(weight_init)    
print(netD)


# In[ ]:


tmp_input = torch.randn(1, 55, 300, device=device)
print(tmp_input.shape)
validity_img = netD(tmp_input)
print(validity_img.shape)


# In[ ]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        n_feature = 100
        #input => [105, 100, 1]
        self.latent_to_features = nn.Sequential(
                                    nn.Linear(33, 1024),
                                    nn.ReLU())
        self.in_layer = nn.Sequential(
                        nn.ConvTranspose1d(1024, 1024, kernel_size=2, stride=1, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(1024))
        #output => [105, 1024, 4]
        self.convT1 = nn.Sequential(
                        nn.ConvTranspose1d(1024, 1024, kernel_size=2, stride=2, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(1024))
        #output => [105, 512, 12]
        self.convT2 = nn.Sequential(
                        nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(512))
        #output => [105, 256, 36]
        self.convT3 = nn.Sequential(
                        nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(512))
        #output => [105, 128, 75]
        self.convT4 = nn.Sequential(
                        nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(256))
        
        self.convT5 = nn.Sequential(
                        nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(128))
        
        self.convT6 = nn.Sequential(
                        nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm1d(64))
        #output => [105, 64, 151]
        self.out_layer = nn.Sequential(
                        nn.ConvTranspose1d(64, 55, kernel_size=2, stride=2, padding=0),
                        nn.Sigmoid())
        
    def forward(self, x):
        x = self.latent_to_features(x)
        x = x.permute(0, 2, 1)
        x = self.in_layer(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        x = self.convT5(x)
        x = self.convT6(x)
        x = self.out_layer(x)
        return x


# In[ ]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        n_feature = 100
        
        #input => [105, 100, 1, 1]
        self.in_layer = nn.Sequential(
                        nn.ConvTranspose2d(100, 1024, kernel_size=3, stride=1, padding=0),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(1024))
        #output => [105, 150, 50]
        self.convT1 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 512, kernel_size=(3, 6), stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(512))
        #output => [105, 100, 200]
        self.convT2 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, kernel_size=(2, 12), stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(256))
        #output => [105, 70, 250]
        self.convT3 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, kernel_size=(2, 20), stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(128))
        
        self.convT4 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=(2, 20), stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(64))
        self.out_layer = nn.Sequential(
                        nn.ConvTranspose2d(64, 1, kernel_size=(7, 16), stride=2, padding=1),
                        nn.Tanh())
        
    def forward(self, x):
        x = self.in_layer(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        x = self.out_layer(x)
        return x


# In[ ]:


# Create the generator
netG = Generator(ngpu=1).to(device)

netG.apply(weight_init)    
print(netG)


# In[ ]:


tmp_noise = torch.randn(1, 1, 33, device=device)
print(tmp_noise.shape)
generated_tmp_img = netG(tmp_noise)
print(generated_tmp_img.shape)


# In[ ]:


def noise(size):
    n = Variable(torch.randn(size, 1, 33))
    if torch.cuda.is_available(): return n.cuda()
    return n


# In[ ]:


test = noise(105)
print(test.shape)


# In[ ]:


#Optimization:
g_optimizer = optim.RMSprop(netG.parameters(), lr=0.00005)
d_optimizer = optim.RMSprop(netD.parameters(), lr=0.00005)


# In[ ]:


#Optimization2:
g_optimizer = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(netD.parameters(), lr=0.00002, betas=(0.5, 0.999))


# In[ ]:


print(g_optimizer)


# In[ ]:


# Number of epochs
num_epochs = 100 #we will need large number of epochs because the dataset is limited

#Loss function
#loss = nn.MSELoss()
loss = nn.BCELoss()


# In[ ]:


def real_data(size):
    #tensor containing ones
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data(size):
    #tensor containing zeros
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


# In[ ]:


def train_discriminator(optimizer, real, fake):
    #reset gardients:
    optimizer.zero_grad()
    
    #train on real data:
    prediction_real = netD(real)
    #calculate error and backpropagation:
    error_real = loss(prediction_real, real_data(real.size(0)))
    error_real.backward()
    
    #train on fake data:
    prediction_fake = netD(fake)
    #calculate error and backpropagation:
    error_fake = loss(prediction_fake, fake_data(fake.size(0)))
    error_fake.backward()
    
    #update weights with gradients
    optimizer.step()
    
    #return total error, and both prediction:
    return error_real + error_fake, prediction_real, prediction_fake


# In[ ]:


def train_generator(optimizer, fake):
    #reset gradients:
    optimizer.zero_grad()
    
    #train:
    generated_data = netD(fake)
    #calculate error and backpropagation:
    error = loss(generated_data, real_data(generated_data.size(0)))
    error.backward()
    
    #update weights:
    optimizer.step()
    
    return error


# In[ ]:


#testing samples:
test_samples = 16
test_noise = noise(test_samples)
print(test_noise.shape)


# In[ ]:


logger = Logger(model_name='VGAN', data_name='TaskDataset')
noise_size = 105


# In[ ]:


def images_to_vectors(images):
    return images.view(images.size(0), 16500)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 55, 300)


# In[ ]:


for epoch in range(11):
    for index, batch in enumerate(all_batches, 1):
        
        #train discriminator:
        
        #1/Real data:
        real = batch
        if torch.cuda.is_available(): real = real.cuda()
        #2/Fake data:
        g_input = noise(noise_size)
        fake = netG(g_input).detach()
        
        d_error, d_real, d_fake = train_discriminator(d_optimizer, real, fake)
        
        #train generator:
        #generate fake data:
        g_input = noise(noise_size)
        fake_generated = netG(g_input)
        g_error = train_generator(g_optimizer, fake_generated)
        
        #log error:
        logger.log(d_error, g_error, epoch, index, 14)
        
        display.clear_output(True)
        # Display Images
        
        #test_images = vectors_to_images(netG(test_noise)).data.cpu()
        #logger.log_images(test_images, test_samples, epoch, index, 14);
        
        # Display status Logs
        logger.display_status(
                epoch, 10, index, 14,
                d_error, g_error, d_real, d_fake
            )
        
        if(index == 3):
            test_output = noise(1)
            output = netG(test_output)
            print('prob: ', netD(output))
            
        if(index == 10):
            test_output = noise(1)
            output = netG(test_output)
            print('prob: ', netD(output))
   
print('finished')


# In[ ]:


for epoch in range(51):
    for index, batch in enumerate(all_batches, 1):
        
        #train discriminator:
        
        #1/Real data:
        real = batch
        if torch.cuda.is_available(): real = real.cuda()
        #2/Fake data:
        g_input = noise(noise_size)
        fake = netG(g_input).detach()
        
        d_error, d_real, d_fake = train_discriminator(d_optimizer, real, fake)
        
        #train generator:
        #generate fake data:
        g_input = noise(noise_size)
        fake_generated = netG(g_input)
        g_error = train_generator(g_optimizer, fake_generated)
        
        #log error:
        logger.log(d_error, g_error, epoch, index, 14)
        
        display.clear_output(True)
        # Display Images
        
        #test_images = vectors_to_images(netG(test_noise)).data.cpu()
        #logger.log_images(test_images, test_samples, epoch, index, 14);
        
        # Display status Logs
        logger.display_status(
                epoch, 10, index, 14,
                d_error, g_error, d_real, d_fake
            )
        
        if(index == 3):
            test_output = noise(1)
            output = netG(test_output)
            print('prob: ', netD(output))
            
        if(index == 10):
            test_output = noise(1)
            output = netG(test_output)
            print('prob: ', netD(output))
   
print('finished')


# In[ ]:


test_output = noise(1)
output = netG(test_output)
print(type(output))
print(len(output))


# In[ ]:



print(output.shape)


# In[ ]:


out_img = vectors_to_images(output)
print(type(out_img))


# In[ ]:


val = netD(output)
print(val)


# In[ ]:


print(out_img.shape)


# In[ ]:


tmp = torch.zeros(55, 300)
tmp = out_img[0, 0, :, :]
host = tmp.cpu()


# In[ ]:


plt.imshow(host.detach().numpy())


# In[ ]:


plt.imshow(img)


# In[ ]:


for index, batch in enumerate(all_batches, 1):
    f
        tmp = batch[0, :, :]
        tmp = torch.unsqueeze(tmp, 0)
        out = netD(tmp)
        print('prob', index, ' ', out)

