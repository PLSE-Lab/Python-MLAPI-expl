#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from PIL import Image
import cv2

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import time
import copy
import glob
import sys
sys.setrecursionlimit(100000)  # To increase the capacity of the stack

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = get_ipython().getoutput("ldconfig -p|grep cudart.so|sed -e 's/.*\\.\\([0-9]*\\)\\.\\([0-9]*\\)$/cu\\1\\2/'")
accelerator = cuda_output[0]

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')


# In[ ]:


# Load data
train_dir = '../input/aptos2019-blindness-detection/train_images/'

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

# Split off data for validation set
train, valid = train_test_split(train, train_size=0.75, test_size=0.25, shuffle=False)


# In[ ]:


print('Number of train samples: ', train.shape[0])
print('Number of validation samples: ', valid.shape[0])
print('Number of test samples: ', test.shape[0])
display(train.head(10))


# In[ ]:


df = pd.DataFrame(train)

ClassCounts = pd.value_counts(df['diagnosis'], sort=True)
print(ClassCounts)
plt.figure(figsize=(10, 7))
ClassCounts.plot.bar(rot=0);
plt.title('Severity Counts for Training Data');


# In[ ]:


# To display 5 unique retina images from each of the 5 classes

j = 1
fig=plt.figure(figsize=(15, 15))
for class_id in sorted(train['diagnosis'].unique()):
    plot_no = j
    for i, (idx, row) in enumerate(train.loc[train['diagnosis'] == class_id].sample(5).iterrows()):
        ax = fig.add_subplot(5, 5, plot_no)
        im = Image.open(f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png")
        plt.imshow(im)
        ax.set_title(f'Label: {class_id}')
        plot_no += 5
    j += 1

plt.show()
plt.savefig("samples_viz.png")


# ## Preprocess the data

# In[ ]:


def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2
    return (ry, rx)

def subtract_gaussian_blur(img):
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
    return a * b + 128 * (1 - b)

def crop_img(img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0
        crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]
        return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img
    return new_img


# In[ ]:


def preprocess(image):
    ry, rx = estimate_radius(img)
    resize_scale = r / max(rx, ry)
    w = min(int(rx * resize_scale * 2), r * 2)
    h = min(int(ry * resize_scale * 2), r * 2)
    img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
    img = crop_img(img, h, w)
    if debug_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
    img = subtract_gaussian_blur(img)
    img = remove_outer_circle(img, 0.9, r)
    img = place_in_square(img, r, h, w)
    image = PIL.ImageOps.autocontrast(image)


# ## Dataloader

# In[ ]:


import PIL

class ImageLoader(Dataset):
    
    def __init__(self, df, datatype):
        self.datatype = datatype
        #self.labels = df['diagnosis'].values
        if self.datatype == 'train':
            self.image_files = [f'../input/aptos2019-blindness-detection/train_images/{i}.png' for i in train['id_code'].values]
            self.transform = transforms.Compose([
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 #transforms.Grayscale(num_output_channels=3),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
            self.labels = train['diagnosis'].values
        elif self.datatype == 'valid':
            self.image_files = [f'../input/aptos2019-blindness-detection/train_images/{i}.png' for i in valid['id_code'].values]
            self.transform = transforms.Compose([
                                                #transforms.Grayscale(num_output_channels=3),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
            self.labels = valid['diagnosis'].values
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
#         image = preprocess(image)

        image = self.transform(image)
        if self.datatype == 'train' or self.datatype == 'valid':
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return image, label
        elif self.datatype == 'test':
#             label = torch.tensor(self.labels[index], dtype=torch.long)
            return image


# In[ ]:


trainloader = torch.utils.data.DataLoader(ImageLoader(df=train, datatype='train'), batch_size=60, shuffle=True)
testloader = torch.utils.data.DataLoader(ImageLoader(df=valid, datatype='valid'), batch_size=60, shuffle=False)  # serving as validation set...


# In[ ]:


print("lenght of dataset:", len(train))
print("length of the loader divided into batches:", len(trainloader))
img, labels = next(iter(trainloader))
print(img.shape)


# In[ ]:


#model = models.densenet121(pretrained=False)
model = models.resnet50(pretrained=True)
model
if train_on_gpu:
    model = model.cuda()
model


# ## We don't need to execute the next cell if we unfreeze the parameters

# In[ ]:


for param in model.parameters():
    param.requires_grad = False


# # Required to re-define the last layer

# In[ ]:


from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1024, 5)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
weights = torch.tensor([2., 11., 5., 13., 12.])
weights = weights.to(device)
criterion = nn.NLLLoss(weight=weights, reduction='mean')
#criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.SGD(model.fc.parameters(), lr=0.0005, momentum=0.9)

#model = load_checkpoint('/kaggle/checkpoint.pth')

model.fc = nn.Linear(512, 5)
model.fc = fc

model.to(device)  
optimizer.state_dict()


# In[ ]:


models.AlexNet()


# models.AlexNet()

# ## To resume the training

# In[ ]:


checkpoint = {'model': fc,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, '/kaggle/checkpoint.pth')
checkpoint = torch.load('/kaggle/checkpoint.pth')
model = checkpoint['model']
model.load_state_dict(checkpoint['state_dict'])
optimizer = checkpoitn['optimizer']


# ## To save the whole model

# In[ ]:


torch.save(model, '/kaggle/checkpoint.pth')
model2 = torch.load('/kaggle/checkpoint.pth')


# ## To save only the trained weights

# In[ ]:


torch.save(model.state_dict(), '/kaggle/checkpoint.pth')


# In[ ]:


model2 = models.resnet50(pretrained=False)
model2.load_state_dict(torch.load('/kaggle/checkpoint.pth'))


# In[ ]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


# In[ ]:


get_ipython().system('ls /kaggle/working')


# In[ ]:


for i, (inputs, labels) in enumerate(trainloader):
    # Move input and label tensors to the GPU
    
    inputs, labels = inputs.to(device), labels.to(device)
    
    start = time.time()

    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i==3:
        break
        
print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")


# In[ ]:


epochs = 5
steps = 0
running_loss = 0
print_every = 10

validation_accuracy = []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    #print(logps)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    #print(ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    #print(top_class)
                    equals = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    validation_accuracy.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}... "
                  f"Validation accuracy: {accuracy/len(testloader):.3f}"
                  )
            
        running_loss = 0
        model.train()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt


# In[ ]:


plt.plot(validation_accuracy, label='Validation accuracy')
plt.legend(frameon=False)


# In[ ]:


print(test['id_code'].values)


# In[ ]:


class SubmissionLoader(Dataset):
    
    def __init__(self, df):
        self.datatype = 'test'
        self.image_files = [f'../input/aptos2019-blindness-detection/test_images/{i}.png' for i in test['id_code'].values]
        self.transform = transforms.Compose([
                                            #transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        self.id_code = test['id_code'].values

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        image = image.convert('RGB')
        image = image.resize((224, 224))
        #image = PIL.ImageOps.autocontrast(image)
        image = self.transform(image)
        id_code = self.id_code[index]
        return image, id_code


# In[ ]:


submissions = torch.utils.data.DataLoader(SubmissionLoader(df=test), batch_size=1, shuffle=False)


# In[ ]:


len(submissions)


# In[ ]:


preds = []
id_codes = []

model.eval()
with torch.no_grad():

    for i, (image, id_code) in enumerate(submissions):

        image = image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        pred = torch.squeeze(top_class).item()
        preds.append(pred)
        id_codes.append(id_code)


# In[ ]:


output = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
preds = list(map(int, preds))
output.diagnosis = preds
output.to_csv("submission.csv", index=False)


# In[ ]:


pd.read_csv('/kaggle/working/submission.csv')


# In[ ]:


freq, _ = np.histogram(output.diagnosis, density=True, bins=5)
freq

