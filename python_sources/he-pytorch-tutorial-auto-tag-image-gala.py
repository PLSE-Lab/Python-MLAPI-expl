#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
from PIL import Image
from IPython.display import display

# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# TEST YOUR VERSION OF PILLOW
# Run this cell. If you see a picture of a cat you're all set!
with Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/image8247.jpg') as im:
    display(im)


# In[ ]:


dftrain=pd.read_csv('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/train.csv')
dftest=pd.read_csv('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/test.csv')
target_map={'Food':0, 'misc':1, 'Attire':2, 'Decorationandsignage':3}
# train
dftrain['Class'].unique()
dftrain['Class']=dftrain['Class'].map(target_map)


# In[ ]:


dftrain[:5000]


# In[ ]:


train=dftrain[:5000]
test=dftrain[5000:]


# In[ ]:


train['Class'].value_counts()


# In[ ]:


train.head()


# In[ ]:


# Start by creating a list
img_sizes = []
rejected = []

for item in train.Image:
    try:
        with Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/'+item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
        
print(f'Images:  {len(img_sizes)}')
print(f'Rejects: {len(rejected)}')


# In[ ]:


df_img = pd.DataFrame(img_sizes)

# Run summary statistics on image widths
df_img[0].describe(),df_img[1].describe()


# This tells us the shortest width is 80, the shortest height is 20, the largest width and height are 80 and 235, and that most images have more than 90 pixels per side. This is useful for deciding on an input size. We'll see in the next section that 224x224 will work well for our purposes (we'll take advantage of some pre-trained models that use this size!)

# ## Image Preprocessing
# Any network we define requires consistent input data. That is, the incoming image files need to have the same number of channels (3 for red/green/blue), the same depth per channel (0-255), and the same height and width. This last requirement can be tricky. How do we transform an 800x450 pixel image into one that is 224x224? In the theory lectures we covered the following:
# * <a href='https://en.wikipedia.org/wiki/Aspect_ratio_(image)'><strong>aspect ratio</strong></a>: the ratio of width to height (16:9, 1:1, etc.) An 800x450 pixel image has an aspect ration of 16:9. We can change the aspect ratio of an image by cropping it, by stretching/squeezing it, or by some combination of the two. In both cases we lose some information contained in the original. Let's say we crop 175 pixels from the left and right sides of our 800x450 image, resulting in one that's 450x450.
# * <strong>scale</strong>: Once we've attained the proper aspect ratio we may need to scale an image up or down to fit our input parameters. There are several libraries we can use to scale a 450x450 image down to 224x224 with minimal loss.
# * <a href=''><strong>normalization</strong></a>: when images are converted to tensors, the [0,255] rgb channels are loaded into range [0,1]. We can then normalize them using the generally accepted values of mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. For the curious, these values were obtained by the PyTorch team using a random 10,000 sample of <a href='http://www.image-net.org/'>ImageNet</a> images. There's a good discussion of this <a href='https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/22'>here</a>, and the original source code can be found <a href='https://github.com/soumith/imagenet-multiGPU.torch/blob/master/donkey.lua#L154'>here</a>.

# ## Transformations
# Before defining our Convolutional Network, let's look at a sample image and perform various transformations on it to see their effect.

# In[ ]:


img1=Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/image9233.jpg')
print(img1.size)
display(img1)


# This is how jupyter displays the original .jpg image. Note that size is given as (width, height).<br>
# Let's look at a single pixel:

# In[ ]:


r, g, b = img1.getpixel((0, 0))
print(r,g,b)


# The pixel at position [0,0] (upper left) of the source image has an rgb value of (90,95,98). This corresponds to <font style="background-color:rgb(90,95,98)">this color </font><br>
# Great! Now let's look at some specific transformations.
# ### <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor'><tt>transforms.ToTensor()</tt></a>
# Converts a PIL Image or numpy.ndarray (HxWxC) in the range [0, 255] to a <tt>torch.FloatTensor</tt> of shape (CxHxW) in the range [0.0, 1.0]

# In[ ]:


transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(img1)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));


# This is the same image converted to a tensor and displayed using matplotlib. Note that the torch dimensions follow [channel, height, width]<br><br>
# PyTorch automatically loads the [0,255] pixel channels to [0,1]:<br><br>
# $\frac{242}{255}=0.94\quad\frac{229}{255}=0.89\quad\frac{213}{255}=0.83$
# 

# In[ ]:


im[:,0,0]


# ### <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize'><tt>transforms.Resize(<em>size</em>)</tt></a>
# If size is a sequence like (h, w), the output size will be matched to this. If size is an integer, the smaller edge of the image will be matched to this number.<br>i.e, if height > width, then the image will be rescaled to (size * height / width, size)

# In[ ]:


transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor()
])
im = transform(img1)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));


# ### <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.CenterCrop'><tt>transforms.CenterCrop(<em>size</em>)</tt></a>
# If size is an integer instead of sequence like (h, w), a square crop of (size, size) is made.
# 
# 
# ## Other affine transformations
# An <a href='https://en.wikipedia.org/wiki/Affine_transformation'><em>affine</em></a> transformation is one that preserves points and straight lines. Examples include rotation, reflection, and scaling. For instance, we can double the effective size of our training set simply by flipping the images.
# ### <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomHorizontalFlip'><tt>transforms.RandomHorizontalFlip(<em>p=0.5</em>)</tt></a>
# Horizontally flip the given PIL image randomly with a given probability.
# 
# ### Scaling is done using <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize'><tt>transforms.Resize(<em>size</em>)</tt></a>

# In[ ]:


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),  # normally we'd set p=0.5
    transforms.RandomRotation(30),
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor()
])
im = transform(img1)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));


# ___
# ## Normalization
# Once the image has been loaded into a tensor, we can perform normalization on it. This serves to make convergence happen quicker during training. The values are somewhat arbitrary - you can use a mean of 0.5 and a standard deviation of 0.5 to convert a range of [0,1] to [-1,1], for example.<br>However, <a href='https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/22'>research has shown</a> that mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] work well in practice.
# 
# ### <a href='https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize'><tt>transforms.Normalize(<em>mean, std</em>)</tt></a>
# Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input tensor
# ### $\quad\textrm {input[channel]} = \frac{\textrm{input[channel] - mean[channel]}}{\textrm {std[channel]}}$

# In[ ]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
im = transform(img1)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));


# Recall that before normalization, the upper-leftmost tensor had pixel values of <tt>[0.3529, 0.3725, 0.3843]</tt>.<br>
# With normalization we subtract the channel mean from the input channel, then divide by the channel std.<br><br>
# $\frac{(0.3529-0.485)}{0.229}=-0.5767\quad\frac{(0.3725-0.456)}{0.224}=-0.3725\quad\frac{(0.3843-0.406)}{0.225}=-0.0964$<br>

# In[ ]:


# After normalization:
im[:,0,0]


# ### Optional: De-normalize the images
# To see the image back in its true colors, we can apply an inverse-transform to the tensor being displayed.

# In[ ]:


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));


# In[ ]:


plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));


# ## Define transforms
# In the previous section we looked at a variety of transforms available for data augmentation (rotate, flip, etc.) and normalization.<br>
# Here we'll combine the ones we want, including the <a href='https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/22'>recommended normalization parameters</a> for mean and std per channel.

# In[ ]:


train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# ## Prepare train and test sets, loaders
# 
# ### Custom data set loader

# In[ ]:


class CustomDataset(Dataset):

    def __init__(self, imgz,labels=None, root_dir='', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = imgz
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.images[idx])
        image = Image.open(img_name).convert('RGB')
#         plt.imshow(image)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


# ## Train and Validation

# In[ ]:


train_data = CustomDataset(imgz=train['Image'].values,labels=train['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/',transform=train_transform)
test_data = CustomDataset(imgz=test['Image'].values,labels=test['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/', transform=test_transform)
total_test_data = CustomDataset(imgz=dftest['Image'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Test Images/', transform=test_transform)


# In[ ]:



torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
total_test_loader = DataLoader(total_test_data, batch_size=4, shuffle=False)

class_names = train.Class.unique()

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


# ### Total Train and Test

# In[ ]:


total_train_data = CustomDataset(imgz=dftrain['Image'].values,labels=dftrain['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/',transform=train_transform)
total_train_loader = DataLoader(total_train_data, batch_size=4, shuffle=True)


# ## Display a batch of images
# To verify that the training loader selects cat and dog images at random, let's show a batch of loaded images.<br>
# Recall that imshow clips pixel values <0, so the resulting display lacks contrast. We'll apply a quick inverse transform to the input tensor so that images show their "true" colors.

# In[ ]:


train_data.__getitem__(2644)


# In[ ]:


# Grab the first batch of 10 images

target_map_inv={0:'Food', 1:'misc', 2:'Attire', 3:'Decorationandsignage'}
from torchvision.utils import make_grid
for images,labels in train_loader: 
    break

# Print the labels
print('Label:', labels.numpy())

print('Class:', *np.array([target_map_inv[i.tolist()] for i in labels]))

im = make_grid(images,nrow=8)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));


# ## Define the model
# We'll start by using a model similar to the one we applied to the CIFAR-10 dataset, except that here we have a binary classification (2 output channels, not 10). Also, we'll add another set of convolution/pooling layers.

# Simple CNN

# In[ ]:


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# <div class="alert alert-info"><strong>Why <tt>(54x54x16)</tt>?</strong><br>
# With 224 pixels per side, the kernels and pooling layers result in $\;(((224-2)/2)-2)/2 = 54.5\;$ which rounds down to 54 pixels per side.</div>

# ### Instantiate the model, define loss and optimization functions
# We're going to call our model "CNNmodel" to differentiate it from an "AlexNetmodel" we'll use later.

# In[ ]:


torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    CNNmodel = CNNmodel.cuda()
    criterion = criterion.cuda()
    
    
# optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
optimizer = torch.optim.SGD(CNNmodel.parameters(), lr=0.001, momentum=0.9)
CNNmodel


# ### Looking at the trainable parameters

# In[ ]:


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')


# In[ ]:


count_parameters(CNNmodel)


# In[ ]:


print(train.shape,test.shape)


# ## Train the model

# In[ ]:



def test_model(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if torch.cuda.is_available():
                X_test = X_test.cuda()
                y_test = y_test.cuda()
            
            inputs, labels = X_test, y_test
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    print('Accuracy of the network on test images: %0.3f %%' % (100 * correct / total))
def train_model(net, trainloader,epochs):
    for epoch in range(epochs): # no. of epochs
        running_loss = 0
#         exp_lr_scheduler.step()
        for b, (X_train, y_train) in enumerate(train_loader):
            # data pixels and labels to GPU if available
            b+=1
            
            if torch.cuda.is_available():
                X_train = X_train.cuda()
                y_train = y_train.cuda()
            inputs, labels = X_train, y_train
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss/len(trainloader)))
        
 
    print('Done Training')
    


# ## Evaluate model performance

# In[ ]:


# plt.plot(train_losses, label='training loss')
# plt.plot(test_losses, label='validation loss')
# plt.title('Loss at the end of each epoch')
# plt.legend();


# In[ ]:


# plt.plot([t for t in train_correct], label='training accuracy')
# plt.plot([t for t in test_correct], label='validation accuracy')
# plt.title('Accuracy at the end of each epoch')
# plt.legend();


# In[ ]:


# print(test_correct)
# print(f'Test accuracy: {test_correct[-1].item()*100/3000:.3f}%')


# ## Download a pretrained model
# Torchvision has a number of proven models available through <a href='https://pytorch.org/docs/stable/torchvision/models.html#classification'><tt><strong>torchvision.models</strong></tt></a>:
# <ul>
# <li><a href="https://arxiv.org/abs/1404.5997">AlexNet</a></li>
# <li><a href="https://arxiv.org/abs/1409.1556">VGG</a></li>
# <li><a href="https://arxiv.org/abs/1512.03385">ResNet</a></li>
# <li><a href="https://arxiv.org/abs/1602.07360">SqueezeNet</a></li>
# <li><a href="https://arxiv.org/abs/1608.06993">DenseNet</a></li>
# <li><a href="https://arxiv.org/abs/1512.00567">Inception</a></li>
# <li><a href="https://arxiv.org/abs/1409.4842">GoogLeNet</a></li>
# <li><a href="https://arxiv.org/abs/1807.11164">ShuffleNet</a></li>
# <li><a href="https://arxiv.org/abs/1801.04381">MobileNet</a></li>
# <li><a href="https://arxiv.org/abs/1611.05431">ResNeXt</a></li>
# </ul>
# These have all been trained on the <a href='http://www.image-net.org/'>ImageNet</a> database of images. Our only task is to reduce the output of the fully connected layers from (typically) 1000 categories to just 2.
# 
# To access the models, you can construct a model with random weights by calling its constructor:<br>
# <pre>resnet18 = models.resnet18()</pre>
# You can also obtain a pre-trained model by passing pretrained=True:<br>
# <pre>resnet18 = models.resnet18(pretrained=True)</pre>
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# 
# Feel free to investigate the different models available. Each one will be downloaded to a cache directory the first time they're accessed - from then on they'll be available locally.
# 
# For its simplicity and effectiveness, we'll use AlexNet:

# In[ ]:


from torchvision import datasets, transforms, models # add models to the list
AlexNetmodel = models.alexnet(pretrained=True)
AlexNetmodel


# In[ ]:


for param in AlexNetmodel.parameters():
    param.requires_grad = False


# In[ ]:


torch.manual_seed(42)
# AlexNetmodel.fc = nn.Sequential(nn.Linear(9216, 1024),
AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(1024, 4),
                                 nn.LogSoftmax(dim=1))

# for param in AlexNetmodel.fc.parameters():
#     param.requires_grad = True
AlexNetmodel


# In[ ]:


# These are the TRAINABLE parameters:
count_parameters(AlexNetmodel)


# In[ ]:


from torch.optim import lr_scheduler


criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    AlexNetmodel = AlexNetmodel.cuda()
    criterion = criterion.cuda()
    
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr=0.0001)
# optimizer = torch.optim.SGD(AlexNetmodel.classifier.parameters(), lr=0.0001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


torch.cuda.is_available()


# ## Train and Validates model performance

# In[ ]:


train_model(AlexNetmodel,train_loader,5)
test_model(AlexNetmodel,test_loader)


# ## Predict on Test Set

# In[ ]:


def prediciton(net, data_loader):
    test_pred = torch.LongTensor()
    for i, data in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        output = net(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    
    return test_pred



train_model(AlexNetmodel,total_train_loader,15)

test_pred = prediciton(AlexNetmodel, total_test_loader)


# In[ ]:


test_pred.numpy().shape


# In[ ]:


dftest['Class']=test_pred.numpy()
dftest.Class=dftest.Class.map(target_map_inv)


# In[ ]:


dftest.to_csv('s6_alexnet.csv',index=False)


# In[ ]:


dftest


# In[ ]:


img1=Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Test Images/image3442.jpg')
print(img1.size)
display(img1)

