#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp ../input/steel-my-models/efficientnet_pytorch-0.5.1.xyz /tmp/pip/cache/efficientnet_pytorch-0.5.1.tar.gz')


# In[ ]:


get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ efficientnet-pytorch')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import albumentations as albu
from albumentations.pytorch import ToTensor
import PIL
import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from torch import nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch.optim import Adam,lr_scheduler

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding, get_model_params

from tqdm import tqdm_notebook, tqdm


# ![image](https://github.com/Lexie88rus/Bengali_AI_Competition/raw/master/assets/samples.png)

# # Bengali.AI EfficientNet Starter Notebook

# Bengali.AI is a wonderful competition, especially for the beginners in deep learning. I created this simple pytorch notebook to demonstrate some code for Bengali hadwrittem symbols classification.
# 
# I tried to add some data visualization tips throughout this notebook, which help to check the inputs and outputs of the model.
# 
# Unfortunately, this code can't be used to make submissions for this competition on Kaggle, but it still can serve as a starting point to develop your own models for Bengali handwriting classification.

# ## Load Data

# Specify the path to data and load the csv files:

# In[ ]:


# setup the input data folder
DATA_PATH = '../input/bengaliai-cv19/'

# load the dataframes with labels
train_labels = pd.read_csv(DATA_PATH + 'train.csv')
test_labels = pd.read_csv(DATA_PATH + 'test.csv')
class_map = pd.read_csv(DATA_PATH + 'class_map.csv')
sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')


# Load the test and train sets:

# In[ ]:


def load_images():
    '''
    Helper function to load all train and test images
    '''
    train_list = []
    for i in range(0,4):
        train_list.append(pd.read_parquet(DATA_PATH + 'train_image_data_{}.parquet'.format(i)))
    train = pd.concat(train_list, ignore_index=True)
    
    test_list = []
    for i in range(0,4):
        test_list.append(pd.read_parquet(DATA_PATH + 'test_image_data_{}.parquet'.format(i)))
    test = pd.concat(test_list, ignore_index=True)
    
    return train, test


# In[ ]:


train, test = load_images()


# ## Image Preprocessing and Data Augmentation

# Preprocessing and data augmentation are exteremely important for the training of deep learning models. I use the adaptive thresholding to binarize the input images and a simple data augmentation pipeline consisting of random crop-resize and slight rotation of the input images.

# In[ ]:


# setup image hight and width
HEIGHT = 137
WIDTH = 236

def threshold_image(img):
    '''
    Helper function for thresholding the images
    '''
    gray = PIL.Image.fromarray(np.uint8(img), 'L')
    ret,th = cv.threshold(np.array(gray),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th

def train_transforms(p=.5):
    '''
    Function returns the training pipeline of augmentations
    '''
    return albu.Compose([
        # compose the random cropping and random rotation
        albu.RandomSizedCrop(min_max_height=(int(HEIGHT // 1.1), HEIGHT), height = HEIGHT, width = WIDTH, p=p),
        albu.Rotate(limit=5, p=p),
        albu.Resize(height = HEIGHT, width = WIDTH)
    ], p=1.0)

def valid_transforms():
    '''
    Function returns the training pipeline of augmentations
    '''
    return albu.Compose([
        # compose the random cropping and random rotation
        albu.Resize(height = HEIGHT, width = WIDTH)
    ], p=1.0)


# ## Define the Dataset

# The next step is to create a custom pytorch dataset, which will produce images and corresponding labels out of the traing dataset:

# In[ ]:


'''
Helper functions to retrieve the images from the dataset in training and validation modes
'''

def get_image(idx, df, labels):
    '''
    Helper function to get the image and label from the training set
    '''
    # get the image id by idx
    image_id = df.iloc[idx].image_id
    # get the image by id
    img = df[df.image_id == image_id].values[:, 1:].reshape(HEIGHT, WIDTH).astype(float)
    # get the labels
    row = labels[labels.image_id == image_id]
    
    # return labels as tuple
    labels = row['grapheme_root'].values[0],     row['vowel_diacritic'].values[0],     row['consonant_diacritic'].values[0]
    
    return img, labels

def get_validation(idx, df):
    '''
    Helper function to get the validation image and image_id from the test set
    '''
    # get the image id by idx
    image_id = df.iloc[idx].image_id
    # get the image by id
    img = df[df.image_id == image_id].values[:, 1:].reshape(HEIGHT, WIDTH).astype(float)
    return img, image_id


# In[ ]:


class BengaliDataset(Dataset):
    '''
    Create a custom Bengali images dataset
    '''
    def __init__(self, df_images, transforms, df_labels = None, validation = False):
        '''
        Init function
        INPUT:
            df_images - dataframe with the images
            transforms - data transforms
            df_labels - datafrane containing the target labels
            validation - flag indication if the dataset is for training or for validation
        '''
        self.df_images = df_images
        self.df_labels = df_labels
        self.transforms = transforms
        self.validation = validation

    def __len__(self):
        return len(self.df_images)

    def __getitem__(self, idx):
        if not self.validation:
            # get the image
            img, label = get_image(idx, self.df_images, self.df_labels)
            # threshold the image
            img = threshold_image(img)
            # transform the image
            aug = self.transforms(image = img)
            return TF.to_tensor(aug['image']), label
        else:
            # get the image
            img, image_id = get_validation(idx, self.df_images)
            # threshold the image
            img = threshold_image(img)
            # transform the image
            aug = self.transforms(image = img)
            # return transformed image and corresponding image_id (instead of label) to create submission
            return TF.to_tensor(aug['image']), image_id


# Now let's check that everything is correct. Let's try to retrieve couple of images from the dataset:

# In[ ]:


# initialize train dataset
train_dataset = BengaliDataset(train, train_transforms(), train_labels)
# create a sample trainloader
sample_trainloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)


# In[ ]:


# plot sample train data
for img, labels in sample_trainloader:
    
    fig, axs = plt.subplots(1, img.shape[0], figsize=(15,10))
    for i in range(0, img.shape[0]):
        axs[i].imshow(TF.to_pil_image(img[i].reshape(HEIGHT, WIDTH)), cmap='gray')
        
        prop = FontProperties()
        prop.set_file('../input/bengaliaiutils/kalpurush.ttf')
        grapheme_root = class_map[(class_map.component_type == 'grapheme_root')                                   & (class_map.label == int(labels[0][i]))].component.values[0]
        
        vowel_diacritic = class_map[(class_map.component_type == 'vowel_diacritic')                                   & (class_map.label == int(labels[1][i]))].component.values[0]
        
        consonant_diacritic = class_map[(class_map.component_type == 'consonant_diacritic')                                   & (class_map.label == int(labels[2][i]))].component.values[0]
        
        axs[i].set_title('{}, {}, {}'.format(grapheme_root, vowel_diacritic, consonant_diacritic), 
                         fontproperties=prop, fontsize=20)
    break;


# ## Define the Model

# In[ ]:


# initialize the efficent net model
efficientnet_b0 = EfficientNet.from_name('efficientnet-b0')


# In[ ]:


'''
Custom model for the Bengali images
backbone model may be replaced with any other archirtecture :)
'''

class BengaliModel(nn.Module):
    '''
    Custom model for Bengali classification
    '''
    def __init__(self, backbone_model):
        super(BengaliModel, self).__init__()
        # the initial layer to convolve into 3 channels
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        # run through the efficientnet
        self.backbone_model = backbone_model
        # linear layers to produce the output labels
        self.fc1 = nn.Linear(in_features=1000, out_features=168) # grapheme_root
        self.fc2 = nn.Linear(in_features=1000, out_features=11) # vowel_diacritic
        self.fc3 = nn.Linear(in_features=1000, out_features=7) # consonant_diacritic
        
    def forward(self, x):
        # pass through the backbone model
        y = self.conv(x)
        y = self.backbone_model(y)
        
        # multi-output
        grapheme_root = self.fc1(y)
        vowel_diacritic = self.fc2(y)
        consonant_diacritic = self.fc3(y)
        
        return grapheme_root, vowel_diacritic, consonant_diacritic


# In[ ]:


# initialize the final model
model = BengaliModel(efficientnet_b0)


# ## Train the Model

# Now we are all set for the modelling:
# 
# First, let's start with defining the hyperparameters. In this notebook I won't be actually training the model, that is why the number of epochs is 0. I trained the model on my own machine and will just load the weights here.

# In[ ]:


test_split = 0.2
batch_size = 128
epochs = 0 # change this value to actually train the model
learning_rate = 0.001
num_workers = 0


# Setup the dataset and samplers:

# In[ ]:


dataset_size = len(train_dataset)

# split the dataset into test and train
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
testloader = DataLoader(train_dataset, batch_size=32, sampler=test_sampler, num_workers=num_workers)


# Optimizer and loss function:

# In[ ]:


# set loss function
criterion = nn.CrossEntropyLoss()

# set optimizer, only train the classifier parameters, feature parameters are frozen
optimizer = Adam(model.parameters(), lr=learning_rate)


# Define a training device:

# In[ ]:


# setup training device
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device


# Setup the logging. I will write the log into pandas DataFrame.

# In[ ]:


train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy'
                                      ,'Test loss', 'Test accuracy'])


# Now we are ready to train! 

# In[ ]:


# move the model to the training device
model = model.to(device)


# In[ ]:


# I'm just loading the weights instead of training
state = torch.load('../input/bengaliaiutils/efficientnet_b0_10.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])


# Training loop:

# In[ ]:


def get_accuracy(ps, labels):
    '''
    Helper function to calculate the accuracy given the labels and the output of the model
    '''
    ps = torch.exp(ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    return accuracy

steps = 0
running_loss = 0
for epoch in range(epochs):
    
    since = time.time()
    
    train_accuracy = 0
    top3_train_accuracy = 0 
    for inputs, labels in tqdm(trainloader):
        steps += 1
        # move input and label tensors to the default device
        inputs, labels = inputs.to(device), [label.to(device) for label in labels]
        
        optimizer.zero_grad()
        
        # forward pass
        grapheme_root, vowel_diacritic, consonant_diacritic  = model.forward(inputs)
        
        # calculate the loss
        loss = criterion(grapheme_root, labels[0]) + criterion(vowel_diacritic, labels[1]) +         criterion(consonant_diacritic, labels[2])
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # get the average accuracy
        train_accuracy += (get_accuracy(grapheme_root, labels[0]) + get_accuracy(vowel_diacritic, labels[1]) +                            get_accuracy(consonant_diacritic, labels[2])) / 3.0

        
    time_elapsed = time.time() - since
    
    test_loss = 0
    test_accuracy = 0
    model.eval()
    # run validation on the test set
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), [label.to(device) for label in labels]
            
            grapheme_root, vowel_diacritic, consonant_diacritic  = model.forward(inputs)
            batch_loss = criterion(grapheme_root, labels[0]) + criterion(vowel_diacritic, labels[1]) + criterion(consonant_diacritic, labels[2])
        
            test_loss += batch_loss.item()

            # Calculate test top-1 accuracy
            test_accuracy += (get_accuracy(grapheme_root, labels[0]) + get_accuracy(vowel_diacritic, labels[1]) +                            get_accuracy(consonant_diacritic, labels[2])) / 3.0
    
    # print out the training stats
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Time per epoch: {time_elapsed:.4f}.. "
          f"Average time per step: {time_elapsed/len(trainloader):.4f}.. "
          f"Train loss: {running_loss/len(trainloader):.4f}.. "
          f"Train accuracy: {train_accuracy/len(trainloader):.4f}.. "
          f"Test loss: {test_loss/len(testloader):.4f}.. "
          f"Test accuracy: {test_accuracy/len(testloader):.4f}.. ")

    # write to the training log
    train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':time_elapsed, 'Avg time per step': time_elapsed/len(trainloader), 'Train loss' : running_loss/len(trainloader),
                                      'Train accuracy': train_accuracy/len(trainloader),'Test loss' : test_loss/len(testloader),
                                      'Test accuracy': test_accuracy/len(testloader)}, ignore_index=True)

    running_loss = 0
    steps = 0
    model.train()


# ## Analyze Results

# Now we can look at the training results:

# In[ ]:


# I load my training stats from the local machine :)
train_stats = pd.read_csv('../input/bengaliaiutils/efficientnet_b0_train_stats_10.csv')


# In[ ]:


# plot the loss
plt.plot(train_stats['Train loss'], label='train')
plt.plot(train_stats['Test loss'], label='test')
plt.title('Loss over epoch')
plt.legend()
plt.show()


# In[ ]:


# plot the accuracy
plt.plot(train_stats['Train accuracy'], label='train')
plt.plot(train_stats['Test accuracy'], label='test')
plt.title('Accuracy over epoch')
plt.legend()
plt.show()


# Let's also visualize some sample predictions from the train set:

# In[ ]:


# plot sample train data
model.eval()
for img, labels in testloader:
    img, labels = img.to(device), [label.to(device) for label in labels]
    grapheme_root, vowel_diacritic, consonant_diacritic  = model.forward(img)
    
    img = img.cpu()
    grapheme_root = grapheme_root.cpu()
    vowel_diacritic = vowel_diacritic.cpu()
    consonant_diacritic = consonant_diacritic.cpu()
    
    # visualize the inputs
    fig, axs = plt.subplots(4, 1, figsize=(10,15))
    for i in range(0, img.shape[0]):
        axs[0].imshow(TF.to_pil_image(img[i].reshape(HEIGHT, WIDTH)), cmap='gray')
        
        prop = FontProperties()
        prop.set_file('../input/bengaliaiutils/kalpurush.ttf')
        grapheme_root_str = class_map[(class_map.component_type == 'grapheme_root')                                   & (class_map.label == int(labels[0][i]))].component.values[0]
        
        vowel_diacritic_str = class_map[(class_map.component_type == 'vowel_diacritic')                                   & (class_map.label == int(labels[1][i]))].component.values[0]
        
        consonant_diacritic_str = class_map[(class_map.component_type == 'consonant_diacritic')                                   & (class_map.label == int(labels[2][i]))].component.values[0]
        
        axs[0].set_title('{}, {}, {}'.format(grapheme_root_str, vowel_diacritic_str, consonant_diacritic_str), 
                         fontproperties=prop, fontsize=20)
        
        # analyze grapheme root prediction
        ps_root = F.softmax(grapheme_root[i])
        top10_p, top10_class = ps_root.topk(10, dim=0)
        
        top10_p = top10_p.detach().numpy()
        top10_class = top10_class.detach().numpy()
        
        axs[1].bar(range(len(top10_p)), top10_p)
        axs[1].set_xticks(range(len(top10_p)))
        axs[1].set_xticklabels(top10_class)
        axs[1].set_title('grapheme_root: {}'.format(labels[0][i]))
        
        # analyze vowel prediction
        ps_vowel = F.softmax(vowel_diacritic[i])
        top11_p, top11_class = ps_vowel.topk(11, dim=0)
        
        top11_p = top11_p.detach().numpy()
        top11_class = top11_class.detach().numpy()
        
        axs[2].bar(range(len(top11_p)), top11_p)
        axs[2].set_xticks(range(len(top11_p)))
        axs[2].set_xticklabels(top11_class)
        axs[2].set_title('vowel_diacritic: {}'.format(labels[1][i]))
        
        # analyze consonant prediction
        ps_cons = F.softmax(consonant_diacritic[i])
        top7_p, top7_class = ps_cons.topk(7, dim=0)
        
        top7_p = top7_p.detach().numpy()
        top7_class = top7_class.detach().numpy()
        
        axs[3].bar(range(len(top7_p)), top7_p)
        axs[3].set_xticks(range(len(top7_p)))
        axs[3].set_xticklabels(top7_class)
        axs[3].set_title('consonant_diacritic: {}'.format(labels[2][i]))
        
        plt.show()
        break;
        
    break;


# ## Create Submission

# The last step is to create a submission:

# In[ ]:


# initialize train dataset
test_dataset = BengaliDataset(test, valid_transforms(), test_labels, validation = True)
sample_validloader = DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=0)


# Take a look at the images from validation set:

# In[ ]:


# plot sample train data
for img, image_ids in sample_validloader:
    fig, axs = plt.subplots(1, img.shape[0], figsize=(15,10))
    for i in range(0, img.shape[0]):
        axs[i].imshow(TF.to_pil_image(img[i].reshape(HEIGHT, WIDTH)), cmap='gray')
        axs[i].set_title(image_ids[i])
    break;


# Create the submission:

# In[ ]:


validloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


# In[ ]:


def get_predicted_label(ps):
    '''
    Helper function to get the predicted label given the probabilities from the model output
    '''
    ps = F.softmax(ps)[0]
    top_p, top_class = ps.topk(1, dim=0)
        
    top_p = top_p.detach().numpy()
    top_class = top_class.detach().numpy()
    
    return top_class[0]


# In[ ]:


# create the submission
# initialize the dataframe
submission = pd.DataFrame(columns=['row_id', 'target'])

for imgs, image_ids in validloader:
    img = imgs[0]
    image_id = image_ids[0]
    
    imgs = imgs.to(device)
    
    # forward pass to get the output
    grapheme_root, vowel_diacritic, consonant_diacritic  = model.forward(imgs)
    
    imgs = imgs.cpu()
    grapheme_root = grapheme_root.cpu()
    vowel_diacritic = vowel_diacritic.cpu()
    consonant_diacritic = consonant_diacritic.cpu()
    
    # get the predicted labels
    grapheme_root_label = get_predicted_label(grapheme_root)
    vowel_diacritic_label = get_predicted_label(vowel_diacritic)
    consonant_diacritic_label = get_predicted_label(consonant_diacritic)
    
    # add the results to the dataframe
    submission = submission.append({'row_id':str(image_id)+'_grapheme_root', 'target':grapheme_root_label}, 
                                   ignore_index=True)
    submission = submission.append({'row_id':str(image_id)+'_vowel_diacritic', 'target':vowel_diacritic_label}, 
                                   ignore_index=True)
    submission = submission.append({'row_id':str(image_id)+'_consonant_diacritic', 'target':consonant_diacritic_label}, 
                                   ignore_index=True)


# Look at the submission file:

# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv')


# ## Conclusion
# 
# In this notebook I created and trained a sample model. This code can't be used for the actual predcitions for the competition. It requires a lot of optiomization, but you can use it as a sample for learning purposes.
# 
# ## References
# 1. [EfficientNet paper](https://arxiv.org/pdf/1905.11946.pdf)
# 2. [efficientnet-pytorch pacckage](https://pypi.org/project/efficientnet-pytorch/)
# 3. [My EDA notebook for Bengali.AI](https://www.kaggle.com/aleksandradeis/bengali-ai-eda)
