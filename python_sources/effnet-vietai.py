#!/usr/bin/env python
# coding: utf-8

# If you run on Google Colab, run the code below to download the dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
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


# Download image and class labels
get_ipython().system('wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/img.jpg')
get_ipython().system('wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/labels_map.txt')
# Get EfficientNet PyTorch
get_ipython().system('pip install efficientnet_pytorch')


# Import libraries

# In[ ]:


import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim

import torchvision.transforms as transforms
import torchvision

from fastprogress import master_bar, progress_bar

from PIL import Image


# # Read dataset

# In[ ]:


data = pd.read_csv("../input/vietai-advance-course-retinal-disease-detection/train.csv")
data.head()


# # Data analyzing

# ## Analyze distribution of 0 and 1 for each label

# In[ ]:


for label in data.columns[1:]:
    print("Distribution of", label)
    print(data[label].value_counts())


# As can be observed, the number of label 0 is much more larger than label 1

# ## Analyze combination of classes

# In[ ]:


LABELS = data.columns[1:]
def build_label(row):
    return ",".join([LABELS[idx] for idx, val in enumerate(row[1:]) if val == 1])
        
data.apply(lambda x: build_label(x), axis=1).value_counts()


# In[ ]:


len(LABELS)


# As we can see, **opacity**, **normal** and **glaucoma** are diseases that share largest proportions in label distribution. The other diseases or combinations just account for small pieces.

# # Observations on the dataset
# The dataset provided is extremely imbalanced. In this baseline model, by simply train the model the original dataset, we will easily get overfitting on the training set and the score on the test set is very low. With the proposed methods below, you will tweak the training process and improve the metric score on the test set:
# - **Image Augmentation**: By augmenting images, we will have more data and make the training set become more regularize. [imgaug](https://github.com/aleju/imgaug) is a very strong augmentation library that you can use in this assignment
# - **Data sampling**: the idea here is to make the distribution between classes in the dataset balance. There are 2 kinds: oversampling and undersampling
# - **Adjust loss function**: the current loss function becomes very small after several epochs. By adding weights, we adjust the loss function to make it suitable for this imbalanced dataset. You can check the [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss) and try applying it.

# # Split the dataset

# For the data provided, we will split the dataset to 80% for training and 20% for validation

# In[ ]:


train_data, val_data = train_test_split(data, test_size=0.2, random_state=2019)


# **To simplify the baseline model, the dataset is splited randomly. However, to improve the model, cross-validation techniques can be applied here**

# # Build and train baseline model

# In this notebook, we will use Pytorch library to implement and train ResNet50 as a baseline model. With initial weights from ImageNet, we will retrain all layers for this problem.

# In[ ]:


IMAGE_SIZE = 224                              # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
BATCH_SIZE = 128                              
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 4           # Parameter used for reducing learning rate
MAX_EPOCHS = 50                              # Maximum number of training epochs


############
USE_BCELOGIT = True
USE_NOTNORMAL_P = False
USE_AUGMENTATION = True
USE_WEIGHT = True
MIXUP = True

# LOSS WEIGHT
pos_w = None
w = None
if USE_WEIGHT:
    pos_w=[1.3, 1.6, 4, 5.6, 5.2, 6, 5.6]
    w = [3., 1, 1, 1, 1,1, 1]


# ## Note on val results:
# - Clean data + BCELogit + MixUp + Loss Weight(pos_w\[0\]=1.3) = 0.78 
# - Unclean data + Augmentation + BCELogit + Loss Weight= 0.79 - Submission 09
# - Unclean data + BCELogit + MixUp + Loss Weight = 0.76
# - Clean data = 0.7546048370816062
# - Clean data + BCELogit = 0.7438
# - Clean data + Loss Weight = 0.7389
# - Clean data  + Not Normal P + Weight + MixUp = 0.721
# - Clean data  + Use_BCELogit + Weight +MixUp + Augment = 0.81
# - Like above + EffNetB0 = 0.8475 -> Tune thres = 0.8489
# - +Blur Augment = 0.8479

# In[ ]:





# # album image aug

# In[ ]:


get_ipython().system('pip install albumentations > /dev/null')


# In[ ]:



from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Resize, GaussianBlur, Normalize
)
from albumentations.pytorch import ToTensor
train_aug = Compose([
    Resize(IMAGE_SIZE, IMAGE_SIZE),
    ShiftScaleRotate(shift_limit=(-0.3, 0.3), scale_limit=0.3, rotate_limit=30, p=0.6),
    HorizontalFlip(p=0.5),

    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensor()
    
])
val_aug = Compose([
    Resize(IMAGE_SIZE, IMAGE_SIZE),
    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensor()

])


# ## Implement Dataset loader
# In Pytorch, you need to subclass the `Dataset` of Pytorch to custom the data loading process. The **Image Augmentation** would be executed in this subclass

# In[ ]:


class FundusDataset(Dataset):
    
    def __init__(self, folder_dir, dataframe, image_size, normalization, is_train=False):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        self.image_paths = [] # List of image paths
        self.image_labels = [] # List of image labels
        self.is_train = is_train
        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size, image_size))
        ]
        
        if is_train:
            image_transformation.append(
                torchvision.transforms.RandomAffine(30, translate=(0.1, 0.2), scale=(0.8, 1.2))
            )
            image_transformation.append(
                torchvision.transforms.RandomHorizontalFlip(p=0.5)
            )
        
        image_transformation.append(transforms.ToTensor())
        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        
        #self.image_transformation = transforms.Compose(image_transformation)
        
        if self.is_train:
            self.image_transformation = train_aug
        else:
            self.image_transformation = val_aug
        
        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row.filename)
            self.image_paths.append(image_path)
            self.image_labels.append(row[1:])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels
        
        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor 
        #image_data = self.image_transformation(image_data)
        
        image_data = np.array(image_data)
        image_data = self.image_transformation(image=image_data)['image']
        
        return image_data, torch.FloatTensor(self.image_labels[index])


# Create training dataset

# In[ ]:


train_dataset = FundusDataset("../input/cleandata-vietai-caothang/cleaned_data/cleaned_data/train", train_data, IMAGE_SIZE, True, USE_AUGMENTATION)


# Besides, `DataLoader` also need to be created. For the training data loader, we need to shuffle the dataset.

# In[ ]:


train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


# Let's check the size of data and label for each iteration

# In[ ]:


for data, label in train_dataloader:
    print(data.size())
    print(label.size())
    break


# We also need to create validation dataloader. Different from training dataloader, we don't shuffle the validation set

# In[ ]:


val_dataset = FundusDataset("../input/cleandata-vietai-caothang/cleaned_data/cleaned_data/train", val_data, IMAGE_SIZE, True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ## Define model
# In the baseline, we use ResNet50 pretrained on ImageNet dataset. The classifier of model would be replaced with a new dense layer to make the output suit the problem.

# In[ ]:


class ResNet50(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        """
        Init model architecture
        
        Parameters
        ----------
        num_classes: int
            number of classes
        is_trained: bool
            whether using pretrained model from ImageNet or not
        """
        super().__init__()
        
        # Load the resnet50 from ImageNet
        self.net = torchvision.models.resnet50(pretrained=is_trained)
        
        # Get the input dimension of last layer
        kernel_count = self.net.fc.in_features
        
        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        if not USE_BCELOGIT:
            self.net.fc = nn.Sequential(nn.Linear(kernel_count, num_classes)
                                       ,nn.Sigmoid()
                                       )
        else:
            self.net.fc = nn.Sequential(nn.Linear(kernel_count, num_classes)
                                       #,nn.Sigmoid()
                                       )
        
    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        output = self.net(inputs)
        if USE_NOTNORMAL_P:
            not_normal_p = 1 - output[:, 0:1]
            output = torch.cat([output[:, 0:1], output[:, 1:] * not_normal_p], dim=1)
        return output


# In[ ]:


## Efficient net
from efficientnet_pytorch import EfficientNet
model_name = 'efficientnet-b0'

class EffNet(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        """
        Init model architecture
        
        Parameters
        ----------
        num_classes: int
            number of classes
        is_trained: bool
            whether using pretrained model from ImageNet or not
        """
        super().__init__()
        
        # Load the resnet50 from ImageNet
        self.net = EfficientNet.from_pretrained(model_name)

        
        # Get the input dimension of last layer
        kernel_count = self.net._fc.in_features
        
        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        if not USE_BCELOGIT:
            self.net._fc = nn.Sequential(nn.Linear(kernel_count, num_classes)
                                       ,nn.Sigmoid()
                                       )
        else:
            self.net._fc = nn.Sequential(nn.Linear(kernel_count, num_classes)
                                       #,nn.Sigmoid()
                                       )
        
    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        output = self.net(inputs)
        if USE_NOTNORMAL_P:
            not_normal_p = 1 - output[:, 0:1]
            output = torch.cat([output[:, 0:1], output[:, 1:] * not_normal_p], dim=1)
        return output


# ## Check GPU available

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# ## Create model and get number of trainable parameters

# Create model and check model architecture

# In[ ]:


model = EffNet(num_classes=len(LABELS)).to(device)
model


# Number of trainable parameters

# In[ ]:


sum(p.numel() for p in model.parameters() if p.requires_grad)


# We need to train about 23 millions parameters

# ## Define loss function, optimizer, and learning rate scheduler

# In[ ]:


# Loss function
if pos_w is not None:
    pos_w = torch.tensor(pos_w, device=device)
if w is not None:
    w = torch.tensor(w, device=device)

if USE_BCELOGIT:
    loss_criteria = nn.BCEWithLogitsLoss(weight=w, pos_weight=pos_w)
else:
    loss_criteria = nn.BCELoss(weight=w)
if device == 'cuda':
    loss_criteria.cuda()


# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

# Learning rate will be reduced automatically during training
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = LEARNING_RATE_SCHEDULE_FACTOR, patience = LEARNING_RATE_SCHEDULE_PATIENCE, mode = 'max', verbose=True)


# ## Compute F1-score 
# Because we have multi labels, we need to calculate F1-score for each class.

# In[ ]:


def multi_label_f1(y_gt, y_pred, threshold=0.5):
    """ Calculate F1 for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    f1_out = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = (y_pred.to("cpu").numpy() > threshold) * 1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:, i], pred_np[:, i]))
    return f1_out


# ## Training each epoch
# This function will be called to train on one epoch
# 

# In[ ]:


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb, augment=True):
    """
    Epoch training

    Paramteters
    -----------
    epoch: int
      epoch number
    model: torch Module
      model to train
    train_dataloader: Dataset
      data loader for training
    device: str
      "cpu" or "cuda"
    loss_criteria: loss function
      loss function used for training
    optimizer: torch optimizer
      optimizer used for training
    mb: master bar of fastprogress
      progress to log

    Returns
    -------
    float
      training loss
    """
    # Switch model to training mode
    model.train()
    training_loss = 0 # Storing sum of training losses
   
    # For each batch
    for batch, (images, labels) in enumerate(progress_bar(train_dataloader, parent=mb)):            

        # Move X, Y  to device (GPU)
        images = images.to(device)
        labels = labels.to(device)
        if MIXUP:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
        # Clear previous gradient
        optimizer.zero_grad()

        # Feed forward the model
        pred = model(images)
        if MIXUP:
            loss = mixup_criterion(loss_criteria, pred, labels_a, labels_b, lam)
        else:
            loss = loss_criteria(pred, labels)
        # Back propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Update training loss after each batch
        training_loss += loss.item()

        mb.child.comment = f'Training loss {training_loss/(batch+1)}'

    del images, labels, loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # return training loss
    return training_loss/len(train_dataloader)


# ## Evaluate model
# This function is used to validate the model on the validation dataset

# In[ ]:


def evaluating(epoch, model, val_loader, device, loss_criteria, mb):
    """
    Validate model on validation dataset
    
    Parameters
    ----------
    epoch: int
        epoch number
    model: torch Module
        model used for validation
    val_loader: Dataset
        data loader of validation set
    device: str
        "cuda" or "cpu"
    loss_criteria: loss function
      loss function used for training
    mb: master bar of fastprogress
      progress to log
  
    Returns
    -------
    float
        loss on validation set
    float
        metric score on validation set
    """

    # Switch model to evaluation mode
    model.eval()

    val_loss = 0                                   # Total loss of model on validation set
    out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
    out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

    with torch.no_grad(): # Turn off gradient
        # For each batch
        for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):
            # Move images, labels to device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Update groundtruth values
            out_gt = torch.cat((out_gt,  labels), 0)

            # Feed forward the model
            ps = model(images)
            loss = loss_criteria(ps, labels)

            # Update prediction values
            out_pred = torch.cat((out_pred, ps), 0)
            
            
            # Update validation loss after each batch
            val_loss += loss
            mb.child.comment = f'Validation loss {val_loss/(step+1)}'

    # Clear memory
    del images, labels, loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # return validation loss, and metric score
    return val_loss/len(val_loader), np.array(multi_label_f1(out_gt, out_pred)).mean()


# ## Fully training
# Fully training the model 

# In[ ]:


# Best F1 value during training
best_score = 0
best_loss = 10
model_path = "resnet50.pth"
training_losses = []
validation_losses = []
validation_score = []


# Config progress bar
mb = master_bar(range(MAX_EPOCHS))
mb.names = ['Training loss', 'Validation loss', 'Validation F1']
x = []

# Training each epoch
for epoch in mb:
    mb.first_bar.comment = f'Best F1 score: {best_score}'
    x.append(epoch)

    # Training
    train_loss = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)
    mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
    training_losses.append(train_loss)

    # Evaluating
    val_loss, new_score = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)
    mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))
    validation_losses.append(val_loss)
    validation_score.append(new_score)

    # Update learning rate
    lr_scheduler.step(new_score)

    # Update training chart
    mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0,MAX_EPOCHS], [0,1])

    # Save model
    if best_score < new_score:
        mb.write(f"Improve F1 from {best_score} to {new_score}")
        best_score = new_score
        best_loss = val_loss
        # Saving model: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(model.state_dict(), model_path)
    if new_score == 1.0 and best_loss > val_loss:
        mb.write(f"Improve score from {best_loss} to {val_loss}")
        best_score = new_score
        best_loss = val_loss
        # Saving model: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(model.state_dict(), model_path)


# # Find threshold

# In[ ]:


model.load_state_dict(torch.load(model_path))
model.eval()
def eval_thres(model, val_loader, threshold=0.5):
    """
    Validate model on validation dataset
    
    Parameters
    ----------
    epoch: int
        epoch number
    model: torch Module
        model used for validation
    val_loader: Dataset
        data loader of validation set
    device: str
        "cuda" or "cpu"
    loss_criteria: loss function
      loss function used for training
    mb: master bar of fastprogress
      progress to log
  
    Returns
    -------
    float
        loss on validation set
    float
        metric score on validation set
    """

    # Switch model to evaluation mode
    model.eval()

    val_loss = 0                                   # Total loss of model on validation set
    out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
    out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

    with torch.no_grad(): # Turn off gradient
        # For each batch
        for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):
            # Move images, labels to device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Update groundtruth values
            out_gt = torch.cat((out_gt,  labels), 0)

            # Feed forward the model
            ps = model(images)

            # Update prediction values
            out_pred = torch.cat((out_pred, ps), 0)
            

    # Clear memory
    del images, labels
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # return validation loss, and metric score
    return np.array(multi_label_f1(out_gt, out_pred, threshold)).mean()

best_score, best_threshold = 0, 0
for threshold in [0.2,0.25,0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    score = eval_thres(model, val_dataloader, threshold)
    if score > best_score:
        best_score = score
        best_threshold = threshold
    print(f'{threshold} - {score}')


# # Inference

# Read the test data

# In[ ]:


test_df = pd.read_csv("../input/vietai-advance-course-retinal-disease-detection/sample_submission.csv")
test_df.head()


# Predict function

# In[ ]:


from torch.nn import functional as F
# Define list of image transformations
image_transformation = [
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
]
# Normalization with mean and std from ImageNet
image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
image_transformation = transforms.Compose(image_transformation)

def predict(image_path, model, device):
    """ Predict image
    
    Parameters
    ----------
    image_path: str
        image path to predict
    model: nn.Module
        model used to predict
    device: str
        'cpu' or 'cuda'
        
    Returns
    -------
    str
        list of label indices
    """
    global image_transformation
    
    # Read image
    image_data = Image.open(image_path).convert('RGB')
    image_data = image_transformation(image_data)
    
    predicted_label = []
    with torch.no_grad():
        ps = model(image_data.unsqueeze(0).to(device))
        if USE_BCELOGIT:
            ps = F.sigmoid(ps)
        ps = ps[0]
        for i in range(ps.size()[0]):
            if ps[i].item() > best_threshold: # Threshold is 0.5
                predicted_label.append(i)
        if len(predicted_label) == 0:
            predicted_label.append(0)
    return " ".join([str(label) for label in predicted_label])



# Load best model weights and switch to evaludation mode

# In[ ]:


model.load_state_dict(torch.load(model_path))
model.eval()


# Predict test images

# In[ ]:


test_df['predicted'] = test_df.filename.map(lambda x: predict(os.path.join("../input/cleandata-vietai-caothang/cleaned_data/cleaned_data/test",x), model, device))


# Write result to submission file

# In[ ]:


test_df.to_csv("submission.csv", index=False)


# In[ ]:


print(model_path)


# <a href="submission.csv">Download submission</a> <br> <a href='resnet50.pth'>Download model</a>
