#!/usr/bin/env python
# coding: utf-8

# # Pytorch from a Beginner's View - Dataloader, Network, and Training
# 
# As a beginner in deep learning and PyTorch, I want to share my understandings and struggles from implementing a recognizer for the MNIST competition on Kaggle. Though the MNIST competition is never a hard problem, I want to use this opportunity as the foundation for other problems in the future. Because of this, I tried to implement the recognizer following a systematic pipeline ("Systematic" in my limited knowledge, I can do better with more experiences in deep learning). Here are the components I will cover in this notebook for a complete recognizer pipeline:
# 
# 1. **Dataset and Dataloader:** Split data into training and validation dataset respectively and randomly construct mini-batch.
# 2. **Network:** Convolutional neural network starting with Basic LeNet-5.
# 3. **Training (and Validation)**: Train the network with customized Dataloader.
# 4. **Data Augmentation:** Add affine image transformations to the digit image. 

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from random import shuffle, randint
from PIL import Image
import math


# # 1. Dataset and Dataloader
# 
# After some search, I found one of the standard ways to load data into models in PyTorch is using the Dataloader. To use the Dataloader, I have to define a Dataset class for the digit images. Then, the Dataloader splits the data into mini-batches for training.
# 
# ### 1.1 Basic Dataset Class
# 
# First, I defined a Dataset class for the digit images. The PyTorch tutorial has a detailed explanation of how to construct a Dataset class ([Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)). Here, the Dataset class reads data from the CSV file when the getitem function is called.

# In[ ]:


class DigitDataset(Dataset):
    """ Digit Dataset """

    def __init__(self, csv_file, root_dir, train=False, transform=None):
        self.digit_df = pd.read_csv(root_dir + csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.digit_df)

    def __getitem__(self, item):
        if self.train:
            digit = self.digit_df.iloc[item, 1:].values
            digit = digit.astype('float').reshape((28, 28))
            label = self.digit_df.iloc[item, 0]
        else:
            digit = self.digit_df.iloc[item, :].values
            digit = digit.astype('float').reshape((28, 28))
            label = 0
        sample = [digit, label]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


# ### 1.2 Basic Transformations
# 
# Dataset class can perform different transformations on the data. Here, I defined two transformations: Regularize pixel value from [0, 255] to [0, 1] and transform data type of the image from Numpy to PyTorch Tensor.

# In[ ]:


class Regularize(object):
    """ Regularize digit pixel value """

    def __init__(self, max_pixel=255):
        self.max_pixel = max_pixel

    def __call__(self, digit):
        assert isinstance(digit, np.ndarray)
        digit = digit / self.max_pixel
        return digit


class ToTensor(object):
    """ Covert ndarrays to Tensors """

    def __call__(self, digit):
        assert isinstance(digit, np.ndarray)
        digit = digit.reshape((1, 28, 28))
        digit = torch.from_numpy(digit)
        digit = digit.float()
        return digit


# Let's test the Dataset class by first defining a Dataset object without any transformation. Here, the result shows the digit image and label get directly from the Dataset object.

# In[ ]:


data_np = DigitDataset('train.csv', '../input/', train=True)
print("Number of Training Images: ", len(data_np))
plt.imshow(data_np[5][0], cmap='gray')
plt.show()
print("Label for the Image: ", data_np[5][1])


# ### 1.3 Dataloader
# 
# With a correctly defined Dataset class, I can now use the Dataloader provided by PyTorch to form mini-batches. The Dataloader prepares mini-batches during iteration. By setting the correct number for num_workers (base on the memory) and make pin_memory to True, the Dataloader can make the training iteration a lot faster on GPU.

# In[ ]:


composed_transform = transforms.Compose([Regularize(), ToTensor()])
data_torch = DigitDataset('train.csv', '../input/', train=True, transform=composed_transform)
dataloader = DataLoader(data_torch,
                       batch_size=4,
                       shuffle=True,
                       num_workers=4)
for i, data in enumerate(dataloader, 0):
    digits, labels = data
    # digits, labels = digits.to("cuda"), labels.to("cuda")
    print("Type of Digits: ", type(digits))
    print("Dimension of the Tensor: ", digits.shape)
    print("Type of Labels: ", type(labels))
    print("Dimension of the Tensor: ", labels.shape)
    if i == 0:
        break


# ### 1.4 Training and Validation Data Split
# 
# Since I have to validate how good is the trained model, I have to split the data from Kaggle into a training set and a validation set. At the same time, I want the validation set has the same class distribution as in the training set. (I tried to find a PyTorch function to do this but failed. Thus, I have to write one by myself. ) To do so, I first defined a function to compute the class distribution of a dataset, which returns how many digits are in the data for each label.

# In[ ]:


def digits_per_class(digit_df, indices):
    assert isinstance(digit_df, pd.DataFrame)
    assert isinstance(indices, list)
    digit_num = [0 for num in range(10)]
    for idx in indices:
        label = digit_df.iloc[idx, 0]
        digit_num[label] += 1
    return digit_num


# Here, the function computed the class distribution of the data given by Kaggle.

# In[ ]:


digit_class_num = digits_per_class(data_torch.digit_df,
                                  [num for num in range(len(data_torch))])
for i, num in enumerate(digit_class_num, 0):
    print("Number of Images for Digit ", i, ": ", num)
print("Overall Images: ", sum(digit_class_num))


# Then, I defined a function to split the dataset into training and validation. The function returns two lists, representing the indices of the training set and the validation set, respectively. The code is relatively straightforward. I first get the class distribution of the complete dataset, then use it to compute the class distribution of the validation set based on the ratio provided to the user. Based on the validation class distribution, I put part of the data into the validation set.

# In[ ]:


def train_validate_split(digit_df, test_ratio=0.2):
    assert isinstance(digit_df, pd.DataFrame)
    digit_num = len(digit_df)
    overall_indices = [num for num in range(digit_num)]
    overall_class_num = digits_per_class(digit_df, overall_indices)
    test_class_num = [int(num*test_ratio) for num in overall_class_num]
    tmp_test_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
    train_indices = []
    val_indices = []
    for idx in overall_indices:
        tmp_label = digit_df.iloc[idx, 0]
        if tmp_test_class_num[tmp_label] < test_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_test_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
    return train_indices, val_indices


# Here shows the class distribution of the validation set is proportional to the training set.

# In[ ]:


train_data, val_data = train_validate_split(data_torch.digit_df)
train_class_num = digits_per_class(data_torch.digit_df, train_data)
val_class_num = digits_per_class(data_torch.digit_df, val_data)
for i, num in enumerate(train_class_num, 0):
    print("Number of Images for Digit ", i, "- Train: ", num, "Validate: ", val_class_num[i])
print("Train Images: ", sum(train_class_num), "Validate Images: ", sum(val_class_num))


# An easy way to use the generated index list is to use the sampler provided by PyTorch. By using the defined sampler in the Dataloader, the Dataloader will only prepare mini-batches from the data included in the sampler.
# 
# Now we have all the necessary things we need on Dataset and Dataloader for training the model.

# In[ ]:


train_sampler = sampler.SubsetRandomSampler(train_data)
train_dataloader = DataLoader(data_torch,
                              batch_size=4,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=4)


# # 2. Network
# 
# After some paper reading, I think the most natural network to start with is the LeNet-5 (If you want to read the paper, [Here is LeCun's LeNet-5 paper from 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)). Honestly, I am amazed by how much they already have back in 1998 that built the foundation of deep learning today.
# 
# This network can be tuned later to maximize the performance by,
# 
# 1. Adding more convolutional filters.
# 2. Adding more convolutional layers.
# 3. Increasing the dimension of fully connected layers.
# 4. Adding batch normalization layers.
# 5. and many other techniques.

# In[ ]:


class BasicLeNet(nn.Module):
    """ Basic LeNet-5 as defined in LeCun's paper"""

    def __init__(self):
        super(BasicLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        """
        Feedforward function

        :param x: input batch
        :return: x
        """
        x = self.features(x)
        x = x.view(x.size(0), 16*4*4)
        x = self.classifier(x)
        return x


# # 3. Training (and Validation)
# 
# Now I want to define a training function that trains any network. The function first creates a training Dataloader and a validation Dataloader, and then it optimizes the network using mini-batches from the Dataloader using iterations.
# 

# In[ ]:


def training(network, criterion, optimizer, epoch_num, test=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Start Training with", device, epoch_num, "overall epoch")
    network.to(device)
    """
    Create Dataloader for training and validating
    """
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    digit_dataset = DigitDataset('train.csv', '../input/', train=True, transform=composed_transform)
    if test:
        train_indices, val_indices = train_validate_split(digit_dataset.digit_df)
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        val_sampler = sampler.SubsetRandomSampler(val_indices)
        train_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        print("Training with validation, ", "Overall Data:", len(train_indices)+len(val_indices))
        print("Training Data:", len(train_indices), "Validate Data:", len(val_indices))
    else:
        train_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_dataloader = None
        print("Training all data, ", "Overall Data:", len(digit_dataset))
    """
    Start Training
    """
    batch_num = 0
    ita = []
    loss_avg = []
    val_acc = []
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            digits, labels = data
            digits, labels = digits.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(digits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_num += 1
            if test == True and i % 500 == 499:
                ita.append(batch_num)
                loss_avg.append(running_loss/500.)
                val_acc.append(validating(network, val_dataloader))
                running_loss = 0.
    if test:
        train_accuracy = validating(network, train_dataloader)
        val_accuracy = validating(network, val_dataloader)
        print('Training accuracy: %.5f' % (train_accuracy))
        print('Validation accuracy: %.5f' % (val_accuracy))
    return network, ita, loss_avg, val_acc


# I also defined a validating function. The function efficiently generates the prediction accuracy of the network. I used this function to generate the training accuracy and validation accuracy after training.

# In[ ]:


def validating(network, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct_num = 0
    total_num = 0
    for i, data in enumerate(loader, 0):
        digits, labels = data
        total_num += labels.size(0)
        digits, labels = digits.to(device), labels.to(device)
        outputs = network(digits)
        _, predicted = torch.max(outputs, 1)
        correct_num += ((predicted == labels).sum().to("cpu")).item()
    accuracy = correct_num / total_num
    return accuracy


# Finally, I can train and test the network. Here I used the Cross-Entropy Loss function and Adam optimizer. After 30 epochs (with 80% of the training data given by Kaggle), the network has a validation accuracy over 98.5% with a converged loss function.

# In[ ]:


lenet = BasicLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters())
lenet, batch_ita, loss_list, val_acc_list = training(lenet, criterion, optimizer, 30)


# In[ ]:


fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(batch_ita, loss_list)
plt.title("Loss function")

plt.subplot(1, 2, 2)
plt.plot(batch_ita, val_acc_list)
plt.title("Validation accuracy")

plt.show()


# # 4. Data Augmentation
# 
# The result using the LeNet-5 from 1998 is very good. To push the result better than 99%, I need to do some additional things. The first extra thing I came to mind immediately is data augmentation. Since deep learning trains a model based on the training data it sees, having more data will benefit and train a better model. The only thing to be careful is not to overfit the model.
# 
# The most common data augmentation technique is the affine transformation on images. [Here is a detailed tutorial on affine transformation in python.](https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/) Here, I created a function for fundamental affine transformation on the digit image from the Kaggle dataset. The function can do three transformations, including rotation, translation, and scaling.

# In[ ]:


def digit_argument(digit, angle, translate, scale):
    digit_img = Image.fromarray(digit)
    t1 = np.array([[1, 0, 14],
                   [0, 1, 14],
                   [0, 0, 1]])
    t2 = np.array([[math.cos(angle), math.sin(angle), 0],
                   [-math.sin(angle), math.cos(angle), 0],
                   [0, 0, 1]])
    t3 = np.array([[scale, 0, 0],
                   [0, scale, 0],
                   [0, 0, 1]])
    t4 = np.array([[1, 0, -14],
                   [0, 1, -14],
                   [0, 0, 1]])
    t5 = np.array([[1, 0, translate[0]],
                   [0, 1, translate[1]],
                   [0, 0, 1]])
    t_inv = np.linalg.inv(t1 @ t2 @ t3 @ t4 @ t5)
    digit_img = digit_img.transform((28, 28),
                                    Image.AFFINE,
                                    data=t_inv.flatten()[:6],
                                    resample=Image.BILINEAR)
    digit_arg = np.asarray(digit_img)
    return digit_arg


# Now let's test the argumentation function.

# In[ ]:


old_digit = data_np[10][0]
print("Digit Image before argumentation: ")
plt.imshow(old_digit, cmap="gray")
plt.show()
print("Digit Image after Rotation: ")
rotate_digit = digit_argument(old_digit, 1, [0, 0], 1)
plt.imshow(rotate_digit, cmap="gray")
plt.show()
print("Digit Image after Translation: ")
translate_digit = digit_argument(old_digit, 0, [-2, -2], 1)
plt.imshow(translate_digit, cmap="gray")
plt.show()
print("Digit Image after Scaling: ")
scale_digit = digit_argument(old_digit, 0, [0, 0], 1.2)
plt.imshow(scale_digit, cmap="gray")
plt.show()


# For each training image, I want to create a new argument image. Thus, the overall training data will be doubled after argumentation. To do so, I have to update the Dataset class to differentiate the original training data and the new argument data.

# In[ ]:


class DigitDataset(Dataset):
    """ Digit Dataset """

    def __init__(self, csv_file, root_dir, train=False, argument=True, transform=None):
        self.digit_df = pd.read_csv(root_dir + csv_file)
        self.transform = transform
        self.train = train
        self.argument = argument

    def __len__(self):
        if self.argument:
            return 2 * len(self.digit_df)
        else:
            return len(self.digit_df)

    def __getitem__(self, item):
        if item < len(self.digit_df):
            if self.train:
                digit = self.digit_df.iloc[item, 1:].values
                digit = digit.astype('float').reshape((28, 28))
                label = self.digit_df.iloc[item, 0]
            else:
                digit = self.digit_df.iloc[item, :].values
                digit = digit.astype('float').reshape((28, 28))
                label = 0
        else:
            assert self.argument and self.train
            digit = self.digit_df.iloc[item % len(self.digit_df), 1:].values
            digit = digit.astype('float').reshape((28, 28))
            rand_theta = (randint(-20, 20) / 180) * math.pi
            rand_x = randint(-2, 2)
            rand_y = randint(-2, 2)
            rand_scale = randint(9, 11) * 0.1
            digit = digit_argument(digit, rand_theta, [rand_x, rand_y], rand_scale)
            label = self.digit_df.iloc[item % len(self.digit_df), 0]
        sample = [digit, label]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


# I also need to redefine the function of splitting the training data and the validation data. The validation data won't contain any argument image, and the training data won't contain any data argument from the validation data. Here I need to be very careful to get the correct validation data.

# In[ ]:


def train_validate_split(digit_df, test_ratio=0.2, argument=True):
    assert isinstance(digit_df, pd.DataFrame)
    digit_num = len(digit_df)
    overall_indices = [num for num in range(digit_num)]
    overall_class_num = digits_per_class(digit_df, overall_indices)
    test_class_num = [int(num*test_ratio) for num in overall_class_num]
    tmp_test_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
    train_indices = []
    val_indices = []
    for idx in overall_indices:
        tmp_label = digit_df.iloc[idx, 0]
        if tmp_test_class_num[tmp_label] < test_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_test_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
            if argument:
                train_indices.append(idx + digit_num)
    return train_indices, val_indices


# Here I also defined a more complicated network than LeNet-5 with more Conv layers and batch normalization.

# In[ ]:


class EnhancedLeNet(nn.Module):

    def __init__(self):
        super(EnhancedLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128*7*7)
        x = self.classifier(x)
        return x


# Finally, I can train and test the new network with data argumentation. After 30 epochs, the network has a validation accuracy over 99% with a converged loss function.

# In[ ]:


lenet = EnhancedLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters())
lenet, batch_ita, loss_list, val_acc_list = training(lenet, criterion, optimizer, 30)


# In[ ]:


fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(batch_ita, loss_list)
plt.title("Loss function")

plt.subplot(1, 2, 2)
plt.plot(batch_ita, val_acc_list)
plt.title("Validation accuracy")

plt.show()


# Now I can test how the model performs on the real test data provided by Kaggle. To get a better testing performance, I have to use all the training data (not split into training and validation set) during training. I also defined a testing function to generate the predictions of the trained model.

# In[ ]:


def testing(network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    digit_dataset = DigitDataset('test.csv', '../input/', train=False, argument=False, transform=composed_transform)
    test_dataloader = DataLoader(
        digit_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_results = []
    for i, data in enumerate(test_dataloader, 0):
        digits, label = data
        digits = digits.to(device)
        outputs = network(digits)
        _, predicted = torch.max(outputs, 1)
        test_results += np.int_(predicted.to("cpu").numpy().squeeze()).tolist()
    """
    Write test results to a csv file
    """
    test_df = pd.read_csv("../input/sample_submission.csv")
    assert (len(test_df) == len(test_results))
    test_df.loc[:, 'Label'] = test_results
    test_df.to_csv('test_results.csv', index=False)
    print("Test Results for Kaggle Generated ...")


# In[ ]:


lenet = EnhancedLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters())
lenet, batch_ita, loss_list, val_acc_list = training(lenet, criterion, optimizer, 50, test=False)
testing(lenet)

