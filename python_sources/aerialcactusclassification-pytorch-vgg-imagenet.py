#!/usr/bin/env python
# coding: utf-8

# # Classification of Catcus

# * The main aim of the competition is to create an algorithm that can identify a specific type of cactus in aerial imagery.
# * This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a columnar cactus (Neobuxbaumia tetetzo). Kaggle has resized the images from the original dataset to make them uniform in size

# In[ ]:


#importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import seaborn as sns
import os

#pytorch imports
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image


# ### Declaring the directory variables

# In[ ]:


train_data_directory = "../input/train/train"
test_data_directory = "../input/test/test"


# # Exploratory Data Analysis

# In[ ]:


#reading the train.csv file for basic data analysis
train_csv_df = pd.read_csv("../input/train.csv")
train_csv_df.head()


# In[ ]:


#dataset dimensions
train_csv_df.shape


# In[ ]:


#checking for missing values in the dataset
train_csv_df.isna().sum()


# In[ ]:


#checking the distribution of classes
train_csv_df.has_cactus.value_counts()


# In[ ]:


#distribution of classes using seaborn plot
plt.style.use("seaborn")
train_csv_df.has_cactus.value_counts().plot(kind = "barh")
plt.ylabel("Classes")
plt.xlabel("Number of catus in each class")
plt.show()


# In[ ]:


#reading the test data set csv
test_data_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


test_data_df.head()


# In[ ]:


#number of images in test data
test_data_df.shape


# In[ ]:





# # Imbalance Dataset

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#split the train data csv into two parts using stratified sampling
train_df, validation_df = train_test_split(train_csv_df, stratify=train_csv_df.has_cactus, test_size=0.2)


# In[ ]:


#shape of train data
train_df.shape


# In[ ]:


#shape of validation data
validation_df.shape


# # Creating a Custom Data Loader Class
# * For converting the dataset to torchvision dataset format.
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# In[ ]:


class AerialCatcusClassification(Dataset):
    def __init__(self,file_data,root_dir,transform=None):
        self.transform = transform
        self.file_data = file_data.values
        self.data_root = root_dir 
            
    def __len__(self):
        return len(self.file_data)
    
    def __getitem__(self, index):
        img_name, label = self.file_data[index]
        img_data = self.pil_loader(os.path.join(self.data_root, img_name))
        #applying transforms if any specified
        if self.transform:
            img_data = self.transform(img_data)
        return img_data, label
          
    def pil_loader(self,path):
        #funtion to read the image and convet to RGB
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


# ## Loading Data

# In[ ]:


#read the train and validation data images
train_data = AerialCatcusClassification(file_data=train_df, root_dir=train_data_directory, transform = transforms.Compose([transforms.ToTensor()]))
validation_data = AerialCatcusClassification(file_data=validation_df, root_dir=train_data_directory, transform = transforms.Compose([transforms.ToTensor()]))

#read the test data
test_data = AerialCatcusClassification(file_data=test_data_df, root_dir=test_data_directory,transform = transforms.Compose([transforms.ToTensor()]))


# In[ ]:


#create a train and validation data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=5, shuffle=True)


# # Visualization Data

# In[ ]:


#custom function to display images

def imshow(img, title):
    
    #convert image from tensor to numpy for visualization
    npimg = img.numpy()
    #define the size of a figure
    plt.figure(figsize = (20, 20))
    plt.axis("off")
    
    #interchaging the image sizes - transposing
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title, fontsize=15)
    plt.show()


# In[ ]:


#function to get images and feed into our custom function 'imshow'

def show_batch_images(dataloader):
    
    #getting the images
    images, labels = next(iter(dataloader))
    #make a grid from those images
    img = torchvision.utils.make_grid(images)
    
    #call our custom function
    imshow(img, title = [str(x.item()) for x in labels])


# In[ ]:


#visualize the training data

for i in range(4):
    show_batch_images(train_loader)


# ## More Transformations
# 
# ### Data Augmentation
# 
# A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.
# 
# To randomly rotate, scale and crop, then flip your images you would define your transforms like this:
# 
# ```python
# train_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                        transforms.RandomResizedCrop(224),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.5, 0.5, 0.5], 
# ```
# 
# You'll also typically want to normalize images with `transforms.Normalize`. You pass in a list of means and list of standard deviations, then the color channels are normalized like so
# 
# ```input[channel] = (input[channel] - mean[channel]) / std[channel]```
# 
# Subtracting `mean` centers the data around zero and dividing by `std` squishes the values to be between -1 and 1. Normalizing helps keep the network work weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.
# 
# You can find a list of all [the available transforms here](http://pytorch.org/docs/0.3.0/torchvision/transforms.html). When you're testing however, you'll want to use images that aren't altered (except you'll need to normalize the same way). So, for validation/test images, you'll typically just resize and crop.
# 

# In[ ]:


transform_train = transforms.Compose([
    #cropping and resizing the image to 224*224
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    #convert image to tensor
    transforms.ToTensor(),
    #normalizing the input mean = 0.5 and std = 0.5 for three channels (R,G,B)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#define the same operations to test data
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[ ]:


#read the train and validation data images
train_data = AerialCatcusClassification(file_data=train_df, root_dir=train_data_directory, transform = transform_train)
validation_data = AerialCatcusClassification(file_data=validation_df, root_dir=train_data_directory, transform = transform_train)

#read the test data
test_data = AerialCatcusClassification(file_data=test_data_df, root_dir=test_data_directory,transform = transform_test)


# In[ ]:


#create a train and validation data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)


# # Modeling
# * Pretrained VGG Net trained on Imagenet dataset

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import copy


# In[ ]:


#number of classes in the data
num_classes = 2
training_batchsize = 10


# In[ ]:


#create a data iterator

dataiter = iter(train_loader)
images, labels = dataiter.next()

#shape of images bunch
print(images.shape)

#shape of first image in a group of 4
print(images[1].shape)

#class label for first image
print(labels)


# In[ ]:


#checking for available gpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


#download the vgg19 model pretrained on imagenet data
vgg = models.vgg19_bn(pretrained = True)


# In[ ]:


#number of trainable parameters in vgg19 - before freezing
print("Number of trainable parameters: ", sum(p.numel() for p in vgg.parameters() if p.requires_grad))


# In[ ]:


#Let's print the names of the layer stacks for our model
for name, child in vgg.named_children():
    print(name)


# In[ ]:


#freeze all the weights
for param in vgg.parameters():
    param.requires_grad = False


# In[ ]:


#number of features in final fc layer
final_in_features = vgg.classifier[6].in_features

#editing the final fc layer
vgg.classifier[6] = nn.Linear(final_in_features, num_classes)


# In[ ]:


#checking the size of trainable layesr
for param in vgg.parameters():
    if param.requires_grad:
        print(param.shape)


# # Train the Model

# In[ ]:


#define the loss function and optimizer for back prop
criterion = nn.CrossEntropyLoss()
vgg = vgg.to(device) #push to gpu

#pass only the trainable parameters into optimizer
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), lr=0.0001, amsgrad=True, weight_decay=1e-4)


# In[ ]:


#custom function to evaluate the model

def evaluation(dataloader, model):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        #pushing the data to gpu if present else cpu
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)

        #number of correct predictions
        correct += (pred == labels).sum().item()

    return 100 * correct/total


# In[ ]:


def train_model(model, criterion, optimizer, num_epochs=25):
    
    since = time.time()
    train_acc, validation_acc = [], []
    best_loss = 1000
    best_model_wts = copy.deepcopy(model.state_dict())

    #define the number of iterations
    n_iter = np.ceil(len(train_loader.dataset)/training_batchsize)

    #iterate or epochs
    for epoch in range(num_epochs):
        running_trainloss = 0.0

        #iterate through all the batches to complete one pass
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #push to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            #set gradients to zero
            optimizer.zero_grad()

            #run the model
            outputs = model(inputs)

            #calculate loss
            loss = criterion(outputs, labels)
            
            #backpropagate the gradients
            loss.backward()
            optimizer.step()

            #total trainingloss
            running_trainloss += loss.item()

            #clear memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        else:
            running_trainloss = running_trainloss / len(train_loader.dataset)

            #doing inference
            validation_accuracy = 0
            training_accuracy = 0
            
            model.eval()
            with torch.no_grad():
                validation_accuracy = evaluation(validation_loader, model)
                training_accuracy = evaluation(train_loader, model)
                
            if running_trainloss < best_loss:
                best_loss = running_trainloss
                best_model_wts = copy.deepcopy(model.state_dict())

            model.train()

            print("Epoch: {}/{}.. ".format(epoch+1, num_epochs), "Training Loss: {:.3f}.. ".format(running_trainloss),
                  "Training Accuracy % : {:.3f}.. ".format(training_accuracy),
                  "Validation Accuracy % : {:.3f}..".format(validation_accuracy))

            train_acc.append(training_accuracy)
            validation_acc.append(validation_accuracy)
    
    time_elapsed = time.time() - since        
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best training loss: {:4f}'.format(best_loss))
    
    #saving the best model
    torch.save(model.state_dict(best_model_wts),"saved.pth")
    print("best model saved")
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_acc, validation_acc


# In[ ]:


#train the model
best_model, train_acc, validation_acc = train_model(vgg, criterion, optimizer_ft,120)


# In[ ]:


#plot the accuracy
plt.plot(train_acc, label='Training Accuracy')
plt.plot(validation_acc, label='Validation Accuracy')
plt.legend(frameon=False)
plt.show()


# In[ ]:


#compute the validation accuracy using best model
validation_acc_bestmodel = evaluation(validation_loader, best_model)
print("validation accuracy %: ", validation_acc_bestmodel)


# In[ ]:


#create a iterator

dataiter = iter(test_loader)
images, labels = dataiter.next()

#shape of images bunch
print(images.shape)

#shape of first image in a group of 4
print(images[1].shape)

#class label for first image
print(labels[1])


# In[ ]:


#custom function for prediction
def testdata_sumission(dataloader, model):
    since = time.time()
    predict = []
    
    model.eval()  # Set model to evaluate mode
    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        
        pred = pred.cpu().numpy()
        #print(pred)
        for i in pred:
            predict.append(i)
      
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return predict


# # Make Final Submission

# In[ ]:


#appending the predicted values
test_data_df['has_cactus'] = testdata_sumission(test_loader, best_model)

#saving the submission file
test_data_df.to_csv('submission.csv', index= False)

