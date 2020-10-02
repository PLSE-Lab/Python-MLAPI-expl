#!/usr/bin/env python
# coding: utf-8

# This command tells you what GPU you have been assigned. If you get P100, you should be very happy, it is an excellent card. Otherwise, K80 or P4 are not as powerful but its free, and its sufficient to help you train.

# In[ ]:


get_ipython().system('nvidia-smi')


# Connect Google Drive to this instance. You can access your google drive files under /content/gdrive/

# In[ ]:


from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)


# This section will unzip the shopee code league file to this instance. This should work if you put the zip file directly in your Google Drive folder. Otherwise, adjust your path correspondingly. This will take roughly 2-3mins
# 
# 

# In[ ]:


get_ipython().system('unzip "./gdrive/My Drive/shopee-product-detection-dataset.zip"')


# In[ ]:


# See how the folders are named for each category
get_ipython().system('ls "./train/train"')


# Libraries to import. I am most familiar with PyTorch, hence the library I use is PyTorch.

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.notebook import tqdm


# **Displaying example images.** Prints a sample image for each category.

# In[ ]:


path="./train/train/"                                       # where the training images are
classes = ["%.2d" % i for i in range(len(os.listdir(path)))]    # ['00','01',...,'41'] i.e. the folders for the images of each corresponding category
for i in classes:
    ls = sorted(os.listdir(path+i))
    print("Number of images for Class {}:".format(i),len(ls))
    img=mpimg.imread(path+'{}/{}'.format(i,ls[1]))          # load images
    imgplot = plt.imshow(img)                               # display images
    plt.show()


# **Create dataset class to pass images into model**. This class stores a list of the path to each training image file. It only loads an item when its called(from the \_getitem_ function). This is done like this because it will require a lot of memory to load all 100k images at once. This way, an image is only loaded when it is needed.

# In[ ]:


class ShopeeDataset(Dataset):
    def __init__(self,directory,transform,train=True):
        self.transform = transform                      # function that turns an image into a tensor that can be passed to the pytorch model
        self.img_ls = list()                            # where the path to each training image file is stored

        classes = ["%.2d" % i for i in range(len(os.listdir(directory)))]           # ['00','01',...,'41'] i.e. the folders for the images of each corresponding category
        for clas in classes:
            path = os.path.join(directory,clas)
            ls = sorted(os.listdir(path))               # obtain list of image file names in a class folder
            if train:
                for img_name in ls[:-50]:                                           # take all images except the last 50 images for each category. This will be used as training data
                    self.img_ls.append((os.path.join(path,img_name),int(clas)))     # adds to img_ls in a tuple format (path,category)
            else:
                for img_name in ls[-50:]:                                           # last 50 images of each category for validation set, for monitoring of current performance
                    self.img_ls.append((os.path.join(path,img_name),int(clas)))
        
    def __getitem__(self, idx):                         # this function will be called by pytorch dataloader class.
        name, label = self.img_ls[idx]
        img = Image.open(name).convert('RGB')           # load image
        img.load()
        img = self.transform(img)                       # apply transformation to turn image into torch tensor
        return {"image": img, "label": label}
    
    def __len__(self):                                  # allows usage of len(<name of dataset>) to obtain size of entire dataset
        return len(self.img_ls)


# This function trains the model for 1 epoch (meaning it iterates through the entire training dataset once). 

# In[ ]:


def train_epoch(model,  trainloader,  criterion, device, optimizer):


    model.train()
    losses = list()
    ct=0
    for batch_idx, data in enumerate(tqdm(trainloader)):       #iterate through dataset. tqdm is the loading bar, allows you to track progress
        ct+=1
        inputs=data['image'].to(device)                 # images
        labels=data['label'].to(device)                 # labels
        optimizer.zero_grad()
        outputs = model(inputs)                         # prediction
        loss = criterion(outputs, labels)               # loss function, tells you how "wrong" your predictions are based on the training dataset.
        loss.backward()                                 # backpropogation
        optimizer.step()                                # gradient descent
        losses.append(loss.item())

    return sum(losses)/ct


# This function tests the performance of your current model on the validation images. Recall we saved the last 50 images of each category as validation images. 

# In[ ]:


def evaluate(model, dataloader, device):

    model.eval()                                                    # set model to evaluation mode
    total=0                                                         # store total number of validation samples
    correct = 0                                                     # store total number of correct prediction samples
    with torch.no_grad():
      for ctr, data in enumerate(dataloader):
          inputs = data['image'].to(device)                         # obtain validation images
          outputs = model(inputs)                                   # pass images to model
          labels = data['label']                                    # actual labels for validation images. Also known as ground truth
          labels = labels.float()
          cpuout= outputs.to('cpu')
          total += len(labels)                                      # total number of samples

          pred = cpuout.argmax(dim=1, keepdim=True)                 # for each sample, the model gives a score for each category. Choose the category that contains the highest score as the prediction
          correct += pred.eq(labels.view_as(pred)).sum().item()     # number of correct predictions

      accuracy = correct / total                                    # accuracy = total correct / total number of samples

    return accuracy


# This function is simply to organize the whole training procedure. It iterates through all the epochs, and evaluates the model after each epoch, so you can see the training progress.

# In[ ]:


def train_modelcv(dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device):

  for epoch in range(num_epochs):                                                           # iterates through specified number of epochs
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train(True)                                                                           
    train_loss=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )    # train model

    model.train(False)
    measure = evaluate(model, dataloader_cvtest, device)                                    # evaluate model

    print('Top 1 Accuracy:', measure, "\n")                                                 # print accuracy for current epoch
  return None


# Let's start training

# In[ ]:


batchsize=64                                                            # number of images being passed to the model at any one time. If you face cuda memory error, reduce batchsize.
lr = 0.01                                                               # how "fast" you want your model to learn. Very important to play around with different values.


data_transform = transforms.Compose([transforms.Resize((224,224)),      # images are of different size, reshape them to same size. Up to you to decide what size to use.
                                     transforms.ToTensor(),             # convert image to torch tensor
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])   # normalizing images helps improve convergence

path = "./train/train"                                                  # where your training images are
dataset_train = ShopeeDataset(path, data_transform,train=True)          # training dataset
dataset_valid = ShopeeDataset(path, data_transform,train=False)         # validation dataset to measure current model performance. (rough gauge)

loadertr=torch.utils.data.DataLoader(dataset_train,batch_size=batchsize,shuffle=True)       # dataloader for training set
loaderval=torch.utils.data.DataLoader(dataset_valid,batch_size=batchsize,shuffle=True)      # dataloader for validation set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # which device to use. If GPU is detected, use GPU, else, use CPU
criterion = nn.CrossEntropyLoss()                                       # Loss function

model = models.resnet18(pretrained=True)                                # transfer learning, Pytorch has a pretrained model trained on ImageNet. Turn this off if you want to, it may or may not help in the end. Helps achieve higher accuracy with less epochs though
model.fc = nn.Linear(512, 42)                                           # as we only have 42 classes, change the last layer to predict on 42 classes.
model.to(device)                                                        # send model to device, and hopefully your device is GPU

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
maxnumepochs=2                                                          # number of epochs to train on. Should most definitely increase this to train more.
train_modelcv(dataloader_cvtrain = loadertr,                            # lets start training
              dataloader_cvtest = loaderval,  
              model = model,  
              criterion = criterion, 
              optimizer = optimizer,
              scheduler = None,
              num_epochs = maxnumepochs,
              device = device)


# Create Dataset class to load testing images

# In[ ]:


class ShopeeTestDataset(Dataset):
    def __init__(self,directory,transform):
        self.transform = transform
        self.img_ls = list()

        ls = sorted(os.listdir(directory))
        for img_name in ls:
            self.img_ls.append((os.path.join(directory,img_name)))
        
    def __getitem__(self, idx):
        name = self.img_ls[idx]
        img = Image.open(name).convert('RGB')
        img.load()
        img = self.transform(img)
        return img,name[12:]
    
    def __len__(self):
        return len(self.img_ls)


# This function collates the predictions for test images and returns a pandas dataframe in the format specified by Shopee

# In[ ]:


def inference(model, dataloader, device):
    # df = pd.DataFrame()
    path_ls = list()
    pred_ls = list()
    model.eval()
    with torch.no_grad():
      for ctr, data in enumerate(dataloader):
          inputs,path = data
          inputs = inputs.to(device)
          outputs = model(inputs)
          cpuout= outputs.to('cpu')
          pred = cpuout.argmax(dim=1).numpy()
          path_ls += list(path)
          pred_ls += list(pred)
    df = pd.DataFrame({'filename':path_ls,'category':pred_ls})

    return df


# Let's start predicting on the test images

# In[ ]:


test_dataset = ShopeeTestDataset("./test/test", data_transform)
loaderte=torch.utils.data.DataLoader(test_dataset,batch_size=256,shuffle=False)
df = inference(model,loaderte,device)                               # predictions are stored in this dataframe
df.to_csv("./gdrive/My Drive/test_prediction.csv",index=False)      # export dataframe as csv. It will export to your Google drive home folder


# Quick peek at the dataframe

# In[ ]:


df.head()


# In[ ]:




