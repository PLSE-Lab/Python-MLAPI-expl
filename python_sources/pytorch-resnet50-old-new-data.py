#!/usr/bin/env python
# coding: utf-8

# ## Aptos 2019 Blindness Detection
# 
# The goal of this competition is to use a set of supplied retina images to predict the severity of any diabetic retinopathy (diabetic blindness) that may be present, according to the following scale:
# * 0 - No DR
# * 1 - Mild
# * 2 - Moderate
# * 3 - Severe
# * 4 - Proliferative DR
# 
# I will be using Pytorch with a pre-trained ResNet-50 model to adjust its weights according to the training set images for predicting the final diagnoses on the test set.
# 
# Also, the training set supplied will be enhanced by adding cropped and resized versions of a previous DR competition held in 2015 (according to ilovescience's kernel https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped ). The current images will also be cropped and resized, similarly to the algorithm used in said kernel, to eliminate blackspace and reduce image size for performance. These modifications will also be replicated on the test set images. To load and modify the images, I will be using PIL and OpenCV.

# In[ ]:


import numpy as np
import pandas as pd
import os
import PIL
import cv2
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms

import matplotlib.pyplot as plt

print(os.listdir('../input/'))

import warnings
warnings.filterwarnings("ignore")


# #### Loading in Data
# 
# Before beginning, we want to load in the supplied datasets into corresponding Pandas dataframes and confirm the structure of the data fields is as expected.

# In[ ]:


# Directories to be accessed for images
train_dir = '../input/aptos2019-blindness-detection/train_images/'
test_dir = '../input/aptos2019-blindness-detection/test_images/'
supp_dir = '../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped/'


# Checking that the data is formatted as expected:

# In[ ]:


train_data = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
print("Training set has %d rows and %d columns" % train_data.shape)
train_data.head()


# In[ ]:


test_data = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
print("Test set has %d rows and %d columns" % test_data.shape)
test_data.head()


# In[ ]:


submission = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
print("Submission set has %d rows and %d columns" % submission.shape)
submission.head()


# As a quick sanity check, we want to confirm that the id_code column is identical for the test_data and submission dataframes:

# In[ ]:


if test_data["id_code"].equals(submission["id_code"]):
    print("The id_code columns are identical")


# With equality of each dataset ensured, we can now simply copy the submission dataframe onto the test dataframe (so that the diagnosis column is added to the test).

# In[ ]:


test_data = submission.copy()
test_data.head()


# Exploring the makeup of the training set:

# In[ ]:


sns.set(style='darkgrid')
sns.countplot(y = 'diagnosis',
              data = train_data)
plt.show()


# From this distribution, it is clear that the dataset is imbalanced in favor of DR-absent cases, as well as moderate DR cases.

# To enhance the training data, the supplemental data from the previous competition will be loaded in (specifically, the resized and cropped versions).

# In[ ]:


supp_data = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels_cropped.csv")
print("Test set has %d rows and %d columns" % supp_data.shape)
supp_data.head()


# The supplemental dataset is improperly labeled as is, this code fixes it:

# In[ ]:


supp_data = supp_data.drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1)
supp_data = supp_data.rename(columns = {"image":"id_code", "level":"diagnosis"})
supp_data.head()


# #### Image Retrieval
# 
# Now the images in each set of data will be loaded in, and the current competition's train and test data will be cropped according to the strategy in ilovescience's kernel (https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped) used to created the supplemental data added to this kernel.

# In[ ]:


def cropImage(image):
# Credits to ilovescience, https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped
# Crops out as much of the surrounding blackspace as possible by forming a bounding
# box around the lens view of the microscope
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = 20 # Grayscale cutoff, increasing speeds up function but may result in more cropping failures
    ret,gray = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x); y = int(y); r = int(r)
    #print(x,y,r)
    if r > 100:
        return output[max(y-r,0):-1 + max(y+r+1,0),max(x-r,0):-1 + max(x+r+1,0)]
    else:
        return image
    
def resizeImage(image, size):
    # Image resizing if needed, preserves aspect ratio
    width = image.shape[1]
    height = image.shape[0]
    if width > size and height > size:
        if width > height:
            newWidth = size
            newHeight = int(height * (newWidth/width))
        else:
            newHeight = size
            newWidth = int(width * (newHeight/height))
        newDims = newWidth, newHeight
        image = cv2.resize(image, newDims, interpolation = cv2.INTER_AREA)
    return image


# #### Running Model Using Pytorch
# 
# Credits go to Alexander Teplyunk for their kernel sections utilizing Pytorch (https://www.kaggle.com/ateplyuk/aptos-pytorch-starter-rnet50), which I adapted for my data setup.
# 
# Pytorch will be used to ingest the data and adjust weights on the pre-trained ResNet-50 model accordingly

# In[ ]:


def getImages(type, dataset):
    images = []
    if type == "train":
        extension = ".png"
        folder = train_dir
        # update_freq - Number of images to go through before providing progress updates
        update_freq = 250
        cropped = False
    elif type == "test":
        extension = ".png"
        folder = test_dir
        update_freq = 250
        cropped = False
    elif type == "supplemental":
        extension = ".jpeg"
        folder = supp_dir
        update_freq = 2500
        cropped = True
    else:
        print("Invalid folder provided, specify either 'train', 'test', or 'supplemental'.")
        return dataset
    ids = dataset['id_code']
    length = len(ids)
    size = 265 # Maximum dimension size to downscale to
    for i in range(length):
        if (i + 1) % update_freq == 0:
            print("Processed image %d of %d" % (i+1,length))
        id = ids[i]
        fileName = folder + id + extension
        im = cv2.cvtColor(cv2.imread(fileName),cv2.COLOR_BGR2RGB)
        im = resizeImage(im, size)
        if not cropped:
            im = cropImage(im)
        images.append(im)
    dataset["image"] = images
    return dataset


# Loading in each dataset:

# In[ ]:


print("Loading in training images...")
train_data = getImages("train", train_data)
print("Done loading training images!")


# In[ ]:


print("Loading in test images...")
test_data = getImages("test", test_data)
print("Done loading test images!")


# In[ ]:


print("Loading in supplemental images...")
supp_data = getImages("supplemental", supp_data)
print("Done loading supplemental images!")


# If the images were properly loaded, a sample image from each set should be displayed below:

# In[ ]:


plt.imshow(train_data["image"][0])
plt.grid(b=None)


# In[ ]:


plt.imshow(test_data["image"][0])
plt.grid(b=None)


# In[ ]:


plt.imshow(supp_data["image"][0])
plt.grid(b=None)


# #### Combining Old and New Data
# 
# To enhance the training set, the supplemental data from the previous blindness competition will be combined with the current training data, we will also see how adding this data affects the score distribution:

# In[ ]:


train_data = train_data.append(supp_data,ignore_index=True)
sns.set(style='darkgrid')
sns.countplot(y = 'diagnosis',
              data = train_data)
plt.show()


# It is clear that adding the supplemental data made the dataset even more imbalanced in favor of DR-absent cases, this may reduce the accuracy of the model on new data.

# In[ ]:


class ImageData(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        label = self.df.diagnosis[index]           
        
        image = self.df.image[index]
        image = self.transform(image)
        return image, label


# In[ ]:


data_transf = transforms.Compose([transforms.ToPILImage(mode='RGB'), 
                                  transforms.Resize(265),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor()])
train = ImageData(df = train_data, transform = data_transf)
train_loader = DataLoader(dataset = train, batch_size=32, drop_last=True, shuffle=True)


# In[ ]:


model = models.resnet50()
model.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))


# In[ ]:


# Freeze model weights
for param in model.parameters():
    param.requires_grad = False


# In[ ]:


# Changing number of model's output classes to 5
model.fc = nn.Linear(2048, 5)


# In[ ]:


# Transfer execution to GPU
model = model.to('cuda')


# In[ ]:


optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Train model\nloss_log=[]\nfor epoch in range(15):    \n    model.train()\n    for ii, (data, target) in enumerate(train_loader):       \n        data, target = data.cuda(), target.cuda()              \n        optimizer.zero_grad()\n        output = model(data)                    \n        loss = loss_func(output, target)\n        loss.backward()\n        optimizer.step()          \n        if ii % 1000 == 0:\n            loss_log.append(loss.item())       \n    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))")


# In[ ]:


test = ImageData(df = test_data, transform = data_transf)
test_loader = DataLoader(dataset = test, shuffle=False)


# #### Creating and Outputting predictions

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Prediction\npredict = []\nmodel.eval()\nfor i, (data, _) in enumerate(test_loader):\n    data = data.cuda()\n    output = model(data)  \n    output = output.cpu().detach().numpy()    \n    predict.append(output[0])')


# In[ ]:


submission['diagnosis'] = np.argmax(predict, axis=1)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

