#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


get_ipython().run_line_magic('matplotlib', 'inline')
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import itertools
import os
import pandas as pd
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input/pytorch-pretrained-models-for-face-detection/"))
print(os.listdir("../input/recognizing-faces-in-the-wild/"))


# In[ ]:


#Checking if CUDA is available or not
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# # Helper Functions

# In[ ]:


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# # Pre processing relationships Data

# In[ ]:


df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")
df.head()


# We can see the data has two columns of people that are related. Let's split these columns into three i.e. Family, Person1 and Person2

# In[ ]:


new = df["p1"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family1"]= new[0]
# making separate last name column from new data frame 
df["Person1"]= new[1]

# Dropping old Name columns
df.drop(columns =["p1"], inplace = True)

new = df["p2"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family2"]= new[0]
# making separate last name column from new data frame 
df["Person2"]= new[1]

# Dropping old Name columns
df.drop(columns =["p2"], inplace = True)
df.head()


# It is essential to check if all the folders present in the above DataFrame exist, else it'll be an error while processing the dataset. We should remove the rows that are duplicate and the rows that don't have any folder existing corresponding to their values

# In[ ]:


root_dir = '../input/recognizing-faces-in-the-wild/train/'
temp = []
for index, row in df.iterrows():
    if os.path.exists(root_dir+row.Family1+'/'+row.Person1) and os.path.exists(root_dir+row.Family2+'/'+row.Person2):
        continue
    else:
        temp.append(index)
        
print(len(temp))
df = df.drop(temp, axis=0)


# There are 236 rows that don't have any folders corresponding to them so they have been removed

# We can also create some more data from Families for people that are not related. Then we can use this data to train our model on classes 'Related' and 'Not Related'. 'Not Related' instances can be numerous since there are many families that are not correlated. However, that would give a lot of data, We can create new data from people within families that are not related. Eg. A son and daughter would be related to their father and their mother, However, the mother and father won't be related themselves

# In[ ]:


#A new column in the existing dataframe with all values as 1, since these people are all related
df['Related'] = 1

#Creating a dictionary, and storing members of each family
df_dict = {}
for index, row in df.iterrows():
    if row['Family1'] in df_dict:
        df_dict[row['Family1']].append(row['Person1'])
    else:
        df_dict[row['Family1']] = [row['Person1']]
        
#For each family in this dictionary, we'll first make pairs of people
#For each pair, we'll check if they're related in our existing Dataset
#If they're not in the dataframe, means we'll create a row with both persons and related value 0
i=1
for key in df_dict:
    pair = list(itertools.combinations(df_dict[key], 2))
    for item in pair:
        if len(df[(df['Family1']==key)&(df['Person1']==item[0])&(df['Person2']==item[1])])==0         and len(df[(df['Family1']==key)&(df['Person1']==item[1])&(df['Person2']==item[0])])==0:
            new = {'Family1':key,'Person1':item[0],'Family2':key,'Person2':item[1],'Related':0}
            df=df.append(new,ignore_index=True)
        
#Storing rows only where Person1 and Person2 are not same
df = df[(df['Person1']!=df['Person2'])]

#len(df[(df['Related']==1)])

print(df['Related'].value_counts())


# From value_counts() it is visible that data is imbalanced and there are many more instances of 1 than 0. So, let's create some more instances of class 0 between two families, such that dataframe becomes balanced

# In[ ]:


extra = df['Related'].value_counts()[1]-df['Related'].value_counts()[0]
while extra>=0:
    rows = df.sample(n=2)
    first = rows.iloc[0,:]
    second = rows.iloc[1,:]
    
    if first.Family1!=second.Family1 and first.Family2!=second.Family2:
        new1 = {'Family1':first.Family1,'Person1':first.Person1,'Family2':second.Family1,'Person2':second.Person1,'Related':0}
        extra=extra-1
        if extra==0:
            break
        new2 = {'Family1':first.Family2,'Person1':first.Person2,'Family2':second.Family2,'Person2':second.Person2,'Related':0}
        extra=extra-1
        
        df=df.append(new1,ignore_index=True)
        df=df.append(new2,ignore_index=True)


# Now all rows with 1 are together and all rows with 0 are together, Rows should be shuffled

# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


df['Related'].value_counts()


# The dataset is now balanced

# In[ ]:


df.head()


# # Custom Dataset Class

# In[ ]:


class FamilyDataset(Dataset):
    """Family Dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.relations)
    
    def __getpair__(self,idx):
        pair = self.root_dir+self.relations.iloc[idx,0] + '/' + self.relations.iloc[idx,1],        self.root_dir+self.relations.iloc[idx,2] + '/' + self.relations.iloc[idx,3]
        return pair
    
    def __getlabel__(self,idx):
        return self.relations.iloc[idx,4]
    
    def __getitem__(self, idx):
        pair =  self.__getpair__(idx)
        label = self.__getlabel__(idx)
        
        first = random.choice(os.listdir(pair[0]))
        second = random.choice(os.listdir(pair[1]))
        
        img0 = Image.open(pair[0] + '/' + first)
        img1 = Image.open(pair[1] + '/'  + second)
#         img0 = img0.convert("L")
#         img1 = img1.convert("L")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return idx,img0,img1,label


# In[ ]:


train_df,valid_df = np.split(df, [int(.8*len(df))])


# In[ ]:


train_transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

valid_transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

train_dataset= FamilyDataset(df=train_df,root_dir="../input/recognizing-faces-in-the-wild/train/",transform=train_transform)
valid_dataset = FamilyDataset(df=valid_df,root_dir="../input/recognizing-faces-in-the-wild/train/",transform=valid_transform)


# # Visualising Data

# In[ ]:


vis_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)

example_batch = next(dataiter)
concatenated = torch.cat((example_batch[1],example_batch[2]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[3].numpy())


# # Defining Neural Network

# In[ ]:


# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(1, 4, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(4),
            
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(4, 8, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(8),

#             nn.ReflectionPad2d(1),
#             nn.Conv2d(8, 8, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(8),

#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(8*224*224, 1024),
#             nn.ReLU(inplace=True),

#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(512, 50),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(50, 25))
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(25,2)
#         )
        
#     def forward_once(self, x):
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
        
#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        
#         output = self.fc2(output1-output2)
        
#         return output


# # VGG Face Net

# In[ ]:


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward_once(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38
    
    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        #euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        difference = output1 - output2
        return difference

def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model


# In[ ]:


vggnet = vgg_face_dag(weights_path="../input/pytorch-pretrained-models-for-face-detection/VGG Face")
vggnet = vggnet.cuda()
for param in vggnet.parameters(): 
    param.requires_grad = False


# In[ ]:


# if train_on_gpu:
#     net = SiameseNetwork().cuda()
# else:
#     net= SiameseNetwork()


# # Classification Model

# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2622, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))
  
    def forward(self, x):
        x = self.fc1(x)
        return x


# In[ ]:


#model = Model().cuda()
#net = nn.Sequential(vgg_face_dag(weights_path="../input/pytorch-pretrained-models-for-face-detection/VGG Face").cuda(),Model().cuda())
net = Model().cuda()


# # Optimizer

# In[ ]:


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.008, momentum=0.9)


# # Configuration Class

# In[ ]:


class Config():
    training_dir = "../input/train/"
    testing_dir = "../input/test/"
    batch_size = 64
    train_number_epochs = 150
    num_workers = 8


# # Training and Validation Dataloader

# In[ ]:


# # percentage of training set to use as validation
# valid_size = 0.2

# # obtain training indices that will be used for validation
# num_train = len(train_dataset)
# indices = list(range(num_train))
# np.random.shuffle(indices)
# split = int(np.floor(valid_size * num_train))
# train_idx, valid_idx = indices[split:], indices[:split]

# # define samplers for obtaining training and validation batches
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,num_workers=Config.num_workers, batch_size = Config.batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset,shuffle=True,num_workers=Config.num_workers, batch_size = Config.batch_size)


# # Training the model

# In[ ]:


train_counter = []
train_loss_history = []
train_iteration_number= 0

valid_counter = []
valid_loss_history = []
valid_iteration_number= 0

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

train_class_correct = list(0 for i in range(2))
train_class_total = list(0 for i in range(2))

valid_class_correct = list(0 for i in range(2))
valid_class_total = list(0 for i in range(2))


# In[ ]:


for epoch in range(0,Config.train_number_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    net.train()
    for i, data in enumerate(train_loader,0):
        row, img0, img1 , label = data
        row, img0, img1 , label = row.cuda(), img0.cuda(), img1.cuda() , label.cuda()
        
        optimizer.zero_grad()
        output1= vggnet(img0,img1)
        output = net(output1)
        _, pred= torch.max(output,1)

        loss = criterion(output,label)
        loss.backward()
        
        optimizer.step()
        
        correct = pred.eq(label.view_as(pred))
        for j in range(len(label)):
                        target = label[j].data
                        train_class_correct[target] += correct[j].item()
                        train_class_total[target] += 1
                        
        if i%30 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch+1,loss.item()))
            train_iteration_number +=30
            train_counter.append(train_iteration_number)
            train_loss_history.append(loss.item())

            for i in range(2):
                if train_class_total[i] > 0:
                        print('\nTraining Accuracy of %5s: %2d%% (%2d/%2d)' % (
                            str(i), 100 * train_class_correct[i] / train_class_total[i],
                            np.sum(train_class_correct[i]), np.sum(train_class_total[i])))

            print('\nTraining Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(train_class_correct) / np.sum(train_class_total),
                np.sum(train_class_correct), np.sum(train_class_total)))
    
    net.eval()
    for i, data in enumerate(valid_loader,0):
        row, img0, img1 , label = data
        row, img0, img1 , label = row.cuda(), img0.cuda(), img1.cuda() , label.cuda()
        
        output1= vggnet(img0,img1)
        output = net(output1)
        #combined = torch.cat([vgg,res1,sen1],1)
        #output= net(combined)
        _, pred= torch.max(output,1)

        loss = criterion(output,label)
        
        correct = pred.eq(label.view_as(pred))
        for j in range(len(label)):
                        target = label[j].data
                        valid_class_correct[target] += correct[j].item()
                        valid_class_total[target] += 1
        valid_loss += loss.item()                
        if i%30 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch+1,loss.item()))
            valid_iteration_number +=30
            valid_counter.append(valid_iteration_number)
            valid_loss_history.append(loss.item())

            for i in range(2):
                if train_class_total[i] > 0:
                        print('\nValdiation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                            str(i), 100 * valid_class_correct[i] / valid_class_total[i],
                            np.sum(valid_class_correct[i]), np.sum(valid_class_total[i])))

            print('\nValdiation Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(valid_class_correct) / np.sum(valid_class_total),
                np.sum(valid_class_correct), np.sum(valid_class_total)))
            
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(net.state_dict(), 'model.pt')
        valid_loss_min = valid_loss


# # Training and Validation Loss

# In[ ]:


print('\n Training Loss History')
show_plot(train_counter,train_loss_history)

print('\n Validation Loss History')
show_plot(valid_counter,valid_loss_history)


# # Loading Saved Model

# In[ ]:


net.load_state_dict(torch.load('model.pt'))


# Now that our model is trained, we'll see how the sample submissions are

# In[ ]:


sample_submission = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")
sample_submission.head()


# In[ ]:


new = sample_submission["img_pair"].str.split("-", n = 1, expand = True)

# making separate first name column from new data frame 
sample_submission["Person1"]= new[0]
# making separate last name column from new data frame 
sample_submission["Person2"]= new[1]

# Dropping old Name columns
sample_submission.head()


# # Custom Test Dataset
# 

# In[ ]:


class FamilyTestDataset(Dataset):
    """Family Dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.relations)
    
    def __getpair__(self,idx):
        pair = self.root_dir+self.relations.iloc[idx,2],        self.root_dir+self.relations.iloc[idx,3]
        return pair
    
    def __getlabel__(self,idx):
        return self.relations.iloc[idx,4]
    
    def __getitem__(self, idx):
        pair =  self.__getpair__(idx)
        
        img0 = Image.open(pair[0])
        img1 = Image.open(pair[1])
#         img0 = img0.convert("L")
#         img1 = img1.convert("L")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return idx,img0,img1


# In[ ]:


transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

test_dataset= FamilyTestDataset(df=sample_submission,root_dir="../input/recognizing-faces-in-the-wild/test/",transform=transform)


# In[ ]:


test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,num_workers=Config.num_workers, batch_size = Config.batch_size)


# In[ ]:


net.eval()
for i, data in enumerate(test_loader,0):
    row, img0, img1 = data
    row, img0, img1 = row.cuda(), img0.cuda(), img1.cuda()
   
    output1= vggnet(img0,img1)
    output = net(output1)
    #output= net(img0,img1)
    _, pred= torch.max(output,1)
     
    count=0
    for item in row:
        sample_submission.loc[item,'is_related'] = pred[count].item()
        count+=1


# In[ ]:


sample_submission.drop(columns =["Person1","Person2"], inplace = True)


# In[ ]:


sample_submission['is_related'].value_counts()


# In[ ]:


output= sample_submission.to_csv('output.csv',index=False)


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


create_download_link(sample_submission)

