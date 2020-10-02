#!/usr/bin/env python
# coding: utf-8

# > # **What Is Our Goal ?**
# In this kernel we will be showing you step by step how to build 2 machine learning models to classify an athelete's gender using the following features:
# - Age
# - Height 
# - Weight
# - Season
# 
# -----
# 
# #  __Table of contents__
# 1. [Python Packages Import](#Python-Packages-Import)
# 2. [Data Import](#Data-Import)
# 3. [ Data Visualization and Overview](#Data-Visualization-and-Overview)
# 4. [ Model 1 - Decision Tree](#Data-Visualization-and-Overview)
#     1. [ Creating Training Set and Test Set for Our Model](#decisionTree1)
#     2. [Training](#decisionTree2)
#     3. [Testing](#decisionTree3)
#     4. [Results](#decisionTree4)
# 5. [ Model 2 - Neural Network](#nn)
#     1.  [ Creating Training Set and Test Set for Our Model](#nn1)
#     2. [Defining the Model](#nn1.2)
#     3.  [Training](#nn2)
#     4. [Testing](#nn3)
#     5. [Results](#nn4)
# 6. [ Summary](#summary)
# 7. [Thanks for reading](#summary2)
# 
# ![Male And Female](https://imgur.com/4Astb6F.png)
# 
# 

# 
# ##  Python Packages Import <a name="Python-Packages-Import"></a>

# In[ ]:


import torch
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Import <a name="Data-Import"></a>
# There are 15 columns: 
# - ID
# - Name
# - Sex
# - Age
# - Height
# - Weight
# - Team
# - NOC
# - Games
# - Year
# - Season
# - City
# - Sport
# - Event
# - Medal
# 

# In[ ]:


athlete = pd.read_csv(filepath_or_buffer='../input/athlete_events.csv')
noc_regions = pd.read_csv(filepath_or_buffer='../input/noc_regions.csv')
athlete.head(5)


# ## Data Visualization and Overview <a name="Data-Visualization-and-Overview"></a>
# In this section I will plot some of the columns' value distributions.
# 
# Our goal in visualizing the data is to see how the values distribute regardless of gender and then spliting the data into male and female, hopefully to see a difference.

# In[ ]:


#Dropping records with NaN
dropped_nan_height = athlete["Height"].dropna()
dropped_nan_weight = athlete["Weight"].dropna()
dropped_nan_age = athlete["Age"].dropna()
#Figure
fig = pyplot.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
pyplot.subplots_adjust(left=1, right=3)
#Plots
sns.distplot(dropped_nan_height, ax=ax1)
sns.distplot(dropped_nan_weight, ax=ax2)
sns.distplot(dropped_nan_age, ax=ax3)


# # Now with respect to Season and Gender
# We can see a few intresting observations :
# 
# Age,Weight,Height the distribution of age, weight, and height behaves approximately like a normal distribution, and therefore.
# 
# With respect to season:
# 1. Winter
#     1. Male athletes are generally taller than female atheletes
#     2. Male athletes significantlly weigh more than female athletes
#     3. Male and female athletes are around the same ages.
# 2. Summer
#     1. Male athletes are generally taller than female athletes but this time not by much (in the winter the difference is  more prominent)
#     2. Male athletes signicantly weigh more than female athletes, in addition the IQR in males weight is bigger than the females.
#     3. Same as in the winter, male and female athletes are around the same ages.

# In[ ]:


pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Height",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Height"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)


# In[ ]:


pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Weight",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Weight"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)


# In[ ]:


pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Age",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Age"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)


# # Machine Learning
# After looking at the data, our intuition told me to try a decision tree before rushing to implement a neural network. We felt  this task can be easily done with a decision tree since we see a significent difference between male and female regarding width and height with respect to season.

# # Model 1 - Decision Tree <a name="decisionTree"></a>
# 

# ## Creating Training Set and Test Set for Our Model <a name="decisionTree1"></a>
# 
# We need to extract a sub dataframe with the features and the correct label.
# 
# Note that I decided that if an athlete is a male then we label his gender as 1 otherwise 0, same for season, summer is labeled as 1 and winter as 0.
# 
# The test set size is 20% of the whole dataset , around 41K records.
# 

# In[ ]:


data_with_predict = ["Season", "Weight", "Height", "Sex","Age"]
features = ["SeasonBinary", "Weight", "Height", "Age"]
data = athlete[data_with_predict].dropna()
data["BinarySex"] = (lambda x:  [1 if t=="M" else 0 for t in x])(data["Sex"])
data["SeasonBinary"] = (lambda x:  [1 if t=="Summer" else 0 for t in x])(data["Season"])
from sklearn.model_selection import train_test_split
from sklearn import tree
train, test = train_test_split(data, test_size=0.2)

X_train = train.as_matrix(columns=features)
Y_train = train.as_matrix(columns=["BinarySex"]).flatten()
X_test = test.as_matrix(columns=features)
Y_test = test.as_matrix(columns=["BinarySex"]).flatten()


# ## Training <a name="decisionTree2"></a>

# In[ ]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)


# ## Testing <a name="decisionTree3"></a>

# In[ ]:


clf.predict(X_test)


# 
# 
# 
# ## Results <a name="decisionTree4"></a>
# Around 80~81% 
# 

# In[ ]:


acc_percentage = np.array(Y_test == clf.predict(X_test)).astype(np.int).sum()/len(Y_test) * 100
print("Accuracy on Test: {:3.4f}%".format(acc_percentage))


# # Model 2 - Neural Network <a name="nn"></a>
# I have decided to use PyTorch, since I have only 4 features we chose not to use a big and deep neural network because we obviosuly would like to avoid overfitting.

# ## Creating Training Set and Test Set for Our Model <a name="nn1"></a>
# The test set size is 20% of the whole dataset , around 41K records.

# In[ ]:


train, test = train_test_split(data, test_size=0.2)
valid_idx = int(len(train) * 0.1)
valid = train[:valid_idx]

class OlympicsDataSet(Dataset):
    def __init__(self,train,transforms=None):
        self.X = train.as_matrix(columns=features)
        self.Y = train.as_matrix(columns=["BinarySex"]).flatten()
        self.count = len(self.X)
        # get iterator
        self.transforms = transforms

    def __getitem__(self, index):
        nextItem = Variable(torch.tensor(self.X[index]).type(torch.FloatTensor))

        if self.transforms is not None:
            nextItem = self.transforms(nextItem[0])

        # return tuple but with no label
        return (nextItem, self.Y[index])

    def __len__(self):
        return self.count # of how many examples(images?) you have
olympicDS = OlympicsDataSet(train)
validDS = OlympicsDataSet(valid)
train_loader = torch.utils.data.DataLoader(olympicDS,
            batch_size=250, shuffle=False)
valid_loader = torch.utils.data.DataLoader(validDS,
            batch_size=1, shuffle=False)
testDS = OlympicsDataSet(test)
test_loader = torch.utils.data.DataLoader(testDS,
            batch_size=1, shuffle=False)


# # Defining the Model <a name="nn1.2"></a>
# Layers sizes:
# - Input Layer = 4
# - FIrst Hidden Layer = 8
# - Second Hidden Layer = 4
# - Output Layer = 2
# 
# Each hidden layer uses ReLU function and the output layer uses LogSoftmax function.

# In[ ]:


epochs = 10
class DNN(nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, num_classes):
        super(DNN, self).__init__()
        self.z1 = nn.Linear(input_size, first_hidden_size) # wx + b
        self.relu = nn.ReLU()
        self.z2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.z3 = nn.Linear(second_hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.z1(x) # input
        out = self.relu(out)

        out = self.z2(out) # first hidden layer
        out = self.relu(out)

        out = self.z3(out) # second hidden layer

        out = self.log_softmax(out) # output
        return out

    def name(self):
        return "DNN"


# ## Training <a name="nn2"></a>
# Optimizer : Adam
# 
# Loss Function : Negative Log Likelihood
# 
# Model Hyper Parameters:
# - 10 Epochs
# - LR = 0.001
# 

# In[ ]:


def train_dnn(net, trainL, validL):
    count = 0
    accuList = []
    lossList = []
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for epc in range(1,epochs + 1):
        print("Epoch # {}".format(epc))
        vcount = 0
        total_loss = 0
        net.train()
        for data,target in trainL:
            optimizer.zero_grad()
            out = net(data)
            loss = F.nll_loss(out, target, size_average=False)
            pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            count += pred.eq(target.data.view_as(pred)).sum()
            # Backward and optimize
            loss.backward()
            # update parameters
            optimizer.step()
        net.eval()
        for data, target in validL:
            out = net(data)
            loss = F.nll_loss(out, target, size_average=False)
            total_loss += loss.item()
            pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            vcount += pred.eq(target.data.view_as(pred)).sum().item()
        
        accuList.append(100. * (vcount / len(validL)))
        lossList.append(total_loss / len(validL))
    
    return accuList, lossList
     
myNet = DNN(4, 8, 4, 2)
accuList, lossList = train_dnn(myNet, train_loader, valid_loader)


# 
# ## Testing <a name="nn3"></a>
# 

# In[ ]:


def test(net, loader):
    net.eval()
    vcount = 0
    count = 0
    total_loss = 0.0
    for data, target in loader:
        out = net(data)
        loss = F.nll_loss(out, target, size_average=False)
        total_loss += loss.item()
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        vcount += pred.eq(target.data.view_as(pred)).sum().item()
    return 100. * (vcount / len(loader)), total_loss / len(loader)
test_acc, test_loss = test(myNet, test_loader)
print("The test set accuracy is {:3.4f}% \n The average loss is : {}".format(test_acc, test_loss))


# 
# ## Results <a name="nn4"></a>
# 

# In[ ]:


pyplot.figure()
pyplot.plot(range(1, epochs + 1), accuList, "b--", marker="o", label='Validation Accuracy')
pyplot.legend()
pyplot.show()
pyplot.figure()
pyplot.plot(range(1, epochs + 1), lossList, "r", marker=".", label='Validation Loss')
pyplot.legend()
pyplot.show()


# # Summary <a name="summary"></a>

# We have seen that male and female athlete weight and height distributions have significant differences and we therefore created 2 ML models.
# 
# First model is a decision tree which achived 80~81% accuracy on the test set .
# 
# Second model is a neural network which achived 77~80% accuracy on the test set.
# 
# Surprisingly we didn't gain any benefit using neural network over a decision tree.
# 

# # Thanks for Reading <a name="summary2"></a>
#  We are new to kaggle , data and machine learning are our passion.
#  
# We would love to hear what you think about our work.
# ![Meow](https://pbs.twimg.com/media/CMl684GUkAASKzX.png)
# 
