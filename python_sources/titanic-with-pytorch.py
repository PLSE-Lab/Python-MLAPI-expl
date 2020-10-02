#!/usr/bin/env python
# coding: utf-8

# # Titanic Machine Learning from Disaster
# 
# 1.   Introduction
# 2.   Data Loading
# 3.   Data Cleaning/ Preprocessing
# 4.   Exploratory Data Analysis or EDA
# 5.   Feature extraction
# 6.   Model Building
# 

# ## Introduction : A short history of Titanic
# 
# RMS Titanic was a passenger liner that struck an iceberg on her maiden voyage from Southampton, England, to New York City, and sank on 15th April 1912, resulting in the deaths of 1,517 people in one of the deadliest peacetime maritime disasters in history.Titanic True Story
# 
# The largest passenger steamship in the world at the time, the Olympic-class RMS Titanic was owned by the White Star Line and constructed at the Harland and Wolff shipyard in Belfast, Ireland, UK.
# 
# After setting sail for New York City on 10th April 1912 with 2,223 people on board, she hit an iceberg four days into the crossing, at 11:40 pm on 14th April 1912, and sank at 2:20 am on the morning of 15th April.
# 
# The high casualty rate resulting from the sinking was due in part to the fact that, although complying with the regulations of the time, the ship carried lifeboats for only 1,178 people. A disproportionate number of men died due to the 'women and children first' protocol that was enforced by the ship's crew.
# (credit : funny-jokes.com)
# 
# In this challenge, we will try to predict what sorts of people were likely to survive. Here, we will use Deep Neural Network using PyTorch to predict which passengers survived the tragedy.
# 

# ## Loading the data set with necessary libraries

# In[ ]:


# For Linear Algebra and Neumerical analysis
import numpy as np 
# For Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# For ploting our data and results 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# For creating the neural network
import torch
from torch import nn, optim
import torch.functional as F

# For spliting the data into training and dev sets
from sklearn.model_selection import train_test_split 

# For randomly shuffling the data
from sklearn.utils import shuffle

import os

# Our avialabe data files
print(os.listdir("../input"))


# Loading the Training and Testing files

# In[ ]:


train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


# ## Data Cleanning/Preprocessing
# 
# 

# Let's have a look at our data first

# In[ ]:


train.head()


# In[ ]:


test.head()


# Checking the information about our training data

# In[ ]:


print("Train info :")
train.info()


# Checking the information about our testing data

# In[ ]:


print("Test info :")
test.info()


# Let's decribe the Training set

# In[ ]:


train.describe()


# Let's describe the test data

# In[ ]:


test.describe()


# The numerical columns present in our training set

# In[ ]:


train.select_dtypes(include=[np.number]).head()


# The catogorical columns present in our training set

# In[ ]:


train.select_dtypes(include=[np.object]).head()


# <br>We have six  Numerical   columns: PassengerId, Survived, Age, SibSp, Parch, Fare </br>
# <br>We have five categorical columns: Name, Sex, Ticket, Cabin, Embarked </br>

# #### Let's map the categorical values to numerical values

# Male to 1 and Female to 0

# In[ ]:


train['Sex'] = train['Sex'].map({'male':1, 'female':0})
test['Sex']  = test[ 'Sex'].map({'male':1, 'female':0})
train.head()


# The ticket column contains two values

# In[ ]:


train.Ticket.head()


# The first part seems like some kind of class and the other part is the ticket number so let's create two seperate columns for them

# In[ ]:


def sp(st):
    li = st.split(' ')
    if len(li) > 1:
        return li
    word = 'NA' # Not Available
    return word, li[0]


# Our function is ready now let's split the columns into two parts

# In[ ]:


train['Class'], train['TicketNo'] = zip(*train['Ticket'].map(sp)) 
test['Class'], test['TicketNo'] = zip(*test['Ticket'].map(sp))


# Let's look at the result

# In[ ]:


train.head()


# Looks good and now we don't need the ticket anymore

# In[ ]:


train = train.drop('Ticket', 1)
test  = test.drop('Ticket',  1)
train.head()


# Now let's map the values of class

# In[ ]:


train['Class'] = train['Class'].map(dict(zip(train.Class.unique(), range(len(train.Class.unique())))))
test['Class'] = test['Class'].map(dict(zip(test.Class.unique(), range(len(test.Class.unique())))))
train.head()


# <br>Let's map the embarked column or feature. There are three value in the embarked feature. </br>
# <br>C : Cherbourg, Q : Queenstown, S : Southampton </br>
# <br>Let's map C as 1, Q as 2, S as 3 </br>

# In[ ]:


train['Embarked'] = train['Embarked'].map({'C':1,'Q':2,'S':3})
test['Embarked']  = test[ 'Embarked'].map({'C':1,'Q':2,'S':3})
train.head()


# A function to check for the NaN(not a number) values

# In[ ]:


def nan_count(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


# In[ ]:


nan_count(train)


# In[ ]:


nan_count(test)


# Since more than 70% data of Cabin is NaN in both train and test, droping it is the best idea

# In[ ]:


train = train.drop('Cabin', 1)
test  = test.drop('Cabin', 1)


# Also the names will just cause noise so also droping them

# In[ ]:


train = train.drop('Name', 1)
test  = test.drop('Name', 1)


# In[ ]:


train = train.drop('PassengerId', 1)
test  = test.drop('PassengerId', 1)


# In[ ]:


train.head()


# **Now we fill the missing values in Age, Embarked and Fare**
# <br>Age with Mean since it's numerical value </br>
# <br>Embarked with Mode since it's a catogorical value </br>
# <br>Fare with Mean since it's numerical value </br>

# In[ ]:


train["Age"] = train.Age.fillna(train.Age.mean()) 
train['Embarked'] = train.Embarked.fillna(train.Embarked.mode()) 

test["Age"] = test.Age.fillna(test.Age.mean()) 
test['Embarked'] = test.Embarked.fillna(test.Embarked.mode())
test["Fare"] = test.Fare.fillna(test.Fare.mean())


# In[ ]:


nan_count(train)


# In[ ]:


nan_count(test)


# A bug that will be fixed later

# In[ ]:


train.TicketNo[train.TicketNo == 'Basle'] = 12
train.TicketNo[train.TicketNo == 'LINE'] = 0


# In[ ]:


train.TicketNo = train.TicketNo.astype(float)


# In[ ]:


test.TicketNo = test.TicketNo.astype(float)


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# Ploting the heat map of the corelation of our data

# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# A deeper look of the corelation of every column with respect to our target column

# In[ ]:


abs(corrmat["Survived"][1:]).plot(kind='bar',stacked=True, figsize=(10,5))


# ## Exploratory Data Analysis or EDA
# According to  Wikipedia,  Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. <br>
# Already we did a little bit of data preprocessing. Here we will describe the data-set using statistical and visualization techniques to bring important aspects of that data into our focus for further analysis. <br>
# 
# Author : Riday

# In[ ]:


# Let's see the survival rate based on Sex
train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)


# In[ ]:


def barChart(Feature):
   survived = train[train['Survived']==1][Feature].value_counts()
   dead = train[train['Survived']==0][Feature].value_counts()
   df = pd.DataFrame([survived,dead])
   df.index = ['Survived','Dead']
   df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


# Let's Draw the Bar Chart to see the survival based on Sex
barChart('Sex')


# In[ ]:


# Let's see the survival rate based on Pclass
train[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)


# In[ ]:


# Let's Draw the Bar Chart to see the survival based on Pclass
barChart('Pclass')


# In[ ]:


# Let's see the survival rate based on Embarked
train[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)


# In[ ]:


# Let's Draw the Bar Chart to see the survival based on Embarked
barChart('Embarked')


# In[ ]:


train.head()


# In[ ]:


train = shuffle(train)
train.head()


# In[ ]:


target = np.array(train.Survived).reshape(len(train), 1)
feat = np.array(train.drop('Survived', 1))[:, :-2]
feat.shape, target.shape


# In[ ]:



feat_train, feat_test, target_train, target_test = train_test_split(feat, target, test_size = 0.07)
target_train.shape, target_test.shape


# In[ ]:


feat_train = torch.from_numpy(feat_train).float().detach().requires_grad_(True)
target_train = torch.from_numpy(target_train).float().detach().requires_grad_(False)

feat_test = torch.from_numpy(feat_test).float().detach().requires_grad_(True)
target_test = torch.from_numpy(target_test).float().detach().requires_grad_(False)
target_train.shape, target_test.shape


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 128)
        self.a1  = nn.ReLU()
#         self.fc2 = nn.Linear(128, 12)
#         self.a2  = nn.ReLU()
        self.output = nn.Linear(128, 1)
        self.aO = nn.Sigmoid()
        
    def forward(self, x):
        x = self.a1(self.fc1(x))
#         x = self.a2(self.fc2(x))
        x = self.aO(self.output(x))
        
        return x

model = Net()
model


# In[ ]:


opti = optim.Adam(model.parameters(), lr=0.03)
criterion = nn.BCELoss()


# In[ ]:


train_loss = []
test_loss  = []

train_acc = []
test_acc  = []


# In[ ]:


D = 200
for epoch in range(4000):
    opti.zero_grad()
    pred = model(feat_train)
    
    loss = criterion(pred, target_train)
    
    loss.backward()
    opti.step()
    
    if not (epoch%D):
        train_loss.append(loss.item())
        
        pred = (pred > 0.5).float()
        acc  = pred == target_train
        train_acc.append(acc.sum().float()/len(acc))
        
    # Calculating the validation Loss
    with torch.no_grad():
        model.eval()
        pred = model(feat_test)
        tloss = criterion(pred, target_test)
        if not (epoch%D):
            test_loss.append(tloss.item())
            
            pred = (pred > 0.5).float()
            acc  = pred == target_test
            test_acc.append(acc.sum().float()/len(acc))
            print(F"{epoch:5d}  |  train accuracy: {train_acc[-1]:0.4f}  |  test accuracy: {test_acc[-1]:0.4f}  |  train loss: {train_loss[-1]:0.4f}  |  test loss: {test_loss[-1]:0.4f}")
    model.train()
            
print("DONE!")


# In[ ]:


plt.plot(train_loss, label='Training loss')
plt.plot(test_loss, label='Validation loss')
plt.legend(frameon=False)


# In[ ]:


plt.plot(train_acc, label='Training accuracy')
plt.plot(test_acc,  label='Validation accuracy')
plt.legend(frameon=False)


# In[ ]:


# Saving the model
torch.save(model.state_dict(), 'checkpoint.pth')


# In[ ]:


# Loading the model
state_dict = torch.load('checkpoint.pth')
# print(state_dict)
model.load_state_dict(state_dict)


# In[ ]:


test_acc[-5:]


# In[ ]:


train_acc[-5:]


# In[ ]:


test_loss[-5:]


# In[ ]:


train_loss[-5:]


# In[ ]:


pid = test.PassengerId
test = test.drop('PassengerId', 1)


# In[ ]:


test.head()


# In[ ]:


test = np.array(test)[:, :-2]


# In[ ]:


test_tensor = torch.from_numpy(test).float().detach().requires_grad_(True)


# In[ ]:


sol = model(test_tensor)


# In[ ]:


sol[:10]


# In[ ]:


sol = sol > 0.5


# In[ ]:


sub = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


sub.head()


# In[ ]:


sub['Survived'] = sol.detach().numpy()


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv("Sollution.csv", index=False)

