#!/usr/bin/env python
# coding: utf-8

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


# # Load Data

# In[ ]:


dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')


# In[ ]:


dataset.columns


# # Chnage Name into 3 categories

# In[ ]:


dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title'] = pd.Series(dataset_title)
dataset['Title'].value_counts()
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Others')


# In[ ]:


dataset_title = [i.split(',')[1].split('.')[0].strip() for i in testset['Name']]
testset['Title'] = pd.Series(dataset_title)
testset['Title'].value_counts()
testset['Title'] = testset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Others')


# # Count Family Size

# In[ ]:


dataset['FN'] = dataset['SibSp'] + dataset['Parch'] + 1
testset['FN'] = testset['SibSp'] + testset['Parch'] + 1


# In[ ]:


def family(x):
    if x < 2:
        return 'S'
    elif x == 2:
        return 'C'
    elif x <= 4:
        return 'M'
    else:
        return 'L'
    
dataset['FN'] = dataset['FN'].apply(family)
testset['FN'] = testset['FN'].apply(family)


# # Fill NUll Value

# In[ ]:


dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
testset['Embarked'].fillna(testset['Embarked'].mode()[0], inplace=True)
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
testset['Age'].fillna(testset['Age'].median(), inplace=True)
testset['Fare'].fillna(testset['Fare'].median(), inplace=True)


# # Drop Columns

# In[ ]:


dataset = dataset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
testset_passengers = testset['PassengerId']
testset = testset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)


# In[ ]:


dataset.iloc[:10]


# In[ ]:


X_train = dataset.iloc[:, 1:9].values
Y_train = dataset.iloc[:, 0].values
X_test = testset.values


# In[ ]:


#print(Y_train.shape)
print(X_train.shape)


# ## Encode string to number

# In[ ]:


# Converting the remaining labels to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])
X_train[:, 6] = labelencoder_X_1.fit_transform(X_train[:, 6])


# In[ ]:


labelencoder_X_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])
X_test[:, 6] = labelencoder_X_2.fit_transform(X_test[:, 6])


# In[ ]:


print(X_test.shape)
print(X_train.shape)


# In[ ]:


X_test[:10, 3]


# In[ ]:


# Converting categorical values to one-hot representation
#0, 1, 4, 5, 6
one_hot_encoder = OneHotEncoder(categorical_features = [0, 1, 4, 5, 6])
X_train = one_hot_encoder.fit_transform(X_train).toarray()
X_test = one_hot_encoder.fit_transform(X_test).toarray()

print(X_test.shape)
print(X_train.shape)


# In[ ]:


print(X_test.shape)
print(X_train.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1)


# ## Define Model

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = F.dropout(x, p=0.4)
        x = F.leaky_relu(x)
        
        x = self.fc2(x)
        x = F.dropout(x, p=0.4)
        x = F.leaky_relu(x)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x
    
net = Net()


# In[ ]:


batch_size = 50
num_epochs = 181
learning_rate = 0.001
batch_no = len(x_train) // batch_size


# In[ ]:


criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# # Train

# In[ ]:


from sklearn.utils import shuffle
from torch.autograd import Variable

min_loss = np.Inf

for epoch in range(num_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    
    train_loss = 0
    valid_loss = 0
    
    for i in range(batch_no):
        x_var = Variable(torch.FloatTensor(x_train))
        y_var = Variable(torch.LongTensor(y_train))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss =criterion(ypred_var, y_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if epoch % 20 == 0:
        test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
        test_y_var = Variable(torch.LongTensor(y_val))
        with torch.no_grad():
            result = net(test_var)
            #loss
            loss = criterion(result, test_y_var)
            #calculate loss
            valid_loss += loss.item()
            
            values, labels = torch.max(result, 1)
            num_right = np.sum(labels.data.numpy() == y_val)
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, train_loss,valid_loss),
                'Accuracy {:.2f}%'.format((num_right / len(y_val))* 100),)
            
        # save model if validation loss has decreased
        if valid_loss <= min_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                min_loss,
                valid_loss))
            torch.save(net.state_dict(), 'model.pt')
            min_loss = valid_loss


# In[ ]:


net.load_state_dict(torch.load('model.pt'))


# # Test

# In[ ]:


# Evaluate the model
test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)
with torch.no_grad():
    result = net(test_var)
    values, labels = torch.max(result, 1)
    num_right = np.sum(labels.data.numpy() == y_val)
    print('Accuracy {:.2f}'.format(num_right / len(y_val)))


# # Find Survied

# In[ ]:


X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=True)
print(X_test_var.shape)
with torch.no_grad():
    test_result = net(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()


# In[ ]:


len(testset_passengers)
len(survived)


# In[ ]:


import csv

data = [['PassengerId', 'Survived']]
for i in range(len(survived)):
    data.append([testset_passengers[i], survived[i]])
    
with open('submission.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(data)
    
print("Complete")

