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


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[ ]:


df.describe(include='all')


# In[ ]:


df.isna().sum()


# In[ ]:


plt.figure(figsize=(20, 10))
df['Age'].plot(kind='hist', bins=25)


# In[ ]:


df.describe()


# In[ ]:


dummy1 = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)
dummy2 = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)

df = pd.concat([df, dummy1, dummy2], axis=1)


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns

corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(240, 15, n=9),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


X = df[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']
X.head()


# In[ ]:


X['Fare'] = X['Fare']/X['Fare'].max()
X.head()


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(X, y, test_size=1/3)


# In[ ]:


train.head()


# In[ ]:


train_labels.head()


# In[ ]:


'''import torch as th
from torch import nn, optim

model = nn.Sequential(
    nn.Linear(5,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,2),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)'''


# In[ ]:


'''epochs =5 
for e in range(epochs):
    running_loss=0
    for i in range(len(X)):
        optimizer.zero_grad()
        
        output = model(train)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    model.train()
    print("Epoch: {}/{}..".format(e+1, epochs),
             "Training Loss: {:.3f}..".format(running_loss/len(X)))'''
    


# In[ ]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(X, y)


# In[ ]:


accuracy = model.score(X, y)
print('Accuracy = ' , accuracy)


# In[ ]:


test_df = pd.read_csv("../input/test.csv")
test_df.head()


# In[ ]:


d1 = pd.get_dummies(test_df['Sex'], prefix='Sex', drop_first=True)
d2 = pd.get_dummies(test_df['Embarked'], prefix='Embarked', drop_first=True)

test_df = pd.concat([test_df, d1, d2], axis=1)
test_df.head()


# In[ ]:





# In[ ]:


test_df['Fare'] = test_df['Fare']/test_df['Fare'].max()


# In[ ]:


test_set = test_df[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
test_set.isnull().sum()


# In[ ]:


test_set['Fare'] = test_set['Fare'].interpolate()


# In[ ]:


result = model.predict(test_set)
result


# In[ ]:


submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                             'Survived': result})
submission.head()


# In[ ]:


submission.to_csv('Submission.csv', index=False)


# In[ ]:




