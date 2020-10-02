#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train.info()
train[0:10]


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'].plot('bar', color='r',width=0.3,title='Survived', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('Survived')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'][[1,2]])
print(train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'][[4,5,6]])


# In[ ]:


#Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["Survived"],  # Make a crosstab
                              columns="count")      # Name the count column

train_outcome


# In[ ]:


# for column
train['Age'] = train['Age'].replace(np.nan, 0)

# for whole dataframe
train = train.replace(np.nan, 0)

# inplace
train.replace(np.nan, 0, inplace=True)

print(train)


# In[ ]:


#Select feature column names and target variable we are going to use for training
Sex = {'male': 1,'female': 2} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
train.Sex = [Sex[item] for item in train.Sex] 
print(train)


# In[ ]:


test.info()
test[0:10]


# In[ ]:


print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")


# In[ ]:


# for column
test['Age'] = train['Age'].replace(np.nan, 0)

# for whole dataframe
test = test.replace(np.nan, 0)

# inplace
test.replace(np.nan, 0, inplace=True)

print(test)


# In[ ]:


#Select feature column names and target variable we are going to use for training
Sex = {'male': 1,'female': 2} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
test.Sex = [Sex[item] for item in test.Sex] 
print(test)


# In[ ]:



features=['Sex','Age']
target = 'Survived'


# In[ ]:


#This is input which our classifier will use as an input.
train[features].head(10)


# In[ ]:


#Display first 10 target variables
train[target].head(10).values


# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000, random_state=42)

# We train model
mlp.fit(train[features],train[target]) 

# We train model
mlp.fit(train[features],train[target]) 


# In[ ]:


#Make predictions using the features from the test data set
predictions = mlp .predict(test[features])

#Display our predictions
predictions


# In[ ]:


# Test score
#score_svmcla = svmcla.score(test[features])
#print(score_svmcla)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

