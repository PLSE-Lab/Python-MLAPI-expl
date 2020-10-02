#!/usr/bin/env python
# coding: utf-8

# First, let's load the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

glassdata = pd.read_csv('../input/glass.csv') 

np.unique(glassdata['Type'])
# Notice that in the description, it has 1-7 types, but the dataset does not have Type 4


# In[ ]:


# look at the first 5 rows of data and their names 
print(glassdata.shape)
glassdata.head()


# In[ ]:


# Store unqiue types
types = glassdata.Type.unique()

alpha = 0.7 # training data ratio

# Splitting glassdata to training and test data
train = pd.DataFrame()
test = pd.DataFrame()
for i in range(len(types)):
    tempt = glassdata[glassdata.Type == types[i]]
    train = train.append(tempt[0:int(alpha*len(tempt))])
    test = test.append(tempt[int(alpha*len(tempt)): len(tempt)])
    # test.append(tempt[int(alpha*len(tempt)): len(tempt)])

# Check whether the dimension match
print (train.shape, test.shape, glassdata.shape)


# In[ ]:


# Take a look at the train data
train.describe()


# In[ ]:


# Both Ba and Fe has too many 0, we will ignore them

print ((train['Ba']==0).sum()/len(train))
print ((train['Fe']==0).sum()/len(train))


# In[ ]:



# Construct correlation matrix for all variables, since Ba and Fe are mostly 0s, we can ignore them

train_variables = train.drop(['Type','Ba', 'Fe'],1)
train_variable_corrmat = train_variables.corr()
print (train_variable_corrmat)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))
sns.pairplot(train_variables,palette='coolwarm')
plt.show()


# In[ ]:


# Visualize the corrlation matrix

corr = train_variables.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar = True,  square = True, annot=True,
           xticklabels= corr.columns.values, yticklabels= corr.columns.values,
           cmap= 'coolwarm')
plt.show()

print(corr)


# In[ ]:


# Find two variables that has the least correlation 
# (negative correlation counts as a 'good' correlation) 
# So we will find two variables whose correlation is closest to 0
import numpy as np
corr_min = abs(train_variable_corrmat).values.min()
np.where(abs(train_variable_corrmat)==corr_min)

train_variables.columns[[3,4]]


# In[ ]:


# Use Logistic Regression on All data
X = train.drop('Type',1)
Y = train['Type']
Z = test.drop('Type',1)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(Z)
# prediction = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(Z)

prediction


# In[ ]:


# Calculate Accuracy
truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction == truth)/(len(truth))
accuracy

# Pretty bad


# In[ ]:


# Use Logistic Regression on Al and Si only
X1 = train[['Al', 'Si']]
Y1 = train['Type']
Z1 = test[['Al', 'Si']]


prediction1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X1, Y1).predict(Z1)


truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction1 == truth)/(len(truth))
accuracy

# Still bad


# In[ ]:


# It seems like by selecting Al and Si, the prediction is not any better.
# Let's plot the histagram for each variables with respect types 
import matplotlib.pyplot as plt

train.columns


# In[ ]:


# Average Bar Plot for each variables 
types = np.unique(train['Type'])

for i in range(len(types)):
    fig = plt.figure()
    average = train[[train.columns[i], "Type"]].groupby(['Type'],as_index=False).mean()
    sns.barplot(x = 'Type', y = train.columns[i], data= average)


# In[ ]:


# By looking at the mean plot above, we can see the significant difference in Mg, Al and K. 
# Go ahead and plot them in 3D 

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

for i in range(len(types)+1):
    count = i+1
    train_tempt = train.loc[train['Type'] == count]
    x = train_tempt['Mg']
    y = train_tempt['Al']
    z = train_tempt['K']
    
    ax.scatter(x, y, z, c= [float(i)/float(len(types)), 0.0, float(len(types)-i)/float(len(types))], marker='o')
    
    ax.set_xlabel(str('Mg'))
    ax.set_ylabel(str('Al'))
    ax.set_zlabel(str('K')) 

    
   
plt.show()


# In[ ]:


# Since there is a red dot that is higher than the other dots, which mess up the scale
# it will be better to show the figures individually

from mpl_toolkits.mplot3d import Axes3D



for i in range(len(types)+1):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    count = i+1
    train_tempt = train.loc[train['Type'] == count]
    x = train_tempt['Mg']
    y = train_tempt['Al']
    z = train_tempt['K']
    
    ax.scatter(x, y, z, c= [float(i)/float(len(types)), 0.0, float(len(types)-i)/float(len(types))], marker='o')
    
    ax.set_xlabel(str('Mg')+str(i+1))
    ax.set_ylabel(str('Al')+str(i+1))
    ax.set_zlabel(str('K')+str(i+1)) 

    
   
plt.show()


# In[ ]:


# Use Logistic Regression on Al, K and Mg
X1 = train[['Al', 'K', 'Mg']]
Y1 = train['Type']
Z1 = test[['Al', 'K', 'Mg']]

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

prediction1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X1, Y1).predict(Z1)


truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction1 == truth)/(len(truth))
print (accuracy)
print (prediction1)
print (truth)


# In[ ]:


# It's getting better, but still not good at all. 
# One of the reasons is that it has very few data for type 3,5 and 6.
# We need to find a way to fix it NEXT TIME

# However, we can still try change alpha to see what will happen

# Store unqiue types
types = glassdata.Type.unique()

alpha = 0.65 # training data ratio

# Splitting glassdata to training and test data
train = pd.DataFrame()
test = pd.DataFrame()
for i in range(len(types)):
    tempt = glassdata[glassdata.Type == types[i]]
    train = train.append(tempt[0:int(alpha*len(tempt))])
    test = test.append(tempt[int(alpha*len(tempt)): len(tempt)])
    

# Use Logistic Regression on All data
X = train.drop('Type',1)
Y = train['Type']
Z = test.drop('Type',1)

prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(Z)
# prediction = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(Z)

truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction == truth)/(len(truth))
print ('The default prediction is '+ str(prediction))
print ('The true value is '+ str(truth))
print ('The default accuracy is ' + str(accuracy))

# ====================================================

# Use Logistic Regression on Al and Si
X1 = train[['Al', 'K', 'Mg']]
Y1 = train['Type']
Z1 = test[['Al', 'K', 'Mg']]

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

prediction1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X1, Y1).predict(Z1)


truth = test['Type']
truth = np.array(truth)

accuracy1 = sum(prediction1 == truth)/(len(truth))
print ('The feature selection prediction is '+ str(prediction1))
print ('The true value is '+ str(truth))
print ('The feature selection accuracy is ' + str(accuracy1))


# In[ ]:


Check https://www.kaggle.com/drwhohu/d/uciml/glass/classification-after-removing-outliers/ for continued analysis.

