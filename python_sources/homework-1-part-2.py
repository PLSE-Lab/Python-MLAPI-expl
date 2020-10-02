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


#import libraries
import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl

#Load dataframe
df = pd.read_csv("../input/diabetes.csv")


# In[ ]:


#Check for Nulls
df.isnull().values.any()


# In[ ]:


#Remove rows where insulin value is zero, as that does not make sense
df=df[df.Insulin!=0]


# In[ ]:


#Visualize Correlation
cor = df.corr()
sb.heatmap(cor)
#This is not quite right yet.


# In[ ]:


#Delete the Skin column
df = df.drop(columns = ["SkinThickness"])


# In[ ]:


#Check data types ==> Change trues to 1 and Falses to 0
#I think this is already fine, but here is the code anyway:
df=df.replace(to_replace=True,value = 1)
df=df.replace(to_replace = False, value = 0)


# In[ ]:


#Check True/False Ratio
#Not sure how to do this


# In[ ]:





# In[ ]:


data = df.drop(columns = ["Outcome"])
data


# In[ ]:


#Split data into training and testing data sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)


# In[ ]:


train_outcomes = train[["Outcome"]].copy()
train_outcomes


# In[ ]:


train_data = train.drop(columns = ["Outcome"])
train_data


# In[ ]:


#Delete rows with 0 values
#This was done above, but here is the code again.
df=df[df.Insulin!=0]


# In[ ]:


#Train model with ID3
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = "entropy")
#I don't understand how the labels work in this method call.
clf = clf.fit(train_data,train_outcomes)
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph


# In[ ]:


#Train model with C4.5


# In[ ]:


#Use predict and metrics on the training data


# In[ ]:


#Use predict and metrics on the testing data


# In[ ]:


#Check out the confusion matrix


# In[ ]:


#Check out the classification report

