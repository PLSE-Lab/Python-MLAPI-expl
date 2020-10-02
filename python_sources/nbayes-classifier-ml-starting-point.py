#!/usr/bin/env python
# coding: utf-8

# Iris species classification using Gaussian Naive Bayes
# ------------------------------------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from sklearn.metrics import accuracy_score
#Will be used to test the accuracy of the model predictions
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
# Any results you write to the current directory are saved as output.


# In[ ]:


iris=pd.read_csv('../input/Iris.csv',index_col=0)
iris.head()


# In[ ]:


iris.head()
def split_data(dataset,ratio):
    #To split the dataset into train and test dataset
    sample=np.random.rand(len(dataset))<ratio
    return(dataset[sample],dataset[~sample])


# LOOKING AND VISUALIZING THE RELATIONSHIPS BETWEEN THE FEATURES

# The following lines of code will give correlations between the various features. Its important to see which features are highly correlated.

# In[ ]:


print (np.corrcoef(iris['SepalLengthCm'],iris['SepalWidthCm']))
print (np.corrcoef(iris['SepalLengthCm'],iris['PetalLengthCm']))
print (np.corrcoef(iris['SepalLengthCm'],iris['PetalWidthCm']))
print (np.corrcoef(iris['SepalWidthCm'],iris['PetalLengthCm']))
print (np.corrcoef(iris['SepalWidthCm'],iris['PetalWidthCm']))
print (np.corrcoef(iris['PetalLengthCm'],iris['PetalWidthCm']))


# One of the best ways to visualize bivariate relationships is using the Seaborn library and using PairGrid and PairPlot methods. They let you create bivariate and univariate plot between the features in a dataset. A good starting step for Exploratory data analysis.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.PairGrid(iris,hue='Species',size=2).map_diag(sns.kdeplot).map_offdiag(plt.scatter)


# In[ ]:


train,test=split_data(iris,0.80)
shape=train.shape[1]
clf=GaussianNB() #Creates a Gaussian NB classifer
clf.fit(train[[0,1,2,3]],train[[4]]) #Fitting the classifier on the training dataset.
pred=clf.predict(np.array(test[[0,1,2,3]])) #Predicting for the test dataset.
print('Accuracy of prediction:',accuracy_score(test[[4]],pred))#Getting the accuracy of the predictions


# **CONCLUSION**
# Naive bayes being a very simple classifer has an accuracy greater than 90% on this dataset. Here, Gaussian distribution has been used. Apart from Gaussian, other distributions can be used which adhere to the data more aptly. I will be learning and using Support Vector machines and applying them for iris data classification. Will post a comparison between NBayes and SVM when its done.
