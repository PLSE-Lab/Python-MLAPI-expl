#!/usr/bin/env python
# coding: utf-8

# Approach taken:
# * After the data was read, it was checked for missing values and class imbalance. 
# * KDE plots were drawn for each column and Kruskal Wallis Test was conducted to determine the relevant features in determining the 'flag' class.
# * On detecting high multicollinearity between the features, PCA was used. 5 Principal COmponents were used. 
# * A SVM model, was built and the F1 score was calculated for said model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Libraries 

# In[ ]:


import seaborn as sns               
import matplotlib.pyplot as plt     
from scipy.stats import *                
from sklearn.decomposition import PCA   
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report


# # Data

# In[ ]:


#reading the dataset
data = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")   


# In[ ]:


#previewing the first 5 rows of the dataframe
data.head()


# In[ ]:


#obtaining the dimensions of the dataframe
data.shape


# In[ ]:


#column names of the dataframe
data.columns


# In[ ]:


#data type of each column
data.dtypes


# In[ ]:


#checking for missing values 
data.isna().sum()


# In[ ]:


#checking for class imbalance
new_data['flag'].value_counts()/new_data.shape[0]


# In[ ]:


#descriptive statistics for each column
data.describe()


# In[ ]:


#segregating the data into independent and dependent variables
X = data.drop(['flag','timeindex'],1)
y = data['flag']


# Feature selection

# In[ ]:


#plotting KDE plots to observe the distribution of each column for each category of 'flag'
for i, col in enumerate(X.columns):
    plt.figure(i)
    dat1=data[col][data['flag']==0]
    dat2 = data.currentBack[data['flag']==1]
    plt.ylabel('Probability')
    plt.xlabel(col)
    plt.title('KDE Plot for column '+ col)
    sns.kdeplot(dat1, label='Anomaly')
    sns.kdeplot(dat2, label='Normal')
    plt.show()


# In[ ]:


#to compute the Kruskal-Wallis test statistic to detect the relationship between y and each continuous variable
for i, col in enumerate(X.columns):
    dat1=data[col][data['flag']==0]
    dat2 = data.currentBack[data['flag']==1]
    print(kruskal(dat1,dat2))


# From the Kruskal Walllis test conducted and KDE plots plotted, it can be seen that all the features are significant in determining the class.

# Multicollinearity check

# In[ ]:


#correlation matrix
cor = round(X.corr(),2)
cor


# In[ ]:


#heatmap depicting the correlation between independent variables
sns.heatmap(cor, cmap="Blues", linewidths=0.3)


# Principal Components

# In[ ]:


#computing principal components to reduce multicollinearity
pca = PCA(n_components=5)     #5 principal components used
X=pca.fit_transform(X)


# In[ ]:


#calculating the variance explained by each principal component
var = pca.explained_variance_ratio_   
var


# In[ ]:


#calculating the cumulative variance explained by the principal components
cum_var = np.cumsum(pca.explained_variance_ratio_)*100
cum_var


# In[ ]:


#dataframe with the principal components
pca_X=pd.DataFrame(X)


# In[ ]:


#correlation matrix for the principal components, to confirm the removal of multicollinearity
cor_pca = round(pca_X.corr(),2)
cor_pca


# # Model Building

# In[ ]:


#splitting the dataset into training and test data; test size = 30%
X_train, X_test, y_train, y_test = train_test_split(pca_X,y, test_size=0.3)


# In[ ]:


#building a SVM model on the training data
svc_model = svm.SVC()
svc_model.fit(X_train,y_train)


# In[ ]:


#predicting the class for the test data 
y_pred = svc_model.predict(X_test)


# In[ ]:


#to get the F1 score for each class
print(classification_report(y_test, y_pred))


# # Sample predictions

# In[ ]:


#reading the sample data
sample = pd.read_csv('../input/bda-2019-ml-test/Test_Mask_Dataset.csv')
sample = sample.drop('timeindex',1)
sample.shape


# In[ ]:


#computing the principal components for the sample data
sample=pca.transform(sample)


# In[ ]:


#predicting the class for the records in the sample data
y = svc_model.predict(sample)
y.shape


# In[ ]:


#reading the Sample Submission csv file
output = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")


# In[ ]:


#updating the values in the 'flag column'
output['flag']=y


# In[ ]:


#writing the new csv file
output.to_csv('Sample Submission.csv', index=False)

