#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read data
cancer_data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

# print first lines
print('First lines :')
print(cancer_data.head())

# print last lines
print('Last lines :')
print(cancer_data.tail())


# In[ ]:


# cancer data columns
cancer_cols = list(cancer_data.columns)[0:-1]
print(cancer_cols)


# In[ ]:


# delete last column
cancer_data = cancer_data[cancer_cols]
# describe data
print(cancer_data.describe())


# Let's see if is there somme missing data

# In[ ]:


for c in cancer_cols:
    temp = cancer_data[c].isnull()
    temp_n = sum(temp)
    if temp_n > 0:
        print(temp_n, ' null values in column ', c)


# When you look at the columns, one may suggest that this geomtric mesurements are correlated : 
# - radius , 
# - area, 
# - perimeter
# 
# This can be confirmed with simple 2d scatters :

# In[ ]:


# plot radius mean againt area mean
plt.subplot(311)
plt.scatter(cancer_data.radius_mean, cancer_data.area_mean, c=cancer_data.diagnosis)

plt.subplot(312)
plt.scatter(cancer_data.radius_mean, cancer_data.perimeter_mean, c=cancer_data.diagnosis)

plt.subplot(313)
plt.scatter(cancer_data.area_mean, cancer_data.perimeter_mean,c=cancer_data.diagnosis)

plt.show()


# Let's see that another way and for all variables (except target)
# 

# In[ ]:


corr = cancer_data[cancer_cols[2:]].corr()
corr.style.background_gradient(cmap='coolwarm')


# As there is a lot of variables that are very correlated, ex : radius an area, let's do a PCA

# In[ ]:


# For better evaluating our coming models, it's better to split data from now :
features_cols = cancer_cols[2:]
X = cancer_data[features_cols]
y = cancer_data.diagnosis
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# Let's create a helper function for PCA :

# In[ ]:


def fit_PCA(train_X, val_X, n_comp):
    # scale
    train_X = StandardScaler().fit_transform(train_X)
    val_X = StandardScaler().fit_transform(val_X)
    
    pca = PCA(n_components=n_comp)
    pca.fit(train_X)
    train_X_pca = pca.transform(train_X)
    val_X_pca = pca.transform(val_X)
    
    df_train_X_pca = pd.DataFrame(data = train_X_pca
             , columns = ['princ_comp_' + str(i+1) for i in range(n_comp)])

    df_val_X_pca = pd.DataFrame(data = val_X_pca
             , columns = ['princ_comp_' + str(i+1) for i in range(n_comp)])
    
    # add diagnosis column back
    df_train_X_pca['diagnosis'] = list(train_y)
    df_val_X_pca['diagnosis'] = list(val_y)
    
    return train_X_pca, val_X_pca, df_train_X_pca, df_val_X_pca
    


# First we run PCA with 2 principal components only

# In[ ]:


train_X_pca, val_X_pca, df_train_X_pca, df_val_X_pca = fit_PCA(train_X, val_X, 2)

# plot princ_comp_1 against princ_comp_2 for train data
plt.scatter(df_train_X_pca.princ_comp_1, df_train_X_pca.princ_comp_2, c=df_train_X_pca.diagnosis)




# Let's do a logistic regression.
# First we'll do it on original data before pca :

# In[ ]:


logisticRegr = LogisticRegression(solver = 'lbfgs')

# fit the model
logisticRegr.fit(train_X, train_y)

# predict on train data
predicted_train_y = logisticRegr.predict(train_X)

# mesure model in train data and validation data
print("precision in train data : " , sum(predicted_train_y==train_y)/len(train_y))
# predict on validation data
predicted_train_y = logisticRegr.predict(val_X)
print("precision in validation data : " , sum(predicted_train_y==val_y)/len(val_y))


# Let's see if pca can do better.
# For this we'll do a for loop to find the best number of components.

# In[ ]:


precisions_train = []
precisions_val = []
for i in range(2,30):
    train_X_pca, val_X_pca, df_train_X_pca, df_val_X_pca = fit_PCA(train_X, val_X, i)
    
    # fit the model
    logisticRegr.fit(train_X_pca, train_y)

    # predict on train data
    predicted_train_y = logisticRegr.predict(train_X_pca)

    # mesure model in train data and validation data
    print("precision in train data : " , sum(predicted_train_y==train_y)/len(train_y), ' n_comp =', i)
    precisions_train.append(sum(predicted_train_y==train_y)/len(train_y))
    # predict on validation data
    predicted_val_y = logisticRegr.predict(val_X_pca)
    print("precision in validation data : " , sum(predicted_val_y==val_y)/len(val_y), ' n_comp =', i)
    precisions_val.append(sum(predicted_val_y==val_y)/len(val_y))
plt.plot(precisions_val)
plt.plot(precisions_train)
plt.show()


# In[ ]:


print(precisions_val)


# Component 4 is the best with precision equals 0.972.
