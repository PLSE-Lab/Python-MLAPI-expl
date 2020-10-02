#!/usr/bin/env python
# coding: utf-8

# **KNN Using for detecting Lower Back Pain Symptoms. **
# 
# In this kernel i will be applying KNN model along with some Sklearn libraries for tuning and accuracy of the model. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  #scale the data 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  # Encode the target variables
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Tuning the model
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Read the data from csv file
data_set= pd.read_csv('../input/Dataset_spine.csv')
print(data_set.head())
#cannot see all the columns?? set the display(Pandas) to show all columns
pd.set_option('display.expand_frame_repr', False)
print(data_set.head()) #ok all you see is numbers except Class_att and Unnamed:13


# #Examine the columns and observe the datatype and see if any missing values
# 
# 

# In[ ]:


data_set.info()


# Lets first encode the Target Variable to numeric values by using Sklearn Label encoder.
# * Either you can use for loop to set 0 or 1 in target column but i used  label encoder because what if target variable has multiple ordinal categorical features like ('abnormal', 'normal', 'x','y','z') label encoder will be easy

# In[ ]:


print('Before:{}'.format(data_set['Class_att'].unique()))
lbl= LabelEncoder()
data_set['Class_att']=lbl.fit_transform(data_set['Class_att'])
print('After:{}'.format(data_set['Class_att'].unique()))


# Fine !!
# 
# Observe the info() table there are 14 columns and 310 rows out of which 12 are float and 2 are object datatype.. 1 column Unname:13 has just 14 entries.
# 
# *Lets first figure out what Unnamed:13 is ? and see whether its useful for calculating target column Class_att
# Sometimes even null values have some meaning - lets find it in below step.
# first fill null values with some 'Unknown' and plot
# 

# In[ ]:


data_set['Unnamed: 13']=data_set['Unnamed: 13'].fillna('Unknown')


# Now that we have filled missing values, lets compare with Target data

# In[ ]:


print(data_set.groupby('Class_att')['Unnamed: 13'].count())   


# Now that from above we have found that the columns Unnamed:13 has 210 features contributing to Abnormal(0) , 100 contributing to normal(1).
# 
# Lets plot using Seaborn just these two columns

# In[ ]:


sns.stripplot(x='Class_att', y='Unnamed: 13', data=data_set)


# Ok, from the above figure missing('Unknown') values has both target values 0  and 1, and 14 other values all contribute to Abnormal.. i dont think this will be useful to determine abnormal or normal because missing('unknown') values have both. 
# 
# Lets delete this column from our dataset

# In[ ]:


data_set= data_set.drop('Unnamed: 13', axis=1)


# Ok , great. Now that we have all numeric 13 columns. lets observe the contribution of feature columns (12 columns ) with target column (1 columns - Class_att) by using Seaborn Pair plot

# In[ ]:


sns.pairplot(data_set, y_vars='Class_att', x_vars=data_set.columns.drop('Class_att'), hue='Class_att')


# Ok to visualize it is little small..but if you zoom it this provides a great explanation each columns contribution with Target variable (Y-axis).
# 
# Also observe carefully that col2 and col6 has negative values. But the min max difference is good for Col2 which is fine..but see for col6 min max difference is huge compare to other columns. So lets scale the data using Sklearn Min max scaler

# In[ ]:


scaler=MinMaxScaler()

data_set['Col6']=scaler.fit_transform(data_set['Col6'].values.reshape(-1,1))


# Now time to train and test the Data and to choose model we did scale and encode the data, the benefit would be applied for non tree based models. So i choosed KNearestNeighbors

# In[ ]:



X= data_set.drop('Class_att', axis=1)
y= data_set['Class_att']


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred= knn.predict(X_test)
print('knn score:{}'.format(knn.score(X_test,y_test)))


# Tuning your model by using GridSearchCV and  Randomized Search CV

# In[ ]:


param_grid={'n_neighbors':range(1,10)}
GS=GridSearchCV(knn, param_grid, cv=5)
GS.fit(X_train, y_train)
print(GS.best_params_)
print(GS.best_score_)


param_grid={'n_neighbors':range(1,10)}
knn=KNeighborsClassifier()
knn_cv=RandomizedSearchCV(estimator=knn, param_distributions=param_grid, cv=4, n_iter=9)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)
print(knn_cv.best_score_)


# Ok So basically n_neighbors=5 (which was default in our model) yields better results.
# 
# Now lets see what classification report say.

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


Finally lets plot ROC curve.


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr,_ = roc_curve(y_test, y_pred)

plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# **Thank you all for watching!!. I will be happy if you find any mistakes or feed back on this. Then only i can learn!**
