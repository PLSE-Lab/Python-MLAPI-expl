#!/usr/bin/env python
# coding: utf-8

# # Suicide Analysis in Sweden
# I am new at machine learning and python. So I try to use what I have learned in order to reinforce. 
# I used WHO Suicide Statistics with REGRESSION algorithm in this part. 
# I plan to revise this document as I learned new things about REGRESSION. If you have any better ideas about using REGRESSION algoritm please let me know. 
# Thanks in advance :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/who_suicide_statistics.csv")
data = data[data['country'] == 'Sweden']  # Sweden data are selected


# In[ ]:



#data.info()
# which fields have null values
missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
data = data.dropna(axis = 0)  # the lines has null values are deleted
data.head()


# In[ ]:


x = np.array(data.loc[:,'year']).reshape(-1,1)
y = np.array(data.loc[:,'suicides_no']).reshape(-1,1)
#Scatter Plot
plt.figure(figsize = [10,10])
plt.scatter(x=x,y=y,)
plt.xlabel('Year')
plt.ylabel('Suicides number')
plt.show()


# In[ ]:


# Lineer Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)  # Prediction Space
#print(predict_space)
lis = ['female', 'male']
lis2 = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
for i in lis:
    for k in lis2:
        data_1 = data[data['sex'] == i]
        data_sex = data_1[data_1['age'] == k ]
        x_sex = np.array(data_sex.loc[:,'year']).reshape(-1,1)
        y_sex = np.array(data_sex.loc[:,'suicides_no']).reshape(-1,1)
        reg.fit(x_sex,y_sex)                                               # Fit
        predicted = reg.predict(predict_space)                     # Prediction
        print( i, k, 'R^2 Score: ', reg.score(x_sex,y_sex))                       # R^2 calculation
        # print(i)
        #plt.figure(figsize = [9,6])
        #print(i,k)
        plt.plot(predict_space, predicted, color = 'black', linewidth = 2)
        plt.scatter(x_sex,y_sex)
        plt.title('Scatter Plot')
        plt.xlabel('Year')
        plt.ylabel('Suicides number')
        plt.show()


# # Conclusion
# Suicide numbers in Sweden are;
# * obviously decreasing in 35-54 years old male people with 84% R^2, 
# * increasing in 5-14 years old female people with 36% R^2 (R^2 is not strong but attention must be paid) 

# ### Other Algoritm Studies:
# **Practice #1: "Gender" prediction with KNN** https://www.kaggle.com/cengizeralp/practice-1-gender-prediction-with-knn
# 

# Thanks to DATAI team for their valuable training notes. I used their documentation as a refference in this study. https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
