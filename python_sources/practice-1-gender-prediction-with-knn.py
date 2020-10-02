#!/usr/bin/env python
# coding: utf-8

# # "Gender" prediction using "Purchase" value and "Occupation" type 
# I am new at machine learning and python. So I try to use what I have learned in order to reinforce. 
# I used Black Friday data with KNN (K-NEAREST NEIGHBORS) algorithm in this part. 
# I plan to revise this document as I learned new things about KNN. If you have any better ideas about using KNN algoritm please let me know. 
# Thanks in advance :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_orj = pd.read_csv("../input/BlackFriday.csv")
data = data_orj.loc [1:10000,'Gender':'Purchase']  # data is sliced in order to study fast


# In[ ]:


data.head()


# In[ ]:


data.info()


# "Product_Category_2" and "Product_Category_3" has null values

# In[ ]:


data.describe()


# In[ ]:


#color_list = ['red' if i=='M' else 'green' for i in data.loc[:,'Gender']]
#d.plotting.scatter_matrix(data.loc[:,data.columns != 'Gender'],
 #                         c = color_list,
  #                        figsize = [15,15],
   #                       diagonal = 'hist',
    #                      alpha = 0.5, 
     #                     s = 100,
      #                    marker = '*')
#plt.show()


# All features are categrical except "Purchase" column. So the matrix scatter plot does not show meaningfull relations (correlations). As you can see from the below outcomes, tha data also is not balanced :( Aynyway, I pretend it is balanced :) Lets begin to use KNN algorithm

# In[ ]:


sns.countplot(x='Gender', data=data)
data.loc[:,'Gender'].value_counts()


# What should we remember here are:
# 1. Features must not be categorical (must be numbers)
# 1. You must have 2 features at least (if you have only one feature and one target you can use correlation simply :))

# In[ ]:


data_knn = data_orj[['Occupation','Gender', 'Purchase']]


# In[ ]:


data_knn.head()


# In[ ]:


data_knn.info()


# In[ ]:


#KNN-2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors = 3, algorithm='kd_tree')

x,y = data_knn.loc[:,data_knn.columns != 'Gender'], data_knn.loc[:,'Gender']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction : {}'.format(prediction))
print('With KNN (K=3) accuracy is: ', knn.score(x_test,y_test))


# In[ ]:


#Best K value selection
neig = np.arange(1,30)
train_accuracy = []
test_accuracy = []
for i, k in enumerate (neig):
    knn = KNeighborsClassifier(n_neighbors = k, algorithm='ball_tree', leaf_size=40,)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
    
# Plot
plt.figure(figsize=(13,8))
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value vs. Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print('Best Accuracy is {} with K = {}'.format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# # Conclusion
# 
# As a result, we can say that using KNN algorithm with K=29, **we can estimate the "Gender" of a person from her/his "Purchase" value and "Occupation" type with 75% accuracy.**
# 

# ### Other Algoritm Studies:
# **Practice #2: Suicide Analysis with REGRESSION** https://www.kaggle.com/cengizeralp/practice-2-suicide-analysis-with-regression

# Thanks to DATAI team for their valuable training notes. I used their documentation as a refference in this study. https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
