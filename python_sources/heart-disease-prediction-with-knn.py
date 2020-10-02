#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# * In this study I will make a prediction of a heart disease presence with KNN. 
# * I will compare the result with [LR method](https://www.kaggle.com/albatros1602/heart-disease-prediction-with-lr) which I studied before.
# * You may want to check my prior study for [visualisation](https://www.kaggle.com/albatros1602/visualization-for-heart-disease-prediction) of this data.
# 
# <font color = 'blue'>
# ## Content
# 1. [About the Dataset](#1)
# 1. [Normalization](#2)
# 1. [Splitting the Data](#3)
# 1. [KNN](#4)  
# 1. [Finding K Value](#5)
# 1. [Conclusion](#6)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id = "1"></a><br>
# ## About the Dataset
# 
# This dataset has 14 features, one of which is the target feature. Target feature tells us weather a patient has a hearth disease or not. About the target feature:
# * 1 = Patient has heart disease
# * 0 = Patient doesn't have heart disease
# 
# You can reach the meanings of the other features from [my prior study](https://www.kaggle.com/albatros1602/visualization-for-heart-disease-prediction).
# 
# Lets check the data first...

# In[ ]:


dt = pd.read_csv('../input/heart-disease-uci/heart.csv')
dt.head()


# In[ ]:


dt.info()


# In[ ]:


Y = dt[dt.target == 1]
N = dt[dt.target == 0]


# In[ ]:


# scatter plot
plt.scatter(Y.age,Y.thalach, color = "red", label = "Heart Diease Present")
plt.scatter(N.age,N.thalach, color = "green", label = "Heart Diease NOT Present")
plt.xlabel("age")
plt.ylabel("thalach")
plt.legend()
plt.show()


# * There is not a very distinct seperation between green and red dots.
# * The KNN method will try to find the nearest "k" number of points to a selected point and will make a prediction according to the value of the nearest points.

# <a id = "2"></a><br>
# ## Normalization
# 
# * We need to normalize the data, otherwise one feature may dominate an other feature.

# In[ ]:


y = dt.target.values
x_dt = dt.drop(["target"], axis = 1)
x = (x_dt - np.min(x_dt))/(np.max(x_dt)-np.min(x_dt))


# <a id = "3"></a><br>
# ## Splitting the Data
# * %80 of the data will be used for training the model
# * %20 of the data will be used for testing the model
# * I chose random_state = 42 in order to obtain the same rows as train and test at each splitting.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# <a id = "4"></a><br>
# ## KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # I chose K = 3 just for now.
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("{} nn score: {} ".format(3,knn.score(x_test,y_test)))


# <a id = "5"></a><br>
# ## Finding K Value
# * Above, I chose K = 3. The accuracy of the model is %83.6
# * The accuracy of the model will change according to the K value
# * Since the aim is to reach the highest accuracy, we need to try different K values and find the best value which gives the highes accuracy.

# In[ ]:


score_list = []
for each in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,30),score_list)
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()


# * For K = 18 or K = 24 the model reaches the highest accuracy which is %85.25

# <a id = "6"></a><br>
# ## Conclusion
# 
# For this data:
# * The accuracy of KNN model is %85.25
# * The accuracy of LR model is %83.61
# * The accuracy of the models may change according to the data.
# * You can reach more ML tutorials at [DATAI Team](https://www.kaggle.com/kanncaa1)'s page.
