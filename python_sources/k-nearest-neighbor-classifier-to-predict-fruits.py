#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting and visualozing data


# In[ ]:


#our dataset
fruits=pd.read_table('../input/fruit_data_with_colors.txt')


# We have loaded our dataset, now we will check it's first five rows to check how our data looks, which features our data have.

# In[ ]:


#checking first five rows of our dataset
fruits.head()


# In[ ]:


# create a mapping from fruit label value to fruit name to make results easier to interpret
predct = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
predct


# Dataset have seven columns containing the information about fruits. Here only two fruits i.e apple and mandarin are seen. Every fruit is described with four features i.e 1) mass of fruit 2) width of fruit 3) what is height and 4) what is color score of fruit. Now we have to check how many fruits are present in our data. 

# In[ ]:


#checking how many unique fruit names are present in the dataset
fruits['fruit_name'].value_counts()


# We have seen that the dataset contains four unique fruits. apple with 19 entries, orange with 19 entries, lemon with 16 entries and mandarin with 5 entries. 

# Now we will store all unique data on four different dataframes.

# In[ ]:


apple_data=fruits[fruits['fruit_name']=='apple']
orange_data=fruits[fruits['fruit_name']=='orange']
lemon_data=fruits[fruits['fruit_name']=='lemon']
mandarin_data=fruits[fruits['fruit_name']=='mandarin']


# In[ ]:


apple_data.head()


# In[ ]:


mandarin_data.head()


# In[ ]:


orange_data.head()


# In[ ]:


lemon_data.head()


# By looking above data, it is shown that for every fruit there is a fruit_label. For apple it is 1, for mandarin it is 2, for orange it is 3 and for lemon it is 4. Now we will visualize this data on plots for further exploration.

# In[ ]:


plt.scatter(fruits['width'],fruits['height'])


# In[ ]:


plt.scatter(fruits['mass'],fruits['color_score'])


# Now we will use K-Nearest Neighbors classifier to predict a new record on the basis of this data. For this we will aplit this dataset into test and train sets. First we will import sklearn library for our model.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X=fruits[['mass','width','height']]
Y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)


# In[ ]:


X_train.describe()


# In[ ]:


X_test.describe()


# Now we will create a KNN classifier for making predictions.

# In[ ]:


knn=KNeighborsClassifier()


# In[ ]:


knn.fit(X_train,y_train)


# We can check the accuracy of our classifier

# In[ ]:


knn.score(X_test,y_test)


# Now we can make predictions with new data as following:

# In[ ]:


#parameters of following function are mass,width and height
#example1
prediction1=knn.predict([['100','6.3','8']])
predct[prediction1[0]]


# In[ ]:


#example2
prediction2=knn.predict([['300','7','10']])
predct[prediction2[0]]


# Yes, our model is running successfully and making accurate predictions.
# Enjoy....!

# In[ ]:




