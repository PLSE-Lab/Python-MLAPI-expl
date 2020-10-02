#!/usr/bin/env python
# coding: utf-8

# # DATASET OVERVIEW

# It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# 
# 1. Id
# 2. SepalLengthCm
# 3. SepalWidthCm
# 4. PetalLengthCm
# 5. PetalWidthCm
# 6. Species (Iris-setosa,Iris-versicolor,Iris-virginica)

# # Visualizing parts of Iris flower
# **Just for knowledge purpose!

# ![irispart.jpg](attachment:irispart.jpg)

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load Dataset

# In[ ]:


data=pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()


# # EXPLORING DATA AND EXPLORATORY DATA ANALYSIS
# 

# In[ ]:


#dimensions of data
print("No of rows in dataset is: ",data.shape[0])
print("No of columns in dataset is: ",data.shape[1])


# In[ ]:


data.info()


# **Our data has no missing values hence it is easier to work with.

# In[ ]:


data.Species.unique()
#There are three unique classes of Iris present in the dataset.


# In[ ]:


plt.figure(figsize=(8,8))
sns.pairplot(data=data,hue='Species')


# We can see that one class i.e Iris Setosa is linearly separable from the other two, where as the other two are not.They are quite clustered together.

# In[ ]:


sns.heatmap(data.corr(),annot=True)


# In[ ]:


data['Species'].value_counts().plot.pie(autopct='%0.2f%%',colors=['blue','green','red'],figsize=(5,5))


# Our dataset has an even distribution of all the three classes.

# In[ ]:


sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=data)


# We cant say that Sepal width increases with sepal length all the time. This increase is observed only for species IRIS-setosa wheras for other it is pretty distributed.

# In[ ]:


sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=data)


# Petal width increases with petal length for all the three classes and all three classes seem to lie in a range of petalWidth and Length. PetalWidthCm and PetalLengthCm are positively correlated that can be seen from the heat-map also.

# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
sns.boxplot(x='Species',y='PetalLengthCm',data=data)
plt.title('Variation of PetalLength with Species \n PLOT 1')
plt.subplot(122)
sns.boxplot(x='Species',y='PetalWidthCm',data=data)
plt.title('Variation of PetalWidth with Species \n PLOT 2')


# The plots show that Iris virginica has the largest petal length and  width. Also, as the petal length and width increases species changes from Iris-setosa to Iris-virginica.

# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
sns.boxplot(x='Species',y='SepalLengthCm',data=data)
plt.title('Variation of SepalLength with Species \n PLOT 1')
plt.subplot(122)
sns.boxplot(x='Species',y='SepalWidthCm',data=data)
plt.title('Variation of SepalWidth with Species \n PLOT 2')


# PLOT1: Shows that as sepalLength increases species changes from Iris-setosa to Iris-virginica.
