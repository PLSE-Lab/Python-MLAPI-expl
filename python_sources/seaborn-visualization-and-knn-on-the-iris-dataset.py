#!/usr/bin/env python
# coding: utf-8

# # Exploration and training of the Iris Dataset

# ### Uses Seaborn and pandas to explore the Iris dataset and find patterns between data of 150 Iris flowers of 3 different subspecies

# Importing files and ignoring seaborn warnings, setting a simple seaborn style

# In[ ]:


import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white',color_codes=True)


# Loading the dataset into df and printing the first 5 rows

# In[ ]:


df=pd.read_csv("../input/Iris.csv")
df.head()


# Each species has 50 samples

# In[ ]:


df['Species'].value_counts()


# Simple scatter plot to see connections between sepal length/width and petal length/width

# In[ ]:


df.plot(kind='scatter',x="SepalLengthCm", y="SepalWidthCm")
df.plot(kind='scatter',x="PetalLengthCm", y="PetalWidthCm")


# Data distribution in the above graph seems to be split into groups,
# so we use a FacetGrid and set the 'hue' to the 'Species' column
# This gives a different color to the points in the scatter plot based on their Species value

# In[ ]:


sns.FacetGrid(df, size=5,hue="Species").map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
sns.FacetGrid(df, size=5,hue="Species").map(plt.scatter,"PetalLengthCm","PetalWidthCm").add_legend()


# So from this plot we can see that Petal length and width play a very important differentiating factor
# (iris setosa has small petal length and width, virginica has the largest)

# Making a boxplot to see the distribution of petal length for each species

# In[ ]:


sns.boxplot(x="Species", y="PetalLengthCm", data=df)


# Boxplot of Petal Width species-wise

# In[ ]:


sns.boxplot(x="Species", y="PetalWidthCm", data=df)


# A stripplot with jitter so we can see the distribution of values more clearly.
# Jitter=True makes it so the points don't fall in a straight line (in this plot X coord.s don't matter)

# In[ ]:


sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")


# Violin plots are the combination of scatter plots and box plots.
# We can see the density of distribution of values for the features

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# KDE: Kernel Density Estimate.
# This shows the distribution density more clearly.
# We use a FacetGrid with hue = 'Species'.
# .add_legend() adds the legend on the top rights.

# In[ ]:


# Distribution density plot KDE (kernel density estimate)
sns.FacetGrid(df, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs)with hue = "Species"

# In[ ]:


sns.pairplot(df.drop("Id", axis=1), hue="Species", size=4)


# # Training models

# ### We will use Logistic Regression and KNN
# #### Train Test Split splits the data into 70:30 ratio. We will train the model on 70% of the data

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics


# Dropping the ID axis because it's not of use in training, and splitting the dataset

# In[ ]:


df.drop('Id',axis=1,inplace=True)
train, test = train_test_split(df, test_size = 0.3)
print(train.shape)
print(test.shape)


# Creating the train and test datasets. X will be the input, y the output

# In[ ]:


train_X=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y=test.Species


# In[ ]:


train_X.head()


# Fitting the data on Logistic Regression:

# In[ ]:


model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y)*100 ,'percent')


# Fitting the data on K Neighbors Classifier with neighbors=3

# In[ ]:


model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y)*100, 'percent')

