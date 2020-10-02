#!/usr/bin/env python
# coding: utf-8

# > # Importing modules and loading data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()


# We can see the dataset is made of 4 independent features, all of which are continues, and they are:
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width
# and 1 dependent categorical variable **Species**.
# 
# Also that the Id column is not set correctly.

# In[ ]:


#Set id column to be index
df.set_index('Id' , inplace = True)
df.head()


# # Data Descritpion

# In[ ]:


print(df.info()) #information on data types,memory usage , null or non-null values


# In[ ]:


print(df.describe()) #descirptive statstics


# **Notice that:**
# 
# * In all of the data (5 columns and 150 entries) there aren't any null values.
# * There doesn't seem to be any outliers. ( from descriptive statstics std).
# 
# The data do not require any cleaning.
# 
# Next let's explore our target variable.

# In[ ]:


df['Species'].value_counts()


# Here we see that the **Species** column has 3 unique values. Also the dataset has equal number of flowers for every species

# # Data Visualization

# In[ ]:


sns.pairplot(df , hue = 'Species')


# From the pairplot above we notice that Petal features has better speration than that of Sepal Features.
# 
# **Note** that we can distinguish the Setosa points using the petals feature. There is a kernal that has used this specific feature to build a simple model specific to distinguish this species.
# 
# Check it here
#  https://www.kaggle.com/viswanathanc/beginner-visualization-using-matplotlib/comments#Notebook-Description

# In[ ]:


#determin bin size based on the values of the dataset
bins_sepal_len = np.arange(df['SepalLengthCm'].min()-0.5 , df['SepalLengthCm'].max()+0.5 , 0.5)
bins_sepal_wid = np.arange(df['SepalWidthCm'].min()-0.5 , df['SepalWidthCm'].max()+0.5 , 0.5)
bins_petal_len = np.arange(df['PetalLengthCm'].min()-0.5 , df['PetalLengthCm'].max()+0.5 , 0.5)
bins_petal_wid = np.arange(df['PetalWidthCm'].min()-0.5 , df['PetalWidthCm'].max()+0.5 , 0.5)

plt.figure(figsize = (15,15))

plt.subplot(221)
plt.hist(data = df, x = 'SepalLengthCm' , bins = bins_sepal_len)
plt.title('Sepal Length')

plt.subplot(222)
plt.hist(data = df, x = 'SepalWidthCm' , bins = bins_sepal_wid)
plt.title('Sepal Width')

plt.subplot(223)
plt.hist(data = df, x = 'PetalLengthCm' , bins = bins_petal_len)
plt.title('Petal Length')

plt.subplot(224)
plt.hist(data = df, x = 'PetalWidthCm' , bins = bins_petal_wid)
plt.title('Petal Width')


# Not much info here to be honest. Let's try again with indivdual species distribution.
# 
# **Note** I am doing the next code in seaborn just to practice. You can get the exact same results (and exact code length) with matplot.

# In[ ]:


#Separ data according to there species
df1 = df[df.Species=='Iris-setosa']
df2 = df[df.Species=='Iris-versicolor']
df3 = df[df.Species=='Iris-virginica']


# In[ ]:




plt.figure(figsize = (15,15))

plt.hist(df1.SepalLengthCm,bins=30)
plt.hist(df2.SepalLengthCm,bins=30)
plt.hist(df3.SepalLengthCm,bins=30)
plt.legend(['Setosa','Versicolor','Virginica'])
plt.title('Sepal Length')

plt.subplot(221)
sns.distplot(df1['SepalLengthCm'] , kde = False , bins = 30)
sns.distplot(df2['SepalLengthCm'] , kde = False , bins = 30)
sns.distplot(df3['SepalLengthCm'] , kde = False , bins = 30)
plt.title('Sepal Length')

plt.subplot(222)
sns.distplot(df1['SepalWidthCm'] , kde = False, bins = 30)
sns.distplot(df2['SepalWidthCm'] , kde = False, bins = 30)
sns.distplot(df3['SepalWidthCm'] , kde = False, bins = 30)
plt.title('Sepal Width')

plt.subplot(223)
sns.distplot(df1['PetalLengthCm'] , kde = False ,bins = 30)
sns.distplot(df2['PetalLengthCm'] , kde = False ,bins = 30)
sns.distplot(df3['PetalLengthCm'] , kde = False ,bins = 30)
plt.title('Petal Length')

plt.subplot(224)
sns.distplot(df1['PetalWidthCm'] , kde = False , bins = 30)
sns.distplot(df2['PetalWidthCm'] , kde = False , bins = 30)
sns.distplot(df3['PetalWidthCm'] , kde = False , bins = 30)
plt.title('Petal Width')


# Another way to find correlation between features is heatmap

# In[ ]:


sns.heatmap(df.corr() , annot=True)


# **Note** the default value for **df.corr()** is **pearson** for pearson correlation which is used to indicate how strongly 2 variables are linearly related. You can choose other correlation methods.
# 
# Check out the documentation:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

# # Calssification

# A simple Dicision Tree Classifier.

# In[ ]:


#drop the results column
results = df['Species']
df = df.drop(columns = ['Species'])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


X_train , X_test , y_train , y_test = train_test_split(df, results, test_size = 0.25, random_state = 0)


# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = accuracy_score(y_test , y_predict)

print(score)


# **One Final Note**
# I have considered re-scaling the features, however it did not seem necessry since all features are close in range and they use the same metric (cm). Also a simple classifier scored over **0.97 accuracy** without any changes in its feature.
