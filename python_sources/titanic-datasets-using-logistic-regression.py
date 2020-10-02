#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))


# **Data Collection**

# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[ ]:


df = pd.read_csv('../input/train_data.csv')


# In[ ]:


df


# In[ ]:


df.dropna(inplace=True)


# **Explorer Dataset**

# In[ ]:


# shape
print(df.shape)


# In[ ]:


#columns*rows
df.size


# How many NA elements in every column

# In[ ]:


df.isnull().sum()


# **For getting some information about the dataset you can use info() command**

# In[ ]:


print(df.info())


# **To check the first 5 rows of the data set, we can use head(5).**

# In[ ]:


df.head(5)


# **To check out last 5 row of the data set, we use tail() function**

# In[ ]:


df.tail() 


# **To pop up 5 random rows from the data set, we can use sample(5) function**

# In[ ]:


df.sample(5) 


# **To give a statistical summary about the dataset, we can use **describe()**

# In[ ]:


df.describe()


# **To check out how many null info are on the dataset, we can use **isnull().sum().**

# In[ ]:


df.isnull().sum()


# **To print dataset columns, we can use columns atribute**

# In[ ]:


df.columns


# **Visualization**

# **Histogram**

# We can also create a histogram of each input variable to get an idea of the distribution.

# In[ ]:


# histograms
df.hist(figsize=(16,47))
plt.figure()


# **Pairplot**

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(df)


# **Exploratory Data Analysis**

# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# **Missing Data**
# 
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass_1',data=df,palette='rainbow')


# In[ ]:


sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


df['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


sns.countplot(x='Title_1',data=df)


# **Building a Logistic Regression model**
# 
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training)
# 
# **Train Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 
                                                    df['Survived'], test_size=0.22, 
                                                    random_state=51)


# **Training and Predicting**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# Let's move on to evaluate our model!

# **Evaluation**
# 
# We can check precision,recall,f1-score using classification report!

# In[ ]:


from sklearn.metrics import classification_report,classification_report


# In[ ]:


print(classification_report(y_test,predictions))

