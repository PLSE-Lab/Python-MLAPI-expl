#!/usr/bin/env python
# coding: utf-8

# **<h1>Machine Learning - The Art of Making Computers Learn</h1>**

# ![](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAz2AAAAJDMzNThmYjA5LTM2ZWYtNDUzZC1iNDQyLTMxNzZkMWYyNGExOQ.jpg)

# <h2>Objective</h2>
# * Understand the possibilities and limitations of Machine Learning.
# * Understand the main ideas behind the most widely used learning algorithms in the industry.
# * How to build predictive models from data and analyze their performance.
# 
# <h1>1.1 Demystifying artificial intelligence & machine learning.</h1>
# <h2>Artificial Intelligence</h2>
# <p>Artificial intelligence (AI) is an area of computer science that emphasizes the creation of intelligent machines that work and react like humans</p>
# <h2>Machine Learning</h2>
# <p>Machine Learning is a subfield within Artificial Intelligence that builds algorithms that allow computers to learn to perform tasks from data instead of being explicitly programmed.</p>
# **Machine + Learn **
# 
# <h3>Traditional Programming</h3>
# <br>
# ![](https://i.imgur.com/31Z2hX7.jpg)
# <h3>Machine Learning</h3>
# <br>
# ![](https://i.imgur.com/BLpEzg2.jpg)
# 

# <h3>AI vs ML vs Deep Learning</h3>
# <br>
# ![](https://media-exp2.licdn.com/mpr/mpr/AAEAAQAAAAAAAA1gAAAAJDA5MzlmNGJlLTg5YWMtNDU5MC1hYWQ5LWQ3YjU1ZDBhY2I4Zg.png)

# <h3>Why Machine Learning is taking off ?</h3>
# * Vast amount of Data
# * Faster computation power of computers
# * Improvement of the learning algorithms.

# <h3>The most common Problems ML can solve</h3>
# * Classification
# * Regression
# * Clustering

# <h2>Applications and Limitations to Machine Learning</h2>
# <h3>Applications</h3>
# 1. Virtual Personal Assistants
# 2. Predictions while Commuting (Driverless cars, traffic prediction etc )
# 3. Videos Surveillance (Crime detection)
# 4. Social Media Services (People recomendation, Face recoginition, Similar Pins etc )
# 5. Email Spam and Malware Filtering
# 6. Online Customer Support
# 7. Product Recommendations
# 8. Banking and financial sector
# 9. Medicine and Healthcare
# 
# <h3>Limitations</h3>
# * Require large amounts of hand-crafted, structured training data
# * No known one-model-fits-all solution exists.
# * Computational and technological barriers can limit real time testing and deployment of ML solutions.
# * ML algorithms does not understand context.
# 
# 

# <h2>There are three types of Machine Learning Algorithms</h2>
# <h3>Supervised Learning</h3>
# <p>The majority of practical machine learning uses supervised learning. Supervised learning is where you have input variables (X) and an output variable (Y ) and you use an algorithm to learn the mapping function from the input to the output. For example: Classification, Regression.</p>
# * Linear Regression
# * Logistic Regression
# * Random Forest
# * SVM
# 
# <h3>Unsupervised Learning</h3>
# <p>Unsupervised learning is where you you only have input data (X) and no corresponding output
# variables. The goal for unsupervised learning is to model the underlying structure or distribution
# in the data in order to learn more about the data. For example: Clustering, Association.</p>
# * K-Means
# * Apriori Algorithm
# 
# <h3>Semi Supervised Learning</h3>
# Problems where you have a large amount of input data (X) and only some of the data is labeled (Y ) are called semi-supervised learning problems. These problems sit in between both supervised and unsupervised learning. A good example is a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

# <h2>1.2 How do machines really learn?</h2>
# 
# ![](https://cdn-images-1.medium.com/max/2000/1*KzmIUYPmxgEHhXX7SlbP4w.jpeg)

# 

# ![](http://oi67.tinypic.com/14dddug.jpg)

# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.

# <h2>Tools used</h2>
# * **The Jupyter Notebook** - The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.
# * **Python** - Python is a powerful high-level, object-oriented programming language created by Guido van Rossum.
# * **Pandas** - Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# * **Matplotlib/ Seaborn** - These are Python visualization libraries.
# * **Scikit** - Scikit-learn is a free software machine learning library for the Python programming language.
# * **NumPy** - NumPy is a Python library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# <h2>1.3 Data really powers everything that we do.</h2>
# **Exploratory Data Analysis(EDA): **
# 1. Analysis of the features.
# 2. Finding any relations or trends considering multiple features.
# 
# **Feature Engineering and Data Cleaning: **
# 1. Adding any few features.
# 2. Removing redundant features.
# 3. Converting features into suitable form for modeling.
# 

# <h3> Import required Libraries</h3>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset into pandas DataFrame (2D data structure ).

# In[ ]:


df = pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)


# In[ ]:


df.head()


# <h3>Analysing The Features</h3>

# In[ ]:


df.columns.values


# <h3>Types Of Features</h3>
# **Categorical Features:**
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.
# * Nominal Variables - No relation between values.( Sex,Embarked)
# * Ordinal Features: Relative ordering or sorting between the values. ( PClass)
# 
# **Continous Feature:**
# A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.
# Example: Age

# In[ ]:


# Data Analysis and Exploration
sns.countplot('Sex',hue='Survived',data=df)
plt.show()


# <h3>Feature Engineering and Data Cleaning </h3>

# In[ ]:


# Age as a Categorical feature
df['Age_band']=0
df.loc[df['Age']<=16,'Age_band']=0
df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1
df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2
df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3
df.loc[df['Age']>64,'Age_band']=4
df.head(2)


# In[ ]:


df['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')#checking the number of passenegers in each band


# In[ ]:


# Cleaning data
df.Age.isnull().sum()
#df.loc[


# In[ ]:


# Converting String Values into Numeric
df['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


pd.crosstab(df.Pclass,df.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


df['Title'] = None
for index,row in enumerate(df['Name']):
    title = row.split(', ')[1].split('. ')[0]
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Mr','Rev', 'Sir']:
        df.loc[index, 'Title'] = 'Mr'
    elif title in [ 'Ms', 'Mme', 'Mrs', 'the Countess','Lady']:
        df.loc[index, 'Title'] = 'Mrs'
    elif title in ['Master']:
        df.loc[index, 'Title'] = 'Master'
    elif title in ['Miss','Mlle']:
        df.loc[index, 'Title'] = 'Ms'
    else:
        df.loc[index, 'Title'] = 'Other'


# In[ ]:


pd.crosstab(df.Title,df.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


df.groupby(['Sex','Survived'])[['Survived']].count().plot(kind='bar')


# In[ ]:


df


# In[ ]:


sns.countplot('Sex', hue='Survived',data=df)


# In[ ]:


df['Sex'].replace(['female','male'],[1,0],inplace=True)


# <h3>Correlation Between The Features</h3>
# * Positive correlation: If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.
# 
# * Negative correlation: If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.

# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# What is overall chance of survival?

# In[ ]:


df['Survived'].mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




