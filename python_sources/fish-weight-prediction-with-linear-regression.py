#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression Project
# #### In this project we will be working with a Fish market data set, indicating the weight of fishs. We will try to create a model that will predict weight in future based of the features of the past weight.
# #### This data set contains the following features:
# - Species :species name of fish
# - Weight : weight of fish in Gram g
# - Length1 : vertical length in cm
# - Length2 : diagonal length in cm
# - Length3 : cross length in cm
# - Height : height in cm
# - Width : diagonal width in cm

# # Gather Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


# get the data
df = pd.read_csv(r'/kaggle/input/fish-market/Fish.csv')


# # Assess Data

# In[ ]:


# get the head of the data
df.head()


# In[ ]:


#shape
df.shape


# In[ ]:


#info
df.info()


# In[ ]:


# columns of the data
for i,col in enumerate(df.columns):
    print(i,col)


# In[ ]:


# check for nulls
df.isnull().sum()


# In[ ]:


# check for duplicated
df.duplicated().sum()


# In[ ]:


# statistical summary
df.describe().T


# In[ ]:


# Get a sample of data
df.sample(5)


# In[ ]:


# check species names
df.Species.unique()


# # Quality
# - The Species columns type need to be category not object

# # Clean Data
# ***The data is clean and doesn't require any fix except convert Species columns type to category***

# ***Define***
# - The Species columns type need to be category not object
# 
# **Code**

# In[ ]:


df.Species = df.Species.astype('category')


# **Test**

# In[ ]:


assert df.Species.dtype == 'category'


# # Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(15,10))
label=['Perch','Bream','Roach','Pike','Smelt','Parkki','Whitefish']
plt.pie(df.Species.value_counts(),explode=[0.1]*len(label),labels=label,autopct='%.1f%%',shadow=True)
plt.axis('equal')
plt.title('Fish Species')
plt.show()


# In[ ]:


df.Species.value_counts()


# ***The plot show that Perch ,Bream species are the most common and Parkki ,Whitefish are less common***

# In[ ]:


# create a function to plot a histogram
def plot_hist(col,style):
    g = sns.FacetGrid(df,hue='Species',palette=style,height=6,aspect=2)
    g.map(plt.hist,col,alpha=0.6,bins=20)
    plt.legend()
    plt.show()    


# In[ ]:


plot_hist('Weight','rainbow')


# In[ ]:


plot_hist('Length1','Set1')


# In[ ]:


plot_hist('Length2','cool')


# In[ ]:


plot_hist('Length3','dark')


# In[ ]:


plot_hist('Height','hot')


# In[ ]:


plot_hist('Width','Set2')


# In[ ]:


# create a function to plot a scatterplot
def plot_scatter(col,style):
    plt.figure(figsize=(12,10))
    sns.scatterplot(data = df,x=col,y='Weight',hue='Species',palette=style)
    plt.xlabel(col)
    plt.ylabel('Weight')
    plt.legend()
    plt.show()


# In[ ]:


plot_scatter('Length1','Set1')


# In[ ]:


plot_scatter('Length2','cool')


# In[ ]:


plot_scatter('Length3','dark')


# In[ ]:


plot_scatter('Height','hot')


# In[ ]:


plot_scatter('Width','Set2')


# ### **The plots show that all feature increasing with the increasing of weight which mean that all fearure are in linear relationship with weight** 

# In[ ]:


# # create a function to plot a boxplot
def plot_box(col,style):
    plt.figure(figsize=(12,10))
    sns.boxplot(data = df,x='Species',y=col,palette=style)
    plt.ylabel(col)
    plt.xlabel('Species')
    plt.show()


# In[ ]:


plot_box('Length1','Set1')


# In[ ]:


plot_box('Length2','cool')


# In[ ]:


plot_box('Length3','dark')


# In[ ]:


plot_box('Height','hot')


# In[ ]:


plot_box('Width','Set2')


# In[ ]:


plot_box('Weight','rainbow')


# # Prepare The Data For The Model

# In[ ]:


# convert species column to nnmbers each one represent a specific species
df.Species =df.Species.cat.codes


# In[ ]:


# indicating the outliers 
df_outlier = df['Weight']
df_outlier_Q1 = df_outlier.quantile(0.25)
df_outlier_Q3 = df_outlier.quantile(0.75)
df_outlier_IQR = df_outlier_Q3 - df_outlier_Q1
df_outlier_lower = df_outlier_Q1 - (1.5 * df_outlier_IQR)
df_outlier_upper = df_outlier_Q3 + (1.5 * df_outlier_IQR)
(df_outlier_lower,df_outlier_upper)


# In[ ]:


df.query('Weight >= 1445.0 or Weight <= -675.0')


# In[ ]:


# drop outliers rows
df.drop([142,143,144],axis=0,inplace=True)
df.reset_index(inplace=True,drop=True)


# In[ ]:


# plot a heatmap
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(),annot=True,cmap='Blues')


# In[ ]:


# drop species columns because it had negative corrleation with weight
df.drop('Species',axis=1,inplace=True)


# In[ ]:


# drop Length2 , Length3 columns because it had 1 corrleation with Length1
df.drop(['Length2','Length3'],axis=1,inplace=True)


# In[ ]:


# get head of data
df.head()


# # Split the data into training set and testing set using train_test_split

# In[ ]:


# import train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Weight',axis=1)
y = df['Weight']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm=LinearRegression()


# ***Train and fit a Linear regression model on the training set***

# In[ ]:


# fit the model to the data
lm.fit(X_train,y_train)


# In[ ]:


# display intercept
lm.intercept_


# In[ ]:


# display coefficient
lm.coef_


# # Predictions and Evaluations

# In[ ]:


predict = lm.predict(X_test)


# In[ ]:


# create a plot for the relation between y_test and the model predicitions
plt.figure(figsize=(12,6))
plt.scatter(y_test,predict,color='orange')
plt.ylabel('predict')
plt.xlabel('y_test')
plt.show()


# In[ ]:


# import mean_squared_error,r2_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


# In[ ]:


# compute root mean_squared_error
np.sqrt(mean_squared_error(y_test,predict))


# In[ ]:


# compute mean_absolute_error
mean_absolute_error(y_test,predict)


# In[ ]:


# compute r2_score
r2_score(y_test,predict)

