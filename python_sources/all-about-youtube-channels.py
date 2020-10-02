#!/usr/bin/env python
# coding: utf-8

# # **All About YouTube Channels**

# ![](https://i.gifer.com/74H4.gif)

# # Introduction
# From Top 5000 Youtube channels data from Socialblade dataset, I decide to make some simple notebook for Exploratory Data Analysis
# and do some wrangling data for put it on my prediction model in future 

# Import the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Let us read file

# In[ ]:


df=pd.read_csv('../input/data.csv')


# ## Data Exploration

# In[ ]:


df.info()


# In[ ]:


df.head(10)


# Now let us change object to int so that the numeric data can be used

# In[ ]:


df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')


# Checking for any Null Values

# In[ ]:


df.isnull().sum()


# Removing the null values from our dataframe

# In[ ]:


df=df.dropna()


# ### Grading System On Social Blade
# Social Blade measures a lot of metrics from the channel, such as subscribers, average views count and a lot of data provided by YouTube's API and from there they rank the channel.
# #### Grades

# In[ ]:


output = df.drop_duplicates()
output.groupby('Grade').size()


# ### Correlation heatmap 

# In[ ]:


sns.heatmap(df.corr(),annot=True)
plt.plot()


# Inference: There is a high correlation between Video Views and Subscribers

# Doughnot Plot

# In[ ]:


labels = ['A++', 'A+', 'A', 'A-','B++']
sizes = [10,40,897,941,2722]
#colors
colors = ['#ffdaB9','#66b3ff','#99ff99','#ffcc99','#ff9999']
#explsion
explode = (0.05,0.05,0.05,0.05,0.05)
plt.figure(figsize=(8,8)) 
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)
p=plt.gcf()
plt.axis('equal')
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:


plt.subplots()
sns.regplot(x=df['Subscribers'], y=df["Video views"], fit_reg=True,scatter_kws={"color":"orange"})
plt.show()


# In[ ]:


sns.regplot(x=df["Video Uploads"], y=df["Video views"], fit_reg=False,scatter_kws={"color":"red"})
plt.show()


# In[ ]:


sns.regplot(x=df["Video Uploads"], y=df["Subscribers"], fit_reg=False,scatter_kws={"color":"blue"})
plt.show()


# Plot by seperating Grades

# In[ ]:


sns.lmplot(x='Subscribers', y='Video views', data=df, fit_reg=False, hue='Grade')
plt.show()


# Ranking by Subscriber Count

# In[ ]:


df.sort_values(by = ['Subscribers'], ascending = False).head(15).plot.barh(x = 'Channel name', y = 'Subscribers')
plt.show()


# Ranking by Viewers count

# In[ ]:


df.sort_values(by = ['Video views'], ascending = False).head(15).plot.barh(x = 'Channel name', y = 'Video views')
plt.show()


# In[ ]:


X = df[['Video Uploads', 'Video views']]
Y = df[['Subscribers']]


# ## Linear Regression
# The term "linearity" in algebra refers to a linear relationship between two or more variables.

# Train-test split

# In[ ]:


X_train, X_test, y_train, y_test =train_test_split(X,Y, test_size = 0.2)


#  Fit a linear regression model

# In[ ]:


lr=LinearRegression()
lr.fit(X_train.dropna(),y_train.dropna())


# In[ ]:


pred_train=lr.predict(X_train)
pred_test=lr.predict(X_test)


# ## Residual Plots
# * Residual plots are a good way to visualize the errors in your data. 
# * If you have done a good job then your data should be randomly scattered around line zero.
# * If you see structure in your data, that means your model is not capturing some thing. Maye be there is a interaction between 2 variables that you are not considering, or may be you are measuring time dependent data. 
# * If you get some structure in your data, you should go back to your model and check whether you are doing a good job with your parameters.

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(lr.predict(X_train),lr.predict(X_train)-y_train,c='b',s=40,alpha=0.5)
plt.scatter(lr.predict(X_test),lr.predict(X_test)-y_test,c='g',s=40)
plt.hlines(y=0,xmin=0,xmax=100000000)
plt.title('Residual Plots using Training(blue) and Test(green) data')
plt.ylabel('Residuals')


# In[ ]:


plt.scatter(y_test,pred_test, color = 'green')
plt.xlabel('Y in test set')
plt.ylabel('Predicted Y')


# ### Please upvote if you find this kernel useful
