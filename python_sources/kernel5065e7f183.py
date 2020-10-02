#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization


# In[3]:


#Reading csv file into pandas dataframe
dataset = pd.read_csv('../input/kc_house_data.csv')


# In[4]:


#Part of EDA - Explore the data
dataset.columns


# In[5]:


#Part of EDA - List few values 
dataset.head()


# In[6]:


#Checking the price values
pd.options.display.float_format = '${:,.0f}'.format
dataset['price'].describe()


# In[7]:


#Checking how the price value is distributed
import seaborn as sns
from scipy.stats import norm
ax = sns.distplot(dataset['price']/1000, bins=20, kde=False);
ax.set(xlabel='price(Units of $1000)', ylabel='Frequency')


# In[12]:


#Checking the training data to see the pattern between size of home and price
data = pd.concat([dataset['price'], dataset['sqft_living']], axis=1)
data.plot.scatter(x='sqft_living', y='price', ylim=(3,8000000));


# In[13]:


print("Rows & Columns of the dataset: ", dataset.shape)


# In[14]:


#Check for missing data
missing_values = dataset.isnull().sum().sort_values(ascending=False)
missing_values.head(100)


# In[15]:


#Training the model to predict the price based on one feature - Sqft_living
space = dataset['sqft_living']
price = dataset['price']
x = np.array(space).reshape(-1, 1)
y = np.array(price)


# In[17]:


#Allocation 20% of the data for validation or testing
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state = 0)


# In[18]:


#Checking the total size of training and test dataset
print("Training set size: ",len(xtrain))
print("Training result size:", len(ytrain))
print("Test set size: ",len(xtest))
print("Test result size: ",len(ytest))


# In[20]:


#Training the linear regression model with one feature data - xtrain and results ytrain 
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(xtrain,ytrain)


# In[21]:


#Accuracy after training with one feature dataset
accuracy = regression.score(xtest, ytest)
#Formatting accuracy value to be a percentage value
"Accuracy: {}%".format(int(round(accuracy * 100)))


# In[22]:


#Generating heatmap to find top 10 features that correlate with price of house
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[23]:


#Creating multiple linear regression based on important features selected from heatmap above
feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'sqft_above', 
                'sqft_living15', 'view', 'sqft_basement', 'lat']
predictors = dataset[feature_cols]
price = dataset['price']


# In[27]:


xtrain,xtest,ytrain,ytest = train_test_split(predictors,price,test_size=0.2,random_state = 0)


# In[28]:


regressor = LinearRegression()
regressor.fit(xtrain, ytrain)


# In[29]:


accuracy = regressor.score(xtest, ytest)
"Accuracy: {}%".format(int(round(accuracy * 100)))

