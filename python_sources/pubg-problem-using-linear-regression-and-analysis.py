#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')


# In[ ]:


train.head()


# **Exploratory Data Analysis
# **
# ****
# Let us first check the kill count of players in each match

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='kills',data=train,palette='RdBu_r')


# As seen, there are many players with 0 kills and the number of people with more kills keeps on decreasing. The maximum kills in a single match is 72.
# Now, let us see how no. of kills depends upon the win placement.

# In[ ]:


sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3)
plt.show()


# Hence, it is clear that killing has a correlation with winning.
# ****
# Let us group these players based on the number of people killed.
# 
# 

# In[ ]:


kills = train.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])

plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()


# Now, let us check how boosting and healing are correlated with win place percentage.

# In[ ]:


sns.jointplot(x="winPlacePerc", y="heals", data=train, height=10, ratio=3)
plt.show()


# In[ ]:


sns.jointplot(x="winPlacePerc", y="boosts", data=train, height=10, ratio=3)
plt.show()


# Clearly, healing and boosting are correlated to winPlacePerc.
# ****
# Let us now check the correlation of all variables with one another.

# In[ ]:


x,y = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=y)
plt.show()


# So in our case we have to check the target variable winPlacePerc. There are a few variables with medium to high correlation with the target variable, with walkDistance having highest positive correlation of 0.8 and killPlace having highest negative correlation of -0.7. 
# ****
# In a game of PUBG, a maximum of 100 players can join at any time. But not everytime all 100 players are there in the game. There is no variable that gives us the number of players joined. So lets create one.

# In[ ]:


train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')


# In[ ]:


data = train.copy()
data = data[data['playersJoined']>49]
plt.figure(figsize=(15,10))
sns.countplot(data['playersJoined'])
plt.title("Players Joined",fontsize=15)
plt.show()


# Based on this variable, you could do feature engineering to improve accuracy of your model. For feature engineering you could refer to this notebook - https://www.kaggle.com/deffro/eda-is-fun
# ****
# I will now build the model with the help of linear regression.

# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Id','groupId','matchId','playersJoined'],axis=1)


# In[ ]:


train['winPlacePerc']=pd.to_numeric(train['winPlacePerc'],errors = 'coerce')


# Now, there is one categorical column, i.e., matchType. So we have to convert that column to numeric entry.

# In[ ]:


train = pd.get_dummies(train,columns=['matchType'])


# In[ ]:


train.head()


# In[ ]:


train = train.dropna(how = 'any')


# We now split our data into train and test data.

# In[ ]:


from sklearn.model_selection import train_test_split
X= train.drop('winPlacePerc',axis= 1)
y= train['winPlacePerc']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.3, random_state = 101)


# We now build our model using Linear Regression.

# In[ ]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)


# In[ ]:


linear_model.score(X_train,y_train)


# As we can see, we get an accuracy of 83.98% which is quiet good. Now let us check our model on the testing data.

# In[ ]:


predictions = linear_model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
linear_model_mse = mean_squared_error(predictions,y_test)
linear_model_mse


# As we can see, we get a mean squared error of 0.015. Great work!
# 

# In[ ]:




