#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h2> First things first</h2>
# We need to load in the data from the csv file into a pandas dataframe and print out the first couple rows to see what we have in terms of data. We also print out the row names to see what type of features we are working with.

# In[ ]:


df = pd.read_csv('../input/train.csv')
print(df.head())


# In[ ]:


print(list(df))


# <h2>Where to start?</h2>
# Since we are trying to predict winPlacePerc, we could explore correlations between vairables and this dependent variable. A heatmap would be a good start to see what kind of correlations we get. From experience playing the game, I think that features revolving around kills will correlate highly with winPlacePerc. I also believe accuracy would be a great indicator; however, that is not an included feature.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), 
            annot=True, 
            linewidths=.7, 
            ax=ax,
            fmt='.1f',
            cmap=sns.cm.rocket_r)


# Well surprisingly the only kill related metric that correlates with win place percentage is killPlace, and that negatively correlates with it. This indicates most of the victors have low kill counts relative to the other players. Maybe going out and killing as much as you can isn't the best strategy! Going to drop the columns that provide no correlation with in place percentage.
# 
# The highest correlated features (including negative correlations) are walking distance, kill place, boosts, and weapons acquired. Lets take a look at these more in depth.

# In[ ]:


df = df.drop(columns = ['groupId', 'maxPlace', 'numGroups', 'roadKills', 'teamKills'])
print(list(df))


# In[ ]:


high_corr = ['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired']
for item in high_corr:
    print('The average for {} was {:.2f}'.format(item, df[item].mean()))
    print('The standard deviation for {} was {:.2f}'.format(item, df[item].std()))
    print('The minimum for {} was {}'.format(item, df[item].min()))
    print('The maximum for {} was {}'.format(item, df[item].max()))
    print('The correlation between win placement and {} is {:.2f}'.format(item, df[item].corr(df['winPlacePerc'])))
    print('________________________')


# <h2>Basic Linear Regression</h2>
# Create a basic linear regression model with the highest correlated features to see results.  Using linear regression as we are predicting a scalar value between 0 and 1.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')

y_train = df['winPlacePerc']
x_train = df[['walkDistance','weaponsAcquired', 'boosts', 'killPlace']]
x_test = test_df[['walkDistance','weaponsAcquired', 'boosts', 'killPlace']]

print(x_train.head())
scaler = MinMaxScaler()
x_train.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']] = scaler.fit_transform(x_train.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']])
x_test.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']] = scaler.fit_transform(x_test.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']])
print(x_train.head())

print(x_train.shape, y_train.shape, x_test.shape)


# In[ ]:


clf = LinearRegression()
clf = clf.fit(x_train, y_train)
fit_score = clf.score(x_train, y_train)
yhat_test = clf.predict(x_test)
print('R2 for train data: {:.4f}'.format(fit_score))


# Training set:<br>
# R2: 0.7565<br><br>    
# 
# Not bad for a linear regression model with only 4 features. Now to write the submission to a file.

# In[ ]:


submission = pd.DataFrame({'Id':test_df['Id'], 
                           'winPlacePerc':yhat_test},
                            columns = ['Id','winPlacePerc'])
submission.to_csv('LR_submission.csv', index=False)


# 
