#!/usr/bin/env python
# coding: utf-8

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


# # The first step is to get the data, and display it.

# In[ ]:


df = pd.read_csv("../input/fifa19/data.csv")


# In[ ]:


df


# # Now, we do not need the strings for our example, only the numbers!

# In[ ]:


#Remove all columns with strings

df = df.select_dtypes(exclude = ['object'])


# In[ ]:


df


# # Check if you have any null values. If so, we need to replace them!

# In[ ]:


df.isnull().sum()


# In[ ]:


#Replacing null values with means of respective columns

for colname in df.columns:
    col_mean = df[colname].mean()
    df[colname] = df[colname].fillna(col_mean)


# In[ ]:


#Now there are no null values!

df.isnull().sum()


# In[ ]:


#Let's check what columns we have.

df.columns


# # We're going to use Agility as the dependent variable.

# In[ ]:


#Checks if Agility is over 70. If so, then put 1, else put 0. A new column called isAgile is created in the process.

df['isAgile'] = (df['Agility'] >= 70).astype(int)


# In[ ]:


#Colourful plot!

import seaborn as sns
sns.catplot(x="Overall", y="Agility", kind="swarm", data=df)


# In[ ]:


#A new column came up on the extreme right! 

df


# # We need to split our data into train and test data.

# In[ ]:


#Masking random rows, to select train and test

msk = np.random.rand(len(df)) < 0.8 


# In[ ]:


train = df[msk]
test = df[~msk]


# # Let X be the set of dependent variables, and y be the independent variable.

# In[ ]:


X = train.drop(columns = ['isAgile']).values
y = train['isAgile'].values


# In[ ]:


print(X.shape)
print(y.shape)


# # Fitting the logistic regression model.

# In[ ]:


from sklearn.linear_model import LogisticRegression
regr = LogisticRegression()
regr.fit(X, y)


# # Finding the accuracy score! It seems to be pretty good.

# In[ ]:


from sklearn import metrics
pred = regr.predict(X)
metrics.accuracy_score(pred, y)


# # We test our model, and obtain the accuracy score. Let's check how accurate our model ACTUALLY is.

# In[ ]:


test_pred = test.drop(columns = ['isAgile'])
prediction = regr.predict(test_pred)


# In[ ]:


correct_output = pd.DataFrame({"ID:":test.ID, "isAgile":test.isAgile})
predicted_output = pd.DataFrame({"ID":test.ID, "isAgile":prediction})


# # Checking if the isAgile column in test and the prediction match.

# In[ ]:


correct_predictions = 0

for i in range(len(correct_output)):
    if(correct_output['isAgile'].iloc[i] == predicted_output['isAgile'].iloc[i]):
        correct_predictions += 1


# # The train and test distribution is randomized, however, regardless, the actual accuracy should always be well above 80%.

# In[ ]:


correct_percent = (correct_predictions/len(correct_output))*100
correct_percent

