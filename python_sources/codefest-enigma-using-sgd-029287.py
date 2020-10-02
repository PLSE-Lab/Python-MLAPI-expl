#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Import the necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Read the Datasets
train = pd.read_csv(os.path.join('../input', 'train_NIR5Yl1.csv'))
test = pd.read_csv(os.path.join('../input', 'test_8i3B3FC.csv'))


# In[ ]:


train.describe()


# In[ ]:


import seaborn as sns
# Explore Fare distribution 
g = sns.distplot(train["Upvotes"], color="m", label="Skewness : %.2f"%(train["Upvotes"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
train["Upvotes"] = train["Upvotes"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


# Explore Fare distribution 
g = sns.distplot(train["Upvotes"], color="m", label="Skewness : %.2f"%(train["Upvotes"].skew()))
g = g.legend(loc="best")


# In[ ]:


g = sns.distplot(train["Reputation"], color="m", label="Skewness : %.2f"%(train["Reputation"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
train["Reputation"] = train["Reputation"].map(lambda i: np.log(i) if i > 0 else 0)

test["Reputation"] = test["Reputation"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g = sns.distplot(train["Reputation"], color="m", label="Skewness : %.2f"%(train["Reputation"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Explore Fare distribution 
g = sns.distplot(train["Views"], color="m", label="Skewness : %.2f"%(train["Views"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
train["Views"] = train["Views"].map(lambda i: np.log(i) if i > 0 else 0)

test["Views"] = test["Views"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g = sns.distplot(train["Views"], color="m", label="Skewness : %.2f"%(train["Views"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Explore Fare distribution 
g = sns.distplot(train["Answers"], color="m", label="Skewness : %.2f"%(train["Answers"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
train["Answers"] = train["Answers"].map(lambda i: np.log(i) if i > 0 else 0)

test["Answers"] = test["Answers"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g = sns.distplot(train["Answers"], color="m", label="Skewness : %.2f"%(train["Answers"].skew()))
g = g.legend(loc="best")


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['ID'], y = train['Upvotes'])
plt.ylabel('Upvotes', fontsize=13)
plt.xlabel('ID', fontsize=13)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['Views'], y = train['Upvotes'])
plt.ylabel('Upvotes', fontsize=13)
plt.xlabel('Views', fontsize=13)
plt.show()


# In[ ]:


#Drop the irrelevant features from the test set
train_labels = np.array(train['Upvotes'])
train_features = train.drop(['Upvotes','ID','Username'], axis=1)
train_features = pd.get_dummies(train_features)
train_features = train_features.fillna(0)
train_features.head(5)


# In[ ]:


#Drop the irrelevant features from the test set
ID_test = test['ID']
test_features = test.drop(['ID','Username'], axis=1)
test_features = pd.get_dummies(test_features)
test_features.head(5)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_features[['Reputation','Views','Answers']] = scaler.fit_transform(train_features[['Reputation','Views','Answers']])
test_features[['Reputation','Views','Answers']] = scaler.fit_transform(test_features[['Reputation','Views','Answers']])
train_features.head(5)


# In[ ]:


#split the training set into train and valid
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(train_features,
train_labels,test_size=0.20, random_state=42)


# In[ ]:


print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Validation Features Shape:', X_valid.shape)
print('Validation Labels Shape:', Y_valid.shape)


# In[ ]:


##from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X_train)  # Don't cheat - fit only on training data
#X_train = scaler.transform(X_train)
#test_features = scaler.transform(test_features)  # apply same transformation to test data


# In[ ]:


from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(loss = 'epsilon_insensitive', verbose=1,eta0=0.1,n_iter=60)

reg.fit(X_train, Y_train)


# In[ ]:


print("training labels",Y_train[1:5])
tr_pred=reg.predict(X_train)
print("predicted labels",tr_pred[1:5])

print("Mean of predictions : ",np.mean(tr_pred))
print("Mean of Y_train  : ",np.mean(Y_train))
print("maximum of predictions : ",max(tr_pred))
print("minimum of predictions : ",min(tr_pred))
print("maximum of Y_train : ",max(Y_train))
print("minimum of Y_train : ",min(Y_train))
errors = (tr_pred-Y_train)/np.mean(Y_train)
print('Mean Absolute Error:', round(np.mean(np.abs(errors)) * 100, 4),'%')

plt.scatter(x=tr_pred,y=Y_train)
plt.xlabel("preds")
plt.ylabel("Y_train")


# In[ ]:


predictions = reg.predict(X_valid)
print(predictions[1:10])
print(Y_valid[1:10])
print("Mean of predictions : ",np.mean(predictions))
print("Mean of valid_labels : ",np.mean(Y_valid))
print("maximum of predictions : ",max(predictions))
print("minimum of predictions : ",min(predictions))
print("maximum of valid_labels : ",max(Y_valid))
print("minimum of valid_labels : ",min(Y_valid))
errors = (predictions-Y_valid)/np.mean(Y_valid)
print(min(errors))
print('Mean Absolute Error:', round(np.mean(np.abs(errors)) * 100, 4),'%')


# In[ ]:


#applying non-linear regression to our dataset
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
tree_reg = RandomForestRegressor(n_estimators = 100, random_state = 100, verbose=1)
tree_reg.fit(X_train, Y_train)


# In[ ]:


Y_pred = tree_reg.predict(X_valid)
error = mean_squared_error(Y_valid, Y_pred)
print(error)


# In[ ]:


import math
print('RMSE:',math.sqrt(error))


# In[ ]:


#Submit the results into the submission file
Upvotes = pd.Series(np.abs(tree_reg.predict(test_features)), name="Upvotes")

results = pd.concat([ID_test,Upvotes],axis=1)

results.to_csv("enigma_submission.csv",index=False)


# In[ ]:




