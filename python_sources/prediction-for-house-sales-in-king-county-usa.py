#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# # Importing the dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
dataset.head()


# # Checking Null values in dataset
# **First of all we will check is there any null or nan value in our dataset.For This I will use two methods.
# 1. By using inbuilt method of our data ,i.e., isnull() method
# 2. By using heatmap function of seaborn library

# In[ ]:


dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# **As we can see here that there is no null value.so there is no need to take care of missing values and we can safely move ahead.**

# In[ ]:


dataset.info()


# **As there is no categorical data so we need not to take care of categorical or string data**

# # Making our dependent and independent features

# **Now making our dependent and independent features to test our model and predict for future values.In my independent values I have removed id and date as they merely do not affect the prices of house.Also I have decided to remove the yr_built , yr_renovated and zipcode as they will decrease the accurracy for the prediction of our model.I will show it to you later**

# In[ ]:


X = dataset.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,17,18,19,20]].values
y = dataset.iloc[:, 2].values


# # Splitting the dataset into the Training set and Test set

# **Here I had given a 20% of my whole dataset to the test data as we want to feed the maximum of our data to our training set so that our model can predict with a very high accuracy**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # Traing the train dataset on Random Forest Regression 

# In[ ]:


regressor = RandomForestRegressor(n_estimators = 30)
regressor.fit(X_train, y_train)


# # Calculating the R^2 for our training set
# **I am calculating the R^2 for my training dataset to see how well my model is adapted to the train dataset to predict housing price. Lets hope good and now waiting....**

# In[ ]:


y_pred_train = regressor.predict(X_train)
print(r2_score(y_train,y_pred_train))


# **Oh Wow we achieved a great accuraccy of around 97% and I hope we achieve nearly the same accuracy for the test set**

# # Predicting the Test set results

# In[ ]:


y_pred_test = regressor.predict(X_test)


# # Calculating r2 Score for model trained on test dataset

# In[ ]:


print(r2_score(y_test,y_pred_test))


# **And we have achieved an accuracy of nearly 88% and I think its quit good for me as a beginner.
# Please Do like and upvote my notebook if you liked it.**
