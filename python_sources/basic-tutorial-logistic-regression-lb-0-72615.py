#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression hands on
# 
# This is a hands on exercise for logistic regression machine learning algorithm. Logistic regression is a very popular classification algorithm . It is a statistical machine learning technique for classifying records. Basically we use one or more independent variables (X) to caluclate dependent variable having a categorical value like 0 or 1 , yes or no, true or false. In this example we are using binary classification i.e 0/1.
# 
# But hey can't we use Linear Regression for that. Yes we can but linear regression gives us the value result  ie. in this case the taget of 5 years will be met or not which is a true (1) or false (0) classification. But well we can't be 100% sure if our target will be met or not. We can only calculate the probability of this. So whenever we have to calculate the probability of the occurence of some event based upon some other given circumstances (features or X in our case) we use Logistic Regression. Linear regression can be used to map those probablistic values into labels or categories later.
# 
# Probability of a 5 years target going to meet  =  1-  (Probability of a 5 years target not going to meet) given circumstances X
# 
# ![Logistic Regression](https://www.solver.com/files/images/xlminer/Lreg/graph.gif)
# 
# Now lets go on with the code

# First we will import all required libraries
# 1. numpy :- For processing data
# 2. matplotlib :- For visulaization
# 3. Seaborn :- Same for visulaization
# 4. LogisticRegression :- For model training and prediction
# 5. train_test_split :- For validating our model
# 6. os :- For reading input
# 7. pandas :- For reading input
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
print(os.listdir("../input"))


# Will be readind data by pandas

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Before processing our data lets have a look at it by printing few of its variables

# In[ ]:


train.shape


# In[ ]:


train.describe()


# We saw that all values are continous and we have a relatively smaller set of data hence LogisticRegression model can work perfectly for us. so lets understand our features than

# In[ ]:


corrmat = train.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# From the above coerelation map we are trying to get coerelation between features and we can see that many of the features are highliy coerelated . For ex :- like 3P Made/ 3PA, DREB/ REB. Now seeing this either we can drop one of the feature out of these pairs. For ex :- we can drop 3PA and REB. But before dropping any column we should first try to fit our data on all columns as losing data most like dont help.
# 
# But in case we have thousands of features there we should drop highly coerelated columns to avoid overfitting.

# Nowe we will see if our data has any null values

# In[ ]:


train.columns[train.isna().any()].tolist()


# So 1 column has null values but instead of dropping it we will fill it with some arbitary high value which can count as an outlier

# In[ ]:


train.fillna(9999, inplace=True)
test.fillna(9999, inplace=True)


# Now we will create or X and Y columns.
# 
# X :- It is traing features and we have to see which features affect out target value. Well name obviously doesnt affect out target result. so is Id and well TARGET_5Yrs is the result so we got to remove that from X
# Y :- It is target value so we assign it train['TARGET_5Yrs']

# In[ ]:


X = train.drop(["TARGET_5Yrs", "Id", "Name"], axis = 1)
Y = train["TARGET_5Yrs"]


# It is very important to validate your model . We can do that by calculating cross validation score. Sklearn provides an amazing function for that which is train_test_split . This divides our traing data into 2 parts. One one part we will do training and on another we will test and calculate cross validation score

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2222)


# Finally lets create our model. We will keep n_jobs=-1 as we want this algotithm to run as fast as it can with multiple threads. Random state is a random number which you can have of your choice. Doesn't really affect much

# In[ ]:


model = LogisticRegression(n_jobs=-1, random_state=10)


# Now we fit our model with train data

# In[ ]:


model.fit(X_train, y_train)


# Lets see our CV score

# In[ ]:


model.score(X_test, y_test)   


# 0.71 its is pretty good for now. Hence we will go ahead with this model

# Now lets prepare our test data. Same thing here also we will be dropping Name and Id column

# In[ ]:


test_X = test.drop(["Name", "Id"], axis=1)


# We will predict on test data amd finally create a submission

# In[ ]:


test_X = model.predict(test_X)


# In[ ]:


submission = pd.DataFrame({"Id":test["Id"],
                         "TARGET_5Yrs":test_X})


# Lets have a look at our submission

# In[ ]:


submission.head()


# Just to be sure I take the above step. Now you can create your submission file and commit your code

# In[ ]:


submission.to_csv("submission.csv", index=False)


# **Hope this was pretty easy and informative. If you like this please upvote and comment for feedback**

# In[ ]:




