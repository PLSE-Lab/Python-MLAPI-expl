#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# My first kaggle project - Titanic survival data analysis. I am going to start by loading the data and libraries then some exploratory data analysis at the end I'll use logistic regression to build the model and to predict using the model 
# 
# * Importing Data and packages
# * Exploratory data analysis & handling missing values
# * Handling Categorical variable
# * Training Test split 
# * Model Building & prediction 
# * Conclusion

# ### Step 1: Importing Data and packages

# In[63]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ### Step 2: Exploratory data analysis & Handling Categorical variable & Missing values

# In[64]:


# Checking for missing values
sns.heatmap(train.isnull(), yticklabels=False,cbar=False)

train.isna().sum()
train["Age"].isna().sum()/len(train) # 19 % of age is missing we can deal with the missing data
train["Cabin"].isna().sum()/len(train) # 77 % of the data is missing so we cant consider Cabin for analysis 
train["Embarked"].isna().sum()/len(train) # Only two missing values so we can assign the most frequest value to this


# Based on the above analysis 19 % of the data in age column is missing we can deal with the missing data. 77 % of the data is missing so we cant consider Cabin for our analysis. Only two missing values so we can assign the most frequest value to this

# In[65]:


# Plotting histogram to check how we can deal with the missing value
# Handling Age missing value

plot = train["Age"].hist(bins = 15, color = 'blue', alpha = 0.8)
plot.set(xlabel = "Age", ylabel= 'Count')  # The age is right skewed so its better to use median to fill na
train["Age"].fillna(train["Age"].median(), inplace = True)


# The column 'age' is right skewed so its better to use median to fill na

# In[66]:


# Handling Embarked missing value
plot = train["Embarked"].hist(bins = 15, color = 'blue', alpha = 0.8)
plot.set(xlabel = "Embarked", ylabel= 'Count')  # Most of the passengers boarded from southhampton
train["Embarked"].fillna("S", inplace = True) # Assigning Southhampton to the missing values


# Most of the passengers boarded from southhampton Assigning Southhampton to the missing values
# 
# No I am going to apply the same changes in the test data as well

# In[67]:


# Applying the smae changes in the test data-set
sns.heatmap(test.isna(), yticklabels=False, cbar=False)
test.isna().sum()


# In[83]:


test["Age"].fillna(test["Age"].median(), inplace = True)
test.isna().sum()  # only 1 value is missing in fare column 


# In[84]:


print(test[test['Fare'].isna()]) # The passenger with na value was from 3rd class


# In[85]:


sns.barplot(x = 'Pclass', y = 'Fare', data = test ) 
np.mean(train[train['Pclass'] == 3]) 
test['Fare'].fillna(13.675550, inplace = True) # Replaced Na with the mean of Pclass 3


# There was only 1 value is missing in fare column and The passenger with na value was from 3rd class so that can be Replaced with the mean of 'Pclass' 3

# In[86]:


train.describe()


# In[87]:


grid = sns.FacetGrid(train, col = 'Survived')
grid.map(sns.distplot, "Age") # More younger survived


# In[88]:


sns.barplot('Pclass', 'Survived', data = train) # No wonders, beind 1st class was safest


# In[89]:


sns.barplot('Embarked', 'Survived', data = train)


# Peple boarded from C point apear to have the highest survival rate and people boarded from S has the least survival rate

# In[90]:


summary = pd.pivot_table(train[['Survived','Pclass', 'Embarked']], index = 'Embarked',
                         columns = 'Pclass', aggfunc = 'sum')
summary


# In[91]:


sns.barplot('Sex', 'Survived', data = train) # Clearly being female incresed the changes of survival


# ### Step 3: Handling categorical data
# * Final training set prepration, fintering irrelavent column
# * Creating dummies variable for the columns with categorical data ('Sex' & 'Embarked')
# * Saperating Explanatory Variables and Response Variables in X and y dataframe

# In[92]:


train.columns
final_train = train[['Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
X = pd.get_dummies(final_train)
y = train['Survived']
print(X.sample(5))


# Holdout testing
# * Spliting the data into train, test set to access the model

# In[93]:


# Test train split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = 0.20,
                                                 random_state = 0)


# ### Step 4:  Logistic regression Model building & Prediction
# * Building LogisticRegression model using sklearn library
# * Predicting using the mode
# * Making confusing matrix to access the model
# 

# In[94]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_train,y_train)


# In[95]:


# Making confusion matrix 
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_pred,y_test)
confusion_mat 


# #### Final Prediction
# Finaly I am going run to model to the test data and add the result in the test data frame
# 

# In[100]:


X_test = test[['Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
X_test = pd.get_dummies(X_test)
final_pred = model.predict(X_test)
test["Survived"] = final_pred 
test.head()


# ### Step 5: Logistic regression conclusion
# As per my analysis, The changes of survival of the passengers on titanic was best if the below criterias are met
# * Female
# * 1st Class passenger
# * Young
# 
# As logistic regression is a linear classifier, its better to run some other classification algorythms line KNN or Naive bayes and compare the result of prediction

# In[ ]:




