#!/usr/bin/env python
# coding: utf-8

# # How INTERACTIONS help in increasing model performance

# This Notebook will help us to understand how interactions helps us to increase model performance. Before getting into the code lets just understand what INTERACTIONS are, and how can we create one.

# ### MODEL COMPLEXITY

# We often create models that end up being an underfitting or overfitting. Both of them has alot to deal with the Bais-variance trade off as both of them affect our model in a bad way. There are several ways to reduce these problems inorder to get a Best fit model, and out of them one way is to add interactions between the features. This may not work always but would definitely add some value to the model. This is one of the way adding noise in the data inorder to make it more complex, but wait doesn't it tend the model to be overfitting ? and the answer is yes and no. Some times models do require extra dimensions to get more grip over the pattern. But some times when there are already enough dimensions available it is not good to use interactions for improving model performance.

# ### INTERACTIONS

# Interaction between the two features is simply adding another feature, whose values are formed by performing product of observations from each of the two features.

# In[ ]:


# This is a simple demonstration of how interactions are made.
# EX:

feat1=[1,2,3,4]
feat2=[2,3,4,5]
inter_feat=[]
for i  in range(0,4):
    inter_feat.append(feat1[i]*feat2[i])
inter_feat


# Let's apply this concept on a stimulated dataset and see how it improves the accuracy.

# ## Interactions in Boston house dataset

# We would be using boston house dataset availabe in sci-kit learn library. These are ready to use datasets which are free from the null values and meant to be used for practice purposes. 

# Let's firstly import all of the libraries that we require.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# I would be writing comments in each of the cell. This would help me to give you some info about whats happening.

# In[ ]:


from sklearn.datasets import load_boston

# This line of code would import our dataset.


# In[ ]:


l=load_boston()

# This is a bunch format datatype which looks almost similar to a dictionary.


# In[ ]:


l.keys()

# This keys would help us to get the info of all that we want to know about the dataset.


# In[ ]:


l.data
# This loads the data.


# In[ ]:


l.feature_names

# This is a list of all the feature's names that we gonna use.


# Let us make two copies of datasets over here and if you ask me why i would just say bear with me :).

# In[ ]:


boston_data1=pd.DataFrame(data=l.data,columns=l.feature_names)
boston_data2=pd.DataFrame(data=l.data,columns=l.feature_names)

# Here we have created two copies of the datasets.


# In[ ]:


# Lets just check the head of our dataset.
# We would load first five rows.

boston_data1.head(n=5)


# Looks good but lets just add target variable to this dataset and then we would start exploring the dataset

# In[ ]:


boston_data1['Target']=l.target
boston_data2['Target']=l.target

# We add target variable in both the datasets. 


# In[ ]:


# Lets now again check the head of the datset.

boston_data1.head(n=5)


# In[ ]:


# Lets just quickly plot a pairplot to have an overview of whats happening.

sns.pairplot(boston_data1)


# Distributions of each of the variables can be explored and can be transformed inorder to get much better results. But as we are concerned with the use of term INTERACTIONS lets just focus more on it.

# It looks like there is alot of correlation between Independent Variables. Lets check the correlation between correlations.

# In[ ]:


corr_dataset=boston_data1.corr()
corr_dataset


# Alot of them have strong correlations in between them(values above 0.60 i.e.,corr>0.60).
# lets understand more using visualisation.

# In[ ]:


sns.heatmap(corr_dataset)


# We can see there are more darker shades and lighter ones which represent strong relationship between them. Which means this dataset has highly correlated variables. Which affects one the Regression assumption.

# - Independent variables should barely correlate with each other.

# Lets try to use LINEAR REGRESSION model for this problem, but lets just divide our dataset to training and testing sets.

# In[ ]:


X=boston_data1.drop('Target',axis=1)
y=boston_data1['Target']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Here a test size of 20% is used with a random state of 40.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr_bef_inter=LinearRegression()

# Linear model before using interactions.


# In[ ]:


lr_bef_inter.fit(X_train,y_train)


# In[ ]:


lr_bef_inter.score(X_test,y_test)


# Looks like a good score but it should have performed well as it mostly had continous features in it.
# But what was the problem. Did correlation between the independent variables had really affected our model ?

# Lets just find it by applying interactions to our model.

# In[ ]:


# Lets again load the correlation dataset.

corr_dataset


# In[ ]:


sns.heatmap(corr_dataset)


# In[ ]:


# This is the function that we made to get interacting features.

def int_feat(cols):
    col1=cols[0]
    col2=cols[1]
    return col1*col2


# This can be done using some libraries from sci-kit learn. But i actually wanted to make You understand how it is actually done in the background. Although there are many other features that had correlation in between them as i wanted to make this as early as possible, so i had to skip some of them. You can use many of them to improve the model.

# In[ ]:


boston_data2['int_CRIM_RAD']=boston_data2[['CRIM','RAD']].apply(int_feat,axis=1)
boston_data2['int_DIS_ZN']=boston_data2[['DIS','ZN']].apply(int_feat,axis=1)
boston_data2['int_NOX_INDUS']=boston_data2[['NOX','INDUS']].apply(int_feat,axis=1)
boston_data2['int_AGE_INDUS']=boston_data2[['AGE','INDUS']].apply(int_feat,axis=1)
boston_data2['int_DIS_INDUS']=boston_data2[['DIS','INDUS']].apply(int_feat,axis=1)
boston_data2['int_RAD_INDUS']=boston_data2[['RAD','INDUS']].apply(int_feat,axis=1)
boston_data2['int_TAX_INDUS']=boston_data2[['TAX','INDUS']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_INDUS']=boston_data2[['LSTAT','INDUS']].apply(int_feat,axis=1)
boston_data2['int_NOX_AGE']=boston_data2[['NOX','AGE']].apply(int_feat,axis=1)
boston_data2['int_NOX_DIS']=boston_data2[['NOX','DIS']].apply(int_feat,axis=1)
boston_data2['int_NOX_RAD']=boston_data2[['NOX','RAD']].apply(int_feat,axis=1)
boston_data2['int_NOX_TAX']=boston_data2[['NOX','TAX']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_RM']=boston_data2[['LSTAT','RM']].apply(int_feat,axis=1)
boston_data2['int_DIS_AGE']=boston_data2[['DIS','AGE']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_AGE']=boston_data2[['LSTAT','AGE']].apply(int_feat,axis=1)


# This lines of code will add new interacted feature of our selected features to our dataset.


# In[ ]:


# We can see some new features been added to our dataset.

boston_data2.head()


# Previously we had about 11 columns but we just increased them to some more and whatever that we have just added are actually the INTERACTING FEATURES. Now lets see by applying them to our model.

# In[ ]:


X_new=boston_data2.drop('Target',axis=1)
y_new=boston_data2['Target']


# In[ ]:


X_New_train, X_New_test, y_New_train, y_New_test = train_test_split(X_new, y_new, test_size=0.20, random_state=40)


# In[ ]:


# This is the new model that we created.

lr_aft_int=LinearRegression()


# In[ ]:


lr_aft_int.fit(X_New_train,y_New_train)


# In[ ]:


lr_aft_int.score(X_New_test,y_New_test)


# #####  This seems to be a very good improvement to our model as adding interactive features actually helped our model to improve.

# In[ ]:


print('Accuracy of the model before adding interacting features     : {} %'.format(lr_bef_inter.score(X_test,y_test)*100))
print('Accuracy of the model after adding interacting features      : {} %'.format(lr_aft_int.score(X_New_test,y_New_test)*100))


# **Note**: This result can be increased by using more interactions and using better models but we don't do this for now as the aim of this notebook is to show how interactions can be effective to boost our results.

# ## How did this happend...?

# - As there were alot of correlation between the independent features, it was just ruining one of the Linear Regression assumption which got compensated after adding INTERACTIONS features.

# - It means sometimes adding complexity in the model actually helps in the improvisation of the model.

# - Alot more can be done to improve the model performance but this was one of the great technique to improve the model performance.

# ### I hope this one actually actually helped you in gaining some knowledge THANK YOU and please give me an upvote if you really liked this.
