#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Scoring customer propensity to purchase
Training a model based on a shoppers actions on a website to identify the best prospects who did not purchase yesterday.


# **Notes**; I am a beginner at machine learning and I've written this kernal to share what I am learning with others, please feel free to share feedback and advice as this will help me and others reading this kernal. This data is sampled and all UserIDs are dummies.

# ### Introduction
# We have many visitors to our website every day, some purchase but many do not. We spend money re-targeting past visitors, we'd like to optomise this activity by targeting the visitors who are more likely to convert. To do this, we've taken data showing which parts of our website users interacted with, our questions are:
# 
# 1. Which of these interactiuons effect a users likelyhood to purchase?
# 2. Can we score visitors from yesterday who did not purchase, to see who the most valauve prospects are?
# 
# ![](https://image.ibb.co/ecGtqy/stats_2.png)

# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics


# First let's load in the training data file and take a look at it...

# In[3]:


train = pd.read_csv('../input/training_sample.csv')

train.dtypes


# Okay, so we have a bunch of integer columns, each one reflecting an action on the website, and oneobject column, which looks like a user identifier, Let's take a look with some more methods...

# In[4]:


print(train.describe())

print(train.info())


# So it looks like this data is just below half a million rows, with a total of 25 columns. Let's take a look at some of the data using HEAD...

# In[ ]:


train.head()


# Here we can see a clear snspshot of the data, we have 1's or 0's in the columns, indicating wheather or not a user interacted with these areas of the website. The last colum shows whether the user ordered or not, this will be important!

# 
# ### Is there any correlation
# In order to answer our first question, we can start by exploring any correlation between there individual website actions and an order, since we have all thes fields in our data.

# We've got quite a few fields, so let's start with a heatmap to view correlations...

# In[ ]:


import seaborn as sns
corr = train.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, vmax=0.5, center=0,
            square=True, linewidths=2, cmap='Blues')
plt.savefig("heatmap.png")
plt.show()


# Interesting - there seems to be a strong correlation between visitors who ordered and visitors who saw the checkout, this makes sense! There are also strong correlations for people who cheked out delivery times and added items to their shopping cart - let's get a closer look at the correlations for orders...

# In[6]:


train.corr()['ordered']


# Alright! Looks like our initial insights from the heatmap were correct, users who checked out the delivery options on a product detail page have an almost 80% correlation to orders, there's definately something in here we can use! But wait...it looks like there isn't much correlation between users on a mobile and orders, so we should proabley remove this field from our predictor.

# ### Let's get predicting!
# First we build our predictor and targets variables, we're going to drop 'ordered' form our predictors, as it is our target variable. We'll also remove 'UserID', as it has no impact on likelyhood to order and 'device_mobile' as we've seen it has a negative correlation to orders.

# In[7]:


# Drop columns with low correlation
predictors = train.drop(['ordered','UserID','device_mobile'], axis=1)
# predictors = train[['checked_delivery_pdp', 'basket_icon_click', 'sign_in', 'saw_checkout']]
targets = train.ordered


# Let's take a look at our predictor columns to check we've included everything we wanted, and not left in something we shouldn't have...

# In[8]:


print(predictors.columns)


# Now we split our data into train and test, with a test size of 30%.

# In[9]:


X_train, X_test, y_train, y_test  =   train_test_split(predictors, targets, test_size=.3)

print( "Predictor - Training : ", X_train.shape, "Predictor - Testing : ", X_test.shape )


# For our model we are going to use a naise bayes classififer, below we instantiate it, fit it, then predict using it, then we an analyse the accuracy of our predictions...

# In[10]:


from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier=classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)

#Analyze accuracy of predictions
sklearn.metrics.confusion_matrix(y_test,predictions)


# And apply an accuracy score to our model...

# In[ ]:


sklearn.metrics.accuracy_score(y_test, predictions)


# ### Now to predict on the previous days visitors!
# Start by loading in our sample data of the **previous days visitors who did not order**.

# In[ ]:


yesterday_prospects = pd.read_csv('../input/testing_sample.csv')


# Now let's explore this DataFrame and check everything is as expected...

# In[ ]:


print(yesterday_prospects.info())


# We're going to drop UserID before we predict on this data, so that is matches our training set, but before we do let's pop it into another variable, so we can pull back this identifier later. Once that's done we can drop our unwanted fields and print the head() to check our data...looking good?

# In[14]:


userids = yesterday_prospects.UserID

yesterday_prospects = yesterday_prospects.drop(['ordered','UserID','device_mobile'], axis=1)

print(yesterday_prospects.head(10))


# Let's check the shape too, to confirm it is what our model will expect to recieve (e.g. the same number of columns)

# In[15]:


yesterday_prospects.shape


# Now we'll run our predictions and insert them into a field called 'propensity', print the head, and check it's all come togeather...

# In[16]:


yesterday_prospects['propensity'] = classifier.predict_proba(yesterday_prospects)[:,1]

print(yesterday_prospects.head())


# Looks good! Now we want to bring out UserIDs back, so we can identify these users in the future (note, these are dummy IDs).

# In[17]:


pd.DataFrame(userids)
results = pd.concat([userids, yesterday_prospects], axis=1)


# All done - let's take a look at our results data frame:

# In[20]:


print(results.head(30))


# In[21]:


results.to_csv('results.csv')

