#!/usr/bin/env python
# coding: utf-8

# # **Logistic Regression**
# 
# Use Logistic Regression model to predict whether or not a user will click on an ad based off the features of that user.

# Import required Libraries

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the advertising.csv file yo a dataframe ad_data

# In[ ]:


ad_data = pd.read_csv("../input/advertising/advertising.csv")


# In[ ]:


ad_data.head()


# In[ ]:


ad_data.info()


# In[ ]:


ad_data.describe()


# # **Exploratory Data Analysis**

# In[ ]:


ad_data['Age'].plot.hist(bins=30)


# In[ ]:


sns.jointplot(x="Age", y="Area Income",data=ad_data)


# In[ ]:


sns.jointplot(x="Age",y="Daily Time Spent on Site",data=ad_data,kind='kde',color='red')


# In[ ]:


sns.jointplot(x="Daily Internet Usage",y="Daily Time Spent on Site",data=ad_data,color='green')


# In[ ]:


sns.pairplot(ad_data,hue="Clicked on Ad")


# # **Logistic Regression**
# Train test split and train the model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=ad_data[["Daily Time Spent on Site", "Age", "Area Income","Daily Internet Usage","Male"]]
y=ad_data['Clicked on Ad']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# # **Evaluating Performance** 
# 
# Evaluating performance with classification report, confusion model and accuracy score of Logistic Regression model

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(logmodel.score(X_test,y_test))


# In[ ]:




