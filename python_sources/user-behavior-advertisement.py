#!/usr/bin/env python
# coding: utf-8

# # Advertising Logistic Regresson

# ### Predict whether or not a user will click on an advertisement

# ## Imports 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ad_data = pd.read_csv('../input/advertising.csv')


# ## Exploratory Data Analysis

# In[ ]:


ad_data.head()


# In[ ]:


ad_data.info()


# In[ ]:


ad_data.describe()


# In[ ]:


ad_data['Age'].plot.hist(bins=30)


# In[ ]:


sns.jointplot(x='Age',y='Area Income',data=ad_data)


# In[ ]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')


# In[ ]:


sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)


# In[ ]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# ## Logistic Regression Model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Train and fit a logistic regression model on the training set

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations

# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




