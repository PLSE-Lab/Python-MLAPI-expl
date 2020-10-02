#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/advertising/advertising.csv')


# In[ ]:


data.head()


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


data['Age'].hist(bins = 30)


# In[ ]:


sns.jointplot(x = 'Age', y = 'Area Income', data = data)


# In[ ]:


sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = data, kind = 'kde')


# In[ ]:


sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = data, color = 'green')


# In[ ]:


sns.pairplot(data, hue = 'Clicked on Ad')


# In[ ]:


#Model building and Predictions


# In[ ]:


data.info()


# In[ ]:


from sklearn.model_selection import train_test_split

X = data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage',  'Male']]
y = data['Clicked on Ad']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size =  0.4, random_state= 101)


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Model Fitting

model.fit(train_X, train_y)


# In[ ]:


prediction = model.predict(test_X)


# In[ ]:


# Model Evaluation


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(test_y, prediction))


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, prediction)


# In[ ]:




