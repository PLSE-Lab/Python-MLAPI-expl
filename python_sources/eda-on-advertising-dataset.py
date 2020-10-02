#!/usr/bin/env python
# coding: utf-8

# # Performing Logistic Regression on Advertising Dataset

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/advertising/advertising.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull()


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='Accent')


# In[ ]:


data.describe()


# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(x='Age', data=data)


# In[ ]:


sns.jointplot(data['Area Income'], data['Age'])


# In[ ]:


sns.jointplot(data['Age'], data['Daily Time Spent on Site'])


# In[ ]:


sns.jointplot(data['Daily Time Spent on Site'], data['Daily Internet Usage'])


# In[ ]:


sns.pairplot(data, hue='Clicked on Ad')


# In[ ]:


data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)


# In[ ]:


X = data.drop(['Clicked on Ad'], axis = 1)              
y = data['Clicked on Ad']                               


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[ ]:


predictions = logmodel.predict(X_test)
predictions


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
accuracy

