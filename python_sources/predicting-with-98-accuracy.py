#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Data Viz
import seaborn as sns
import matplotlib.pyplot as plt

#ML
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, LeaveOneOut


# # Using machine learning to predict types of wine

# In[ ]:


data = pd.read_csv('/kaggle/input/winedataset/WineDataset.csv')
data.head()


# In[ ]:


# 3 categories
data['Wine'].unique()


# In[ ]:


data.shape


# In[ ]:


X = data.drop('Wine', axis=1).values #features


# In[ ]:


y = data['Wine'].values #labels


# In[ ]:


trainX, testX, trainy, testy = train_test_split(X, y, random_state=3)


# In[ ]:


model = GaussianNB()


# In[ ]:


model.fit(trainX, trainy)


# In[ ]:


p = model.predict(testX)


# In[ ]:


accuracy_score(testy, p)


# In[ ]:


scores = cross_val_score(model, X, y, cv=LeaveOneOut())


# In[ ]:


#Perfect
scores.mean()


# In[ ]:


cmap = sns.cm.rocket_r
sns.heatmap(confusion_matrix(testy, p), cmap=cmap, annot=True)

plt.title('Confusion Matrix')

plt.xlabel('Predictions')
plt.ylabel('Real values')

plt.show()

