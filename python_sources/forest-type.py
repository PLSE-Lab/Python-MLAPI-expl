#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


for column in train.columns:
    print(column, train[column].nunique())


# In[ ]:


sns.pairplot(train[['Cover_Type', 'Elevation', 'Slope', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']])


# In[ ]:


sns.pairplot(train[['Cover_Type', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']])


# In[ ]:


sns.pairplot(train[['Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']])


# In[ ]:


#train = train.drop(["Id"], axis = 1)

#test_ids = test["Id"]
#test = test.drop(["Id"], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[ ]:


model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


model.score(X_train, y_train)


# In[ ]:


predictions = model.predict(X_val)
accuracy_score(y_val, predictions)


# In[ ]:


test.head()


# In[ ]:


test_pred = model.predict(test)


# In[ ]:


print(test_pred)


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'ID': test_ids,
                       'TARGET': test_pred})
output.to_csv('submission.csv', index=False)

