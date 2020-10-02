#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
test = pd.read_csv("../input/liverpool-ion-switching/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


plt.figure(figsize=(15,7))
plt.xlabel("Open Channels")
plt.ylabel("Counts of Open Channel")
sns.countplot(train['open_channels'])


# In[ ]:


train.groupby('signal')['open_channels'].apply(lambda x: len(set(x))).plot()


# In[ ]:


X = train[['time', 'signal']].values
y = train['open_channels'].values


# In[ ]:


sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:




