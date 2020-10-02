#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv("../input/dataset.csv")


# In[ ]:


data.head()


# In[ ]:


labels=data.activity


# In[ ]:


data_dropped=data.drop(["username","activity"],axis=1)


# In[ ]:


data_dropped.head()


# In[ ]:


data_dropped=data_dropped.set_index("date")


# In[ ]:


data_dropped.head()


# In[ ]:


features=data_dropped.values


# In[ ]:


features


# In[ ]:


data_time_dropped=data_dropped.drop(["time"],axis=1)


# In[ ]:


data_time_dropped.head()


# In[ ]:


features=data_time_dropped.values


# In[ ]:


features


# In[ ]:


LABELS=labels.values


# In[ ]:


FEATURES=features


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(FEATURES,LABELS,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RanFor=RandomForestClassifier(n_estimators=100,random_state=1)


# In[ ]:


RanFor.fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import *


# In[ ]:


accuracy_score(y_train,RanFor.predict(x_train))


# In[ ]:


accuracy_score(y_test,RanFor.predict(x_test))


# In[ ]:


precision_score(y_test,RanFor.predict(x_test))


# In[ ]:


recall_score(y_test,RanFor.predict(x_test))


# In[ ]:


f1_score(y_test,RanFor.predict(x_test))


# In[ ]:




