#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/breastCancer.csv")


# In[ ]:


df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)


# In[ ]:


df.diagnosis = [1 if each =="M" else 0 for each in df.diagnosis]


# In[ ]:


y = df.diagnosis.values
x_data = df.drop(["diagnosis"], axis=1)


# In[ ]:


x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state=42)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


print("Score : ",model.score(x_test, y_test))
