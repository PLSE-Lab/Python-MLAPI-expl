#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


df = pd.read_csv("../input/breastCancer.csv")


# In[ ]:


df.drop(["id","Unnamed: 32"], axis=1, inplace = True)


# In[ ]:


M = df[df.diagnosis=="M"]
B = df[df.diagnosis == "B"]


# In[ ]:


df.diagnosis = [1 if each =="M" else 0 for each in df.diagnosis]
y = df.diagnosis.values
x_data = df.drop(["diagnosis"],axis=1)


# In[ ]:


df.describe()


# In[ ]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .3, random_state=1)


# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state=1)


# In[ ]:


svm.fit(x_train, y_train)


# In[ ]:


print("accuracy of SVM algorithms", svm.score(x_test,y_test))

