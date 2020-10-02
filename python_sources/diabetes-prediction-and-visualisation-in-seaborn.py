#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import operator


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv("diabetes.csv")
data.head()


# In[ ]:


print(data.info())


# In[ ]:


print(data.describe())


# In[ ]:


g=data["Glucose"]!=0
data=data[g]


# In[ ]:


bp=data["BloodPressure"]!=0
data=data[bp]


# In[ ]:


st=data["SkinThickness"]!=0
data=data[st]


# In[ ]:


i=data["Insulin"]!=0
data=data[i]


# In[ ]:


b=data["BMI"]!=0
data=data[b]


# In[ ]:


data.describe()


# In[ ]:


sns.set_style("darkgrid")
sns.FacetGrid(data, hue="Outcome", )    .map(plt.scatter, "Pregnancies", "Outcome")    .add_legend()
plt.show()


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(data, hue="Outcome", kind="scatter")
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


x=data.drop(["Outcome"], axis=1)
y=data["Outcome"]

x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=2)

cls=KNeighborsClassifier(n_neighbors=28, weights='distance')
cls.fit(x_train, y_train)
prediction=cls.predict(x_test)


# In[ ]:


print(metrics.accuracy_score(prediction, y_test))
print(metrics.f1_score(prediction , y_test, average='macro'))
print(metrics.confusion_matrix(prediction, y_test))

