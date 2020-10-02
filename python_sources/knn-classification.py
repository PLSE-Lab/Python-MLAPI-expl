#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


fruits = pd.read_table("../input/fruits-with-colors-dataset/fruit_data_with_colors.txt")


# In[ ]:


fruits.head(0)


# In[ ]:


lookup_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))


# In[ ]:


lookup_name


# In[ ]:


x = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)


# In[ ]:


knn.fit(x_train,y_train)     #traning the classifier


# In[ ]:


knn.score(x_test,y_test) #testing the classifier and checking accuracy


# In[ ]:


prediction = knn.predict([[20,4,6,7]])
lookup_name[prediction[0]]


# In[ ]:


prediction = knn.predict([[100,4,6,7]])
lookup_name[prediction[0]]


# In[ ]:




