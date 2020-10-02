#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Different types of Iris flowers

# In[ ]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# # Analysing and Visualizing the Data

# In[ ]:


iris = sns.load_dataset('iris')
iris.head()


# In[ ]:


# Normal pairplot with numerical data
sns.pairplot(iris)


# In[ ]:


# sepating the data as per the species
sns.pairplot(iris, hue= 'species')


# In[ ]:


sns.set_style('darkgrid')
setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)


# In[ ]:


versicolor = iris[iris['species']=='versicolor']
sns.kdeplot(versicolor['sepal_width'], versicolor['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)


# In[ ]:


virginica = iris[iris['species']=='virginica']
sns.kdeplot(virginica['sepal_width'], virginica['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)


# # Prediction

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X = iris.drop('species', axis=1)
y = iris['species']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)

model = SVC()
model.fit(train_X, train_y)

prediction = model.predict(test_X)


# # Accuracy

# In[ ]:


print(confusion_matrix(test_y, prediction))
print('\n')
print(classification_report(test_y, prediction))


# As we can see above we got 98% accuracy with classificaion_report and showing one wrong predicted value in confusion_matrix

# In[ ]:




