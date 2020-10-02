#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nbody {background-color: gainsboro;} \na {color: #37c9e1; font-family: 'Roboto';} \nh1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;} \nh2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}\nh4 {color: #818286; font-family: 'Roboto';}\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      \n</style>")


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.tail()


# In[ ]:


iris.shape


# In[ ]:


sns.pairplot(iris)


# In[ ]:


sns.barplot(data=iris, x='SepalLengthCm', y='Species')


# In[ ]:


lookup_iris_name = dict(zip(iris.Id.unique(), iris.Species.unique() ))


# In[ ]:


lookup_iris_name


# In[ ]:


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:



knn.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


knn.score(X_test, y_test)


# In[ ]:


iris_pred = knn.predict([[5.1, 3.5, 1.4, 0.2]])

iris_pred[0]


# In[ ]:


pred = knn.predict(X_test)
cm = confusion_matrix(y_test, pred)
cr = classification_report(y_test, pred)


# In[ ]:


print(cm)
print("*"*40)
print(cr)


# In[ ]:


i=0.0
for i in range(0, 10, ):
    iris_pred = knn.predict([[float(i/2), float(i+0.2), float(i*2.0), float(i+0.3)]])
    print(iris_pred[0])
    print("*"*10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




