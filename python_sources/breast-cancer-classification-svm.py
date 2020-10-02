#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# * **IMPORT AND VISUALIZE THE DATA**

# <img src="https://i.imgur.com/rOPlBpK.jpg" width="800px">

# In[ ]:


data = pd.read_csv("../input/data.csv")
data.head()


# Here we have dataset with 2 different classes. M (malignant), B (benign). we have 30 different features in each row<br>
# that describes our data. our dataset is linearly seperable using our 30 features.<br>
# we have 569 instances:<br>
#     212 M<br>
#     357 B<br>

# In[ ]:


data.keys()   # name of all the columns in our data


# In[ ]:


data.describe()


# In[ ]:


# we don't need id, diagnosis and also last column which is NaN
X = data.iloc[:, 2:-1]


# In[ ]:


X.shape


# In[ ]:


Y = data.iloc[:, 1]
Y.shape


# In[ ]:


Y = [1 if i=='M' else 0 for i in Y]


# In[ ]:


Y[1:5]


# In[ ]:


data.tail()


# In[ ]:


# when you specify thr hue, it will show which part of the data is which class based on the target values
sns.pairplot(data, hue='diagnosis', vars = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"])


# As we can see for area_mean, radius_mean and radius_mean features M, B classes have visibly different values<br>
# which is useful for classifiying two labels

# In[ ]:


sns.countplot(data['diagnosis'])


# In[ ]:


sns.scatterplot(x = 'fractal_dimension_se', y = 'concavity_worst', hue = 'diagnosis', data = data)


# In[ ]:


sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = data)


# as it is visible above our two features have very different scales and we need to scale the features to normalize <br>
# data for better results

# In[ ]:


plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(), annot= True)


# * **TRAIN AND TEST MODEL**

# <img src="https://i.imgur.com/yHjAnnu.jpg" width="600px">

# Support Vector Machine is a binary classifier that can detect two classes.
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[ ]:


min_train = X_train.min()
range_train = (X_train-min_train).max() # find biggest difference between min value and any point of dataset
X_train= (X_train - min_train)/range_train


# In[ ]:


min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test= (X_test - min_test)/range_test


# In[ ]:


sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', data = X_train)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


len(y_train)


# In[ ]:


classifier = SVC()
classifier.fit(X_train, y_train)


# In[ ]:


y_predict = classifier.predict(X_test)
y_predict


# In[ ]:


cm = confusion_matrix(y_test, y_predict)


# In[ ]:


sns.heatmap(cm, annot= True)


# <img src="https://i.imgur.com/E0Agtsd.jpg" width="600px">
# <br>
# <br>
# <br>
# <img src="https://i.imgur.com/jggKs6h.jpg" width="600px">
# 

# In[ ]:


param_grid = {'C':[0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel':['rbf']}


# In[ ]:


grid = GridSearchCV(SVC(), param_grid, verbose= 4, refit=True)
grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[ ]:


optimized_preds = grid.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, optimized_preds)
sns.heatmap(cm, annot= True)


# While we have more type 1 error, number of type two errors has decreased to 0

# In[ ]:


print(classification_report(y_test, y_predict))

