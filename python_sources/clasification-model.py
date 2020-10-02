#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
dataset.head()


# ## Encoding the columns with bicategorical data

# In[ ]:


dataset['workex'].replace(to_replace = 'Yes', value = 1, inplace = True)
dataset['workex'].replace(to_replace = 'No', value = 0, inplace = True)
dataset['status'].replace(to_replace = 'Placed', value = 1, inplace = True)
dataset['status'].replace(to_replace = 'Not Placed', value = 0, inplace = True)
dataset.head()


# ## Data Cleaning

# In[ ]:


data = dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p','mba_p', 'status']]
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data.head()


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state =1)


# ## Training the Kernel SVM model on the Training set

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# ## Predicting the Training set results

# In[ ]:


y_pred_train = classifier.predict(X_train)


# ## Assessing Model Performance on training set

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
accuracy_score(y_train, y_pred_train)


# ## Predicting the generated Test set results

# In[ ]:


y_pred_test = classifier.predict(X_test)


# ## Assessing Model Performance on test set

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
accuracy_score(y_test, y_pred_test)


# In[ ]:




