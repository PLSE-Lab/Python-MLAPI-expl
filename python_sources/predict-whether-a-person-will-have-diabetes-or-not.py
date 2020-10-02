#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3>Import the Dataset </h3>

# In[ ]:


data = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')
data.head()


# <h3>Remove zero values and Import the mean values</h3>

# In[ ]:


not_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for column in not_zero:
    data[column] = data[column].replace(0,np.NaN)
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.NaN,mean)
    


# <h3>Split the Dataset</h3>

# In[ ]:


X = data.iloc[:, 0:8]
y = data['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)


# <h3> Choose a value of 'K' </h3><br>
# <h4><ol><li>sqrt(n), Where 'n' is total number of data points</li><br>
#     <li>Odd value of 'K' is selected to avoid confusion between two classes of data</li></ol> </h4>

# In[ ]:


import math
math.sqrt(len(y_test))


# <h4> '20'  is even number,  so we will choose odd number for near the 20.<br><br><br> K = 19 </h4><br><br>

# <h3>KNN - Algorithm</h3>

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')
knn.fit(X_train,y_train)


# <h3> Predict the Values</h3>

# In[ ]:


y_pred = knn.predict(X_test)
y_pred


# <h3>Evaluate the Score</h3>

# In[ ]:


accuracy_score(y_pred,y_test)


# <h3> Make a Prediction</h3>

# In[ ]:


prediction=knn.predict([[6,148.0,62.0,35.0,455.0,33.6,0.627,30]])
if prediction ==1:
    print("The person have Diabetes")
else:
    print("The person is not have Diabetes")
prediction


# <br>
# <h2>Conclusion:</h2><br>
#     
# <h3> When insulin level is high, that person will be diagnosed with diabetes. So the prediction is correct.</h3> 
