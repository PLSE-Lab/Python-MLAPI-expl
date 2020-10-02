#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Successfully importing the h20 library and initializing it.
# Since, all the data is stored on server.

# In[ ]:


import h2o


# In[ ]:


h2o.init()


# # Importing the data to h2o cluster.

# In[ ]:


data=h2o.import_file("/kaggle/input/iris/Iris.csv")


# In[ ]:


x=data.names
print(x)


# In[ ]:


y="Species"
x.remove(y)


# # Splitting the data into train data which will train our model and test data through which we will to predict.

# In[ ]:


train, test=data.split_frame([0.8])


# # Importing the model

# In[ ]:


from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# # Training the model

# In[ ]:


m=H2ODeepLearningEstimator()
m.train(x,y,train)


# # And finally, predicting the species.

# In[ ]:


m.predict(test)


# # Analyzing some error metrics.

# In[ ]:


m.mse()


# In[ ]:


m.confusion_matrix(train)


# In[ ]:


m.confusion_matrix(test)


# In[ ]:





# In[ ]:




