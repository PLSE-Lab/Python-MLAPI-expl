#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd
import os
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn.preprocessing import StandardScaler as ss
from time import time


# ##Define Working Directory and set diplay number of colm limit

# In[ ]:


pd.options.display.max_columns = 800
os.chdir("../input")


# #### Start H2o

# In[ ]:


h2o.init()


# ### Read the Zipped dataset

# In[ ]:


tr_data=h2o.import_file("fashion-mnist_train.csv")
type(tr_data)
tst_data = h2o.import_file("fashion-mnist_test.csv")


# ### Step 4 Seperate the Predictor (X=> col 2 to 785: pixels) and Target Data set (Y=> col 1 : label)

# In[ ]:


tr_data["label"]=tr_data["label"].asfactor()
tr_data['label'].levels()
X_columns = tr_data.col_names[1:785]
Y_columns = tr_data.col_names[0]
Y_columns


# ### Step 5 Scale X using StandardScaler() 
# ##### As the Data has no categorical Variable, Standardization is not required.
# 
# ### Step 6 Splitting Data in Train and Test set using split_frame

# In[ ]:


train,test = tr_data.split_frame(ratios= [0.7])
train.shape


# In[ ]:


test.shape


# ### Step 7 Modelling and Training using train Data from Train csv

# In[ ]:


Model = H2ODeepLearningEstimator(
                             distribution="multinomial",
                             activation = "RectifierWithDropout",
                             hidden = [50,50,50],
                             input_dropout_ratio=0.2,
                            standardize=True,
                             epochs = 100
                             )


# ##### Train the Model

# In[ ]:


start=time()
Model.train(x = X_columns,
           y = Y_columns,
           training_frame=train
           )
Stop=time()
f"Processing Time = {(Stop-start)/60} min"


# # Predict the test data

# In[ ]:


Out=Model.predict(test[:,1:785])
Out.shape


# ### Data collection for Validating Prediction

# In[ ]:


act=test['label'].as_data_frame()
act['predict'] = Out[0].as_data_frame()
act.head()


# In[ ]:


act.columns


# ### Check the actual v/s Predicted

# In[ ]:


check=(act['label'] == act['predict'])
Accuracy=np.sum(check)/check.size
f"The Prediction is {Accuracy*100} % accurate"


# ### Step 7 Predicting Test.csv Data

# In[ ]:


Out=Model.predict(tst_data[:,1:785])
Out.shape


# ### Data collection for Validating Prediction

# In[ ]:


act=tst_data['label'].as_data_frame()
act['predict'] = Out[0].as_data_frame()
act.head()


# In[ ]:


act.columns


# Accuracy of Prediction for Test.csv

# In[ ]:


check=(act['label'] == act['predict'])
Accuracy=np.sum(check)/check.size
f"The Prediction is {Accuracy*100} % accurate"

