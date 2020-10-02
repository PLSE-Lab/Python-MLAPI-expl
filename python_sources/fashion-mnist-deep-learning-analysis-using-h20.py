#!/usr/bin/env python
# coding: utf-8

# # Deep Learning on Fashion MNIST Dataset
# 
# ## Import standard libraries
# 
# ## Import preprocessing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import time

from sklearn.preprocessing import StandardScaler

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# In[ ]:


os.listdir("../input")


# ## H20 initiation

# In[ ]:


h2o.init()


# In[ ]:


## Read the dataset - both train and test using H20


# In[ ]:


train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")
test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head(5)


# ## To check the data as image!!!
# 
# ### First convert into Pandas Dataframe
# #### Take any one row and first column till last column
# #### Reshape into 28x28 matrix
# #### use imshow to display the image

# In[ ]:


tmp_df = train.as_data_frame()


# In[ ]:


tmp_df.label.unique()


# In[ ]:


tmp_r = tmp_df.values[5,1:]
tmp_r.shape


# In[ ]:


tmp_r = tmp_r.reshape(28,28)


# In[ ]:


tmp_r.shape


# In[ ]:


plt.imshow(tmp_r)
plt.show()


# #### ******************** Data Modelling using H20 ***********************************
# 
# #### This data is a balanced set and no need to use SMOTE

# #### Separate out target and predictors from train data

# In[ ]:


y_target = train.columns[0]
y_target


# In[ ]:


X_predictors = train.columns[1:785]


# In[ ]:


train["label"] = train["label"].asfactor()


# In[ ]:


train["label"].levels()


# In[ ]:


model_h2o = H2ODeepLearningEstimator(
                distribution="multinomial",
                activation="RectifierWithDropout",
                hidden=[50,50,50],
                input_dropout_ratio=0.2,
                standardize=True,
                epochs=1000
                )


# In[ ]:


start = time()
model_h2o.train(X_predictors, y_target, training_frame= train)
end = time()
(end-start)/60


# In[ ]:


result = model_h2o.predict(test)


# In[ ]:


result.shape
result.as_data_frame().head(5)


# In[ ]:


re = result.as_data_frame()


# In[ ]:


re["predict"]


# In[ ]:


re["actual"] = test["label"].as_data_frame().values


# In[ ]:


out = (re["predict"] == re["actual"])
np.sum(out)/out.size


# In[ ]:




