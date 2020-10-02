#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Objective: To learn modeling with H2O Deaplearning 


# In[ ]:


# Call libraries
# basic libraries
import pandas as pd
import numpy as np
import os
# For plotting
import matplotlib.pyplot as plt

# For measuring time elapsed
from time import time

# Model building
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# Change ipython options to display all data columns
pd.options.display.max_columns = 300


# In[ ]:


#Read file
train = pd.read_csv("../input/fashion-mnist_train.csv")


# In[ ]:


train.shape       # (60000,785)


# In[ ]:


train.head(1)


# In[ ]:


train['label'].value_counts() # balanced classes 


# In[ ]:


# Get the first row excluding first column
#    First column contains class labels and 
#    other columns contain pixel-intensity values
abc = train.values[1, 1:]
abc.shape    # (784,)


# In[ ]:


abc = abc.reshape(28,28)   # Reshape to 28 X 28


# In[ ]:


# And plot it
plt.imshow(abc)
plt.show()


# In[ ]:


#Get list of predictor column names and target column names
#     Column names are given by H2O when we converted array to
#     H2o dataframe
X_columns = train.columns[1:786]        # Only column names. No data


# In[ ]:


y_columns = train.columns[0]
y_columns


# In[ ]:


train["label"].unique()


# In[ ]:


h2o.init()


# In[ ]:


train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")
test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")


# In[ ]:


#Get list of predictor column names and target column names
#     Column names are given by H2O when we converted array to
#     H2o dataframe
X_columns = train.columns[1:785]        # Only column names. No data
  # C1 to C786


# In[ ]:


y_columns = train.columns[0]
y_columns


# In[ ]:


train["label"]=train["label"].asfactor()


# In[ ]:


train['label'].levels()


# There are [multiple algorithms](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)  available in the H2O module. 

# In[ ]:


model = H2ODeepLearningEstimator(
                             distribution="multinomial",
                             activation = "RectifierWithDropout",
                             hidden = [32,32,32],
                             input_dropout_ratio=0.2,  
                             standardize=True,
                             epochs = 500
                             )


# In[ ]:


start = time()
model.train(X_columns,
               y_columns,
               training_frame = train)


end = time()
(end - start)/60


# In[ ]:


result = model.predict(test[: , 1:785])


# In[ ]:


result.shape       # 5730 X 3


# In[ ]:


result.as_data_frame().head(10)   # Class-wise predictions


# In[ ]:


#  Ground truth
#      Convert H2O frame back to pandas dataframe
xe = test['label'].as_data_frame()
xe['result'] = result[0].as_data_frame()
xe.head()
xe.columns


# In[ ]:


#Accuracy
out = (xe['result'] == xe['label'])
np.sum(out)/out.size

