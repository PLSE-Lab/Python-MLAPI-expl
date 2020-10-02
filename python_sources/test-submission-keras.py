#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
tf.__version__, keras.__version__


# In[ ]:


ls -lh ../input/galaxy-zoo-cleaned/


# # Load model

# In[ ]:


model = keras.models.load_model('../input/galaxy-zoo-cleaned/trained_simple_cnn_galaxy.h5')
model.summary()


# # Load test data

# In[ ]:


df_id = pd.read_csv('../input/galaxy-zoo-cleaned/galaxy_id_test.csv', 
                    header=None, index_col=0, names=['GalaxyID'])
df_id.head()


# In[ ]:


get_ipython().run_line_magic('time', "ds_test = xr.load_dataset('../input/galaxy-zoo-cleaned/galaxy_test.nc')")
ds_test


# In[ ]:


X_test = ds_test['image_test'].values
X_test.shape


# In[ ]:


get_ipython().run_line_magic('time', 'y_test_pred = model.predict(X_test)')


# In[ ]:


y_test_pred.shape


# In[ ]:


# copied from sample submission csv
class_header_str = 'Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6'

class_header = class_header_str.split(',')
len(class_header), class_header[:5]


# In[ ]:


df_value = pd.DataFrame(y_test_pred, columns=class_header)

submission_df = pd.concat([df_id, df_value], axis=1)
submission_df


# In[ ]:


get_ipython().run_line_magic('time', "submission_df.to_csv('test_submission_simple_cnn.csv', index=False)")


# In[ ]:


get_ipython().system('head -n 2 test_submission_simple_cnn.csv')


# In[ ]:




