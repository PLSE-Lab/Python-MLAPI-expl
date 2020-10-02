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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[ ]:


tf.config.experimental.list_physical_devices('GPU')


# In[ ]:


ls -lh ../input/galaxy-zoo-cleaned/


# In[ ]:


get_ipython().run_line_magic('time', "ds = xr.load_dataset('../input/galaxy-zoo-cleaned/galaxy_train.nc')")
ds


# # Explore data

# In[ ]:


ds['image_train'].isel(sample=range(0, 50, 10)).plot(col='sample', row='channel', x='x', y='y')


# In[ ]:


ds['label_train'].isel(sample=range(0, 50, 10)).plot(col='sample')


# # Fitting model

# In[ ]:


get_ipython().run_cell_magic('time', '', "X = ds['image_train'].data\ny = ds['label_train'].data\n\nX_train, X_valid, y_train, y_valid = train_test_split(\n    X, y, test_size=0.2, random_state=0)")


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(37)
])


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.compile('adam', 'mse')\nhistory = model.fit(\n    X_train, y_train, batch_size=32, epochs=15,\n    validation_data=(X_valid, y_valid)\n)")


# In[ ]:


pd.DataFrame(history.history).plot(marker='o')


# # Evaluate on train & validation data

# In[ ]:


get_ipython().run_line_magic('time', 'y_train_pred = model.predict(X_train)')
r2_score(y_train, y_train_pred)


# In[ ]:


rmse(y_train, y_train_pred)


# In[ ]:


get_ipython().run_line_magic('time', 'y_valid_pred = model.predict(X_valid)')
r2_score(y_valid, y_valid_pred)


# In[ ]:


rmse(y_valid, y_valid_pred)


# # Save trained model

# In[ ]:


get_ipython().run_line_magic('time', "model.save('trained_simple_cnn_galaxy.h5')")


# In[ ]:


ls -lh trained_simple_cnn_galaxy.h5


# In[ ]:




