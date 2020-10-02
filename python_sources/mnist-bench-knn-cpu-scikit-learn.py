#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# # Load datasets

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\npath_input = Path('/kaggle/input/digit-recognizer/')\n\ndf_train = pd.read_csv(path_input / 'train.csv')\ndf_test  = pd.read_csv(path_input / 'test.csv')\n\nX_train = df_train.iloc[:, 1:].values.reshape(-1, 28, 28)\ny_train = df_train.iloc[:, 0 ].values\n\nX_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)\n\nX_test  = df_test.values.reshape(-1, 28, 28)")


# In[ ]:


fig = plt.figure(figsize=(18, 9))
for row in range(5):
    for col in range(10):
        ax = fig.add_subplot(5, 10, 1+10*row+col)
        ax.imshow(X_train[10*row+col])
        ax.set_axis_off()
        ax.set_title(y_train[10*row+col])


# # Train

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = KNeighborsClassifier(n_neighbors=10)\nmodel.fit(X_train.reshape(-1, 28*28), y_train)')


# # Predict

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nh_valid = model.predict(X_valid.reshape((-1, 28*28)))')


# # validation

# In[ ]:


confusion_matrix(y_valid, h_valid)


# In[ ]:


accuracy_score(y_valid, h_valid)


# In[ ]:


fig = plt.figure(figsize=(18, 9))
for row in range(5):
    for col in range(10):
        ax = fig.add_subplot(5, 10, 1+10*row+col)
        ax.imshow(X_valid[y_valid != h_valid][10*row+col])
        ax.set_axis_off()
        ax.set_title(h_valid[y_valid != h_valid][10*row+col])

