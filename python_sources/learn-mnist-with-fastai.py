#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from pathlib import Path
path = Path('/kaggle/input/digit-recognizer/')
out = Path('/kaggle/working/')
trainfile = path/'train.csv'
testfile  = path/'test.csv'
sample = path/'sample_submission.csv'


# In[ ]:


from fastai.tabular import *


# In[ ]:


dftrain = pd.read_csv(trainfile)
dftest  = pd.read_csv(testfile)
dftest['label'] = 'EmptyLabel'


# In[ ]:


dep_var   = 'label'
procs = [Categorify, Normalize]
valid_idx = range(dftrain.shape[0] - 2000, dftrain.shape[0])


# In[ ]:


data = TabularDataBunch.from_df(
    path, dftrain, dep_var, 
    valid_idx=valid_idx,
    procs=procs,
    cat_names=cat_names,
    test_df=dftest
)


# In[ ]:


def show_training_example(data):
    """
    Plot an entry of the data.
    
    This works by extracting values from the data loader, reordering them
    according to their orgininal pixel columns and then plotting
    """
    (cat_x,cont_x),y = next(iter(data.train_dl)) # Get a batch from the data loader
    pix_id = data.train_ds.cont_names # Get the labels. We assume the order matches that of the data
    pix_vals = to_np(cont_x)[0] # Get the pixel values
    label = to_np(y)[0]
    d = {int(p.replace('pixel', '')): v for p, v in zip(pix_id, pix_vals)} # Match the pixel label to the data value
    s = np.array([d[k] for k in sorted(d)]) # Order and reconvert to array
    plt.imshow(s.reshape(28, 28)) # Plot
    plt.title(label)

show_training_example(data)


# In[ ]:


learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


predictions, _ = learn.get_preds(ds_type=DatasetType.Test)
predicted_labels = np.argmax(predictions, 1)


# In[ ]:


def show(i):
    tableline, label = data.test_ds[i]
    x, pix_vals = tableline.data
    pix_id = data.test_ds.cont_names # Get the labels. We assume the order matches that of the data
    d = {int(p.replace('pixel', '')): v for p, v in zip(pix_id, pix_vals)} # Match the pixel label to the data value
    s = np.array([d[k] for k in sorted(d)]) # Order and reconvert to array
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(s.reshape(28, 28)) # Plot
    ax[0].set_title(int(predicted_labels[i]))
    ref = dftest.loc[i].drop(['label']).fillna(0).values
    ax[1].imshow(ref.reshape(28, 28))
    
show(18)


# In[ ]:


submission = pd.DataFrame()
submission['ImageId'] = range(len(predicted_labels))
submission['Label'] = predicted_labels


# In[ ]:


submission.to_csv(out/'mnist.csv', index=False)


# In[ ]:




