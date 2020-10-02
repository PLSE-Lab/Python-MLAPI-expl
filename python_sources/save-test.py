#!/usr/bin/env python
# coding: utf-8

# **How many yards will an NFL player gain after receiving a handoff?**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, tqdm
from kaggle.competitions import nflrush
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import lightgbm as lgb


# In[ ]:


env = nflrush.make_env()


# In[ ]:


dfs = []
for df, sample in tqdm.tqdm(env.iter_test()):
    dfs.append(df)
    pred = np.zeros((1, 199))
    env.predict(pd.DataFrame(data=pred, columns=sample.columns))


# In[ ]:


test = pd.concat(dfs, axis=0)
test.shape
    


# In[ ]:


test.to_csv('test.csv', index=False)


# In[ ]:



env.write_submission_file()

