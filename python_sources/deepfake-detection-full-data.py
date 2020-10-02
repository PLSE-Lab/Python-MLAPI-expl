#!/usr/bin/env python
# coding: utf-8

# # Kaggle Kernal with all the datasets preloaded
# View this disussion for more information: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/134420

# In[ ]:


import os
import glob
import cv2
import pandas as pd
import re
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


meta = glob.glob('../input/deepfake-detection-faces-*/*.csv')
meta.sort(key=lambda f: int(re.sub('\D', '', f)))

dfs = []
for path in meta:
    df = pd.read_csv(path)
    df['path'] = ''
    path = path.split("/")[:-1]
    path = path[0] + '/' + path[1] + '/' + path[2] + '/'
    for i in range(len(df)):
        df.loc[i]['path'] = f'{path}{df.loc[i]["filename"][:-4]}'
    dfs.append(df)

train_df = pd.concat(dfs)
train_df = train_df.reset_index(drop=True)
len(train_df)


# In[ ]:


part = 16
for j in range((49-16)+1):
    if part+j != 17:
        meta = pd.read_csv(f'../input/dfdc-part-{part+j}/images/metadata{part+j}.csv')
    else:
        meta = pd.read_csv(f'../input/dfdc-part-{part+j}/images/metadata{part+j}.json', index_col=0)
    meta['path'] = ''
    print(part+j)
    del_idxs = []
    for i in range(len(meta)):
        if os.path.isdir(f'../input/dfdc-part-{part+j}/images/{meta.loc[i]["filename"][:-4]}'):
            if len(os.listdir(f'../input/dfdc-part-{part+j}/images/{meta.loc[i]["filename"][:-4]}')) < 5:
                del_idxs.append(i)
            else:
                meta.loc[i]['path'] = f'../input/dfdc-part-{part+j}/images/{meta.loc[i]["filename"][:-4]}'
        else:
            del_idxs.append(i)
    print(del_idxs)
    for idx in del_idxs:
        meta = meta.drop(idx)
    train_df = pd.concat([train_df,meta])
    train_df = train_df.reset_index(drop=True)
len(train_df)


# In[ ]:


train_df


# In[ ]:




