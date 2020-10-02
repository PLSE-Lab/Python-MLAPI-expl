#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_train = pd.read_csv(f'../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')[['id', 'comment_text', 'toxic']]\ndf_train = df_train[(df_train['toxic'] < 0.2) | (df_train['toxic'] > 0.5)]\ndf_train = df_train.append(\n    pd.read_csv(f'../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')[['id', 'comment_text', 'toxic']]\n)\ndf_train['lang'] = 'en'\n\nfor lang in ['es', 'tr', 'it', 'ru', 'pt', 'fr']:\n    df_lang = pd.read_csv(f'../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-{lang}-cleaned.csv')[[\n        'id', 'comment_text', 'toxic', \n    ]]\n    df_lang['lang'] = lang\n    df_train = df_train.append(df_lang)\n\n\ndf_train = df_train[~df_train['comment_text'].isna()]\ndf_train = df_train.drop_duplicates(subset='comment_text')\ndf_train['toxic'] = df_train['toxic'].round().astype(np.int32)")


# In[ ]:


df_train['toxic'].hist(bins=20);


# In[ ]:


df_train['lang'].hist(bins=20);


# In[ ]:


df_train.to_csv('train_data.csv', index=False)


# In[ ]:




