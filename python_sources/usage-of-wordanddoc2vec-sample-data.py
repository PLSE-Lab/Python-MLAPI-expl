#!/usr/bin/env python
# coding: utf-8

# num features = 3  
# sample data

# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


from feature_eng.neg_smpl3 import (
    WordAndDoc2vec,
    MySparseMatrixSimilarity,
    Seq, Seq2, Dic4seq,
    get_sim
)


# In[ ]:


import os.path
import sys
import re
import itertools
import csv
import datetime
import pickle
import random
from collections import defaultdict, Counter
import gc

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import gensim
from sklearn.metrics import f1_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
import gensim
from keras.preprocessing.sequence import skipgrams
import tensorflow as tf


# In[ ]:


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, cmap=cmap, **kwargs)
def scatter(x, y, color, **kwargs):
    plt.scatter(x, y, marker='.')


# # Create Sample Data

# In[ ]:


NN_word = 2000
NN_sentence = 10000
NN_SEG = 7


# In[ ]:


product_list = [ee+1 for ee in range(NN_word)]
user_list = [ee+1 for ee in range(NN_sentence)]


# In[ ]:


a, _ = divmod(len(user_list), NN_SEG)
print(a)
cls_user = [int(user_id / (a+1)) for user_id in range(1, 1+len(user_list))]


# In[ ]:


a, _ = divmod(len(product_list), NN_SEG)
print(a)
cls_prod = [int(prod_id / (a+1)) for prod_id in range(1, 1+len(product_list))]


# In[ ]:


random.seed(0)

X_list = []

for ii in range(len(user_list)):
    cls = cls_user[ii]
    product_group = np.array(product_list)[np.array(cls_prod) == cls]
    nword = random.randint(5, 20)
    prods = random.sample(product_group.tolist(), nword)
    irow = np.zeros((1,NN_word))
    irow[0,np.array(prods)-1] = 1
    X_list.append(irow)

X = np.concatenate(X_list)
print(X.shape)
X


# In[ ]:


X_df = pd.DataFrame(X, dtype=int)
X_df.index = ['r'+ee.astype('str') for ee in (np.arange(X_df.shape[0])+1)]
X_df.columns = ['c'+ee.astype('str') for ee in np.arange(X_df.shape[1])+1]
print(X_df.shape)
X_df.head()


# In[ ]:


X_df.values.shape


# In[ ]:


plt.figure(figsize=(10, 10))
plt.imshow(X_df.values.T)


# # Define Original Class for This Methodology

# In[ ]:


from collections.abc import Sequence

class DocSeq(Sequence):
    '''
    doc_dic  : doc_name (unique)
    word_dic : index=0 must be place holder.
    '''
    def __init__(self, df):
        self.df = df
        self.cols = self.df.columns.values
        
        self.doc_dic = gensim.corpora.Dictionary([df.index.values.tolist()], prune_at=None)
        
        
        '''
        index=0 must be place holder.
        '''
        self.word_dic = gensim.corpora.Dictionary([['PL_DUMMY']], prune_at=None)
        
        self.word_dic.add_documents([list(self.cols)], prune_at=None)
    
    def __getitem__(self, idx):
        return self._get(idx)
    
    def _get(self, idx):
        try:
            ebid = self.doc_dic[idx]
        except KeyError:
            raise IndexError
        irow = self.df.loc[ebid]
        res = []
        for icol in self.cols:
            if irow[icol] == 1:
                res.append(icol)
        return res
    
    def __len__(self):
        return self.df.shape[0]


# In[ ]:


doc_seq = DocSeq(X_df)
len(doc_seq)


# # Create Model

# In[ ]:


ls -la ../input/wordanddoc2vec-sample-data


# In[ ]:


# %%time
# corpus_csr = gensim.matutils.corpus2csc(
#     (doc_seq.word_dic.doc2bow(ee) for ee in doc_seq),
#     num_terms=max(doc_seq.word_dic.keys())+1
# ).T
# corpus_csr.shape


# In[ ]:


# scipy.sparse.save_npz('tt01_corpus_csr', corpus_csr)
# doc_seq.word_dic.save('tt01_word_dic')
# doc_seq.doc_dic.save('tt01_doc_dic')


# In[ ]:


src_dir = '../input/wordanddoc2vec-sample-data'
corpus_csr = scipy.sparse.load_npz(os.path.join(src_dir, 'tt01_corpus_csr.npz'))
word_dic = gensim.corpora.dictionary.Dictionary.load(os.path.join(src_dir, 'tt01_word_dic'))
doc_dic = gensim.corpora.dictionary.Dictionary.load(os.path.join(src_dir, 'tt01_doc_dic'))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'wd2v = WordAndDoc2vec(corpus_csr, word_dic, doc_dic)\nwd2v')


# In[ ]:


num_features = 3
#wd2v.make_model(num_features=num_features)
wd2v.make_model(num_features=num_features, embeddings_val=0.1)


# In[ ]:


wgt_prod = wd2v.wgt_col
print(wgt_prod.shape)
df = pd.DataFrame(wgt_prod[:,:5])
sns.pairplot(df, markers='.')


# In[ ]:


wgt_user = wd2v.wgt_row
print(wgt_user.shape)
df = pd.DataFrame(wgt_user[:,:5])
sns.pairplot(df, markers='.')


# # Train

# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

def lr_schedule(epoch):
    lr0 = 0.02
    epoch1 = 16
    epoch2 = 16
    epoch3 = 16
    epoch4 = 16
    
    if epoch<epoch1:
        lr = lr0
    elif epoch<epoch1+epoch2:
        lr = lr0/2
    elif epoch<epoch1+epoch2+epoch3:
        lr = lr0/4
    elif epoch<epoch1+epoch2+epoch3+epoch4:
        lr = lr0/8
    else:
        lr = lr0/16
    
    if divmod(epoch,4)[1] == 3:
        lr *= (1/8)
    elif divmod(epoch,4)[1] == 2:
        lr *= (1/4)
    elif divmod(epoch,4)[1] == 1:
        lr *= (1/2)
    elif divmod(epoch,4)[1] == 0:
        pass
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [lr_scheduler]

hst = wd2v.train(epochs=8, verbose=2,
           use_multiprocessing=True, workers=4,
           callbacks=callbacks)
hst_history = hst.history


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["acc"], label="accuracy")
ax[2].set_title('learning rate')
ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
ax[0].legend()
ax[1].legend()
ax[2].legend()


# # Show Features (col side)

# In[ ]:


wgt_prod = wd2v.wgt_col[doc_seq.word_dic.doc2idx(['PL_DUMMY']+X_df.columns.tolist())]
print(wgt_prod.shape)
df = pd.DataFrame(wgt_prod[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, height=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)


# In[ ]:


df = pd.DataFrame(wgt_prod[:,:5])
df['cls'] = ['zz'] + ['c'+str(ii) for ii in cls_prod]
sns.pairplot(df, markers='o', hue='cls', hue_order=['c'+str(ee) for ee in range(7)]+['zz'], height=3.5, diag_kind='hist')


# # Show Features (row side)

# In[ ]:


wgt_user = wd2v.wgt_row[doc_seq.doc_dic.doc2idx(X_df.index.tolist())]
print(wgt_user.shape)
df = pd.DataFrame(wgt_user[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, height=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)


# In[ ]:


df = pd.DataFrame(wgt_user[:,:5])
df['cls'] = ['c'+str(ii) for ii in cls_user]
sns.pairplot(df, markers='o', hue='cls', hue_order=['c'+str(ee) for ee in range(7)], height=3.5, diag_kind='hist')


# # Similarity

# In[ ]:


sim = wd2v.sim
print(sim.num_features)
print(sim.sim_row)


# ## Get Document 'r1'

# In[ ]:


query = sim.sim_row.index[sim.row_dic.token2id['r1']]
query


# In[ ]:


sim.get_sim_byrow(query, num_best=20)


# In[ ]:




