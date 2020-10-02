#!/usr/bin/env python
# coding: utf-8

# translated train1 + train2, updated
# 
# 6 folds with L1 error threshold 0.36

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

import os, gc

from tqdm.notebook import tqdm
from transformers import TFAutoModel, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

SEED = 6  # original seed from input datasets
N_PSEUDO = 2  # N copies of test pseudo labels
er_threshold = 0.36  # max L1 error threshold between blended predictions and ground-truth labels

SUB_SAMPLE = 0.74


# In[ ]:


valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
y_valid = valid.toxic = valid.toxic.astype(np.float32)
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids']).astype(np.int32)

MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
x_test  = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)

del test
del valid
gc.collect()


# In[ ]:


import glob
ds1 = sorted(glob.glob('../input/jigsaw20-xlm-rtt42-translations-ds*/*.parquet.gzip'))
ds2 = sorted(glob.glob('../input/jigsaw20-xlm-rtt42-translations2-ds*/*.parquet.gzip'))
assert len(ds1) == len(ds2)
cols = ['toxic', 'label', 'token']

langs = set()
for i, (t1, t2) in enumerate(zip(ds1, ds2)):
    lang = t1[-15:-13]
    assert lang == t2[-15:-13]
    langs.add(lang)
    print(i, lang, t1, t2, langs)
    t1 = pd.read_parquet(t1)
    t2 = pd.read_parquet(t2)
    print(t1.shape[0] + t2.shape[0])

    folds = sorted(t1.fold.unique())
    assert t2.fold.isna().sum() == 0,  t2.fold.value_counts()

    for fold in folds:
        print(fold, t1.loc[t1.fold == fold].shape, t2.loc[t2.fold == fold].shape)
        f = t1.loc[t1.fold == fold, cols].append(t2.loc[t2.fold == fold, cols], ignore_index=True)
        print(f.shape, 'mean:', f.toxic.mean(), 'ratio:', (f.toxic > 0.5).mean())

        # sub-sample
        f = f.sample(frac=SUB_SAMPLE, random_state=SEED, weights=0.01+f.toxic)
        print(f.shape, 'mean:', f.toxic.mean(), 'ratio:', (f.toxic > 0.5).mean())
        f.to_parquet(f'fold{fold}_{lang}.parquet.gzip', compression='gzip')


# In[ ]:


del t1
del t2
del f
gc.collect()
langs


# In[ ]:


# pseudo labels
pseudo = pd.read_csv("/kaggle/input/jigsaw20-ensemble06-14-lb9491/submission.csv")
pseudo.toxic -= pseudo.toxic.min()
pseudo.toxic /= pseudo.toxic.max()
pseudo['token'] = [x for x in x_test]
pseudo


# In[ ]:


for fold in folds:
    df = pd.concat([pd.read_parquet(f'fold{fold}_{lang}.parquet.gzip') for lang in langs],
                   ignore_index=True)
    print(fold, df.shape)
    df['l1er'] = abs(df.toxic - df.label)
    # use er_threshold/2 as toxic is already a 50% blend with the GT labels
    ax = df.toxic.hist(bins=100, log=True, alpha=0.6)
    df = df.loc[df.l1er < er_threshold/2, ['toxic', 'token']]
    ax = df.toxic.hist(bins=100, log=True, alpha=0.4, ax=ax)
    print(fold, df.shape)

    for n in range(N_PSEUDO):
        df = df.append(pseudo[['toxic', 'token']], ignore_index=True)
    print(fold, df.shape)

    ax = pseudo.toxic.hist(bins=100, log=True, alpha=0.4, ax=ax)
    ax = df.toxic.hist(bins=100, log=True, alpha=0.3, ax=ax)
    plt.legend(['all', 'cut', 'pseudo', 'final'])
    plt.savefig(f'fold{fold}.png')
    plt.show()
    print(df.shape, 'mean:', df.toxic.mean(), 'ratio:', (df.toxic > 0.5).mean())

    # shuffle and save
    df = df.sample(frac=1, random_state=SEED)
    np.savez_compressed(f'jigsaw20_ds{len(df)}tt{SEED}_fold{fold}.npz',
                        np.array(df.token.tolist()), x_valid, x_test,
                        df.toxic.values, y_valid)
    del df
    gc.collect()
get_ipython().system('ls -sh *.npz')


# In[ ]:


get_ipython().system('rm *.parquet.gzip')

