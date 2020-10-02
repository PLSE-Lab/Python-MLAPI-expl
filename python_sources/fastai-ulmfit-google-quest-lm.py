#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
from functools import partial
from pathlib import Path

from fastai.text import *
from fastai.callbacks import *
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

home = Path(".")
input_dir = Path("/kaggle/input/google-quest-challenge/")

get_ipython().system('mkdir -p ~/.fastai/models/')
get_ipython().system('cp -R ../input/fastai-wt103-models/* ~/.fastai/models/')


# This notebook is to show how one can fine tune the language model for transfer learning and pre-process data.
# 
# The whole workflow:
# 1. Fine tune fastai wt103 lm on the competition text
# 2. Train a classification model using encodings from the previous step
# 
# The language model uses public test set, which is onle 13% of the whole test set. To improve results, you want to train LM on the private test set also. In other words, include the LM fine tuning in the submission notebook.

# In[ ]:


BS = 256


# In[ ]:


pd.read_csv(input_dir/"sample_submission.csv").head(5)


# In[ ]:


raw_test = pd.read_csv(input_dir/"test.csv"); raw_test.tail(3)


# In[ ]:


raw_train = pd.read_csv(input_dir/"train.csv"); raw_train.tail(3)


# # Preprocess

# Using all the text data to fine tune LM

# In[ ]:


lm_df = raw_train.append(raw_test, ignore_index=True, sort=False)


# In[ ]:


np.random.seed(42)


# In[ ]:


lm_df = lm_df.iloc[np.random.permutation(len(raw_train))]
cut = int(0.2 * len(lm_df)) + 1
train_lm_df, valid_lm_df = lm_df[cut:], lm_df[:cut]


# In[ ]:


data_lm = TextLMDataBunch.from_df(home, train_lm_df, valid_lm_df,
                                  text_cols=["question_title", "question_body", "answer"],
                                  mark_fields=True,
                                  bs=BS)


# In[ ]:


data_lm.show_batch()


# In[ ]:


data_lm.save('./data_lm_export.pkl')


# In[ ]:


labels = raw_train.columns[(raw_train.columns.str.startswith("question_")) |
                           (raw_train.columns.str.startswith("answer_"))].to_list()
labels = list(filter(lambda x: x not in ['question_title',
                                         'question_body',
                                         'question_user_name',
                                         'question_user_page',
                                         'answer_user_name',
                                         'answer_user_page',], labels))
assert len(labels) == 30


# In[ ]:


data_clas = TextClasDataBunch.from_csv(home, input_dir/"train.csv", test=input_dir/"test.csv",
                                       vocab=data_lm.train_ds.vocab, bs=BS,
                                       text_cols=["question_title", "question_body", "answer"],
                                       mark_fields=True,
                                       label_cols=labels)


# In[ ]:


data_clas.show_batch(reverse=True)


# In[ ]:


data_clas.save('./data_clas.pkl')


# # Fine tune LM

# In[ ]:


data_lm = load_data(home, 'data_lm_export.pkl', bs=BS)


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5,
                               metrics=[accuracy, Perplexity()],
                               callback_fns=[partial(EarlyStoppingCallback, monitor="perplexity", mode="min", patience=5),
                                             partial(SaveModelCallback, monitor="perplexity", mode="min", name="best_model")])
learn = learn.to_fp16()


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=5)


# In[ ]:


lr = 5e-02
moms = (0.8, 0.7)
wd=0.1


# In[ ]:


learn.fit_one_cycle(5, slice(lr), moms=moms, wd=wd)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(30, slice(lr/2), moms=moms, wd=wd,
                    callbacks=[SaveModelCallback(learn, monitor="perplexity", name="best_model"),
                               ReduceLROnPlateauCallback(learn, monitor="perplexity", patience=5,
                                                         min_delta=0.1, min_lr=1e-6)])


# In[ ]:


learn.save_encoder('ft_enc')


# In[ ]:


learn.predict("As a non-mathematician, I am somewhat", n_words=10)

