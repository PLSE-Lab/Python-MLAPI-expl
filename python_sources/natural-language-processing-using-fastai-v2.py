#!/usr/bin/env python
# coding: utf-8

# ## 1. Setup the environment
# 
# Let's first install the [fastaiv2](https://github.com/fastai/fastai2) library. The documentation of the library is available at [fastai v2 docs].(http://https://dev.fast.ai/)

# In[ ]:


get_ipython().system('pip install fastai2')


# import the module of natural language processing which is named in fastai `text`.

# In[ ]:


from fastai2.text.all import *


# ## 2. Load the data

# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')

# print the size of the training and test data
print(train.shape, test.shape)


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# ## 3. Language Model

# In[ ]:


total = pd.concat((train.drop('target', axis=1),test), axis=0)
total.shape


# In[ ]:


total.head()


# In[ ]:


dls_lm = TextDataLoaders.from_df(total, path='.', valid_pct=0.1, is_lm=True, text_col ='text')


# In[ ]:


dls_lm.show_batch()


# In[ ]:


learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(1, 1e-1)


# In[ ]:


learn.save('1epoch')


# In[ ]:


learn = learn.load('1epoch')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(6, 2e-3)


# In[ ]:


learn.save_encoder('finetuned')


# ## 4. Classification

# In[ ]:


dls_clas = TextDataLoaders.from_df(train, path='.', text_col='text', label_col='target', valid_pct=0.1, text_vocab=dls_lm.train.vocab)
dls_clas.show_batch(max_n=4)


# In[ ]:


len(dls_clas.train.vocab[0]), len(dls_lm.train.vocab)


# In[ ]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)


# In[ ]:


learn = learn.load_encoder('finetuned')


# In[ ]:


learn.fit_one_cycle(2, 2e-2)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(2, slice(1e-2/(2.6**4),1e-2))


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-3))


# ## 5. Preparing submission file

# In[ ]:


test_dl = dls_clas.test_dl(test)


# In[ ]:


preds, _, classes = learn.get_preds(dl=test_dl, with_decoded=True)


# In[ ]:


df = pd.DataFrame({
    'id': test_dl.get_idxs(),
    'target': classes
})


# In[ ]:


df.head()


# In[ ]:


df = df.sort_values(by='id')
df = df.reset_index(drop=True)
df.id = test.id.values
df.head()


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:




