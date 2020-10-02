#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.text import *
from fastai import *
import re


# In[ ]:


df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
df.shape


# In[ ]:


df.head()


# Each line contains one review along with the corresponding country, designation, points awarded, price, province, region, variety and winery.<br>
# First, we want to create a language model, which gains an appreciation for wine semantics. We will do this by creating a language model from the wine descriptions (using transfer learning as discussed below).

# ## Language Model Using ULMFiT
# 

# In[ ]:


data_lm = (TextList.from_df(df=df,path='.',cols='description') 
            .random_split_by_pct(0.1)
            .label_for_lm()           
            .databunch(bs=48))


# In[ ]:


data_lm.save('tmp_lm')


# NOTE: In contrast to image classification (whereby images being an array of pixel values can be used as inputs for a CNN), the descriptions are composed of words and therefore mathematical functions are useless. Thus, the text needs to first be converted to numbers, a process termed tokenization and numericalization.

# ## Tokenization
# This first step splits the raw sentences into words (or more correctly, tokens). Whilst this can be completed simply by splitting the sentences by spaces, we can achieve a more refined tokenization result by capturing:
# - punctuation
# - contractions of two different words e.g. isn't or don't
# - non-text e.g. HTML code
# 
# NOTE: special tokens are also implemented (tokens beginning with xx), to replace unknown tokens or to introduce different text fields e.g. capitilization.

# ## Numericalization
# After the tokens have been developed from the text, these are converted to a list of integers representing all the words i.e. our vocabulary.<br>
# NOTE: Only tokens that appear at list twice are retained, with a maximum vocabulary size of 60,000 (by default). The remaining tokens are replaced by the unknown token `UNK`.
# 
# The correspondance from ids tokens is stored in the `vocab` attribute of our datasets, in a dictionary called `itos` (for int to string).

# In[ ]:


data_lm.vocab.itos[:10]


# In[ ]:


data_lm.train_ds[0][0]


# In[ ]:


data_lm.train_ds[0][0].data[:10]


# In[ ]:


data_lm = TextLMDataBunch.load('.', 'tmp_lm', bs=48)


# In[ ]:


data_lm.show_batch()


# Here we will take advantage of transfer learning and the fastai provided model WT103. This model was pretrained on a cleaned subset of wikipeia called [wikitext-103](https://einstein.ai/research/blog/the-wikitext-long-term-dependency-language-modeling-dataset)). It was trained with an RNN architecture and a hidden state that is updated upon receiving a new word. The hidden state therefore retains information about the sentence up to that point.<br>
# This understanding of the text is utilized to build the classifier, however, we first need to fine-tine the pretrained model to our wine domain. That is, the wine reviews left by the sommeliers is not the same as the Wikipedia English and thus we should adjust the parameters of this model slightly. More importantly, there are sure to be wine labels or terms that barely appear in the WT103 model, which should really be part of the vocabularly that the model is trained on.

# In[ ]:


fnames=['../input/wt1031/itos_wt103.pkl',
       '../input/wt1031/lstm_wt103.pth']


# In[ ]:


def language_model_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
                  drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False, pretrained_model=None,
                  pretrained_fnames:OptStrTuple=None, **kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data`."
    dps = default_dropout['language'] * drop_mult
    vocab_size = len(data.vocab.itos)
    model = get_language_model(vocab_size, emb_sz, nh, nl, pad_token, input_p=dps[0], output_p=dps[1],
                weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn)
    learn = LanguageLearner(data, model, bptt, split_func=lm_split, **kwargs)
    if pretrained_model is not None:
        model_path = Path('../input/wt1031/')
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn


# In[ ]:


learn = language_model_learner(data_lm,path='.', pretrained_model=' ', drop_mult=0.3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=10)


# In[ ]:


learn.fit_one_cycle(1, 5e-2)


# In[ ]:


learn.save('fit_head')


# In[ ]:


learn.load('fit_head');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, 1e-3)


# In[ ]:


learn.save('fine_tuned')


# ## Test Sentence Completion

# In[ ]:


learn.load('fine_tuned');


# In[ ]:


TEXT = "i taste hints of"
N_WORDS = 40
N_SENTENCES = 2


# In[ ]:


print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# # Data Classifier

# In[ ]:


min_samples=10
lst=df.variety.value_counts()
wines=lst[lst>min_samples].keys()
subdf=df[df.variety.isin(wines)]


# In[ ]:


subdf.shape,df.shape


# In[ ]:


data_clas = (TextList.from_df(df=subdf,path='.',cols='description', vocab=data_lm.vocab)
             .random_split_by_pct(0.1)
             .label_from_df('variety')
             .databunch(bs=48))


# In[ ]:


data_clas.save('tmp_clas')


# In[ ]:


data_clas = TextClasDataBunch.load('.', 'tmp_clas', bs=48)


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')
learn.freeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=8)


# In[ ]:


learn.fit_one_cycle(1, 2e-2)
print ('')


# In[ ]:


learn.save('first')


# In[ ]:


learn.load('first');


# In[ ]:


for i in range(2,5):
    learn.freeze_to(-i)
    learn.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    learn.save('sub-'+str(i))
    print ('')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-5/(2.6**4),1e-3))
print ('')


# In[ ]:


learn.save('final')


# ## Further Training

# In[ ]:


learn.fit_one_cycle(5, slice(1e-5,1e-3))
print ('')


# In[ ]:


learn.save('final')


# ## Predictions On Validation Set

# In[ ]:


learn.show_results(rows=10)


# ## Predictions On Fake Reviews

# In[ ]:


learn.predict("tannins are well proportioned both grained and supple")[0]


# In[ ]:


learn.predict("a light wine with hints of bitterness and fruit")[0]


# In[ ]:


learn.predict("a wine full of flavor and color, mostly white")[0]

