#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ###  ULMFit
# 
# This kernel is the initial version of using universal language model for text classification, which is currently the state of the art method for text classification. To put this thing in this shape, it took a lot of effort in configuration, Finally I figured out the way to use it kaggle kernel.
# 
# Majority of the code is taken from FastAI course. I extend my Sincere gratitude  to Jeremey Howard and Sebastian Ruder who are contributing  amazing stuff in NLP.
# 
# 
# **Understanding ULMFit**
# 
# Transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100x more data.
# 
# At first we train a language model with the given text and use it for classification....
# 
# ![ULM FIT](http://nlp.fast.ai/images/ulmfit_approach.png)
# 
# **Tasks done in this notebook**
# 
# 1) Training a language model.
# 
# 2) Using Language model encoder for classification
# 
# **TODO**
# 
# 1) Predictions on test data.
# 
# 2) Blending both forward and backward trained language model for classification.
# 

# In[ ]:


get_ipython().system('pip uninstall -y torch')


# In[ ]:


get_ipython().system('conda install -y pytorch=0.3.1.0 cuda80 -c soumith')


# In[ ]:


import torch
torch.__version__


# In[ ]:


get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip uninstall -y torchtext')
get_ipython().system('pip install torchtext==0.2.3')

from fastai.text import *
import html


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')


# In[ ]:


trn_df,val_df = sklearn.model_selection.train_test_split(train_df, test_size=0.1)


# In[ ]:


trn_texts = trn_df['question_text']
val_texts = val_df['question_text']

trn_labels = trn_df['target']
val_labels = val_df['target']


# In[ ]:


trn_labels.value_counts()


# In[ ]:


col_names = ['labels','text']


# In[ ]:


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag


# #### Speeding up the tokenization:
# 
# 
# 
# For Speeding up the tokenization process we are breaking frame into chunks

# In[ ]:


df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)


# In[ ]:


get_ipython().system('mkdir ../updated')


# In[ ]:


get_ipython().system('ls -lart ../')


# In[ ]:


df_trn.to_csv('../updated/train.csv', header=False, index=False)
df_val.to_csv('../updated/test.csv', header=False, index=False)


# In[ ]:


chunksize=24000


# In[ ]:


re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


# In[ ]:


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


# In[ ]:


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


# In[ ]:


df_trn = pd.read_csv('../updated/train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv('../updated/test.csv', header=None, chunksize=chunksize)


# In[ ]:


tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)


# In[ ]:


np.save('../updated/tok_trn.npy', tok_trn)
np.save('../updated/tok_val.npy', tok_val)


# In[ ]:


tok_trn = np.load('../updated/tok_trn.npy')
tok_val = np.load('../updated/tok_val.npy')


# In[ ]:


freq = Counter(p for o in tok_trn for p in o)


# In[ ]:


max_vocab = 60000
min_freq = 2


# In[ ]:


itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')


# In[ ]:


stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[ ]:


trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])


# In[ ]:


np.save('../updated/trn_ids.npy', trn_lm)
np.save('../updated/val_ids.npy', val_lm)
pickle.dump(itos, open('../updated/itos.pkl', 'wb'))


# In[ ]:


trn_lm = np.load('../updated/trn_ids.npy')
val_lm = np.load('../updated/val_ids.npy')
itos = pickle.load(open('../updated/itos.pkl', 'rb'))


# In[ ]:


vs=len(itos)
vs,len(trn_lm)


# #### Getting the wikipedia language model

# In[ ]:


get_ipython().system('wget -P ../updated/ http://files.fast.ai/models/wt103/fwd_wt103.h5 ')


# In[ ]:


wgts = torch.load('../updated/fwd_wt103.h5', map_location=lambda storage, loc: storage)


# In[ ]:


enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)


# In[ ]:


PATH=Path('../updated')


# In[ ]:


get_ipython().system('wget -P ../updated/ http://files.fast.ai/models/wt103/itos_wt103.pkl ')
itos2 = pickle.load((PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# In[ ]:


em_sz,nh,nl = 400,1150,3


# In[ ]:


new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


# #### Set the map with our inputs

# In[ ]:


wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


# In[ ]:


import gc
gc.enable()


# In[ ]:


gc.collect()


# ### Language model in action.....

# In[ ]:


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# In[ ]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


# In[ ]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


# In[ ]:


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[ ]:


learner.model.load_state_dict(wgts)


# In[ ]:


lr=1e-2
lrs = lr


# In[ ]:


learner


# In[ ]:


### Taking too long to train as per the kernel requirement I couldn't commit it

#learner.fit(lrs, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# **Stay tune to be continued**

# In[ ]:




