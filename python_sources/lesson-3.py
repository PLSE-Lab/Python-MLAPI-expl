#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai.text import *

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
#Remember to use num_workers=0 when creating the DataBunch.


# In[4]:


path = Path('../input/')
path.ls()


# In[5]:


df = pd.read_csv(path/'train-balanced-sarcasm.csv')
df.head()


# In[6]:


df['comment'][1]


# In[7]:


#data = TextDataBunch.from_csv(path, 'train-balanced-sarcasm.csv', num_workers=0)


# Creating our data failed because our csv had NaN values in our comments column.  We will have to sanitize our dataframe first.  df.dropna() will drop all rows with NaNs (df.dropna(1) will drop all columns.  By default, the parameter value is 0 for rows).

# In[8]:


badinputs = df.loc[lambda x: x['comment'].isna()]
badinputs.head()


# In[9]:


df = df.dropna()
df['comment'][56267:56270]


# In[10]:


random_seed(123,True)
rand_df = df.assign(is_valid = np.random.choice(a=[True,False],size=len(df),p=[0.2,0.8]))
rand_df.head()


# In[11]:


random_seed(1006,True)
bs=48
data_lm = (TextList.from_df(rand_df, path, cols="comment")
                .split_from_df(col='is_valid')
                .label_for_lm()
                .databunch(bs=bs))


# In[12]:


data_lm.save('../working/data_lm.pkl')


# In[13]:


data_lm = load_data('../working','data_lm.pkl',bs=bs)
data_lm.show_batch()


# In[14]:


random_seed(100,True)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=15)


# In[15]:


random_seed(111, True)
learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))


# In[16]:


learn.save('fit_head')


# In[17]:


learn.load('fit_head');


# In[18]:


learn.unfreeze()


# In[19]:


random_seed(444,True)
learn.fit_one_cycle(10,2e-2,moms=(0.8,0.7))


# In[20]:


learn.save('fine_tuned')


# In[21]:


learn.load('fine_tuned');


# In[22]:


learn.save_encoder('fine_tuned_enc')


# In[23]:


data_clas = (TextList.from_df(rand_df, vocab=data_lm.vocab, cols="comment")
                .split_from_df(col='is_valid')
                .label_from_df(cols='label')
                .databunch(bs=bs))
data_clas.save('../working/data_clas.pkl')


# In[24]:


data_clas = load_data('../working','data_clas.pkl',bs=bs)
data_clas.show_batch()


# In[25]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')


# In[26]:


learn.lr_find()
learn.recorder.plot()


# In[27]:


random_seed(678,True)
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[28]:


learn.save('first')


# In[29]:


learn.load('first');


# In[30]:


random_seed(777,True)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[31]:


learn.save('second')


# In[32]:


learn.load('second');


# In[33]:


random_seed(999,True)
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(2e-3/(2.6**4),2e-3), moms=(0.8,0.7))


# In[34]:


learn.save('third')


# In[35]:


learn.load('third');


# In[80]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# In[81]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn,preds,y,losses)
interp.plot_confusion_matrix()


# In[95]:


losses,idxs = interp.top_losses()
len(data_clas.valid_ds)==len(losses)==len(idxs)
idxs[:10]


# In[116]:


for i in range(10):
  print(df['comment'][idxs[i]],df['label'][idxs[i]],losses[i])


# In[86]:


# Sarcastic
learn.predict("What could possibly go wrong?")


# In[85]:


# Sincere
learn.predict("I think that is a really good idea.")


# In[83]:


# Sarcastic
learn.predict("Obviously this is all your fault.")


# In[87]:


# Sincere
learn.predict("Honestly, this is your fault.")


# In[89]:


# ???
learn.predict("Good job, learner!")

