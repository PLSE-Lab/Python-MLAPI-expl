#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.text import *
from fastai import *


# In[ ]:


path = Path('/kaggle/input/jigsaw-toxic-comment-classification-challenge/')
path.ls()
get_ipython().system('mkdir data')
get_ipython().system('pwd')
get_ipython().system('cp -a {path}/*.* ./data/')
get_ipython().system('ls data')

path = Path('/kaggle/working/data/')
path.ls()


# In[ ]:


df_train = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')
df = df_train.append(df_test)

df.head()


# In[ ]:


df['comment_text'][1]


# Create a TextLMDataBunch

# In[ ]:


bs = 64
data_lm = (TextList.from_df(df, path, cols='comment_text')
                .split_by_rand_pct(0.1)
                .label_for_lm()
                .databunch(bs=bs))


# In[ ]:


data_lm.show_batch()


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=15)


# In[ ]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('fit_head')


# In[ ]:


learn.load('fit_head');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(6, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.save('fine_tuned')


# In[ ]:


learn.load('fine_tuned');


# In[ ]:


TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2


# In[ ]:


print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[ ]:


# save the encoder for next step use
learn.save_encoder('fine_tuned_enc')


# Now create the classifier

# In[ ]:


test = pd.read_csv(path/"test.csv")
test_datalist = TextList.from_df(test, cols='comment_text')


# In[ ]:


data_cls = (TextList.from_csv(path, 'train.csv', cols='comment_text', vocab=data_lm.vocab)
                .split_by_rand_pct(valid_pct=0.1)
                .label_from_df(cols=['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate'], label_cls=MultiCategoryList, one_hot=True)
                .add_test(test_datalist)
                .databunch())
data_cls.save('data_clas.pkl')


# In[ ]:


data_clas = load_data(path, 'data_clas.pkl', bs=bs)


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')


# In[ ]:


# learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3)
# learn = language_model_learner(data, pretrained_model=URLs.WT103, drop_mult=0.3)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('first')


# In[ ]:


learn.load('first');


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn.save('second');
learn.load('second');


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn.save('third')
learn.load('third');


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# In[ ]:


learn.predict("I really loved that movie, it was awesome!")


# In[ ]:


learn.show_results()


# In[ ]:


preds, target = learn.get_preds(DatasetType.Test, ordered=True)
labels = preds.numpy()


# In[ ]:


labels


# In[ ]:


test_id = test['id']
label_cols = ['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']

submission = pd.DataFrame({'id': test_id})
submission = pd.concat([submission, pd.DataFrame(preds.numpy(), columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


# import os
# os.chdir(r'kaggle/working')

# from IPython.display import FileLink
# FileLink(r'df_name.csv')

