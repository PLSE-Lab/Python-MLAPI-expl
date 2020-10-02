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


# In[ ]:


from fastai.text import *


# In[ ]:


train = pd.read_csv("../input/train.tsv", sep='\t')
test = pd.read_csv("../input/test.tsv", sep='\t')
#train = train[0:1000]
train['Sentiment'] = train['Sentiment'].apply(str)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test_id = test['PhraseId']


# In[ ]:


test['Phrase'][0]


# In[ ]:


train['Sentiment'].unique()


# In[ ]:


data = (TextList.from_df(train, cols='Phrase')
                .split_by_rand_pct(0.2)
                .label_for_lm()  
                .databunch(bs=48))
data.show_batch()


# In[ ]:


learn = language_model_learner(data,AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-2, moms=(0.8,0.7))


# In[ ]:


# Tune a little more
learn.unfreeze()
learn.fit_one_cycle(5, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# In[ ]:


test_datalist = TextList.from_df(test, cols='Phrase', vocab=data.vocab)

data_clas = (TextList.from_df(train, cols='Phrase', vocab=data.vocab)
             .split_by_rand_pct(0.2)
             .label_from_df(cols= 'Sentiment')
             .add_test(test_datalist)
             .databunch(bs=32))

data_clas.show_batch()


# In[ ]:


learn_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn_classifier.load_encoder('fine_tuned_enc')
learn_classifier.freeze()


# In[ ]:


learn_classifier.lr_find()


# In[ ]:


learn_classifier.recorder.plot()


# In[ ]:


learn_classifier.fit_one_cycle(5, 2e-2, moms=(0.8,0.7))


# In[ ]:


learn_classifier.freeze_to(-2)
learn_classifier.fit_one_cycle(5, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn_classifier.freeze_to(-3)
learn_classifier.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn_classifier.show_results()


# In[ ]:


preds, target = learn_classifier.get_preds(DatasetType.Test, ordered=True)
labels = np.argmax(preds, axis =1)


# In[ ]:


submission = pd.DataFrame({'PhraseId': test_id, 'Sentiment': labels})
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




