#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the wonderful work done at:
# 
# https://www.kaggle.com/atikur/google-quest-fastai-v1
# 
# https://medium.com/@nikkisharma536/deep-learning-on-multi-label-text-classification-with-fastai-d5495d66ed88

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import spearmanr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.text import *


# In[ ]:


data=pd.read_csv('/kaggle/input/google-quest-challenge/train.csv',index_col='qa_id')
data.head(5)


# In[ ]:


print(data.shape)
print(list(data.columns))


# In[ ]:


data.isnull().sum()


# In[ ]:


val=data[:1000]
train=data[1000:]
print("Training data shape is:",train.shape)
print("Validation data shape is:",val.shape)


# In[ ]:


cols=['question_title', 'question_body', 'question_user_name', 'question_user_page', 'answer', 'answer_user_name', 'answer_user_page', 'url', 'category', 'host']
data_bunch = (TextList.from_df(data, cols=cols)
                .split_by_rand_pct(0.2)
                .label_for_lm()  
                .databunch(bs=48))
data_bunch.show_batch()


# In[ ]:


learn = language_model_learner(data_bunch,AWD_LSTM,pretrained_fnames=['/kaggle/input/wt103-fastai-nlp/lstm_fwd','/kaggle/input/wt103-fastai-nlp/itos_wt103'],pretrained=True,drop_mult=0.7)

# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot()

# Fit the model based on selected learning rate
learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))

# Save the encoder for use in classification
learn.save_encoder('fine_tuned_enc')


# In[ ]:


target_cols=['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']
test=pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
#test_data = TextList.from_df(test, cols=cols, vocab=data_bunch.vocab)
data_clas = TextClasDataBunch.from_df('.', train, val, test,
                  vocab=data_bunch.vocab,
                  text_cols=cols,
                  label_cols=target_cols,
                  bs=32)


# In[ ]:


class RhoMetric(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.corr = 0.
        self.count= 0.
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        self.count += torch.tensor(1.)
        y_pred=last_output.cpu().numpy()
        y_true=last_target.cpu().numpy()
        score=0.0
        if np.ndim(y_pred) == 2:
            for i in range(30):
                score += np.nan_to_num(spearmanr(y_pred[:, i], y_true[:,i]).correlation) / 30
        else:
            score = spearmanr(y_true, y_pred).correlation / 30
        
        self.corr=self.corr + torch.tensor(score)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.corr/self.count)


# In[ ]:


rhometric=RhoMetric()
learn_classifier = text_classifier_learner(data_clas, AWD_LSTM,pretrained=False,drop_mult=0.7,metrics=[rhometric])
fnames = ['/kaggle/input/wt103-fastai-nlp/lstm_fwd.pth','/kaggle/input/wt103-fastai-nlp/itos_wt103.pkl']
learn_classifier.load_pretrained(*fnames, strict=False)
# load the encoder saved  
learn_classifier.load_encoder('fine_tuned_enc')
learn_classifier.freeze()

# select the appropriate learning rate
learn_classifier.lr_find()

# we typically find the point where the slope is steepest
learn_classifier.recorder.plot()


# In[ ]:


# Fit the model based on selected learning rate
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
learn_classifier.fit_one_cycle(5, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.unfreeze()
learn_classifier.fit_one_cycle(20, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn_classifier.show_results()


# In[ ]:


preds, target = learn_classifier.get_preds(DatasetType.Valid, ordered=True)
labels = preds.numpy()


# In[ ]:


labels.shape


# In[ ]:


from scipy.stats import spearmanr
score = 0
for i in range(30):
    score += np.nan_to_num(spearmanr(val[target_cols].values[:, i], labels[:,i]).correlation) / 30
score


# In[ ]:


preds_test, target_test = learn_classifier.get_preds(DatasetType.Test, ordered=True)
labels_test = preds_test.numpy()


# In[ ]:


labels_test.shape


# In[ ]:


submission=pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
print(submission.shape)
submission.loc[:,target_cols]=labels_test
print(submission.shape)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

