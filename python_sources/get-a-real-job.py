#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from fastai.text import *
from fastai import *
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


jobs  = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
jobs.head()


# In[ ]:


jobs.fillna(value='NA',inplace=True)


# In[ ]:


jobs.fraudulent.value_counts()


# This could possibly be because of a huge class imbalance, seeing that there are far less number of fradulent cases than the real ones.

# In[ ]:


sns.set_style('darkgrid')
plt.figure(1,figsize=(20,8))
sns.countplot(hue=jobs.fraudulent,x=jobs.employment_type)
plt.title('Fraudulence Distribution based on Type of Employment Opportunity')
plt.xlabel('Employment Type')
plt.ylabel('No. of Jobs')


# Clearly, there is a class imbalance. Therefore, simple accuracy is not a good metric for prediction. I will hence be using Cohen's Kappa. But that is for later, first I need to make a language model.

# In[ ]:


path = Path('/kaggle/working/')
data_lm = (TextList.from_df(jobs,path,cols=['company_profile','description','requirements','benefits'])
                  .split_by_rand_pct(0.2)
                  .label_for_lm()
                  .databunch(bs=128))


# In[ ]:


data_lm.show_batch(rows=6)


# In[ ]:


learn = language_model_learner(data_lm,AWD_LSTM,metrics=[accuracy,Perplexity()],model_dir='/kaggle/working/',drop_mult=0.3).to_fp16()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# `to_fp16()` uses [Mixed Precision Training](https://arxiv.org/abs/1710.03740) that decreases the training time. Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. By keeping certain parts of the model in the 32-bit types for numeric stability, the model will have a lower step time and train equally as well in terms of the evaluation metrics such as accuracy.

# In[ ]:


import gc
gc.collect()


# In[ ]:


learn.fit_one_cycle(5, 1e-01, moms = (0.8,0.7))


# The language model can predict the next word correctly one out of two times, which is quite good.

# In[ ]:


learn.save_encoder('lm')


# In[ ]:


learn = None
gc.collect()


# In[ ]:


jobs0 = jobs[jobs['fraudulent']==0][:1400].copy()
jobs1 = jobs[jobs['fraudulent']==1].copy()
train = pd.concat([jobs0,jobs1])


# In[ ]:


label_cols = ['department','employment_type','required_experience','industry','function','required_education','title','company_profile','description','requirements','benefits']


# In[ ]:


data_cls = (TextList.from_df(train,path,cols=label_cols,vocab=data_lm.vocab)
                    .split_by_rand_pct(0.2,seed=64)
                    .label_from_df(cols='fraudulent')
                    .databunch(bs=128))


# In[ ]:


data_cls.show_batch(rows=6)


# In[ ]:


clf = None
gc.collect()


# In[ ]:


f_score = FBeta()
f_score.average = 'macro'
kappa = KappaScore()
kappa.weights = "quadratic"

clf = text_classifier_learner(data_cls,AWD_LSTM,metrics=[accuracy, f_score, kappa],drop_mult=0.3).to_fp16()
clf.load_encoder('/kaggle/working/lm');


# In[ ]:


gc.collect()


# In[ ]:


clf.lr_find()
clf.recorder.plot()


# In[ ]:


clf.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))


# In the last epoch, f1 score is 0.925419 and Cohen's Kappa is 0.855083. Nice. Confusion matrix can help further tune the model.

# In[ ]:


interp = TextClassificationInterpretation.from_learner(clf)
interp.plot_confusion_matrix()


# Pretty good if you ask me! The model correctly predicts the fraudulence of 278 out of 290 postings which are not fraud, and 145 out of 163 postings which are fraud.

# In[ ]:




