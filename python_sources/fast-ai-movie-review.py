#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install fastai==1.0.42


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


from fastai import *
from fastai.text import * 
from fastai.gen_doc.nbdoc import *
from fastai.datasets import * 
from fastai.datasets import Config
from pathlib import Path
import pandas as pd


# In[ ]:


#import fastai; 
#fastai.show_install(1)


# In[ ]:


path = Path('../input/')


# In[ ]:


df = pd.read_csv(path/'train.tsv', sep="\t")
df.head()


# In[ ]:


df.shape


# In[ ]:


df['Sentiment'].value_counts()


# In[ ]:


df_test = pd.read_csv(path/'test.tsv', sep="\t")
df_test.head()


# In[ ]:


df = pd.DataFrame({'label':df.Sentiment, 'text':df.Phrase})
df.head()


# In[ ]:


test_df = pd.DataFrame({'PhraseId':df_test.PhraseId, 'text':df_test.Phrase})
test_df.head()


# In[ ]:


df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
df_test['text'] = test_df['text'].str.replace("[^a-zA-Z]", " ")


# In[ ]:


import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 


# In[ ]:


stop_words = stopwords.words('english')


# In[ ]:


# tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split()) 


# In[ ]:


# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 
                                    item not in stop_words]) 


# In[ ]:


# de-tokenization 
detokenized_doc = [] 


# In[ ]:


for i in range(len(df)):
    t =' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 


# In[ ]:


df['text'] = detokenized_doc
df.head()


# In[ ]:


# de-tokenization 
detokenized_doc = [] 


# In[ ]:


df_test.head()


# In[ ]:


# tokenization 
tokenized_doc = df_test['text'].apply(lambda x: x.split()) 


# In[ ]:


# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 
                                    item not in stop_words]) 


# In[ ]:


for i in range(len(df_test)):
    t =' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 


# In[ ]:


test_df.head()


# In[ ]:


test_df['text'] = detokenized_doc
test_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split 
# split data into training and validation set 
df_trn, df_val = train_test_split(df, stratify = df['label'],  test_size = 0.2, random_state = 12)
df_trn.shape, df_val.shape, test_df.shape


# In[ ]:


#data_lm = (TextList.from_csv(path, '/kaggle/working/train.csv', cols='text') 
#                   .random_split_by_pct(0.1)
#                   .label_for_lm()
#                   .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='text'))
#                   .databunch())


# In[ ]:


# Language model data 
data_lm = TextLMDataBunch.from_df(train_df = df_trn, 
                                  valid_df = df_val,
                                  test_df = test_df,
                                  text_cols=['text'],
                                  path = "") 


# In[ ]:


# Classifier model data 
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, 
                                      valid_df = df_val,
                                      test_df = test_df,
                                      vocab=data_lm.train_ds.vocab, 
                                      bs=32)


# In[ ]:


learn = language_model_learner(data_lm, pretrained=True,arch=AWD_LSTM, drop_mult=0.7) #pretrained_model=URLs.WT103
#learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)


# In[ ]:


#learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, model_dir="/tmp/model/", drop_mult=0.3)
learn.model


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


#learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
#learn.fit_one_cycle(1, 1e-1)
#learn.save('mini_train_lm')
#learn.save_encoder('mini_train_encoder')


# In[ ]:


# train the learner object with learning rate = 1e-2 
learn.fit_one_cycle(3, 1e-2)


# In[ ]:


#learn.unfreeze()
#learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))


# In[ ]:


learn.predict("This is a review about", n_words=10)


# In[ ]:


#learn.show_results()


# In[ ]:


learn.save_encoder('ft_enc')


# # Classifier

# https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb

# In[ ]:


learn = text_classifier_learner(data_clas, drop_mult=0.7) 
learn.load_encoder('ft_enc')


# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        #print("inputs",inputs.shape)
        #print("target",targets.shape)
        if self.logits:
            BCE_loss = F.cross_entropy_with_logits(inputs, targets, reduction='none')
            #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            #BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


# In[ ]:


learn.loss_func = FocalLoss()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, 1e-3)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))


# In[ ]:


# and plot the losses of the first cycle
learn.recorder.plot_losses()


# # Validation Set

# In[ ]:


# get predictions 
preds, targets = learn.get_preds(DatasetType.Valid) 
predictions = np.argmax(preds, axis = 1) 


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
#predictions = model.predict(X_test, batch_size=1000)

LABELS = ['negative','somewhat negative','neutral','somewhat positive','positive'] 

confusion_matrix = metrics.confusion_matrix(targets, predictions)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()


# In[ ]:


#??TextList.from_csv


# In[ ]:


#data_clas = (TextList.from_csv(path, '/kaggle/working/train.csv',cols='text', vocab=data_lm.vocab) #test='test'
#    .split_from_df(col='is_valid') #is_valid
#    .label_from_df(cols='target')
#    .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='text'))
#    .databunch(bs=42))


# In[ ]:


#type(data_clas.test_dl)


# In[ ]:


#data_clas.show_batch()


# In[ ]:


#??text_classifier_learner()


# In[ ]:


#data_clas.c


# In[ ]:


#len(data_clas.vocab.itos)


# In[ ]:


#learn = text_classifier_learner(data_clas, drop_mult=0.5) #metrics=[accuracy_thresh]
#learn.load_encoder('mini_train_encoder')
#learn.freeze()
#learn.model


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.crit = F.binary_cross_entropy
#learn.crit = F.binary_cross_entropy_with_logits


# In[ ]:


#learn.metrics = [accuracy, fbeta] #r2_score, top_k_accuracy


# In[ ]:


#learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
#learn.fit_one_cycle(1, slice(1e-3,1e-2))
#learn.save('mini_train_clas')


# https://docs.fast.ai/text.html

# In[ ]:


#learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[ ]:


#learn.freeze_to(-2)
#learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

#learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


#learn.freeze_to(-3)
#learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


#learn.unfreeze()
#learn.fit_one_cycle(15, slice(2e-3/100, 2e-3))

#learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# In[ ]:


# get predictions
#preds, targets = learn.get_preds()

#predictions = np.argmax(preds, axis = 1)
#pd.crosstab(predictions, targets)


# In[ ]:


#learn.show_results(rows=20)


# In[ ]:


# Language model data
#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")

# Classifier model data
#data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[ ]:


#type(learn.data.test_dl)


# # Predict Test

# In[ ]:


probs, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


probs.shape


# In[ ]:


probs[0]


# In[ ]:


preds = probs.argmax(dim=1)


# In[ ]:


#preds = np.argmax(probs, axis=1)


# In[ ]:


ids = df_test["PhraseId"].copy()


# In[ ]:


submission = pd.DataFrame(data={
    "PhraseId": ids,
    "Sentiment": preds
})
submission.to_csv("submission.csv", index=False)
submission.head(n=10)


# In[ ]:


#df.head()


# In[ ]:


#from sklearn.model_selection import train_test_split

# split data into training and validation set
#df_trn, df_val = train_test_split(df, stratify = df['target'], test_size = 0.4, random_state = 12)


# In[ ]:


# Language model data
#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "../input")


# In[ ]:


# Classifier model data
#data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[ ]:




