#!/usr/bin/env python
# coding: utf-8

# ## Overview of ULMFiT

# ULMFiT is essentially a method to enable transfer learning for any NLP task and achieve great results. All this, without having to train models from scratch. 
# 
# ULMFiT achieves state-of-the-art result using novel techniques like:
# 
# Discriminative fine-tuning
# Slanted triangular learning rates, and
# Gradual unfreezing
# 
# Link to paper -
# https://arxiv.org/abs/1801.06146

# This method involves fine-tuning a pre-trained language model (LM), trained on the Wikitext 1 Million token dataset to your own custom datset same as we do transfer learning in image processing.

# ## Import Libraries

# In[ ]:


from fastai import *
from fastai.text import *
from fastai.callbacks import *
from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

import torch
print("Cuda available" if torch.cuda.is_available() is True else "CPU")
print("PyTorch version: ", torch.__version__)


# ## Data Preprocessing
# 
# Though Jeremy discourages cleaning of input data (removing stop words, part of speech etc), we can do using NLTK libraries if the need arises. Although I have not seen much impact of data cleaning on the overall model accuracy after training.

# For simplicity I am only taking 10K rows, but eventually we will train on all data

# In[ ]:


train_df = pd.read_csv('../input/train.csv', nrows=10000)
test_df = pd.read_csv('../input/test.csv', nrows=2000)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


len(train_df), len(test_df)


# New labels True, False. True if target value >0.4

# In[ ]:


train_df['label'] = (train_df['target'] >= 0.4)


# In[ ]:


train_df['label'].value_counts()


# In[ ]:


train_df[['target','comment_text','label']].sample(10)


# Splitting data 20% for validation at each epoch

# In[ ]:


train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'])


# In[ ]:


len(train_df), len(val_df)


# ### Language Model
# 
#     

# We create a language model based on the given vocabulary and the architecture AWD_LSTM. We only take into account the comment_text input. To quickly gain understanding of the dataset and the potential performance of our model, we will only take a subset of  sentences from our original dataset df_train. Eventually, we will train our model on the entire dataset.

# In[ ]:


data_lm = TextLMDataBunch.from_df(
    path='',
    train_df=train_df,
    valid_df=val_df,
    test_df=test_df,
    text_cols=['comment_text'],
    label_cols=['label'],
    #label_cols=['target_better'],
    #classes=['target_better'],
    min_freq=3
)


# In[ ]:


data_lm.show_batch()


# Visualizing tokenization
# ULMFit doesn't feed the different texts separately but concatenate them all together in a big array. To create the batches, it splits this array into bs chunks of continuous texts.

# In[ ]:


x,y = next(iter(data_lm.train_dl))
example = x[:15,:15].cpu()
texts = pd.DataFrame([data_lm.train_ds.vocab.textify(l).split(' ') for l in example])
texts


# ### Fine Tuning Language Model

# In[ ]:


learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.6)


# 1cycle policy
# 
# We will use the one cycle policy proposed by Leslie Smith, arXiv, April 2018. The policy brings more disciplined approach for selecting hyperparameters such as learning rate and weight decay. This can potentially save us a lot of time from training with suboptimal hyperparameters. In addititon, Fastai library has implemented a training function for one cycle policy that we can use with only a few lines of code.
# 

# In[ ]:


learn.lr_find(start_lr=1e-6, end_lr=1e2)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(cyc_len=3, max_lr=1e-01)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(cyc_len=5, max_lr=1e-3, moms=(0.8, 0.7))


# In[ ]:


learn.save_encoder('ft_enc')


# In[ ]:


learn.predict("Thank",n_words=5)


# ## Training Text Classifier

# ### Classifier data
# since the text don't all have the same length, LM can't easily collate them together in batches. To help with this fastai uses two different techniques:
# 
# * padding: each text is padded with the PAD token to get all the ones picked to the same size. 
# * sorting the texts (ish): to avoid having together a very long text with a very short one (which would then have a lot of PAD tokens), it regroups the texts by order of length.
# * 
# This is all done behind the scenes by fastai library

# In[ ]:


data_class = TextClasDataBunch.from_df(
    path='',
    train_df=train_df,
    valid_df=val_df,
    test_df=test_df,
    text_cols=['comment_text'],
    label_cols=['label'],  
    min_freq=3,
    vocab=data_lm.train_ds.vocab,
    #label_delim=' '
)


# In[ ]:


iter_dl = iter(data_class.train_dl)
_ = next(iter_dl)
x,y = next(iter_dl)
x[-10:,:10]


# data is padded with token 1 at beginning

# In[ ]:


learn = text_classifier_learner(data_class, arch=AWD_LSTM, drop_mult=0.6)
learn.load_encoder('ft_enc')
learn.freeze()


# Apart from AWD_LSTM, I will try with other available architectures like TransformerXL
# 
# Link to AWD_LSTM paper
# https://arxiv.org/pdf/1708.02182

# In[ ]:


learn.lr_find(start_lr=1e-8, end_lr=1e2)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(cyc_len=2, max_lr=1e-2)
#learn.fit_one_cycle(1, 1e-2)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(3, slice(1e-4,1e-2))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:



learn.fit_one_cycle(10, slice(1e-5,1e-3),callbacks=[SaveModelCallback(learn, name="best_lm")])


# In[ ]:


learn.recorder.plot_losses()


# Graph clearly shows high variance and overfitting. One reason could be data imbalance

# In[ ]:


preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)


# In[ ]:



p = preds[0][:,1]
test_df['prediction'] = p
test_df.sort_values('prediction', inplace=True)
test_df.reset_index(drop=True, inplace=True)
ii = 100
print(test_df['comment_text'][ii])
print(test_df['prediction'][ii])


# In[ ]:


learn.show_results()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# Confusion matrix can help us understand the ratio of false negatives and positives and it's a fast way looking at our model's performance. This is a simple table that shows the counts in a way of actual label vs. predicted label. 

# As we can see from the figure that model is skewed in favor of the high freaquency class due to data imbalance. 

# In[ ]:


preds = learn.get_preds(ds_type=DatasetType.Valid, ordered=True)
p = preds[0][:,1]


# In[ ]:


preds,y, loss = learn.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))


# In[ ]:


from sklearn.metrics import roc_curve, auc
# probs from log preds
probs = np.exp(preds[:,1])
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

# Compute ROC area
roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[ ]:


interp2 = TextClassificationInterpretation.from_learner(learn) 
interp2.show_top_losses(10)


# We can see the contribution of each token to the classifier model

# In[ ]:


print(interp2.show_intrinsic_attention("Thank you, for your comments"))

