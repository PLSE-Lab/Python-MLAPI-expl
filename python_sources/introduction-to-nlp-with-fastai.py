#!/usr/bin/env python
# coding: utf-8

# # An introduction to Fastai for NLP Text Classification
# 
# This notebook will describe the basic steps to create a text classifier with the library Fastai. It is very simple and is based on the example/overview provided in the website of fastai. Some modifications have been included, but they are minor changes. I am not trying to get the best model, but with this guide you can improve the model performance applying some improvement (using some embeddings, optimizing the model, trying pretrained models,...) 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.text import * 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


base_path="../output"
text_columns=['text']
label_columns=['target']
BATCH_SIZE=128


# ## Loading the data

# In[ ]:


train= pd.read_csv('../input/nlp-getting-started/train.csv')
test= pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()


# In[ ]:


tweets = pd.concat([train[text_columns], test[text_columns]])
print(tweets.shape)


# ## Train a text language model
# Create a databunch for a text language model to get the data ready for training a language model. The text will be processed, tokenized and numericalized by a default processor, if you want to apply a customized tokenizer or vocab, you just need to create them.

# In[ ]:


data_lm = (TextList.from_df(tweets)
           #Inputs: all the text files in path
            .split_by_rand_pct(0.15)
           #We randomly split and keep 10% for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=BATCH_SIZE))
data_lm.save('tmp_lm')


# In[ ]:


data_lm.show_batch()


# Now we can create a language model based on the architecture AWD_LSTM (a detailed explanation can be found on the fastai website https://docs.fast.ai/text.models.html#AWD_LSTM):

# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)


# In[ ]:


print('Model Summary:')
print(learn.layer_groups)


# Lets train our language model. First, we call lr_find to analyze and find an optimal learning rate for our problem, then we fit or train the model for a few epochs. Finally we unfreeze the model and runs it for a few more epochs. 
# So we have a encoder trained and ready to be used for our classifier and it is recorded on disk.

# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, 1e-2)
learn.save('lm_fit_head')


# In[ ]:


learn.load('lm_fit_head')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)


# In[ ]:


learn.save_encoder('ft_enc')


# ## Building and training a text classifier

# Now that we have a language model trained, we can create a text classifier on it.

# In[ ]:


data_clas = (TextList.from_df(train, cols=text_columns, vocab=data_lm.vocab)
             .split_by_rand_pct(0.15)
             .label_from_df('target')
             .add_test(test[text_columns])
             .databunch(bs=BATCH_SIZE))

data_clas.save('tmp_clas')


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)


# Next step, to load the encoder previously trained (the language model).

# In[ ]:


learn.load_encoder('ft_enc')


# Now, the training cycle is repeated: lr_find, freeze except last layer,..., unfreeze the model and saving the final trained model.

# In[ ]:


learn.freeze_to(-1)
learn.summary()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, 1e-3)


# In[ ]:


learn.save('stage1')


# In[ ]:


learn.load('stage1')
learn.freeze_to(-2)
learn.fit_one_cycle(5, slice(5e-3/2., 5e-3))
learn.save('stage2')


# In[ ]:


learn.load('stage2')
learn.unfreeze()
learn.fit_one_cycle(5, slice(2e-3/100, 2e-3))


# In[ ]:


learn.export()
learn.save('final')


# ## Some analysis and interpretation of the results
# There are some functions to explore the behaviour of our model. For example we can explore the confusion matrix and show the top losses of our classifier.

# In[ ]:


from fastai.vision import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)


# In[ ]:


interp = TextClassificationInterpretation.from_learner(learn)
interp.show_top_losses(10)


# ## Predicting and creating a submission file
# Now we can predict on the test set and create the submission file requiered by the competition:

# In[ ]:


learn.predict(test.loc[0,'text'])


# In[ ]:


def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in learn.data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]


# In[ ]:


test_preds = get_preds_as_nparray(DatasetType.Test)


# In[ ]:


sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample_submission['target'] = np.argmax(test_preds, axis=1)
sample_submission.to_csv("predictions.csv", index=False, header=True)


# In[ ]:


sample_submission['target'].value_counts()


# In[ ]:




