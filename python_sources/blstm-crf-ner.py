#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install -q tensorflow_gpu>=2.0')
get_ipython().system('pip install ktrain')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.environ['DISABLE_V2_BEHAVIOR'] = '1'


# In[ ]:


import tensorflow as tf; print(tf.__version__)


# In[ ]:


import ktrain
from ktrain import text


# In[ ]:


DATAFILE = './../input/entity-annotated-corpus/ner_dataset.csv'
(trn, val, preproc) = text.entities_from_txt(DATAFILE,
                                             embeddings='word2vec',
                                             sentence_column='Sentence #',
                                             word_column='Word',
                                             tag_column='Tag', 
                                             data_format='gmb')


# In[ ]:


text.print_sequence_taggers()


# In[ ]:


model = text.sequence_tagger('bilstm-crf', preproc)


# In[ ]:


learner = ktrain.get_learner(model, train_data=trn, val_data=val)


# In[ ]:


# find good learning rate
#learner.lr_find()             # briefly simulate training to find good learning rate
#learner.lr_plot()             # visually identify best learning rate


# In[ ]:


learner.fit(1e-3, 1)


# In[ ]:


learner.validate(class_names=preproc.get_classes())


# In[ ]:


learner.view_top_losses(n=1)


# In[ ]:


predictor = ktrain.get_predictor(learner.model, preproc)


# In[ ]:


predictor.predict('As of 2019,Narendra modi has been prime minister of india.')

