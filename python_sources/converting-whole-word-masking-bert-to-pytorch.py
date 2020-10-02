#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')
get_ipython().system('cp -r ../input/jigsaw-pytorch-pretrained-bert/repository/huggingface-pytorch-pretrained-BERT-3fc63f1/ ./')
get_ipython().system('pip install ./huggingface-pytorch-pretrained-BERT-3fc63f1/.')


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip')
get_ipython().system('unzip wwm_uncased_L-24_H-1024_A-16.zip')
get_ipython().system('rm wwm_uncased_L-24_H-1024_A-16.zip')


# In[ ]:


get_ipython().system('mkdir bert_large_uncased_whole_word_masking')
get_ipython().system('cp ./wwm_uncased_L-24_H-1024_A-16/bert_config.json ./bert_large_uncased_whole_word_masking/')

get_ipython().system('pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch     wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt     wwm_uncased_L-24_H-1024_A-16/bert_config.json     bert_large_uncased_whole_word_masking/pytorch_model.bin')

get_ipython().system('rm -rf wwm_uncased_L-24_H-1024_A-16')
get_ipython().system('tar czf bert_large_uncased_whole_word_masking.tar.gz bert_large_uncased_whole_word_masking')
get_ipython().system('rm -rf bert_large_uncased_whole_word_masking')


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip')
get_ipython().system('unzip wwm_cased_L-24_H-1024_A-16.zip')
get_ipython().system('rm wwm_cased_L-24_H-1024_A-16.zip')


# In[ ]:


get_ipython().system('mkdir bert_large_cased_whole_word_masking')
get_ipython().system('cp ./wwm_cased_L-24_H-1024_A-16/bert_config.json ./bert_large_cased_whole_word_masking/')

get_ipython().system('pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch     wwm_cased_L-24_H-1024_A-16/bert_model.ckpt     wwm_cased_L-24_H-1024_A-16/bert_config.json     bert_large_cased_whole_word_masking/pytorch_model.bin')

get_ipython().system('rm -rf wwm_cased_L-24_H-1024_A-16')
get_ipython().system('tar czf bert_large_cased_whole_word_masking.tar.gz bert_large_cased_whole_word_masking')
get_ipython().system('rm -rf bert_large_cased_whole_word_masking')


# In[ ]:


get_ipython().system('rm -rf ./huggingface-pytorch-pretrained-BERT-3fc63f1')

