#!/usr/bin/env python
# coding: utf-8

# # Stage 1 Models

# ## Hearkilla

# ### Roberta large

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/roberta-large-code/infer.py')


# ### Xlnet

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/xlnet-base/infer.py')


# ### Roberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/roberta-base/infer.py')


# ### Distilroberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/distil-roberta/infer.py')


# ## Theo

# ### Distilbert

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_distilbert.py')


# ### Bert base

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_bert_base.py')


# ### Bert large wwm

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_bert_wwm.py')


# ### Albert large

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_albert.py')


# ## Anton

# ### Bertweet

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n!pip install /kaggle/input/bertweet-libs/sacrebleu-1.4.10-py3-none-any.whl\n!cp -R /kaggle/input/bertweet-libs/fairseq-0.9.0/fairseq-0.9.0 /kaggle/working\n!cp -R /kaggle/input/bertweet-libs/fastBPE-0.1.0/fastBPE-0.1.0/ /kaggle/working\n\n!pip install /kaggle/working/fairseq-0.9.0/\n!pip install /kaggle/working/fastBPE-0.1.0/\n\n!python ../input/tweet-inference-scripts/inference_bertweet.py')


# ### Roberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_roberta_anton.py')


# ## Hiki

# ### Roberta large

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_roberta_large_hiki.py')


# ### Roberta

# In[ ]:


get_ipython().run_cell_magic('time', '', '!python ../input/tweet-inference-scripts/inference_roberta_hiki.py')


# # Level 2 models

# In[ ]:


get_ipython().system('python ../input/tweet-inference-scripts-lvl-2/inference_cnn_0_2.py')


# In[ ]:


get_ipython().system('python ../input/tweet-inference-scripts-lvl-2/inference_cnn_1_2.py')


# In[ ]:


get_ipython().system('python ../input/tweet-inference-scripts-lvl-2/inference_wavenet_2.py')


# In[ ]:


get_ipython().system('python ../input/tweet-inference-scripts-lvl-2/inference_wavenet_6_2.py')


# # Ensemble

# In[ ]:


import pickle

import numpy as np
import pandas as pd


def string_from_preds_char_level(texts, preds):
    selected_texts = []
    n_models = len(preds)

    for idx in range(len(texts)):
        data = texts[idx]

        start_probas = np.mean(
            [preds[i][0][idx] for i in range(n_models)], 0)
        end_probas = np.mean(
            [preds[i][1][idx] for i in range(n_models)], 0)

        start_idx = np.argmax(start_probas)
        end_idx = np.argmax(end_probas)

        if end_idx < start_idx:
            selected_text = data
        else:
            selected_text = data[start_idx: end_idx]

        selected_texts.append(selected_text.strip())

    return selected_texts


df_test = pd.read_csv(
    '../input/tweet-sentiment-extraction/test.csv').fillna('')
df_test['selected_text'] = ''
sub = pd.read_csv(
    '../input/tweet-sentiment-extraction/sample_submission.csv')


preds_1 = np.load('preds_char_test_cnn_0_2.npy')
preds_2 = np.load('preds_char_test_cnn_1_2.npy')
preds_3 = np.load('preds_char_test_wavenet_2.npy')
preds_4 = np.load('preds_char_test_wavenet_6_2.npy')

test_preds = (preds_1 + preds_2 + preds_3 + preds_4) / 4

selected_texts = string_from_preds_char_level(
    df_test['text'].values, test_preds)

sub['selected_text'] = selected_texts
df_test['selected_text'] = selected_texts
sub.to_csv('submission.csv', index=False)


# In[ ]:




