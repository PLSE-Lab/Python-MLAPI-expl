#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/OpenNMT/OpenNMT-py/')
get_ipython().system('pip install -r OpenNMT-py/requirements.opt.txt')
get_ipython().system('pip install gcsfs')
get_ipython().system('pip install rouge > /dev/null')


# In[ ]:


import numpy as np 
import pandas as pd 
from IPython.display import display, Markdown

from rouge import Rouge 
import os
from sklearn.utils import shuffle
from pathlib import Path
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Read data**
# 
# In addition to dataset [News about major cryptocurrencies 2013-2018 (40k)](https://www.kaggle.com/kashnitsky/news-about-major-cryptocurrencies-20132018-40k) i used
# [US Financial News Articles](https://www.kaggle.com/jeet2016/us-financial-news-articles)
# 

# In[ ]:


PATH_TO_FINNEWS = Path('gs://cryptosum_dataset')
PATH_TO_CRYPTONEWS = Path('../input/news-about-major-cryptocurrencies-20132018-40k/')


# In[ ]:


train_df_crypto = pd.read_csv(PATH_TO_CRYPTONEWS / 'crypto_news_parsed_2013-2017_train.csv')
valid_df = pd.read_csv(PATH_TO_CRYPTONEWS / 'crypto_news_parsed_2018_validation.csv')
train_df_fin = pd.read_csv(PATH_TO_FINNEWS / 'Financial_news_dataset.csv')


# In[ ]:


train_df_fin.head()


# Simple preprocessing. 
# Remove rows if they have empty text or title, also remove rows with a short text.

# In[ ]:


valid_df['text'].fillna(' ', inplace=True)

train_df_crypto = train_df_crypto.dropna()
train_df_crypto = train_df_crypto[train_df_crypto['title']!=' ']

train_df_fin = train_df_fin.dropna()
train_df_fin = train_df_fin[train_df_fin['title']!=' ']
train_df_fin = train_df_fin[train_df_fin['text']!=' ']
train_df_fin = train_df_fin[train_df_fin['title'].str.len()>15]
train_df_fin = train_df_fin[train_df_fin['text'].str.len()>15]


# In[ ]:


train_df_fin['text'].apply(lambda s: len(s.split())).describe()


# In[ ]:


train_df_fin['title'].apply(lambda s: len(s.split())).describe()


# In[ ]:


train_df_crypto['title'].apply(lambda s: len(s.split())).describe()


# # Preprocessing the data
# For the current dataset, I additionally truncate the source length at 400 tokens and the target at 15. Also, note that in current dataset models work better if the target surrounds sentences with tags such that a sentence looks like  `<t> w1 w2 w3 </t>` 

# In[ ]:


max_str_length = 400
max_title_length = 15

title_val = '<t> ' + valid_df['title'].apply(lambda x : ' '.join(x.split()[:max_title_length])) + ' </t>' + ' \n'
text_val = valid_df['text'].apply(lambda x : ' '.join(x.split()[:max_str_length])) + ' \n'

title_tr = '<t> ' + train_df_crypto['title'].apply(lambda x : ' '.join(x.split()[:max_title_length])) + ' </t>' + ' \n'
text_tr = train_df_crypto['text'].apply(lambda x : ' '.join(x.split()[:max_str_length]))  + ' \n'

title_tr_fin = '<t> ' + train_df_fin['title'].apply(lambda x : ' '.join(x.split()[:max_title_length])) + ' </t>' + ' \n'
text_tr_fin = train_df_fin['text'].apply(lambda x : ' '.join(x.split()[:max_str_length]).replace('\n', '')) + ' \n'

train_text = text_tr.values.tolist() + text_tr_fin.values.tolist()
train_target = title_tr.values.tolist() + title_tr_fin.values.tolist() 


# In[ ]:


train_text, train_target = shuffle(train_text, train_target)


# In[ ]:


with open("val_text.txt", 'w') as f:
    f.writelines(text_val.values.tolist())
with open("val_target.txt", 'w') as f:
    f.writelines(title_val.values.tolist())

with open("tr_target.txt", 'w') as f:
    f.writelines(train_target)
with open("tr_text.txt", 'w') as f:
    f.writelines(train_text)


# Since I am using [copy-attention](https://papers.nips.cc/paper/5866-pointer-networks.pdf) in the model, I need to preprocess the dataset such that source and target are aligned and use the same dictionary. This is achieved by using the options dynamic_dict and share_vocab. 
# 
# **Command used:**

# <code>!python OpenNMT-py/preprocess.py \
# -train_src tr_text.txt \
# -train_tgt tr_target.txt \
# -valid_src val_text.txt  \
# -valid_tgt val_target.txt \
# -dynamic_dict \
# -share_vocab \
# -overwrite \
# -shard_size 20000 \
# -src_vocab_size 50000 \
# -tgt_vocab_size 50000 \
# -save_data preprocessed_data/cryptodata \
# -tgt_seq_length 20 \
# -src_seq_length 405 \
# -seed 17 \
# -lower  </code>

# # Train the pointer-generator model for 80000 steps. 
# [pointer-generator networks (See 2017) ](https://arxiv.org/pdf/1704.04368.pdf)
# 
# 
# > **copy_attn:** This is the most important option, since it allows the model to copy words from the source.
# 
# > **global_attention mlp: **This makes the model use the attention mechanism introduced by [Bahdanau et al.](https://arxiv.org/abs/1409.0473) instead of that by [Luong et al.](https://arxiv.org/abs/1508.04025)(global_attention dot).
# 
# > **reuse_copy_attn:** This option reuses the standard attention as copy attention. Without this, the model learns an additional attention that is only used for copying.
# 
# > **copy_loss_by_seqlength:** This modifies the loss to divide the loss of a sequence by the number of tokens in it. In practice, we found this to generate longer sequences during inference. However, this effect can also be achieved by using penalties during decoding.
# 
# > **bridge:** This is an additional layer that uses the final hidden state of the encoder as input and computes an initial hidden state for the decoder. Without this, the decoder is initialized with the final hidden state of the encoder directly.
# 
# > **optim SGD:** Alternative training procedures such as adam with initial learning rate 0.001 converge faster than sgd, but achieve slightly worse results. I additionally set the maximum norm of the gradient to 2, and renormalize if the gradient norm exceeds this value.
# 
# > I am using using a 128-dimensional word-embedding, and 256-dimensional 1 layer LSTM. On the encoder side, I use a bidirectional LSTM (brnn), which means that the 256 dimensions are split into 128 dimensions per direction.
# 
# > I additionally set the maximum norm of the gradient to 2, and renormalize if the gradient norm exceeds this value. 
# 
# **commands used:**

# <code>!python OpenNMT-py/train.py \
# -data preprocessed_data/cryptodata \
# -save_model model/BiLSTM_copy_attn \
# -copy_attn \
# -global_attention mlp \
# -word_vec_size 128 \
# -rnn_size 256 \
# -layers 1 \
# -encoder_type brnn \
# -train_steps 80000 \
# -valid_steps 500 \
# -save_checkpoint_steps 5000 \
# -max_grad_norm 2 \
# -dropout 0.3 \
# -batch_size 32 \
# -valid_batch_size 32 \
# -optim sgd \
# -learning_rate 1 \
# -copy_loss_by_seqlength \
# -reuse_copy_attn \
# -bridge \
# -seed 17 \
# -world_size 1 \
# -gpu_ranks 0 </code>

# # Evaluation

# In[ ]:


get_ipython().system('python OpenNMT-py/translate.py -gpu 0 -batch_size 20 -beam_size 3 -model /kaggle/input/onmt-bilstm-copy-attn/BiLSTM_copy_attn_step_80000.pt -src val_text.txt -output predicted_title.txt -min_length 8 -max_length 15 -length_penalty avg -alpha 0.8 -verbose -replace_unk -dynamic_dict -block_ngram_repeat 3 -ignore_when_blocking "." "</t>" "<t>" ')


# In[ ]:


get_ipython().system("sed -i 's/ <\\/t>//g' predicted_title.txt ")
get_ipython().system("sed -i 's/<t> //g' predicted_title.txt")


# In[ ]:


with open('predicted_title.txt', 'r') as f:
    pred_titles = f.readlines()
pred_titles = [x.replace('\n', '').lower() for x in pred_titles]

true_val_titles = valid_df['title'].str.lower().tolist()

import string
punctuation = string.punctuation

true_titles = []
for tr in true_val_titles:
    for p in punctuation:
        tr = tr.replace(p, f' {p} ')
    true_titles.append(tr.lower().replace('  ', ' '))

predicted_titles = []
for pred in pred_titles:
    for p in punctuation:
        pred = pred.replace(p, f' {p} ')
    predicted_titles.append(pred.lower().replace('  ', ' '))


# In[ ]:


from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(hyps=predicted_titles, refs=true_titles, avg=True, ignore_empty=True)


# In[ ]:


scores


# In[ ]:


final_metric = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
final_metric


# # Eyeballing the results: good and bad cases

# In[ ]:


scores_by_example = rouge.get_scores(hyps=predicted_titles, refs=true_titles, avg=False, ignore_empty=True)
scores_by_example = np.array([(x['rouge-1']['f'] + x['rouge-2']['f'] + x['rouge-l']['f']) / 3 for x in scores_by_example])


# In[ ]:


def print_result(index):
    display(Markdown('> **Rouge:** ' + str(round(scores_by_example[index], 3))))
    display(Markdown('> **Title:** ' + valid_df['title'].iloc[index]))
    display(Markdown('> **Generated:** ' + pred_titles[index]))
    display(Markdown('> **Text:** ' + valid_df['text'].iloc[index]))
    print('_' * 60)


# In[ ]:


top_best_10 = scores_by_example.argsort()[-10:]
top_worst_10 = scores_by_example.argsort()[:10]


# In[ ]:


for i in top_best_10:
    print_result(i)


# In[ ]:


for i in top_worst_10:
    print_result(i)

