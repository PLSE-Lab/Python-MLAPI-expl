#!/usr/bin/env python
# coding: utf-8

# #### In this kernel, I will show you how to train language models, such as BERT, from scratch on TPUs!
# 
# #### If you like this kernel, consider upvoting it and the associated datasets:
# 
# - https://www.kaggle.com/abhishek/bert-master
# - https://www.kaggle.com/abhishek/hindi-oscar-corpus
# - https://www.kaggle.com/abhishek/bert-base-uncased

# In[ ]:


get_ipython().system(' pip install -U tokenizers')


# In[ ]:


get_ipython().system('pip install tensorflow==1.15')


# In[ ]:


import tokenizers


# In[ ]:


bwpt = tokenizers.BertWordPieceTokenizer(
    vocab_file=None,
    add_special_tokens=True,
    unk_token='[UNK]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
    wordpieces_prefix='##'
)


# In[ ]:


bwpt.train(
    files=["../input/hindi-oscar-corpus/hi_dedup_1000.txt"],
    vocab_size=30000,
    min_frequency=3,
    limit_alphabet=1000,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]']
)


# In[ ]:


bwpt.save("/kaggle/working/", "hindi")


# In[ ]:


cd ../input/bertsrc/


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('python create_pretraining_data.py     --input_file=/kaggle/input/hindi-oscar-corpus/hi_dedup_1000.txt     --output_file=/kaggle/working/tf_examples.tfrecord     --vocab_file=/kaggle/working/hindi-vocab.txt     --do_lower_case=True     --max_seq_length=128     --max_predictions_per_seq=20     --masked_lm_prob=0.15     --random_seed=42     --dupe_factor=5')


# In[ ]:


get_ipython().system('python run_pretraining.py     --input_file=gs://tf-lang-model/*.tfrecord     --output_dir=gs://tf-lang-model/model/     --do_train=True     --do_eval=True     --bert_config_file=/kaggle/input/bert-base-uncased/config.json     --train_batch_size=32     --max_seq_length=128     --max_predictions_per_seq=20     --num_train_steps=20     --num_warmup_steps=10     --learning_rate=2e-5     --use_tpu=True     --tpu_name=$TPU_NAME')

