#!/usr/bin/env python
# coding: utf-8

# # Try language modeling fine tuning on WikiText data

# 1. Clone Transformers library
# 2. Install requirements
# 3. Add data - Used Keras_transformers dataset which has WikiText dataset added
# 4. Move data from input_data folder to working directory where Tokenized and transformed data will be stored
# 5. Use pretrained LM model to get baseline score
# 6. Run Fine tune LM code (Storing various checkpoints)
# 7. It shows perplexity as evaluation metric ( Perplxity here is Masked LM loss )
# 8. Also, inspect the cached dataset used by the model
# 

# In[ ]:


get_ipython().system('git clone https://github.com/huggingface/transformers')


# In[ ]:


cd transformers


# In[ ]:


get_ipython().system('pip install .')
get_ipython().system('pip install -r ./examples/requirements.txt')


# In[ ]:


cd .. 


# In[ ]:


get_ipython().system('dir transformers/examples/')


# In[ ]:


get_ipython().system('ls ../input/kerastransformer/example/wikitext-2-raw-v1/wikitext-2-raw/')


# In[ ]:


pwd


# In[ ]:


#!rm -r /kaggle/working/outputs
get_ipython().system('mkdir /kaggle/working/outputs')
get_ipython().system('ls /kaggle/working/')


# In[ ]:


get_ipython().system('mkdir /kaggle/working/input_data')
get_ipython().system('dir /kaggle/working/')


# In[ ]:


get_ipython().system('cp /kaggle/input/kerastransformer/example/wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw /kaggle/working/input_data')
get_ipython().system('cp /kaggle/input/kerastransformer/example/wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw /kaggle/working/input_data')
get_ipython().system('du -s -h /kaggle/working/input_data/*')


# In[ ]:


get_ipython().system('head -n 4 /kaggle/working/input_data/wiki.test.raw ')


# **Get pretrained model and baseline perplexity score on test dataset**

# In[ ]:


# from transformers import BertTokenizer, BertConfig, BertForMaskedLM
# #help(BertModel)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')


# In[ ]:


# #!rm -r /kaggle/working/output_pretrained
# !mkdir /kaggle/working/output_pretrained
# model.save_pretrained("/kaggle/working/output_pretrained")
# tokenizer.save_pretrained("/kaggle/working/output_pretrained")

# !du -s -h /kaggle/working/output_pretrained/*


# In[ ]:


# !python transformers/examples/run_language_modeling.py \
#     --output_dir=/kaggle/working/output_pretrained \
#     --model_type=bert \
#     --model_name_or_path=/kaggle/working/output_pretrained \
#     --train_data_file=/kaggle/working/input_data/wiki.train.raw \
#     --do_eval \
#     --eval_data_file=/kaggle/working/input_data/wiki.test.raw \
#     --mlm


# In[ ]:


# Baseline perplexity score for pretrained model

#!head /kaggle/working/output_pretrained/eval_results.txt


# Pretrained Masked LM bert model gives baseline perplexity of **9.2404** on test wiki data

# In[ ]:


# !rm -r /kaggle/working/output_pretrained
# !rm /kaggle/working/input_data/bert_cached_lm_510_wiki.test.raw


# RUN FINE TUNING

# In[ ]:


get_ipython().system('python transformers/examples/run_language_modeling.py     --output_dir=/kaggle/working/outputs     --model_type=bert     --model_name_or_path=bert-base-uncased     --do_train     --train_data_file=/kaggle/working/input_data/wiki.train.raw     --do_eval     --eval_data_file=/kaggle/working/input_data/wiki.test.raw     --num_train_epochs=1     --save_steps=400     --eval_all_checkpoints     --mlm')


# In[ ]:


get_ipython().system('du -s -h /kaggle/working/outputs/*')


# **EVALUATE**

# In[ ]:


import glob
import os
from transformers import WEIGHTS_NAME
print(WEIGHTS_NAME)
directory_cpt = list(os.path.dirname(c) for c in sorted(glob.glob("/kaggle/working/outputs" + "/**/" + WEIGHTS_NAME, recursive=True)))
print(directory_cpt)


# In[ ]:


for directory_cpt_name in directory_cpt:
    tmp_dir = directory_cpt_name+"/eval_results.txt"
    print("\n"+tmp_dir)
    get_ipython().system('head {tmp_dir}')


# In[ ]:


get_ipython().system('head -n 10 /kaggle/working/outputs/vocab.txt')


# In[ ]:


get_ipython().system('head -n 10 /kaggle/working/outputs/special_tokens_map.json')


# In[ ]:


get_ipython().system('head /kaggle/working/outputs/config.json')


# In[ ]:


# Load cached data and see
import pickle
with open("/kaggle/working/input_data/bert_cached_lm_510_wiki.train.raw", "rb") as handle:
    tmp_data = pickle.load(handle)


# In[ ]:


print(type(tmp_data))
print(len(tmp_data))
print(tmp_data[0])


# In[ ]:


# Length of each sentence
print(len(tmp_data[0]))
print(len(tmp_data[1]))


# In[ ]:


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/kaggle/working/outputs")
print(tokenizer.decode(tmp_data[0]))


# In[ ]:


print(tokenizer.max_len)
print(tokenizer.max_len_single_sentence)


# In[ ]:


get_ipython().system('head -n 15 /kaggle/working/input_data/wiki.test.raw')


# In[ ]:


from transformers import pipeline
help(pipeline)


# In[ ]:



fill_mask = pipeline(
    "fill-mask",
    model="/kaggle/working/outputs",
    tokenizer="/kaggle/working/outputs"
)


# In[ ]:


result = fill_mask( "Robert Boulter is an [MASK] film actor")
result


# In[ ]:




