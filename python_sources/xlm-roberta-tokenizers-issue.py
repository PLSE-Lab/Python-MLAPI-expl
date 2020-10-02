#!/usr/bin/env python
# coding: utf-8

# ### Question: Why is the hugging face encoding not match the google sentencepiece encoding? See comparison below...
# 
# 

# ### XLM-RoBERTa Hugging Face:
# 
# ![image.png](attachment:image.png)
# 
# Source: https://huggingface.co/transformers/_modules/transformers/tokenization_xlm_roberta.html#XLMRobertaTokenizer

# In[ ]:


from transformers import XLMRobertaTokenizer

tokenizer_xlmroberta = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

for i in range(5):
    print(i,":",tokenizer_xlmroberta.decode(i))


# In[ ]:


tokenizer_xlmroberta.encode("positive negative neutral")


# ### XLM-RoBERTa Google sentencepiece:
# 
# * Downloaded sentencepiece.bpe.model from link on hugging face source code https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-sentencepiece.bpe.model
# 
# * Currently loading it from Abhishek's uploaded dataset "xlm-roberta-base"
# 
# 
# **Source:** https://github.com/google/sentencepiece

# In[ ]:


import sentencepiece as spm

XLM_ROBERTA_PATH = '/kaggle/input/xlm-roberta-base/'
tokenizer_xlmroberta_ = spm.SentencePieceProcessor()
tokenizer_xlmroberta_.load(f"{XLM_ROBERTA_PATH}sentencepiece.bpe.model") 


# In[ ]:


print(tokenizer_xlmroberta_.encode_as_ids("positive negative neutral"))


# ## Comparison?
# 
# They seem to be off by 1 index... 

# In[ ]:


## Hugging Face:
tokenizer_xlmroberta.encode("I don't understand why",add_special_tokens=False)


# In[ ]:


## Sentencepiece:
tokenizer_xlmroberta_.encode_as_ids("I don't understand why")


# In[ ]:


print("Index |","HugFace |","SentPiece")
for i in range(20):
    print('{:3n}:     {:6s}     {:6s}'.format(i,tokenizer_xlmroberta.decode(i),tokenizer_xlmroberta_.decode_ids([i])))


# In[ ]:


print("DONE: Successful Commit")

