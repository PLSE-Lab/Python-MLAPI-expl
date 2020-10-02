#!/usr/bin/env python
# coding: utf-8

# <centre> ![image.png](attachment:image.png) 
# 
# # <center>   HuggingFace Tokenizer Cheat-Sheet

# ### This notebook contains handy information to help building NLP models using Hugging Face Library (https://huggingface.co/transformers). 
# 
# #### I will keep updating this notebook. 
# 
# 
# Please upvote if this is useful!
# 

# ## Import Useful Packages

# In[ ]:


import os
import pandas as pd
import numpy as np
import transformers
import tokenizers


# # TOKENIZERS

# ## <span style="color:green">BERT Base Uncased </span>
# 
# ### Offline loading in Kaggle Notebook (Thanks to https://www.kaggle.com/abhishek)
# 
# ```
# https://www.kaggle.com/abhishek/bert-base-uncased
# ```
# 
# That contains:
# 
# ```
# config.json
# vocab.txt
# pytorch_model.bin
# ```
# 
# <span style="color:blue">Example usage: </span>
# ```
# import tokenizers
# 
# TOKENIZER = tokenizers.BertWordPieceTokenizer(
#     f"{BERT_PATH}/vocab.txt", 
#     lowercase=True
# )
# ```
# 
# 
# ### Online loading in Kaggle Notebook (needs Internet) using ```BertTokenizer```
# 
# ```
# https://huggingface.co/transformers/_modules/transformers/tokenization_bert.html
# ```
# 
# <span style="color:blue">Example usage: </span>

# In[ ]:


from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
enc = TOKENIZER.encode("Hello there!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
print("[CLS]: " + str(enc[0]))
print("[SEP]: " + str(enc[4]))
print("[PAD]: " + str(TOKENIZER.encode("[PAD]")[1]))


# **Note: There are also tons of other pretrained models other than ```"bert-base-uncased"```:**
# ```
# PRETRAINED_INIT_CONFIGURATION = {
#     "bert-base-uncased": {"do_lower_case": True},
#     "bert-large-uncased": {"do_lower_case": True},
#     "bert-base-cased": {"do_lower_case": False},
#     "bert-large-cased": {"do_lower_case": False},
#     "bert-base-multilingual-uncased": {"do_lower_case": True},
#     "bert-base-multilingual-cased": {"do_lower_case": False},
#     "bert-base-chinese": {"do_lower_case": False},
#     "bert-base-german-cased": {"do_lower_case": False},
#     "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
#     "bert-large-cased-whole-word-masking": {"do_lower_case": False},
#     "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
#     "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
#     "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
#     "bert-base-german-dbmdz-cased": {"do_lower_case": False},
#     "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
#     "bert-base-finnish-cased-v1": {"do_lower_case": False},
#     "bert-base-finnish-uncased-v1": {"do_lower_case": True},
#     "bert-base-dutch-cased": {"do_lower_case": False},
# }
# ```

# ## <span style="color:green">RoBERTa </span>
# 
# ### Offline loading in Kaggle Notebook (Thanks to https://www.kaggle.com/abhishek)
# 
# ```
# https://www.kaggle.com/abhishek/roberta-base
# ```
# 
# That contains:
# 
# ```
# config.json
# vocab.json
# merges.txt
# pytorch_model.bin
# ```
# 
# 
# <span style="color:blue">Example usage: </span>
# ```
# import tokenizers
# 
# ROBERTA_PATH = "../input/roberta-base"
# TOKENIZER = tokenizers.ByteLevelBPETokenizer(
#     vocab_file=f"{ROBERTA_PATH}/vocab.json", 
#     merges_file=f"{ROBERTA_PATH}/merges.txt", 
#     lowercase=True,
#     add_prefix_space=True
# )
# ```
# 
# 
# ### Online loading in Kaggle Notebook (needs Internet) using ```RobertaTokenizer```
# 
# ```
# https://huggingface.co/transformers/_modules/transformers/tokenization_roberta.html
# ```
# 
# <span style="color:blue">Example usage: </span>

# In[ ]:


from transformers import RobertaTokenizer
TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
enc = TOKENIZER.encode("Hello there!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
print("[CLS]: " + str(enc[0]))
print("[SEP]: " + str(enc[4]))
print("<pad>: " + str(TOKENIZER.encode("<pad>")[1]))


# **Note: There are also tons of other pretrained models other than ```"roberta-base"```:**
# 
# ```
# PRETRAINED_VOCAB_FILES_MAP = {
#     "vocab_file": {
#         "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
#         "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
#         "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
#         "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
#         "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
#         "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
#     },
#     "merges_file": {
#         "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
#         "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
#         "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
#         "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
#         "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
#         "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
#     },
# }
# ```

# ## <span style="color:green">ALBERT</span>
# 
# ### Offline loading in Kaggle Notebook
# 
# ```
# [TODO] Planning to create a dataset in future!
# ```
# 
# 
# 
# ### Online loading in Kaggle Notebook (needs Internet) using ```AlbertTokenizer```
# 
# ```
# https://huggingface.co/transformers/_modules/transformers/tokenization_albert.html
# ```
# 
# <span style="color:blue">Example usage: </span>

# In[ ]:


from transformers import AlbertTokenizer
TOKENIZER = AlbertTokenizer.from_pretrained("albert-base-v1")
enc = TOKENIZER.encode("Hello there!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
print("[CLS]: " + str(enc[0]))
print("[SEP]: " + str(enc[4]))
print("<pad>: " + str(TOKENIZER.encode("<pad>")[1]))


# **Note: There are also tons of other pretrained models other than ```"albert-base-v1"```:**
# ```
# PRETRAINED_VOCAB_FILES_MAP = {
#     "vocab_file": {
#         "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-spiece.model",
#         "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-spiece.model",
#         "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-spiece.model",
#         "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-spiece.model",
#         "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model",
#         "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-spiece.model",
#         "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-spiece.model",
#         "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-spiece.model",
#     }
# }
# 
# ```

# ## <span style="color:green">Bart </span>
# 
# ### Offline loading in Kaggle Notebook
# 
# ```
# [TODO]
# ```
# 
# 
# ### Online loading in Kaggle Notebook (needs Internet) using ```RobertaTokenizer```
# 
# ```
# https://huggingface.co/transformers/_modules/transformers/modeling_bart.html
# ```
# 
# <span style="color:blue">Example usage: </span>

# In[ ]:


from transformers import BartTokenizer
TOKENIZER = BartTokenizer.from_pretrained('bart-large')
enc = TOKENIZER.encode("Hello there!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
print("[CLS]: " + str(enc[0]))
print("[SEP]: " + str(enc[4]))
print("<pad>: " + str(TOKENIZER.encode("<pad>")[1]))


# 
# **Note: There are also tons of other pretrained models other than ```"bart-large"```:**
# 
# ```
# BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
#     "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin",
#     "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/pytorch_model.bin",
#     "bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/pytorch_model.bin",
#     "bart-large-xsum": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-xsum/pytorch_model.bin",
# }
# ```

# ## <span style="color:green">ELECTRA </span>
# 
# ### Offline loading in Kaggle Notebook (Thanks to https://www.kaggle.com/ratan123)
# 
# ```
# https://www.kaggle.com/ratan123/electra-base
# ```
# 
# 
# <span style="color:blue">Example usage: </span>
# ```
# import tokenizers
# 
# ELECTRA_PATH = "/kaggle/input/electra-base/"
# TOKENIZER = tokenizers.BertWordPieceTokenizer(
#     f"{ELECTRA_PATH}/vocab.txt", 
#     lowercase=True
# )
# ```
# 
# ### Online loading in Kaggle Notebook (needs Internet) using ```ElectraTokenizer```
# 
# ```
# https://huggingface.co/transformers/_modules/transformers/tokenization_electra.html
# ```
# 
# <span style="color:blue">Example usage: </span>

# In[ ]:


from transformers import BertTokenizer
# class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end tokenization: punctuation splitting + wordpiece.
TOKENIZER = BertTokenizer.from_pretrained('google/electra-base-generator')
enc = TOKENIZER.encode("Hello there!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
print("[CLS]: " + str(enc[0]))
print("[SEP]: " + str(enc[4]))
print("[PAD]: " + str(TOKENIZER.encode("[PAD]")[1]))

