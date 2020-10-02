#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Install dependencies
get_ipython().system('pip uninstall -y tensorflow')
get_ipython().system('pip install transformers')
# transformers version at notebook creation --- 2.5.1
# tokenizers version at notebook creation --- 0.5.2


# In[ ]:


from pathlib import Path


# In[ ]:


# paths = [str(x) for x in Path(".").glob("/kaggle/input/quoratext/file_text.txt")]


# In[ ]:


# paths


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from pathlib import Path\n\nfrom tokenizers import ByteLevelBPETokenizer\n\n\n# Initialize a tokenizer\ntokenizer = ByteLevelBPETokenizer()\n\n# Customize training\ntokenizer.train(files="/kaggle/input/quoratext/file_text.txt", vocab_size=52_000, min_frequency=2, special_tokens=[\n    "<s>",\n    "<pad>",\n    "</s>",\n    "<unk>",\n    "<mask>",\n])')


# In[ ]:


get_ipython().system('mkdir EsperBERTo')
tokenizer.save("EsperBERTo")


# In[ ]:


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)


# In[ ]:


tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)


# In[ ]:


tokenizer.encode("Mi estas Julien.")


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


# Check that PyTorch sees it
import torch
torch.cuda.is_available()


# In[ ]:


# Get the example scripts.
get_ipython().system('wget -c https://raw.githubusercontent.com/huggingface/transformers/master/examples/run_language_modeling.py')


# In[ ]:


import json
config = {
	"architectures": [
		"RobertaForMaskedLM"
	],
	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"layer_norm_eps": 1e-05,
	"max_position_embeddings": 514,
	"model_type": "roberta",
	"num_attention_heads": 12,
	"num_hidden_layers": 6,
	"type_vocab_size": 1,
	"vocab_size": 520
}
with open("./EsperBERTo/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {
	"max_len": 512
}
with open("./EsperBERTo/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)


# In[ ]:


cmd =	"""
  python run_language_modeling.py
  --train_data_file ../input/quoratext/file_text.txt
  --output_dir ./EsperBERTo-small-v1
	--model_type roberta
	--mlm
	--config_name ./EsperBERTo
	--tokenizer_name ./EsperBERTo
	--do_train
	--line_by_line
	--learning_rate 1e-4
	--num_train_epochs 1
	--save_total_limit 2
	--save_steps 500
	--per_gpu_train_batch_size 16
	--seed 42
""".replace("\n", " ")


# In[ ]:


get_ipython().run_cell_magic('time', '', '!{cmd}')


# In[ ]:


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="../working/EsperBERTo-small-v1",
    tokenizer="../working/EsperBERTo-small-v1"
)

# The sun <mask>.
# =>

result = fill_mask("Data Science <mask>.")


# In[ ]:


result = fill_mask("Data Science in <mask>.")
result


# In[ ]:


import re


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu
def evaluation(actual,autocomplete):
    result = fill_mask(autocomplete)
    sentances = [i['sequence'] for i in result]
    list_of_sentances = []
    print(sentances)
    for i in sentances:
        line = re.sub('<s>', '', i)
        line = re.sub("</s>","",line)
        list_of_sentances.append(line.split())
    print(list_of_sentances)
    print(actual)
    score = sentence_bleu(list_of_sentances,actual.split())
    print(score)
    return score
    
    


# In[ ]:


print(evaluation("what is datascience","what is <mask>"))


# In[ ]:




