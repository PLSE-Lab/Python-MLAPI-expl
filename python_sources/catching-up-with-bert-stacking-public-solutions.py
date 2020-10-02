#!/usr/bin/env python
# coding: utf-8

# # Catching up with BERT: stacking public solutions
# 
# We have at least three solutions using different architectures: let's add them up and see what happens!
# 
# I use two great public kernels by Keyi Tang and one of my own:
# 1. [https://www.kaggle.com/keyit92/end2end-coref-resolution-by-attention-rnn](https://www.kaggle.com/keyit92/end2end-coref-resolution-by-attention-rnn)
# 2. [https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention](https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention)
# 3. [https://www.kaggle.com/mamamot/fastai-awd-lstm-solution-0-71-lb](https://www.kaggle.com/mamamot/fastai-awd-lstm-solution-0-71-lb)

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

import os
print(os.listdir("../input"))


# Load:

# In[ ]:


input_path = Path("../input")
submission_file_name = "submission.csv"
right_answers = pd.read_csv(input_path/"gendered-pronoun-resolution"/"test_stage_1.tsv", sep="\t", index_col="ID")

results = [
    pd.read_csv(input_path/"end2end-coref-resolution-by-attention-rnn"/submission_file_name, index_col="ID"),
    pd.read_csv(input_path/"fastai-awd-lstm-solution-0-71-lb"/submission_file_name, index_col="ID"),
    pd.read_csv(input_path/"coref-by-mlp-cnn-coattention"/submission_file_name, index_col="ID"),
]


# Blend:

# In[ ]:


stacked = sum(results) / len(results)


# Submit:

# In[ ]:


stacked.to_csv("submission.csv")

