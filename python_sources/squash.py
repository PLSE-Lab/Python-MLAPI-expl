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


get_ipython().system('ls')


# In[ ]:


get_ipython().system('git clone https://github.com/martiansideofthemoon/squash-generation')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os
os.chdir('/kaggle/working/squash-generation')
get_ipython().system('ls')


# In[ ]:


get_ipython().system('pip install -r requirements.txt')


# In[ ]:


get_ipython().system('ls ')


# In[ ]:


# !cp -r /kaggle/input/gpt2-corefs/gpt2_corefs_question_generation /kaggle/working/squash-generation/question-generation/


# In[ ]:


os.chdir('/kaggle/working/squash-generation/pytorch-pretrained-BERT')
get_ipython().system('pip install --editable .')


# In[ ]:


os.chdir('/kaggle/working/squash-generation')
get_ipython().system('ls')


# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


# !python squash/extract_answers.py --key quac_869


# In[ ]:


# !python question-generation/interact.py --model_checkpoint /kaggle/input/gpt2-corefs/gpt2_corefs_question_generation/ --model_type gpt2 --key quac_869


# In[ ]:


# !ls squash/temp/quac_869


# In[ ]:


# with open("/kaggle/working/squash-generation/squash/temp/quac_869/generated_questions.json") as f:
#   print(f.read())
    


# In[ ]:


# import json 
  
# # Opening JSON file 
# f = open('/kaggle/working/squash-generation/squash/temp/quac_869/generated_questions.json','r') 
  
# # returns JSON object as  
# # a dictionary 
  


# In[ ]:


# print(f)


# In[ ]:


# import json


# json_object = json.loads(f.read())

# json_formatted_str = json.dumps(json_object, indent=2)

# print(json_formatted_str)


# python -m torch.distributed.launch --nproc_per_node=4 question-generation/train.py --eval_before_start --n_epochs 4 --model_checkpoint gpt2 --dataset_path data/temp_dataset/instances_corefs --dataset_cache data/temp_dataset/cache_corefs --output_dir question-generation/gpt2_corefs_question_generation
# 

# In[ ]:


get_ipython().system('ls')


# In[ ]:


import torch


# In[ ]:


# !ls /kaggle/input/temp-dataset/temp_dataset/






# In[ ]:


# !rm data/temp_dataset/README.md


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls /kaggle/input/pickkle')


# In[ ]:





# In[ ]:


get_ipython().system('cp -r /kaggle/input/pickkle/*  /kaggle/working/squash-generation/data/temp_dataset/')


# In[ ]:


get_ipython().system('ls data/temp_dataset')


# In[ ]:


# !pip install git+https://github.com/lanpa/tensorboardX


# In[ ]:


# !mkdir question-generation/gpt2_corefs_question_generations
# !ls question-generation


# In[ ]:


get_ipython().system('python -m spacy download en')


# In[ ]:


#!pip install git+https://github.com/lanpa/tensorboardX


# In[ ]:


get_ipython().system('pip install tensorboardX==1.6')


# In[ ]:


get_ipython().system('ls data/temp_dataset/temp_dataset')


# In[ ]:


# !ls question-generation/gpt2_corefs_question_generations


# In[ ]:


# !cp question-generation/gpt2_corefs_question_generation/pytorch_model.bin question-generation/gpt2_corefs_question_generations


# In[ ]:


# !ls question-generation/gpt2_corefs_question_generations


# In[ ]:


get_ipython().system('python -m torch.distributed.launch --nproc_per_node=1 question-generation/train.py --eval_before_start --n_epochs=1 --model_checkpoint gpt2 --train_batch_size 2 --valid_batch_size 2 --gradient_accumulation_steps 16  --dataset_path data/temp_dataset/instances_corefs --dataset_cache data/temp_dataset/cache_corefs --output_dir question-generation/gpt2_corefs_question_generation')


# In[ ]:





# In[ ]:


torch.cuda.get_device_name(0)


# In[ ]:




