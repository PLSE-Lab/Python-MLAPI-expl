#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.getcwd()
os.chdir('/kaggle/working')
os.mkdir('data_token')
os.mkdir('model')
os.mkdir('tensorboard_summary')


# In[ ]:


cd data_token


# In[ ]:


os.mkdir('tokenized')


# In[ ]:


import os
os.chdir('/kaggle/input/gpt2chinesemaster/GPT2-Chinese-master')
os.getcwd()
get_ipython().system("python train.py --raw --epochs=10 --tokenized_data_path='/kaggle/working/data_token/tokenized/' --batch_size=4 --log_step=20 --output_dir='/kaggle/working/model/' --writer_dir='/kaggle/working/tensorboard_summary/'")

