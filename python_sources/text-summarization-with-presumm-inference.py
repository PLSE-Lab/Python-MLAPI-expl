#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This kernel demonstrates how to use the open source code [PreSumm](http://github.com/nlpyang/PreSumm) released by **yang liu** and to conduct text summarization task on your own dataset. It covers the following contents
# 
# - Evaluation on your own text files to generate summary
# 
# ***Notice that:** The Presumm code used here has been modified to run in kaggle and process the customized data.*

# # Setup 
# Install the required package from **requirements.txt** in presummy folder

# In[ ]:


get_ipython().system('pip install -r /kaggle/input/pre-summy/PreSumm/requirements.txt')


# # Read the raw description and summary data

# Glimpse on our News summary data released by **Kondalarao Vonteru** in [here](https://www.kaggle.com/sunnysai12345/news-summary#news_summary_more.csv)

# In[ ]:


import pandas as pd 
summary_df = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')
summary_df


# # Data Preprocess

# The inference support summarizing raw text input. The example data format are in the [raw_data directory ](https://github.com/nlpyang/PreSumm/tree/dev/raw_data)Each line in the text document should be a source text to be summarized.  
# Hence, at first we should convert our raw data from *.csv *format to *.txt* format.(*We only process 50 samples here to save kernel commit time*)
# 
# 
# 

# In[ ]:


import pandas as pd
from nltk.tokenize import sent_tokenize

news_df = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')
news_txt = news_df['text']


with open('/kaggle/working/temp_news_text.txt', 'w') as save_txt:
    for i in range(50):
        save_txt.write(news_txt[i].strip() + '\n')


# ## Take a glimpse on few sample

# In[ ]:


with open('/kaggle/working/temp_news_text.txt', 'r') as save_txt:
    f= save_txt.readlines()


# In[ ]:


print(f[0])


# In[ ]:


print(f[1])


# # Model Inference

# In[ ]:


get_ipython().system('mkdir logs')
get_ipython().system('mkdir temp')
get_ipython().system('mkdir models')
get_ipython().system('mkdir result')


# In[ ]:


import os
os.chdir('/kaggle/input/pre-summy/PreSumm/src/')


# In[ ]:


get_ipython().system('python train.py -task abs -mode test_text -text_src /kaggle/working/temp_news_text.txt -test_from /kaggle/input/absbert-weights/model_step_148000.pt -model_path /kaggle/working/models -visible_gpus -1 -alpha 0.2 -result_path /kaggle/working/result/news -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/cnndm.log')


# ## Take a glimpse on the prediction summary

# In[ ]:


import os
os.chdir('/kaggle/working/result/')


# In[ ]:


with open('news.-1.candidate', 'r') as save_txt:
    s= save_txt.readlines()
    


# In[ ]:


print(s[0])


# In[ ]:


print(s[1])


# # Remove generate data to save commit time

# In[ ]:


import os
os.chdir('/kaggle/working/')


# In[ ]:


get_ipython().system('rm -r logs')
get_ipython().system('rm -r temp')
get_ipython().system('rm -r models')
get_ipython().system('rm -r result')


# In[ ]:




