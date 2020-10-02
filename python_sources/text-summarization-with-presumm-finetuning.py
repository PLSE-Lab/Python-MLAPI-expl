#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This kernel demonstrates how to use the open source code [PreSumm](http://github.com/nlpyang/PreSumm) released by **yang liu** and to conduct text summarization task on your own dataset. It covers the following contents
# 
# - 1. Process the data by using **process.py** from PreSumm repo and [**StanfordCoreNLP**](https://stanfordnlp.github.io/CoreNLP/download.html) tool
# - 2. Fine-tuning the text summarization model and save weights
# 
# ***Notice that:** The Presumm code used here has been modified to run in kaggle and process the customized data.*

# # Setup 
# Install the required package from **requirements.txt** in presummy folder

# In[ ]:


# !pip install -r /kaggle/input/pre-summy/PreSumm/requirements.txt


# # Read the raw description and summary data

# Glimpse on our News summary data released by **Kondalarao Vonteru** in [here](https://www.kaggle.com/sunnysai12345/news-summary#news_summary_more.csv)

# In[ ]:


import pandas as pd 
summary_df = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')
summary_df


# # Data Preprocess

# Our data process includes the following four steps:  
# - 1. Convert our raw data to **.story** format and store in */kaggle/working/news_sum/* . This enables us to use **preprocess.py** code to do further process  
# - 2. Tokenize **.story** file via using [**StanfordCoreNLP**](https://stanfordnlp.github.io/CoreNLP/index.html). Stanford CoreNLP is written in **Java**; recent releases require Java **1.8+**. You need to have Java installed to run CoreNLP. However, you can interact with CoreNLP via the command-line or its web service; many people use CoreNLP while writing their own code in Javascript, Python, or some other language.  
# 
#     (*Here we activate Stanford CoreNLP by the code in **preprocess.py***)
# - 3. Convert tokenized data as **json** format 
# - 4. Convert **json** file to **.pt** format for model finetuning
# 

# ## Create required folder for the preprocess and modeling fine-tuning

# ## **folder name -> data file**
# * **news_sum** -> .story data  
# * **merged_stories_tokenized** -> tokenized data  
# * **json_data** ->  json files
# * **bert_data** ->  .pt files
# * **logs** ->  for storing logs information during preprocess and finetuning
# * **temp** -> cache model config data
# * **bertsumextabs** ->  save finetuning model weights****
# 

# In[ ]:


import os 
os.chdir('/kaggle/working')


# In[ ]:


get_ipython().system('mkdir news_sum')
get_ipython().system('mkdir merged_stories_tokenized')
get_ipython().system('mkdir json_data')
get_ipython().system('mkdir bert_data')
get_ipython().system('mkdir logs')
get_ipython().system('mkdir temp')
get_ipython().system('mkdir bertsumextabs ')


# ## 1. Make raw description and summary data as *.story*** files
# Here we preprocess only 100 files to save kernel commit time 

# In[ ]:


# import os
# def create_story_files(df,num_file=100,filename=None):
#     for i in range(num_file):
#         doc = df['text'][i] + '\n'*2 + '@highlight' + '\n'*2 + df['headlines'][i]
#         file_name = os.path.join(filename,(str(i) + '.story'))
#         with open(file_name,'w') as story_file:
#             story_file.write(doc)


# In[ ]:


# create_story_files(summary_df,num_file=30000,filename='/kaggle/working/news_sum')


# ## 2. Make  *.story* files as tokenized data file via StandfordCoreNLP

# In[ ]:


import os
os.chdir('/kaggle/input/pre-summy/PreSumm/src/')


# In[ ]:


# !python preprocess.py -mode tokenize -raw_path /kaggle/working/news_sum -save_path /kaggle/working/merged_stories_tokenized  -log_file /kaggle/working/logs/cnndm.log


# ## 3. Make  *tokenized data* files as *.json file* 

# In[ ]:


# !python preprocess.py -mode format_to_lines -raw_path /kaggle/working/merged_stories_tokenized -save_path /kaggle/working/json_data/news -n_cpus 1 -use_bert_basic_tokenizer false -log_file /kaggle/working/logs/cnndm.log 


# ## 4. Make .json file as .pt file for BERT model

# In[ ]:


# !python preprocess.py -mode format_to_bert -raw_path /kaggle/working/json_data -save_path /kaggle/working/bert_data  -lower -n_cpus 1 -log_file /kaggle/working/logs/cnndm.log 


# In[ ]:


import os
os.chdir('/kaggle/working/')

get_ipython().system('rm -r news_sum')
get_ipython().system('rm -r merged_stories_tokenized')
get_ipython().system('rm -r json_data')


# # Model Finetuning

# * The code here uses a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences.   
# * The original [paper](https://arxiv.org/abs/1908.08345) includes a two-staged fine-tuning approach *(finetuning on extractive BERT and then abstractive BERT)* can further boost the quality of the generated summaries, but here we only do the second stage finetuning on the **abstractive summarization model**.

# In[ ]:


import os
os.chdir('/kaggle/input/pre-summy/PreSumm/src/')


# In[ ]:


# !python train.py  -task abs -mode train -train_from /kaggle/input/absbert-weights/model_step_149000.pt -bert_data_path /kaggle/working/bert_data/news  -dec_dropout 0.2  -model_path /kaggle/working/bertsumextabs -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 150000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0  -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/abs_bert_cnndm


# # Evaluation - Rouge-1 , Rouge-L 

# We can get Average **Rouge-1: 0.522**,  **Rouge-L:0.487** after finetuning

# In[ ]:


os.chdir('/kaggle/input/pre-summy/PreSumm/src')


# In[ ]:


# !python train.py -task abs -mode test -model_path /kaggle/working/bertsumextabs -test_from /kaggle/working/bertsumextabs/model_step_150000.pt -batch_size 100 -test_batch_size 100 -bert_data_path /kaggle/working/bert_data/news -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/abs_bert_cnndm  -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.82 -min_length 10 -result_path /kaggle/working/logs/abs_bert_cnndm


# In[ ]:


os.chdir('/kaggle/working/')


# In[ ]:


# with open('logs/abs_bert_cnndm.150000.gold','r') as s:
#     summary = s.readlines()

# for i in range(10):
#     print(summary[i])


# In[ ]:


# with open('logs/abs_bert_cnndm.150000.candidate','r') as s:
#     cand = s.readlines()

# for i in range(10):
#     print(cand[i])


# In[ ]:


# !pip install rouge


# In[ ]:


# from rouge import Rouge
# rouge = Rouge()
# rouge_1= 0
# rouge_l=0
# for i in range(len(summary)):
#     scores = rouge.get_scores(summary[i],cand[i])
#     rating = 'good' if scores[0]['rouge-l']['r']>0.5 else 'bad'
#     rouge_1+=float(scores[0]['rouge-1']['r'])
#     rouge_l+=float(scores[0]['rouge-l']['r'])
    
# rouge_1/=len(summary)
# rouge_l/=len(summary)


# In[ ]:


# print('Average Rouge-1: {},  Rouge-L:{}'.format(rouge_1 ,rouge_l))


# # Remove generate data to save commit time

# In[ ]:


import os
os.chdir('/kaggle/working/')


# In[ ]:


get_ipython().system('rm -r bert_data')
get_ipython().system('rm -r logs')
get_ipython().system('rm -r temp')
get_ipython().system('rm -r bertsumextabs ')

