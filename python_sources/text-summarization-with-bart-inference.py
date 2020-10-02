#!/usr/bin/env python
# coding: utf-8

# # Introduction

# [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/bart) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.

# # BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

# [BART](https://github.com/pytorch/fairseq/tree/master/examples/bart) is sequence-to-sequence model trained with denoising as pretraining objective. We show that this pretraining objective is more generic and show that we can match RoBERTa Results on SQuAD and GLUE and gain state-of-the-art results on summarization (XSum, CNN dataset), long form generative question answering (ELI5) and dialog response genration (ConvAI2). See the associated paper for more details.

# # Setup

# In[ ]:


# !pip install fairseq


# # Make evaluation dataset from .csv files

# In[ ]:


# import pandas as pd
# df_summary = pd.read_csv("../input/news-summary/news_summary_more.csv")
# news_txt = df_summary['text']


# In[ ]:


# with open('/kaggle/working/news_text.txt','w') as save_txt:
#     for i in range(100):
#         save_txt.write(news_txt[i].strip()+'\n')


# # Load model for evaluation

# In[ ]:


# import torch

# bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
# bart.cuda()
# bart.eval()
# bart.half()
# count = 1
# bsz = 32


# In[ ]:


# with open('/kaggle/working/news_text.txt') as source, open('/kaggle/working/test.hypo', 'w') as fout:
#     sline = source.readline().strip()
#     slines = [sline]
#     for sline in source:
#         if count % bsz == 0:
#             with torch.no_grad():
#                 hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

#             for hypothesis in hypotheses_batch:
#                 fout.write(hypothesis + '\n')
#                 fout.flush()
#             slines = []

#         slines.append(sline.strip())
#         count += 1


# # Save generated summary and store as *"test.hypo"* file

# In[ ]:


# with open('/kaggle/working/news_text.txt', 'r') as fout:
#     src=fout.readlines()


# In[ ]:


# with open('/kaggle/working/test.hypo', 'r') as fout:
#     s=fout.readlines()


# In[ ]:


# i=12
# print('Source text:'+'\n'+src[i])


# In[ ]:


# print('Summary text:'+'\n'+s[i])

