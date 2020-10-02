#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# !ls


# In[ ]:



# !nvcc --version
# !nvidia-smi


# In[ ]:


# !pip install torch==1.1.0


# In[ ]:


# %%bash
# export CUDA_HOME=/usr/local/cuda-10.1
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension


# In[ ]:


pip install text2text


# In[ ]:


from numpy import nan
import nltk
import json


# In[ ]:


import pandas as pd
df=pd.read_csv('/kaggle/input/stanford-plato-corpus/ranked_sentences.csv')


# In[ ]:


df


# In[ ]:


df['filename'][1]


# In[ ]:


data_l = list()
for data in df['data']:
    data_l.append(eval(data))
df['data'] = data_l
    


# In[ ]:


tagged = list()
for rank in df['tagged_ranked']:
    tagged.append(eval(rank))
df['tagged_ranked'] = tagged


# In[ ]:


untagged = list()
for rank in df['untagged_ranked']:
    untagged.append(eval(rank))
df['untagged_ranked'] = untagged


# In[ ]:


#Transformer model that can generate questions given a sentence
from text2text.text_generator import TextGenerator
qg = TextGenerator(output_type="question")


# In[ ]:


#SOTA transformer model to find answers from comprehension question answering
from transformers import pipeline
nlp=pipeline('question-answering')


# In[ ]:



# for i in df['untagged_ranked'][0][0:10]:
#     print(qg.predict([i[1]]))


# In[ ]:





# In[ ]:


new_df_clean = pd.read_csv('/kaggle/input/stanford-plato-corpus/clean.csv')
new_df_unclean = pd.read_csv('/kaggle/input/stanford-plato-corpus/unclean.csv')


# In[ ]:


new_df_clean.shape


# In[ ]:


# new_df_unclean.head()


# In[ ]:


# new_df_clean = pd.DataFrame(columns=['title', 'filename', 'question', 'sentence', 'summarized_paragraph', 'sec_id', 'par_id'])
# new_df_unclean = pd.DataFrame(columns=['title', 'filename', 'question', 'sentence', 'summarized_paragraph', 'sec_id', 'par_id'])


# In[ ]:



for val in range(16, 20):
    count = 0
    title = df['title'][val]
    filename = df['filename'][val]
    for i in df['tagged_ranked'][val][0:100]:
#         print('sentence number:', count, '/100')
        count += 1
        i1=i[1].split('#')[2]
        tupd=i[1].split('#')[1].split(',')
        tup=(float(tupd[0]),float(tupd[1]))
#         print(tup)
        questions=qg.predict([i1,i1])
        for q in questions: 
            s=nlp({'question':q[0], 'context':i1 })
#             print(q)
#             print(s)
            for j in df['data'][val]:
                if tup in j.keys():
                    new_df_unclean = new_df_unclean.append({'title':title, 
                                                           'filename':filename,
                                                           'question':q[0], 
                                                           'sentence':i1, 
                                                           'summarized_paragraph':j[tup],
                                                           'sec_id':tup[0], 
                                                           'par_id':tup[1]}, ignore_index=True)
#                     q_uncleandict[q[0]]=j[tup]
                    if q[1].lower() in nltk.word_tokenize(s['answer'].lower()):
#                         q_cleandict[q[0]]=j[tup]
                        new_df_clean = new_df_clean.append({'title':title, 
                                                           'filename':filename,
                                                           'question':q[0], 
                                                           'sentence':i1, 
                                                           'summarized_paragraph':j[tup],
                                                           'sec_id':tup[0], 
                                                           'par_id':tup[1]}, ignore_index=True)
        new_df_unclean
        new_df_clean
filename = 'clean.csv'
filename1 = 'unclean.csv'
new_df_clean.to_csv(filename)
new_df_unclean.to_csv(filename1)
#     with open(filename, 'w') as fp:
#         json.dump(q_cleandict, fp)
#     with open(filename1, 'w') as fp1:
#         json.dump(q_uncleandict, fp1)
# #     final_cleandict[df['filename'][val]] = q_cleandict
#     final_uncleandict[df['filename'][val]]=q_uncleandict
# with open('cleaned_questions.json','w') as fp:
#     json.dump(q_cleandict,fp)
# with open('uncleaned_questions.json','w') as fp1:
#     json.dump(q_uncleandict,fp1)
                
        
    
    


# In[ ]:


# questions_dict
        


# In[ ]:


get_ipython().system('ls')


# In[ ]:


new_df_clean


# In[ ]:




