#!/usr/bin/env python
# coding: utf-8

# This kernel just wants to show why luck (or repeated submissions) may play a part in getting a perfect score: 
# * some cipher texts match mutliple targets... 
# * however if you have the right chunking, you can guess sometimes guess the right target...
# * and sometimes you cannot as some identical plaintext chunks belong to different targets
# 
# Hat tip to RS Turley for all the great material he shares on this competition

# In[ ]:


import numpy as np 
import pandas as pd 

import os

from IPython.core.display import display
from sklearn.datasets import fetch_20newsgroups


# # Loading the Dataset (competition & 20newsgroup original)

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


competition_path = '20-newsgroups-ciphertext-challenge'


# In[ ]:


test = pd.read_csv('../input/' + competition_path + '/test.csv').rename(columns={'ciphertext' : 'text'})


# In[ ]:


train_p = fetch_20newsgroups(subset='train')
test_p = fetch_20newsgroups(subset='test')


# In[ ]:


df_p = pd.concat([pd.DataFrame(data = np.c_[train_p['data'], train_p['target']],
                                   columns= ['text','target']),
                      pd.DataFrame(data = np.c_[test_p['data'], test_p['target']],
                                   columns= ['text','target'])],
                     axis=0).reset_index(drop=True)


# In[ ]:


df_p['text'] = df_p['text'].map(lambda x: x.replace('\r\n','\n').replace('\r','\n').replace('\n','\n '))


# In[ ]:


df_p.loc[df_p['text'].str.endswith('\n '),'text'] = df_p.loc[df_p['text'].str.endswith('\n '),'text'].map(lambda x: x[:-1])


# In[ ]:


df_p['target'] = df_p['target'].astype(np.int8)


# # Loading the Cipher #1 Substitution Map

# In[ ]:


cipher_path = 'cipher-1-cipher-2-full-solutions'
cipher1_map = pd.read_csv('../input/'+ cipher_path + '/cipher1_map.csv')
translation_1 = str.maketrans(''.join(cipher1_map['cipher']), ''.join(cipher1_map['plain']))


# # Showing an Example Cipher in the Test Set that Matches Exactly Multiple Targets

# In[ ]:


test.loc[40]


# In[ ]:


c_text = test.loc[40,'text']


# In[ ]:


t_text = test.loc[40,'text'].translate(translation_1)


# In[ ]:


t_text


# In[ ]:


df_p.loc[[4473,7227],'text'].str.contains(t_text,regex=False)


# In[ ]:


df_p.loc[[4473,7227],'target']


# # However Using Chunked Plaintexts Point to the Right Target

# In[ ]:


df_p_extract = df_p[df_p['text'].str.contains(t_text,regex=False)]


# In[ ]:


df_p_extract


# In[ ]:


p_text_chunk_list = []
p_text_index_list = []

chunk_size = 300

for p_index, p_row in df_p_extract.iterrows():
    p_text = p_row['text']
    p_text_len = len(p_text)
    if p_text_len > chunk_size:
        for j in range(p_text_len // chunk_size):
            p_text_chunk_list.append(p_text[chunk_size*j:chunk_size*(j+1)])
            p_text_index_list.append(p_index)
        if p_text_len%chunk_size > 0:
            p_text_chunk_list.append(p_text[chunk_size*(p_text_len // chunk_size):(chunk_size*(p_text_len // chunk_size)+p_text_len%chunk_size)])
            p_text_index_list.append(p_index)
    else:
        p_text_chunk_list.append(p_text)
        p_text_index_list.append(p_index)


# In[ ]:


df_p_chunked = pd.DataFrame({'text' : p_text_chunk_list, 'p_index' : p_text_index_list})


# In[ ]:


df_p_chunked = pd.merge(df_p_chunked, df_p.reset_index().rename(columns={'index' : 'p_index'})[['p_index','target']],on='p_index',how='left')


# In[ ]:


df_p_chunked[df_p_chunked['text'].str.contains(t_text,regex=False)]


# # But in some cases we cannot do better

# In[ ]:


test.loc[31525]


# In[ ]:


t_text = test.loc[31525,'text'].translate(translation_1)


# In[ ]:


t_text


# In[ ]:


df_p_extract = df_p.loc[[11001,13188]]


# In[ ]:


p_text_chunk_list = []
p_text_index_list = []

chunk_size = 300

for p_index, p_row in df_p_extract.iterrows():
    p_text = p_row['text']
    p_text_len = len(p_text)
    if p_text_len > chunk_size:
        for j in range(p_text_len // chunk_size):
            p_text_chunk_list.append(p_text[chunk_size*j:chunk_size*(j+1)])
            p_text_index_list.append(p_index)
        if p_text_len%chunk_size > 0:
            p_text_chunk_list.append(p_text[chunk_size*(p_text_len // chunk_size):(chunk_size*(p_text_len // chunk_size)+p_text_len%chunk_size)])
            p_text_index_list.append(p_index)
    else:
        p_text_chunk_list.append(p_text)
        p_text_index_list.append(p_index)


# In[ ]:


df_p_chunked = pd.DataFrame({'text' : p_text_chunk_list, 'p_index' : p_text_index_list})


# In[ ]:


df_p_chunked = pd.merge(df_p_chunked, df_p.reset_index().rename(columns={'index' : 'p_index'})[['p_index','target']],on='p_index',how='left')


# In[ ]:


df_p_chunked[df_p_chunked['p_index'] == 11001]


# In[ ]:


df_p_chunked[df_p_chunked['p_index'] == 13188]


# In[ ]:


df_p_chunked[df_p_chunked['p_index'] == 11001].iloc[-1]['text'] == df_p_chunked[df_p_chunked['p_index'] == 13188].iloc[-1]['text']


# In[ ]:


df_p_chunked[df_p_chunked['p_index'] == 11001].iloc[-1]['text'] == t_text

