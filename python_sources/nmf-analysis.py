#!/usr/bin/env python
# coding: utf-8

# # NMF Analysis on News Articles 
# * contact info: madeleinecheyette@gmail.com
# * input: files of news article data that have been retokenized to include 'hillary_clinton' and 'donald_trump,' data originally from https://www.kaggle.com/snapcrack/all-the-news
# * output: files pre_lib.csv, post_lib.csv, pre_cons.csv, post_cons.csv, which contain 15 NMF topics for liberal & conservative articles before and after the 2016 election
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import gensim
import datetime
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/retokenize.csv', index_col=0)
os.system('read in data')


# **Isolate Articles for Each Publication**

# In[ ]:


NYT_id_list = df[df['publication'] == "New York Times"]["id"].tolist()

df_copy_NYT = df.copy(deep=True)

df_copy_NYT = df_copy_NYT[df_copy_NYT['id'].isin(NYT_id_list)]

print('New York Times data')
print(df_copy_NYT.tail(15))

# New York post
NYP_id_list = df[df['publication'] == "New York Post"]["id"].tolist()

df_copy_NYP = df.copy(deep=True)

df_copy_NYP = df_copy_NYP[df_copy_NYP['id'].isin(NYP_id_list)]


# Breitbart
Breitbart_id_list = df[df['publication'] == "Breitbart"]["id"].tolist()

df_copy_Breitbart = df.copy(deep=True)

df_copy_Breitbart = df_copy_Breitbart[df_copy_Breitbart['id'].isin(Breitbart_id_list)]

df_copy_Breitbart.tail(15)


#CNN
CNN_id_list = df[df['publication'] == "CNN"]["id"].tolist()

df_copy_CNN = df.copy(deep=True)

df_copy_CNN = df_copy_CNN[df_copy_CNN['id'].isin(CNN_id_list)]

df_copy_CNN.tail(15)



#Guardian
Guardian_id_list = df[df['publication'] == "Guardian"]["id"].tolist()

df_copy_Guardian = df.copy(deep=True)

df_copy_Guardian = df_copy_Guardian[df_copy_Guardian['id'].isin(Guardian_id_list)]


#Vox
Vox_id_list = df[df['publication'] == "Vox"]["id"].tolist()

df_copy_Vox = df.copy(deep=True)

df_copy_Vox = df_copy_Vox[df_copy_Vox['id'].isin(Vox_id_list)]


# In[ ]:


#BuzzFeed
Buzz_id_list = df[df['publication'] == "Buzzfeed News"]["id"].tolist()

df_copy_Buzz = df.copy(deep=True)

df_copy_Buzz = df_copy_Buzz[df_copy_Buzz['id'].isin(Buzz_id_list)]


#NationalReview
NR_id_list = df[df['publication'] == "National Review"]["id"].tolist()

df_copy_NR = df.copy(deep=True)

df_copy_NR = df_copy_NR[df_copy_NR['id'].isin(NR_id_list)]


#Fox News 
Fox_id_list = df[df['publication'] == "Fox News"]["id"].tolist()

df_copy_Fox = df.copy(deep=True)

df_copy_Fox = df_copy_Fox[df_copy_Fox['id'].isin(Fox_id_list)]

#Rueters 
Reuters_id_list = df[df['publication'] == "Reuters"]["id"].tolist()

df_copy_Reuters = df.copy(deep=True)

df_copy_Reuters = df_copy_Reuters[df_copy_Reuters['id'].isin(Reuters_id_list)]


# In[ ]:


#Atlantic
Atlantic_id_list = df[df['publication'] == "Atlantic"]["id"].tolist()

df_copy_Atlantic = df.copy(deep=True)

df_copy_Atlantic = df_copy_Atlantic[df_copy_Atlantic['id'].isin(Atlantic_id_list)]


#Atlantic
BI_id_list = df[df['publication'] == "Business Insider"]["id"].tolist()

df_copy_BI = df.copy(deep=True)

df_copy_BI = df_copy_BI[df_copy_BI['id'].isin(BI_id_list)]


#NPR
NPR_id_list = df[df['publication'] == "NPR"]["id"].tolist()

df_copy_NPR = df.copy(deep=True)

df_copy_NPR = df_copy_NPR[df_copy_NPR['id'].isin(NPR_id_list)]


#Talking Points Memo
TPM_id_list = df[df['publication'] == "Talking Points Memo"]["id"].tolist()

df_copy_TPM = df.copy(deep=True)

df_copy_TPM = df_copy_TPM[df_copy_TPM['id'].isin(TPM_id_list)]


#Talking Points Memo
WP_id_list = df[df['publication'] == "Washington Post"]["id"].tolist()

df_copy_WP = df.copy(deep=True)

df_copy_WP= df_copy_WP[df_copy_WP['id'].isin(WP_id_list)]


# **Split Content Based on 2016 Election Date **

# In[ ]:


split_date = datetime.date(2016,11,8)


# In[ ]:


#make pre-election values
pre_c1_text = df_copy_NR[(pd.to_datetime(df_copy_NR['date']) <split_date)]["content"].append(pre_NYP).append(df_copy_Breitbart[(pd.to_datetime(df_copy_Breitbart['date']) <split_date)]["content"])


pre_d1_text = df_copy_CNN[(pd.to_datetime(df_copy_CNN['date']) <split_date)]["content"].append(df_copy_Atlantic[(pd.to_datetime(df_copy_Atlantic['date']) <split_date)]["content"]).append(df_copy_NYT[(pd.to_datetime(df_copy_NYT['date']) <split_date)]["content"])

#NYP, Fox, NR
pre_c4_text =  df_copy_NYP[(pd.to_datetime(df_copy_NYP['date']) <split_date)]["content"].append(df_copy_Fox[(pd.to_datetime(df_copy_Fox['date']) <split_date)]["content"]).append(df_copy_NR[(pd.to_datetime(df_copy_NR['date']) <split_date)]["content"])
#TPM, CNN, NYT
pre_d4_text = df_copy_TPM[(pd.to_datetime(df_copy_TPM['date']) <split_date)]["content"].append(df_copy_CNN[(pd.to_datetime(df_copy_CNN['date']) <split_date)]["content"]).append(df_copy_NYT[(pd.to_datetime(df_copy_NYT['date']) <split_date)]["content"])

#TPM, NYT, Buzz
pre_d5_text = df_copy_TPM[(pd.to_datetime(df_copy_TPM['date']) <split_date)]["content"].append(df_copy_NYT[(pd.to_datetime(df_copy_NYT['date']) <split_date)]["content"]).append(df_copy_Buzz[(pd.to_datetime(df_copy_Buzz['date']) <split_date)]["content"])


# In[ ]:


#make post-election values
post_c1_text = df_copy_NR[(pd.to_datetime(df_copy_NR['date']) >split_date)]["content"].append(pre_NYP).append(df_copy_Breitbart[(pd.to_datetime(df_copy_Breitbart['date']) >split_date)]["content"])


post_d1_text = df_copy_CNN[(pd.to_datetime(df_copy_CNN['date']) >split_date)]["content"].append(df_copy_Atlantic[(pd.to_datetime(df_copy_Atlantic['date']) >split_date)]["content"]).append(df_copy_NYT[(pd.to_datetime(df_copy_NYT['date']) >split_date)]["content"])

#NYP, Fox, NR
post_c4_text =  df_copy_NYP[(pd.to_datetime(df_copy_NYP['date']) >split_date)]["content"].append(df_copy_Fox[(pd.to_datetime(df_copy_Fox['date']) >split_date)]["content"]).append(df_copy_NR[(pd.to_datetime(df_copy_NR['date']) >split_date)]["content"])

#TPM, NYT, Buzz
post_d5_text = df_copy_TPM[(pd.to_datetime(df_copy_TPM['date']) >split_date)]["content"].append(df_copy_NYT[(pd.to_datetime(df_copy_NYT['date']) >split_date)]["content"]).append(df_copy_Buzz[(pd.to_datetime(df_copy_Buzz['date']) >split_date)]["content"])


# In[ ]:


pre_cons = pre_c1_text.append(pre_c4_text)
pre_lib = pre_d1_text.append(pre_d5_text)


# In[ ]:


post_cons = post_c1_text.append(post_c4_text)
post_lib = post_d1_text.append(post_d5_text)


# **Use TFIDF Vecotrizer as Input to NMF**

# In[ ]:


vector = TfidfVectorizer(stop_words = 'english')
tfidf_post_cons = vector.fit_transform(post_cons)
terms_post_cons = vector.get_feature_names()


# In[ ]:


vector = TfidfVectorizer(stop_words = 'english')
tfidf_post_lib = vector.fit_transform(post_lib)
terms_post_lib = vector.get_feature_names()


# In[ ]:


vector = TfidfVectorizer(stop_words = 'english')
tfidf_pre_lib = vector.fit_transform(pre_lib)
terms_pre_lib = vector.get_feature_names()


# In[ ]:


vector = TfidfVectorizer(stop_words = 'english')
tfidf_pre_cons = vector.fit_transform(pre_cons)
terms_pre_cons = vector.get_feature_names()


# **NMF Analysis**

# In[ ]:


def get_nmf_topics(vectorizor, feat_names, n_top_words, num_topics):
    nmf  = NMF(n_components = num_topics)
    nmf.fit(vectorizor)
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = nmf.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);


# In[ ]:


df = get_nmf_topics(tfidf_pre_cons, terms_pre_cons, 10, 15)
df.to_csv('pre_cons.csv')


# In[ ]:


df = get_nmf_topics(tfidf_post_cons, terms_post_cons, 10, 15)
df.to_csv('post_cons.csv')


# In[ ]:


df = get_nmf_topics(tfidf_pre_lib, terms_pre_lib, 10, 15)
df.to_csv('pre_lib.csv')


# In[ ]:


df = get_nmf_topics(tfidf_post_lib, terms_post_lib, 10, 15)
df.to_csv('post_lib.csv')


# **Code for Elbow Visualizations**

# In[ ]:


# K = range(30,40)
# SSE = []
# for k in K:
#     nmf =  NMF(n_components = k)
#     nmf.fit(tfidf_pre_cons)
#     SSE.append(nmf.reconstruction_err_)
    
# import matplotlib.pyplot as plt
# plt.plot(K,SSE,'bx-')
# plt.title('Elbow Method')
# plt.xlabel('cluster numbers')
# plt.show()
# os.system('finished elbow')


# In[ ]:




