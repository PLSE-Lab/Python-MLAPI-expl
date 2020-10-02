#!/usr/bin/env python
# coding: utf-8

# # Mentioned drugs in the COVID-19 articles.
# 
# In this work i will try to find drugs that mentioned in all article and try to understand those drugs.
# I will use a small dataset.
# ### Update 1:
# 1-I have found another useful dataset which is richer. The source can be found below.
# 2-I checked drugs with their class and will do some wordcloud next.
# ### Update 2:
# 1- I  created a wordcloud on all sentence that mentioned about antivirals.
# 2- I merge sentences and mentioned antivirals as a single dataframe.
# 3- As a beta feature, i applied a sentiment analysis method but it will need some update due to week performance(some sentence gives no result, may be it is because sentences doesn't contain enough words to understand if the sentence postive or negative).
# ### Update 3:
# 1- I have done working on sentiment analysis of sentences of antiviral mentioned. 
# 2-I did weighted average on order by polarity-subjectivity scores including value counts of antiviral.
# 3 The same work can also be done for other drug categories.

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from tqdm.notebook import tqdm
pd.set_option('display.max_colwidth', -1)
import os


# In[ ]:


dir_list = [
    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',
    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',
    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'
]
results_list = list()
for target_dir in dir_list:
    
    print(target_dir)
    
    for json_fp in tqdm(glob(target_dir + '/*.json')):

        with open(json_fp) as json_file:
            target_json = json.load(json_file)

        data_dict = dict()
        data_dict['doc_id'] = target_json['paper_id']
        data_dict['title'] = target_json['metadata']['title']

        abstract_section = str()
        for element in target_json['abstract']:
            abstract_section += element['text'] + ' '
        data_dict['abstract'] = abstract_section

        full_text_section = str()
        for element in target_json['body_text']:
            full_text_section += element['text'] + ' '
        data_dict['full_text'] = full_text_section
        
        results_list.append(data_dict)
        
    
df_results = pd.DataFrame(results_list)
df_results.head()        


# Reading Drug dataset.

# In[ ]:


df_results.info()


# In[ ]:


dfdrugs=pd.read_csv('/kaggle/input/drug-data/drugsComTest_raw.csv')
dfdrugs.info()


# I will split every word in all articles. To join it let's call the column drugName

# In[ ]:


freq = pd.DataFrame(' '.join(df_results['full_text']).split(), columns=['drugName']).drop_duplicates()
freq.head()


# We will split all texts of all article.

# In[ ]:


result = pd.merge(freq, dfdrugs, on=['drugName'])


# In[ ]:


result.head()


# In[ ]:


result.sample(10)


# In[ ]:


result.drugName.value_counts()


# Results show us that drugs had been used for different cases and rated based on effects.

# Let's do some filter:
# * removing drugs that has lower then 7 rating

# In[ ]:


result=result.where(result['rating']>7.0).dropna()


# We have 13891 column left.

# We know that one of the main condition for Covid-19 is Cough so lets try to see drugs that been used for this purpose. I also Included headache so you can try yourself.

# In[ ]:


result.where(result['condition'].str.contains('Cough')).dropna().sample(5)
#result.where(result['condition'].str.contains('Headache')).dropna()
#result.where(result['condition'].str.contains('Cluster Headaches')).dropna()


# An example: lets see where Benzonatate mentioned in the articles.

# In[ ]:


articles=df_results['full_text'].values
for text in articles:
    for sentences in text.split('.'):
        if 'Benzonatate' in sentences:
            print(sentences)        


# Another drug called Codeine.

# In[ ]:


for text in articles:
    for sentences in text.split('.'):
        if 'Codeine' in sentences:
            print(sentences)


# I will use another dataset here, the source can be found below.

# In[ ]:


dfdrugs2=pd.read_csv('../input/usp-drug-classification/usp_drug_classification.csv')
dfdrugs2['drugName']=dfdrugs2['drug_example']
dfdrugs2.head()


# In[ ]:


result2 = pd.merge(freq, dfdrugs2, on=['drugName'])


# In[ ]:


result2.head()


# **Mentioned drugs category.**

# In[ ]:


result2['usp_category'].value_counts()[:10]


# **Antivirals**

# In[ ]:


antivirals=list(result2.drugName.where(result2['usp_category']=='Antivirals').dropna().unique())
antivirals[:5]


# An example of sentences that mention an antiviral named entecavir. Next work will be try to analyze if this drug mentioned in a positive or negative way.

# In[ ]:


for text in articles:
    for sentences in text.split('.'):
        if 'entecavir' in sentences:
            print(sentences) 


# In[ ]:


CA=list(result2.drugName.where(result2['usp_category']=='Cardiovascular Agents').dropna().unique())


#  A wordcloud on all sentences for Cardiovascular Agents.

# In[ ]:


Cardiovascular_Agents =[]
for text in articles:
    for sentences in text.split('.'):
        if any(word in sentences for word in CA):
            Cardiovascular_Agents .append(sentences)
            #print(sentences) 


# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Cardiovascular_Agents))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# A wordcloud on all sentences for antivirals.

# In[ ]:


antivirals_all=[]
for text in articles:
    for sentences in text.split('.'):
        if any(word in sentences for word in antivirals):
            antivirals_all.append(sentences)
            #print(sentences) 


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(antivirals_all))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Antivirals wordcloud&sentiment analysis.

# I matched drugs and the sentences that include that drug. Note that at same cases one sentences may include many drugs.

# In[ ]:


df = pd.DataFrame(antivirals_all, columns=['sentence']) 
def matcher(x):
    for i in antivirals:
        if i.lower() in x.lower():
            return i
    else:
        return np.nan
    
df['Match'] = df['sentence'].apply(matcher)    
df.sample(5)


# In[ ]:


df['Match'].value_counts()[:10]


# In[ ]:


df['sentence'] = df['sentence'].astype(str)
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['sentence'] = df['sentence'].str.replace('[^\w\s]','')

stop = stopwords.words('english')
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
st = PorterStemmer()
df['sentence'] = df['sentence'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

def senti(x):
    return TextBlob(x).sentiment  
 
df['senti_score'] = df['sentence'].apply(senti)


# In[ ]:


df.sample(5)


# In[ ]:


polarity=[]
subjectivity=[]
for i, j in df.senti_score:
    polarity.append(i)
    subjectivity.append(j)


# * Now finally we can see which antivirals mentioned positive with our sentiment analysis method. 

# In[ ]:


df['subjectivity']=subjectivity
df['polarity']=polarity
df.where(df['polarity']==1).dropna().head(5)


# * We will use weighted rating to balance the polarity and subjectivity then we can see which one has the best score on those values including number mentions.
# [source](http://github.com/pytmar/Python-Code-Collection/blob/master/weighted-ratings.py)

# In[ ]:


vote_data=df[['Match', 'polarity']]
items=vote_data['Match']#item's column
votes=vote_data['polarity']#vote's column
num_of_votes=len(items)
    
m=min(votes)
avg_votes_for_item=vote_data.groupby('Match')['polarity'].mean()#mean of each item's vote
mean_vote=np.mean(votes)#mean of all votes
pol=pd.DataFrame(((num_of_votes/(num_of_votes+m))*avg_votes_for_item)+((m/(num_of_votes+m))*mean_vote))
pol.head()


# In[ ]:


vote_data=df[['Match', 'subjectivity']]
items=vote_data['Match']#item's column
votes=vote_data['subjectivity']#vote's column
num_of_votes=len(items)
    
m=min(votes)
avg_votes_for_item=vote_data.groupby('Match')['subjectivity'].mean()#mean of each item's vote
mean_vote=np.mean(votes)#mean of all votes
sub=pd.DataFrame(((num_of_votes/(num_of_votes+m))*avg_votes_for_item)+((m/(num_of_votes+m))*mean_vote))
sub.head()


# In[ ]:


on_weighted_score=pd.concat([pol, sub.reindex(pol.index)], axis=1).sort_values(by=['polarity', 'subjectivity'], ascending=False)
value_count=pd.DataFrame(df['Match'].value_counts())
bests=pd.concat([value_count, on_weighted_score.reindex(value_count.index)], axis=1).sort_values(by=['polarity', 'subjectivity'], ascending=False)
bests.head(5)


# ## resources:
# * https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
# * https://www.kaggle.com/iancornish/drug-data
# * https://www.youtube.com/watch?v=S6GVXk6kbcs&lc=z23czv4rezzqspkcnacdp434abyko0xfj3zyelkza01w03c010c
# * https://www.kaggle.com/bgoss541/training-set-labeling-jump-start-umls-linking
# * https://www.kaggle.com/danofer/usp-drug-classification
# * https://data-science-blog.com/blog/2018/11/04/sentiment-analysis-using-python/

# **Final Note**:  I am not an expert on medicine just tried to do some text mining and crossing my borders on text mining. Please type down if we found any issue or have suggestion.
# Thanks.
