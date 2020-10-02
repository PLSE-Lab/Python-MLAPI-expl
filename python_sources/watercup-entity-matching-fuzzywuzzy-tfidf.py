#!/usr/bin/env python
# coding: utf-8

# This is a inprogress notebook.
# 
# Notebook submitted in response to task#3:
# https://www.kaggle.com/bombatkarvivek/paani-foundations-satyamev-jayate-water-cup/tasks?taskId=348
# 
# Aim is to find method that will identify the correct pair of District-Taluka-Village among different datasets.

# # Pain of Entity Matching
# 
# - 'Entity Matching' is common task in most of the data engineering pipeline which joins multiple datasets.    
# - Complexity of this problem could escalate as dataset coming from different sources.  
# - While working WaterCup dataset, we realise there are quite a lot of time we have names of the places typed differently in different datasets. 
# - That leads us to creating a mapping of names manually, something like this:   
# `_df_ListOfTalukas = _df_ListOfTalukas.replace('Ahmednagar','Ahmadnagar') \ . 
#                                         .replace('Buldhana','Buldana') \ 
#                                         .replace('Sangli','Sangali') \ 
#                                         .replace('Nashik','Nasik')`
# Of course this is not way to go with bigger datasets and more granular mapping!
# - In this notebook we will try to address tbnis issue using various traditional and some non-traditional but innovative methods!

# In[ ]:


import geopandas as gpd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_mh_all_villages = gpd.read_file('../input/mh-villages-v2w2/MH_Villages v2W2.shp')[['DTNAME','GPNAME','VILLNAME']]
# ['DTNAME','GPNAME','VILLNAME']
print(df_mh_all_villages.shape)
df_mh_all_villages.T


# In[ ]:


df_ListOfTalukas = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv")
print(df_ListOfTalukas.shape)
df_ListOfTalukas.T


# ### Lets Join on District-DTNAME for all matching non-matching records. 48926 + 184
# 
# Pandas join,  
# Outer for union and inner for intersection.

# In[ ]:


df_join_district = pd.merge(df_mh_all_villages,
                            df_ListOfTalukas,
                           how='inner',
                           left_on=['DTNAME','GPNAME'],
                           right_on=['District','Taluka'],
                           indicator=True)


# In[ ]:


print(df_join_district.shape)
df_join_district.T


# In[ ]:


# df_join_district[['_merge']].nunique()
df_join_district.groupby('_merge').count()


# In[ ]:


df_StateLevelWinners = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv')
print(df_StateLevelWinners.shape)
df_StateLevelWinners.T


# In[ ]:


_df_join_villages = pd.merge(df_mh_all_villages,
                           df_StateLevelWinners,
                           left_on=['DTNAME','GPNAME','VILLNAME'],
                           right_on=['District','Taluka','Village'],
                           indicator=True)


# In[ ]:


_df_join_villages.shape


# #### No records ware found when joined on Village level!

# In[ ]:


df_join_villages = pd.merge(df_mh_all_villages,
                           df_StateLevelWinners,
                           left_on=['VILLNAME'],
                           right_on=['Village'],
                           indicator=True)


# In[ ]:


print(df_join_villages.shape)
df_join_villages.T


# In[ ]:


df_join_villages[df_join_villages['DTNAME']=='Satara']


# In[ ]:





# ## 1. Fuzzy maching

# In[ ]:


# df_fuzzy_matching = df_join_villages
# 'DTNAME','GPNAME','VILLNAME'
df_mh_all_villages['merge_entities'] = df_mh_all_villages[df_mh_all_villages.columns[:3]]                                         .apply(lambda x: ','.join(x.dropna().astype(str))
                                               ,axis=1)


# In[ ]:


df_mh_all_villages.T


# In[ ]:


df_StateLevelWinners['merge_entities'] = df_StateLevelWinners[df_StateLevelWinners.columns[:3]]                         .apply(lambda x: ','.join(x.dropna().astype(str)), axis =1)


# In[ ]:


df_StateLevelWinners.T


# [](http://)### Using fuzzywuzzy.ratio lets try to identify the matcnhing entities with matching score more than 80%.   
# Lets run this over small subset of data, i.e. District = Satara 

# In[ ]:


from fuzzywuzzy import fuzz


# In[ ]:


for ind, row1 in df_mh_all_villages[df_mh_all_villages['DTNAME']=='Satara'].iterrows():
    for ind, row2 in df_StateLevelWinners[df_StateLevelWinners['District']=='Satara'].iterrows():
        matching_ratio = fuzz.ratio(row1['merge_entities'], row2['merge_entities'])
        if matching_ratio > 80:
            print(row1['merge_entities'] + '  *  ' +
                  row2['merge_entities'] + '  :  ' + 
                  str(matching_ratio) )


# Of course above logic will explode if applied to entire dataset!  
# Below loop took good amount of time to finish. 

# In[ ]:


for ind, row1 in df_mh_all_villages.iterrows():
    for ind, row2 in df_StateLevelWinners.iterrows():
        matching_ratio = fuzz.ratio(row1['merge_entities'], row2['merge_entities'])
        if matching_ratio > 80:
            print(row1['merge_entities'] + '  *  ' +
                  row2['merge_entities'] + '  :  ' + 
                  str(matching_ratio) )


# #### As we see above we got some success in matching.

# ## 2. TF-IDF

# ### Lets try more sophosticated algorithm, TfidfVectorizer from sklearn   
# 
# Source:   
# https://colab.research.google.com/drive/1qhBwDRitrgapNhyaHGxCW8uKK5SWJblW#scrollTo=xo-X_nds97UN

# In[ ]:


import regex as re


# In[ ]:


def ngrams(string, n=3):
#     string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


org_names = df_mh_all_villages['merge_entities'].unique()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(org_names)


# In[ ]:


get_ipython().system('pip install sparse_dot_topn ')


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# clean_org_names = pd.read_excel('Gov Orgs ONS.xlsx')
# clean_org_names = clean_org_names.iloc[:, 0:6]
merged_entities_watercup = df_StateLevelWinners['merge_entities'].unique()
print('Vecorizing the data - this could take a few minutes for large datasets...')

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(merged_entities_watercup)
print('Vecorizing completed...')
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
org_column = 'merge_entities' #column to match against in the messy data
merged_entities_all_villages = set(df_mh_all_villages[org_column].values) # set used for increased performance
###matching query:
def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices
import time
t1 = time.time()
print('getting nearest n...')
distances, indices = getNearestN(merged_entities_all_villages)
t = time.time()-t1
print("COMPLETED IN:", t)
merged_entities_all_villages = list(merged_entities_all_villages) #need to convert back to a list
print('finding matches...')
matches = []
for i,j in enumerate(indices):
  temp = [round(distances[i][0],2), df_StateLevelWinners.values[j][0][5],merged_entities_all_villages[i]]
  matches.append(temp)
print('Building data frame...')  
matches = pd.DataFrame(matches, columns=['Match confidence','entities_watercup','entities_all_villages'])
print('Done')


# In[ ]:


matches.T


# ### Matching score lower is better!

# In[ ]:



matches[matches['Match confidence'] < 0.7]
# matches.query('Match confidence < 0.5')

