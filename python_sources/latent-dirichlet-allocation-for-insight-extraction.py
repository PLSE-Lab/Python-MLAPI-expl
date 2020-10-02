#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load libraries

import numpy as np 
import pandas as pd 
import json
import glob
import re
import pickle
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[ ]:


#generate lists of all the json's of all the papers in each 

biorxiv_list = glob.glob("/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*.json")
comm_list = glob.glob("/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/*.json")
noncomm_list = glob.glob("/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/*.json")
pmc_list = glob.glob("/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/*.json")


# In[ ]:


#These functions will parse the different fields in the jsons

def ParseAbstract(full_paper):
    try:
        return full_paper['abstract'][0]['text']
    except:
        return 'No Abstract'

    
def ParseRefEntries(full_paper):
    try:
        return pd.DataFrame(full_paper['ref_entries']).iloc[0].values
    except:
        return "No Entries"

def TagKeyword(jsonfile_str):
    full_paper = json.load(open(jsonfile_str, "r"))
    paper_id = full_paper['paper_id']
    title = full_paper['metadata']['title']
    abstract = ParseAbstract(full_paper)
    body = full_paper['body_text'][0]['text']
    ref_entries = ParseRefEntries(full_paper)
    bib_entries = full_paper['bib_entries']
    back_matter = full_paper['back_matter']
    #print(full_paper.keys())
    return [paper_id,title,abstract,body,ref_entries,bib_entries,back_matter]
    


# In[ ]:


#aggregate all of the papers into one np array
df = []
print('running on biorxiv')
for i in range(len(biorxiv_list)):
    df.append(TagKeyword(biorxiv_list[i]))

    
print('running on common use subset')
for i in range(len(comm_list)):
    df.append(TagKeyword(comm_list[i]))

    
print('running on noncommon use subset')
for i in range(len(noncomm_list)):
    df.append(TagKeyword(noncomm_list[i]))

    
print('running on pmc custom license')
for i in range(len(pmc_list)):
    df.append(TagKeyword(pmc_list[i]))


# In[ ]:


#convert np array to data frame, label columns
alltext_df = pd.DataFrame(df,columns=['paper_id','title','abstract','body','ref_entries','bib_entries','back_matter'])


# In[ ]:


#pickle alltext_df so that it can be easily accessed without running the code above every session
#comment out the pickle dump so it doesn't overwrite, load the pickle back to the original df
#pickle.dump(alltext_df,open( "alltext_dataframe.pkl", "wb" ))
alltext_df = pickle.load(open('alltext_dataframe.pkl','rb'))


# In[ ]:


#check if the df loaded correctly
alltext_df.head()


# Will flag keywords in corpus that will allow for brute force clustering of papers across all sources. 

# In[ ]:


#flag texts with specific keywords in the title, abstract, or body text
alltext_df['CoV_flag'] = alltext_df['title'].str.contains('CoV') | alltext_df['title'].str.contains('coronavirus', case=False) |                          alltext_df['abstract'].str.contains('CoV') | alltext_df['abstract'].str.contains('coronavirus', case=False) |                          alltext_df['body'].str.contains('CoV') | alltext_df['body'].str.contains('coronavirus', case=False)

alltext_df['SARS_flag'] = alltext_df['title'].str.contains('SARS') |                           alltext_df['abstract'].str.contains('SARS') |                           alltext_df['body'].str.contains('SARS')

alltext_df['MERS_flag'] = alltext_df['title'].str.contains('MERS') |                           alltext_df['abstract'].str.contains('MERS') |                           alltext_df['body'].str.contains('MERS')

#truncate "antibod" to include both singular and plural
alltext_df['antibody_flag'] = alltext_df['title'].str.contains('antibod') |                               alltext_df['abstract'].str.contains('antibod') |                               alltext_df['body'].str.contains('antibod')

alltext_df['vaccine_flag'] = alltext_df['title'].str.contains('vaccin') |                              alltext_df['abstract'].str.contains('vaccin') |                              alltext_df['body'].str.contains('vaccin')

alltext_df['treatment_flag'] = alltext_df['title'].str.contains('treatment') |                              alltext_df['abstract'].str.contains('treatment') |                              alltext_df['body'].str.contains('treatment')

alltext_df['cure_flag'] = alltext_df['title'].str.contains('cure') |                              alltext_df['abstract'].str.contains('cure') |                              alltext_df['body'].str.contains('cure')

alltext_df['theraputics_flag'] = alltext_df['title'].str.contains('therap') |                              alltext_df['abstract'].str.contains('therap') |                              alltext_df['body'].str.contains('therap')


# By creating a subset of texts that contain keywords of interest, we can then add certain obvious words to the stop words list, thus leaving the remaining tf-idf and LDA analysis to extract less common keywords shared amongst the subset of literature. In this way, keywords of interest can be whittled down from a larger list of keywords to see frequently discussed concepts in the subset of texts of interest. We are extracting additional concepts and keywords that are found in a subset of texts known to have an initial keyword in common that we are interested in. 

# In[ ]:


def display_topics(model, feature_names, no_top_words):  
    #this function parses the extracted keywords from the LDA model
    extracted_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        extracted_keywords.extend([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return extracted_keywords



def LDAwithExclusions(alltext_df,keyword_exclusion_list,no_features=50,no_topics=20,no_top_words=20): 
    #this function fits the input subset of text to Latent Dirichlet Allocation
    keyword_exclusion_list = text.ENGLISH_STOP_WORDS.union(keyword_exclusion_list)

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10, max_features=no_features, stop_words=keyword_exclusion_list)
    tf = tf_vectorizer.fit_transform(alltext_df[alltext_df['vaccine_flag']==True]['title'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    return display_topics(lda,tf_feature_names,no_top_words)


def FlagNewWords(alltext_df,list_of_words): 
    #this function allows for the brute force flagging and subsetting of additional keywords in order to force the model away or towards specific words
    for word in list_of_words:
        alltext_df[word+'_flag'] = alltext_df['title'].str.contains(word) |                                    alltext_df['abstract'].str.contains(word) |                                    alltext_df['body'].str.contains(word)
    return alltext_df


# e.g. below, I'm looking for texts that have to do with antibodies. At first, with no stop words, it returns topics that are relatively obvious such as "respiratory" or "protein". After iteratively running the LDAwithExclusions function more and more 'obvious' and/or unhelpful keywords are added to the stop words list and the LDA extracts less obvious keywords that the antibody subset of texts have in common.   

# In[ ]:


text_subset = alltext_df[alltext_df['antibody_flag']==1] 
keyword_exclusion_list = ['coronavirus','cov','protein','influenza','human','sars','virus','viral','vaccine',
                          'infection','antibody','neutralizing','respiratory','antibodies','immune','vaccines',
                          'infectious','responses','response','syndrome','development','dna','vaccination',
                          'immunity','protection','cell','mice','disease','specific']


# Below, counts of the keywords in each topic extracted by the LDA show the relative strength of the partner keywords.
# 

# In[ ]:


dftest = LDAwithExclusions(text_subset,keyword_exclusion_list)

keyword_counts = pd.DataFrame([[term,dftest.count(term)] for term in pd.Series(dftest).unique()],columns=['Keyword','Count']).sort_values('Count',ascending=False)
keyword_counts.head(15)


# It can be seen that 'recombinant', 'neutralization', and 'immunogenicity' are topics that are strongly related to the original topic of inquiry: 'antibodies'. This strength corresponds to how much of the literature discusses the related topic within the subset.
# 
# This process of subsetting, fitting, adding stopwords, and refitting can be iterated over by subject matter experts in order to extract concepts from the body of texts
# 
# For example, below I pivot on the concept of "recombinant" and rerun the model which extracts the concepts of things like 'antgens' and 'spike' which can inspire new pivots and subsearches within the body of literature

# In[ ]:


recombinant_subset = FlagNewWords(alltext_df,['recombinant','neutralization'])
recombinant_subset = recombinant_subset[(recombinant_subset['recombinant_flag']==1)]


# In[ ]:


keyword_exclusion_list.append('recombinant')
dftest = LDAwithExclusions(recombinant_subset,keyword_exclusion_list)

keyword_counts = pd.DataFrame([[term,dftest.count(term)] for term in pd.Series(dftest).unique()],columns=['Keyword','Count']).sort_values('Count',ascending=False)
keyword_counts.head(15)


# Here I started with the concept of "antibodies" which drew interest to "recombinant" and lead me to "immunogenicity", "antigens", "attenuated", etc. 

# In[ ]:




