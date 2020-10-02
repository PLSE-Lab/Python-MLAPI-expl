#!/usr/bin/env python
# coding: utf-8

# # Exploratory - Language use and text properties
# 
# ## Objective 
# Scientific literature is usually written in English, however due to the nature of this crisis some articles are in several languages. 
# 
# The aim of the notebook is to **explore language distribution** on the dataset and generate a data attribute. 
# 
# Language may be helpful: 
#  - To focus text analysis in English documents 
#  - Validate if prioritization of English documents is filtering meaningful info. 
#  - Decide if additional effort is worth on the analysis of non-english document
#  - Help people with appropiate language skills (reading or analysis resources) to focus on part of the collection. 
# 
# ## Procedure 
#  - Use a language detection library to tag each record. 
#  - Title and abstract are used for language detection. Access to full text may only provide marginal improvement. 
#  - Some quality issues on the content of title and abstract fields were discovered.  
#     
# ## Conclusion 
#   - Most articles are in English, with small proportions in French, Spanish, German or Italian
#   - A visual inspection of results shows that results are quite accurate for Englush even if the text are not long.  
#   - Nevertheless, for those records that have no abstract and short text, English articles get tagged as other languages. 
#   - There are only three article in Chinese (abstract, title is in English)
#   - **Should Chinese be better represented? Could we missed something?**
#   - As English records are the vast majority, probably we should ignore language detection so far and work with the full set after cleaning empty records. 
# 

# ## Data Loading

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file,
                 dtype={'Microsoft Academic Paper ID': str,
                        'pubmed_id': str, 
                        "title":str,
                        "absract":str,
                        "WHO #Covidence": str})
    print(f'Loaded metadata with {len(df)} records')
    return df


# In[ ]:


METADATA_FILE = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

df = load_metadata(METADATA_FILE)


# ## Data validation and exploration

# In[ ]:


df[["title","abstract"]].head()


# We already see that some of the colums that may help to detect language have values that are not strings, like NaN or may be empty.  
# Some other records may have been incorrectly extracted, they have TOC as content (Table of Contents??). 
# So we define a helper function to filter.

# In[ ]:


def isNaN(string):
    return string != string
print('hello', isNaN('hello'))
print(np.nan,isNaN(np.nan))


# We plot title length and abstract length. Further exploration shows that some of the shorter text are incorrect extractions.

# In[ ]:


df['has_title'] = df.title.apply(lambda x: not isNaN(x) and x != 'TOC')
df['has_abstract'] = df.abstract.apply(lambda x: not isNaN(x) and x != 'TOC')


# In[ ]:


df['has_title'].value_counts()


# In[ ]:


df['has_abstract'].value_counts()


# In[ ]:


n_records = df.shape[0]
pct_has_title = df['has_title'].sum()/n_records * 100
pct_has_abstract = df['has_abstract'].sum()/n_records * 100
pct_has_title_and_abstract = df[df['has_title'] & df['has_abstract']].shape[0]/n_records * 100
pct_has_title_or_abstract = df[df['has_title'] | df['has_abstract']].shape[0]/n_records * 100

print(f"Number of records: {n_records}")
print(f"Records with title: {pct_has_title:.2f}%")
print(f"Records with abstract: {pct_has_abstract:.2f}%")
print(f"Records with both: {pct_has_title_and_abstract:.2f}%")
print(f"Records with text: {pct_has_title_or_abstract:.2f}%")


# There are a few articles that have no title (0.4%), a larger proportion that have no abstract (20%) and even some do not have text. For those, withouth text there are no text mining to use if you don't get the full text.

# In[ ]:


df['title_len'] = df["title"].apply(lambda x : len(str(x)))
df['abstract_len'] = df["abstract"].apply(lambda x : len(str(x)))


# In[ ]:


df[df.has_title].title_len.hist(bins=30)


# In[ ]:


df[df.has_abstract].abstract_len.hist()


# In[ ]:


df[df.title_len < 15][["title_len","title","abstract"]]


# In[ ]:


df[~df.has_abstract][["title_len","abstract_len","title","abstract"]]


# In[ ]:


df[(df.abstract_len > 5) & (df.abstract_len < 30) ][["title_len","abstract_len","title","abstract"]]


# **TODO**: Some more filtering on title and abstract may be required for further analysis. 
#   - Journal names 
#   - Appendix, Image, Index, Authors, Not available, Announcement  
#   
# Errors may be introduced in the metadata extraction step, but maybe no action is required. Alternatively: use stopwords. **

# ## Language detection
# 
# We start by using langtedect library and use the first predicted language. 
# 

# In[ ]:


import langdetect as ld

SEED= 53 

from langdetect import DetectorFactory
DetectorFactory.seed = SEED


# In[ ]:


def lang_detect(title, abstract):
    try:
        str_abstract = '' if (isNaN(abstract)) else abstract
        str_title = '' if (isNaN(title)) else title
        return ld.detect((str_title + ' ' + str_abstract).strip())
    except: 
        return '--'

print(lang_detect(df.iloc[0].title, 'Hola'))
print(lang_detect(df.iloc[0].title, df.iloc[0].abstract))
print(lang_detect(df.iloc[0].title, np.nan))
print(lang_detect(np.nan, df.iloc[0].title))
print(lang_detect(np.nan, np.nan))


# In[ ]:


df.head().title


# In[ ]:


df.head().apply(lambda x: lang_detect(x.title, x.abstract), axis = 1)


# In[ ]:


df['lang'] = df.apply(lambda x: lang_detect(x.title, x.abstract), axis = 1)


# ## Language Distribution
# 
# Finally, we have a look to the language distribution and some of the results to assert the classification. 
# 
# Results seemed good enough

# In[ ]:


df['lang'].value_counts().plot(kind="barh")


# In[ ]:


df['lang'].value_counts()


# In[ ]:


df['lang'].value_counts()/n_records


# Almost all articles are ain English, about 96% of them. The percentage seems increasing with time.  
# 
# Second language is French (1%) and then Spanish, German and Italian. 

# ### Exploring some results by language

# In[ ]:


show_cols = ['lang','source_x','title','abstract','publish_time','journal','WHO #Covidence']


# In[ ]:


df[df['lang']=='en'][show_cols].head()


# In[ ]:


df[df['lang']=='fr'][show_cols]


# In[ ]:


df[df['lang']=='es'][show_cols]


# There are some errors, at least when abstract is not available. 

# In[ ]:


df[df['lang']=='de'][show_cols]


# In[ ]:


df[df['lang']=='it'][show_cols]


# In[ ]:


df[df['lang']=='zh-cn'][show_cols]


# In[ ]:


df['lang'].apply(lambda x: x not in ['en','fr','es','de'])


# In[ ]:


df[df['lang'].apply(lambda x: x not in ['en','fr','es','de','it'])][show_cols]


# There are some errors, at least when abstract is not available. As far as title is the only available text and it is short some articles get tagged as other language than English. 
# 
# There are only three articles in Chinese. Chinese articles seem to have English title. 
# 
# ** Should the Chinese literature be better represented? ** I am sure that Chinese scientist do publish in Englisg, but is it possible that a relevant part of the literature could also be in Chinese, specially as they suffered the crisis first. 
# 
# Can we collect data on Chinese journal?  

# In[ ]:




