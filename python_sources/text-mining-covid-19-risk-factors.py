#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re 
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 
# ## Helper Functions

# In[ ]:


def df_nan_filter(df, filter_cols):
    row_nans_mask = df[filter_cols].isnull().any(axis=1)
    dropped_rows = df.loc[row_nans_mask]
    filtered_df = df.loc[~row_nans_mask]
    return filtered_df, dropped_rows


# ### Info about json schema

# In[ ]:


with open('/kaggle/input/CORD-19-research-challenge/json_schema.txt', 'r') as f:
    print(f.read())


# ### Reading the license agreement

# In[ ]:


#Uncomment if online mode is turned on
get_ipython().system('pip install tika')


# In[ ]:


# Tika is useful for reading PDFs in Python
from tika import parser
raw = parser.from_file('/kaggle/input/CORD-19-research-challenge/COVID.DATA.LIC.AGMT.pdf')
print(raw['content'])


# # Exploratory Data Analysis (EDA) on Meta Data

# ### Reading about Metadata

# In[ ]:


with open('/kaggle/input/CORD-19-research-challenge/metadata.readme', 'r') as f:
    print(f.read())


# **Sources:**
# * CZI: Chan Zuckerberg Initiative, a company aimed to "advance human potential and promote equality in areas such as health, education, scientific research and energy".
# * PMC: PubMed Central, a free full-text archive of biomedical and life sciences journal literature at the U.S. National Institutes of Health's National Library of Medicine (NIH/NLM).
# * bioRxiv: an open access preprint repository for the biological sciences. As preprints, papers hosted on bioRxiv are not peer-reviewed, but undergo basic screening and checked against plagiarism.
# * medRxiv: a preprint service for the medicine and health sciences and provides a free online platform for researchers to share, comment, and receive feedback on their work. 
# 

# In[ ]:


meta_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv') 


# In[ ]:


meta_df.shape


# In[ ]:


meta_df.columns


# In[ ]:


meta_df.isnull().sum()


# In[ ]:


meta_df, dropped_rows = df_nan_filter(meta_df, ['title'])


# In[ ]:


meta_df.shape


# In[ ]:


meta_df['title']


# In[ ]:


# Look at the distribution of article titles in terms of total count of each
meta_df['title'].value_counts()


# ### Functions used to find most commonly used words in article titles taken from https://www.kaggle.com/paultimothymooney/most-common-words-in-the-cord-19-dataset

# In[ ]:


def count_ngrams(df,column,begin_ngram,end_ngram):
    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe
    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(df['title'].dropna())
    frequencies = sum(sparse_matrix).toarray()[0]
    most_common = pd.DataFrame(frequencies, 
                               index=word_vectorizer.get_feature_names(), 
                               columns=['frequency']).sort_values('frequency',ascending=False)
    most_common['ngram'] = most_common.index
    most_common.reset_index()
    return most_common

def word_cloud_function(df,column,number_of_words):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=number_of_words,
                          width=1000,height=1000,
                         ).generate(word_string)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def word_bar_graph_function(df,column,title):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    plt.show()


# ### Most Common Words in Titles

# In[ ]:


plt.figure(figsize=(10,10))
word_bar_graph_function(meta_df,'title','Most common words in titles of papers in CORD-19 dataset')


# ### Most Common Journals

# In[ ]:


value_counts = meta_df['journal'].value_counts()
value_counts_df = pd.DataFrame(value_counts)
value_counts_df['journal_name'] = value_counts_df.index
value_counts_df['count'] = value_counts_df['journal']
fig = px.bar(value_counts_df[0:20], 
             x="count", 
             y="journal_name",
             title='Most Common Journals in CORD-19 Dataset',
             orientation='h')
fig.show()


# ### Most Common Dates of Publication

# In[ ]:


value_counts = meta_df['publish_time'].value_counts()
value_counts_df = pd.DataFrame(value_counts)
value_counts_df['which_year'] = value_counts_df.index
value_counts_df['count'] = value_counts_df['publish_time']
fig = px.bar(value_counts_df[0:5], 
             x="count", 
             y="which_year",
             title='Most Common Dates of Publication',
             orientation='h')
fig.show()


# In[ ]:


# Idea is to gather all most popular articles that have most popular words, from the most popular journal from the most popular year, with the most 
# common n-gram phrase and then data mine these articles (and then extend this to top 3 words/journals/years/etc?)


# In[ ]:


# Observe value distribution of publish times
meta_df['publish_time'].value_counts()


# In[ ]:


# Observe value distribution of journals
meta_df['journal'].value_counts()


# ### Let's see if PLoS One articles can be found in the set of articles that contain 'virus' in the title and that were published in 2020...

# In[ ]:


# Create a DataFrame where title and publish_time are the most popular values
virus_2020_df = meta_df.loc[(meta_df['title'].str.contains("virus")) & (meta_df['publish_time'] == '2020')]


# In[ ]:


virus_2020_df['journal'].value_counts()


# ### Ok it looks like none of the top journals are PLoS One... do any articles come from this journal, though?

# In[ ]:


virus_2020_df.loc[virus_2020_df['journal'] == 'PLoS One']


# ### Looks like PLoS One is no where to be found in this subset of data...

# # Data on potential risks factors
# 

# In[ ]:


biorxiv_df = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")


# In[ ]:


biorxiv_df.shape


# In[ ]:


clean_commerical_df = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")


# In[ ]:


clean_commerical_df.shape


# In[ ]:


clean_noncommerical_df = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")


# In[ ]:


clean_noncommerical_df.shape


# In[ ]:


clean_pmc_df = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")


# In[ ]:


clean_pmc_df.shape


# Since we don't care about which of these sources the risk factor info comes from... I will combine all of these data sources to analyze (since columns are the same).

# In[ ]:


clean_df = pd.concat([biorxiv_df, clean_commerical_df, clean_noncommerical_df, clean_pmc_df])


# In[ ]:


clean_df.shape


# In[ ]:


clean_df.columns


# ### See which columns have NaNs and then remove the NaNs if the column is of interest

# In[ ]:


clean_df.isnull().sum()


# Let's remove all rows with NaNs in the **Abstract** column so that we can search for information on risk factors (i.e. smoking) in the abstracts

# In[ ]:


clean_abstract_df, abstract_dropped_rows = df_nan_filter(clean_df, ['abstract'])


# In[ ]:


clean_abstract_df.shape


# In[ ]:


smoking_abstracts_df = clean_abstract_df.loc[clean_abstract_df['abstract'].str.contains("smoking")].reset_index(drop=True)


# In[ ]:


smoking_abstracts_df


# In[ ]:


smoking_abstract_text = ''.join(str(elem) for elem in list(smoking_abstracts_df['abstract']))


# In[ ]:


smoking_abstract_text


# Using RegEx to find all sentences that contain the word 'Smoking'

# In[ ]:


regExSmokingAbstractResults = re.findall(r"([^.]*?smoking[^.]*\.)",smoking_abstract_text)    


# In[ ]:


regExSmokingAbstractResults


# In[ ]:


regExSmokingAbstractResultsText = ''.join(str(elem) for elem in regExSmokingAbstractResults)
regExSmokingAbstractResultsText


# In[ ]:





# # Using Pytextrank to summarize text about smoking (taken from https://medium.com/@aneesha/beyond-bag-of-words-using-pytextrank-to-find-phrases-and-summarize-text-f736fa3773c5)

# In[ ]:


#Uncomment if online mode is turned on
get_ipython().system('pip install pytextrank')


# In[ ]:


import pytextrank


# In[ ]:


#spaCy is a library for advanced Natural Language Processing in Python and Cython. It's built on the very latest research, and was designed from day one to be used 
# in real products. spaCy comes with pretrained statistical models and word vectors, and currently supports tokenization for 50+ languages.
import spacy
# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
nlp


# In[ ]:


# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)


# In[ ]:


doc = nlp(regExSmokingAbstractResultsText)
smoking_strings = [x for x in doc._.phrases if 'smoking' in str(x)] 


# In[ ]:


get_ipython().system('pip install git+https://github.com/boudinfl/pke.git')


# # Using PKE (https://github.com/boudinfl/pke) Paper = http://aclweb.org/anthology/C16-2015

# In[ ]:


import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy
extractor.load_document(input=regExSmokingAbstractResultsText, language='en')

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()


# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)


# The topics listed above demonstrate that smoking status (most likely how often someone smokes) is considered an 'important topic' in the article abstracts, which are focused on understanding COVID-19.

# # Trying to use Rake (https://pypi.org/project/rake-nltk/) from the natural language toolkit to find key phrases

# In[ ]:


get_ipython().system('pip install rake-nltk')


# In[ ]:


from rake_nltk import Rake

rAbstract = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

rAbstract.extract_keywords_from_text(regExSmokingAbstractResultsText)

rAbstract.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.


# In[ ]:


abstractRankedPhrases = rAbstract.get_ranked_phrases()


# In[ ]:


for phrase in abstractRankedPhrases:
    if 'smoking' in phrase:
        print(phrase)


# In[ ]:


from rake_nltk import Rake

rMainText = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

rMainText.extract_keywords_from_text(''.join(str(elem) for elem in re.findall(r"([^.]*?smoking[^.]*\.)",''.join(str(elem) for elem in list(clean_abstract_df['text'])))))

rMainText.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.

maintextRankedPhrases = rMainText.get_ranked_phrases()

for phrase in maintextRankedPhrases:
    if 'smoking' in phrase:
        print(phrase)


# As we can see, certain risk factors, based on the text mining results, most likely include: 
# * tension 
# * smoking 
# * alcohol 
# * salt 
# * animal fats 
# * body weight 
# 

# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




