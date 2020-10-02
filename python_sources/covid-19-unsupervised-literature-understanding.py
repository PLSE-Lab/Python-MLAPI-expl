#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 Unsupervised Literature Understanding system**
# 
# ![](https://sportslogohistory.com/wp-content/uploads/2018/09/georgia_tech_yellow_jackets_1991-pres-1.png)
# 
# ***IF YOU FIND THIS USEFUL, PLEASE UPVOTE IT.***
# 
# **PROBLEM:** When a new virus is discovered and causes a pandemic, it is important for scientists to get information coming from all scientific sources that may help them combat the pandemic.  The challenege, however, is that the number of scientific papers created is large and the papers are published very rapidly, making it nearly impossible for scientists to digest and understand important data in this mass of data.
# 
# **SOLUTION:** Create an unsupervised scientific literature understanding system that can take in common terms and analyze a very large corpus of scientific papers and return highly relevant text excerpts from papers containing topical data relating to the common text inputed, allowing a single researcher or small team to gather targeted information and quickly and easily locate relevant text in the scientific papers to answer important questions about the new virus from a large corpus of documents.
# 
# **APPROACH:** The current implementation uses Pandas built in search technology to search all paper abstracts for the keywords realting to topics where specific answers are desired.  Once the dataframe slice is returned, the abstracts are then parsed into sentence and word levels to understand which of the abstracts likley contain the most relevant answers to the keyword topics.
# 
# **Enhancements:** 2020-03-23 - added NLTK stemming for search terms. 2020-03-24 - dropping duplicate rows from the dataframe
# 
# **Enhancements: (COMING SOON)** The system currenlty requires some human thought regarding the crafting of keyword combinations to return the desired information from the tasks. We are working on updates to the system that will take in natrual languge questions and provide very detailed responses.  We are working on some word adjacencey, synonym and probablisitc scoring analysis so the system will be able learn from the corpus on its own how to ferret out relevant and responsive information despite vague or incomplete natural language inputs.
# 
# **Pros:** Currently the system is a very simple **(as Einstein said "make it as simple as possible, but no simpler")**, but quite effective solutiuon to providing insight on topical queries.
# 
# **Cons:** Currently the system requires some human understanding in crafting keyword combinations and the systems brute force approach may miss relevant documents because of its very specific approach to "relevance" of document and sentence content that requires the presence of all keywords.
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df

# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])
print ('ALL CORD19 articles',df.shape)
#fill na fields
df=df.fillna('no data provided')
#drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")
#keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
#show 5 lines of the new dataframe
df=search_focus(df)


# In[ ]:


import functools
from IPython.core.display import display, HTML
from nltk import PorterStemmer

#tell the system how many sentences are needed
max_sentences=10

# function to stem keywords into a common base word
def stem_words(words):
    stemmer = PorterStemmer()
    singles=[]
    for w in words:
        singles.append(stemmer.stem(w))
    return singles

# list of lists for topic words realting to tasks
display(HTML('<h1>What is known about transmission, incubation, and environmental stability?</h1>'))
tasks = [['transmission','humidity'],['effective','reproductive','number'],['surface','persist','days'],["incubation", "period", "days"],["contagious", "incubation"],["asymptomatic","transmission"],['children'],['season'],['prevention','control'],['adhesion'],['environmental'],["comorbidities"],['disease', 'model'],['phenotypic'],['immune','response'],['movement','control'],['protective','equipment'],["blood type","type"],['smoking'],["common","symptoms"]]
# loop through the list of lists
for search_words in tasks:
    str1=''
    # a make a string of the search words to print readable search
    str1=' '.join(search_words)
    search_words=stem_words(search_words)
    # add cov to focus the search the papers and avoid unrelated documents
    search_words.append("covid-19")
    # search the dataframe for all the keywords
    dfa=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]
    search_words.pop()
    search_words.append("cov")
    dfb=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]
    # remove the cov word for sentence level analysis
    search_words.pop()
    #combine frames with COVID and cov and drop dups
    frames = [dfa, dfb]
    df1 = pd.concat(frames)
    df=df.drop_duplicates()
    
    display(HTML('<h3>Task Topic: '+str1+'</h3>'))
    # record how many sentences have been saved for display
    sentences_used=0
    # loop through the result of the dataframe search
    for index, row in df1.iterrows():
        #break apart the absracrt to sentence level
        sentences = row['abstract'].split('. ')
        #loop through the sentences of the abstract
        for sentence in sentences:
            # missing lets the system know if all the words are in the sentence
            missing=0
            #loop through the words of sentence
            for word in search_words:
                #if keyword missing change missing variable
                if word not in sentence:
                    missing=1
            # after all sentences processed show the sentences not missing keywords limit to max_sentences
            if missing==0 and sentences_used < max_sentences:
                sentences_used=sentences_used+1
                authors=row["authors"].split(" ")
                link=row['doi']
                title=row["title"]
                display(HTML('<b>'+sentence+'</b> - <i>'+title+'</i>, '+'<a href="https://doi.org/'+link+'" target=blank>'+authors[0]+' et al.</a>'))
print ("done")

