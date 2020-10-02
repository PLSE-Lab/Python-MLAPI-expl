#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 TASK - 1 transmission incubation environment 2.0**
# 
# **REVISION - 2020-03-30 code cleaned, modularized and commented**
# 
# ![](https://sportslogohistory.com/wp-content/uploads/2018/09/georgia_tech_yellow_jackets_1991-pres-1.png)
# 
# **PROBLEM:** When a new virus is discovered and causes a pandemic, it is important for scientists to get information coming from all scientific sources that may help them combat the pandemic.  The challenege, however, is that the number of scientific papers created is large and the papers are published very rapidly, making it nearly impossible for scientists to digest and understand important data in this mass of data.
# 
# **SOLUTION:** Create an unsupervised scientific literature understanding system that can take in common terms and analyze a very large corpus of scientific papers and return highly relevant text excerpts from papers containing topical data relating to the common text inputed, allowing a single researcher or small team to gather targeted information and quickly and easily locate relevant text in the scientific papers to answer important questions about the new virus from a large corpus of documents.
# 
# **APPROACH:** The newest implementation outlined:
# - open the meta CSV file in a dataframe - 44K entries
# - drop the duplicates by abstract
# - loop (comprehension) a list of lists for the keyword for searches. eg. search=[['incubation','period','range']]
# - stem the keywords so incubation becomes incubat etc.
# - append indpendently - different variations of corona references - (covid, cov, -cov, hcov) to the keywords avoids pulling older research
# - query function in pandas to query the above terms with the keywords - the abstract has to have all the keywords and at least one of (covid, cov, -cov, hcov) 
# - drop the duplicates in the df again after the queries
# - caclulate a relevance score for all the abstracts in the query result df
# - raw_score = total count of the keywords in the abstract
# - final_score = (raw_score/len(abstract))*raw_score
# - sort the df on the final_score ascending=False (best on top)
# - drop rows with a relevance score < .02
# - parse the relevant abstarcts into sentences split('. ') works well.
# - test if keywords are in the sentences
# - if sentence has all keywords, it is added to the df for display.
# - if you are seeking statistics in the data, adding % to the search terms works well
# - df with search results displayed in HTML - currently limiting results to 3 for ease of scanning
# 
# **Pros:** Currently the system is a very simple **(as Einstein said "make it as simple as possible, but no simpler")**, but quite effective solutiuon to providing insight on topical queries.  The code is easy to read and simple to understand.
# 
# **Cons:** Currently the system requires some human understanding in crafting keyword combinations, however, the recent relevance measure changes and stemming of keywords, have made it so the keywords can be very close to the NL questions and return very good results.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# keep only docsuments with covid -cov-2 and cov2
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
print (df.shape)
#drop duplicates
#df=df.drop_duplicates()
#drop NANs 
df=df.fillna('no data provided')
df = df.drop_duplicates(subset='title', keep="first")
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
#show 5 lines of the new dataframe
df=search_focus(df)
print (df.shape)
df.head()


# In[ ]:


import functools
from IPython.core.display import display, HTML
from nltk import PorterStemmer

# function to stem keyword list into a common base word
def stem_words(words):
    stemmer = PorterStemmer()
    singles=[]
    for w in words:
        singles.append(stemmer.stem(w))
    return singles

def search_dataframe(df,search_words):
    search_words=stem_words(search_words)
    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]
    return df1

# function analyze search results for relevance with word count / abstract length
def search_relevance(rel_df,search_words):
    rel_df['score']=""
    search_words=stem_words(search_words)
    for index, row in rel_df.iterrows():
        abstract = row['abstract']
        result = abstract.split()
        len_abstract=len(result)
        score=0
        for word in search_words:
            score=score+result.count(word)
        final_score=(score/len_abstract)
        rel_score=score*final_score
        rel_df.loc[index, 'score'] = rel_score
    rel_df=rel_df.sort_values(by=['score'], ascending=False)
    #rel_df= rel_df[rel_df['score'] > .01]
    return rel_df

# function to get best sentences from the search results
def get_sentences(df1,search_words):
    df_table = pd.DataFrame(columns = ["pub_date","authors","title","excerpt","rel_score"])
    search_words=stem_words(search_words)
    for index, row in df1.iterrows():
        pub_sentence=''
        sentences_used=0
        #break apart the absracrt to sentence level
        sentences = row['abstract'].split('. ')
        #loop through the sentences of the abstract
        highligts=[]
        for sentence in sentences:
            # missing lets the system know if all the words are in the sentence
            missing=0
            #loop through the words of sentence
            for word in search_words:
                #if keyword missing change missing variable
                if word not in sentence:
                    missing=1
                #if '%' in sentence:
                    #missing=missing-1
            # after all sentences processed show the sentences not missing keywords
            if missing==0 and len(sentence)<1000 and sentence!='':
                sentence=sentence.capitalize()
                if sentence[len(sentence)-1]!='.':
                    sentence=sentence+'.'
                pub_sentence=pub_sentence+'<br><br>'+sentence
        if pub_sentence!='':
            sentence=pub_sentence
            sentences_used=sentences_used+1
            authors=row["authors"].split(" ")
            link=row['doi']
            title=row["title"]
            score=row["score"]
            linka='https://doi.org/'+link
            linkb=title
            sentence='<p fontsize=tiny" align="left">'+sentence+'</p>'
            final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)
            to_append = [row['publish_time'],authors[0]+' et al.',final_link,sentence,score]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
    return df_table
    
################ main program ########################

display(HTML('<h1>Task 1: What is known about transmission, incubation, and environmental stability?</h1>'))

display(HTML('<h3>Results currently limited to two (10) for ease of scanning</h3>'))

# list of lists of search terms
questions=[
['Q: What is the range of incubation periods for the disease in humans?'],
['Q: How long are individuals are contagious?'],
['Q: How long are individuals are contagious, even after recovery.'],
['Q: Does the range of incubation period vary across age groups?'],
['Q: Does the range of incubation period vary with children?'],
['Q: Does the range of incubation period vary based on underlying health?'],
['Q: What do we know about the basic reproduction number?'],
['Q: What is the prevalance of asymptomatic transmission?'],
['Q: What do we know about asymptomatic transmission in children?'],
['Q: What do we know about seasonality of transmission?'],
['Q: Informing decontamination based on physical science of the coronavirus?'],
['Q: What do we know about stability of the virus in environmental conditions?'],
['Q: What do we know about persistence of the virus on various substrates? (e.g., nasal discharge, sputum, urine, fecal matter, blood)'],
['Q: What do we know about persistence of the virus on various surfaces? (e,g., copper, stainless steel, plastic) '],
['Q: What do we know about viral shedding duration?'],
['Q: What do we know about viral shedding in fecal/stool?'],
['Q: What do we know about viral shedding from nasopharynx?'],
['Q: What do we know about viral shedding in blood?'],
['Q: What do we know about viral shedding in urine?'],
['Q: What do we know about implemtation of diagnostics?'],
['Q: What do we know about disease models?'],
['Q: What do we know about models for disease infection?'],
['Q: What do we know about models for disease transmission?'],
['Q: Are there studies about phenotypic change?'],
['Q: What is know about adaptations (mutations) of the virus?'],
['Q: What do we know about the human immune response and immunity?'],
['Q: Is population movement control effective in stopping transmission (spread)?'],
['Q: Effectiveness of personal protective equipment (PPE)?'],
['Q: What is the role of environment in transmission?']
]
#search=[['incubation','period','range','mean','%']]
search=[['incubation','period','range'],
['viral','shedding','duration'],
['asymptomatic','shedding'],
['incubation','period','age','statistically','significant'],
['incubation','period','child'],
['incubation','groups','risk'],
['basic','reproduction', 'number','%'],
['asymptomatic','infection','%'],
['asymptomatic','children'],
['seasonal','transmission'],
['contaminat','object'],
['environmental', 'conditions'],
['sputum','stool','blood','urine'],
['persistence','surfaces'],
['duration','viral','shedding'],
['shedding','stool'],
['shedding','nasopharynx'],
['shedding','blood'],
['shedding','urine'],
['diagnostics','point'],
['model', 'disease'],        
['model', 'infection'],
['model', 'transmission'],
['phenotypic'],
['mutation'],
['immune','response'],
['restriction', 'movement'],
['protective','clothing'],
['transmission','routes']
]
q=0
for search_words in search:
    str1=''
    # a make a string of the search words to print readable version above table
    str1=' '.join(questions[q])
    
    #search the dataframe for all words
    df1=search_dataframe(df,search_words)

    # analyze search results for relevance 
    df1=search_relevance(df1,search_words)

    # get best sentences
    df_table=get_sentences(df1,search_words)
    
    length=df_table.shape[0]
    #limit 3 results
    df_table=df_table.head(15)
    df_table=df_table.drop(['rel_score'], axis=1)
    #convert df to html
    df_table=HTML(df_table.to_html(escape=False,index=False))
    
    # display search topic
    display(HTML('<h3>'+str1+'</h3>'))
    
    #display table
    if length<1:
        print ("No reliable answer could be located in the literature")
    else:
        display(df_table)
    q=q+1
print ('done')

