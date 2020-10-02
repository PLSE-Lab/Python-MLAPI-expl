#!/usr/bin/env python
# coding: utf-8

# # Here is a link to a web based version of a question answering tool.
# 
# http://edocdiscovery.com/covid_19/index.php 
# 
# You will have to fork this notebook and run in edit mode to see it work as Kaggle front end will not allow input - see error below.

# In[ ]:


import numpy as np 
import pandas as pd

# keep only docsuments with covid -cov-2 and cov2
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    frames=[dfa,dfb,dfc]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df

# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])
print (df.shape)
#drop duplicate abstracts
df = df.drop_duplicates(subset='title', keep="first")
#drop NANs 
df=df.dropna()
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()

# search focus keeps abstracts with the anchor words covid,-cov-2,hcov2
df=search_focus(df)

#show 5 lines of the new dataframe
#print (df.shape)
#df.head()
print ('data loaded')

###################### LOAD PACKAGES ##########################
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import functools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from IPython.core.display import display, HTML

stop=set(stopwords.words('english'))
stop.update(('sars-cov-2','coronavirus','covid-19','wuhan','2019-ncov','2020','2021','covid19','2019','sarscov2', 'disease covid19','acute', 'respiratory', 'syndrome','province', 'china','world','organization','emergency','novel','homeless'))


##################### FUNCTIONS ##############################

def remove_stopwords(query,stopwords):
    qstr=''
    qstr=qstr.join(query)
    #remove punctuaiton
    qstr = "".join(c for c in qstr if c not in ('!','.',',','?','(',')'))
    text_tokens = word_tokenize(qstr)
    #remove stopwords
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    #stem words
    tokens_without_sw=stem_words(tokens_without_sw)
    return tokens_without_sw

# function to stem keyword list into a common base word
def stem_words(words):
    stemmer = PorterStemmer()
    singles=[]
    for w in words:
        singles.append(stemmer.stem(w))
    return singles


# function search df abstracts for relevant ones
def search_relevance(rel_df,search_words):
    rel_df['score']=""
    search_words=stem_words(search_words)
    for index, row in rel_df.iterrows():
        abstract = row['abstract']
        result = abstract.split()
        len_abstract=len(result)
        score=0
        missing=0
        for word in search_words:
            score=score+result.count(word)
            if word not in result:
                    missing=missing+1
        missing_factor=1-(missing/len(search_words))
        final_score=(score/len_abstract)* missing_factor 
        rel_df.loc[index, 'score'] = final_score*1000000
    rel_df=rel_df.sort_values(by=['score'], ascending=False)
    return rel_df

def clean_results(raw_topics,stop):
    #remove stop words abstracts of relevant passages
    raw_topics['nostop'] = raw_topics['abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    raw_topics["nostop"] = raw_topics['nostop'].str.replace('[^\w\s]','')
    #rel_topics=Counter(" ".join(df1["nostop"]).split()).most_common(100)
    return raw_topics

def vectorize_results(df1):
    word_vectorizer = TfidfVectorizer(ngram_range=(3,4), analyzer='word')
    #word_vectorizer = CountVectorizer(ngram_range=(3,4), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(df1['nostop'])
    frequencies = sum(sparse_matrix).toarray()[0]
    ngrams=word_vectorizer.get_feature_names()
    results=pd.DataFrame(columns=['ngram','frequency'])
    results['ngram']=ngrams
    results['frequency']=frequencies
    dfa = results[results['frequency'] > .5]
    dfa=dfa.sort_values(by=['frequency'], ascending=False)
    dfa=dfa.head(0)
    return dfa

def prep_final_query(df_vectorized,query):
    addl_terms = df_vectorized["ngram"].tolist()
    # add the phrases to the query and drop duplicates
    for phrase in addl_terms:
        words=phrase.split()
        for word in words:
            query.append(word)
    query=stem_words(query)
    # drop duplicates
    final_query=[]
    final_query=[final_query.append(x) for x in query if x not in final_query] 
    return query
    
# function parse sentences for query NEED TO modularize
def get_sentences(df1,search_words):
    df_table = pd.DataFrame(columns = ["pub_date","authors","title","excerpt","sent_score"])
    for index, row in df1.iterrows():
        pub_sentence=''
        sentences_used=0
        hi_score=0
        best_sentence=''
        #break apart the absract to sentence level
        sentences = row['abstract'].split('. ')
        #loop through the sentences of the abstract
        highligts=[]
        for sentence in sentences:
            # missing lets the system know if all the words are in the sentence
            missing=0
            score=0
            missing_factor=0
            final_score=0
            #loop through the words of sentence
            for word in search_words:
                #if keyword missing change missing variable
                score=score+sentence.count(word)
                if word not in sentence:
                    missing=missing+1
            missing_factor=1-(missing/len(search_words))
            final_score=(score/len(sentence))* missing_factor
            # after all sentences processed show the sentences not missing keywords
            if len(sentence)>100 and len(sentence)<1000 and sentence!='':
                sentence=sentence.capitalize()
                if sentence[len(sentence)-1]!='.':
                    sentence=sentence+'.'
                
                if final_score>=hi_score:
                    hi_score=final_score
                    best_sentence=sentence
                    pub_sentence=sentence
        if pub_sentence!='':
            sentence=pub_sentence
            sentences_used=sentences_used+1
            authors=row["authors"].split(" ")
            link=row['doi']
            title=row["title"]
            linka='https://doi.org/'+link
            linkb=title
            sentence='<p fontsize=tiny" align="left">'+sentence+'</p>'
            final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)
            to_append = [row['publish_time'],authors[0]+' et al.',final_link,sentence,hi_score]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
        #print (hi_score)
        #print (best_sentence)
    return df_table
    

###################### MAIN PROGRAM ###########################

display(HTML('<h1>Question Answering System</h1>'))

# Loop Natural Language Questions 
#for question in questions:
while question != 'exit':
    display(HTML('<h3>Type your question below and hit enter</h3>'))
    display(HTML('<h4>enter exit to end program</h4>'))
    display(HTML('<b>For Best Answers Use Targeted Questions: e.g.<br>What is the incubation period range?<br>What is the communicable period? <br> Time duration from first positive test to clearance? <br>What type of comorbidities did patients have? <br>What is the basic reproduction number? <br> Is the virus persistent on surfaces? <br> What is the incubation period across age groups?<br>Does it spread through surface contamination?<br>What were the most contaminated surfaces and objects? <br> Are temperature and humidity a factor? etc.</b>'))
    
    question = input()
    
    #str1=''
    # a make a string of the search words to print readable version above table
    #str1=' '.join(question)
    
    # remove punctuation, stop words and stem words from NL question
    query=remove_stopwords(question,stopwords)

    # search dataframe abstracts for most relevant results
    df1=search_relevance(df,query)
    
    # clean the result abstracts
    df_clean=clean_results(df1,stop)
    
    # vectorize the abstracts and return top ngrams from corpus
    #df_vectorized=vectorize_results(df_clean)
    
    # create final query
    #final_query=prep_final_query(df_vectorized,query)
    
    # get best sentences
    df_table=get_sentences(df1,query)
    
    # sort df by sentence rank scores
    df_table=df_table.sort_values(by=['sent_score'], ascending=False)
    df_table=df_table.drop(['sent_score'], axis=1)
    
    #limit number of results
    df_table=df_table.head(10)
    
    #convert df to html
    df_table=HTML(df_table.to_html(escape=False,index=False))
    
    # display search topic
    #display(HTML('<h3>'+question+'</h3>'))
    
    # show the HTML table with responses
    display(df_table)
   
print ('done')
    

