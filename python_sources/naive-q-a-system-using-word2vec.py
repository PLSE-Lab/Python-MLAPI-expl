#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# 
# I imported the clean biorxiv file from the data conversion and cleaning kernel by xhlulu. https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv.
# 
# ## Word2Vec
# Word2Vec is a method to represent words in a numerical - vector format such that words that are closely related to each other are close to each other in numeric vector space. This method was developed by Thomas Mikolov in 2013 at Google.
# 
# Each word in the corpus is modeled against surrounding words, in such a way that the surrounding words get maximum probabilities of occurence. The mapping that allows this to happen , becomes the word2vec representation of the word. The number of surrounding words can be chosen through a model parameter called "window size". The length of the vector representation is chosen using the parameter 'size'.
# 
# In this notebook, the library gensim is used to construct the word2vec models
# 
# ## Reading Comprehension
# 
# Reading comprehension is a way to answer questions with respect to the given text. This is same as the English tests we used to get back in school, where a paragraph would be given about a certain subject and related questions are asked.
# 
# One of the naive ways to answer the questions was to look at the question and to find the paragraph/sentence that closely resembled the question semantically. We are going to do that here, using word2vec representations
# 
# ## Library Load
# 
# In the following code snippet, we look at the cleaned csv data and take a random sample of 300 articles for the sake of memory.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd #
import numpy as np
import os
import re
import gensim
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

biorxiv = pd.read_csv("/kaggle/input/clean-csv/biorxiv_clean.csv")
biorxiv.shape
biorxiv.head()

biorxiv = biorxiv[['paper_id','title','text']].dropna().drop_duplicates()
pmc = pd.read_csv('/kaggle/input/clean-csv-new/clean_pmc.csv')
pmc = pmc[['paper_id','title','text']].dropna().drop_duplicates()

biorxiv = pd.concat([biorxiv,pmc]).drop_duplicates()

biorxiv = biorxiv.sample(n=300)

biorxiv.head()


# ## Sentence breakdown
# 
# As we look for answers in sentences,we break down each article into its sentence constituents.

# In[ ]:


biorxiv_split = pd.concat([pd.Series(row['paper_id'], row['text'].split('.')) for _, row in biorxiv.iterrows()]).reset_index()


# In[ ]:


biorxiv_split.columns = ['sentences','paper_id']
biorxiv_split = biorxiv_split.replace('\n','', regex=True)


# To ease text processing for english words, the spacy's english module library is used. This helps in tackling tokenization

# In[ ]:


get_ipython().system(' python -m spacy download en_core_web_sm')
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


# Before feeding the data into the word2vector (skip-gram) model, the text data is converted to a list object that is passed. The following code snippet removes stopwords, punctuations and stems words so as to remove noise.

# In[ ]:



stemmer = SnowballStemmer("english")

def text_clean_tokenize(article_data):
    
    review_lines = list()

    lines = article_data['text'].values.astype(str).tolist()

    for line in lines:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('','',string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        words = [stemmer.stem(w) for w in words]

        review_lines.append(words)
    return(review_lines)
    
    
review_lines = text_clean_tokenize(biorxiv)


# The resulting list is then passed to the `gensim.models.Word2Vec()` function. Each word is represented by a vector that is 1000 elements long.And at a time, four words surrounding the context word is used to train the model.

# In[ ]:


model =  gensim.models.Word2Vec(sentences = review_lines,
                               size=1000,
                               window=2,
                               workers=4,
                               min_count=2,
                               seed=42,
                               iter= 50)

model.save("word2vec.model")


# After the numeric vector representation of each word is obtained, these are used to create numeric representations of papers, sentence-wise. For each paper, the word2vec representations of each constituent words is found and the word2vec representation of each sentence is found by averaging.

# In[ ]:


import spacy
nlp = en_core_web_sm.load()
def tokenize(sent):
    doc = nlp.tokenizer(sent)
    return [token.lower_ for token in doc if not token.is_punct]

new_df = (biorxiv_split['sentences'].apply(tokenize).apply(pd.Series))

new_df = new_df.stack()
new_df = (new_df.reset_index(level=0)
                .set_index('level_0')
                .rename(columns={0: 'word'}))

new_df = new_df.join(biorxiv_split,how='left')

new_df = new_df[['word','paper_id','sentences']]
word_list = list(model.wv.vocab)
vectors = model.wv[word_list]
vectors_df = pd.DataFrame(vectors)
vectors_df['word'] = word_list
merged_frame = pd.merge(vectors_df, new_df, on='word')
merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby(['paper_id','sentences']).mean().reset_index()
del merged_frame
del new_df
del vectors


# ## Questions
# We get the list of questions as mentioned by the providers of this dataset. The questions are stored in a dataframe format.

# In[ ]:


questions = {
    'questions' : ["What is known about transmission, incubation, and environmental stability of COVID?",
                "What do we know about COVID risk factors?","What do we know about virus genetics, origin, and evolution of COVID?","What do we know about vaccines and therapeutics for COVID?"]
}
questions = pd.DataFrame(questions)


# After we get the questions, each question is converted to its word2vec representation.

# In[ ]:


new_df = (questions['questions'].apply(tokenize).apply(pd.Series))

new_df = new_df.stack()
new_df = (new_df.reset_index(level=0)
                .set_index('level_0')
                .rename(columns={0: 'word'}))

new_df = new_df.join(questions,how='left')

new_df = new_df[['word','questions']]
word_list = list(model.wv.vocab)
vectors = model.wv[word_list]
vectors_df = pd.DataFrame(vectors)
vectors_df['word'] = word_list
merged_frame = pd.merge(vectors_df, new_df, on='word')
question2vec = merged_frame.drop('word',axis=1).groupby(['questions']).mean().reset_index()


# For each question, the cosine similarity is calculated against all the sentences. The sentences with top 10 scores are printed as the answers.

# In[ ]:


from numpy import dot
from numpy.linalg import norm


for i in range(len(question2vec)):
    tmp = question2vec.iloc[[i]]
    tmp = tmp.drop('questions',axis=1)
    a = np.array(tmp.values)
    list_of_scores = []
    for j in range(len(merged_frame_rolled_up)):
        tmp_ = merged_frame_rolled_up.iloc[[j]]
        tmp_ = tmp_.drop(['paper_id','sentences'],axis=1)
        b = np.array(tmp_.values)
        b = b.T
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        list_of_scores.append(float(cos_sim))
    df_answer = pd.DataFrame()
    df_answer['sentence'] = merged_frame_rolled_up['sentences'].tolist()
    df_answer['scores'] = list_of_scores
    df_answer['question'] = question2vec.iloc[i]['questions']
    df_answer.sort_values(by='scores',ascending=False,inplace=True)
    print('---------------------------- \n')
    print('\n Answers for question: \n')
    print(question2vec.iloc[i]['questions'])
    print(df_answer.head(10)['sentence'].values)
        
        


# ## Changes
# * Using more advanced techniques to imitate a Q & A bot.
# * Using Decoder & Encoder methodologies to get results in form of an answer. 
# * Getting a single paragraph answer with sentences that are connected to each other.
# 
# Thanks for reading!
