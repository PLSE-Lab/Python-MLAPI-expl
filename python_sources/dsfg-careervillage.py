#!/usr/bin/env python
# coding: utf-8

# ## 1. Data Overview
# For detailed description, refer to the competetion page https://www.kaggle.com/c/data-science-for-good-careervillage.
# 
# ![data overview](https://storage.googleapis.com/kagglesdsdata/datasets/308369/627264/data_overview.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1566501804&Signature=eSwTN5SXDbNeZzPrFTnGNxRIRluacFVt3HOAdJVPoFRNxwA%2FdswsTPcQGj5umyrbP4q%2F86eiDmE446qrQydcjvFE1Pk0%2FneKQZnTkYbajyWNu7TAU2mG2sRedXR59mBaOhJzI7rs8rIF8xaV8WSAcM2etIXM%2BvXg7%2Fmq%2BMIAuDRUnlfVCoxDwJNQoMBZ91ypJnVH4s1Exi335vAWjTGsXR2LNgSKvdw00%2BlVgFSibSnzvi7qvaV4WEPEoZMjDqB3y70lqDz3VrYsf9O%2BUJ4bdFRg5BFDsL%2B0t%2FhWf3oKG2z3L1j8NwoNkeUAitqnWHX3gAe13dkQAe0ovV3J4Cca8g%3D%3D)

# ## 2. Data import

# In[ ]:


import os
import copy
import datetime
import warnings

import random
from datetime import datetime
import re

import numpy as np
from scipy.stats import t
import pandas as pd
import keras

from matplotlib import pyplot as plt
import matplotlib as mpl

from wordcloud import WordCloud

import seaborn as sns


# In[ ]:


np.random.seed(42)

DATA_PATH = '../input/data-science-for-good-careervillage/'
SPLIT_DATE = '2019-01-01'


# In[ ]:


# Read CSV
answers = pd.read_csv(DATA_PATH+'answers.csv')
answer_scores = pd.read_csv(DATA_PATH+'answer_scores.csv')
comments = pd.read_csv(DATA_PATH+'comments.csv')
emails = pd.read_csv(DATA_PATH+'emails.csv')
groups = pd.read_csv(DATA_PATH+'groups.csv')
group_memberships = pd.read_csv(DATA_PATH+ 'group_memberships.csv')
matches = pd.read_csv(DATA_PATH+ 'matches.csv')
professionals = pd.read_csv(DATA_PATH +'professionals.csv')
questions = pd.read_csv(DATA_PATH+ 'questions.csv')
question_scores = pd.read_csv(DATA_PATH+'question_scores.csv')
school_memberships = pd.read_csv(DATA_PATH +'school_memberships.csv')
students = pd.read_csv(DATA_PATH+ 'students.csv')
tags = pd.read_csv(DATA_PATH+ 'tags.csv')
tag_questions = pd.read_csv(DATA_PATH +'tag_questions.csv')
tag_users = pd.read_csv(DATA_PATH+ 'tag_users.csv')


# ## 3. Data processing

# In[ ]:


# Convert datetime format
answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)
comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'], infer_datetime_format=True)
emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'], infer_datetime_format=True)
professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'], infer_datetime_format=True)
questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)
students['students_date_joined'] = pd.to_datetime(students['students_date_joined'], infer_datetime_format=True)


# In[ ]:


# Get the first and last date of activities for each professionals and students

# Answer activity
temp = answers.groupby('answers_author_id').max()['answers_date_added']
professionals = professionals.merge(pd.DataFrame(temp.rename('last_answer')),                                    left_on='professionals_id',right_index=True,how='left')
temp = answers.groupby('answers_author_id').min()['answers_date_added']
professionals = professionals.merge(pd.DataFrame(temp.rename('first_answer')),                                    left_on='professionals_id',right_index=True,how='left')
# Question activity
temp = questions.groupby('questions_author_id').max()['questions_date_added']
students = students.merge(pd.DataFrame(temp.rename('last_question')),                                    left_on='students_id',right_index=True,how='left')
temp = questions.groupby('questions_author_id').min()['questions_date_added']
students = students.merge(pd.DataFrame(temp.rename('first_question')),                                    left_on='students_id',right_index=True,how='left')
# Comment activity
temp = comments.groupby('comments_author_id').max()['comments_date_added']
students = students.merge(pd.DataFrame(temp.rename('last_comment')),                                    left_on='students_id',right_index=True,how='left')
professionals = professionals.merge(pd.DataFrame(temp.rename('last_comment')),                                    left_on='professionals_id',right_index=True,how='left')
temp = comments.groupby('comments_author_id').min()['comments_date_added']
students = students.merge(pd.DataFrame(temp.rename('first_comment')),                                    left_on='students_id',right_index=True,how='left')
professionals = professionals.merge(pd.DataFrame(temp.rename('first_comment')),                                    left_on='professionals_id',right_index=True,how='left')


# ## 4. EDA

# ### 4.1 Data consumption overview
# * Professionals: Location, Industry, Headline, Tags, Groups, Schools, Answers, and Comments,
# * Students: Location, Tags, Groups, Schools, Questions, and Comments,

# In[ ]:


# Professionals
xTick=['Location','Industry', 'Headline', 'Tags', 'Groups', 'Schools', 'Answers', 'Comments']
xidx=range(len(xTick))
total=professionals.shape[0]
yNMiss = []
# Location
yNMiss.append(professionals['professionals_location'].count())
# Industry
yNMiss.append(professionals['professionals_industry'].count())
# Headline
yNMiss.append(professionals['professionals_headline'].count())
# Tags
temp=tag_users.groupby('tag_users_user_id').min()['tag_users_tag_id']
yNMiss.append(professionals.merge(pd.DataFrame(temp),left_on='professionals_id',                                  right_index=True,how='left')['tag_users_tag_id'].count())
# Groups
yNMiss.append(professionals.merge(group_memberships,left_on='professionals_id',                                  right_on='group_memberships_user_id',how='left')['group_memberships_group_id'].count())
# Schools
yNMiss.append(professionals.merge(school_memberships,left_on='professionals_id',                                  right_on='school_memberships_user_id',how='left')['school_memberships_school_id'].count())
# Answers
yNMiss.append(professionals['first_answer'].count())
# Comments
yNMiss.append(professionals['first_comment'].count())

# Plot
yMiss = [total-x for x in yNMiss]
p1=plt.bar(xidx,yNMiss)
p2=plt.bar(xidx,yMiss,bottom=yNMiss)
plt.xticks(xidx, xTick, rotation='vertical')
plt.legend((p1[0],p2[0]),('Existing','Missing'),bbox_to_anchor=(1,.5))
plt.show()


# In[ ]:


# Students
xTick=['Location','Tags', 'Groups', 'Schools', 'Questions', 'Comments']
xidx=range(len(xTick))
total=students.shape[0]
yNMiss = []
# Location
yNMiss.append(students['students_location'].count())
# Tags
temp=tag_users.groupby('tag_users_user_id').min()['tag_users_tag_id']
yNMiss.append(students.merge(pd.DataFrame(temp),left_on='students_id',                                  right_index=True,how='left')['tag_users_tag_id'].count())
# Groups
yNMiss.append(students.merge(group_memberships,left_on='students_id',                                  right_on='group_memberships_user_id',how='left')['group_memberships_group_id'].count())
# Schools
yNMiss.append(students.merge(school_memberships,left_on='students_id',                                  right_on='school_memberships_user_id',how='left')['school_memberships_school_id'].count())
# Answers
yNMiss.append(students['first_question'].count())
# Comments
yNMiss.append(students['first_comment'].count())

# Plot
yMiss = [total-x for x in yNMiss]
p1=plt.bar(xidx,yNMiss)
p2=plt.bar(xidx,yMiss,bottom=yNMiss)
plt.xticks(xidx, xTick, rotation='vertical')
plt.legend((p1[0],p2[0]),('Existing','Missing'),bbox_to_anchor=(1,.5))
plt.show()


# ### 4.2 Locations distribution
# 

# In[ ]:


# Professionals
professionals['professionals_location'].value_counts(ascending=True).tail(30).plot.barh()
plt.show()


# In[ ]:


# Students
students['students_location'].value_counts(ascending=True).tail(30).plot.barh()
plt.show()


# ### 4.3 Users registered per year
# 

# In[ ]:


pd.DataFrame({'Prof':professionals['professionals_date_joined'].dt.year.value_counts().sort_index(),              'Stu':students['students_date_joined'].dt.year.value_counts().sort_index()}).plot()
plt.show()


# ### 4.4 Questions & answers posted per year

# In[ ]:


pd.DataFrame({'Q':questions['questions_date_added'].dt.year.value_counts().sort_index(),              'A':answers['answers_date_added'].dt.year.value_counts().sort_index()}).plot()
plt.show()


# ### 4.5 Student-Questions & Professional-Answers count distributions

# In[ ]:


# S-Q distribution
questions['questions_author_id'].value_counts().hist(bins=range(1,25),grid=False)
plt.xlabel('Question #')
plt.ylabel('Student #')
plt.show()


# In[ ]:


# P-A distribution
answers['answers_author_id'].value_counts().hist(bins=range(1,50),grid=False)
plt.xlabel('Answer #')
plt.ylabel('Prof #')
plt.show()


# ### 4.6 Question-Answer cont distribution

# In[ ]:


answers['answers_question_id'].value_counts().hist(bins=range(0,20),grid=False)
plt.xlabel('Answer #')
plt.ylabel('Question #')
plt.show()


# ### 4.7 Top tags in questions

# In[ ]:


tag_questions.merge(tags,left_on='tag_questions_tag_id',right_on='tags_tag_id',how='left')['tags_tag_name']            .value_counts(ascending=True).tail(20).plot.barh()
plt.show()


# ### 4.8 Top tags in students

# In[ ]:


tag_users[tag_users['tag_users_user_id'].isin(students['students_id'])]        .merge(tags,left_on='tag_users_tag_id',right_on='tags_tag_id',how='left')['tags_tag_name']        .value_counts(ascending=True).tail(20).plot.barh()
plt.show()


# ### 4.9 Top tags in professionals

# In[ ]:


tag_users[tag_users['tag_users_user_id'].isin(professionals['professionals_id'])]        .merge(tags,left_on='tag_users_tag_id',right_on='tags_tag_id',how='left')['tags_tag_name']        .value_counts(ascending=True).tail(20).plot.barh()
plt.show()


# ### 4.10 Average question response days per year

# In[ ]:


afirst = answers.groupby('answers_question_id').min()['answers_date_added'].rename('firstA')
alast = answers.groupby('answers_question_id').max()['answers_date_added'].rename('lastA')
temp = questions.merge(pd.DataFrame(afirst),left_on='questions_id',right_index=True,how='left')            .merge(pd.DataFrame(alast),left_on='questions_id',right_index=True,how='left')[['questions_date_added','firstA','lastA']]
temp['firstA']=(temp['firstA']-temp['questions_date_added']).dt.days
temp['lastA']=(temp['lastA']-temp['questions_date_added']).dt.days
temp['year']=temp['questions_date_added'].dt.year
temp.groupby('year').mean().plot()
plt.ylabel('Days')
plt.show()


# ### 4.11 Emails per year

# In[ ]:


emails['emails_date_sent'].dt.year.value_counts().sort_index().plot()
plt.show()


# ### 4.12 Questions-email distribution

# In[ ]:


matches.groupby('matches_email_id').size().hist(bins=range(15),grid=False)
plt.xlabel('Question #')
plt.ylabel('Email #')
plt.show()


# ## 5. NLP - Content Similarity Filtering
# 
# Build a LDA model on the full question text (headline+body). Given a new question, find the most similar question and get the professionals who answered that question. 

# ### 5.1 Process training data

# In[ ]:


import spacy


# In[ ]:


# Create full text for questions
testData = questions.head(100)
testData['full_txt'] = testData['questions_title']+' '+testData['questions_body']
testData['full_txt']= testData['full_txt'].str.replace('#',' ')

# Apply spaCy part-of-speech to tokenize the text
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']
nlp = spacy.load("en_core_web_sm")
dpipe = nlp.pipe(testData['full_txt'],disable=["parser","ner"])
tokens=[]
for doc in dpipe:
    tokens.append([t.lower_ for t in doc if (t.pos_ in token_pos and not t.is_stop and t.is_alpha)])


# ### 5.2 Train LDA model

# In[ ]:


import gensim


# In[ ]:


# Create Gensim dic from the tokens
no_below = 20
no_above = 0.6
keep_n = 8000
lda_dic = gensim.corpora.Dictionary(tokens)
##lda_dic.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

# Create corpus from the dic
lda_corpus = [lda_dic.doc2bow(doc) for doc in tokens]

# Create tf-idf from corpus
tfidf = gensim.models.TfidfModel(lda_corpus)
tfidf_corpus = tfidf[lda_corpus]

# Train model
num_topics = 19
passes = 15
chunksize = 1000
alpha = 1/50
seed=13
lda_model = gensim.models.ldamodel.LdaModel(tfidf_corpus, num_topics=num_topics,                                             id2word = lda_dic, passes=passes,                                            chunksize=chunksize,update_every=0,                                            alpha=alpha, random_state=seed)


# ### 5.3 Find similar questions and corresponding professionals

# In[ ]:


# Convert the query to LDA space
doc = "I would love to be a civil engineer one day"
vec_bow = lda_dic.doc2bow(doc.lower().split())
vec_corpus = tfidf[vec_bow]
vec_lda = lda_model[vec_corpus] 

# Transform tfidf corpus to LDA space and index it
index = gensim.similarities.MatrixSimilarity(lda_model[tfidf_corpus])

# Perform a similarity query against the tfidf corpus
sims = index[vec_lda]  
sims = sorted(enumerate(sims), key=lambda item: -item[1])

# Get the most similar question
q_id = testData['questions_id'][sims[0][0]]
print(testData.iloc[sims[0][0]])
print(testData['full_txt'][sims[0][0]])

# Get the professionals
prof_id = answers['answers_author_id'][answers['answers_question_id']==q_id]
print(prof_id)


# ## 6. Collaberative Filtering
# 
# The approach above can find most-likely professaionals who had answered similar questions. We can further find more professaionals who behave similarly as the most-likely ones, though they haven't answered any of the top similar questions. 

# ### 6.1 Create question-professional matrix

# In[ ]:


from scipy.sparse import csr_matrix

# Reindex professionals and questions with integer
temp=answers.merge(professionals,how='inner',left_on='answers_author_id',right_on='professionals_id')
temp=temp.merge(questions,how='inner',left_on='answers_question_id',right_on='questions_id')
temp['question'] = temp['questions_id']                        .apply(lambda x : np.argwhere(questions['questions_id'] == x)[0][0])
temp['prof'] = temp['professionals_id']                        .apply(lambda x : np.argwhere(professionals['professionals_id'] == x)[0][0])


# Create a sparse matrix for co-occurences
occurences = csr_matrix((questions.shape[0], professionals.shape[0]), dtype='int8')
def set_occurences(q, p):
    occurences[q, p] += 1
temp.apply(lambda row: set_occurences(row['question'], row['prof']), axis=1)
occurences


# ### 6.2 Construct prof-prof matrix
# A co-occurrence matrix, the element of which indicates how many times two professionals answered the same question

# In[ ]:


cooc = occurences.transpose().dot(occurences)
cooc.setdiag(0)


# ### 6.3 Log-Likelihood Ratio (LLR)
# The LLR function is computing the likelihood of two events, A and B appear together.
# 
# k11, number of when both events appeared together
# 
# k12, number of B appear without A
# 
# k21, number of A appear without B
# 
# k22, number of other things appeared without both of them
# 
# 

# In[ ]:


# Functions to compute LLR
def xLogX(x):
    return x * np.log(x) if x != 0 else 0.0
def entropy(x1, x2=0, x3=0, x4=0):
    return xLogX(x1 + x2 + x3 + x4) - xLogX(x1) - xLogX(x2) - xLogX(x3) - xLogX(x4)
def LLR(k11, k12, k21, k22):
    rowEntropy = entropy(k11 + k12, k21 + k22)
    columnEntropy = entropy(k11 + k21, k12 + k22)
    matrixEntropy = entropy(k11, k12, k21, k22)
    if rowEntropy + columnEntropy < matrixEntropy:
        return 0.0
    return 2.0 * (rowEntropy + columnEntropy - matrixEntropy)
def rootLLR(k11, k12, k21, k22):
    llr = LLR(k11, k12, k21, k22)
    sqrt = np.sqrt(llr)
    if k11 * 1.0 / (k11 + k12) < k21 * 1.0 / (k21 + k22):
        sqrt = -sqrt
    return sqrt

# Compute LLR
row_sum = np.sum(cooc, axis=0).A.flatten()
column_sum = np.sum(cooc, axis=1).A.flatten()
total = np.sum(row_sum, axis=0)
pp_score = csr_matrix((cooc.shape[0], cooc.shape[1]), dtype='double')
cx = cooc.tocoo()
for i,j,v in zip(cx.row, cx.col, cx.data):
    if v != 0:
        k11 = v
        k12 = row_sum[i] - k11
        k21 = column_sum[j] - k11
        k22 = total - k11 - k12 - k21
        pp_score[i,j] = rootLLR(k11, k12, k21, k22)
        
result = np.flip(np.sort(pp_score.A, axis=1), axis=1)
result_indices = np.flip(np.argsort(pp_score.A, axis=1), axis=1)


# ### 6.4 Find similar professionals

# In[ ]:


# Use the professional id from 5.3
x = prof_id.iloc[0]
prof_num = np.argwhere(professionals['professionals_id'] == x)
# Find similar professionals
print(result[prof_num][0][0])
print(result_indices[prof_num][0][0])
# Display the most similar professional id
print(professionals.loc[result_indices[prof_num][0][0][0],'professionals_id'])

