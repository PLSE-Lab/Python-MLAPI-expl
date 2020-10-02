#!/usr/bin/env python
# coding: utf-8

# **Table of Contents**
# 
# <a href='#1. Introduction'>1. Introduction</a> <br/>
# <a href='#2. Data Munging'>2. Data Munging</a> <br/>
# <a href='#3. Recommendation System #1'>3. Recommendation System #1</a> <br/>
# <a href='#4. Recommendation System #2'>4. Recommendation System #2</a> <br/>
# <a href='#5. Combine both Recommendation Systems'>5. Combine both Recommendation Systems</a> <br/>
# 

# <a id='1. Introduction'></a>
# 
# > **Introduction**
# 
# Below is our approach to solve the recommendation problem using two recommendation systems. The workflow is in the diagram below. 
# 
# ![Image](https://i.imgur.com/vqGHNfX.jpg)

# The system not only identifies the best professionals based on their answers so far, it also identifies professionals who have never answered any questions. The goal here is to encourage new professionals to be engaged in the system much more thereby contributing to the environment. 
# 
# The steps are described as follows - 
# 
# 1. When a new question comes in from a user, it is sent to the the Rec Sys 1. 
# 2. Rec Sys 1 - This system identifies all questions similar to the new question. It considers the question title, body and tags to identify similar questions already answered in the database. 
# 3. It then identifies professionals who have best answered these similar questions. The professionals are identified using the similarity score, answer score and time to answer. 
# 3. The identified professionals are then passed to another recommendation system - Rec Sys 2. This system searches in a subset of professionals who have never answered a question, and identifies the most similar in that group. 
# 4. Combine the output from both systems - a. best professionals from the group of answers and b. best professionals who have never answered a question so far
# 
# By providing a recommendation of professionals who have already answered similar question, this system gives a better chance at the question being answered quickly. However, by also providing a recommendation of professionals who haven't answered questions so far, this system provides a way to engage with those users and get them to answer more questions by leveraging instances where past professionals have answered a question. 

# **Imports and set parameters**

# In[ ]:


import pandas as pd
import pickle
import os
import numpy as np
import scipy
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
import spacy
import re
import en_core_web_sm
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords


# In[ ]:


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']


# In[ ]:


data_folder = "../input/"
professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'))
tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))
students = pd.read_csv(os.path.join(data_folder, 'students.csv'))
tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))
groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))
emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))
group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))
answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))
answer_scores = pd.read_csv(os.path.join(data_folder, 'answer_scores.csv'))
comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))
matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))
tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))
questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))
question_scores = pd.read_csv(os.path.join(data_folder, 'question_scores.csv'))
school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))


# <a></a>

# **Merge for questions dataframe**

# <a id='2. Data Munging'></a>
# > **Data Munging**
# 
# We thought it works best to combine all datasets and relevant fields into one big dataframe. There are two major data objects - Questions and Professionals. We will combine the relevant dataframes into one big **question** dataframe, and similarly do the same for the **professionals** dataframe and then merge them together for our main **features** dataframe. 

# In[ ]:


#get questions dataframe
#looking at the data, we realize that some questions are asked by professionals, 
#so we also need to get data from the professionals data frame in this context
prof_author = professionals.rename( columns = {
    'professionals_id':'author_id',
    'professionals_location': 'author_location',
    'professionals_date_joined': 'author_date_joined'
})
students_author = students.rename( columns = {
    'students_id':'author_id',
    'students_location': 'author_location',
    'students_date_joined': 'author_date_joined'
})
authors = students_author.append(prof_author, sort=True)
authors.drop(['professionals_headline', 'professionals_industry'], axis=1, inplace=True)

#join questions with students
df = pd.merge(questions, authors, how='left',
                   left_on=['questions_author_id'],
                   right_on=['author_id'])
#merge with question tags
tag_merge = pd.merge (tag_questions, tags, how='left',
                     left_on=['tag_questions_tag_id'], right_on=['tags_tag_id'])
q_tags = tag_merge.groupby('tag_questions_question_id')['tags_tag_name'].apply(lambda x: pd.unique(x.values)).rename("question_tags").reset_index()
df_2 = pd.merge(df, q_tags, how='left', left_on=['questions_id'], right_on=['tag_questions_question_id'])


# **Get Professionals dataframe**

# In[ ]:


#get professionals dataframe
#looking at the data, we realize that some questions are actually answered by Students, so we need to get them here as well
prof_answerers = professionals.rename( columns = {
    'professionals_id':'answerers_id',
    'professionals_location': 'answerers_location',
    'professionals_date_joined': 'answerers_date_joined',
    'professionals_headline' : 'answerers_headline',
    'professionals_industry' : 'answerers_industry'
})
students_answerers = students.rename( columns = {
    'students_id':'answerers_id',
    'students_location': 'answerers_location',
    'students_date_joined': 'answerers_date_joined'
})
answerers = students_answerers.append(prof_answerers, sort=True)
prof1 = pd.merge (answerers, answers, how='outer',
                     left_on=['answerers_id'], right_on=['answers_author_id'])
#merge prof tags
tag_merge = pd.merge (tag_users, tags, how='left',
                     left_on=['tag_users_tag_id'], right_on=['tags_tag_id'])
p_tags = tag_merge.groupby('tag_users_user_id')['tags_tag_name'].apply(lambda x: pd.unique(x.values)).rename("prof_tags").reset_index()

prof2 = pd.merge(prof1, p_tags, how='left', left_on=['answerers_id'], right_on=['tag_users_user_id'])

#merge group types
group_cats = pd.merge (group_memberships, groups, how='left',
                     left_on=['group_memberships_group_id'], right_on=['groups_id'])
group_cats = group_cats.groupby('group_memberships_user_id')['groups_group_type'].apply(lambda x: pd.unique(x.values)).rename("group_type").reset_index()

prof3 = pd.merge(prof2, group_cats, how='left', left_on=['answerers_id'], right_on=['group_memberships_user_id'])

#merge school counts
school_counts = school_memberships.groupby(['school_memberships_user_id']).size().reset_index(name='school_count')
prof4 = pd.merge(prof3, school_counts, how='left', left_on=['answerers_id'], right_on=['school_memberships_user_id'])


# **Join questions and answers**

# In[ ]:


#join question and answers
question_answer = pd.merge(df_2, prof4, how='left', left_on=['questions_id'], right_on=['answers_question_id'])

#join answer scores
ans_scores = answer_scores.rename(columns={'id': 'answer_scores_id', 'score':'answer_score'})
question_answer_2 = pd.merge(question_answer, ans_scores, how='left', left_on=['answers_id'], right_on=['answer_scores_id'])

#join question scores
qtn_scores = question_scores.rename(columns={'id': 'question_scores_id', 'score':'question_score'})
question_answer_3 = pd.merge(question_answer_2, qtn_scores, how='left', left_on=['questions_id'], right_on=['question_scores_id'])


# **Lets also get a list of professionals who haven't answered any questions to use in <a href='#4. Recommendation System #2'> Recommendation System #2.</a> <br/>**

# In[ ]:


#get profs who haven't answered any questions
prof_with_no_answers_1 = pd.merge (prof4, answers, how='left',
                     left_on=['answerers_id'], right_on=['answers_author_id'], indicator=True)
prof_with_no_answers = prof_with_no_answers_1[prof_with_no_answers_1.answerers_headline.notnull()]
prof_with_no_answers = prof_with_no_answers[['answerers_id', 'answerers_headline', 'answerers_industry', 'group_type', 'school_count', 'prof_tags']]
prof_with_no_answers.drop_duplicates(subset='answerers_id', inplace=True)
prof_with_no_answers.reset_index(inplace=True, drop=True)
prof_with_no_answers.head(5)


# <a id='3. Recommendation System #1'></a>
# > Recommendation System #1 
# 
# Build a model and recommendation system to get a new question and identify similar questions, and thereby professionals who best answered those similar questions. 

# In[ ]:


features = question_answer_3.copy()
#combine question title and body into one text
features['questions_full_text'] = features['questions_title'] +  " " + features['questions_body']
#Replace Nan with an empty string
features['questions_full_text'] = features['questions_full_text'].fillna('')


# **Process Steps**
# 
# 1. Pre Process data
# 2. Build Models
# 3. NLP Tokenization and identify similarity

# In[ ]:


def pre_process(data):
    # Function to pre_proces the data and convert to data works. 
    
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.compile(r'<[^>]+>').sub('', x) for x in data] #Remove HTML-tags
    
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    return data_words
    


# Standard tokenization typically breaks words into single word token and interprets meaning thereafer. 
# 
# For example - a question such as 'How do I learn Clinical Data Science' - gets split into individual words as 'Clinical', 'Data' and 'Science'. Therefore it tries and assigns this question to a Clinical Physician or a Scientist. In this instance the model should combine the relevant words into one single token such as 'Clinical-Data-Science' to assign to the appropriate professional. 
# 
# Using gensim models it combines relevant phrases comprised of two or three words pairs into single tokens, two or three word pairs to avoid the issue above.  
# 
# Parameters in the below function can be adjusted as needed. 

# In[ ]:


def build_models(data_words, min_count, threshold):
    # Function to build bigram and trigram models. 
    # Parameters: 
    # data_words - data_words returned by the pre_processing function used above
    # min_count - minimum setting threshold to create the phrase. 
    # threshold - Represent a score threshold for forming the phrases (higher means fewer phrases). 
    #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=threshold) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=threshold)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return bigram_mod, trigram_mod


# In[ ]:


def nlp_tokenization (data_words, bigram_mod, trigram_mod):
    #Function to create tokens based on the data words and models created so far
    
   # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        #https://spacy.io/api/annotation"""
        def token_filter(token):
            #Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list
            return not token.is_stop and token.is_alpha and token.pos_ in token_pos
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]
            texts_out.append([token.lemma_.lower() for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv and remove stop words
    processed_tokens = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return processed_tokens


# **Create models based on the available questions data**

# In[ ]:


#get data_words for all questions texts
data_words = pre_process(features.questions_full_text.values.tolist())
#create models using the data_words. Params can be adjusted if needed
bigram_mod_questions, trigram_mod_questions = build_models(data_words, 2, 20)
#create tokens
features['nlp_questions_tokens'] = nlp_tokenization(data_words, bigram_mod_questions, trigram_mod_questions)


# **Get similar questions**
# 
# Function to get similar questions based on a new question

# In[ ]:


def get_similar_items(features, token_list, new_text, bigram_mod, trigram_mod):
    #function to get similar items based on new_text 
    #features - dataframe to merge the final recommended list
    #token_list - list of tokens to compare the new_text
    #new_text - new item to compare
    #bigram_mod - name of the bigram model used
    #trigram_mod - name of the trigram model used
    
    #get tokens into a list
    nlp_items_corpus = [' '.join(x) for x in token_list]
    new_data_words = pre_process(new_text)
    nlp_new_item_corpus = [' '.join(x) for x in nlp_tokenization(new_data_words, bigram_mod, trigram_mod)]
    
    #vectorize the tokens
    vectorizer = TfidfVectorizer()
    vectorizer.fit(nlp_items_corpus)
    items_tfidf = vectorizer.transform(nlp_items_corpus)
    new_item_tfidf = vectorizer.transform(nlp_new_item_corpus)
    
    #find a cosine similarity between new item and the items list
    sim = cosine_similarity(items_tfidf, new_item_tfidf)
    sim_df = pd.DataFrame({'similarity': sim[:,0]})
    
    #merge with features dataset and return
    item_sim = features.join(sim_df)
    return item_sim


# **Example inference with a new question**

# In[ ]:


new_question_text = ['How do I learn clinical data science?']
similar_questions = get_similar_items(features, features['nlp_questions_tokens'], new_question_text, bigram_mod_questions, trigram_mod_questions)
#sort for items with highest similarity
item_sim = similar_questions.sort_values('similarity', ascending=False)
item_sim['similarity'].head(5)


# <a id='4. Recommendation System #2'></a>
# > **Recommendation System #2** 
# 
# Use a similar workflow as above to build models for recommendation system #2. Here we will try to find similar professional given the attributes of a professional in the system.
# 
# Please note that we are only using a subset of the data with professionals who havent answered any questions so far

# In[ ]:


#Append all text attributes into one field
features_prof = prof_with_no_answers.copy()
features_prof['prof_full_text'] = features_prof['answerers_headline'].astype(str) + " " +     features_prof['answerers_industry'].astype(str) + " " +     features_prof['group_type'].astype(str) +  " " +     features_prof ['school_count'].astype(str) +  " " +     features_prof['prof_tags'].astype(str)
features_prof['prof_full_text'] = features_prof['prof_full_text'].fillna('')


# **Build models using this dataset**

# In[ ]:


data_words = pre_process(features_prof.prof_full_text.values.tolist())
bigram_mod_prof, trigram_mod_prof = build_models(data_words, 2, 20)
features_prof['nlp_prof_tokens'] = nlp_tokenization(data_words, bigram_mod_prof, trigram_mod_prof)


# **Example inference using a new professional text**

# In[ ]:


new_prof_text = ['Enterprise Risk Management - Technology Risk Information Technology and Services nan nan information-technology-and-services leadership team-leadership career-development risk risk-management cybersecurity']
similar_profs = get_similar_items(features_prof, features_prof['nlp_prof_tokens'], new_prof_text, bigram_mod_prof, trigram_mod_prof)
item_sim = similar_profs.sort_values('similarity', ascending=False)
item_sim['similarity'].head(5)


# <a id='5. Combine both Recommendation Systems'></a>
# > **Combine both recommendation systems**
# 
# Get a list of recommended professionals from RecSys #1, and use that to identify a list of professionals using Rec sys #2

# In[ ]:


#create a new feature - time taken to answer a question
similar_questions['questions_added_datetime'] = pd.to_datetime(similar_questions['questions_date_added'], infer_datetime_format=True)
similar_questions['answers_added_datetime'] = pd.to_datetime(similar_questions['answers_date_added'], infer_datetime_format=True)
similar_questions['time_to_answer'] = similar_questions['answers_added_datetime'] - similar_questions['questions_added_datetime']
similar_questions['time_to_answer'].head(5)


# Sort the similar questions in this order to get the recommended professionals list - 
# 
# 1. Similarity - Descending
# 2. answer_score - Descending
# 3. time_to_answer - Ascending (quickest to answer)

# In[ ]:


similar_questions = similar_questions.sort_values(['similarity', 'answer_score', 'time_to_answer'], ascending=[False, False, True])
#Get Top 5 professionals. Please update this number as you see fit. 
Recommended_profs = similar_questions.head(5)


# In[ ]:


Recommended_profs['prof_full_text'] = Recommended_profs['answerers_headline'].astype(str) + " " +     Recommended_profs['answerers_industry'].astype(str) + " " +     Recommended_profs['group_type'].astype(str) +  " " +     Recommended_profs ['school_count'].astype(str) +  " " +     Recommended_profs['prof_tags'].astype(str)
Recommended_profs['prof_full_text'] = Recommended_profs['prof_full_text'].fillna('')


# In[ ]:


def get_similar_profs(new_prof_text, return_count):
    #function to get similar profs for model 2. 
    #new_prof_text - new prof text to find similarities
    #return_count - number of new professionals to return
    new_prof_list = new_prof_text.split()
    similar_profs = get_similar_items(features_prof, features_prof['nlp_prof_tokens'], new_prof_list, bigram_mod_prof, trigram_mod_prof)
    item_sim = similar_profs.sort_values('similarity', ascending=False).head(return_count)
    return item_sim


# **Run a loop on all Recommended Professionals to identify similar professionals in the not answered pool**

# In[ ]:


Recommended_not_answered_profs = pd.DataFrame()
for new_prof_text in (Recommended_profs['prof_full_text']):
    temp = get_similar_profs(new_prof_text, 5)
    Recommended_not_answered_profs = Recommended_not_answered_profs.append(temp)


# In[ ]:


#Append both dataframes to get final recommendations
Final_recs = Recommended_profs.append(Recommended_not_answered_profs, sort=False)


# In[ ]:


Final_recs[['answerers_id', 'answerers_headline', 'answerers_industry', 'group_type', 'school_count', 'prof_tags']]


# **Summary**
# 
# With the following approach we hope to accommplish two goals:
#     1. Leverage historical data to match new questions to professionals based on question similarity.
#     2. Increase the likelihood for new users to start contributing within the community by providing them questions that have a high frequency of answers based on similar professional profiles.
#     
# Lastly, we hope this approach will both allow for an efficient implementation into production quickly as well as provide a way to greatly reduce the number of false positive matches between questions and professionals.
