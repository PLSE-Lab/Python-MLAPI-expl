#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation for Career Village Recommender System
# 
# by Marsh [ @vbookshelf ]<br>
# 9 April 2019

# <hr>

# ## 2. Prepare the Data

# ### 2.1. Import libraries

# In[ ]:


# Set a seed value
from numpy.random import seed
seed(101)

import pandas as pd
import numpy as np
import os

import pickle
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# ### 2.2. Check what folders are available

# In[ ]:


os.listdir('../input')


# ### 2.3. Read the Data

# In[ ]:



df_questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
df_answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')
df_professionals = pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')

df_comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')
df_tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')
df_tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

print(df_questions.shape)
print(df_answers.shape)
print(df_professionals.shape)
print(df_comments.shape)
print(df_tags.shape)
print(df_tag_users.shape)


# ### 2.4. Add the answer authors currently not listed in professionals.csv
# 
# There are 102 answer author id's in answers.csv that are not listed in professionals.csv. Here we'll add those missing id's to the data imported from professionals.csv. We'll save the updated dataframe as a pickle file. We're doing this so that we will not lose answer data when we merge dataframes later. If we don't do this then if an answer author's id is not listed in df_professionals, that person's answers will be automatically deleted when we merge the dataframes.

# In[ ]:


# Check if the answer authors listed in df_answers are listed in df_professionals

def check_if_present(x):
    prof_list = list(df_professionals['professionals_id'])
    if x in prof_list:
        return 1
    else:
        return 0
    
df_answers['check'] = df_answers['answers_author_id'].apply(check_if_present)

# Check for a mismatch

df = df_answers[df_answers['check'] == 0]
# drop duplicates in df_missing
df = df.drop_duplicates('answers_author_id')

print(len(df), ' professionals are missing from df_professionals.')


# == Insert the missing answer authors into the df_professionals dataframe == #

# select the missing answer authors
df_missing = df_answers[df_answers['check'] == 0]

# drop duplicates in df_missing
df_missing = df_missing.drop_duplicates('answers_author_id')

# select only one column
df_missing = df_missing[['answers_author_id']]

# change the id column name to be the same as df_professionals
new_names = ['professionals_id']
df_missing.columns = new_names

# drop the check column from df_missing
df_answers = df_answers.drop('check', axis=1)

# concat df_professionals and df_missing
df_professionals = pd.concat([df_professionals, df_missing], axis=0).reset_index(drop=True)

print('df_professionals has now been updated.')

# save df_professionals as a pickle file
#pickle.dump(df_professionals,open('df_professionals.pickle','wb'))

# load df_professionals
#df_professionals = pickle.load(open('df_professionals.pickle','rb'))


# In[ ]:


df_professionals.head()


# ### 2.5. Identify the high school students who have registered as professionals
# There are 26 professionals who are high school students according to their professional_headline. It could be that new student users registered under the wrong section of the website or it could be that they were once high school students but have since graduated. We won't remove these students from the list of professionals.

# In[ ]:


# replace Nan values with nothing
df_professionals = df_professionals.fillna('')

def identify_highschoolers(x):
    # convert words to lower case
    x = x.lower()
    
    if 'student at' in x:
        if 'high school' in x:
            return 1
        else:
            return 0

df_professionals['check'] = df_professionals['professionals_headline'].apply(identify_highschoolers)

# filter out the high school students
df_highschoolers = df_professionals[df_professionals['check'] == 1]

# get a list of professional id's of the high school students
highschoolers_list = list(df_highschoolers['professionals_id'])

# drop the 'check' column
df_professionals = df_professionals.drop('check', axis=1)

# print the number of highschoolers who are professionals
print(len(highschoolers_list), ' high school students are listed as professionals.')


# ### 2.6. Add a column showing how many days each professional has been part of CareerVillage

# In[ ]:


# Change the time stamp column to a pandas datetime column
df_professionals['professionals_date_joined'] = pd.to_datetime(df_professionals['professionals_date_joined'])

# Get the most recent date in the dataframe
max_date = df_professionals['professionals_date_joined'].max()

# Create a new column showing how many days that
# professional has been a member

def num_days_member(x):
    num_days = (max_date - x).days
    
    return num_days

df_professionals['num_days_member'] = df_professionals['professionals_date_joined'].apply(num_days_member)

df_professionals.head()


# ### 2.7. Merge df_questions, df_answers and df_professionals

# In[ ]:


# Merge the question and answer dataframes
df_question_answers = df_questions.merge(right=df_answers, how='inner', 
                                         left_on='questions_id', 
                                         right_on='answers_question_id')

# Merge df_question_answers with df_professionals
df_qa_prof = df_question_answers.merge(right=df_professionals, 
                                          left_on='answers_author_id', 
                                          right_on='professionals_id')


# Add the questions_title to questions_body
df_qa_prof['quest_text'] = df_qa_prof['questions_title'] + ' ' + df_qa_prof['questions_body']

# Add the professionals headline and industry to the answers body
df_qa_prof['answers_text'] = df_qa_prof['professionals_headline'] + ' ' + df_qa_prof['professionals_industry'] + ' ' + df_qa_prof['answers_body']


# Quick check: The number of rows should be the same for both dataframes.
print(df_question_answers.shape)
print(df_qa_prof.shape)


# ### 2.8. Clean the text

# In[ ]:


# replace all missing values with nothing
df_qa_prof = df_qa_prof.fillna('')


def process_text(x):
    
    # remove the hash sign
    x = x.replace("#", "")
    
    # remove the dash sign with a space
    #x = x.replace("-", " ")
    
    # Remove HTML
    x = BeautifulSoup(x).get_text()
    
    # convert words to lower case
    x = x.lower()
    
    # remove the word question
    x = x.replace("question", "")
    
    # remove the word career
    x = x.replace("career", "")
    
    # remove the word study
    x = x.replace("study", "")
    
    # remove the word student
    x = x.replace("student", "")
    
    # remove the word school
    x = x.replace("school", "")
    
    # Remove non-letters
    x = re.sub("[^a-zA-Z]"," ", x)
    
    # Remove stop words
    # Convert words to lower case and split them
    words = x.split()
    stops = stopwords.words("english")
    x_list = [w for w in words if not w in stops]
    # convert the list to a string
    x = ' '.join(x_list)
    
    return x

# clean the questions_text column
df_qa_prof['quest_text'] = df_qa_prof['quest_text'].apply(process_text)

# clean the answers_text
df_qa_prof['answers_text'] = df_qa_prof['answers_text'].apply(process_text)


# ### 2.9. Save df_professionals and df_qa_prof

# In[ ]:


# save df_professionals as a pickle file
pickle.dump(df_professionals,open('df_professionals.pickle','wb'))

# save df_qa_prof as a pickle file
pickle.dump(df_qa_prof,open('df_qa_prof.pickle','wb'))

# Code for loading df_qa_prof
# df_qa_prof = pickle.load(open('df_qa_prof.pickle','rb'))


# In[ ]:


# Check that the new file has been created
get_ipython().system('ls')


# ### 2.10. Display the df_qa_prof dataframe
# 
# This is the pre-processed dataframe that we'll use in all three models. Just a reminder that this is a merged dataframe that includes questions, answers and professionals. Professionals who didn't answer any questions are not included. Also remember that we created a new column called quest_text where each cell contains both the question title and the question body.

# In[ ]:


df_qa_prof.shape


# In[ ]:


df_qa_prof.head(3)


# In[ ]:





# In[ ]:





# In[ ]:




