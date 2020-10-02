#!/usr/bin/env python
# coding: utf-8

# **Updated Version **
# 
# Some lessons learned since creating this kernel:
# 
# 1. This solution relies on hashtags being present in the student's question. If a hashtag is not present then this solution will fail. Some questions don't have hashtags.
# 2. There are 102 professional id's that are listed in answers.csv which are not included in professionals.csv
# 3. When merging questions.csv, answers.csv and professionals.csv - only professionals who have answered questions will be present in the merged dataframes. Those professionals who did not answer questions are automatically dropped.

# <hr>

# ## What is the objective?
# 
# > ### Develop a method to recommend relevant questions to the professionals who are most likely to answer them.

# ## Summary
# 
# We will start by creating a dataframe containing a list of all professionals who have answered questions in the past. This dataframe will inculde the hash tags that were part of the questions that they answered. It will also include the total number of questions they have answered. We will use this info to match professionals to questions and identify professionals who are both able and likely to answer those questions. This is a vintage solution. We won't be using machine learning.

# <hr>

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


# What files are available?
os.listdir('../input')


# 

# ## Section 1
# The goal of this section is to create a dataframe containing the id's of all professionals that have ever answered questions. This dataframe will have a column containing all the hash tags that were part of the questions each professional answered.

# ### Merge questions.csv and answers.csv

# In[ ]:


df_questions = pd.read_csv('../input/questions.csv')
df_answers = pd.read_csv('../input/answers.csv')

print(df_questions.shape)
print(df_answers.shape)


# In[ ]:


# source: https://www.kaggle.com/crawford/starter-kernel

df_question_answers = df_questions.merge(right=df_answers, how='inner', 
                                         left_on='questions_id', 
                                         right_on='answers_question_id')

df_question_answers.shape


# In[ ]:


df_question_answers.head()


# ### Extract the hash tags from the questions

# In[ ]:


def extract_hashtags(x):
    # split the string into a list
    a = x.split()
    # extract the hashtags
    hash_tags = [i for i in a if i.startswith("#")]
    
    # Convert the list to a string with a space at the end.
    # This space is needed when we sum the strings later.
    result = ' '.join(hash_tags) + " "
    
    return result

# create a new column containing the hash tags
df_question_answers['question_hash_tags'] = df_question_answers['questions_body'].apply(extract_hashtags)

df_question_answers.head()


# In[ ]:


# Select only the columns we need
df_answers = df_question_answers[['answers_author_id', 'question_hash_tags']]

# Create a new dataframe where we will store the total number of questions answered
df_count = df_answers.copy()

# Create an additional column that will show how many questions a professional has answered.
# The total will appear after we run pandas "Groupby" below.
df_count['num_questions_answered'] = 1


# In[ ]:


# Get the number of questions that each professional has answered
df_count = df_count.groupby('answers_author_id').sum()

df_count.head()


# In[ ]:


# Groupby the answers_author_id column and sum.
# This is essentially adding (+) the strings associated with each professional id.

df_prof_tags = df_answers.groupby('answers_author_id').sum()

# Add the num_questions_answered column to df_prof_tags
df_prof_tags['num_questions_answered'] = df_count['num_questions_answered']

# reset the index
df_prof_tags.reset_index(inplace=True)


df_prof_tags.head()


#  This is the final dataframe containing a list of professionals that have previously answered questions and the hash tags that were part of the questions that they answered. The total number of questions that each professional answered is also shown.

# In[ ]:


df_prof_tags.head()


# ## Section 2
# In this section we will use the above df_prof_tags dataframe to generate a list of professionals to whom a notification should be sent when a student asks a question on a specific subject.

# ### Let's consider the following question that was asked by Priyanka, a high school student from Bangalore:

# In[ ]:


df_questions = pd.read_csv('../input/questions.csv')
question = df_questions.loc[1,'questions_body']

question


# Note that she has tagged this question:
# - #military
# - #army

# In[ ]:


# 1. Extract the hash tags from the question

def extract_hashtags(question):
    a = question.split()
    # extract the hashtags
    quest_tags = [i for i in a if i.startswith("#")]
    
    return quest_tags

# call the function
quest_tags = extract_hashtags(question)

quest_tags


# In[ ]:


# 2. Filter out all professionals who have previously answered questions with
# at least one of those hash tags.

df_prof = df_prof_tags.copy()

# Return 1 if a professional has anawered a question with those tags
def filter_on_tag(x):    
        if x != 1:
            if item in x:
                return 1
            else:
                return x

for item in quest_tags:
    print(item)
    df_prof['tag_found'] = df_prof['question_hash_tags'].apply(filter_on_tag)
    
# filter out rows where the 'tag_found' column has value 1
df_filtered = df_prof[df_prof['tag_found'] == 1]

df_filtered.shape


# We see that there are 68 professionals that have previously answered questions that contained at least one these tags.

# In[ ]:


df_filtered.head()


# Let's reduce this number by considering only those professionals who have answered more than 50 questions. This helps us to identify those professionals who are likely to respond to an email notification. The fact that they have answered many questions in the past is a possible indicator that:<br>
# (i) they are keen on helping students<br>
# (ii) they are willing to take time to write answers<br>
# (iii) they have experience
# 

# In[ ]:


df_filtered = df_filtered[df_filtered['num_questions_answered'] > 50]

# Check the number of professionals identified
df_filtered.shape


# We now have a list of 14 professionals to whom we could send an email notification.

# In[ ]:


# Display the 14 professionals
df_filtered.head(20)


# ### These are their unique id's:

# In[ ]:


youve_got_mail = list(df_filtered['answers_author_id'])

youve_got_mail


# ## Conclusion

# To conclude, let's take a look at how one professional answered Priyanka's question.

# In[ ]:


df_answers = pd.read_csv('../input/answers.csv')
df_answers.loc[2,'answers_body']


# These are the tags that this professional follows:

# In[ ]:


df_tag_users = pd.read_csv('../input/tag_users.csv')
df_tag_users[df_tag_users['tag_users_user_id'] == 'cbd8f30613a849bf918aed5c010340be']


# These are the tag id's that correspond to #military and #army:

# In[ ]:


df_tags = pd.read_csv('../input/tags.csv')

print(df_tags[df_tags['tags_tag_name'] == 'military'])
print(df_tags[df_tags['tags_tag_name'] == 'army'])


# This person has answered this question quite thoroughly and gracefully. Yet it's interesting to note that the hash tags that this professional follows do not include either #military (id 27) or #army (id 18016). He or she probably found this question by scrolling through a forum. This is why, when creating this solution, I chose to use question hash tags and not consider hash tags that professionals follow. 

# Thank you for reading and good luck in this competition.

# In[ ]:




