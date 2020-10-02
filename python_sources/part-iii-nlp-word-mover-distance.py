#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
from nltk import word_tokenize
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from time import time
from tqdm import tqdm
stop_words = stopwords.words('english')
warnings.filterwarnings('ignore')
root_path = '../input/data-science-for-good-careervillage/'
print('The csv files provided are:\n')
print(os.listdir(root_path))
model = KeyedVectors.load_word2vec_format('../input/word2vec-google/GoogleNews-vectors-negative300.bin', binary=True)


# In[19]:


df_emails = pd.read_csv(root_path + 'emails.csv')
df_questions = pd.read_csv(root_path + 'questions.csv')
df_professionals = pd.read_csv(root_path + 'professionals.csv')
df_comments = pd.read_csv(root_path + 'comments.csv')
df_tag_users = pd.read_csv(root_path + 'tag_users.csv')
df_group_memberships = pd.read_csv(root_path + 'group_memberships.csv')
df_tags = pd.read_csv(root_path + 'tags.csv')
df_answer_scores = pd.read_csv(root_path + 'answer_scores.csv')
df_students = pd.read_csv(root_path + 'students.csv')
df_groups = pd.read_csv(root_path + 'groups.csv')
df_tag_questions = pd.read_csv(root_path + 'tag_questions.csv')
df_question_scores = pd.read_csv(root_path + 'question_scores.csv')
df_matches = pd.read_csv(root_path + 'matches.csv')
df_answers = pd.read_csv(root_path + 'answers.csv')
df_school_memberships = pd.read_csv(root_path + 'school_memberships.csv')


# This notebook is a continuation of these notebooks - [Part I: Yet Another EDA, Strategy & Useful Links](https://www.kaggle.com/akshayt19nayak/part-i-yet-another-eda-strategy-useful-links) and [Part II: Tag RecSys - Cosine + Levenshtein Dist](https://www.kaggle.com/akshayt19nayak/part-ii-tag-recsys-cosine-levenshtein-dist). The notebook is inspired by [this tutorial](https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html) on Word Mover Distance. There are 3 sections to this notebook: 
# - [Preprocessing and computing WMD Similarity](#Preprocessing-and-computing-WMD-Similarity)
# - [Recommender System](#Recommender-System)
# - [Real-Time Implementation - FINAL](#Real-Time-Implementation---FINAL)

# ## Preprocessing and computing WMD Similarity

# In[ ]:


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return(doc)


# In[ ]:


clean_corpus = [] 
documents = df_questions['questions_title'].tolist()  #wmd_corpus, with no pre-processing (so we can see the original documents).
for text in tqdm(df_questions['questions_title']):
    text = preprocess(text)   
    clean_corpus.append(text)        


# In[47]:


start = time()
#We won't be training our own Word2Vec model and we'll instead use the pretrained vectors
num_best = 10
instance = WmdSimilarity(clean_corpus, model, num_best=num_best)  
print('Cell took %.2f seconds to run.' % (time() - start))


# ## Recommender System 

# To make recommendations based on a question:
# 
# - Find questions that are similar to the one under consideration
# - Find if the similar questions have been answered
# - If yes, find if the professional is active. Active professional has to have answered a question within the last 1 year
# - If multiple professionals fit the criteria, rank them based on the proportion of questions they have answered within 24-48 hours [since that is a key metric](https://www.kaggle.com/c/data-science-for-good-careervillage/discussion/84845#latest-496046)

# In[22]:


#To see the profile of the volunteers and the questions that they have answered
df_questions['questions_date_added'] = pd.to_datetime(df_questions['questions_date_added'])
df_answers['answers_date_added'] = pd.to_datetime(df_answers['answers_date_added'])
df_answers_professionals = pd.merge(df_answers, df_professionals, left_on='answers_author_id', right_on='professionals_id', how='outer')
df_questions_answers_professionals = pd.merge(df_questions, df_answers_professionals, left_on='questions_id', right_on='answers_question_id')
df_qap_time_taken = df_questions_answers_professionals.groupby(['professionals_id','questions_id']).agg({'questions_date_added':min, 'answers_date_added':min})
df_qap_time_taken['less_than_2_days'] = df_qap_time_taken['answers_date_added'] - df_qap_time_taken['questions_date_added'] < '2 days'
df_qap_time_taken = df_qap_time_taken.reset_index().groupby('professionals_id', as_index=False).agg({'less_than_2_days':np.mean})
last_date = df_questions['questions_date_added'].max() #date of the last question asked on the platform
df_ap_grouped = df_answers_professionals.groupby('professionals_id').agg({'answers_date_added':max}).apply(lambda x:
                                                                                          (last_date-x).dt.days)
df_ap_grouped.rename(columns={'answers_date_added':'days_since_answered'}, inplace=True)
active_professionals = df_ap_grouped[df_ap_grouped['days_since_answered']<365].index


# We'll give the same examples we did in the second notebook

# ### Example Recommendation 1

# In[ ]:


sent = 'Should I declare a minor during undergrad if I want to be a lawyer?'
topk = 5
query = preprocess(sent)
sims = instance[query]  #A query is simply a "look-up" in the similarity class.
#Print the query and the retrieved documents, together with their similarities.
print('Question:')
print(sent)
#We won't consider the first index since that is the question itself
for i in range(1,topk+1): 
    print('\nsim = %.4f' % sims[i][1])
    print(documents[sims[i][0]])


# In[ ]:


idx = [tup[0] for tup in sims][:5]
author_id = df_answers[df_answers['answers_question_id'].isin(df_questions.iloc[idx]['questions_id'])]['answers_author_id']
active_author_id = author_id[author_id.isin(active_professionals)]
df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)
print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())
print('The profile of the professionals:')
df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]


# ### Example Recommendation 2

# In[ ]:


sent = 'My current plan is to go to a one year film college to get a certificate in screenwriting. Many people have mentioned that you really don\'t need a film degree to get into film, so a certificate is fine. Is this true?'
topk = 5
query = preprocess(sent)
sims = instance[query]  #A query is simply a "look-up" in the similarity class.
#Print the query and the retrieved documents, together with their similarities.
print('Question:')
print(sent)
for i in range(1,topk+1):
    print('\nsim = %.4f' % sims[i][1])
    print(documents[sims[i][0]])


# In[ ]:


idx = [tup[0] for tup in sims][:5]
author_id = df_answers[df_answers['answers_question_id'].isin(df_questions.iloc[idx]['questions_id'])]['answers_author_id']
active_author_id = author_id[author_id.isin(active_professionals)]
df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)
print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())
print('The profile of the professionals:')
df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]


# The recommendations in both cases do look relevant. There are a few recommendations that are common to both the tag-based recommender and WMD Similarity based recommender. Hashtags are generally used in the question body and in our case we have computed WMD similarity on the questions title. A problem with the Word Mover Distance is that it is a computationally intensive function and the time taken to compute the similarity increases with the increase in length of the string. Thus it may not be the best idea to use it on question bodies and it's better to use it on the questions title. Coupling the two - WMD Similarity on questions title + Tag RecSys on questions body (hashtags), can give us a powerful way of computing question similarity. Given a question, what can also be done is that we can find out the most similar question and the professional who answered it, and then compute professional similarity based on professionals headline or professionals industry. They can then be ranked on the basis of the proportion of questions answered within 48 hours. This is especially useful if the same professional comes up as the top recommendation for every question and we can use something like [stable marriage](https://www.geeksforgeeks.org/stable-marriage-problem/) to assign questions to different professionals.

# ### Active-Passive Indifference 

# One particular advantage of WMD is that it doesn't matter if the question is in the active or the passive voice

# In[ ]:


sent = 'If I want to be a lawyer, should I declare a minor during undergrad?' 
query = preprocess(sent)
sims = instance[query]  #A query is simply a "look-up" in the similarity class.
#Print the query and the retrieved documents, together with their similarities.
print('Question:')
print(sent)
for i in range(0,topk+1): 
    print('\nsim = %.4f' % sims[i][1])
    print(documents[sims[i][0]])


# The most similar question is the first example we used for demonstration purposes. Thus WMD doesn't care about the order in which words appear in the two strings.

# ## Real-Time Implementation - FINAL

# ### Pseudo Code 

# The markdown cell messes up the formatting of the pseudo code so I'll be using triple quotes

# In[ ]:


"""
Let's concatenate the professionals' industry (which is finite and exhaustive) and the professionals' headline. Let's call it industry_headline
active_pros = pros who have answered questions in the last 1 year
new_pros = pros who have registered in the last 1 year

For each question:
1. Compute similarity with a question asked in the past - on the basis on tags using Tag RecSys (if tags are available) and on the basis of questions title using WMD Similarity

2. Go through the pipeline for making recommendations (https://www.kaggle.com/akshayt19nayak/part-ii-tag-recsys-cosine-levenshtein-dist#Recommender-System) and make a list of active_pros that have answered questions having a similarity score above a certain threshold with the question under consideration. Let's call it reco_active_pros. 
(As far as the similarity score is concerned we can make a weighted score of (0.5 * tag similarity + 0.5 * wmd similarity) and then decide on a threshold)

3. count_active = 0, count_new = 0 
For each pro in reco_active_pros:
    - if count_active <= k:
        - if (pro has already been recommended 2 times in the last 3 days):
            - continue (i.e move onto the next pro)
        - else:
            - recommend 
            count_active = count_active + 1
    Let's call this final list as final_reco_active_pros. The number k is user defined i.e how many active pros should we actually recommend?
    
    - if count_new <= k:
        calculate WMD Similarity of a pro with each new_pros industry_headline or/and tags by using Tag RecSys. Select the ones above a certain similarity threshold. Let's call them sim_new_pros
        - For each new_pro in sim_new_pros:
            - if count_new <= k
                - if (new_pro has already been recommended 2 times in the last 3 days):
                    - continue (i.e move onto the next new_pro)
                - else:
                    - recommend 
                    count_new = count_new + 1
    Let's call this final list as final_reco_new_pros
    
Thus for every question we will make a recommendation of final_reco_active_pros + final_reco_new_pros
"""


# ### Things to consider  

# This is in reference to [this discussion](https://www.kaggle.com/c/data-science-for-good-careervillage/discussion/90249#latest-520960). Some of these questions have been answered in greater detail in different sections of the three notebooks
# 
# * **Did you decide to predict Pros given a Question, or predict Questions given a Pro? Why?**
# 
# We are only predicting pros given a question. As per the pseudo code, we iterate through each tagged question, find out similar questions and then recommend professionals - both new and active. It's just something that I decided to base the working of my recommender system on since it's important that every student's question gets answered and in this way we get to ensure that in the best way possible.
# 
# * **Does your model address the "cold start" problem for Professionals who just signed up but have not yet answered any questions? How'd you approach that challenge?**
# 
# Yes. In the second part of point number 3 in the pseudo code, we are finding new professionals who are most similar to the ones who are in the list of recommended active professionals. In order to do this this we can create a new feature called industry_headline that is a concatenated string of professionals_industry and professionals_headline. Using this, we can compute the WMD Similarity of the top professionals in reco_active_pros with the ones in new_pros and thus, we can encourage the latter to answer questions on the platform.
# 
# * **Did your model have any novel approaches to ensuring that "no question gets left behind"?**
# 
# For about 97.3% of questions, we can use both Tag RecSys as well as WMD Similarity. For the rest we can use the latter, since every question will have a question title by default.
# 
# * **What types of models did you try, and why did you pick the model architecture that you picked?**
# 
# I was addressing one level of complexity/ one problem at a time and while I was doing my EDA, certain problems such as the enormous number of tags and the multiple variations of each tag, stood out. I realised that metadata in this format is highly useful and it gives us standardised entities in a way. Also, using pretrained word vectors gives us context that is otherwise not available but possible due to transfer learning. I used this approach as it's simple enough to understand, control and debug.
# 
# * **Did you engineer any new data features? Which ones worked well for your model?**
# 
# For each of the sections under EDA, multiple features were created to get a better understanding of the data. For the recommender system, we created the feature 'less_than_2_days', that is used to rank professionals based on the proportion of questions they have answered within 48 hours. 
# 
# * **Is there anything built into your approach to protect certain Professionals from being "overburdened" with a disproportionately high share of matches?**
# 
# Yes, this is precisely what point number 3 in the pseudo code deals with. The first part *Exploits* - questions are recommended to those professionals who are most likely to answer based on similarity and proportion of questions answered within 48 hours and the second part *Explores* - finds professionals who are most similar to the ones we expect an answer from.
# 
# * **What do you wish you could use in the future (that perhaps we could try out on our own as an extension of your work)?**
# 
# I have mentioned some of this in the Insights and Strategy sections within each sub-heading under EDA. With that being said, I would have also liked to work on text summarization. Given similar questions, extract relevant statements from the bodies of their answers and present them to students who have matching queries. Here are some interesting articles that I found on [Medium](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1) and [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/).
