#!/usr/bin/env python
# coding: utf-8

# Given a professional, the goal is to return a list of questions sorted by the predicted likelihood of the professional answering the question. This can be treated like a recommendation problem with implicit feedback, where the professional is the user and the question is the item. A professional answering a question can be used as a form of implicit feedback. The  [Implicit](https://github.com/benfred/implicit) library is used  to implement a collaboritve filtering algorithm that is based on the method used in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf).

# In[ ]:


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


# In[ ]:


# install implicit package
# !pip install implicit
import implicit


# In[ ]:


get_ipython().system('export OPENBLAS_NUM_THREADS=1')


# In[ ]:


FACTORS = 25

# load data
df_professionals = pd.read_csv(f'../input/professionals.csv')
df_questions = pd.read_csv(f'../input/questions.csv')
df_answers = pd.read_csv(f'../input/answers.csv')


# In[ ]:


# helpers for converting id's to indices and vice-versa
p_id_to_idx = { p_id: i for i, p_id in zip(df_professionals.index, df_professionals.professionals_id) }
p_idx_to_id = { i: p_id for p_id, i in p_id_to_idx.items() }
q_id_to_idx = { q_id: i for i, q_id in zip(df_questions.index, df_questions.questions_id) }
q_idx_to_id = { i: q_id for q_id, i in q_id_to_idx.items() }


# In[ ]:


# helper methods
def get_prof(prof_id):
    return df_professionals[df_professionals.professionals_id == prof_id][['professionals_industry', 'professionals_headline']] 

def get_q_and_a(prof_id):
    pd_q_and_a = pd.merge(left=df_questions, right=df_answers.rename(columns={'answers_question_id': 'questions_id'}), how='left')
    return pd_q_and_a[pd_q_and_a.answers_author_id == prof_id][['questions_id','questions_title', 'questions_body', 'answers_body']]

def recommend(prof_id, model, item_users):
    q_rec_idxs = [q_idx for q_idx, _ in model.recommend(p_id_to_idx[prof_id], item_users.tocsr().T)]
    return df_questions[df_questions.index.isin(q_rec_idxs)][['questions_id','questions_title', 'questions_body']]

def train(item_users, user_factors=None, item_factors=None, factors=FACTORS):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=0.01, dtype=np.float64, iterations=10)
    if user_factors is not None:
        model.user_factors = user_factors
    if item_factors is not None:
        model.item_factors = item_factors
    confidence = 20
    model.fit(confidence * item_users)
    return model

def explain(item_users,p_id, q_id, ):
    _, contributions, _ = model.explain(p_id_to_idx[p_id], item_users.tocsr().T, 
              q_id_to_idx[q_id])
    q_ids = [q_idx_to_id[q_id] for q_id, _ in contributions]
    return df_questions[df_questions.questions_id.isin(q_ids)]

def similar(model, q_id):
    q_idxs_and_scores = model.similar_items(q_id_to_idx[q_id])
    q_idxs = [q_idx for q_idx, _ in q_idxs_and_scores]
    return df_questions[df_questions.index.isin(q_idxs)]


# To train the model, we create an items-users matrix where the questions are the items and the users are the professionals. The value of element i, j is set to 1 if the i-th professional answered the j-th question. Otherwise, it is set to 0

# In[ ]:


# save index of of professional and question if an answer exists
p_and_q_idxs = set()
for p_id, q_id in zip(df_answers.answers_author_id, df_answers.answers_question_id):
    if p_id in p_id_to_idx and q_id in q_id_to_idx:
        p_and_q_idxs.add((p_id_to_idx[p_id], q_id_to_idx[q_id]))
        
        
p_idxs, q_idxs = [], []
for p_idx, q_idx in p_and_q_idxs:
    p_idxs.append(p_idx)
    q_idxs.append(q_idx)

#create sparse matrix
P = df_professionals.shape[0]
Q = df_questions.shape[0]
item_users = coo_matrix((np.ones(len(p_idxs)), (q_idxs, p_idxs)), shape=(Q, P))


# In[ ]:


# create model and fit using the matrix, and a confidence level
model = train(item_users)


# Lets take a look at the results. I've chosen a professional who is in the software industry and has answered 3 questions about software engineering.

# In[ ]:


sample_dev_prof_id = '1ec14aee9311480681dfa81b0f193de8'
get_prof(sample_dev_prof_id)


# In[ ]:


get_q_and_a(sample_dev_prof_id)


# It seems like a fair assumption that this particular professional is more likely to answer questions about software engineering. Here are the questions that the model recommends:

# In[ ]:


recommend(sample_dev_prof_id, model, item_users).values


# The model predicts questions related to software engineering, which makes sense for this particular example.

# ## Improving the Model

# Lets look at another professional in the Architecture industry.

# In[ ]:


sample_architect_prof_id = 'b45e7851aded479a92282ea3c66300ab'
get_prof(sample_architect_prof_id)


# In[ ]:


get_q_and_a(sample_architect_prof_id)


# Based on the questions that the professional has answered, it seems that he or she is more likely to answer a question if it is about architecture. Here are the questions that the model recommends:

# In[ ]:


recommend(sample_architect_prof_id, model, item_users).values


# The model failed to recommend any questions about architecture.
# 
# The algorithm is trying to estimate user-factor vectors for each user and item-factor vectors for each item. The vectors are initialized to random values initially. However, what if we used what we know to initialize these vectors to values that may lead to improved recommendations? For example, let's assume that professionals in the architecture industry are more likely to answer questions about architecture. To model, this, we can set the first index in the user-factor vector for professionals with `professionals_industry` equal to "Architecture" to a high value, and set the first index in the item-factor vector for questions with `questions_title` or `questions_body` containing the word "architecture" to a high value.
# 
# This thinking behind this is from the NVBSVM++ algorithim discussed in a  [fast.ai](https://www.fast.ai/) lecture (which referenced [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018)).

# In[ ]:


# initialize the user and item factor vectors
user_factors = np.random.rand(df_professionals.shape[0], FACTORS).astype(np.float32) * 0.1
item_factors = np.random.rand(df_questions.shape[0], FACTORS).astype(np.float32) * 0.1

# modify the user factor vector for professionals in the architecture industry 
for i, industry in zip(df_professionals.index, df_professionals.professionals_industry.values):
    if industry == industry and 'architecture' in industry.lower():
        # make the value of the first index much larger than the others
        user_factors[i, 0] = 5.0

# modify the item factor vector for questions that contain the word architecture        
for i, title, body in zip(df_questions.index, df_questions.questions_title, df_questions.questions_body):
    title, body = title.lower(), body.lower()
    if 'architecture' in title or 'architecture' in body:
        # make the value of the first index much larger than the others
        item_factors[i, 0] = 5.0


# In[ ]:


model = train(item_users, user_factors=user_factors, item_factors=item_factors)


# In[ ]:


recommend(sample_architect_prof_id, model, item_users).values


# The change caused the algorithm to recommend more architecture related questions.
# 
# What about professionals that have not answered any questions? Unfortunately, because these professionals will not provide the model with any implicit feedback, their user-factor vectors will be set to 0.

# In[ ]:


sample_bio_prof_id = 'b2def296aff34e5da6b405fae8969e73'
get_prof(sample_bio_prof_id)


# In[ ]:


get_q_and_a(sample_bio_prof_id)


# In[ ]:


recommend(sample_bio_prof_id, model, item_users).values


# For each answer, we assume that a professional likes the question with a certain level of confidence. This confidence value is arbitrarily set to 50. To solve the cold start issue mentioned above, for professionals who have not answered questions we can assume that a professional likes the question, based on data from both the professional and the question, with a much lower level of confidence.

# In[ ]:


# save index of of professional and question if an answer exists
p_and_q_idxs = set()
for p_id, q_id in zip(df_answers.answers_author_id, df_answers.answers_question_id):
    if p_id in p_id_to_idx and q_id in q_id_to_idx:
        p_and_q_idxs.add((p_id_to_idx[p_id], q_id_to_idx[q_id]))
        
        
p_idxs, q_idxs = [], []
for p_idx, q_idx in p_and_q_idxs:
    p_idxs.append(p_idx)
    q_idxs.append(q_idx)

# get professionals that have not answered a question
df_professionals_cold = df_professionals[~df_professionals.professionals_id.isin(np.unique(df_answers.answers_author_id.values))]
        
# get the indices of professionals in the biotechnology and biomedical engineering industry        
bio_p_idxs, bio_q_idxs = [], []
for i, industry in zip(df_professionals_cold.index, df_professionals_cold.professionals_industry.values):
    if industry == industry and industry.lower() in ['biotechnology', 'biomedical engineering']:
        bio_p_idxs.append(i)

# get the indices of questions with the terms biotechnology or biomedical engineering in them
for i, title, body in zip(df_questions.index, df_questions.questions_title, df_questions.questions_body):
    title, body = title.lower(), body.lower()
    if 'biomedical' in title or 'biomedical' in body or 'biotechnology' in title or 'biotechnology' in body:
        bio_q_idxs.append(i)

# save the pairs of indices to two arrays        
bp_idxs, bq_idxs = [], []        
for i in bio_q_idxs:
    for j in bio_p_idxs:
            bq_idxs.append(i)
            bp_idxs.append(j)
    

# #create sparse matrix
P = df_professionals.shape[0]
Q = df_questions.shape[0]

# assign difference values for questions that professionals answered 
# and questions/professionals that contain biotechnology/biomedical engineering
answer_values = np.full((len(p_idxs), 1), 1.0)
bio_values = np.full((len(bp_idxs), 1), 0.001)

values = np.concatenate((answer_values, bio_values))
updated_item_users = coo_matrix((values[:, 0], (q_idxs + bq_idxs, p_idxs + bp_idxs)), shape=(Q, P), dtype=np.float64)


# In[ ]:


model = train(updated_item_users)


# In[ ]:


recommend(sample_bio_prof_id, model, item_users).values


# Now the recommendations include biotechnology and biomedical engineering related questions.
