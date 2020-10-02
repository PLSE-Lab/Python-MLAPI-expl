#!/usr/bin/env python
# coding: utf-8

# # Quora Question Pairs - Identifying Duplicate Questions
# In this notebook, our main goal is to identify which questions asked are duplicates of questions that have already been asked.
# We have to submit a binary prediction against a log - loss metric.
# I am doing a late submission to the competition, but this was a very interesting NLP problem so I wanted to solve it!
# Score on the private leaderboard - 0.353 

# **(1). IMPORTING ALL THE REQUIRED MODULES**

# In[17]:


# Data Analysis of the Quora Question Pairs Dataset
# Using the XGBoost Classifier for prediction
# By - Omkar Sabnis: 22-05-2018


# IMPORTING MODULES
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__)
colors = sns.color_palette()
import os,gc
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb


# **(2). LOADING AND PRINTING SOME USEFUL INFORMATION**

# In[24]:


# LOADING AND VISUALIZING THE DATASET
trainingset = pd.read_csv("../input/train.csv")
testingset = pd.read_csv("../input/test.csv")

print("Number of question pairs in trainingset: ", len(trainingset))
print("Number of duplicate pairs: " , round(trainingset['is_duplicate'].mean()*100,2))
questions = pd.Series(trainingset['qid1'].tolist() + trainingset['qid2'].tolist())
print("Number of questions in trainingset:", len(np.unique(questions)))
print("Multiple questions : ", np.sum(questions.value_counts()>1))

plt.figure(figsize=(10,9))
plt.hist(questions.value_counts(),bins=60)
plt.yscale('log',nonposy='clip')
plt.xlabel('Occurence of questions')
plt.ylabel('Number of questions')
plt.show()


# **Conclusion:**
# 1. Many questions appear very few times, while there is a question that has appeared 160 times!
# 2. There are 36.92 % repeated questions.
# 3. Since we are using log-loss, if we predict the mean value of the label, it will be good enough!

# **(3). Feature Analysis:**
# I have used the word share feature from the benchmark model for checking the features and their impact.

# In[23]:


# FEATURE ANALYSIS USING WORD_MATCH_SHARE
stopword = set(stopwords.words('english'))
def word_match_share(row):
    ques1 = {}
    ques2 = {}
    for i in str(row['question1']).lower().split():
        if i not in stopword:
            ques1[i] = 1
    for i in str(row['question2']).lower().split():
        if i not in stopword:
            ques2[i] = 1
    if len(ques1) == 0 or len(ques2) == 0:
        return 0
    q1_shared = [word for word in ques1.keys() if word in ques2]
    q2_shared = [word for word in ques2.keys() if word in ques1]
    rr = (len(q1_shared)+len(q2_shared))/(len(ques1)+len(ques2))
    return rr
plt.figure(figsize=(10,9))
trainmatch = trainingset.apply(word_match_share,axis=1,raw=True)
plt.hist(trainmatch[trainingset['is_duplicate']==0],bins=10,normed=True,label='NOT DUPLICATE')
plt.hist(trainmatch[trainingset['is_duplicate']==1],bins=10,normed=True,label='DUPLICATE')
plt.legend()
plt.title("LABEL DISTRIBUTION - FEATURE ANALYSIS")
plt.show()


# From the graph, we can see that the feature has an impact as it can easily distinguish between the duplicate and non-duplicate  questions!

# In[7]:


# AUC ON TESTING SET
from sklearn import metrics
prob = trainingset['is_duplicate'].mean()
f,t,threshold = metrics.roc_curve(trainingset['is_duplicate'], np.zeros_like(trainingset['is_duplicate']) + prob)
print("Score: ", metrics.auc(f,t))
submissionprob = pd.DataFrame({'test_id':testingset['test_id'],'is_duplicate':prob})
print(submissionprob.head())


# **(4). Text Analysis - Character and Word counts **

# In[22]:


# STUDY OF CHARACTER COUNT AND PROBABILITY OF DUPLICATE QUESTIONS
trainques = pd.Series(trainingset['question1'].tolist() + trainingset['question2'].tolist()).astype(str)
testques = pd.Series(testingset['question1'].tolist() + testingset['question2'].tolist()).astype(str)

train = trainques.apply(len)
test = trainques.apply(len)
plt.figure(figsize=(10,9))
plt.hist(train,bins=100,range=[0,180],color=colors[4],normed=True,label='train')
plt.hist(test,bins=100,range=[1,180],color=colors[3],normed=True,label='test')
plt.title("Character Count vs Probability")
plt.legend()
plt.xlabel("Characters")
plt.ylabel("Probability")
plt.show()
print("Training Mean: ", train.mean())
print("Training Standard Deviation: ", train.std())
print("Testing Mean: ", test.mean())
print("Testing Standard Deviation: ", test.std())


# **Conclusion:**
# 1. Questions have characters between 15 - 150.
# 2. 150 words seems to be a word limit as there is a drop. 
# 3. Maybe the outliers are dummy questions added to the dataset?

# In[25]:


# STUDY OF WORD COUNT AND PROBABILITY OF DUPLICATE QUESTIONS
train = trainques.apply(lambda x: len(x.split(' ')))
test = testques.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(10,9))
plt.hist(train,bins=100,range=[0,50],color='green',normed=True,label='train')
plt.hist(test,bins=100,range=[0,50],color='red',normed=True,label='test')
plt.title("Word Count vs Probability")
plt.legend()
plt.xlabel("Words")
plt.ylabel("Probability")
plt.show()
print("Training Mean: ", train.mean())
print("Training Standard Deviation: ", train.std())
print("Testing Mean: ", test.mean())
print("Testing Standard Deviation: ", test.std())


# **Conclusion:**
# 1. Most questions have 10 words in them.
# 2. The test set is much larger, however, the histogram shows cluttered question length.
# 3. The test set is a dummy, which is proven by this histogram - due to it being so clustered.

# **(5). Rebalancing the Data**
# Since, the data in the training set has 37% duplicity and the testing set has only 17% duplicity, it can reduce the XGBoost accuracy. By rebalancing the training set to have 17% duplicity, we can increase the performance.

# In[26]:


# REBALANCING THE DATA
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = trainmatch
x_test['word_match'] = testingset.apply(word_match_share,axis=1,raw=True)
y_train = trainingset['is_duplicate'].values
pos = x_train[y_train == 1]
neg = x_train[y_train == 0]

#OVERSAMPLE THE NEGATIVE CLASS
p = 0.165
scale = ((len(pos)/(len(pos)+len(neg)))/p)-1
while scale >1:
    neg = pd.concat([neg,neg])
    scale-=1
neg = pd.concat([neg,neg[:int(scale*len(neg))]])
x_train = pd.concat([pos,neg])
y_train = (np.zeros(len(pos))+1).tolist() + np.zeros(len(neg)).tolist()
del pos,neg


# In[ ]:


# SPLITTING THE DATASET
x_train,x_check,y_train,y_check = train_test_split(x_train,y_train,test_size = 0.2,random_state=0)


# In[ ]:


#XGBOOST CLASSIFIER
parameters = {}
parameters['objective'] = 'binary:logistic'
parameters['eval_metric'] = 'logloss'
parameters['eta'] = 0.05
parameters['max_depth'] = 5

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_check, label=y_check)

checklist = [(dtrain, 'train'), (dvalid, 'valid')]

bst = xgb.train(parameters, dtrain, 400, checklist, early_stopping_rounds=50, verbose_eval=10)


# In[1]:


# GENERATING SUBMISSION FILE
dtest = xgb.DMatrix(x_test)
ptest = bst.predict(dtest)
submission = pd.DataFrame()
submission['test_id'] = testingset['test_id']
submission['is_duplicate'] = ptest
submission.to_csv("XGBOOST.csv",index=False)
print(submission.head())

