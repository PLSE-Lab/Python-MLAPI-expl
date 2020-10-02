#!/usr/bin/env python
# coding: utf-8

# ## Wiki title baseline
# 
# This is a simple baseline only using the wiki title supplied from the URL in the data set.
# Similar to the best baseline described in the article describing the corpus, pronouns refer to the entity that is a substring of the title (e.g, ```"Dehner" in "Jeremy Dehner"```. This is known as a strong coreference baseline, i.e., taking the main protoganist in the entire article. This seems to work to a certain extend here as well, but note that the creators of the corpus created a balanced selection of what they call Page Entity:
# > Page Entity. Pronouns in a Wikipedia page
# > often refer to the entity the page is about. We
# > include such examples in our dataset but balance them 1:1 against examples that do not
# > include mentions of the page entity.
# 
# Running this baseline on the training data (=```gap_test```) shows a precision for A ab B of about 0.78 (but lower recall).
# 
# The baseline also takes into account the prior and the precision values for A, B, and NEITHER. The probability for a value is set to the precision vallue calculated on the training data and the probabilities for the other two values are set to the precision * prior of the respective value.
# 
# https://arxiv.org/abs/1810.05201
# 
# Baldridge, J., Webster, K., Recasens, M., and Axelrod, V. Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns. 2018.
# Transactions of the Association of Computational Linguistics
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.metrics import f1_score, log_loss, precision_score, confusion_matrix, classification_report

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test_stage_1 = pd.read_csv("../input/test_stage_1.tsv", sep="\t")


# In[ ]:


test_stage_1[0:5]


# In[ ]:


# assigning the GAP dev data as test data
test_df = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", delimiter='\t')
# assigning the GAP test data as train data
train_df = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')
valid_df = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')


# In[ ]:


# using the full set of training and validation data
train_df = pd.concat([train_df,valid_df])


# In[ ]:


train_df.head()


# In[ ]:


def scrape_url(url):
    '''
    get the title of the wikipedia page and replace "_" with white space
    '''
    return url[29:].lower().replace("_"," ")

def check_name_in_string(name,string):
    '''
    check whether the name string is a substring of another string (i.e. wikipedia title)
    '''

    return name.lower() in string



def predict_coref(df):
    pred =[]
    for index, row in df.iterrows():
        wiki_title = scrape_url(row["URL"])
        if (check_name_in_string(row["A"],wiki_title)):
            pred.append("A")
        else:
            if (check_name_in_string(row["B"],wiki_title)):
                pred.append("B")
            else:
                pred.append("NEITHER")
    return pred

train_pred = predict_coref(train_df)
test_pred = predict_coref(test_df)


# In[ ]:


train_len = len(train_df)
A_prior = len(train_df[train_df["A-coref"] == True])/train_len
B_prior = len(train_df[train_df["B-coref"] == True])/train_len
Neither_prior = len(train_df[(train_df["A-coref"] == False) & (train_df["B-coref"] == False)])/train_len

print("A prior: "+str(A_prior))

print("B prior: "+str(B_prior))

print("NEITHER prior: "+str(Neither_prior))


# In[ ]:


gold_train = []
for index, row in train_df.iterrows():
    if (row["A-coref"]):
        gold_train.append("A") 
    else:
        if (row["B-coref"]):
            gold_train.append("B") 
        else:
            gold_train.append("NEITHER")
            
gold_test = []
for index, row in test_df.iterrows():
    if (row["A-coref"]):
        gold_test.append("A") 
    else:
        if (row["B-coref"]):
            gold_test.append("B") 
        else:
            gold_test.append("NEITHER")


# In[ ]:



print(f1_score( gold_train, train_pred, average='micro'))
print(classification_report( gold_train, train_pred))
print(confusion_matrix(gold_train, train_pred))


# In[ ]:


def prec_prob(gold, pred, test):
    '''
    Using the training set to determine the precision by class
    and assigning it to the test data set
    '''
    scores = []
    precision = precision_score(gold, pred,  average=None,
                                labels=['A','B','NEITHER'])
    A_prec = precision[0]
    B_prec = precision[1]
    Neither_prec = precision[2]
    for ante in test:
        if (ante == 'A'):
            scores.append([A_prec, B_prec*B_prior, Neither_prec*Neither_prior])
        else:
            if (ante =='B'):
                scores.append([A_prec*A_prior, B_prec, Neither_prec*Neither_prior])
            else:
                scores.append([A_prec*A_prior,B_prec*B_prior,Neither_prec])
    return scores


# In[ ]:



scores_train = prec_prob(gold_train, train_pred, train_pred)
log_loss(gold_train,scores_train)


# In[ ]:



scores_test = prec_prob(gold_train, train_pred, test_pred)
log_loss(gold_test,scores_test)


# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission_stage_1.csv")


# In[ ]:


sample_submission[['A','B','NEITHER']] = scores_test


# In[ ]:





# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)


# In[ ]:




