#!/usr/bin/env python
# coding: utf-8

# I'm beginner at NLP, so I learn a lot from kernel. Thank you for post great kernel!<br>
# This is my cheap approach, "Which word make possibility of insincere sentence high?"

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_text = train_data['question_text']
test_text = test_data['question_text']
train_target = train_data['target']
all_text = train_text.append(test_text)


# In[ ]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_text)

train_text_features_tf = tfidf_vectorizer.transform(train_text)
test_text_features_tf = tfidf_vectorizer.transform(test_text)


# In[ ]:


target0_index = list(train_data.query("target == 0").index)
target1_index = list(train_data.query("target == 1").index)
print(len(target0_index))
print(len(target1_index))


# In[ ]:


# check the difference between target1 and target0
target0_score = train_text_features_tf[target0_index]
target1_score = train_text_features_tf[target1_index]

diff_score = np.mean(target1_score, axis=0) - np.mean(target0_score, axis=0)
diff_ary = np.argsort(-diff_score).tolist()[0]
print(diff_score)


# In[ ]:


# insincere words
for i in range(30):
    insincere_word = tfidf_vectorizer.get_feature_names()[diff_ary[i]]
    insincere_data = train_data[train_data['question_text'].str.contains(insincere_word)]
    print("======================")
    print(insincere_word)
    print("len: {}".format(len(insincere_data)))
    print("mean: {}".format(np.mean(insincere_data["target"])))
    print("======================")


# In[ ]:


# NOT insincere words
for i in range(30):
    insincere_word = tfidf_vectorizer.get_feature_names()[diff_ary[-(i+1)]]
    insincere_data = train_data[train_data['question_text'].str.contains(insincere_word)]
    print("======================")
    print(insincere_word)
    print("len: {}".format(len(insincere_data)))
    print("mean: {}".format(np.mean(insincere_data["target"])))
    print("======================")


# Below words may be insincere:
# 
# * Race
# * Sex
# * skin color
# 
# Below words might not be insincere:
# * work (job, work, learn, business, ... engineering is very low! :D )
# * positive word(good, best)
# 
# 

# In[ ]:




