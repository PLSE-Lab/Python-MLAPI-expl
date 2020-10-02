#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# I am loosely following the tutorial [here](#https://www.kaggle.com/philculliton/nlp-getting-started-tutorial) as I know nothing of NLP.
# 
# 
# This is the first time I have done NLP and I have purposefully not looked at anyone else's notebook and therefore I do not know what the best practices are for this. I shall go through and do the things that seem intuitive. Hopefully I will make lots of mistakes and learn a lot.
# 
# After I have submitted this, I shall start a new kernel and learn from everyone else who has submitted a notebook.
# 
# :)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train_df.head()


# Looking at the columns we have the obvious one which is text, I am not sure what keyword or location would be. Let's explore

# In[ ]:


train_df.keyword.value_counts()


# In[ ]:


train_df.location.value_counts()


# So location, vague to precise, and I am still not quite sure what keyword is. 
# 
# I wonder how the target is spread, knowing kaggle and the learner challenges it will be nice and evenly spread.

# In[ ]:


train_df.target.value_counts()


# Not quite a perfect distribution, although I have no idea at how imbalanced a data set has to be before the model is impaired. I suppose it would be nice to no at which location the most misleading tweets are from.
# 
# Let's have a look at the text

# In[ ]:


train_df['text'].head()[0]


# Interesting...
# 
# As someone who finds religion hilarious, this is an amusing tweet to get for the first one.
# 
# 
# So now i need to figure out how to turn the text into a feature that can be used by any modelling package.
# 
# 

# In[ ]:



sample_text = ['Hello world', '. ~ #### world', 'this is #sample text']
count_vectorizer = feature_extraction.text.CountVectorizer()
count_vectorizer.fit_transform(sample_text).todense()


# If my understanding is correct, count_vectorizer looks finds all the words in the entire list (in this case) and that is the length of each vector, each unique word in input is assigned a position. In the case above 'world' is the 5th entry of each vector. If 'world' is in the input text the 5th entry for its vector representation is set to 1, else 0. 

# In[ ]:


count_vectorizer.vocabulary_


# This seems like a very simple way of representing a document or documents of text and is easily understandale, but what of the other offering of sci-kit learn.
# 
# #### Term frequenct Inverse document frequency TF-IDF
# 
# from sklearn - Equivalent to CountVectorizer followed by TfidfTransformer.
# 
# Term frequency - how often a term appears
# Inverse document frequency - scales words based on there appearance in the document, therefore words like 'the' will count less towards importance?
# 
# This may be more usefull in extracting those words which are less frequent common in the english langauge such as 'volcano'?
# 
# From what I understand, it takes word in the document and divides it by the total number of words int he document. In the tweets example, each tweet is a document.
# 
# And then the inverse frequency portion does some maths and gives a weighting to rare words. [Here](#https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76) is a good article on it 

# In[ ]:


tfidvectorizer = feature_extraction.text.TfidfVectorizer()

tfidvectorizer.fit_transform(sample_text).todense()


# Looking at the output from my test example it is less obvious to me how this is working

# In[ ]:


tfidvectorizer.vocabulary_ # same as count_vectorizer.


# #### Hash vectorizer
# 
# From what I have read this implementation is designed to be memory efficient, and you can select how many features you want to pick-up.

# In[ ]:


hashvectorizer = feature_extraction.text.HashingVectorizer()

hashvectorizer.fit_transform(sample_text).todense()


# In[ ]:


hashvectorizer.get_stop_words


# ## Modelling

# In[ ]:


tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()
count_vectorizer = feature_extraction.text.CountVectorizer()
hash_vectorizer = feature_extraction.text.HashingVectorizer()


# In[ ]:



def model_score(model, features, target):        
    return model_selection.cross_val_score(model, features, target, cv=5, scoring="f1")

def clean(data):    
    mean = round(data.mean(), 2)
    std = round(data.std(), 2)
    
    return f'mean: {mean} +/- {std}' 


# In[ ]:


tfidf_features = tfidf_vectorizer.fit_transform(train_df["text"]).todense()
count_features = count_vectorizer.fit_transform(train_df["text"]).todense()


# In[ ]:


hash_score = model_score(linear_model.RidgeClassifier(), tfidf_features, train_df["target"])
clean(hash_score)


# So, 0.63 is our target to beat

# In[ ]:


from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB


# with 
# 
# features = tfidvectorizer.fit_transform(train_df["text"]).todense()
# 
#     - clean(model_score(LogisticRegression(solver = 'lbfgs'), features, train_df["target"])) = 'mean: 0.64 +/- 0.05'
#     - clean(model_score(SGDClassifier(), features, train_df["target"])) = 'mean: 0.63 +/- 0.05'
#     - clean(model_score(PassiveAggressiveClassifier(), features, train_df["target"])) = 'mean: 0.59 +/- 0.07'
#     - clean(model_score(Perceptron(), features, train_df["target"])) = 'mean: 0.58 +/- 0.05'
#     - clean(model_score(ExtraTreeClassifier(), features, train_df["target"])) = 'mean: 0.53 +/- 0.06'
#     - clean(model_score(GaussianNB(), features, train_df["target"])) = 'mean: 0.59 +/- 0.02'
#     
# Add the different methods of vectorizing as additinal features?

# Perhaps removing non alphanumeric will help? For good measure I threw in a 'lower()' as I read that it is common to do with NLP

# In[ ]:


import re
def only_alpha(x):
    return re.sub(r'[\W_]', ' ', x).lower()

train_df['text_alpha_num'] = train_df['text'].apply(func = only_alpha)


# In[ ]:


clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['text_alpha_num']).todense() , train_df["target"]))


# No change whatsoever! That is good in one respect as we have the same result with fewer 'words'.
# 
# How about using stop words? This works by removing the really common words, such as 'the', 'a' 'etc'. However, seeing as we are using tfidf I am not sure it will help.

# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[ ]:


tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(stop_words = stop_words)
clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['text_alpha_num']).todense() , train_df["target"]))


# Worse with stopwords!

# In[ ]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def stemmer(x):
    return ' '.join([porter.stem(x) for x in x.split(' ')])

train_df['stemmed_words'] = train_df['text_alpha_num'].apply(func = stemmer)


# In[ ]:


train_df['stemmed_words'][0]


# In[ ]:


clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['stemmed_words']).todense() , train_df["target"]))


# Slightly worse. What about using both new features?

# In[ ]:


stemmed = tfidf_vectorizer.fit_transform(train_df["stemmed_words"]).todense()
stemmed_data = pd.DataFrame(stemmed, columns=tfidf_vectorizer.get_feature_names())

alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()
alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())

data = pd.concat([stemmed_data,alpha_data], axis =1)

clean(model_score(LogisticRegression(solver = 'lbfgs'), data , train_df["target"]))


# Even worse...

# ### Let's have a look at the keyword column

# In[ ]:


train_df['keyword'].value_counts()


# Looks like we could clean this up a bit. removing the '%20' which I believe represents a space.

# In[ ]:


train_df['keyword_test'] = train_df['keyword'].fillna('no keyword')


# In[ ]:


clean(model_score(LogisticRegression(solver = 'lbfgs'),tfidf_vectorizer.fit_transform(train_df['keyword_test']).todense() , train_df["target"]))


# In[ ]:


clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['keyword_test']).todense() , train_df["target"]))


# I don't think I have ever seen a score that bad before! I would stuggle to design a model that bad.

# What about extracting hashtags from the twwets?

# In[ ]:


def get_hashtag(x):
    return ' '.join(list(re.findall(r"#(\w+)", x)))

train_df['tags'] = train_df['text'].apply(func = get_hashtag)


# In[ ]:


train_df['tags'].value_counts()


# Seems unlikely that this will help us as the vast majority of people do not use hash tags it turns out. Although from the ones we can see they don't seem to be about disasters

# In[ ]:


train_df.head()


# Let's fill in the tags column

# In[ ]:



def insert_tags(x):
    if x == '':
        return 'no tags'
    else:
        return x

train_df['tags'] = train_df['tags'].apply(func = insert_tags)


# In[ ]:


train_df['tags'].value_counts()


# In[ ]:


clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['tags']).todense() , train_df["target"]))


# Interesting that GuassianNB does so well with this... LogisiticRegression scores 0.17, which is what I would have expected.
# 
# let's sort out location

# In[ ]:


train_df['location'].fillna('no location').value_counts()


# In[ ]:


train_df['location'] = train_df['location'].fillna('no location')


# In[ ]:


clean(model_score(GaussianNB(),tfidf_vectorizer.fit_transform(train_df['location']).todense() , train_df["target"]))


# LogisticRegression = 'mean: 0.29 +/- 0.02'

# In[ ]:


stemmed = tfidf_vectorizer.fit_transform(train_df["stemmed_words"]).todense()
stemmed_data = pd.DataFrame(stemmed, columns=tfidf_vectorizer.get_feature_names())

alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()
alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())

data = pd.concat([stemmed_data,alpha_data], axis =1)

clean(model_score(LogisticRegression(solver = 'lbfgs'), data , train_df["target"]))


# In[ ]:


alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()
alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())

tags_tfidf = tfidf_vectorizer.fit_transform(train_df["tags"]).todense()
tags_data = pd.DataFrame(tags_tfidf, columns=tfidf_vectorizer.get_feature_names())

keys_tfidf = tfidf_vectorizer.fit_transform(train_df["keyword_test"]).todense()
keys_data = pd.DataFrame(keys_tfidf, columns=tfidf_vectorizer.get_feature_names())


location_tfidf = tfidf_vectorizer.fit_transform(train_df["location"]).todense()
location_data = pd.DataFrame(location_tfidf, columns=tfidf_vectorizer.get_feature_names())

data = pd.concat([alpha_data,tags_data,keys_data,location_data], axis =1)


# In[ ]:


clean(model_score(GaussianNB(),data , train_df["target"]))


# LogisticRegression: 'mean: 0.46 +/- 0.08'
# 
# Interesting, when we combine features, the model gets very poor.
# 
# Maybe, if I add the keywrds and tags etc to the text it will be better?

# In[ ]:


train_df['tags'][0] + ' ' + train_df['location'][0]


# In[ ]:


data['combined'] = train_df['tags'] + ' ' + train_df['location'] + ' ' + train_df["keyword_test"] + ' ' + train_df["text_alpha_num"]


# In[ ]:


clean(model_score(GaussianNB(), tfidf_vectorizer.fit_transform(data['combined']).todense() , train_df["target"]))


# LogisticRegression = 'mean: 0.52 +/- 0.08'
# 

# So, using the best, I have come up with so far, I shall try and optimise the model

# In[ ]:


tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()
alpha_num = tfidf_vectorizer.fit_transform(train_df["text_alpha_num"]).todense()
alpha_data = pd.DataFrame(alpha_num, columns=tfidf_vectorizer.get_feature_names())


# solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# 
#     for i in solvers:
#         print(i)
#         print(clean(model_score(LogisticRegression(solver = i), alpha_data , train_df["target"])))
# 
# 
#         newton-cg
#         mean: 0.64 +/- 0.05
#         lbfgs
#         mean: 0.64 +/- 0.05
#         liblinear
#         mean: 0.64 +/- 0.05
#         sag
#         mean: 0.64 +/- 0.05
#         saga
#         mean: 0.64 +/- 0.05
# 
# 

# In[ ]:





# max_iter = [10,100,1000,10000]
# 
#     for i in max_iter:
#         print(i)
#         print(clean(model_score(LogisticRegression(solver = 'lbfgs' ,max_iter = i, n_jobs = -1), alpha_data , train_df["target"])))
# 
#     10
#     mean: 0.64 +/- 0.05
#     100
#     mean: 0.64 +/- 0.05
#     1000
#     mean: 0.64 +/- 0.05

# OKay, so nothing I have done has improved the score, let's submit!

# In[ ]:


clf = LogisticRegression(n_jobs = -1, random_state = 1337)
clf.fit(alpha_data , train_df["target"])


# In[ ]:



test_df['text_alpha_num'] = test_df['text'].apply(func = only_alpha)

alpha_num_test = tfidf_vectorizer.transform(test_df["text_alpha_num"]).todense()
alpha_data_test = pd.DataFrame(alpha_num_test, columns=tfidf_vectorizer.get_feature_names())


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(alpha_data_test)

sample_submission.head()

sample_submission.to_csv('submission.csv', index=False)


