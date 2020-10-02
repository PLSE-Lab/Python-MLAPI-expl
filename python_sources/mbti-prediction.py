#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier,plot_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from itertools import compress


from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/mbti_1.csv')


# In[ ]:


df.head()


# In[ ]:


dist = df['type'].value_counts()
dist


# In[ ]:


dist.index


# In[ ]:


plt.hlines(y=list(range(16)), xmin=0, xmax=dist, color='skyblue')
plt.plot(dist, list(range(16)), "D")
# plt.stem(dist)
plt.yticks(list(range(16)), dist.index)
plt.show()


# Interesting observation: Although INTP, INFJ and INFP are claimed to be the rarest types, in this dataset, they are pretty prevalent. It is probably because introverts tend to have much time online than extroverts, who might be busy socializing :D

# In[ ]:


df['seperated_post'] = df['posts'].apply(lambda x: x.strip().split("|||"))
df['num_post'] = df['seperated_post'].apply(lambda x: len(x))


# How much each type of personality posts?

# In[ ]:


df.head()


# In[ ]:


num_post_df = df.groupby('type')['num_post'].apply(list).reset_index()


# In[ ]:


rcParams['figure.figsize'] = 10,5
sns.violinplot(x='type',y='num_post',data=df)
plt.xlabel('')
plt.ylabel('Number of posts')


# In[ ]:


def count_youtube(posts):
    count = 0
    for p in posts:
        if 'youtube' in p:
            count += 1
    return count
        
df['youtube'] = df['seperated_post'].apply(count_youtube)


# In[ ]:


sns.violinplot(x='type',y='youtube',data=df)
plt.xlabel('')
plt.ylabel('Number of posts which mention Youtube')
plt.show()


# In[ ]:


plt.hist(df['youtube'])
plt.title('Distribution of number of posts containing Youtube across individuals')
plt.show()


# In[ ]:


df['seperated_post'][1]


# Clean the text and use simple BoW with multiclass classification Logistic Regression

# In[ ]:


# Before expanding the dataframe, give everyone an unique ID?
df['id'] = df.index


# In[ ]:


len(df)


# In[ ]:


expanded_df = pd.DataFrame(df['seperated_post'].tolist(), index=df['id']).stack().reset_index(level=1, drop=True).reset_index(name='idposts')


# In[ ]:


expanded_df.head()


# In[ ]:


expanded_df=expanded_df.join(df.set_index('id'), on='id', how = 'left')


# In[ ]:


expanded_df=expanded_df.drop(columns=['posts','seperated_post','num_post','youtube'])


# Text are cleaned so that:
# - words are in lowercase
# - urls are removed
# - all numbers are removed
# - all usernames are replaced by the word 'user'
# - all punctuation are removed

# In[ ]:


def clean_text(text):
    result = re.sub(r'http[^\s]*', '',text)
    result = re.sub('[0-9]+','', result).lower()
    result = re.sub('@[a-z0-9]+', 'user', result)
    return re.sub('[%s]*' % string.punctuation, '',result)
    


# In[ ]:


final_df = expanded_df.copy()


# In[ ]:


final_df['idposts'] = final_df['idposts'].apply(clean_text)


# In[ ]:


final_df.head()


# In[ ]:


cleaned_df = final_df.groupby('id')['idposts'].apply(list).reset_index()


# In[ ]:


cleaned_df.head()


# In[ ]:


df['clean_post'] = cleaned_df['idposts'].apply(lambda x: ' '.join(x))


# In[ ]:


df.head()


# Build the vocabulary from 1500 words that are not common words or MBTI personalities, and appear 0.1-0.7 of the time.

# In[ ]:


vectorizer = CountVectorizer(stop_words = ['and','the','to','of',
                                           'infj','entp','intp','intj',
                                           'entj','enfj','infp','enfp',
                                           'isfp','istp','isfj','istj',
                                           'estp','esfp','estj','esfj',
                                           'infjs','entps','intps','intjs',
                                           'entjs','enfjs','infps','enfps',
                                           'isfps','istps','isfjs','istjs',
                                           'estps','esfps','estjs','esfjs'],
                            max_features=1500,
                            analyzer="word",
                            max_df=0.8,
                            min_df=0.1)


# In[ ]:


corpus = df['clean_post'].values.reshape(1,-1).tolist()[0]
vectorizer.fit(corpus)
X_cnt = vectorizer.fit_transform(corpus)


# In[ ]:


X_cnt


# In[ ]:


# Transform the count matrix to a tf-idf representation
tfizer = TfidfTransformer()
tfizer.fit(X_cnt)
X = tfizer.fit_transform(X_cnt).toarray()


# In[ ]:


X.shape


# In[ ]:


all_words = vectorizer.get_feature_names()
n_words = len(all_words)


# In[ ]:


df['fav_world'] = df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
df['info'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['decision'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['structure'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)


# In[ ]:


df.head()


# In[ ]:


X_df = pd.DataFrame.from_dict({w: X[:, i] for i, w in enumerate(all_words)})


# In[ ]:


def sub_classifier(keyword):
    y_f = df[keyword].values
    X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_df, y_f, stratify=y_f)
    f_classifier = XGBClassifier()
    print(">>> Train classifier ... ")
    f_classifier.fit(X_f_train, y_f_train, 
                     early_stopping_rounds = 10, 
                     eval_metric="logloss", 
                     eval_set=[(X_f_test, y_f_test)], verbose=False)
    print(">>> Finish training")
    print("%s:" % keyword, sum(y_f)/len(y_f))
    print("Accuracy %s" % keyword, accuracy_score(y_f_test, f_classifier.predict(X_f_test)))
    print("AUC %s" % keyword, roc_auc_score(y_f_test, f_classifier.predict_proba(X_f_test)[:,1]))
    return f_classifier


# In[ ]:


fav_classifier = sub_classifier('fav_world')


# In[ ]:


info_classifier = sub_classifier('info')


# In[ ]:


decision_classifier = sub_classifier('decision')


# In[ ]:


str_classifier = sub_classifier('structure')


# In[ ]:


rcParams['figure.figsize'] = 20, 10
plt.subplots_adjust(wspace = 0.5)
ax1 = plt.subplot(1, 4, 1)
plt.pie([sum(df['fav_world']), 
         len(df['fav_world']) - sum(df['fav_world'])], 
        labels = ['Extrovert', 'Introvert'],
        explode = (0, 0.1),
       autopct='%1.1f%%')

ax2 = plt.subplot(1, 4, 2)
plt.pie([sum(df['info']), 
         len(df['info']) - sum(df['info'])], 
        labels = ['Sensing', 'Intuition'],
        explode = (0, 0.1),
       autopct='%1.1f%%')

ax3 = plt.subplot(1, 4, 3)
plt.pie([sum(df['decision']), 
         len(df['decision']) - sum(df['decision'])], 
        labels = ['Thinking', 'Feeling'],
        explode = (0, 0.1),
       autopct='%1.1f%%')

ax4 = plt.subplot(1, 4, 4)
plt.pie([sum(df['structure']), 
         len(df['structure']) - sum(df['structure'])], 
        labels = ['Judging', 'Perceiving'],
        explode = (0, 0.1),
       autopct='%1.1f%%')

plt.show()


# At this point, we have 4 classifiers for each of the preferences. The accuracy for each of them are:
# - Favorite world (Extrovert/Introvert) : 0.81
# - Information (Sensing/ Intuition): 0.86
# - Decision (Thinking, Feeling): 0.77
# - Structure (Judging, Perceiving): 0.73
# 
# We will use RandomSearchCV with 5-fold cross validation to tune hyperparameters for XGBoost.

# In[ ]:


# Get the default params for the current E/I classifier
fav_classifier.get_xgb_params()


# We will use random grid search for random hyperparameters tuning. In this simple example, we will tune for <b> bolded features </b>:
# - <b>min_child_weight</b>: the minimum sum of weights of all observations required in a child. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values will lead to under-fitting.
# - <b>max_depth</b>: maximum depth of the tree. Too high values will lead to over-fitting and vice versa. Typical values are between 3 and 10.
# - max_leaf_nodes: number of maximum leaves in the tree. Can only be tuned interchangably with max_depth
# - <b>gamma</b>: minimum loss in reduction required to make a split. 
# - <b>subsample</b>: the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting. Typical values are between 0.5 and 1.
# - <b>colsample_bytree</b>: denotes the number of features randomly selected for a tree. Typical values are between 0.5 and 1.
# - lambda: L2 regularization term on weights (not used very often but might be helpful to reduce overfitting).
# - alpha: L1 regularization term on weights (not used very often but might be helpful in case of high dimensions as it creates more sparse trees)
# 

# In[ ]:


# set up parameters grids
params = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7]
        }


# In[ ]:


# xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
#                     silent=True, nthread=1)


# In[ ]:


# keyword = 'fav_world'
# y = df[keyword].values
# folds = 3
# param_comb = 5

# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=1, cv=skf.split(X,y), verbose=3, random_state=1001 )


# In[ ]:


# random_search.fit(X, y)


# In[ ]:


# subsample=0.6, min_child_weight=1, max_depth=5, gamma=0.5, colsample_bytree=1.0 is the best hyperparameters in the search


# This search takes a very long time and the auc doesn't improve substantially. One interesting thing about the search was that Kaggle server might run out of time on multiple workers/ doesn't have enough resources, hence the number of workers is set to 1. 

# ## Features importance
# What words are associated with our personalities?

# In[ ]:


plot_importance(fav_classifier, max_num_features = 20)
plt.title("Features associated with Extrovert")
plt.show()


# In[ ]:


plot_importance(info_classifier, max_num_features = 20)
plt.title("Features associated with Sensing")
plt.show()


# In[ ]:


plot_importance(decision_classifier, max_num_features=20)
plt.title("Features associated with Thinking")
plt.show()


# In[ ]:


plot_importance(str_classifier, max_num_features=20)
plt.title("Features associated with Judging")
plt.show()


# ### What words are most common among personality types

# In[ ]:


# Start with one review:
def generate_wordcloud(text, title):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize = 40)
    plt.show()


# In[ ]:


df_by_personality = df.groupby("type")['clean_post'].apply(' '.join).reset_index()


# In[ ]:


df_by_personality.head()


# In[ ]:


for i, t in enumerate(df_by_personality['type']):
    text = df_by_personality.iloc[i,1]
    generate_wordcloud(text, t)


# In[ ]:


test_string = 'I like to observe, think, and analyze to find cons and pros. Based on my analysis, I like to create a solution based on cost effective analysis to maximize the resource to improve the performance. I like talking to my friends. I like to read and learn. I simulate a lot of different situations to see how I would react. I read or watch a lot to improve myself. I love talking to them and seeing what they have been up to. I have a variety of friends, and I appreciate they all experience different things. Listening to their emotion, experience, and life is always great.'.lower()
final_test = tfizer.transform(vectorizer.transform([test_string])).toarray()


# In[ ]:


test_point = pd.DataFrame.from_dict({w: final_test[:, i] for i, w in enumerate(all_words)})


# In[ ]:


fav_classifier.predict_proba(test_point) #[I, E]


# In[ ]:


info_classifier.predict_proba(test_point) #[N,S]


# In[ ]:


decision_classifier.predict_proba(test_point) #[F,T]


# In[ ]:


str_classifier.predict_proba(test_point) #[P,J]


# In[ ]:





# In[ ]:




