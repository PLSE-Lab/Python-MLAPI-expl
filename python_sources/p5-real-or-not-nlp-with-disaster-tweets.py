#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install jcopml')


# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
# from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


# # Import Data

# In[ ]:


df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df.head()

# why "id" is not set as index? because it is treated as list in the phase of model prediction to test dataset


# In[ ]:


# plot missing value
plot_missing_value(df, return_df=True)


# In[ ]:



# drop too missing values data
df.drop(columns=["keyword", "location"], inplace=True)


# In[ ]:


plot_missing_value(df, return_df=True)


# In[ ]:


print('the data is ready to be engineered')
df.head(10)


# In[ ]:


# total of words of old df
total_words_old_df = df.text.apply(lambda x: len(x.split(" "))).sum()
print(total_words_old_df)


# ## Feature Engineering - Basic Text Processing

# In[ ]:


import nltk 
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation


# ##### Display text samples

# In[ ]:


for i in range(len(df.text)):
    print(df.text[i])
    if i == 10:
        break


# In[ ]:


print("The length of dataframe is", len(df), "rows")


# ### Text Processing

# ##### 1) Normalization to Lower Case & Removing "https: ..."

# In[ ]:


def clean_text(text):
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = text.split()
    text = " ".join(text)
    return text

df1 = df.text.apply(str).apply(lambda x:clean_text(x))


# ##### Display text samples

# In[ ]:


for i in range(len(df1)):
    print(df1[i])
    if i == 10:
        break


# In[ ]:


print("The length of dataframe is", len(df1), "rows")


# ##### 2) Sentence & Word Tokenization; Punctuation and Words Removal 

# ##### resource:
# 
# - Lemmatization: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
# - Stemming: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/ <br>  https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# - Replacing values in DataFrame: https://www.python-course.eu/pandas_replacing_values.php
# - Converting list into pandas series: https://www.geeksforgeeks.org/creating-a-pandas-series-from-lists/

# In[ ]:


df1_clean_text = []
for i in range(len(df1)):
    x = df1[i]
    # x_sent_token = sent_tokenize(x)
    # x_sent_token
    x_word_tokens = word_tokenize(x)
    # x_word_tokens
#     print(x)
    
#     print(df1[i])
    
    # punctuation removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens if w not in punctuation]
#     print(x_word_tokens_removed_punctuations, "punctuation")
    
    # numeric removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens_removed_punctuations if w.isalpha()]
#     print(x_word_tokens_removed_punctuations, "numeric")
    
    # stopwords removal
    x_word_tokens_removed_punctuation_removed_sw = [w for w in x_word_tokens_removed_punctuations if w not in stopwords.words('english')]
#     print(x_word_tokens_removed_punctuation_removed_sw, "stopwords")

    # rejoining the words into one string/sentence as inputted before being tokenized
    x_word_tokens_removed_punctuation_removed_sw = " ".join(x_word_tokens_removed_punctuation_removed_sw)
#     print(x_word_tokens_removed_punctuation_removed_sw)
    
    df1_clean_text.append(x_word_tokens_removed_punctuation_removed_sw)


# ##### Sample Text

# In[ ]:


# text vs processed text
for i,j in zip(df1[0:10], df1_clean_text[0:10]):
    print(i)
    print(j)
    print()


# In[ ]:


# list (df1_clean_text) to series (df1_clean_text_series)

# list
print(type(df1_clean_text))
print(len(df1_clean_text))

# converting list to pandas series
df1_clean_text_series = pd.Series(df1_clean_text)

print(type(df1_clean_text_series))
print(len(df1_clean_text_series))


# In[ ]:


# new series
df1_clean_text_series.head()


# ##### joining the series into dataframe

# In[ ]:


# new df
df['text'] = df1_clean_text_series
df.head(10)


# In[ ]:


# total of words of old df vs new df 

total_words_new_df = df.text.apply(lambda x: len(x.split(" "))).sum()

print("old df: ", total_words_old_df, "words")
print("new df: ", total_words_new_df, "words")
print("text processing has reduced the number of words by", round((total_words_old_df-total_words_new_df)/total_words_old_df*100), "%")


# # Dataset Splitting

# In[ ]:


X = df.text
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# Check the imbalance dataset
y_train.value_counts() / len(y_train) *100

# the train data is more or less balance


# In[ ]:


X_train.head(), X_test.head(), y_train.head(), y_test.head()


# In[ ]:


X_train.head(11)


# # Visualize the Target Label

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


y_train.shape, y_test.shape


# ##### y_train dataset

# In[ ]:


sns.set(style="darkgrid")
sns.countplot(x=y_train)
plt.title("y_train");


# ##### y_test dataset

# In[ ]:


sns.set(style="darkgrid")
sns.countplot(x=y_test)
plt.title("y_test");


# # Training

# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 


# #### TF-IDF

# In[ ]:


pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

model_logreg_tfidf = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model_logreg_tfidf.fit(X_train, y_train)

print(model_logreg_tfidf.best_params_)
print(model_logreg_tfidf.score(X_train, y_train), model_logreg_tfidf.best_score_, model_logreg_tfidf.score(X_test, y_test))


# #### Bag of Words - CountVectorizer

# In[ ]:


pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

# model = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model_logreg_bow = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_jobs=-1, verbose=1)
model_logreg_bow.fit(X_train, y_train)

print(model_logreg_bow.best_params_)
print(model_logreg_bow.score(X_train, y_train), model_logreg_bow.best_score_, model_logreg_bow.score(X_test, y_test))


# # Linear SVM
# 
# source: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568#:~:text=Linear%20Support%20Vector%20Machine%20is,the%20best%20text%20classification%20algorithms.&text=We%20achieve%20a%20higher%20accuracy,5%25%20improvement%20over%20Naive%20Bayes.

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 


# ##### BoW

# In[ ]:


pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
    'algo__penalty': ['l2', 'l1', 'elasticnet'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [7, 8, 9, 10, 11, 12, 13, 14, 15],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
# model_sgd_bow = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1)
model_sgd_bow = RandomizedSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1, random_state=42)
model_sgd_bow.fit(X_train, y_train)


print(model_sgd_bow.best_params_)
print(model_sgd_bow.score(X_train, y_train), model_sgd_bow.best_score_, model_sgd_bow.score(X_test, y_test))


# ##### TFIDF

# In[ ]:


pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
    'algo__penalty': ['l2', 'l1', 'elasticnet'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [7, 8, 9, 10, 11, 12, 13, 14, 15],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
model_sgd_tfidf = RandomizedSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1, random_state=42)
# modelt_sgd_fidf = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1)
model_sgd_tfidf.fit(X_train, y_train)


print(model_sgd_tfidf.best_params_)
print(model_sgd_tfidf.score(X_train, y_train), model_sgd_tfidf.best_score_, model_sgd_tfidf.score(X_test, y_test))


# ## Save & Submit Model with JCOPML

# ##### Save Model

# In[ ]:


from jcopml.utils import save_model


# In[ ]:


# save_model(model, "disaster_tweet_v1.pkl")
# save_model(model, "disaster_tweet_v2.pkl")
# save_model(model, "disaster_tweet_v3.pkl")
# save_model(model, "disaster_tweet_v4.pkl")
# save_model(model, "disaster_tweet_v5.pkl")
# save_model(model, "disaster_tweet_v6.pkl")
# save_model(model, "disaster_tweet_v7.pkl")
# save_model(model, "disaster_tweet_v8.pkl")
# save_model(model, "disaster_tweet_v9.pkl")
# save_model(model, "disaster_tweet_v10.pkl")
save_model(model, "disaster_tweet_v11.pkl")


# ##### Submit Model

# In[ ]:


df_submit = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_submit.drop(columns=["keyword", "location"], inplace=True)
df_submit_id_list = df_submit.id.values.tolist()
df_submit_text_list = df_submit.text.values.tolist()
df_submit.head()


# In[ ]:


print("df_submit.id: ", len(df_submit.id))
print("df_submit.text: ", len(df_submit.text))
print("df_submit_id_list: ", len(df_submit_id_list))
print("df_submit_text_list: ", len(df_submit_text_list))


# In[ ]:


target = modeltfidf.predict(df_submit_text_list)
target


# In[ ]:


print(type(target))
print(len(target))


# In[ ]:


df_submit_final = pd.DataFrame({
    "id": df_submit_id_list,
    "target": target
})


# In[ ]:


# df_submit_final.set_index('id', inplace=True)
df_submit_final.head()


# In[ ]:


# df_submit_final.to_csv("disaster_tweet_v1.csv")
# df_submit_final.to_csv("disaster_tweet_v2.csv")
# df_submit_final.to_csv("disaster_tweet_v3.csv")
# df_submit_final.to_csv("disaster_tweet_v4.csv")
# df_submit_final.to_csv("disaster_tweet_v5.csv")
# df_submit_final.to_csv("disaster_tweet_v6.csv") #numeric removal, tfidf
# df_submit_final.to_csv("disaster_tweet_v7.csv") #numeric removal, bow, n_iter default, random_state default
# df_submit_final.to_csv("disaster_tweet_v8.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v9.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v10.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v11.csv") #SGD Classifier
df_submit_final.to_csv("disaster_tweet_v12.csv") #SGD Classifier + tfidf


# # Save & Submit Model with Pickle

# In[ ]:


import pickle


# In[ ]:


pickle.dump(model_logreg_tfidf, open("model_logreg_tfidf.pkl", 'wb'))
pickle.dump(model_logreg_bow, open("model_logreg_bow.pkl", 'wb'))
pickle.dump(model_sgd_tfidf, open("model_sgd_tfidf.pkl", 'wb'))
pickle.dump(model_sgd_bow, open("model_sgd_bow.pkl", 'wb'))


# In[ ]:


# model = pickle.load(open("knn.pkl", 'rb'))


# # Conclusion

# ### Linear SVM is the best algorithm for classification in the case "Disaster Tweets
# #### Parameter
# - 'algo__alpha': 0.0003, 
# - 'algo__loss': 'log', 
# - 'algo__max_iter': 8, 
# - 'algo__penalty': 'elasticnet', 
# - 'algo__tol': 0.0001}
# 
# #### Score
# - train: 0.960919540229885 
# - 0.7876847290640395 
# - test: 0.8214051214707814
# 
# #### Further Steps
# 1. Use RandomizedSearchCV to once again crosscheck whether the score can be better by tuning the parameter
# 2. Try to apply LSTM to improve the score
