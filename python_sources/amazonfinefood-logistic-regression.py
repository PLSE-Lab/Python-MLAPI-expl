#!/usr/bin/env python
# coding: utf-8

# **Amazon Fine Food: Preprocessing and classification using logistic Regression**
# 
# Accuracy: 94%
# 
# Confusion Matrix: [[ 2704, 980]
#                    [812, 25504]]

# In[ ]:


#importing all the modules that might be used later on
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")



import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.snowball import SnowballStemmer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# In[ ]:


df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')


# In[ ]:


print(df.shape)
df.head()


# Lets Begin by deleting the reviews that are same across various products of same type i.e reviews that are shared amongst different products

# In[ ]:


df = df.drop_duplicates(subset = {'UserId', 'Time', 'Text'})
df.shape


# Since we want to simply classify a review as either good or bad, we will remove the rows with score 3 because it signifies the review is neither good nor bad

# In[ ]:


df = df.loc[lambda df: df['Score'] != 3]
print(df.shape)
df['Score'].unique()


# Now I'll replace score greater than 3 with 1 to signify a good review and score less than 3 with 0 to signify a bad review

# In[ ]:


def scorer(x):
    if x > 3:
        return 1
    return 0

scores = df['Score']
scores_binary = scores.map(scorer)
df['Score'] = scores_binary
df['Score'].unique()


# Now I'll remove rows in which helpfulness numerator is greater than helpfulness denominator because thats not possible and if that is happening it means that it is wrong observation

# In[ ]:


df = df[df.HelpfulnessDenominator >= df.HelpfulnessNumerator]
df.shape


# Text is the major factor while deciding score, so I'll seprate out texts from the dataframe

# In[ ]:


df = df.sort_values('Time', axis = 0, inplace = False, ascending = True)
texts = df['Text']
texts.head()


# Since the extracted text would have things like numbers, url etc the things that arent really important for the decision making so i'll remove them.

# In[ ]:


# removing all the url from the text
def remove_url(s):
  return re.sub(r'http\S+', '', s)
test = "hello https://www.google.com/ world"
print(remove_url(test))
texts = texts.map(remove_url)


# In[ ]:


# removing all the tags from the text
def remove_tag(s):
  return re.sub(r'<.*?>', ' ', s)
test = "<p> hello world </p>"
print(remove_tag(test))
texts = texts.map(remove_tag)


# In[ ]:


#converting strings into only lowercase.
def lower_words(s):
   return s.lower()
test = "HELLO world"
print(lower_words(test))
texts = texts.map(lower_words)


# In[ ]:


# decontracting all contracted words

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
test = "it'll take time to complete this notebook"
print(decontracted(test))
texts = texts.map(decontracted)


# In[ ]:


# deleting all the words with numbers in them

def remove_words_with_nums(s):
  return re.sub(r"\S*\d\S*", "", s)
test = "hello0 world"
print(remove_words_with_nums(test))
texts = texts.map(remove_words_with_nums)


# In[ ]:


# deleting words with special character in them

def remove_special_character(s):
  return re.sub('[^A-Za-z0-9]+', ' ', s)
test = "hello-world"
print(remove_special_character(test))
texts = texts.map(remove_special_character)


# In[ ]:


# defining the set of stop words according to our problem basically we'll remove all the negations from the pre-defined set of stopwords
# i removed some stopwords from basic english language stopwords set, the removed elements are related to negations that generally express a negative emotion

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren'])


# In[ ]:


def remove_stopword(s):
    res = ' '.join([word for word in s.split(' ') if word not in stopwords])
    return res

test = "hello my world"
print(remove_stopword(test))
texts = texts.map(remove_stopword)


# We have two options when it comes to reducing the words to their root form, stemming and lemmatization, none of the two techinque is better than other so I would apply both of them and then see which one of them provides better result and will use that one for future reference.

# In[ ]:


test_texts = texts[:10000:].copy()
test_texts.shape


# In[ ]:


#Stemming (Snowball Stemmer)
stemmer = SnowballStemmer('english')
def stemming(s):
    res = ' '.join([stemmer.stem(word) for word in s.split(' ')])
    return res
test = "running and walking"
print(stemming(test))
stemmed_texts = test_texts.map(stemming)


# In[ ]:


lemmatizer = WordNetLemmatizer()
def lemmatization(s):
    res = ' '.join([lemmatizer.lemmatize(word) for word in s.split(' ')])
    return res
test = "was running"
lemmatized_texts = test_texts.map(lemmatization)
print(lemmatization(test))


# For the basic comparison i'll apply bag of words model on texts, stemmed_texts, and lemmatized_texts and i'll continue with the one that provides best accuracy

# In[ ]:


#texts
X = test_texts
y = df['Score'][:10000:].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)
print('AUC: ', roc_auc_score(y_test, predictions))


# In[ ]:


#texts
X = stemmed_texts
y = df['Score'][:10000:].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)
print('AUC: ', roc_auc_score(y_test, predictions))


# In[ ]:


#texts
X = lemmatized_texts
y = df['Score'][:10000:].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)
print('AUC: ', roc_auc_score(y_test, predictions))


# We'll continue with the stemmed version as it performs similar to othe ones but will provide us with smaller vectors

# In[ ]:


text = texts[:100000:]
from nltk.stem import PorterStemmer
st = PorterStemmer()
stemmed_data = []
for review in text:
    stemmed_data.append(st.stem(review))
print('Done')


# In[ ]:


X = stemmed_data
y = df['Score'][:100000:].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
count_vect = CountVectorizer()
train_bow = count_vect.fit_transform(X_train)
test_bow = count_vect.transform(X_test)
print(train_bow.shape)


# In[ ]:


c_dist = []
for x in range(-2, 3):
    mul = 10 ** (-x + 1)
    center = 10 ** x
    for y in range(-5,6):
        c_dist.append(y/mul + center)
print(c_dist)
max_iter = []
for x in range (75, 130, 5):
    max_iter.append(x)
print(max_iter)
param_dist = {'C' : c_dist, 'max_iter' : max_iter}
print(param_dist)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(train_bow, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(test_bow)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(train_bow, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(test_bow)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


count_vect = CountVectorizer(ngram_range = (1, 2))
train_bow = count_vect.fit_transform(X_train)
test_bow = count_vect.transform(X_test)
print(train_bow.shape)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(train_bow, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(test_bow)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(train_bow, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(test_bow)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Tfidf

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf_vect = TfidfVectorizer()
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean = False)
x_train = sc.fit_transform(train_tfidf)
x_test = sc.transform(test_tfidf)


# In[ ]:


c_dist = []
for x in range(-2, 3):
    mul = 10 ** (-x + 1)
    center = 10 ** x
    for y in range(-5,6):
        c_dist.append(y/mul + center)
print(c_dist)
max_iter = []
for x in range (75, 130, 5):
    max_iter.append(x)
print(max_iter)
param_dist = {'C' : c_dist, 'max_iter' : max_iter}
print(param_dist)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(x_train, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(x_test)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(train_tfidf, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(test_tfidf)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean = False)
x_train = sc.fit_transform(train_tfidf)
x_test = sc.transform(test_tfidf)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(x_train, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(x_test)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(x_train, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(x_test)
from sklearn.metrics import accuracy_score
print('Accuracy :', accuracy_score(y_test, pred)*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# AVG W2VEC

# In[ ]:


list_of_sent_train = []
for i in X_train:
    sent = []
    for word in i.split():
        sent.append(word)
    list_of_sent_train.append(sent)


# In[ ]:


from gensim.models import Word2Vec
w2v_model = Word2Vec(list_of_sent_train,min_count = 5,size = 50,workers = 4)
sent_vectors_train = []
for sent in list_of_sent_train:
    sent_vec = np.zeros(50)
    cnt_word = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_word += 1
        except:
            pass
    sent_vec /= cnt_word
    sent_vectors_train.append(sent_vec)
print(len(sent_vectors_train))


# In[ ]:


list_of_sent_test = []
for i in X_test:
    sent = []
    for word in i.split():
        sent.append(word)
    list_of_sent_test.append(sent)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec
w2v_model = Word2Vec(list_of_sent_test,min_count = 5,size = 50,workers = 4)
sent_vectors_test = []
for sent in list_of_sent_test:
    sent_vec = np.zeros(50)
    cnt_word = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_word += 1
        except:
            pass
    sent_vec /= cnt_word
    sent_vectors_test.append(sent_vec)
print(len(sent_vectors_test))


# In[ ]:


np.where(np.isnan(sent_vectors_test))


# In[ ]:


sent_vectors_train = pd.DataFrame(sent_vectors_train)
sent_vectors_test = pd.DataFrame(sent_vectors_test)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(sent_vectors_train)
sent_vectors_train = imputer.transform(sent_vectors_train)
sent_vectors_test = imputer.transform(sent_vectors_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
w2v_train = sc.fit_transform(sent_vectors_train)
w2v_test = sc.transform(sent_vectors_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(w2v_train, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(w2v_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print('accuracy is',acc*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
w2v_train = sc.fit_transform(sent_vectors_train)
w2v_test = sc.transform(sent_vectors_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(w2v_train, y_train)


# In[ ]:


print(random_model.best_estimator_)
pred = random_model.predict(w2v_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print('accuracy is',acc*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# tfidf w2v

# In[ ]:


tfidf_vect = TfidfVectorizer()
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)


# In[ ]:


tf_idf_feat = tfidf_vect.get_feature_names()
tfidf_sent_vec_train = []
row = 0
w2v_model = Word2Vec(list_of_sent_train,min_count = 5,size = 50,workers = 4)
for sent in list_of_sent_train:
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            tfidf = train_tfidf[row,tf_idf_feat.index(word)]
            sent_vec += (vec*tfidf)
            weight_sum += tfidf
        except:
            pass
    sent_vec/= weight_sum
    tfidf_sent_vec_train.append(sent_vec)
    row += 1


# In[ ]:


tf_idf_feat = tfidf_vect.get_feature_names()
tfidf_sent_vec_test = []
row = 0
w2v_model = Word2Vec(list_of_sent_test,min_count = 5,size = 50,workers = 4)
for sent in list_of_sent_test:
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            tfidf = test_tfidf[row,tf_idf_feat.index(word)]
            sent_vec += (vec*tfidf)
            weight_sum += tfidf
        except:
            pass
    sent_vec/= weight_sum
    tfidf_sent_vec_test.append(sent_vec)
    row += 1


# In[ ]:


np.where(np.isnan(tfidf_sent_vec_train))


# In[ ]:


tfidf_sent_vec_train = pd.DataFrame(tfidf_sent_vec_train)
tfidf_sent_vec_test = pd.DataFrame(tfidf_sent_vec_test)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(tfidf_sent_vec_train)
tfidf_sent_vec_train = imputer.transform(tfidf_sent_vec_train)
tfidf_sent_vec_test = imputer.transform(tfidf_sent_vec_test)


# In[ ]:


sc =  StandardScaler()
tfidf_w2v_train = sc.fit_transform(tfidf_sent_vec_train)
tfidf_w2v_test = sc.transform(tfidf_sent_vec_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(tfidf_w2v_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
print(random_model.best_estimator_)
pred = random_model.predict(tfidf_w2v_test)
acc = accuracy_score(y_test,pred)
print('accuracy is',acc*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


sc =  StandardScaler()
tfidf_w2v_train = sc.fit_transform(tfidf_sent_vec_train)
tfidf_w2v_test = sc.transform(tfidf_sent_vec_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)
random_model.fit(tfidf_w2v_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
print(random_model.best_estimator_)
pred = random_model.predict(tfidf_w2v_test)
acc = accuracy_score(y_test,pred)
print('accuracy is',acc*100)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test , pred)
print(confusion)
df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])
sns.heatmap(df_cm ,annot = True)
plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **Things to do in future**
# * Multicollinearity Check    

# In[ ]:




