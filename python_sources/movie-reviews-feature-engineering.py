#!/usr/bin/env python
# coding: utf-8

# ** Introduction: Movie Review Sentiment Analysis**
# 
# This notebook is for those who are new to machine learning and deep learning industry. I want to introduce the basic concept to learn data, understand the data using some statistics and virtualization tools in order to start machine learning instead of jumping to a complicated model. Any suggestion is much appreciated.
# 
# Sentiment analysis is an example of supervised machine learning task a labelled dataset which containing text documents and their labels is used for a train a classifier.

# **Objective:**
# 
# EDA/ feature engineering  for movie review sentiment analysis

# ** DATA**
# 
# The data is a movie review dataset for text classification. The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis. Reviews are classified into 5 following categories.
# 
#  * Negative
#  * Somewhat negative
#  * Neutral
#  * Somewhat positive
#  * Positive.

# **Import libraries :**

# In[ ]:


#basic 

import pandas as pd
import numpy as np
pd.set_option('max_colwidth',400)
import string

#Graph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')

#machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import hstack


#Deep Learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

#NLP
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
import textblob
import re


# **Read Data**

# In[ ]:


train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv' , sep="\t")
sub =pd.read_csv('../input/sampleSubmission.csv')

train.shape , test.shape


# In[ ]:


train.head()


# Each Sentence has been parsed into many phrases by the parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data. Let's look at sentence number two with it phrases.

# In[ ]:


train.loc[train['SentenceId'] == 2]


# * Let's check missing value

# In[ ]:


def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

missing_data_df = check_missing(train)
missing_data_test = check_missing(test)
print('Missing data in train set: \n' , missing_data_df.head())
print('\nMissing data in test set: \n'  ,missing_data_test.head())


# No missing data in both train and test set. Go Further and find something interesting in data.

# ** Class Imbalanced **

# In[ ]:


temp = train['Sentiment'].value_counts()

trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of Class Label in train dataset",
    xaxis=dict(
        title='Class Label',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Occurance of Class label',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Sentiment')


# The reviews are not evenly spread into categories.
# majority of the reviews are classified as "neutral" class.
# Ohh it's imbalanced data

# **Feature Engineering:**
# Here, I have explained Feature Engineering in deeply.
# 
# Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms works well. 

# 1 )New features are created from text feature.
# * Number of character in Phrase
# * Number of words in Phrase
# * Number of punctuation_count
# * Number of uppercase words 
# * Number of stopwords
# * Number of positive words in Phrase
# * Number of negative words in Phrase
# 
# 2 )Frequency distribution of Part of Speech Tags:
# * Noun Count
# * Verb Count
# * Adjective Count
# * Adverb Count
# * Pronoun Count
# 
# 3)Sentiment analysis using Textblob library
# * Subjectivity
# * polarity

# Let's first create features from a text. But, First of all, merge train and test set.

# In[ ]:


length = len(train)
df = pd.concat([train, test], axis=0)


# In[ ]:


# generate clean text from Phrase 
def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))

df['Clean_text'] = df['Phrase'].apply(lambda x : review_to_words(x))

df['Clean_text'].replace('', str('something'), inplace=True)


# In[ ]:


df['char_count'] = df['Phrase'].apply(len)
df['word_count'] = df['Phrase'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['Phrase'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['title_word_count'] = df['Phrase'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['Phrase'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
df["stopword_count"] = df['Phrase'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))


# We required to find positive words and negative words in order to create a feature number of positive words and the number of negative words in Phrase.

# Let's consider words of a phrase as positive words which classified into class Positive(label: 4)

# In[ ]:


positive = df['Clean_text'][df['Sentiment']== 4 ]

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, max_features = 6877)

pos_words = vectorizer.fit_transform(positive)
pos_words = pos_words.toarray()
pos= vectorizer.get_feature_names()
print ("Total number of positive words : " ,len(pos))

dist = np.sum(pos_words, axis=0)
postive_new= pd.DataFrame(dist)
postive_new.columns=['word_count']
postive_new['word'] = pd.Series(pos, index=postive_new.index)
top = postive_new.sort_values(['word_count'] , ascending = False )


# Let's consider words of a phrase as negative words which classified into class Negative( class label : 0)

# In[ ]:


negative=df['Clean_text'][df['Sentiment']== 0]

neg_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, max_features = 6891)

neg_words = neg_vectorizer.fit_transform(negative)
neg_words = neg_words.toarray()
neg= neg_vectorizer.get_feature_names()
print ("Total number of negative words :",len(neg))

dist = np.sum(neg_words, axis=0)
negative_new= pd.DataFrame(dist)
negative_new.columns=['word_count']
negative_new['word'] = pd.Series(neg, index=negative_new.index)
top_neg = negative_new.sort_values(['word_count'] , ascending = False )


# Now we count the number of positive words and number of negative words in a Phrase.
# 
# one more feature is : Ratio of positive word count to negative word count

# In[ ]:


def count_word(x , pos_tag):
    cnt = 0
    if pos_tag:
        for e in x.split():
            if e in pos:
                cnt = cnt + 1
    else:
        for e in x.split():
            if e in neg:
                cnt = cnt + 1
    return cnt
    
df['pos_cnt'] = df['Clean_text'].apply(lambda x : count_word(x , pos_tag = True))
df['neg_cnt'] = df['Clean_text'].apply(lambda x : count_word(x, pos_tag = False))

df['Ratio'] = df['pos_cnt'] / (df['neg_cnt']+0.0001)


# Let's create Frequency distribution of Part of Speech Tags features:

# In[ ]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    wiki = textblob.TextBlob(x)
    for tup in wiki.tags:
        ppo = list(tup)[1]
        if ppo in pos_family[flag]:
            cnt += 1

    return cnt

df['noun_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'pron'))


# Let's create the feature using textblob package. TextBlob is a Python library for processing textual data. It provides a consistent API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and more. 

# In[ ]:


def getSentFeat(s , polarity):
    sent = textblob.TextBlob(s).sentiment
    if polarity:
        return sent.polarity
    else :
        return sent.subjectivity
    
df['polarity'] = df['Phrase'].apply(lambda x: getSentFeat(x , polarity=True))
df['subjectivity'] = df['Phrase'].apply(lambda x: getSentFeat(x , polarity=False))


# In[ ]:


#separate train and test data
train = df[:length]
test = df[length:]
train.shape, test.shape


# In[ ]:


train.describe()


# In[ ]:



plt.figure(figsize=(12,6))
plt.subplot(121)
sns.violinplot(y='pos_cnt',x='Sentiment', data=train,split=True)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('# of Positive words ', fontsize=12)
plt.title("Number of Positive word in each review", fontsize=15)

plt.subplot(122)
sns.violinplot(y='neg_cnt',x='Sentiment', data=train,split=True)
plt.xlabel('Class label', fontsize=12)
plt.ylabel('# of negative words', fontsize=12)
plt.title("Number of Negative words in each review", fontsize=15)

plt.show()


# A Violin Plot is used to visualise the distribution of the data and its probability density. Let's check correlation with target Sentiment.

# In[ ]:


f,ax = plt.subplots(figsize=(15,15))    #correlation between numerical values' maps
sns.heatmap(train.corr() , annot = True, linewidths = .5, fmt= '.1f', ax=ax , vmin=-1, vmax=1)
plt.legend()
plt.show() 


# **Baseline Model**

# In[ ]:


# Standardize numeric feature

ss = StandardScaler()
num_col = [ 'pos_cnt', 'neg_cnt' , 'Ratio','polarity','subjectivity' ,
            'char_count' , 'word_count' , 'word_density' , 'punctuation_count','title_word_count' ,
           'upper_case_word_count' ,'stopword_count' ,
            'adv_count' ,'verb_count','adj_count', 'pron_count' , 'noun_count']

X_num = ss.fit_transform(train[num_col].fillna(-1).clip(0.0001 , 0.99999))

y = train['Sentiment']


# In[ ]:


# vectorization of text data

count_vect = CountVectorizer(analyzer='word', ngram_range=(1,2))

X_txt = count_vect.fit_transform(train['Phrase'])


# Let's train a model on numeric feature and see which feature is work best for model training.

# In[ ]:



X_train, X_val, Y_train, Y_val = train_test_split(X_num, y, test_size=0.10, random_state=1234)

clf = LogisticRegression(C=3)

clf.fit(X_train,Y_train)
clf.score(X_val,Y_val)


# Observe feature importance

# In[ ]:


plt.figure(figsize=(16,22))
plt.suptitle("Feature importance",fontsize=20)
gridspec.GridSpec(3,2)
plt.subplots_adjust(hspace=0.4)
plt.subplot2grid((3,2),(0,0))
sns.barplot(num_col,clf.coef_[0],color=color[0])
plt.title("class : Negative",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(0,1))
sns.barplot(num_col,clf.coef_[1] , color=color[1])
plt.title("class : Somewhat negative",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(1,0))
sns.barplot(num_col,clf.coef_[2],color=color[2])
plt.title("class : Neutral",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(1,1))
sns.barplot(num_col,clf.coef_[3],color=color[3])
plt.title("class : Somewhat positive",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(2,0))
sns.barplot(num_col,clf.coef_[4],color=color[4])
plt.title("class : Positive",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.show()


# Positive word count, negative word count, the ratio of positive word count to negative word count , polarity and subjectivity feature sound great for the model.

# Let's train model on both feature text and numeric feature.

# In[ ]:


x = hstack((X_num,X_txt)).tocsr()

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.10, random_state=1234)

clf1 = LogisticRegression(C = 3)

clf1.fit(X_train,Y_train)
clf1.score(X_val,Y_val)


# **Deep learning Baseline Model**

# In[ ]:


#pre-processing of data for keras model

train_DL = train.drop(['PhraseId' , 'SentenceId' , 'Sentiment'] , axis =1)
y = train['Sentiment']

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

X_train, X_val, Y_train, Y_val = train_test_split(train_DL, y_ohe, test_size=0.10, random_state=1234)

tk = Tokenizer(lower = True, filters='', num_words= 15000)
tk.fit_on_texts(train_DL['Phrase'])

train_tokenized = tk.texts_to_sequences(X_train['Phrase'])
valid_tokenized = tk.texts_to_sequences(X_val['Phrase'])

max_len = 80
X_train_txt = pad_sequences(train_tokenized, maxlen = max_len)
X_valid_txt = pad_sequences(valid_tokenized, maxlen = max_len)

X_num_train = ss.transform(X_train[num_col].fillna(-1).clip(0.0001 , 0.99999))
X_num_valid = ss.transform(X_val[num_col].fillna(-1).clip(0.0001 , 0.99999))


# In[ ]:


inp = Input(shape = (max_len,))
input_num = Input((len(num_col), ))

x = Embedding(15000 , 100 ,mask_zero=True)(inp)
x = LSTM(128, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)(x)
x = LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False)(x)

x_num = Dense(64, activation="relu")(input_num)   
X_num = Dropout(0.2)(x_num)
X_num = Dense(32, activation = "relu")(X_num)

xx = concatenate([x_num, x])
xx = BatchNormalization()(xx)
xx = Dropout(0.1)(Dense(20, activation='relu') (xx))

outp = Dense(5, activation = "softmax")(xx)

model = Model(inputs = [inp,input_num], outputs = outp)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_train_txt,X_num_train], Y_train, validation_data=([X_valid_txt,X_num_valid], Y_val),
         epochs=6, batch_size=128, verbose=1)

accuracy = model.evaluate([X_valid_txt,X_num_valid], Y_val )[1]
accuracy


# **Next steps:**
# 
# * Add Embedding  features vector
# * hyperparameter Tuning
# * Stacking 
# 
# To be continued. Please stay tuned!

# In[ ]:




