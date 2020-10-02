#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import *
from plotly import tools
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import re
# Natural Language Tool Kit 
import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from collections import Counter
import cufflinks as cf
import string 
get_ipython().system('pip install simpletransformers')
from simpletransformers.classification import ClassificationModel
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train_df[train_df["target"] == 0]["text"].values[1]


# In[ ]:


print(train_df)


# In[ ]:


y_train=train_df['target']
x_train = train_df.drop(labels=["target"],axis=1)
y_train.value_counts()
sns.countplot(y_train)


# In[ ]:


x_train.isnull().any().describe()
test_df.isnull().any().describe()


# In[ ]:


missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=train_df[missing_cols].isnull().sum().index, y=train_df[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=test_df[missing_cols].isnull().sum().index, y=test_df[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)

plt.show()


# Locations
# 

# In[ ]:


print(f'Number of unique values in location = {train_df["location"].nunique()} (Training) - {test_df["location"].nunique()} (Test)')


# In[ ]:


cnt_ = train_df['location'].value_counts()
cnt_.reset_index()
cnt_ = cnt_[:20,]
trace1 = go.Bar(
                x = cnt_.index,
                y = cnt_.values,
                name = "Number of tweets in dataset according to location",
                marker = dict(color = 'rgba(200, 74, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )

data = [trace1]
layout = go.Layout(barmode = "group",title = 'Number of tweets in dataset according to location')
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# Locations vs Targets

# In[ ]:


train1_df = train_df[train_df["target"]==1]
train0_df = train_df[train_df["target"]==0]
cnt_1 = train1_df['location'].value_counts()
cnt_1.reset_index()
cnt_1 = cnt_1[:20,]

cnt_0 = train0_df['location'].value_counts()
cnt_0.reset_index()
cnt_0 = cnt_0[:20,]


# In[ ]:


trace1 = go.Bar(
                x = cnt_1.index,
                y = cnt_1.values,
                name = "real disaste",
                marker = dict(color = 'rgba(255, 74, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )
trace0 = go.Bar(
                x = cnt_0.index,
                y = cnt_0.values,
                name = "fake disaster",
                marker = dict(color = 'rgba(79, 82, 97, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )


data = [trace0,trace1]
layout = go.Layout(barmode = 'stack',title = 'Number of tweets in dataset according to location')
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# Keywords
# 

# In[ ]:


print(f'Number of unique values in keyword = {train_df["keyword"].nunique()} (Training) - {test_df["keyword"].nunique()} (Test)')


# In[ ]:


## Distribution per keywords
cnt2 = train_df['keyword'].value_counts()
cnt2.reset_index()
cnt2 = cnt_[:30,]
trace1 = go.Bar(
                x = cnt2.index,
                y = cnt2.values,
                name = "Number of tweets in dataset according to keyword",
                marker = dict(color = 'rgba(200, 74, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )

data = [trace1]
layout = go.Layout(barmode = "group",title = 'Number of tweets in dataset according to keyword')
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:



cnt_1 = train1_df['keyword'].value_counts()
cnt_1.reset_index()
#cnt_1 = cnt_1[:30,]

cnt_0 = train_df['keyword'].value_counts()
cnt_0.reset_index()
#cnt_0 = cnt_0[:30,]


# In[ ]:


trace1 = go.Bar(
                x = cnt_1.index,
                y = cnt_1.values,
                name = "real disaste",
                marker = dict(color = 'rgba(255, 74, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )
trace0 = go.Bar(
                x = cnt_0.index,
                y = cnt_0.values,
                name = "fake disaster",
                marker = dict(color = 'rgba(79, 82, 97, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )


data = [trace0,trace1]
layout = go.Layout(barmode = 'stack',title = 'Number of tweets in dataset according to location')
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:


train_df['target_mean'] = train_df.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(8, 72), dpi=100)

sns.countplot(y=train_df.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=train_df.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

train_df.drop(columns=['target_mean'], inplace=True)


# In[ ]:


#Fill NA 

for df in [train_df, test_df]:
    for col in ['keyword', 'location']:
        df[col] = df[col].fillna(f'no_{col}')


# MetaData features
# 

# In[ ]:


#tweets lenght - characters count
#train_df['length']= train_df['text'].apply(len)


# In[ ]:


train_df['tweet_len']= train_df['text'].apply(len)
train_df['text']
test_df['tweet_len']= test_df['text'].apply(len)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
for label, group in train_df.groupby('target'):
    sns.distplot(group['text'].str.len(), label=str(label), ax=ax)
plt.xlabel('# of characters')
plt.ylabel('density')
plt.legend()
sns.despine()


# In[ ]:


data = [
    go.Box(
        y=train_df[train_df['target']==0]['tweet_len'],
        name='Fake'
    ),
    go.Box(
        y=train_df[train_df['target']==1]['tweet_len'],
        name='Real'
    )
]
layout = go.Layout(
    title = 'Comparison of text length in Tweets '
)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


#word count
train_df['word_count']= train_df['text'].apply(lambda x: len(str(x).split()))
test_df['word_count']= test_df['text'].apply(lambda x: len(str(x).split()))


# In[ ]:


data = [
    go.Box(
        y=train_df[train_df['target']==0]['word_count'],
        name='Fake'
    ),
    go.Box(
        y=train_df[train_df['target']==1]['word_count'],
        name='Real'
    )
]
layout = go.Layout(
    title = 'Comparison of word_count in Tweets '
)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


# unique_word_count
train_df['unique_word_count'] = train_df['text'].apply(lambda x: len(set(str(x).split())))
test_df['unique_word_count'] = test_df['text'].apply(lambda x: len(set(str(x).split())))


# In[ ]:


data = [
    go.Box(
        y=train_df[train_df['target']==0]['unique_word_count'],
        name='Fake'
    ),
    go.Box(
        y=train_df[train_df['target']==1]['unique_word_count'],
        name='Real'
    )
]
layout = go.Layout(
    title = 'Comparison of unique word_count in Tweets '
)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


# punctuation_count
train_df['punctuation_count'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test_df['punctuation_count'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[ ]:


# mean of word lenght
train_df['mean_word_length'] = train_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df['mean_word_length'] = test_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[ ]:


METAFEATURES = ['word_count', 'unique_word_count', 'tweet_len', 'mean_word_length','punctuation_count']
DISASTER_TWEETS = train_df['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(train_df.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0], color='green')
    sns.distplot(train_df.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='red')

    sns.distplot(train_df[feature], label='Training', ax=axes[i][1])
    sns.distplot(test_df[feature], label='Test', ax=axes[i][1])
    
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

plt.show()


# NGRAMs
# option1 :
# 

# In[ ]:


def generate_ngrams (text,n=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams= zip(*[token[i:] for i in range(n)])
    return[' '.join(ngram) for ngram in ngrams]
N=100


# In[ ]:


#uni
from collections import defaultdict
disaster_unigrams= defaultdict(int)
nondisaster_unigrams = defaultdict(int)

for tweet in train_df[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet):
        disaster_unigrams[word]+=1
    
for tweet in train_df[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet):
        nondisaster_unigrams[word]+=1
        
df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])


# In[ ]:


# bigrams
disaster_bigrams = defaultdict(int)
nondisaster_bigrams = defaultdict(int)

for tweet in train_df[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n=2):
        disaster_bigrams[word] += 1
        
for tweet in train_df[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n=2):
        nondisaster_bigrams[word] += 1
        
df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])


# In[ ]:


#trigram
disaster_trigrams = defaultdict(int)
nondisaster_trigrams = defaultdict(int)

for tweet in train_df[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n=3):
        disaster_trigrams[word] += 1
        
for tweet in train_df[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n=3):
        nondisaster_trigrams[word] += 1
        
df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])


# In[ ]:


import re

def generate_ngrams(text, n=1):
    # Convert to lowercases
    s = s.str.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.str())
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

plt.tight_layout()

sns.barplot(y=df_disaster_unigrams[0].values[:N], x=df_disaster_unigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_unigrams[0].values[:N], x=df_nondisaster_unigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common unigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common unigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
plt.tight_layout()

sns.barplot(y=df_disaster_bigrams[0].values[:N], x=df_disaster_bigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_bigrams[0].values[:N], x=df_nondisaster_bigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(20, 50), dpi=100)

sns.barplot(y=df_disaster_trigrams[0].values[:N], x=df_disaster_trigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_trigrams[0].values[:N], x=df_nondisaster_trigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=11)

axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# import re
# from nltk.util import ngrams
# 
# train_df['text'] = train_df['text'].str.lower()
# s = re.sub(r'[^a-zA-Z0-9\s]', ' ',train_df['text'].str)
# tokens = [token for token in train_df.text.str.split(" ") if token != ""]
# output = list(ngrams(tokens, 1))
# output

# > - GLOVE 1 **

# In[ ]:


def build_vocab(tweets):
    vocab = {}        
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# In[ ]:


train_tweets = train_df['text'].apply(lambda s: s.split()).values
train_vocab = build_vocab(train_tweets)
test_tweets = test_df['text'].apply(lambda s: s.split()).values
test_vocab = build_vocab(test_tweets)
corpus_tweets=df['text'].apply(lambda s: s.split()).values
corpus_vocab=build_vocab(corpus_tweets)


# In[ ]:


#embedding Glove
embeddings_glove = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)


# In[ ]:


#checking coverage
def check_coverage(vocab, embeddings, embeddings_name, dataset_name):
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word]= embeddings[word]
            n_covered+=vocab[word]
        except:
            oov[word]=vocab[word]
            n_oov+=vocab[word]
            
    vocab_coverage=len(covered)/len(vocab)
    text_covrage= (n_covered/(n_covered+n_oov))
    print('{} Embeddings cover {:.2%} of {} vocab'.format(embeddings_name, vocab_coverage, dataset_name))
    #sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    #return sorted_oov

train_oov_glove = check_coverage(train_vocab, embeddings_glove, 'GloVe', 'Training')
test_oov_glove = check_coverage(test_vocab, embeddings_glove, 'GloVe', 'Test')


# In[ ]:


def clean(tweet):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    url = re.compile(r'https?://\S+|www\.\S+')
    # Punctuations at the start or end of words    
    for punctuation in "#@!?()[]*%":
        tweet = tweet.replace(punctuation, f' {punctuation} ').strip()
    tweet = tweet.replace('...', ' ... ').strip()
    tweet = tweet.replace("'", " ' ").strip()   
    #webpages
    tweet= url.sub(r'',tweet)
    #emojis
    tweet = emoji_pattern.sub(r'', tweet)
    #spellchecker
    #tweet=correct_spellings(tweet)
    
    url.sub(r'',tweet)
    #https
    if 'http' not in tweet:
        tweet = tweet.replace(":", " : ").strip() 
        tweet = tweet.replace(".", " . ").strip() 
    return  tweet


# In[ ]:


train_df['text'] = train_df['text'].apply(lambda x : clean(x))
test_df['text'] = test_df['text'].apply(lambda x : clean(x))
df['text'] = df['text'].apply(lambda x : clean(x))


# In[ ]:


train_tweets_cleaned = train_df['text'].apply(lambda s: s.split()).values
train_vocab_cleaned = build_vocab(train_tweets_cleaned)
test_tweets_cleaned = test_df['text'].apply(lambda s: s.split()).values
test_vocab_cleaned = build_vocab(test_tweets_cleaned)
corpus_tweets=df['text'].apply(lambda s: s.split()).values
corpus_vocab=build_vocab(corpus_tweets)

train_oov_glove = check_coverage(train_vocab_cleaned, embeddings_glove, 'GloVe', 'Training')
test_oov_glove = check_coverage(test_vocab_cleaned, embeddings_glove, 'GloVe', 'Test')


# Spellchecker
# 

# #!pip install pyspellchecker
# from spellchecker import SpellChecker
# 
# spell = SpellChecker()
# def correct_spellings(tweet):
#     corrected_text = []
#     misspelled_words = spell.unknown(tweet.split())
#     for word in tweet.split():
#         if word in misspelled_words:
#             corrected_text.append(spell.correction(word))
#         else:
#             corrected_text.append(word)
#     return " ".join(corrected_text)
#         
# tweet = "corect me plese"
# correct_spellings(tweet)

# In[ ]:


df=pd.concat([train_df,test_df])
df.shape


# In[ ]:


def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus


# In[ ]:


from tqdm import tqdm
from nltk.tokenize import word_tokenize
stop=set(stopwords.words('english'))
corpus=create_corpus(df)
#corpus_test=create_corpus(test_df)


# In[ ]:


embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r')  as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


# In[ ]:


MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[ ]:


word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[ ]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec


# Readability features

# In[ ]:


tqdm.pandas()
get_ipython().system('pip install textstat')
import textstat


# In[ ]:


def plot_readability(a,b,title,bins=0.1,colors=['mediumvioletred', 'goldenrod']):
    trace1 = ff.create_distplot([a,b], [" disaster","Not disaster"], bin_size=bins, colors=colors, show_rug=False)
    trace1['layout'].update(title=title)
    py.iplot(trace1, filename='Distplot')
    table_data= [["Statistical Measures"," Not real disaster tweets","real disaster tweets"],
                ["Mean",np.mean(a),np.mean(b)],
                ["Standard Deviation",pstdev(a),pstdev(b)],
                ["Variance",pvariance(a),pvariance(b)],
                ["Median",median(a),median(b)],
                ["Maximum value",max(a),max(b)],
                ["Minimum value",min(a),min(b)]]
    trace2 = ff.create_table(table_data)
    py.iplot(trace2, filename='Table')


# In[ ]:


fre_notreal=np.array(train_df["text"][train_df["target"] == 0].progress_apply(textstat.flesch_reading_ease))
fre_real = np.array(train_df["text"][train_df["target"] == 1].progress_apply(textstat.flesch_reading_ease))
plot_readability(fre_notreal,fre_real,"Flesch Reading Ease",20)


# In[ ]:


fkg_notreal = np.array(train_df["text"][train_df["target"] == 0].progress_apply(textstat.flesch_kincaid_grade))
fkg_real = np.array(train_df["text"][train_df["target"] == 1].progress_apply(textstat.flesch_kincaid_grade))
plot_readability(fkg_notreal,fkg_real,"Flesch Kincaid Grade",4,['#C1D37F','#491F21'])


# Topic modeling

# LDA
# 

# In[ ]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
parser = English()
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# In[ ]:


notreal_text = train_df["text"][train_df["target"] == 0].progress_apply(spacy_tokenizer)
real_text = train_df["text"][train_df["target"] == 1].progress_apply(spacy_tokenizer)
#count vectorization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer_notreal = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
notreal_vectorized = vectorizer_notreal.fit_transform(notreal_text)
vectorizer_real = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
real_vectorized = vectorizer_real.fit_transform(real_text)


# In[ ]:


from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD


lda_notreal = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
notreal_lda = lda_notreal.fit_transform(notreal_vectorized)
lda_real = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
real_lda = lda_real.fit_transform(real_vectorized)


# In[ ]:


def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 


# In[ ]:



selected_topics(lda_notreal, vectorizer_notreal)


# In[ ]:


import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_notreal, notreal_vectorized, vectorizer_notreal, mds='tsne')
dash


# In[ ]:


pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_real, real_vectorized, vectorizer_real, mds='tsne')
dash


# 

# In[ ]:


tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())


# Model on TFidf

# In[ ]:


from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_df,train_df['target'].values,test_size=0.15)


# In[ ]:


#model = linear_model.LogisticRegression(C=5., solver='sag')
#model.fit(X_train, y_train)


# In[ ]:


train_y = train_df["target"].values

def runModel(train_X, train_y, test_X, test_y, test_X2):
    model = linear_model.LogisticRegression(C=5., solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:,1]
    pred_test_y2 = model.predict_proba(test_X2)[:,1]
    return pred_test_y, pred_test_y2, model

print("Building model.")
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break


# In[ ]:


runModel(dev_X, dev_y, val_X, val_y, test_tfidf)


# In[ ]:


from tqdm import tqdm
def threshold_search(y_true, y_proba):
#reference: https://www.kaggle.com/hung96ad/pytorch-starter
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.001 for i in range(1000)]):
        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
search_result = threshold_search(val_y, pred_val_y)
search_result


# In[ ]:


#sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
#y_pre=model.predict(test_tfidf)
#y_pre=np.round(y_pre).astype(int).reshape(3263)
#sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})
#sub.to_csv('submission.csv',index=False)


# In[ ]:


print("F1 score at threshold {0} is {1}".format(0.381, metrics.f1_score(val_y, (pred_val_y>0.381).astype(int))))
print("Precision at threshold {0} is {1}".format(0.381, metrics.precision_score(val_y, (pred_val_y>0.381).astype(int))))
print("recall score at threshold {0} is {1}".format(0.381, metrics.recall_score(val_y, (pred_val_y>0.381).astype(int))))


# In[ ]:


import eli5
eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')


# Simple transformers
# 

# In[ ]:


get_ipython().system('pip install simpletransformers')
import os, re, string
import random

import numpy as np
import pandas as pd
import sklearn

import torch

from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[ ]:



seed = 1337
torch.cuda.manual_seed(seed)
bert_uncased = ClassificationModel('bert', 'bert-large-uncased') 


# In[ ]:


custom_args = {'fp16': False, # not using mixed precision 
               'train_batch_size': 4, # default is 8
               'gradient_accumulation_steps': 2,
               'do_lower_case': True,
               'learning_rate': 1e-05, # using lower learning rate
               'overwrite_output_dir': True, # important for CV
               'num_train_epochs': 2} # default is 1


# In[ ]:


train_df=train_df.iloc[:,3:5]


# In[ ]:


train_y = train_df["target"].values
model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args) 
model.train_model(train_df)


# In[ ]:


#test_df= test_df.iloc[:,0]
test_df.head()

predictions, raw_outputs = model.predict(test_df['text'])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = predictions
sample_submission.to_csv("submission.csv", index=False)


# Model deep learning seq
# 
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from keras.optimizers import Adam

model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)


model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
optimizer= Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer= optimizer,metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


train=tweet_pad[:train_df.shape[0]]
test=tweet_pad[train_df.shape[0]:]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,train_df['target'].values,test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# In[ ]:


history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)


# Prediction and submission

# In[ ]:





# In[ ]:


test_df.shape
test.shape


# In[ ]:



#sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
#y_pre=model.predict(test)
#y_pre=np.round(y_pre).astype(int).reshape(3263)
#sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})
#sub.to_csv('submission.csv',index=False)

