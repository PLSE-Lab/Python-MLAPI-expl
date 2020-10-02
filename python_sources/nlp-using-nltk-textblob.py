#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from collections import defaultdict
from collections import  Counter
from textblob import TextBlob 
from nltk.corpus import words
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import nltk
from nltk.corpus import wordnet
from nltk.corpus import words
from string import punctuation
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
import re
from html import unescape
from wordcloud import WordCloud, STOPWORDS

np.random.seed(0);


# In[ ]:


get_ipython().system(' pip install wordsegment -q')
from wordsegment import load, segment
load()


# In[ ]:


style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


TRAIN = '/kaggle/input/nlp-getting-started/train.csv'
TEST = '/kaggle/input/nlp-getting-started/test.csv'
SAMPLE_SUBMISSION = '/kaggle/input/nlp-getting-started/sample_submission.csv'

ticker = ticker.EngFormatter(unit='')
words = set(nltk.corpus.words.words())


# In[ ]:


def get_polarity(text):
    try:
        pol = TextBlob(text).sentiment.polarity
    except:
        pol = 0.0
    return pol

def statistical_features(df,text_col):
    df['word_count'] = df[text_col].apply(lambda x : len(x.split()))
    df['char_count'] = df[text_col].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['punc_count'] = df[text_col].apply(lambda x : len([a for a in x if a in punctuation]))
    df['stop_count'] = df[text_col].apply(lambda x : len([a for a in x if a in STOPWORDS]))
    df['polarity'] = df[text_col].apply(get_polarity)
    df["has_hashtag"] = df[text_col].apply(lambda text: 1 if "#" in text else 0)
    df["has_mention"] = df[text_col].apply(lambda text: 1 if "@" in text else 0)
    df["has_exclamation"] = df[text_col].apply(lambda text: 1 if "!" in text else 0)
    df["has_question"] = df[text_col].apply(lambda text: 1 if "?" in text else 0)
    df["has_url"] = df[text_col].apply(lambda text: 1 if "http" in text else 0)
    df["has_RT"] = df[text_col].apply(lambda text: 1 if "RT" in text else 0)
    return df

def get_data(df,filter_val):
    return df[df['target']==filter_val]

def create_corpus(df,target):
    corpus=[]
    
    for x in df[df['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i.lower())
    return corpus

def get_Distribution(corpus,values):
    dic=defaultdict(int)
    for word in corpus:
        if word in values:
            dic[word]+=1
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)
    return top

def most_common_words(corpus):
    counter=Counter(corpus)
    most=counter.most_common()
    x=[]
    y=[]
    for word,count in most[:100]:
        if (word not in STOPWORDS and word not in punctuation) :
            x.append(word)
            y.append(count)
    return x,y

def get_top_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


train_df  = pd.read_csv(TRAIN,encoding='utf-8-sig')
test_df  = pd.read_csv(TEST,encoding='utf-8-sig')

print('There are {} rows and {} columns in train'.format(train_df.shape[0],train_df.shape[1]))
print('There are {} rows and {} columns in test'.format(test_df.shape[0],test_df.shape[1]))


# In[ ]:


plt.style.use('fivethirtyeight')
with plt.xkcd():
    fig = plt.figure(figsize = (7, 5))
    ax = fig.add_subplot(111)
    ax = sns.countplot(data=train_df,x='target')
    ax.set_ylabel('Sample Size')
    ax.set_xlabel('Target Class')
    ax.yaxis.set_major_formatter(ticker)
    ax.set_title('Target Class Population Distribution')
    ax= plt.xticks([0,1], ['Fake Tweet', 'Disaster Tweet'])
    plt.grid()


# In[ ]:


plt.style.use('fivethirtyeight')
with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax = train_df.keyword.value_counts()[:20].plot(kind='barh', color='#1B86BA')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Keywords')
    ax.set_title('Top 20 keywords in text')
    plt.grid()


# In[ ]:


plt.style.use('fivethirtyeight')
with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax = train_df.location.value_counts()[:20].plot(kind='barh', color='#1B86BA')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Location')
    ax.set_title('Top 20 Location in text')
    plt.grid()


# In[ ]:


train_df = statistical_features(train_df,'text')
test_df = statistical_features(test_df,'text')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['word_count'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['word_count'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Word Count Distribution')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['char_count'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['char_count'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Character Count Distribution')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['word_density'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['word_density'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Word Density Distribution')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['punc_count'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['punc_count'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Punctuation Count Distribution')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['stop_count'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['stop_count'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Stopwords Distribution')


# In[ ]:


with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = sns.distplot(get_data(train_df,0)['polarity'], fit=norm, kde=False,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2 = sns.distplot(get_data(train_df,1)['polarity'], fit=norm, kde=False,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Polarity Distribution')


# In[ ]:


corpus_0 = create_corpus(train_df,0)
corpus_1 =create_corpus(train_df,1)


# In[ ]:


top_fake_x, top_fake_y = zip(*get_Distribution(corpus_0,STOPWORDS)[:30])
top_disaster_x,top_disaster_y = zip(*get_Distribution(corpus_1,STOPWORDS)[:30])

with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1.barh(top_fake_x,top_fake_y,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2.barh(top_disaster_x,top_disaster_y,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Top 30 Stop Word')


# In[ ]:


top_fake_x, top_fake_y = zip(*get_Distribution(corpus_0,punctuation)[:30])
top_disaster_x,top_disaster_y = zip(*get_Distribution(corpus_1,punctuation)[:30])

with plt.xkcd():
    fig =plt.figure(figsize=(15,8))
    ax1 = plt.subplot(121,frameon=True)
    ax1.barh(top_fake_x,top_fake_y,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2.barh(top_disaster_x,top_disaster_y,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Top 30 Punctions')


# In[ ]:


top_fake_x, top_fake_y = most_common_words(corpus_0)
top_disaster_x,top_disaster_y = most_common_words(corpus_1)

with plt.xkcd():
    fig =plt.figure(figsize=(20,12))
    ax1 = plt.subplot(121,frameon=True)
    ax1.barh(top_fake_x,top_fake_y,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2.barh(top_disaster_x,top_disaster_y,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Most Common Word\n(Exclusing Punctuation & StopWords)')


# In[ ]:


top_fake_x, top_fake_y = map(list,zip(*get_top_bigrams(corpus_0)[:30]))
top_disaster_x,top_disaster_y = map(list,zip(*get_top_bigrams(corpus_1)[:30]))

with plt.xkcd():
    fig =plt.figure(figsize=(30,18))
    ax1 = plt.subplot(121,frameon=True)
    ax1.barh(top_fake_x,top_fake_y,color="#1B86BA")
    ax1.set_title('Fake Tweet')
    ax2 = plt.subplot(122,frameon=True)
    ax2.barh(top_disaster_x,top_disaster_y,color="#E36149")
    ax2.set_title('Disaster Tweet')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Most Common Bi-Gram')


# In[ ]:


def remove_mention_hashtag(text):
    """https://medium.com/analytics-vidhya/working-with-twitter-data-b0aa5419532"""
#     cleantext = re.sub(r"#(\w+)", '', text, flags=re.MULTILINE).strip()
    cleantext = re.sub(r"@(\w+)", '', text, flags=re.MULTILINE).strip()
    return cleantext

def html_escape(text):
    """https://docs.python.org/3/library/html.html"""
    return unescape(text)

def web_urls(text):
    """https://stackoverflow.com/a/11332543"""
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text).strip()

def clean_text(text):
    """https://stackoverflow.com/a/57030875"""
    text_nonum = re.sub(r'\d+', '', text)
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in punctuation])
    return text_nopunct

add_stopwords = ["http","https","amp","rt","t","c","the"]

def stopWords(text):
    """https://medium.com/analytics-vidhya/working-with-twitter-data-b0aa5419532"""
    return ' '.join(token.lower() for token in word_tokenize(text) if 
                    ((token.lower() not in STOPWORDS) and (token.lower() not in add_stopwords)
                    ))
def removeNonAscii(s): 
    """https://stackoverflow.com/a/1342373"""
    return "".join(i for i in s if ord(i)<128)

def remove_double_space(text):
    """https://stackoverflow.com/a/57030875"""
    return re.sub('\s+', ' ', text).strip()

def remove_len_words(text):
    return re.sub(r'\b\w{1,2}\b', '', text)

def corrections(text):
    return TextBlob(text).correct()

def clean_dataframe(df,clean_field):
    df['cleaned_text'] = df[clean_field].apply(lambda x: html_escape(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: remove_mention_hashtag(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: web_urls(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: removeNonAscii(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: stopWords(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: clean_text(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: remove_len_words(x))
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: remove_double_space(x))
    return df


# In[ ]:


cleaned_train_df = clean_dataframe(train_df,'text')
cleaned_test_df = clean_dataframe(test_df,'text')


# In[ ]:


from pprint import pprint
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


# In[ ]:


trainIndex, testIndex = list(), list()
for i in range(cleaned_train_df.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
        
trainData = cleaned_train_df.loc[trainIndex]
testData = cleaned_train_df.loc[testIndex]

cv = CountVectorizer(stop_words = 'english',strip_accents='ascii')
vect = cv.fit(cleaned_train_df.cleaned_text)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words = 'english',strip_accents='ascii')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=0)),
])
pipeline.fit(trainData.cleaned_text,trainData.target);
y_pred = pipeline.predict(testData.cleaned_text)
f1_score(testData.target, y_pred, average='micro')


# In[ ]:


def wordCloud(df1,tex1,df2,tex2,title,feature='cleaned_text'):
    all_words_1 = ' '.join([text for text in df1[feature]])
    wordcloud1 = WordCloud(width=1000, height=900, random_state=21, max_font_size=110).generate(all_words_1)
    all_words_2 = ' '.join([text for text in df2[feature]])
    wordcloud2 = WordCloud(width=1000, height=900, random_state=21, max_font_size=110).generate(all_words_2)
    
    fig =plt.figure(figsize=(30,12))
    ax1 = plt.subplot(121,frameon=True)
    ax1 = plt.imshow(wordcloud1, interpolation="bilinear")
    ax1 = plt.axis('off')
    ax1 = plt.title(tex1)
    ax2 = plt.subplot(122,frameon=True)
    ax2 = plt.imshow(wordcloud2, interpolation="bilinear")
    ax2 = plt.title(tex2)
    ax2 = plt.axis('off')
    fig.suptitle(f'Word Cloud\n{title} Set')


# In[ ]:


wordCloud(cleaned_train_df,'Train Tweet',cleaned_test_df,'Test Tweet','Train vs Validation','cleaned_text')


# In[ ]:


wordCloud(cleaned_train_df[cleaned_train_df.target==0],'Fake Tweet',cleaned_train_df[cleaned_train_df.target==1],'Disaster Tweet','Fake vs Disaster from Training Set','cleaned_text')


# In[ ]:


"""https://www.machinelearningplus.com/nlp/lemmatization-examples-python/"""
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()

cleaned_train_df['lemmatize_text'] = cleaned_train_df['cleaned_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(x)]))
cleaned_test_df['lemmatize_text'] = cleaned_test_df['cleaned_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(x)]))


# In[ ]:


wordCloud(cleaned_train_df,'Train Tweet(Lemmatized)',cleaned_test_df,'Test Tweet(Lemmatized)','Train vs Validation','lemmatize_text')


# In[ ]:


wordCloud(cleaned_train_df[cleaned_train_df.target==0],'Fake Tweet(Lemmatized)',cleaned_train_df[cleaned_train_df.target==1],'Disaster Tweet(Lemmatized)','Fake vs Disaster from Training Set','lemmatize_text')


# In[ ]:


def normalization(df,clean_field):
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bmom\b', 'mother', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bomg\b', 'oh my god', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\blol\b', 'laugh out loud', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bhaha\b', 'laugh', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bgovt\b', 'government', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\blmao\b', 'laughing my ass off', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bmilitants\b', 'militant', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bguys\b', 'guy', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\binfo\b', 'information', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bphotos\b', 'photo', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bwomens\b', 'women', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bbusinesses\b', 'business', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bppl\b', 'people', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\btraumatised\b', 'trauma', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bmadhya pradesh\b', 'madhya_pradesh', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\byearold\b', 'year old', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\bpanicking\b', 'panic old', x))
    df['cleaned_text'] = df[clean_field].apply(lambda x: re.sub(r'\busa|libya|pamela geller|bob apocalypse wither|mod|tonto|faux leather|enugu|calgary|nws|apc|refugio|cameroon\b', '', x))
    return df['cleaned_text']


# In[ ]:


cleaned_train_df['cleaned_text'] = normalization(cleaned_train_df,'cleaned_text')
cleaned_test_df['cleaned_text'] = normalization(cleaned_test_df,'cleaned_text')


# In[ ]:


trainData = cleaned_train_df.loc[trainIndex]
testData = cleaned_train_df.loc[testIndex]

cv = CountVectorizer(stop_words = 'english',strip_accents='ascii')
vect = cv.fit(cleaned_train_df.cleaned_text)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words = 'english',strip_accents='ascii')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=0)),
])
pipeline.fit(trainData.cleaned_text,trainData.target);
y_pred = pipeline.predict(testData.cleaned_text)
f1_score(testData.target, y_pred, average='micro')


# In[ ]:


cleaned_train_df['lemmatize_text'] = normalization(cleaned_train_df,'lemmatize_text')
cleaned_test_df['lemmatize_text'] = normalization(cleaned_test_df,'lemmatize_text')


# In[ ]:


trainData = cleaned_train_df.loc[trainIndex]
testData = cleaned_train_df.loc[testIndex]

cv = CountVectorizer(stop_words = 'english',strip_accents='ascii')
vect = cv.fit(cleaned_train_df.cleaned_text)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words = 'english',strip_accents='ascii')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=0)),
])
pipeline.fit(trainData.lemmatize_text,trainData.target);
y_pred = pipeline.predict(testData.lemmatize_text)
f1_score(testData.target, y_pred, average='micro')


# In[ ]:


from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


# In[ ]:


get_ipython().run_cell_magic('time', '', "embedding_dict = {}\nnum_lines = sum(1 for line in open('../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt','r'))")


# In[ ]:


with open("../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt","r") as f:
    for line in tqdm(f, total=num_lines):
        values = line.split()
        word = values[0].replace("<", "").replace(">", "")
        vectors = np.asarray(values[1:],'float32')
        embedding_dict[word] = vectors


# In[ ]:


def get_text_features(df,text):
    """https://www.kaggle.com/ykskks/lightgbm-starter"""
    text_feature_df = pd.DataFrame()
    
    # get word vector for each word and average them
    for i, text in enumerate(df[text].values):
        word_vecs = []
        for word in text.split():
            if word in embedding_dict:
                word_vecs.append(embedding_dict[word])
        tweet_vec = np.mean(np.array(word_vecs), axis=0)
        text_feature_df[i] = pd.Series(tweet_vec)
        
    text_feature_df = text_feature_df.T
    text_feature_df.columns = [f"GloVe_{j+1}" for j in range(200)]
    
    return text_feature_df


# In[ ]:


train_text_feat = get_text_features(cleaned_train_df,'lemmatize_text')
test_text_feat = get_text_features(cleaned_test_df,'lemmatize_text')


# In[ ]:


basic_feat = ['word_count','char_count','word_density','punc_count','stop_count','polarity','has_hashtag','has_mention','has_exclamation','has_question','has_url','has_RT']
# combine two types of features
train_feat = pd.concat([cleaned_train_df[basic_feat], train_text_feat], axis=1)
target = cleaned_train_df.target
test_feat = pd.concat([cleaned_test_df[basic_feat], test_text_feat], axis=1)


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_feat))
predictions = np.zeros(len(test_feat))
feature_importance_df = pd.DataFrame()

params = {"learning_rate": 0.01,
          "max_depth": 8,
         "boosting": "gbdt",
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
          "colsample_bytree": 0.5,
         "min_data_in_leaf": 50,
         "bagging_seed": 42,
          "lambda_l2": 0.0001,
         "metric": "binary_logloss",
         "random_state": 42}

for fold, (train_idx, val_idx) in enumerate(skf.split(train_feat, target)):
    print(f"Fold {fold+1}")
    train_data = lgb.Dataset(train_feat.iloc[train_idx], label=target.iloc[train_idx])
    val_data = lgb.Dataset(train_feat.iloc[val_idx], label=target.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(params, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof[val_idx] = clf.predict(train_feat.iloc[val_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train_feat.columns
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).head(50)
    
    predictions += clf.predict(test_feat, num_iteration=clf.best_iteration) / skf.n_splits


# In[ ]:


plt.style.use('fivethirtyeight')
with plt.xkcd():
    fig = plt.figure(figsize=(22, 18))
    ax = fig.add_subplot(111)
    ax = sns.barplot(feature_importance_df["importance"], feature_importance_df.index,color="#1B86BA")
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (LGBM)')
    ax.xaxis.set_major_formatter(ticker)
    plt.grid()


# In[ ]:


thresholds = [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55]
f1s = []

for threshold in thresholds:
    pred_bin = np.where(oof>threshold, 1, 0)
    f1 = f1_score(target, pred_bin)
    f1s.append(f1)

print(f1s)


# In[ ]:


# submit
spsbm = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
spsbm["target"] = np.where(predictions>0.45, 1, 0)
spsbm.to_csv("submission_lgbm.csv",index=False)

