#!/usr/bin/env python
# coding: utf-8

# ## Introduction 
# 
# This is a feature engineering notebook for the DonorsChoose.org Application Screening competition. The objective is to predict whether teachers' project proposals are accepted or rejected. In this notebook, I have described different types of features that can be engineered with the given dataset. These features can be used in the classification models.  
# 
# ### Contents
# 
# 1. Aggregated Features
# 2. Date-Time Features
# 3. Text Based Features
# 4. NLP Based Features
# 5. TF-IDF Features
#     - Word Level TF-IDF
#     - Character Level TF-IDF
# 6. Word Embedding Features
# 7. Topic Modelling Features
# 8. Count Features

# In[ ]:


# !! Important !!
# Set small_run = False, to run this feature engineering notebook for entire dataframe
# Setting small_run = True, runs the notebook only for top 100 rows of the dataframe
# I have added this flag so that this notebook can be executed in kaggle kernal

run_for_small_data = True


# In[ ]:


# import the required libraries 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence, text
from keras.layers import Input, Embedding

from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

import datetime as dt
import pandas as pd
import numpy as np
import warnings
import string

# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation


# In[ ]:


# read data files 

id_column = "id"
missing_token = " UNK "

train = pd.read_csv("../input/donorschoose-application-screening/train.csv", parse_dates=["project_submitted_datetime"])
test = pd.read_csv("../input/donorschoose-application-screening/test.csv", parse_dates=["project_submitted_datetime"])
rc = pd.read_csv("../input/donorschoose-application-screening/resources.csv").fillna(missing_token)

df = pd.concat([train, test], axis=0) 


# ### 1. Aggregated Features
# 
# Features obtained by aggregating the fields from resources data and the training data
# 
# - **Feature 1,2,3 - Min Price, Max Price, Mean Price**: Min, Max, and Mean value of Price of resources requested.
# 
# - **Feature 4,5,6 - Min Quantity, Max Quantity, Mean Quantity**: Min, Max, and Mean value of Quantity of resources requested.
# 
# - **Feature 7,8,9 - Min Total Price, Max Total Price, Mean Total Price**: Min, Max, and Mean value of Total Price of resources requested.
# 
# - **Feature 10,11,12 - Sum of Total Price**: Total price of all the resoruces requested by the teacher in a proposal
# 
# - **Feature 13 - Items Requested**: Total unique number of items requested by the teacher in a proposal
# 
# - **Feature 14 - Quantity**: Total number of quantities requested by the teacher in a proposal

# In[ ]:


rc['total_price'] = rc['quantity']*rc['price']
agg_rc = rc.groupby('id').agg({'description':'count', 'quantity':'sum', 'price':'sum', 'total_price':'sum'}).rename(columns={'description':'items'})

for func in ['min', 'max', 'mean']:
    agg_rc_temp = rc.groupby('id').agg({'quantity':func, 'price':func, 'total_price':func}).rename(columns={'quantity':func+'_quantity', 'price':func+'_price', 'total_price':func+'_total_price'}).fillna(0)
    agg_rc = agg_rc.join(agg_rc_temp)

agg_rc = agg_rc.join(rc.groupby('id').agg({'description':lambda x:' '.join(x.values.astype(str))}).rename(columns={'description':'resource_description'}))

df = df.join(agg_rc, on='id')

if run_for_small_data:
    df = df.head(100)


# In[ ]:


df[['price', 'total_price', 'items', 'quantity', 'min_price', 'min_total_price', 'min_quantity', 
    'max_price', 'max_total_price', 'max_quantity', 'mean_price', 'mean_total_price', 'mean_quantity']].head(10)


# 
# ### 2. Datetime Features 
# 
# Features extracted from project submitted datetime
# 
# - **Feature 15 - Year of Submission**: Value of year when the proposal was submitted
# - **Feature 16 - Month of Submission**: Month number (values between 1 to 12) when the proposal was submitted
# - **Feature 17 - Week Day of Submission**: Week Day value (values between 1 to 7) when the proposal was submitted
# - **Feature 18 - Hour of Submission**: Value of time hour (values between 0 to 23) when the proposal was submitted
# - **Feature 19 - Year Day of Submission**: Year Day (values between 1 to 365) when the proposal was submitted
# - **Feature 20 - Month Day of Submission**: Month Day (values between 1 to 31) when the proposal was submitted
# 
# 

# In[ ]:


# extracting datetime features using datetime module 
df["Year"] = df["project_submitted_datetime"].dt.year
df["Month"] = df["project_submitted_datetime"].dt.month
df['Weekday'] = df['project_submitted_datetime'].dt.weekday
df["Hour"] = df["project_submitted_datetime"].dt.hour
df["Month_Day"] = df['project_submitted_datetime'].dt.day
df["Year_Day"] = df['project_submitted_datetime'].dt.dayofyear


# In[ ]:


df[['Year', 'Month', 'Weekday', 'Hour', 'Month_Day', 'Year_Day']].head(10)


# 
# ### 3. Text based features 
# 
# Features extracted from proposal essay text and resources description
# 
# - **Feature 21: Length of Essay 1** - total number of characters in essay 1 including spaces
# - **Feature 22: Length of Essay 2** - total number of characters in essay 2 including spaces
# - **Feature 23: Length of Essay 3** - total number of characters in essay 3 including spaces
# - **Feature 24: Length of Essay 4** - total number of characters in essay 4 including spaces
# - **Feature 25: Length of Project Title** - total number of characters in project title including spaces
# - **Feature 26: Word Count in the Complete Essay** - total number of words in the complete essay text
# - **Feature 27: Character Count in the Complete Essay** - total number of characters in complete essay text
# - **Feature 28: Word Density of the Complete Essay** - average length of the words used in the essay
# - **Feature 29: Puncutation Count in the Complete Essay** - total number of punctuation marks in the essay
# - **Feature 30: Upper Case Count in the Complete Essay** - total number of upper count words in the essay
# - **Feature 31: Title Word Count in the Complete Essay** - total number of proper case (title) words in the essay
# - **Feature 32: Stopword Count in the Complete Essay** - total number of stopwords in the essay
# 

# In[ ]:


# fillup empty values with missing token 
df['project_essay_3'] = df['project_essay_3'].fillna(missing_token)
df['project_essay_4'] = df['project_essay_4'].fillna(missing_token)

# extract length of each essay and title
df["essay1_len"] = df['project_essay_1'].apply(len)
df["essay2_len"] = df['project_essay_2'].apply(len)
df["essay3_len"] = df['project_essay_3'].apply(len)
df["essay4_len"] = df['project_essay_4'].apply(len)
df["title_len"] = df['project_title'].apply(len)


# In[ ]:


df[['essay1_len', 'essay2_len', 'essay3_len', 'essay4_len', 'title_len']].head(10)


# In[ ]:


# combine the project essays to create a complete essay text
df['text'] = df.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                            str(row['project_essay_2']), 
                                            str(row['project_essay_3']), 
                                            str(row['project_essay_4'])]), axis=1)

# extract features from text
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
df['stopword_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))


# In[ ]:


df[['char_count', 'word_count', 'word_density', 'punctuation_count', 'title_word_count', 'upper_case_word_count', 'stopword_count']].head(10)


# 
# ### 4. More NLP based features 
# 
# Part of Speech and Sentiment related features from the text. I have used python's textblob package to get the sentiment related features and part-of-speech tags of the tokens in the sentence. 
# 
# - **Feature 33: Article Polarity** - total number of characters in essay 1 including spaces
# - **Feature 34: Article Subjectivity** - total number of characters in essay 2 including spaces
# - **Feature 35: Noun Count** - total number of characters in essay 3 including spaces
# - **Feature 36: Verb Count** - total number of characters in essay 4 including spaces
# - **Feature 37: Adjective Count** - total number of characters in project title including spaces
# - **Feature 38: Adverb Count** - total number of words in the complete essay text
# - **Feature 39: Pronoun Count** - total number of characters in complete essay text
# 

# In[ ]:


# functions to get polatiy and subjectivity of text using the module textblob
def get_polarity(text):
    try:
        textblob = TextBlob(unicode(text, 'utf-8'))
        pol = textblob.sentiment.polarity
    except:
        pol = 0.0
    return pol

def get_subjectivity(text):
    try:
        textblob = TextBlob(unicode(text, 'utf-8'))
        subj = textblob.sentiment.subjectivity
    except:
        subj = 0.0
    return subj


# change df_small to df to create these features on complete dataframe
df['polarity'] = df['text'].apply(get_polarity)
df['subjectivity'] = df['text'].apply(get_subjectivity)


# In[ ]:


df[['polarity', 'subjectivity']].head(10)


# In[ ]:


pos_dic = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def pos_check(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count'] = df['text'].apply(lambda x: pos_check(x, 'noun'))
df['verb_count'] = df['text'].apply(lambda x: pos_check(x, 'verb'))
df['adj_count'] = df['text'].apply(lambda x: pos_check(x, 'adj'))
df['adv_count'] = df['text'].apply(lambda x: pos_check(x, 'adv'))
df['pron_count'] = df['text'].apply(lambda x: pos_check(x, 'pron'))


# In[ ]:


df[['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']].head(10)


# 
# ### 5. TF-IDF Features
# 
# Tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# - TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
# - IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
# 
# Reference: http://www.tfidf.com/
# 
# 
# 
# - **Feature 40:** Word Level N-Gram TF-IDF of Article Text
# - **Feature 41:** Word Level N-Gram TF-IDF of Project Title
# - **Feature 42:** Word Level N-Gram TF-IDF of Resource Text
# - **Feature 43:** Character Level N-Gram TF-IDF of Article Text
# - **Feature 44:** Character Level N-Gram TF-IDF of Project Title
# - **Feature 45:** Character Level N-Gram TF-IDF of Resource Text
# 

# In[ ]:


df['article_text'] = df.apply(lambda row: ' '.join([str(row['project_essay_1']), str(row['project_essay_2']), 
                                         str(row['project_essay_3']), str(row['project_essay_4'])]), axis=1)
df['resource_text'] = df.apply(lambda row: ' '.join([str(row['resource_description']), str(row['project_resource_summary'])]), axis=1)

resource_text = list(df['resource_text'].values)
title_text = list(df['project_title'].values)
article_text = list(df['article_text'].values)

# word level tf-idf for article text
vect_word = TfidfVectorizer(max_features=2500, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
vect_word.fit(article_text)
tfidf_complete = vect_word.transform(article_text)

# word level tf-idf for project title
vect_word = TfidfVectorizer(max_features=500, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
vect_word.fit(title_text)
tfidf_title = vect_word.transform(title_text)

# word level tf-idf for resource text
vect_word = TfidfVectorizer(max_features=1000, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
vect_word.fit(resource_text)
tfidf_resource = vect_word.transform(resource_text)


# In[ ]:


#  create a dictionary mapping the tokens to their tfidf values
tfidf = dict(zip(vect_word.get_feature_names(), vect_word.idf_))
tfidf = pd.DataFrame(columns=['title_word_tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['title_word_tfidf']


# In[ ]:


# features with highest tf-idf (in title)
tfidf.sort_values(by=['title_word_tfidf'], ascending=False).head(10)


# Similarly we can generate character level tf-idfs

# In[ ]:


# character level tf-idf for article text
char_word = TfidfVectorizer(max_features=2000, analyzer='char', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
char_word.fit(article_text)
tfidf_complete_char = char_word.transform(article_text)

# character level tf-idf for project title
char_word = TfidfVectorizer(max_features=500, analyzer='char', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
char_word.fit(title_text)
tfidf_title_char = char_word.transform(title_text)

# character level tf-idf for resource text
char_word = TfidfVectorizer(max_features=600, analyzer='char', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
char_word.fit(resource_text)
tfidf_resource_char = char_word.transform(resource_text)


# 
# ### 6. Word Embeddings
# 
# **Feature 46:** WordEmbedding Vectors of text data
# 
# Word Embedding Vectors can be trained itself using the corpus or they can be generated using Pre-Trained word embeddings. 
# 
# A word embedding is a class of approaches for representing words and documents using a dense vector representation. It is an improvement over more the traditional bag-of-word model encoding schemes where large sparse vectors were used to represent each word or to score each word within a vector to represent an entire vocabulary. These representations were sparse because the vocabularies were vast and a given word or document would be represented by a large vector comprised mostly of zero values. Instead, in an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space. The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used.
# 
# Reference: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# 

# In[ ]:


xtrain = df.text.values

# load the pre-trained word-vectors
embeddings_index = {}

EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'
f = open(EMBEDDING_FILE)
for line in f:
    if run_for_small_data and len(embeddings_index) == 100:
      break
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# perform pre-processing in keras
max_features = 100000 # max number of words to use in word embedding matrix
max_len = 300 # max length of the word embedding vector

# Tokenization of text data
token = text.Tokenizer(num_words=max_features)
token.fit_on_texts(list(xtrain))
word_index = token.word_index

# Create sequence of Tokens and Pad them to create equal length vectors
xtrain_seq = token.texts_to_sequences(xtrain)
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

# Create an embedding matrix of words in the data
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# #### Using Word Embedding Features
# 
# **Option 1:** Create Sentence Vectors
# 
# There are differet methods to get the sentence vectors :
# 
# - Doc2Vec : Train your dataset using Doc2Vec and then use the sentence vectors.
# - Average of Word2Vec vectors : Take the average of all the word vectors in a sentence. This average vector will represent the sentence vector. In this notebook I have used this approach. 
# - Average of Word2Vec vectors with TF-IDF : Take the word vectors, multiply it with their TF-IDF scores and take the average to get sentence vector.
# 
# 
# **Option 2:** Use Word Embeddings Directly
# 
# Keras offers an Embedding layer that can be used for neural networks on text data. It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with Keras. The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.

# In[ ]:


# Option One: Create Sentence to vector
    
# function to generate sentence vector of the sentence
def sent2vec(sentence):
    M = []
    for w in word_tokenize(sentence):
#     for w in word_tokenize(unicode(sentence, 'utf8')):
        if not w.isalpha():
            continue
        if w in embeddings_index:
            M.append(embeddings_index[w])
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

xtrain_vector = [sent2vec(x) for x in xtrain[:10]]
xtrain_vector = np.array(xtrain_vector)

# Option Two: Use the word embeddings directly in deep neural network

input_layer = Input((max_len, ))
embedding_layer = Embedding(len(word_index)+1, max_len, weights=[embedding_matrix], trainable=False)(input_layer)


# Lets view the word embedding vector

# In[ ]:


# these word vectors can be directly used in the model
xtrain_vector


# ### 7. Topic Modelling Features
# 
# **Feature 47:** Topic Modelling Features 
# 
# I have used LDA for generating Topic Modelling Features. Latent Dirichlet Allocation (LDA) is an algorithm used to discover the hidden topics that are present in a corpus. LDA starts from a fixed number of topics. Each topic is represented as a distribution over words, and each document is then represented as a distribution over topics. Although the tokens themselves are meaningless, the probability distributions over words provided by the topics provide a sense of the different ideas contained in the documents.
# 

# In[ ]:


# create count vectorizer first
cvectorizer = CountVectorizer(min_df=4, max_features=4000, ngram_range=(1,2))
cvz = cvectorizer.fit_transform(df['text'])

# generate topic models using Latent Dirichlet Allocation
lda_model = LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20, random_state=42)
X_topics = lda_model.fit_transform(cvz)


# In[ ]:


n_top_words = 10
topic_summaries = []

# get topics and topic terms
topic_word = lda_model.components_ 
vocab = cvectorizer.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))


# Lets view some of the topics obtained

# In[ ]:


X_topics[:5]


# **Please note that the quality of Word Embeddings and Topic Modelling Features 
# will be poor in case of small data run, but they are improved if the models are run on complete data frame**

# 
# ### 8. Count Features
# 
# **Feature 48 - 60:** Count Features 
# 
# There are some categorical features in the dataset, which can be represented as the count features. 

# In[ ]:


features_for_count = ['school_state', 'teacher_id', 'teacher_prefix', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']
features_for_count += ['Year', 'Year_Day', 'Weekday', 'Month_Day', 'Month', 'Hour']
for col in features_for_count:
    aggDF = df.groupby(col).agg('count')
    aggDF[col] = aggDF.index
    tempDF = pd.DataFrame(aggDF[['project_submitted_datetime', col]], columns = ['project_submitted_datetime', col])
    tempDF = tempDF.rename(columns={'project_submitted_datetime': col+"_count"})
    df = df.merge(tempDF, on=col, how='left')


# In[ ]:


df[[x+"_count" for x in features_for_count]].head(10)


# Thanks for viewing the notebook. 
