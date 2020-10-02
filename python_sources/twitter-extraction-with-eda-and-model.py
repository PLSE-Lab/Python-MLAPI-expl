#!/usr/bin/env python
# coding: utf-8

# ## Importing Necesary Library

# In[ ]:


import spacy

#spacy.prefer_gpu()


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


from sklearn.feature_extraction.text import CountVectorizer
import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/ import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
import string

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=10,6
plt.rcParams['axes.grid']=True
plt.gray()

use_cuda = True


# In[ ]:


# Reading the data
BASE_PATH = '/kaggle/input/tweet-sentiment-extraction/'

train_df = pd.read_csv(BASE_PATH + 'train.csv')
test_df = pd.read_csv( BASE_PATH + 'test.csv')
submission_df = pd.read_csv( BASE_PATH + 'sample_submission.csv')


# In[ ]:


# Checking the shape of train and test data
print(train_df.shape)
print(test_df.shape)


# In[ ]:


# Checking Missing value in the training set
print(train_df.isnull().sum())
# Checking Missing Value in the testing set
print(test_df.isnull().sum())


# In[ ]:


# Droping the row with missing values
train_df.dropna(axis = 0, how ='any',inplace=True)


# In[ ]:


# Positive tweet
print("Positive Tweet example :",train_df[train_df['sentiment']=='positive']['text'].values[0])
#negative_text
print("Negative Tweet example :",train_df[train_df['sentiment']=='negative']['text'].values[0])
#neutral_text
print("Neutral tweet example  :",train_df[train_df['sentiment']=='neutral']['text'].values[0])


# In[ ]:


# Distribution of the Sentiment Column
train_df['sentiment'].value_counts()


# In[ ]:


sns.countplot(x=train_df['sentiment'],data=train_df)
plt.show()


# In[ ]:


train_df['sentiment'].value_counts(normalize=True)


# In[ ]:


train_df['sentiment'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Sentiments')
plt.ylabel('Percentage')
plt.show()


# In[ ]:


# text preprocessing helper functions

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text


# In[ ]:


# Applying the cleaning function to both test and training datasets
train_df['text_clean'] = train_df['text'].apply(str).apply(lambda x: text_preprocessing(x))
test_df['text_clean'] = test_df['text'].apply(str).apply(lambda x: text_preprocessing(x))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# Analyzing Text statistics

train_df['text_len'] = train_df['text_clean'].astype(str).apply(len)
train_df['text_word_count'] = train_df['text_clean'].apply(lambda x: len(str(x).split()))


# In[ ]:


train_df.head()


# In[ ]:


# Let's create three separate dataframes for positive, neutral and negative sentiments. 
#This will help in analyzing the text statistics separately for separate polarities.

pos = train_df[train_df['sentiment']=='positive']
neg = train_df[train_df['sentiment']=='negative']
neutral = train_df[train_df['sentiment']=='neutral']


# In[ ]:


pos.head()


# In[ ]:


# Sentence length analysis

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(pos['text_len'],bins=50,color='g')
plt.title('Positive Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')


plt.subplot(1, 3, 2)
plt.hist(neg['text_len'],bins=50,color='r')
plt.title('Negative Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')


plt.subplot(1, 3, 3)
plt.hist(neutral['text_len'],bins=50,color='y')
plt.title('Neutral Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')
plt.show()


# ##### The histogram shows that the length of the cleaned text ranges from around 2 to 140 characters and generally,it is almost same for all the polarities

# In[ ]:


#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


#Distribution of top unigrams
pos_unigrams = get_top_n_words(pos['text_clean'],20)
neg_unigrams = get_top_n_words(neg['text_clean'],20)
neutral_unigrams = get_top_n_words(neutral['text_clean'],20)

df1 = pd.DataFrame(pos_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 unigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 unigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 unigram in Neutral text')
plt.show()


# In[ ]:


def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


#Distribution of top Bigrams
pos_bigrams = get_top_n_gram(pos['text_clean'],(2,2),20)
neg_bigrams = get_top_n_gram(neg['text_clean'],(2,2),20)
neutral_bigrams = get_top_n_gram(neutral['text_clean'],(2,2),20)

df1 = pd.DataFrame(pos_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 Bigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 Bigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 Bigram in Neutral text')
plt.show()


# In[ ]:


# Finding top trigram
pos_trigrams = get_top_n_gram(pos['text_clean'],(3,3),20)
neg_trigrams = get_top_n_gram(neg['text_clean'],(3,3),20)
neutral_trigrams = get_top_n_gram(neutral['text_clean'],(3,3),20)

df1 = pd.DataFrame(pos_trigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 trigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_trigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 trigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_trigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 trigram in Neutral text')
plt.show()


# In[ ]:


#  Exploring the selected_text column

positive_text = train_df[train_df['sentiment'] == 'positive']['selected_text']
negative_text = train_df[train_df['sentiment'] == 'negative']['selected_text']
neutral_text = train_df[train_df['sentiment'] == 'neutral']['selected_text']


# In[ ]:


negative_text.head()


# In[ ]:



# Positive text
print("Positive Text example :",positive_text.values[0])
#negative_text
print("Negative Tweet example :",negative_text.values[0])
#neutral_text
print("Neutral tweet example  :",neutral_text.values[0])


# In[ ]:


# Preprocess Selected_text

positive_text_clean = positive_text.apply(lambda x: text_preprocessing(x))
negative_text_clean = negative_text.apply(lambda x: text_preprocessing(x))
neutral_text_clean = neutral_text.apply(lambda x: text_preprocessing(x))


# In[ ]:


negative_text_clean.head()


# In[ ]:


#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


top_words_in_positive_text = get_top_n_words(positive_text_clean)
top_words_in_negative_text = get_top_n_words(negative_text_clean)
top_words_in_neutral_text = get_top_n_words(neutral_text_clean)

p1 = [x[0] for x in top_words_in_positive_text[:20]]
p2 = [x[1] for x in top_words_in_positive_text[:20]]


n1 = [x[0] for x in top_words_in_negative_text[:20]]
n2 = [x[1] for x in top_words_in_negative_text[:20]]


nu1 = [x[0] for x in top_words_in_neutral_text[:20]]
nu2 = [x[1] for x in top_words_in_neutral_text[:20]]


# In[ ]:


# Top positive word
sns.barplot(x=p1,y=p2,color = 'green')
plt.xticks(rotation=45)
plt.title('Top 20 Positive Word')
plt.show()

sns.barplot(x=n1,y=n2,color='red')
plt.xticks(rotation=45)
plt.title('Top 20 Negative Word')
plt.show()

sns.barplot(x=nu1,y=nu2,color='yellow')
plt.xticks(rotation=45)
plt.title('Top 20 Neutral Word')
plt.show()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


#Wordclouds
# Wordclouds to see which words contribute to which type of polarity.

from wordcloud import WordCloud
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(positive_text_clean))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive text',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(negative_text_clean))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative text',fontsize=40);

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(neutral_text_clean))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutral text',fontsize=40)


# In[ ]:


# https://www.kaggle.com/ekhtiar/unintended-eda-with-tutorial-notes
def generate_word_cloud(df_data, text_col):
    # convert stop words to sets as required by the wordcloud library
    stop_words = set(stopwords.words("english"))
    
    data_neutral = " ".join(df_data.loc[df_data["sentiment"]=="neutral", text_col].map(lambda x: str(x).lower()))
    data_positive = " ".join(df_data.loc[df_data["sentiment"]=="positive", text_col].map(lambda x: str(x).lower()))
    data_negative = " ".join(df_data.loc[df_data["sentiment"]=="negative", text_col].map(lambda x: str(x).lower()))

    wc_neutral = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_neutral)
    wc_positive = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_positive)
    wc_negative = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_negative)

    # draw the two wordclouds side by side using subplot
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].set_title("Neutral Wordcloud" , fontsize=10)
    ax[0].imshow(wc_neutral, interpolation="bilinear")
    ax[0].axis("off")
    
    ax[1].set_title("Positive Wordcloud", fontsize=10)
    ax[1].imshow(wc_positive, interpolation="bilinear")
    ax[1].axis("off")
    
    ax[2].set_title("Negative Wordcloud", fontsize=10)
    ax[2].imshow(wc_negative, interpolation="bilinear")
    ax[2].axis("off")
    plt.show()
    
    return [wc_neutral, wc_positive, wc_negative]


# In[ ]:


train_text_wc = generate_word_cloud(train_df, "text")


# In[ ]:


train_sel_text_wc = generate_word_cloud(train_df, "selected_text")


# In[ ]:


train_df.head()


# In[ ]:


train_df['Num_words_text'] = train_df['text'].apply(lambda x: len(str(x).split()))


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df.loc[train_df['Num_words_text']>=3]


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


def save_model(output_dir, nlp, new_model_name):
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


# In[ ]:


def train(train_data, output_dir, n_iter=30, model=None):
    """Load the model,set up the pipeline and train the entity recognizer"""
    if model is not None:
        nlp=spacy.load(model) #load existing spaCy model
        print("Loaded model '%s'" %model)
    else:
        nlp = spacy.blank("en") #create blank Language class
        print("Created blank 'en' model ")
        
        # The pipeline execution
        # Create the built-in pipeline components and them to the pipeline
        # nlp.create_pipe works for built-ins that are registered in the spacy
        
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner,last=True)
            
        # otherwise, get it so we can add labels
        
        else:
            ner = nlp.get_pipe("ner")
            
        # add labels 
        for _, annotations in train_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
        
        # get names of other pipes to disable them during training
        
        pipe_exceptions = ["ner","trf_wordpiecer","trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        #other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        
        with nlp.disable_pipes(*other_pipes): # training of only NER
            
            # reset and intialize the weights randoml - but only if we're
            # training a model
            
            if model is None:
                nlp.begin_training()
            else:
                nlp.resume_training()
            
            for itn in tqdm(range(n_iter)):
                random.shuffle(train_data)
                losses={}
                
                # batch up the example using spaCy's mnibatch
                batches = minibatch(train_data,size=compounding(4.0,1000.0,1.001))
                
                for batch in batches:
                    texts , annotations = zip(*batch)
                    nlp.update(
                        texts, #batch of texts
                        annotations, # batch of annotations
                        drop = 0.5,  # dropout - make it harder to memorise data
                        losses = losses,
                )
            print("Losses", losses)
        save_model(output_dir, nlp, 'st_ner')
                    


# In[ ]:


def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    else:
        model_out_path = 'models/model_neu'
    return model_out_path


# In[ ]:



def get_training_data(sentiment):
    train_data=[]
    '''
    Returns Training data in the format needed to train spacy NER
    '''
    for index,row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start,end,'selected_text']]}))
    return train_data


# In[ ]:


# Training Positive sentiments
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

# Training model iteration 
train(train_data, model_path, n_iter=10, model=None)


# In[ ]:


# Training Negative Sentiment

sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

# Training model iteration 
train(train_data, model_path, n_iter=10, model=None)


# In[ ]:


# Training Neutral Sentiment

sentiment = 'neutral'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

# Training model iteration 
train(train_data, model_path, n_iter=10, model=None)


# In[ ]:


def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text


# In[ ]:


TRAINED_MODELS_BASE_PATH = '../input/tse-spacy-model/models/'


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
        
    jaccard_score = 0
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        text = row.text
        if row.sentiment == 'neutral':
            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)
        elif row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
        else:
            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text) 
        
    print(f'Average Jaccard Score is {jaccard_score / train_df.shape[0]}') 


# In[ ]:


def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text


# In[ ]:


MODELS_BASE_PATH = '../input/tse-spacy-model/models/'


# In[ ]:


selected_texts = []

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(MODELS_BASE_PATH + 'model_neu')
        
    for index, row in test_df.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) < 2:
#             output_str = text
#             selected_texts.append(predict_entities(text, model_neu))
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
test_df['selected_text'] = selected_texts


# In[ ]:


test_df.head(10)


# In[ ]:


submission_df['selected_text'] = test_df['selected_text']
submission_df.to_csv("submission.csv", index=False)
display(submission_df.head(10))

