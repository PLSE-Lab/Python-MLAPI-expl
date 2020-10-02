#!/usr/bin/env python
# coding: utf-8

# # Team EN3 Classification Predict

# <a id='Table_Contents'></a><br>
# ### Table of Contents
# 
# 1. [Introduction](#intro)
#  * Abstract
# 
# 2. [Data](#imports_data)
#  * Importing libraries
#  * Comet
#  * Loading Data
#  
# 3. [Exploratory Data Analysis of Raw tweets](#EDA_raw)
#  * Text Statistics
#  * Stop Words
#  * Ngrams Analysis
#  * Topic Modelling
#  * NER Analysis
# 
# 4. [Preprocessing](#cleaning)
#  * Target and input vairable
#  * Data Cleaning
# 
# 5. [Exploratory Data Analysis after cleaning tweets](#EDA_clean)
#  * Token
#  * Stemming
#  * Stopwords
#  * Transformation
#  * Visualing WordClouds
#  * Visualising Bar Plots
#  * Visualing Hashtags
# 
# 6. [Vectorization](#vector)
#  * CountVector
#  * TF-IDF
#  * Word2Vec
# 
# 7. [Modelling](#modeling)
#  * Trial and error
#  * Split train 
#  * Naive Bayes Classifier
#  * Linear Support Vector Classifier
#  * Passive Aggressive Classifier
#  * Logistic Regression Classifier
#  * K Nearest Neighbours Clasiffier
#  * Gradient Boosting Classifier
#  * Performance Metrics of Best Model
#  * Vec2Word on best models
#  
# 
# 8. [Hypertuning](#tuning)
#  * RandomSearchCV 
#  * GridSearchCV 
#  * Testing
# 
# 9. [Evaluation](#eval)
# 
# 10. [Conclusion](#con)
# 
# 11. [Submission](#pkl)

# <a id='intro'></a><br>
# ## 1. Introduction
# [Back to Table of Contents](#Table_Contents)

# ### Abstract

# <p>Twitter is a platform widely used by people to express their opinions and show their sentiments based on a certain topic or situation. Sentiment analysis is a way to analyse the data in depth and get an understanding of how the tweet comes accross as positive, neutral or negative. One major topic in this world today is climate change and whether it exists. Companies are starting to offer products and services that are environmentally friendly and sustainable. It is very important for these companies to understand how people perceive climate change and whether a person believes in climate change based on what they tweet. The tweet format is very small, which generates many problems when analysing due to the slang, abbreviations, misspelled words etc. This notebook will do exploritory data analysis and preproccssing of data, in order to transform the data into a tidy tweet format and classify the user's sentiment by analysing the tweets into negative(-1), neutral(0), positive(1), news(2). This will be accomplished by building supervised learning models using python and natural language processing libraries</p>
# <p>The aim of this notebook is to detect sentiment in tweets to determine if people believe in climate change or not. A postive sentiment meaning they know climate change exists and a negative sentiment to express that climate change is not real.</p>

# <a id='imports_data'></a><br>
# ## 2.0 Data
# [Back to Table of Contents](#Table_Contents)

# ### Importing libraries

# In[ ]:


# import comet_ml in the top of your file
# from comet_ml import Experiment

#Inspecting
import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
from collections import Counter
import pyLDAvis.gensim
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from spacy import displacy
import gensim
import spacy
nlp = spacy.load("en_core_web_lg")

#visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from wordcloud import WordCloud
from PIL import Image
import collections
from matplotlib import style
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [16, 12]})
plt.style.use('seaborn-whitegrid')
sns.set_style('dark')

#Warnings
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Balance data
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

#Cleaning
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist

#Modeling
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, RandomizedSearchCV


#metrics for analysis
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, f1_score

# Import Pickle for streamlit Application
import pickle


# ### Comet

# In[ ]:


# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="ZO3kD6D1uVgIr9CHdhUPQaU3B",
#                         project_name="climate-change-belief-analysis", workspace="helloaggregator")


# ### Loading  data

# In[ ]:


train = pd.read_csv('../input/dataset/train.csv')
test = pd.read_csv('../input/dataset/test.csv')


# <a id='EDA_raw'></a><br>
# ## 3.0 Exploratory Data Analysis of Raw Tweets
# [Back to Table of Contents](#Table_Contents)

# ### Text Statistics

# #### Number of characters present in each message/tweet.

# In[ ]:


plt.title('Number of Characters Present in Tweet')
train['message'].str.len().hist()


# * For years, Twitter has had a 140 character limit. 
# * However, in 2017, the network increased the limit to 280 characters.
# * Our dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018 
#   - this probably explains why the majority of tweets are below 140 characters.
# * Sprout Social says the ideal Length of a Tweet is 71-100 characters.
# * According to Buddy Media, Tweets with 100 characters get 17% higher engagement rates than longer Tweets.
# 
# Key maximum word count and character limits on Twitter:
#     * Maximum Tweet length: 280 characters
#     * DMs: 10,000 characters
#     * Handle maximum length: 15 characters
#     * Twitter profile name maximum length: 20

# #### Analysing word length

# In[ ]:


train['message'].str.split().apply(lambda x : [len(i) for i in x]). map(lambda x: np.mean(x)).hist()


# The average amount of characters in a word is around 5

# ### Stopwords

# In[ ]:


# Fetch stopwords so it doesn't take away from Ngram analysis
stop = set(stopwords.words('english'))


# ### Ngrams Analysis

# #### Most common words

# In[ ]:


# Create corpus
corpus=[]
new= train['message'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1


# In[ ]:


counter=Counter(corpus)
most=counter.most_common()

x, y = [], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
plt.title('Most Common Words')
plt.show()


# It is no surprise that 'climate' is the top word as that is the topic. The word appears over 12,000 times.
# * RT is the second word, appearing just shy of 10,000. A safe assumption here is that a large number of the 
# tweets are not original, but just re-tweets.
# * Appearing under 9,000 times 'change' has the highest association with climate related tweets. This is       followed by global and then warming.
# * At just under 2,000 appearances, Trump is the highest mentioned person. This number will of course change as we come to realise that his mentions can be by title, first name or by surname.
# * Let's move to Ngram analysis for more insights.

# #### Bigram Analysis

# In[ ]:


def get_top_ngram(corpus, n=None):
    
    '''
    Takes a list of words and groups then in terms of ngrams depending on how many words you want to group, returns 
    a word count based on the number of times ngram appears
    
    Parameters
    -----------
    corpus: list
            input list of strings
    n: int
       input the number of ngrams needed
       
    Output
    ----------
    Output: Returns a tuple list with specified number of words grouped and counts the frequency
    
    '''
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

top_n_bigrams = get_top_ngram(train['message'],2)[:40]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x = y,y = x)
plt.title('Bigram analysis')
plt.show()


# 'Climate change' is the highest mentioned bigram. This is followed by a url, leading to an assumption that tweets also carry links to other content. 'Global warming' is another popular bigram.

# #### Trigram analysis

# In[ ]:


top_tri_grams=get_top_ngram(train['message'],n=3)
x,y=map(list,zip(*top_tri_grams))
sns.barplot(x=y,y=x)
plt.title('Trigram Analysis')
plt.show()


# Once again, climate change dominates and the sharing of urls is high. The aspect noticed is people using the word 'believe' in climate change or not.

# ### Topic Modelling

# In[ ]:


def preprocess_train(df):
    '''
    Creates a list of lemmetized words that must have a length greater than 2 from an input of a dataframe
    
    Parameters
    -----------
    df: Dataframe
        Input needs to be dataframe
        
    Output
    -----------
    corpus: Returns a list of lemmatized words
    
    '''
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for train in df['message']:
        words = [w for w in word_tokenize(train) if (w not in stop)]
        
        words = [lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus


# In[ ]:


#Create corpus
corpus = preprocess_train(train)

# Create tuple vectorised words
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]

# creat a weight for topics of vectorized words
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)


# In[ ]:


#visual the the top ten topics
style.use('dark_background')
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
vis


# Topic 1 shows the words that appear the most, adding on from the words mentioned above are 'EPA', 'fight', 'paris', 'china', 'scientist' and 'SenSanders' seems to be the hot topics. Topic 3 shows the likes of 'amp', 'LeoDicaprio' and 'BeforeTheFlood' which shows postive words of climate change as Leonardo has strong views of helping climate change from his speech at the oscar awards. 'BeforeTheFlood', presented by National Geographic, features Leonardo DiCaprio on a journey as a United Nations Messenger of Peace.

# ### NER Analysis

# In[ ]:


def ner(text):
    '''
    Takes in  text and returns entity label of text using the natural language processor on python
    
    Parameters
    ----------
    text: String
          input a string
    ent: String
         Input a string ant it will return the entity you desire
    
    Output
    ---------
    output: Entity labelled string
            Returns a label depending on the context of string
    
    '''
    doc=nlp(text)
    return [X.label_ for X in doc.ents]


# In[ ]:


#create labels for all the tweets
ent=train['message'].apply(lambda x : ner(x))
ent=[x for sub in ent for x in sub]
counter=Counter(ent)
count=counter.most_common()

#Plot the labels that occurred the most from the tweets
x,y=map(list,zip(*count))
sns.barplot(x=y,y=x)
plt.title('The most Occurring Entities')
plt.show()


# It shows that the most entities tweeted are organisations, people then countries. This makes sense because these entities play a huge role climate change espcially organisations.

# #### Most common GPE

# In[ ]:


def ner(text,ent="GPE"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]


# In[ ]:


gpe = train['message'].apply(lambda x: ner(x,"GPE"))
gpe = [i for x in gpe for i in x]
counter = Counter(gpe)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('Most Common GPE')
plt.show()


# #### Most common person

# In[ ]:


per = train['message'].apply(lambda x: ner(x,"PERSON"))
per = [i for x in per for i in x]
counter = Counter(per)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('The most Common Person')
plt.show()


# #### Most Common Organisation

# In[ ]:


gpe = train['message'].apply(lambda x: ner(x,"ORG"))
gpe = [i for x in gpe for i in x]
counter = Counter(gpe)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('The most Common Organisations')
plt.show()


# From the entity analysis it shows that a lot of data needs to be cleaned as some of the organisations shown are links or latin symbols and not all the entity labels have been labelled correctly. However, the entity analysis has given good insight as the countries USA, China, UK and the city Paris seem to show the most in the tweets. The names that appear the most is first and for most Donald Trump, followed by Scott Pruit, Al Gore, Obama and Hilary Clinton which all have to do with mainly political figures in America. The top organisations found from the list are EPA which is the environmental Protection agency, Exxon which is an oil company and the UN known as the United Nations which has framework convention on climate change. NASA which stands for National Aeronautics and Space Administration, qhich have been collecting data with their Earth-orbiting satellites and other technological advances, have enabled scientists to see the big picture, collecting many different types of information about our planet and its climate on a global scale. Since tweets can use slang and mispelled words, this analysis can give you some good insight but not the whole picture, so further analysis will be done once the text is cleaned.

# <a id='cleaning'></a><br>
# ## 4.0 Preprocessing
# [Back to Table of Contents](#Table_Contents)

# ### Target and input variable

# In[ ]:


test.head()


# In[ ]:


train.head()


# #### Target Variable

# Lets plot the target variable to understand if the sentiments are balanced or imbalanced data.

# In[ ]:


sns.factorplot('sentiment',data = train, kind='count',size=6,aspect = 1.5, palette = 'PuBuGn_d') 
plt.suptitle("Climate Sentiment Bar Graph",y=1)
plt.show()


# To understand the imbalance of the data better, a pie plot will be shown with percentages of each sentiment

# In[ ]:


# total number of negative ,neutral, positive and news posts.
climate_sentiment  = train['sentiment'].value_counts()

# pie plot for total percentage of climate change sentiment
plt.figure(figsize=(5,5))
labels = 'Positive','News','Neutral','Negative'
sizes = climate_sentiment.tolist()
colors = ['green', 'purple', 'blue','red']
explode = (0, 0, 0,0) 

# Plot
plt.suptitle("Climate Sentiment Pie Chart",y=1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# * The positive sentiment counts are significantly higher followed by news, then neutral and lastly anti. 
# * The categories for the labelled data that is going to predict unseen data is unbalanced and this will cause the model to predict sentiment = 1 very well as it dominates the four sentiment categories, however, this will be a problem in predicting the other sentiment (-1,0,2) accurately.

# #### Input Variables

# In[ ]:


#Analyse the text tweets for cleaning
for index,text in enumerate(train['message'][35:]):
  print('Tweet %d:\n'%(index+1),text)


# From looking the text above, we can create Textcounts from notcing a few trends, this will help compute some basic statistics on the text variables:
# 
# <p><b>count_words:</b> Number of words in a tweet</p>
# <p><b>count_mentions:</b> referrals to other twitter accounts, starts with @</p>
# <p><b>count_hasgtags:</b> numner of tag words,starts with #</p>
# <p><b>count_capital_words:</b> Number of uppercase words are sometimes used as a way to express feelings</p>
# <p><b>count_number_exl_quest:</b> count number of question marks and exclamations</p>
# <p><b>count_urls:</b> number of links in tweet, starts with https</p>
# <p><b>count_emojis:</b> number of emojis, might be a good sign of the sentiment</p>

# In[ ]:


# number of words in a tweet
df_eda = pd.DataFrame()
df_eda['count_words'] = train['message'].apply(lambda x: len(re.findall(r'\w+',x)))

# referrals to other twiiter accounts
df_eda['count_mentions'] = train['message'].apply(lambda x: len(re.findall(r'@\w+',x)))

# number of hashtags 
df_eda['count_hashtags'] = train['message'].apply(lambda x: len(re.findall(r'#\w+',x)))

# Number of upper case words 3 or more to ignore RT
df_eda['count_capital_words'] = train['message'].apply(lambda x: len(re.findall(r'\b[A-Z]{3,}\b',x)))

#count number of exclamation marks and questions marks 
df_eda['count_exl_quest'] = train['message'].apply(lambda x: len(re.findall(r'!|\?',x)))

#count number of urls
df_eda['count_urls'] = train['message'].apply(lambda x: len(re.findall(r'http.?://[^\s]+[\s]?',x)))

#count the number of emojis
df_eda['count_emojis'] = train['message'].apply(lambda x: emoji.demojize(x)).apply(lambda x: len(re.findall(r':[a-z_&]+:',x)))

#add the dependent varaible for further analysis
df_eda['sentiment'] = train.sentiment
df_eda.head()


# Lets see how the text count analysis relates to the sentiment categoroies by plotting graphs

# In[ ]:


# Comment
column_names = [col for col in df_eda.columns if col != 'sentiment']
for i in column_names:
    bins = np.arange(df_eda[i].min(),df_eda[i].max()+1)
    g = sns.FacetGrid(data=df_eda,col='sentiment',size=5, hue = 'sentiment',palette="PuBuGn_d")
    g = g.map(sns.distplot, i, kde= False, norm_hist = True,bins = bins)
    plt.show()


# <b> Number of words per tweet </b>
# <ul>
# <li> The number of words used in each tweet is relatively low, the largest number of words is 30 and the lowest number of words being 5. Therefore when cleaning data, be careful not to remove a lot of words. The distribution between 20-30 words seems to be the peak for sentiments = (-1, 0,1). However, the distribution for a sentiment = 2, shows a peak between 15-25 words. Therefore, no relationship was found when comparing number of words and the value of sentiment it portrays.</li>
# </ul>
# <p> </p>
# <b> Number of mentions</b>
# <ul>
# <li> Each sentiment has atleast one referral to another twitter account since the peak is at one, therefore seems to be no relationship with number of mentions and sentiment.</li>
# </ul>
# <p> </p>
# <b> Number of hashtags</b>
# <ul>
# <li> The distributions between graphs show the peak at zero therefore most of the tweets do not have hastags. Showing no change of number of hashtags with sentiment rating</li>
# </ul>
# <p> </p>
# <b> Number of Capital letters containig 3 or more consecutively</b>
# <ul>
# <li> Most of the tweets seem to show no Capatilized words therefore, no relationship between capitalized words and sentiment</li>
# </ul>
# <p> </p>
# <b> Number of exclamation and question marks</b>
# <ul>
# <li> Most of the tweets seem to show no exclamation or question marks therefore, no relationship between capitalized words and sentiment</li>
# </ul>
# <p> </p>
# <b> Number of URLs</b>
# <ul>
# <li> The tweets with no url link seem to have less sentiment than the tweets that have atleast one url link, this can be compared whe observing sentiment = 2 with the other sentiments. Therefore a relationship seems to show when comparing url links and sentiment</li>
# </ul>
# <p> </p>
# <b> Number of Emojis</b>
# <ul>
# <li> Most of the tweets seem to show no use of emojis therefore, no relationship between emojis and sentiment was detected</li>
# </ul>
# <p> </p>

# ### Data Cleaning

# <p>Before we start using the tweets to train the model for predictions. It is important to clean the data and check for repetions of rows in order to determine whether it will improve the base model of 0.75260 as the f1-Score. The base model is expected to improve once the tweet texts are have been cleaned to reduce the noise obtained within each tweet.</p>

# #### Remove Duplicate Rows

# In[ ]:


#Check for duplicates
duplicate_rows_train = train['message'].duplicated().sum()
duplicate_rows_test = test['message'].duplicated().sum()
print('There are ',duplicate_rows_train,' duplicated rows for the training set')
print('There are ',duplicate_rows_test,' duplicated rows for the test set')


# In[ ]:


# Drop duplicate rows/retweets
train = train.drop_duplicates(subset='message', keep='first',)
train = train.reset_index()
train.drop('index',inplace=True,axis =1)
train.head()


# In[ ]:


# Cleaning the data 
def data_preprocessing(train,test):
    '''
    Cleaning the data based on analysis which includes removing capilised letters, changing contractions like don't to do not,
    replace urls, replace emjicons, remove digits and lastly remove any funny characters in tweets
    
    Parameters
    ----------
    train: data frame
          The data frame of training set
    test: data frame
          The data frame of test set
          
    Output
    ---------
    train: Adds column of tidy tweets to train dataframe
    test: Adds column of tidy tweets to test dataframe
    '''
    def remove_capital_words(df,column):
        df_Lower = df[column].map(lambda x: x.lower())
        return df_Lower
    train['tidy_tweet'] = remove_capital_words(train,'message')
    test['tidy_tweet'] = remove_capital_words(test,'message')
    contra_map = {
                    "ain't": "am not ",
                    "aren't": "are not ",
                    "can't": "cannot",
                    "can't've": "cannot have",
                    "'cause": "because",
                    "could've": "could have",
                    "couldn't": "could not",
                    "couldn't've": "could not have",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hadn't've": "had not have",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'd've": "he would have",
                    "he'll": "he will",
                    "he'll've": "he will have",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is",
                    "i'd": "I would",
                    "i'd've": "I would have",
                    "i'll": "I will",
                    "i'll've": "I will have",
                    "i'm": "I am",
                    "i've": "I have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have",
                    "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not",
                    "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she would",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so is",
                    "that'd": "that would",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why is",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you would",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"}
    contractions_re = re.compile('(%s)' % '|'.join(contra_map.keys()))
    def contradictions(s, contractions_dict=contra_map):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)
    train['tidy_tweet']=train['tidy_tweet'].apply(lambda x:contradictions(x))
    test['tidy_tweet']=test['tidy_tweet'].apply(lambda x:contradictions(x))
    def replace_url(df,column):
        df_url = df[column].str.replace(r'http.?://[^\s]+[\s]?', 'urlweb ')
        return df_url
    train['tidy_tweet'] = replace_url(train,'tidy_tweet')
    test['tidy_tweet'] = replace_url(test,'tidy_tweet')
    def replace_emoji(df,column):
        df_emoji = df[column].apply(lambda x: emoji.demojize(x)).apply(lambda x: re.sub(r':[a-z_&]+:','emoji ',x))
        return df_emoji
    train['tidy_tweet'] = replace_emoji(train,'tidy_tweet')
    test['tidy_tweet'] = replace_emoji(test,'tidy_tweet')
    def remove_digits(df,column):
        df_digits = df[column].apply(lambda x: re.sub(r'\d','',x))
        return df_digits
    train['tidy_tweet'] = remove_digits(train,'tidy_tweet')
    test['tidy_tweet'] = remove_digits(test,'tidy_tweet')	
    def remove_patterns(df,column):
        df_char = df[column].apply(lambda x:  re.sub(r'[^a-z# ]', '', x))
        return df_char
    train['tidy_tweet'] = remove_patterns(train,'tidy_tweet')
    test['tidy_tweet'] = remove_patterns(test,'tidy_tweet')   
    return train,test
(train,test) = data_preprocessing(train,test)


# In[ ]:


#Analyse the cleaned tweets
for index,text in enumerate(train['tidy_tweet'][35:]):
  print('Tweet %d:\n'%(index+1),text)


# <a id='EDA_clean'></a><br>
# ## 5.0 Exploratory Data Analysis after cleaning tweets
# [Back to Table of Contents](#Table_Contents)

# ### Tokenisation

# It is used to describe the process of converting each tweet into a list of tokens, words we actually want. Word tokenizer can be used to find the list of words in a string

# In[ ]:


# Get Tokens of clean tweets
train['token'] = train['tidy_tweet'].apply(lambda x: x.split())
test['token'] = test['tidy_tweet'].apply(lambda x: x.split())


# In[ ]:


train['token'].head()


# ###  Stemming

# Stemming is the process of reducing words to their base or root form. For example, if we were to stem the following words: "runners", "running", "ran", the result would be a single word "run"

# In[ ]:


# use stemming process on clean tweets
stemmer = PorterStemmer()

train['stemming'] = train['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
test['stemming'] = test['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
train['stemming'].head()


# ### Stopwords

# Firstly going to analyse the words after using the stemmatization process to see which words appear the most and to add to the stopword dictionary that has been created in the nltk library.

# In[ ]:


#create list of all cleaned text appearing in tweets
stemma_list_all = []
for index, rows in train.iterrows():
    stemma_list_all.append(rows['stemming'])
flatlist_all = [item for sublist in stemma_list_all for item in sublist]
flatlist_all


# In[ ]:


#Count the number of words apppearing in all the tweets
frequency_dist = FreqDist(flatlist_all)
freq_dist = dict(frequency_dist)
sorted(freq_dist.items(), key= lambda x:-x[1])

#Make Data frame 
df_all = pd.DataFrame(freq_dist.items(),columns = ['Word','Occurrence'])
# Sort values 
df_all = df_all.sort_values('Occurrence', ascending=False)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))

# Plot horizontal bar graph
df_all.iloc[:60].sort_values(by='Occurrence').plot.barh(x='Word',
                      y='Occurrence',
                      ax=ax,
                      color="deepskyblue")

ax.set_title("Plot 4: Common Words Found in all Tweets")

plt.show()


# This shows that the most common words are climate, change, rt, urlweb, global, warm and trump. Therefore some of these words like 'rt' can be added to the stopwords as it adds no sentiment. It going to be exciting to see which words show up the the most for each sentiment after cleaning the stopwords. What is notible here is the large amount of short words appearing in tweets

# In[ ]:


#check for stopwords in train
stop = stopwords.words('english')
train['stopwords'] = train['stemming'].apply(lambda x: len([i for i in x if i in stop]))
train[['stemming','stopwords']].head()


# In[ ]:


#check for stopwords in test
stop = stopwords.words('english')
test['stopwords'] = test['stemming'].apply(lambda x: len([i for i in x if i in stop]))
test[['stemming','stopwords']].head()


# As noticed above, there seems to be quite a few stopwords for the train and specifically more for test. These could add noise in predicting sentiment so it is important to remove them as the length of the words average around 20 per tweet.

# In[ ]:


#create my own stop words from analysis and comparing with general stopwords
stopwords_own =[ 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him',
                'his','himself','she','her','hers','herself','it','itself','they','them','their','theirs','themselves','what',
                'which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has',
                'had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while',
                'of','at','by','for','with','about','against','between','into','through','during','before','after','above',
                'below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here',
                'there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only',
                'own','same','so','than','too','very','s','t','can','will','just','should','now','d','ll','m','o','re','ve','y',
               #my own stopwords found from analysis
                'u','doe','going','ha','wa','l', 'thi','becaus','rt']


# In[ ]:


# def remove_strop_words(df,column):
def remove_stopwords(df,column):
    '''
    Removing the stop words from the clean tweets
    
    Parameters
    ----------
    df: data frame
        Input a dataframe
    column: String
        name of column from data frame
        
    Output
    ----------
    output: df
            Returns a dataframe with no stopwords
    
    '''
    df_stopwords = df[column].apply(lambda x: [item for item in x if item not in stopwords_own])
    return df_stopwords
train['stem_no_stopwords'] = remove_stopwords(train,'stemming')
test['stem_no_stopwords'] = remove_stopwords(test,'stemming')
train['stem_no_stopwords'].head()


# ### Transformation

# Creating a string of tidy tweets for further analysis with visualisation

# In[ ]:


def convert_st_str(df,column):
    '''
    Changes list of strings into one string per row in dataframe
    
    Parameters
    -----------
    df: data frame
        Takes  in a dataframe
        
    Output
    -----------
    output: df_str
            Returns a dataframe with a string instead of list for each row 
    '''
    df_str = df[column].apply(lambda x: ' '.join(x))
    return df_str
train['clean_tweet'] = convert_st_str(train,'stem_no_stopwords')
test['clean_tweet'] = convert_st_str(test,'stem_no_stopwords')
train['clean_tweet'].head()


# ###  Visualising WordClouds

# <p>Plotting a WordCloud will help the common words used in a tweet. The most important analysis is understanding sentiment and the wordcloud will show the common words used by looking at the train dataset</p>
# <p>A word cloud is an image made of words that together resemble a cloudy shape. The clouds give greater prominence to words that appear more frequently in the source text. You can tweak your clouds with different fonts, layouts, and color schemes.</p>

# ####  sentiment of 2

# In[ ]:


#Create WordCloud Plot
news_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 2]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(news_words)
print(wordcloud)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for News Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ####  sentiment of 1

# In[ ]:


#Create WordCloud Plot
pro_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 1]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200,background_color='white').generate(pro_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for postive Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ####  Sentiment of 0

# In[ ]:


#Create WordCloud Plot
neutral_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 0]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(neutral_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for neutral Sentiment")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# #### Sentiment of -1

# In[ ]:


#Create WordCloud Plot
anti_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == -1]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(anti_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for negative Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# <p> When analysing each wordcloud it shows 'climat chang' which is 'climate change' as the most occuring word for any sentiment therefore stating that the data was collected with anyone referencing climate change to view each persons sentiment. The other interesting point to notice is that 'global warming' is mentioned way more in the neutral and negative sentiment than in the positve and news sentiments. The 'urlweb' seems to be appearing in every sentiment but more in sentiment 2 than any other sentiment. We will need to do some further analysis to see exactly which words are appearing the most in each sentiment. The same goes for the peoples names as it seems 'stevesgoddard' shows a lot in the negative sentiment where as for the positve sentiment of 2 shows 'stephenschelegel' appears substantially. Another interesting point, the word cloud for sentiment 2 shows no slang and the use of fuller phrases compared to others. For example instead of just saying 'Trump', 'president Donald Trump' is used.</p>

# ###  Visualising Barplots 

# Creating barplots will give show a very clear representation in terms of the amount of words for each sentiment, the horizontal plot will give insight and show clearly which words are used more dominantly 

# ####  Create a bigram for each sentiment

# #### Plot of Sentiment 2

# In[ ]:


df_news = train[train.sentiment == 2]
top_bi_grams_news=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_news))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 2 sentiment')


# #### Plot of Sentiment 1

# In[ ]:


df_news = train[train.sentiment == 1]
top_bi_grams_pos=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_pos))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 1 sentiment')


# ####  Plot of sentiment 0

# In[ ]:


df_news = train[train.sentiment == 0]
top_bi_grams_neutr=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_neutr))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 0 sentiment')


# #### Plot of sentiment -1

# In[ ]:


df_news = train[train.sentiment == -1]
top_bi_grams_neg = get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_neg))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of -1 sentiment')


# <p>This shows some good insight on paired words  per sentiment as the WorldCloud gave us the idea of how single words are used for each sentiment. This shows for positive sentiment of 2 that there are many links shared for giving a greater sentiment, this proves that the statistical analysis of url links was accurate, However that doesn't mean if someone put a link that it would automatically be a positive sentiment. The other interesting insights are 'donald trump', 'scott pruit' and 'fight climate' for sentiment of 2. Scott Pruit is the head of the EPA organisation that was mentioned above, which makes it clear why he would show signs of news sentiment, fight climate shows how people feel towards climate change.</p>
# <p> Sentiment of 1 shows new aspects of 'believe climate' and 'change denier', deniers work actively to mislead the public and delay policy action to address climate change. therefore the word change means that the public needs to stop being misleaded therefore it is intutively a positive sentiment.
# <p> Sentiment of 0 shows aspects like 'club penguin' which is not shown in any other sentiment, this speaks about intuitively the ice caps melting for penguins, which should show signs rather of positve sentiment, so its interesting that this pair of words is shown in the neutral sentiment.</p> 
# <p> Sentiment of -1 shows some interesting paried features, 'man made', 'made climate' and 'al gore'. Al Gore is a environmentalist and politican in America, he has a campaign to teach people about global warming, so it is very interesting that he appears in the negative sentiment tweets. The other interesting debate and it may be the reason why man made showed up in the tweets, the debate is about is climate change a natural event meaning that humans have add no impact on it, or is it man made therefore humans have had an impact on the environment.</p>

# ### Hashtag plots

# In[ ]:


# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[ ]:


# extracting hashtags from  tweets

HT_neutral = hashtag_extract(train['clean_tweet'][train['sentiment'] == 0])


HT_pro = hashtag_extract(train['clean_tweet'][train['sentiment'] == 1])

HT_news = hashtag_extract(train['clean_tweet'][train['sentiment'] == 2])


HT_anti = hashtag_extract(train['clean_tweet'][train['sentiment'] == -1])
# unnesting list
HT_neutral = sum(HT_neutral,[])
HT_pro = sum(HT_pro,[])
HT_news = sum(HT_news,[])
HT_anti = sum(HT_anti,[])


# #### Sentiment of 2

# In[ ]:


a = nltk.FreqDist(HT_news)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags  
d = d.sort_values(by = 'Count',ascending = False)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d[0:5], x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for news (2) Sentiment")
plt.show()


# #### Sentiment of 1

# In[ ]:


a = nltk.FreqDist(HT_pro)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for positive (1) Sentiment")
plt.show()


# #### Sentiment of 0

# In[ ]:


a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for neutral (0) Sentiment")
plt.show()


# #### Sentiment of -1

# In[ ]:


a = nltk.FreqDist(HT_anti)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for negative (-1) Sentiment")
plt.show()


# For the positve sentiments it seems to show more hashtags focus on the environment and the effects that climate change is having on the enviroment. For the neutral and negative sentiments it seems that the hashtags are focusing on  'trump hashtag' and 'maga' which means Make America Great Again, which could possibly state that they are believing in what trump is doing to he environment

# <a id='vector'></a><br>
# ## 6. Vectorization
# [Back to Table of Contents](#Table_Contents)

# ### CountVector

# The bag of words shows a representation of its words, disregarding grammar and word order but keeping multiplicity

# In[ ]:


#Create Count Vector
cv = CountVectorizer(max_df = 0.90,min_df = 2, max_features = 1000)
bow = cv.fit_transform(train['clean_tweet'])


# In[ ]:


#Plot Count Vector
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(20, 7))
sns.barplot(x ="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.title("Count Vectorizer plot")
plt.show();


# Here we can analyse some clear topics. For example, #0 talking about how trump is impacting climate change for a better economy, #1 talking about how climate change is affecting the sea level because of melting ice caps and #6 talking about protecting the ebvironment and fight against trump city.

# ###  TF-IDF Features

# Frequency Inverse Document Frequency: Some words have high frequency but very little meaningful information. This feature takes into account both the frequency and how meaningful the information is.

# In[ ]:


#Create TF -IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(train['clean_tweet'])


# In[ ]:


#Plot TF-IDF
word_freq = dict(zip(cv.get_feature_names(), np.asarray(tfidf.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(20, 7))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.title("Plot 16: TD - IDF plot")
plt.show();


# As you can see the TF-IDF weights the words differently than the Count Vectorizer from analysing the plots, 'urlweb' seems to be waited much higher compared to the count vectorizer which just counts the amount of words for all tweets.

# ### Word2Vec

# Word2vec is a two-layer neural net processes that convert words into corresponding vectors in such way that the semantically similar vectors are close to each other in N-dimensional space. Where N refers to the dimensions of the vector, in other words it inputs text corpus and outputs text vector. There are two features with word2vec: Skip Gram and Continuous Bag of Words Models. In the Skip Gram model, the context words are predicted using the base word. For instance, given a sentence "I love to dance in the rain", the skip gram model will predict "love" and "dance" given the word "to" as input. Skip-gram was the type used for the wrd2vec created below.

# In[ ]:


#Create Word2Vec 
tokenised_tweet = train['clean_tweet'].apply(lambda x: x.split()) #tokenising
test_tokenised_tweets = test['clean_tweet'].apply(lambda x: x.split())

model_w2v = Word2Vec(            
            tokenised_tweet,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34) 
model_w2v.train(tokenised_tweet,total_examples= len(train['clean_tweet']), epochs=20)


# In[ ]:


model_w2v.wv.most_similar(positive="realdonaldtrump")


# ####  Preparing word2vec feature set

# In[ ]:


#The below function will be used to create a vector for each tweet by taking the average of the vectors
def word_vector(tokens, size):
    '''
    create a vector for each tweet by taking the average of the vectors of the words present in the tweet
    
    Parameters
    ----------
    tokens: list of strings
            Input of tokens per tweet
    size: int
          how many words to vectorize
          
    Output
    ---------
    output: Average vector per token in tweets
    
    '''
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec


# #### preparing word2vec features for models

# In[ ]:


#create word2vec dataframe
wordvec_arrays = np.zeros((len(tokenised_tweet), 200))

for i in range(len(tokenised_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenised_tweet[i], 200)
    
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape


# <a id='modeling'></a><br>
# ## 7.0 Modelling
# [Back to Table of Contents](#Table_Contents)

# A classification model tries to draw some conclusion from the input values given for training. It will predict the class labels/categories for the new data.
# * The Dataset being used here is a multiclass with four classes to predict namely negative, neutral and positive, news sentiments.
# * Hence we use various classification models to classify our data and test the accuracy of classification.
# * The Classifiers being used now are,
#     1. Naive Bayes Classifier
#     2. Linear Support Vector Classifier
#     3. Passive Aggressive Classifier
#     4. Logistic Regression Classifier
#     5. K Nearest Neighbours Clasiffier
#     6. Gradient Boosting Classifier
# * For evaluating the model we check for the Accuracy and F1 scores of the models for performance evaluation.

# ### Trial and Errors 

# <p>The target variable had unbalanced data, so it was imperative to fix this to improve the model in predicting all classes. Oversampling, Undersampling and a balance of Oversampling/Undersamping were created in balancing the target variable in order to improve the scores. The scores for all the models improved in the training set, but as it was submitted in kaggle, there were scores less than 0.7 being achieved, therefore it was overfitting the training results. The conclusion was that we needed another method for balancing the data or more tweets to succeed in the methods mentioned above. This code was not included as it did not perform well and overfitted the models.</p> 
# 
# <p>The clean data was tested and the models performed worse on kaggle with f1 scores of around 0.69, trial and error was used to clean the tweets in various ways where a method was taken out from the precprocessing and other methods like text blogging and vader were added in. The score increased to 0.713 but did not achieve better results than the raw text. For this reason the raw tweets were used to train the models</p>

# ### Split train and test data

# In[ ]:


X = train['message']
y = train['sentiment']
# X_vec = wordvec_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train_vec, X_test_vec, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)


# ### Naive Bayes

# The Bernoulli naive Bayes calssifier assumes that all features are binary such that they only take in two values. The classifier uses Bayes Theorem. It predicts membership probabilities for each class (in this case sentiment) such as the probability given a certain obervation belongs to a particular class. The class with highest probability is considered as the most likely class in this case sentiment 1. Given the features, it selects proability of class with highest outcome.

# ####  Pipeline

# In[ ]:


# TF-IDF Features
pipe1 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',BernoulliNB())])
pipe1.fit(X_train,y_train)
#prediction set
prediction_nb = pipe1.predict(X_test)


# ####  Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_nb),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_nb))


# In[ ]:


# Print overall acuracy
print(accuracy_score(y_test,prediction_nb))


# The result of the f1-score shows that the Naive Bayes classifier predicted sentiment 1 and 2 very well, averaging a score around 0.75, but it could not predict the scores of sentiment 0 and 1, averaging a score of 0.045. This decreased the weighted score and shows why its at 0.58. The accuracy is also not high enough to predict unseen data. 

# ### Linear Support Vector Classifier

# <p>A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. Since it is a linear svc the hyperplane will be predicting each class with a linear kernal. It is vary similar to SVC with kernal = 'linear' and the multiclass support is handled according to a one-vs-the-rest scheme.</p> 
# 
# 

# ####  Pipeline

# In[ ]:


pipe2 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',LinearSVC(random_state = 42))])
pipe2.fit(X_train,y_train)
#prediction set
prediction_lsvc = pipe2.predict(X_test)


# ####  Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_lsvc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_lsvc))


# In[ ]:


# Print overall acuracy
print(accuracy_score(y_test,prediction_lsvc))


# The result of the f1-score shows that the Linear SVC predicted sentiment 1 and 2 very well, averaging a score around 0.79, but had a problem with predicting scores of 0 and 1, averaging a score of 0.47 which is a major improvement from the Naive Bayers Classfier. The imbalance in data shows that it is predicting two classes better than the others. The accuracy is better than the Naives Bayers model.

# ###  Passive Aggressive Classifier

# <p>The goal is to find a hyperplane that seperates all the sentiment classes, on each round of analysing an observation and makes a prediction on the current hypothesis. It then compares prediction to true y and suffers a loss based on the difference. The goal is to make cumulative loss as small as possible. Finally, the hypothesis gets updated according to previous hypothesis and rhe current example.</p>

# ####  Pipeline

# In[ ]:


pipe3 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',PassiveAggressiveClassifier(random_state = 42))])
pipe3.fit(X_train,y_train)
#prediction set
prediction_pas = pipe3.predict(X_test)


# ####  Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_pas),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_pas))


# In[ ]:


# Print overall acuracy
print(accuracy_score(prediction_pas,y_test))


# The result of the f1-score shows that sentiment 1 and 2 averaged a score around 0.78, but had a problem with predicting scores of 0 and 1, averaging a score of 0.45 which shows a slight drop compared to Linear SVC. The imbalance in is still having a major effect. The accuracy has droped slightly comapared to linear svc

# ### Logistic Regression Classifier

# <p>It is a statistical method for analysing a data set in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).</p>
# <p>The goal of logistic regression is to find the best fitting model to describe the relationship between the dichotomous characteristic of interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables. This is better than other binary classification like nearest neighbor since it also explains quantitatively the factors that lead to classification.</p>

# ####  Pipeline

# In[ ]:


pipe4 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',LogisticRegression(random_state = 42))])
pipe4.fit(X_train,y_train)
#prediction set
prediction_lr = pipe4.predict(X_test)


# ####  Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_lr),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_lr))


# In[ ]:


# Print overall acuracy
print(accuracy_score(y_test,prediction_lr))


# The result of the f1-score shows that sentiment 1 and 2 averaged a score around 0.78, but had a problem with predicting scores of 0 and 1, averaging a score of 0.38 which shows a  drop compared to passive aggressive classifier. The imbalance in is still having a major effect. The accuracy has improved slightly compared to passive aggresive classifier.

# ### K Nearest Neighbours Classifier

# K-nearest neighbors (KNN) algorithm uses 'feature similarity' to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors.

# #### Pipeline

# In[ ]:


pipe5 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',KNeighborsClassifier())])
pipe5.fit(X_train,y_train)
#prediction set
prediction_knnc = pipe5.predict(X_test)


# ####  Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_knnc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_knnc))


# In[ ]:


# Print overall acuracy
print(accuracy_score(y_test,prediction_knnc))


# The result of the f1-score shows that sentiment 1 and 2 averaged a score around 0.72, but had a problem with predicting scores of 0 and 1, averaging a score of 0.36 which shows a slight drop compared to logistic regression. The imbalance in is still having a major effect. The accuracy has dropped quite a bit compared to logistic regression.

# ### Gradient Boosting Classifier

# Gradient boosting is a type of machine learning boosting. It relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for this next model in order to minimize the error.

# #### Pipeline

# In[ ]:


pipe6 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',GradientBoostingClassifier(random_state = 42))])
pipe6.fit(X_train,y_train)
#prediction set
prediction_gbc = pipe6.predict(X_test)


# #### Analysis

# In[ ]:


# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_gbc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df


# In[ ]:


# print classification report
print(classification_report(y_test,prediction_gbc))


# In[ ]:


# Print overall acuracy
print(accuracy_score(y_test,prediction_gbc))


# The result of the f1-score shows that sentiment 1 and 2 averaged a score around 0.73, but had a problem with predicting scores of 0 and 1, averaging a score of 0.33 which shows a slight drop compared to K Nearest Neighbours classifier. The imbalance in is still having a major effect. The accuracy has improved slightly compared to K Nearest Neighbours.

# ### Preformance metric of best model

# In[ ]:


#Calculating f1 - scores
nb_f1 = round(f1_score(y_test,prediction_nb, average='weighted'),2)
lsvc_f1 = round(f1_score(y_test,prediction_lsvc, average='weighted'),2)
pac_f1 = round(f1_score(y_test,prediction_pas, average='weighted'),2)
lr_f1 = round(f1_score(y_test,prediction_lr, average='weighted'),2)
knnc_f1 = round(f1_score(y_test,prediction_knnc, average='weighted'),2)
gbc_f1 = round(f1_score(y_test,prediction_gbc, average='weighted'),2)

dict_f1 = {'BernoulliNB':nb_f1,'LinearSVC':lsvc_f1,'PassiveAggressiveClassifier':pac_f1,
                      'LogisticRegression':lr_f1, 'KNeighborsClassifier':knnc_f1,'GradientBoostingClassifier':gbc_f1}
f1_df = pd.DataFrame(dict_f1,index=['f1_score'])
f1_df = f1_df.T
f1_df.sort_values('f1_score',ascending = False)


# The summary of all the f1 scores which shows three models being the top performers, the logistic regression model will be used for futher analysis as it has a slightly higher accuracy compared to passive aggressive classfier, along side with the top performing model.

# ### Word2Vec on top performing models 

# In[ ]:


#Model LSVC
model_lsvc = LinearSVC(random_state = 42)
model_lsvc.fit(X_train_vec,y_train)

#Predict LSVC
predict_vec_lsvc = model_lsvc.predict(X_test_vec)

#Model LR
model_lr = LogisticRegression(random_state = 42)
model_lr.fit(X_train_vec,y_train)
#Predict LR
predict_vec_lr = model_lsvc.predict(X_test_vec)

#Comparing f1 score and accuracy to see if model improved
lsvc_vec_f1 = round(f1_score(predict_vec_lsvc,y_test, average='weighted'),2)
lr_vec_f1 = round(f1_score(predict_vec_lr,y_test, average='weighted'),2)
lsvc_vec_acc = accuracy_score(predict_vec_lsvc,y_test)
lr_vec_acc = accuracy_score(predict_vec_lr,y_test)

#Dict
dict1 = {'Linear SVC vec2word':[lsvc_vec_f1,lsvc_vec_acc],'Logisitc Regression vec2word':[lr_vec_f1 ,lr_vec_acc]}

#Dataframe
gs_rs_df = pd.DataFrame(dict1,index =['f1 score','accuracy']).T
gs_rs_df = gs_rs_df.sort_values('f1 score',ascending =False)
gs_rs_df


# The word2vec performs slightly worse than the TF-IDF when comparing the f1 score and accuracy therefore the TF-IDF will be used for hypertuning on the models that preformed the best.

# <a id='tuning'></a><br>
# ## 8. Hypertuning
# [Back to Table of Contents](#Table_Contents)

# The results showed Linear Support Vector Classifier and Logistic Regression model performing best, mainly observing the weighted f1 score and the accuracy of the model. The logistic regression had an accuracy of 0.71 and an average weighted f1 score of 0.69. The linear support vector classifier had an accuracy of 0.71 and a weighted f1 score of 0.71. Therefore by tuning the parameters of the TD-IDF and Support Vector Classifier, the f1-score should improve. There will be one more model tuned to see if the results improve for the second best performing models based on the weight f1-score

# ## RandomSearchCV

# In[ ]:


#Tuning parameters for TD-IDF first
pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC(random_state = 42))])

parameters = {'tfidf': [TfidfVectorizer()],
           'tfidf__max_df': [0.25,0.5,0.75],
           'tfidf__ngram_range':[(1, 1),(1,2),(2, 2)],
           'tfidf__min_df':(1,2),
           'tfidf__norm':['l1','l2']},

grid_search_tune = RandomizedSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)


# In[ ]:


#Tuning parameters for Linear Support Vector and Passive Agressive Classifiers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.25, min_df=2, ngram_range=(1, 2))),
    ('clf', LinearSVC(random_state = 42))])

parameters = [{'clf':[LinearSVC(random_state = 42)],
           'clf__penalty':['l1','l2'],
           'clf__C':np.logspace(0, 4, 10),
           'clf__class_weight':['balanced',None]},
           {'clf':[LogisticRegression(random_state = 42)],
            'clf__penalty' : ['l1', 'l2'],
            'clf__C' : np.logspace(0, 4, 10),
            'clf__solver' : ["newton-cg", "lbfgs", "liblinear"],
            'clf__class_weight':['balanced',None]}]

grid_search_tune = RandomizedSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)


# ## GridSearchCV

# In[ ]:


#Tuning parameters for TD-IDF first
pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC(random_state = 42))])

parameters = {'tfidf': [TfidfVectorizer()],
           'tfidf__max_df': [0.25,0.5,0.75],
           'tfidf__ngram_range':[(1, 1),(1,2),(2, 2)],
           'tfidf__min_df':(1,2),
           'tfidf__norm':['l1','l2']},

grid_search_tune = GridSearchCV(pipeline, parameters, cv=4, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)


# In[ ]:


#Tuning parameters for Linear Support Vector and Passive Agressive Classifiers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, min_df=2, ngram_range=(1, 2))),
    ('clf', LinearSVC(random_state = 42))])

parameters = [{'clf':[LinearSVC(random_state = 42)],
           'clf__penalty':['l1','l2'],
           'clf__C':np.logspace(0, 4, 10),
           'clf__class_weight':['balanced',None]},
           {'clf':[LogisticRegression(random_state = 42)],
            'clf__penalty' : ['l1', 'l2'],
            'clf__C' : np.logspace(0, 4, 10),
            'clf__solver' : ["newton-cg", "lbfgs", "liblinear"],
            'clf__class_weight':['balanced',None]}]

grid_search_tune = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)


# ### Testing

# #### RandomSearchCV & GridSearchCV

# ##### Pipeline

# In[ ]:


#pipeline for random search cv
randomcv = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer(max_df=0.25, min_df=2, ngram_range=(1, 2))),('classifier',LinearSVC(C=7.742636826811269, random_state=42))])
randomcv.fit(X_train,y_train)
#prediction set
prediction_lsvc_best1 = randomcv.predict(test['message'])

#pipeline for grid search cv
gridcv = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer(max_df=0.75, min_df=2, ngram_range=(1, 2))),('classifier', LinearSVC(C=2.7825594022071245, random_state=42))])
gridcv.fit(X_train,y_train)
#prediction set
prediction_lsvc_best2 = gridcv.predict(X_test)


# #####  Analysis

# In[ ]:


#Calculating f1 - scores
randomcv_f1 = round(f1_score(y_test,prediction_lsvc_best1, average='weighted'),2)
gridcv_f1 = round(f1_score(y_test,prediction_lsvc_best1, average='weighted'),2)
randomcv_acc = accuracy_score(y_test,prediction_lsvc_best1,)
gridcv_acc = accuracy_score(y_test,prediction_lsvc_best2)
dict1 = {'RandomSearch':[randomcv_f1,randomcv_acc],'GridSearch':[gridcv_f1,gridcv_acc]}

gs_rs_df = pd.DataFrame(dict1,index =['f1 score','accuracy']).T
gs_rs_df = gs_rs_df.sort_values('f1 score',ascending =False)
gs_rs_df


# <a id='eval'></a><br>
# ## 9. Evaluation
# [Back to Table of Contents](#Table_Contents)

# <h3> Summary of analysis before modelling</h3>
# <p> The exploratory data analysis gave insight into words used the most in this global topic, and revealed the most prominent personalities that drive and influence opinions.
#    
# People like Donald Trump, Barack Obama, Leonardo Dicaprio, Scott Pruitt, Al Gore Hilary Clinton and Steve Goddard featured mostly. These people do not share same views on climate change but their opinions drive the debate. For example, Donald Trump seems to resonate well with non-believers and his actions are largely seen as poralizing Notable as well is that climate change is largely a political topic.
#    
# Organisations that appeared a lot in these tweets were the Environmental Protection Agency(EPA) where Scott Pruitt was the Administrator from February 17, 2017, to July 6, 2018. Exxon, an oil and and gas company also featured prominently. The United Nations (UN) also had a lot of mentions.
#    
# Another discovery was that appeared a lot in the tweets had 'RT' which means that people retweeted their feelings of sentiment instead of saying anything personal.
#    
# The countries that appeared the most were USA and China. Paris is the most prominent city driven by Paris Climate Agreement and the US walk out.
#    
# The words that appeared the most from the WordClouds are fight climate, scientist, news, warm urlweb (urlweb means a link was originally here), change denier, not believe, believe climate, cause global and the realdonaldtrump.
#    
# <p> The people that are metioned above play a major role in climate change. Trump's first term has been a relentless drive for no restrictions on fossil energy development. This in other words was to create more jobs and GDP for America with no sentiment towards the environment. He changed all the laws that Obama started to protect the environment such as the safety rules for for offshore drilling operations. Leonardo Dicaprio has played a monumental movement in helping climate change where he has a foundation called the Leonardo DiCaprio Foundation, this is intuitively why his name appears a lot in the tweets. Steve Goddard is an environmentalist and is on both sides of the climate debate. He is famous for writing the article The Register, which describes Arctic Sea ice is not receding and claimed that data from the National Snow and Ice Data Center (NSIDC) showing the opposite was incorrect. China is the worlds largest emiiter of carbon dioxide since the economy is growing on a large scale and they have one of the hugest populations. The second country that emmits the most carbon dioxide is the USA. The Paris climate change agreement is the reasons for the city to appear so much in the tweets. The agreement includes commitments from all major emitting countries to cut their climate-altering pollution and to strengthen those commitments over time. This is some of the insight found from analysis and you can see why these organisations, places, people and choice of words have appeared on the topic of climate change.</p>
# 
# <p> The preprocessing of data was done based on the statistical analysis of counting each of the specific characters to see if it added to sentiment. The only aspects left of the tweet were lowercase words and hashtags. The stopwords words used a process called stemming to get the routes of the words, this would help the process of vectorization. The stopwords were removed to decrease the noise from the the actual words that add sentiment. The repeated rows in the train set were removed as it would not add any insight in the training the model. The vectors that can be used to train the models were explored and it showed that TF-IDF and Vec2Word uses a more sophisticated weighting system rather than CountVectorizer therefore they were used in modelling section.
# 
# It is worth noting that text analysis of tweets would differ from normal literature. This is mainly driven by restrictions on characters per tweet which then influences language usage. So words are likely to be shortened and use of slang is common. Notably though, the language used for Sentiment 2, which is News - differs significantly as fuller phrases, peoples names and titles are used.
# 
# The use of raw data with minimal cleaning helps maintain the integrity and sentiment of each tweet.</p>
# 
# <h3> Modelling </h3>
# <p> The models that were used for classification were Naive Bayes Classifier, Linear Support Vector Classifier, Passive Aggressive Classifier, Logistic Regression Classifier, K Nearest Neighbours Classifier and Gradient Boosting Classifier. The two best performing models were logistic regression and linear support vector classifier based off the clean tweets, the accuracy of the Logistic regression was slightly better and so was the f1-score. The raw data seems to achieve better results for the linear models than it did for any of the other models. This shows that the target variables were seperated linearly where a hyperplane could classify better which features distinguished each sentiment.</p>
# 

# <a id='con'></a><br>
# ## 10. Conclusion
# [Back to Table of Contents](#Table_Contents)

# The imbalance of data seemed to have the biggest affect on the results of our models, it had a major problem with predicting sentiments of 0 and -1, the cleaning of data seemed to take away from the tweet messages since the length of the average tweets was around 15-25 words, taking away from this seemed to make it harder for the models to predict each of the sentiments. The hypertuning did not improve the models substantially and only showed slight improvement. In terms of recommendations, first to focus on finding a methods to balance the target variable. Secondly, to spend a lot more time on traning the models and tuning them.

# <a id='pkl'></a><br>
# ## 11. Submission
# [Back to Table of Contents](#Table_Contents)

# In[ ]:


my_submission = pd.DataFrame({'tweetid': test.tweetid, 'sentiment': prediction_lsvc_best1})
# you could use any filename. We choose submission here
my_submission.to_csv('finalsubmission.csv', index=False)


# In[ ]:




