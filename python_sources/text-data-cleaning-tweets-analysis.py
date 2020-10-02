#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Problem Statement - Predict Tweets Category (0/1)
# 
# 
#  
#  
# ** If you find the content informative to any extend, kindly encourge me by upvoting. **

# # Load Libraries

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 100)


# In[ ]:


# Load dataset
def load_data():
    data = pd.read_csv('../input/Data.csv')
    return data


# In[ ]:


tweet_df = load_data()
tweet_df.head()


# In[ ]:


print('Dataset size:',tweet_df.shape)
print('Columns are:',tweet_df.columns)


# In[ ]:


tweet_df.info()


# In[ ]:


sns.countplot(x = 'ADR_label', data = tweet_df)


# In[ ]:


df  = pd.DataFrame(tweet_df[['UserId', 'Tweet']])


# # Exploratory Data Analysis
# ## Wordcloud Visualization 

# In[ ]:


from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
# Start with one review:
df_ADR = tweet_df[tweet_df['ADR_label']==1]
df_NADR = tweet_df[tweet_df['ADR_label']==0]
tweet_All = " ".join(review for review in df.Tweet)
tweet_ADR = " ".join(review for review in df_ADR.Tweet)
tweet_NADR = " ".join(review for review in df_NADR.Tweet)

fig, ax = plt.subplots(3, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_ADR = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_ADR)
wordcloud_NADR = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_NADR)

# Display the generated image:
ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_ADR, interpolation='bilinear')
ax[1].set_title('Tweets under ADR Class',fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_NADR, interpolation='bilinear')
ax[2].set_title('Tweets under None - ADR Class',fontsize=30)
ax[2].axis('off')

#wordcloud.to_file("img/first_review.png")


# Quick Notes:-
#     - Few high frequency tokens such as 'treatment', 'patient', 'therapy' are frequently used in both the categorical classes (ADR/non-ADR)
#     - Removing these words along with stops words would not impact the performance.

# ## Pre-processing text data
# Most of the text data are cleaned by following below steps. 
# 
# 1. Remove punctuations 
# 2. Tokenization - Converting a sentence into list of words
# 3. Remove stopwords
# 4. Lammetization/stemming - Tranforming any form of a word to its root word
# 

# ### Remove punctuations

# In[ ]:


string.punctuation


# In[ ]:


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['Tweet'].apply(lambda x: remove_punct(x))
df.head(10)


# ## Tokenization

# In[ ]:


def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
df.head()


# ## Remove stopwords
# 
# - Identified few more words to be removed along with English stopwords
#          - (['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
#               'treatment', 'associated', 'patients', 'may','day', 'case','old'])

# In[ ]:


stopword = nltk.corpus.stopwords.words('english')
#stopword.extend(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
#               'treatment', 'associated', 'patients', 'may','day', 'case','old'])


# In[ ]:


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
df.head(10)


# ##  Stemming and Lammitization
# 
# Ex - developed, development

# In[ ]:


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))
df.head()


# In[ ]:


wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
df.head()


# In[ ]:


def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


# ## Vectorisation
# 
#     - Cleaning data in single line through passing clean_text in the CountVectorizer

# In[ ]:


countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['Tweet'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())


# In[ ]:


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()


# ## Feature Creation

# Assuming a hypothesis to create new features
# #### Hypothesis:
#     - N0 : ADR - has long text length than NADR
#     - N1: ADR - has not long text length than NADR

# In[ ]:


ADR_tweet_1 = tweet_df[tweet_df['ADR_label'] == 1]['Tweet'].apply(lambda x: len(x) - len(' '))
ADR_tweet_0 = tweet_df[tweet_df['ADR_label'] == 0]['Tweet'].apply(lambda x: len(x) - len(' '))


# In[ ]:


bins_ = np.linspace(0, 450, 70)

plt.hist(ADR_tweet_1, bins= bins_, normed=True, alpha = 0.5, label = 'ADR')
plt.hist(ADR_tweet_0, bins= bins_, normed=True, alpha = 0.1, label = 'None_ADR')
plt.legend()


# ## Summary of Exploratory Data Analysis
#         - As any text data, tweets are quite unclean having punctuations, numbers and short cuts.
#         - Most of the short cuts or abbreviations can either be transformed or dropped. I have dropped them here for, I don't have complete details on them.
#         - There are few words such as patient, therapy, etc, occuring most frequently than other words and are common in all classification categories. As high frequency of these words in this dataset does not add to the word significance and dropping them along with stop words will not influence much on each result or metrics.
#         - Among stemmer and lemmatizer, stemmer looks to be working in this dataset, however, later can also be used and try during model building and evaluation.
#         - As part of Feature creation, text lenght of ADR class and None ADR class is compared. Though there is not much difference in text legth pattern for the same. Other hypothesis can also be developed to generate more features such as frequency of 'renal failure' in ADR tweets compared to 'NONE-ADR' tweets. Selection of the key words certainly depends on questions like - if tweets are for one product or several products combined.

# *This notebook is under progress and further code for model building, model evaluation and hyper parameter tuning will be added soon. *
#     
