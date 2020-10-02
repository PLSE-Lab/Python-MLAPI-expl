#!/usr/bin/env python
# coding: utf-8

# <h1>Baseline Classification models</h1>
# 
# In this notebook, I'll do all the text preprocessing to our data and then I'll start with some baseline classifier models like BOW to classify the disaster tweets.  In the subsequent notebooks, I'll use more advanced methodologies to try to improve on the predictions from this model. 

# <h2>Load the data</h2>
# 
# Let's first start with loading the data

# In[ ]:


import pandas as pd

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print('Train dataframe shape:', train.shape)
print('Test dataframe shape:', test.shape)


# In[ ]:


train.head()


# So, disaster tweets are labeled as 1. 

# In[ ]:


test.head()


# <h2>Text Preprocessing</h2>
# 
# Clean up the text using some basic NLP methods

# In[ ]:


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
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
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
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
"so's": "so as",
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
"you've": "you have"
}


# In[ ]:


import re 

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)            if contraction_mapping.get(match)             else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

expand_contractions("y'all I've don't I'd we're")


# In[ ]:


from bs4 import BeautifulSoup
import nltk
import numpy as np

wpt = nltk.WordPunctTokenizer()

stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text, expand=True):
    # strip html tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # remove URL
    text = re.sub(r'http://\S+|https?://\S+|www\.\S+', '', text)
    
    if expand:
        text = expand_contractions(text)

    # lower case, remove special characters, numbers and strip leading and trailing whitespaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    
    # remove emojis
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # tokenize
    tokens = wpt.tokenize(text)
    
    # filter out stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # Join tokens
    text = ' '.join(filtered_tokens)
    return text

clean_corpus = np.vectorize(clean_text)


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['clean_text'] = clean_corpus(train['text'])")


# In[ ]:


train.head()


# <h2>EDA</h2>
# 
# Let's visualize the data and look at some distributions to get a feel for the data and identify imbalances in our features.

# <h3>Target Distribution</h3>
# 
# Let's look at the labeled train data to see what the distribution of the target feature looks like. 

# In[ ]:


import matplotlib.pyplot as plt

train['target'].value_counts().plot(kind='bar', figsize=(10, 6), rot=0)
plt.xlabel('Target label', labelpad=14)
plt.ylabel('Counts of Target', labelpad=14)
plt.title('Counts of Each Labeled Target', y=1)


# In[ ]:


train['target'].value_counts()/len(train)


# Roughly, 43% are disaster tweets and 57% are non-disaster tweets.  

# <h3>Word distribution</h3>
# 
# Frequency of words across the tweets. We'll look at word clouds as an informal visualization and then try to get actual frequencies by unigram, bigrams and trigrams. 

# In[ ]:


train_0 = train[train['target']==0]
train_1 = train[train['target']==1]


# <h4>Word Cloud</h4>
# Let's use one of my favorite visuals -- word clouds -- to see the frequency of words in tweets per target, i.e., 0 or 1

# In[ ]:


from wordcloud import WordCloud

def plot_wordcloud(text, ax, clean_text=False, mask=None, max_words=200, max_font_size=100, 
                   title=None, title_size=40):

    if clean_text:
        text = clean_text(text)
    
    wordcloud = WordCloud(background_color='black',
                         stopwords = stopwords,
                         max_words = max_words,
                         max_font_size = max_font_size,
                         random_state = 0,
                         width = 800, 
                         height = 800, 
                         mask = mask)
    wordcloud.generate(text)
  
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title)


# Let's see some of the most frequent words in the di

# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
plot_wordcloud(' '.join(train_0['clean_text']), ax=axis1, title='Frequent words in non-disaster tweets')
plot_wordcloud(' '.join(train_1['clean_text']), ax=axis2, title='Frequent words in disaster tweets')


# Wordclouds immediately show some overall differences in the frequency of words. As expected, there is a higher frequency of words like 'death', 'fire', 'suicide', 'flood' etc. in disaster tweets relative to non-disaster tweets.  But, this is obviously very informal.  Let's try to formalize this a bit and get some real numbers!

# <h4>Unigram Frequencies</h4>

# In[ ]:


def generate_ngrams(text, n=1):
    tokens = [token for token in text.lower().split(' ') if token !='']
    ngrams = zip(*([tokens[i:] for i in range(n)]))
    return [' '.join(ngram) for ngram in ngrams]

# test ride the function
train_0['clean_text'][:10].apply(lambda x: generate_ngrams(x, n=2))


# Tokenize the text and get each word's frequency

# In[ ]:


from collections import Counter

train_0_unigrams = generate_ngrams(' '.join(train_0['clean_text']), n=1)
train_1_unigrams = generate_ngrams(' '.join(train_1['clean_text']), n=1)

train_0_counter = pd.DataFrame(Counter(train_0_unigrams).items(), columns = ['token', 'frequency'])
train_0_counter = train_0_counter.sort_values('frequency', ascending=False)
train_1_counter = pd.DataFrame(Counter(train_1_unigrams).items(), columns = ['token', 'frequency'])
train_1_counter = train_1_counter.sort_values('frequency', ascending=False)


# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
train_0_counter[:50].plot.barh(x='token', y='frequency', ax=axis1, title='Non disaster tweets token frequency')
train_1_counter[:50].plot.barh(x='token', y='frequency', ax=axis2, title='Disaster tweets token frequency')
fig.tight_layout()


# Seems that non-disaster related tweets have a higher frequency of non-influential words, whereas disaster tweets have a higher frequency of relevant tokens like 'fire', 'disaster', 'suicide', 'storm', etc. 

# <h4>Bigram Frequencies</h4>
# 
# Now, let's look at bigram frequencies.  These may be more relevant to us

# In[ ]:


train_0_bigrams = generate_ngrams(' '.join(train_0['clean_text']), n=2)
train_1_bigrams = generate_ngrams(' '.join(train_1['clean_text']), n=2)

train_0_counter = pd.DataFrame(Counter(train_0_bigrams).items(), columns = ['token', 'frequency'])
train_0_counter = train_0_counter.sort_values('frequency', ascending=False)
train_1_counter = pd.DataFrame(Counter(train_1_bigrams).items(), columns = ['token', 'frequency'])
train_1_counter = train_1_counter.sort_values('frequency', ascending=False)


# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
train_0_counter[:50].plot.barh(x='token', y='frequency', ax=axis1, title='Non disaster tweets token frequency')
train_1_counter[:50].plot.barh(x='token', y='frequency', ax=axis2, title='Disaster tweets token frequency')
fig.tight_layout()


# Cool! The bigrams are definitely more informative for us humans. Not sure if bigrams would actually help when training the models since the key disaster-related words like 'suicide', 'wildfire', 'bomber', etc. are still fairly independent and predictive. Either way, let's look at trigrams, just for fun. 

# <h4>Trigram frequencies</h4>

# In[ ]:


train_0_trigrams = generate_ngrams(' '.join(train_0['clean_text']), n=3)
train_1_trigrams = generate_ngrams(' '.join(train_1['clean_text']), n=3)

train_0_counter = pd.DataFrame(Counter(train_0_trigrams).items(), columns = ['token', 'frequency'])
train_0_counter = train_0_counter.sort_values('frequency', ascending=False)
train_1_counter = pd.DataFrame(Counter(train_1_trigrams).items(), columns = ['token', 'frequency'])
train_1_counter = train_1_counter.sort_values('frequency', ascending=False)


# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
train_0_counter[:50].plot.barh(x='token', y='frequency', ax=axis1, title='Non disaster tweets token frequency')
train_1_counter[:50].plot.barh(x='token', y='frequency', ax=axis2, title='Disaster tweets token frequency')
fig.tight_layout()


# Trigrams definitely seem extraneous. 

# <h4>Meta features</h4>
# 
# I'll look at the following meta features since they might be useful features to consider in the final model:
# 
# 1. Number of words in the text
# 2. Number of characters in the text

# In[ ]:


train['num_words'] = train['text'].apply(lambda x: len(x.split(' ')))
train['num_characters'] = train['text'].apply(lambda x: len(x))


# In[ ]:


train.head()


# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

train.boxplot(column=['num_words'], by=['target'], ax=axis1)
train.boxplot(column=['num_characters'], by=['target'], ax=axis2)


# On average, there are more characters in a disaster tweet, but not significant. 

# <h4>Save the cleaned data</h4>
train.to_csv('/kaggle/data/output/train_cleaned.csv', index=False)
# In[ ]:


train.to_csv('train_cleaned.csv', index=False)


# In[ ]:


test['clean_text'] = clean_corpus(test['text'])
test.head()


# In[ ]:


test.to_csv('test_cleaned.csv', index=False)


# In[ ]:





# In[ ]:




