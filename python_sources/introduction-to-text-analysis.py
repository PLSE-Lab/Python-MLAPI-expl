#!/usr/bin/env python
# coding: utf-8

# # Introduction to Text Analysis
# 
# ![](https://www.kdnuggets.com/images/sentiment-fig-1-689.jpg)
# 
# Text analysis is a process for parsing texts in order to extract some machine-readable facts. In this tutorial, we will play with sentiment analysis using Food Reviews dataset from Amazon. 
# 
# ### What will you learn?
# 
# Some basic operations of the following libraries:
# * `NLTK`: a leading platform for building Python programs to work with human language data.
# * `pandas`: a library for tabular data, provides fast, flexible, and expressive data structures designed to make working with structured.
# * `matplotlib`:  a plotting library, produces publication quality figures in a variety of hardcopy formats.

# In[ ]:


# Let's import what we gonna to use firstly.
import nltk
import pandas as pd


# # Load the dataset

# In[ ]:


# read a tabular dataset.
dataset = pd.read_csv('../input/Reviews.csv')


# > # Overview

# In[ ]:


# the function `head(n: int)` will show the first n rows of data. The default value of n is 5.
dataset.head()


# In[ ]:


dataset['Score'].value_counts()


# In[ ]:


for ele in dataset['Score'].values:
    mapping(ele)


# In[ ]:


# For easy-understanding, we modify this task to fine-grained estimation to binary classification.
def mapping(x):
    if x > 4: return 1
    if x <= 4: return 0

dataset['label']= dataset['Score'].apply(mapping)

# We would only use texts and labels, so just keep it simple.
dataset = dataset[['Text', 'label']]


# In[ ]:


# Check the dataset again.
dataset.head()


# # Tokenization
# ![](https://i.imgur.com/0E4JLiZ.png)

# In[ ]:


s = 'a b c'
s.split(' ')


# In[ ]:


def split_str(x):
    return x.split(' ')


# In[ ]:


# We can tokenize a sentence using a tokenizer or splitting by space simply.
dataset['tokenized_sentences_nltk'] = dataset['Text'].apply(nltk.word_tokenize) # it take some times
dataset['tokenized_sentences_naive'] = dataset['Text'].apply(lambda s: s.split(' '))


# ## Remove the stopwords

# In[ ]:


stopwords = set(nltk.corpus.stopwords.words())
print(stopwords)


# In[ ]:


def clean_stopwords(sentence):
    res = []
    for word in sentence:
        if word not in stopwords:
            res.append(word)
    return res
    # return [w if w not in stopwords for w in sentence]

dataset['tokenized_sentences_nltk_remove_stopwords'] = dataset['tokenized_sentences_nltk'].apply(clean_stopwords)


# In[ ]:


pd.set_option('display.max_colwidth', 300)


# In[ ]:


dataset[['Text', 'tokenized_sentences_nltk', 'tokenized_sentences_naive', 'tokenized_sentences_nltk_remove_stopwords']].head()


# In[ ]:


# Extract the data
cleaned_texts = dataset['tokenized_sentences_nltk_remove_stopwords'].values
labels = dataset['label'].values


# # A simple algorithm
# 
# Let's assume words in a sentence are totally independent which means we can consider each single word as a feature. So, for a sentence with 10 words, it would has 10 features and what we're going to do is predicting the sentiments using the words tendency.
# 
# ![](https://i.imgur.com/StdAo5H.png)

# In[ ]:


# Calculate the word frequencies and word energies
word_frequency = {} 
word_energy = {}

# To handle missing key, you can use defaultdict which would initialize a item that miss it's key with a default value. 
# Or you can just use the `Counter` class in this situation.
# ---
# from collections import defaultdict, Counter
# ...
# ---

for text, label in zip(cleaned_texts, labels):
    for word in text:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
            
        if label == 1:
            if word not in word_energy:
                word_energy[word] = 1
            else:
                word_energy[word] += 1
        else:
            if word not in word_energy:
                word_energy[word] = -1
            else:
                word_energy[word] -= 1


# # Normalization

# In[ ]:


for word in word_energy:
    word_energy[word] /= word_frequency[word]


# # Remove the unreliable results

# In[ ]:


reliable_word_energy = {}
for word in word_energy:
    # we assume that the energies would be reliable only for words that appear more than 500 times in our corpus. 
    if word_frequency[word] > 500: 
        reliable_word_energy[word] = word_energy[word]


# # Visualization

# In[ ]:


top_30_positive_words = [v[0] for v in sorted(reliable_word_energy.items(), key=lambda x: x[1], reverse=True)[:30]]
top_30_negative_words = [v[0] for v in sorted(reliable_word_energy.items(), key=lambda x: x[1], reverse=False)[:30]]


# In[ ]:


from wordcloud import WordCloud
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_word_clouds(keywords):
    wordcloud = WordCloud().generate(' '.join(keywords))
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")


# In[ ]:


plot_word_clouds(top_30_positive_words)


# In[ ]:


plot_word_clouds(top_30_negative_words)


# # Calculate the energies

# In[ ]:


def calc_energies(sentence):
    score = 0
    for word in sentence:
        if word in reliable_word_energy:
            score += reliable_word_energy[word]
    score /= len(sentence) # normalization
    return score

dataset['sentiment_energy'] = dataset['tokenized_sentences_nltk_remove_stopwords'].apply(calc_energies)


# # Search the best threshold

# In[ ]:


import numpy as np

def threshold_searching(left=-1, right=1, num_thresholds=101):
    record = []
    for i in np.linspace(left, right, num_thresholds):
        record.append(((dataset['sentiment_energy'] > i).astype('int') == dataset['label']).sum() / len(dataset))
    plt.title('Threshold Searching')
    plt.xlabel('Threshold Value')
    plt.ylabel('Accuracy')
    plt.plot(np.linspace(left, right, num_thresholds), record)
    best_train_threshold = np.linspace(left, right, num_thresholds)[record.index(max(record))]
    best_train_accuracy = max(record)
    return best_train_threshold, best_train_accuracy

best_train_threshold, best_train_accuracy = threshold_searching()
print(best_train_threshold, best_train_accuracy)


# In[ ]:


def our_sentiment_classifier(sentence, threshold=best_train_threshold):
    energy = calc_energies(sentence)
    if energy > threshold:
        return 1
    else:
        return 0


# In[ ]:


print('prediction', our_sentiment_classifier(['i', 'will', 'not', 'buy', 'it', 'again']))
print('prediction', our_sentiment_classifier(['it', 'is', 'really', 'delicious']))


# In[ ]:




