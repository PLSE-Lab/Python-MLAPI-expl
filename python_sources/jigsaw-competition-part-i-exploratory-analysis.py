#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. [Introduction](#1)
# 2. [Exploratory Analysis](#2)
#  - 2.1 [Class Distribution](#3)
#  - 2.2 [Token Size Distribution](#4)
#  - 2.3 [Sentiment Analysis](#5)
#  - 2.4 [Word Cloud](#6)

# # 1. Introduction <a></a>
# In order to understand the training data and address any possible shortcomings that may affect the models' performance beforehand, a detailed exploratory analysis has been performed. A basic EDA for text classification contains:
# 
# * **Data types and format**
# * **Basic statistics** - average number of words per sequence, median, max and min
# * **Size of the dataset and number of classes**
# * **Number of samples per class** - useful to check potential class imbalance
# * **Number of words per sample** - helps understand if context capabilities may be required i.e. for long sentences meaning may fade or become more complex with increased size
# * **Distribution of words per class** - helps unveil which words/tokens are informative to class membership
# 
# Other more complex visualizations may help us improve our understanding of the data, such as:
# * **Sentiment analysis** - visualization of positivity/negativity/complexity/etc. of the text data per class
# * **Word cloud** - visualization of top N most common tokens per class
# 
# First, import required libraries and datasets:

# In[ ]:


import numpy as np
import pandas as pd
import json
from tqdm import tqdm
tqdm.pandas()
pd.set_option("display.precision", 2)

import warnings
warnings.filterwarnings('ignore')

# import os for system interaction and garbage collector
import os
import gc

# import textblog for text processing
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import gensim for statistical semantical analysis and structure
import gensim

from sklearn.model_selection import KFold

from keras.layers import *
from keras.initializers import *
from keras.constraints import *
from keras.regularizers import *
from keras.activations import *
from keras.optimizers import *
import keras.backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from IPython.display import SVG
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import zipfile

# unzip file to specified path
def import_zipped_data(file, output_path):
    with zipfile.ZipFile("../input/jigsaw-toxic-comment-classification-challenge/"+file+".zip","r") as z:
        z.extractall("/kaggle/input")
        
datasets = ['train.csv', 'test.csv', 'test_labels.csv']

kaggle_home = '/kaggle/input'
for dataset in datasets:
    import_zipped_data(dataset, output_path = kaggle_home)


# In[ ]:


test_df = pd.read_csv('/kaggle/input/test.csv')
train_df = pd.read_csv('/kaggle/input/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


# col with input text
TEXT = 'comment_text'


# # 2. Exploratory Analysis <a></a>
# ## 2.1 Class Distribution
# Let's visualize the number of toxic and non-toxic comments contained in the dataset:

# In[ ]:


non_toxic = len(train_df[train_df['toxic'] == 0])
toxic = len(train_df[train_df['toxic'] == 1])
print(f'There are {non_toxic} non-toxic comments, representing a {round((non_toxic/len(train_df)*100))}% of the total {len(train_df)} samples collected.')
print(f'Only {toxic} - about a {round((toxic/len(train_df))*100)}% - are toxic comments.')


# In[ ]:


plt.barh(['non_toxic', 'toxic'], [non_toxic, toxic], color = 'r', alpha = 0.5)
plt.title('Toxicity distribution')
plt.show()


# There is a clear **class inbalance**, as only about 10% of the comments are toxic. It is reasonable to assume that, given homogeneous sampling, there will always be a bigger proportion non-toxic comments.
# 
# Now we'll explore the toxicity subtypes. Are they imbalanced as well?

# In[ ]:


labels = ['obscene', 'threat', 'insult', 'identity_hate']
class_cnt = {}
for label in labels:
    # count number of samples per toxicity type
    class_cnt[label] = len(train_df[train_df[label] == 1])
    
# sort dict from bigger to lower key value
class_cnt = {k: v for k, v in sorted(class_cnt.items(), key = lambda item: item[1], reverse = True)}


# In[ ]:


plt.bar(*zip(*class_cnt.items()), color = 'r', alpha = 0.5)
plt.title('Toxicity type distribution')
plt.show()


# In[ ]:


print(f'The percentage respect to toxic comments of each toxicity subtype are:')
for label in labels:
    print(f'>> {label} comments: {round((class_cnt[label]/toxic)*100)}%')


# In[ ]:


print(f'The percentage respect to toxic comments of each toxicity subtype are:')
for label in labels:
    print(f'>> {label} comments: {round((class_cnt[label]/len(train_df))*100)}%')


# It looks like obscene and insult toxicity types are almost on pair. On the other hand, identity hate appears in a much lower percentage, and threat comments comprise only 3% of the toxic comments and less than a 1% of the total dataset. This fact may lead to poor classification performance in regards to these classes. On the model evaluation stage, we'll pay attention to the ratios of False Negatives and False Positives for the minority subgroups.
# 
# Lastly, let's take a look at the distribution of comments tagged as severly toxic:

# In[ ]:


labels = ['toxic', 'severe_toxic']
class_cnt = {}
for label in labels:
    # count number of samples per toxicity type
    class_cnt[label] = len(train_df[train_df[label] == 1])
    
# sort dict from bigger to lower key value
class_cnt = {k: v for k, v in sorted(class_cnt.items(), key = lambda item: item[1], reverse = True)}


# In[ ]:


plt.bar(*zip(*class_cnt.items()), color = 'r', alpha = 0.5)
plt.title('Toxicity level distribution')
plt.show()


# They comprise less that 10% of all toxic comments. Since severely toxic comments are sometimes also labeled as toxic, the discrimination may be more difficult.

# ## 2.2 Token Size Distribution
# Depending on the format of the input sequence tokens, different models may be better suitted for the classification tasks. Google recommends computing the word/sequence ration in order to define the complexity of the model, as detailed [here](https://developers.google.com/machine-learning/guides/text-classification/step-2-5). 
# 
# First, we'll obtain the number of characters per input sequence.
# 
# ### Number of characters per sentence

# In[ ]:


# compute character length of comments
lengths = train_df[TEXT].apply(len)
lengths_df = lengths.to_frame()
# print basic metrics
lengths.mean(), lengths.std(), lengths.min(), lengths.max()


# In[ ]:


lengths = train_df[TEXT].apply(len)
lengths_df = lengths.to_frame()
sns.boxplot(x=lengths_df, color='r')
plt.title('Boxplot of characters per sentence')
plt.show()


# Use IQR to filter out the outliers:

# In[ ]:


Q1, Q3 = lengths_df.quantile(0.25), lengths_df.quantile(0.75)
IQR = Q3 - Q1
IQR


# In[ ]:


legths_df_iqr = lengths_df[lengths_df[TEXT] < int(round(IQR))]
sns.boxplot(x=legths_df_iqr, color='r')
plt.title('Boxplot of characters per sentence without IQR Outliers')
plt.show()


# In[ ]:


lengths = train_df[TEXT].apply(len)
train_df['lengths'] = lengths
lengths = train_df.loc[train_df['lengths']<1125]['lengths']
sns.distplot(lengths, color='r')
plt.title('Number of characters per sentence')
plt.show()


# Most input texts are below the 400 character mark.

# ### Number of words per sentence

# In[ ]:


words = train_df[TEXT].apply(lambda x: len(x) - len(''.join(x.split())) + 1)
train_df['words'] = words
words = train_df.loc[train_df['words']<200]['words']
sns.distplot(words, color='r')
plt.title('Number of words per sentence')
plt.show()


# Words per sentence follows a distribution very similar to that of the number of characters.

# ### Average Word Length

# In[ ]:


avg_word_len = train_df[TEXT].apply(lambda x: 1.0*len(''.join(x.split()))/(len(x) - len(''.join(x.split())) + 1))
train_df['avg_word_len'] = avg_word_len
avg_word_len = train_df.loc[train_df['avg_word_len']<10]['avg_word_len']
sns.distplot(avg_word_len, color='b')
plt.title('Average word length')
plt.show()


# Average word length follows a simple bell-shaped normal distribution with a mean of around 4.5
# ### Word size per label (toxic/non-toxic)

# In[ ]:


# take a small sample of training dataset to speed up sentiment analysis
tiny_train_df = train_df.sample(n=10000)


# In[ ]:


import matplotlib.patches as mpatches
non_toxic_0 = tiny_train_df.loc[(tiny_train_df.toxic<0.5) & (tiny_train_df.words<200)]['words']
toxic_1 = tiny_train_df.loc[(tiny_train_df.toxic>0.5) & (tiny_train_df.words<200)]['words']
sns.distplot(non_toxic_0, color='green')
sns.distplot(toxic_1, color='red')
red_patch = mpatches.Patch(color='red', label='Toxic')
green_patch = mpatches.Patch(color='green', label='Non-Toxic')
plt.legend(handles=[red_patch, green_patch])
plt.title('Toxicity per word number in text')
plt.show()


# Interestingly, there is an greater proportion of toxic comments that are shorter, which suggests a possible correlation between these two metrics.

# ## 2.2 Sentiment Analysis

# ### Sentiment - negativity

# In[ ]:


# take a small sample of training dataset to speed up sentiment analysis
tiny_train_df = train_df.sample(n=10000)


# In[ ]:


sia = SentimentIntensityAnalyzer()
non_toxic_0 = tiny_train_df.loc[tiny_train_df.toxic<0.5][TEXT].apply(lambda x: sia.polarity_scores(x))
toxic_1 = tiny_train_df.loc[tiny_train_df.toxic>0.5][TEXT].apply(lambda x: sia.polarity_scores(x))


# In[ ]:


sns.distplot([polarity['neg'] for polarity in non_toxic_0], color='green')
sns.distplot([polarity['neg'] for polarity in toxic_1], color='red')
red_patch = mpatches.Patch(color='red', label='Toxic')
green_patch = mpatches.Patch(color='green', label='Non-Toxic')
plt.legend(handles=[red_patch, green_patch])
plt.title('Distribution of negativity in comments')
plt.show()


# The toxic comments distribution has a higher mean than the non-toxic, which shows that such comments generally tend to be more negative on average. 

# ### Sentiment - positivity

# In[ ]:


sns.distplot([polarity['pos'] for polarity in non_toxic_0], color='green')
sns.distplot([polarity['pos'] for polarity in toxic_1], color='red')
red_patch = mpatches.Patch(color='red', label='Toxic')
green_patch = mpatches.Patch(color='green', label='Non-Toxic')
plt.legend(handles=[red_patch, green_patch])
plt.title('Distribution of positivity in comments')
plt.show()


# Both distributions are very similar. This shows that there is no significant difference in terms of positivity levels between toxic and non-toxic comments.

# ### Sentiment - neutrality

# In[ ]:


sns.distplot([polarity['neu'] for polarity in non_toxic_0], color='green')
sns.distplot([polarity['neu'] for polarity in toxic_1], color='red')
red_patch = mpatches.Patch(color='red', label='Toxic')
green_patch = mpatches.Patch(color='green', label='Non-Toxic')
plt.legend(handles=[red_patch, green_patch])
plt.title('Distribution of neutrality in comments')
plt.show()


# The non-toxic comments distribution clearly has a higher mean than the toxic distribution and higher neutrality on average. This may be due to toxic comments expressing more extreme emotions - like hate or anger.

# ### Sentiment - compoundness / complexity of comment)

# In[ ]:


sns.distplot([polarity['compound'] for polarity in non_toxic_0], color='green')
sns.distplot([polarity['compound'] for polarity in toxic_1], color='red')
red_patch = mpatches.Patch(color='red', label='Toxic')
green_patch = mpatches.Patch(color='green', label='Non-Toxic')
plt.legend(handles=[red_patch, green_patch])
plt.title('Distribution of complexity in comments')
plt.show()


# Interentingly, the distribution of complexity in toxic comments is skewed towards the left. This matches with the idea that toxic comments are less gramatically complex, with many of the consisting in short, aggregsive threats or insults.
# 
# On the other hand, non-toxic comments may be written with a more constructive tone, often trying to express an opinion or make a point, thus using more complex structures.

# ## 2.3 Word Cloud

# In[ ]:


from wordcloud import WordCloud

def class_wordcloud(dataframe, label, max_words):
    # data preprocessing: concatenate all reviews per class
    text = " ".join(x for x in dataframe[dataframe[label]==1].comment_text)

    # create and generate a word cloud image
    wordcloud = WordCloud(max_words=max_words, background_color="white", collocations=False).generate(text)

    # display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most popular {max_words} words in class {label}")
    plt.show()


# In[ ]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for label in labels:
    class_wordcloud(train_df, label, 50)


# In contrast, the 50 most popular words for comments that are not labeled as toxic are:

# In[ ]:


# data preprocessing: concatenate all reviews per class
text = " ".join(x for x in train_df[train_df['toxic']==0].comment_text)

# create and generate a word cloud image
wordcloud = WordCloud(max_words=50, background_color="white", collocations=False).generate(text)

# display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title(f"Most popular 50 words for non-toxic comments")
plt.show()


# In[ ]:




