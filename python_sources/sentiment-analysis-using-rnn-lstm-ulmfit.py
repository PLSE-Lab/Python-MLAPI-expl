#!/usr/bin/env python
# coding: utf-8

# # Twitter Airline Sentiment Analysis (ULMFiT)

# ## Introduction

# This notebook explores the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) and tries to predict tweet sentiment using a language model and RNN via Fast.ai's library for Universal Language Model Fine-tuning for Text Classification ([ULMFiT](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)).

# ## Setup

# Importing packages.

# In[ ]:


# Basic packages
import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt

# Modeling, selection, and evaluation
from fastai.text import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preparation

# First we read in the data and have a look at the columns we can use and explore. 

# In[ ]:


# Read file into dataframe
pd.set_option('display.max_colwidth', -1)
df = pd.read_csv('../input/Tweets.csv')
df = df.reindex(np.random.permutation(df.index))  
df.head()


# Next we look at the distribution of the main dependent variable: airline_sentiment, and its breakdown across airlines and tweet length. 

# In[ ]:


df['airline_sentiment'].value_counts().plot(kind='bar')


# In[ ]:


df['airline'].value_counts().plot(kind='bar')


# In[ ]:


df.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar', stacked=True)


# In[ ]:


df['tweet_length'] = df['text'].apply(len)
df.groupby(['tweet_length', 'airline_sentiment']).size().unstack().plot(kind='line', stacked=False)


# We see that there isn't a lot of correlation between the number of positive / neutral tweets and the tweet length, but for negative tweets the distribution is heavily skewed towards longer tweets. This is possibly because the angier the tweeter, the more they have to say. Next we'll see if the same observation can be made about the average and median sentiment confidence as well.

# In[ ]:


df[['tweet_length', 'airline_sentiment', 'airline_sentiment_confidence']].groupby(['tweet_length', 'airline_sentiment']).mean().unstack().plot(kind='line', stacked=False)


# In[ ]:


df[['tweet_length', 'airline_sentiment', 'airline_sentiment_confidence']].groupby(['tweet_length', 'airline_sentiment']).median().unstack().plot(kind='line', stacked=False)


# There doesn't seem to be a discernable relationship between the confidence and the tweet length. Next we randomize and split the data, then write to CSVs. 

# In[ ]:


test_percentage = 0.1
df.sort_index(inplace=True)
cutoff = int(test_percentage * df.shape[0])
df[['airline_sentiment', 'text']][:cutoff].to_csv('Tweets_filtered_test.csv', index=False, encoding='utf-8')
df[['airline_sentiment', 'text']][cutoff:].to_csv('Tweets_filtered_train.csv', index=False, encoding='utf-8')
df[['text']][cutoff:].to_csv('Tweets_text_only_train.csv', index=False, encoding='utf-8')


# ### Tokenization

# We read in the data and add new words to our dictionary, as well as create a representation of words using numbers. 

# In[ ]:


data = TextClasDataBunch.from_csv('.', 'Tweets_filtered_train.csv')
data.show_batch()


# The following changes have been made to the text for ease of modeling:
# - split on space and punctuation symbols
# - the "'s" are grouped together in one token
# - the contractions are separated like this: "did", "n't"
# - there are several special tokens (all those that begin by xx), to replace unknown tokens (see below)

# In[ ]:


data.vocab.itos[:10]


# Looking into the dataset we can now see the current representation post-processing, in both text form and numerical.

# In[ ]:


print(data.train_ds[0][0])
print(data.train_ds[1][0])
print(data.train_ds[2][0])


# In[ ]:


print(data.train_ds[0][0].data[:10])
print(data.train_ds[1][0].data[:10])
print(data.train_ds[2][0].data[:10])


# ## Language Model

# We'll be using a language model provided from the fastai library and applying the pre-calculated weights from wikitext-103. This will provide a word embedding scheme that aligns with the corpus of airline tweets and will hopefully capture sufficient "meaning" in each word. The language model here will not be optimal as tweets sometimes do not obey conventional spelling and grammar, especially hastags. 

# In[ ]:


bs = 24
seed = 333


# First we ingest the data once again but have a 10% holdout only since we'll be using it to calibrate the language model. 

# In[ ]:


data_lm = (TextList.from_csv('.', 'Tweets_text_only_train.csv')
            .random_split_by_pct(0.1, seed = seed)
           #We randomly split and keep 10% for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
data_lm.save('data_lm.pkl')


# In[ ]:


# data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()


# Now we define the language model and set the learning rates. 

# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=15)


# Next we fit the model for a few cycles by running 1 epoch and then unfreezing and running more epochs to fine tune.  

# In[ ]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('fit_head')
# learn.load('fit_head')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.save('fine_tuned')


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# The language model is a bit worse than I would have expected; this is likely due to the fact that tweets dont always follow proper English spelling and grammar, making it difficult for the model to correctly predict. 

# ## Classifier

# Next we'll use the encoder from the language model in our classifier, which has a similar LSTM architecture but will predict the sentiment instead of the next word in a tweet. The model architecture here presents some advantages over traditional bags-of-words 

# In[ ]:


data_clas = (TextList.from_csv('.', 'Tweets_filtered_train.csv', cols = 'text')               
             .random_split_by_pct(0.1, seed = seed)
             .label_from_df(cols=0)
             .databunch(bs=bs))
data_clas.save('data_clas.pkl')
data_clas.show_batch()


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# We train by gradually unfreezing layers and then running an epoch each time, in accordance with the suggestions in the ULMFiT paper.

# In[ ]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('first')
# learn.load('first)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn.save('second')
# learn.load('second')


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# ### Evaluate Performance on Test Set

# In[ ]:


test_df = pd.read_csv("Tweets_filtered_test.csv", encoding="utf-8")
test_df['airline_sentiment'].value_counts().plot(kind='bar')


# In[ ]:


test_df['pred_sentiment'] = test_df['text'].apply(lambda row: str(learn.predict(row)[0]))
print("Test Accuracy: ", accuracy_score(test_df['airline_sentiment'], test_df['pred_sentiment']))


# In[ ]:


test_df[:20]


# Plot confusion matrix to see where the areas of misclassification are. 

# In[ ]:


# Confusion matrix plotting adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


plot_confusion_matrix(test_df['airline_sentiment'], test_df['pred_sentiment'], classes=['negative', 'neutral', 'positive'], title='Airline sentiment confusion matrix')
# confusion_matrix(test_df['airline_sentiment'], test_df['pred_sentiment'], labels=['positive', 'neutral', 'negative'])
plt.show()


# From the confusion matrix, we can see that within True Negatives, the prediction accuracy is pretty high. The model has a very hard time classifying neutral tweets, and often misclassifies them as negative. For True Positives, the overall accuracy is pretty good but a surprisingly large number get classified as negative. Let's look at these in closer detail:

# In[ ]:


test_df.loc[(test_df['airline_sentiment'] == 'positive') & (test_df['pred_sentiment'] == 'negative')]


# It's not immediately clear what the common thread that led these tweets to be mislabelled but possible causes include missing signals from hastags, non-grammatical sentences, mis-spellings, etc. which generally contribute to the classification error. 
# 
# In the future, it would be good to spend more time on feature engineering and fitting the language model so that it's more suited to the vocabulary and style of language that is in tweets. From the language model and processing steps, we can see that the model accuracy is not very high and that, even as humans, it's hard to discern the sentiment from the post-processed / tokenized text. One possible way of addressing this issue is to use a much larger corpus of tweets when developing the language model. 
