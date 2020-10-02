#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from fastai.text import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# # Loading and Visualization of distribution of the Data

# In[ ]:


df = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')


# In[ ]:


df.head()


# In[ ]:


# Distribution of the respective 3 sentiments.
sns.set(style="darkgrid")
ax = sns.countplot(x="airline_sentiment", data=df)


# In[ ]:


# Distribution of the negative sentiment.
ax = sns.countplot(x='negativereason',data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)


# # **Preprocessing the Twitter data**
# The decisions in which to preprocess were based on intuitions (for example, our assumption is that removing a hashtag would reduce noise and improve signal), as well as research paper results (removing numbers, replacing urls, and replacing user mentions were shown to improve accuracy on Twitter Sentiment Analysis).
# This is based on the code from "A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis" by Dimotrios Effrosynidis.

# In[ ]:


import re

def removeUnicode(text):
  """ Removes unicode strings like "\u002c" and "x96" """
  text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
  text = re.sub(r'[^\x00-\x7f]',r'',text)
  return text
  
def replaceURL(text):
  """Replaces url address with "url" """
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
  text = re.sub(r'#([^\s]+)', r'\1', text)
  return text

def replaceAtUser(text):
  """ Replaces "@user" with "atUser" """
  # text = re.sub('@[^\s]+','atUser',text)
  text = re.sub('@[^\s]+','',text)
  return text

def removeHashtagInFrontOfWord(text):
  """ Removes hastag in front of a word """
  text = re.sub(r'#([^\s]+)', r'\1', text)
  return text

def removeNumbers(text):
  """ Removes integers """
  text = ''.join([i for i in text if not i.isdigit()])         
  return text

def removeEmoticons(text):
  """ Removes emoticons from text """
  text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
  return text


""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
  patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
  for (pattern, repl) in patterns:
      (text, count) = re.subn(pattern, repl, text)
  return text


# In[ ]:


def preprocessTwitterData(df):
  """Function to apply text preprocessing functions to a dataframe"""
  
  # remove unicode
  df['text'] = df['text'].apply(removeUnicode)
  
  # replace url
  df['text'] = df['text'].apply(replaceURL)
  
  # replace '@' signs
  df['text'] = df['text'].apply(replaceAtUser)
  
  
  # replace hastags
  df['text'] = df['text'].apply(removeHashtagInFrontOfWord)
  
  # remove numbers in the tweets
  df['text'] = df['text'].apply(removeNumbers)
  
  # remove the emoticons
  df['text'] = df['text'].apply(removeEmoticons)
  
  # replace contractions
  df['text'] = df['text'].apply(replaceContraction)
  
# Call the function and preprocess the data  
preprocessTwitterData(df)

# Since we don't need the rest of the columns in the data, subindex the relevant columns and make this the new dataframe
df = df[['text','airline_sentiment']]

# Split the dataset into a train and test set.
# Using a validation set is built into the fastai API, so we don't need to do this split ourselves

# use an 80-20 split for the train and test sets
df_train, df_test = train_test_split(df,test_size=0.1,random_state=20)

# Convert the cleaned training and testing data into their own CSV files which we can import later to perform modeling on them

df_train.to_csv('twitter_data_cleaned_train.csv')
df_test.to_csv('twitter_data_cleaned_test.csv')


# # Training Phase
# First we will have a language model and then add a classification model for enhanced results.

# In[ ]:


# Create a 'TextLMDataBunch' from a csv file.
# We specify 'valid=0.1' to signify that when we want to actually put this into our language model, we'll be setting off 10% of it for a validation set
data_batch = TextLMDataBunch.from_csv(path='',csv_name='twitter_data_cleaned_train.csv',valid_pct=0.1)

# run this to see how the batch looks like
data_batch.show_batch()


# Initially we gonna set a language model ,train it and save the encodings i.e weights

# In[ ]:


# pass in our 'data_lm' objet to specify our Twitter data
# pass in AWD_LSTM to specify that we're using this particular language model
tweet_model = language_model_learner(data_batch, AWD_LSTM, drop_mult=0.3)


# In[ ]:


tweet_model.model


# In[ ]:


# fastai learning-rate finding
# implemented using fastai callbacks
#learning_rate_finder is used to find the optimal learning rate .
tweet_model.lr_find()


# In[ ]:


# plot the graph we were talking about earlier
tweet_model.recorder.plot()


# From the graph, we shall decide to take 1e-1 since the learning rate as the loss becomes minimum after this.
# 

# In[ ]:


# We set cycle_len to 1 because we only train with one epoch 'moms' refers to a tuple with the form (max_momentum,min_momentum)
tweet_model.fit_one_cycle(cyc_len=1,max_lr=1e-1,moms=(0.85,0.75))


# In[ ]:


# unfreeze the LSTM layers of the model
tweet_model.unfreeze()


# In[ ]:


#Now let's train the model
tweet_model.fit_one_cycle(cyc_len=5, max_lr=slice(1e-1/(2.6**4),1e-1), moms=(0.85, 0.75))


# In[ ]:


# save the encoder 
tweet_model.save_encoder('encoder')


# Now let's change the model according to our application i.e classification.

# In[ ]:


# create 'TextClasDataBunch'
# pass in vocab to ensure the vocab is the
# same one that was modified in the fine-tuned LM
data_class = TextClasDataBunch.from_csv(path='',csv_name='twitter_data_cleaned_train.csv',
                              vocab=data_batch.train_ds.vocab,bs=32,text_cols='text',label_cols='airline_sentiment')

# show what our batch looks like
data_class.show_batch()


# In[ ]:


# create new learner object with the 'text_classifier_learner' object.
# The concept behind this learner is the same as the 'language_model_learner'.
# It can similarly take in callbacks that allow us to train with special optimization methods. We use a slightly bigger dropout this time
tweet_model = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)

# load the fine-tuned encoder onto the learner
tweet_model.load_encoder('encoder')

# look at the model
tweet_model.model


# # Gradual Unfreezing
# i.e unfreezing and training layers one by one from the top i.e last layer to prevent forgetting of features.

# In[ ]:


# find the optimal learning rate, just like we did before
tweet_model.lr_find()

# plot it
tweet_model.recorder.plot()


# In[ ]:


# like we did before, we choose a learning rate before
# the minimum of the graph and use the 1cycle policy
tweet_model.fit_one_cycle(5,1e-1,moms=(0.8,0.7))


# In[ ]:


# unfreeze next layer
tweet_model.freeze_to(-2)

# train with next layer unfrozen, apply discriminative fine-tuning
tweet_model.fit_one_cycle(5,slice(1e-2/(2.6**4),1e-2))


# In[ ]:


# repeat the process
tweet_model.freeze_to(-3)
tweet_model.fit_one_cycle(5,slice(1e-2/(2.6**4),1e-2))


# In[ ]:


# now unfreeze everything
tweet_model.unfreeze()
tweet_model.fit_one_cycle(5,slice(1e-2/(2.6**4),1e-2))


# # Prediction Phase

# In[ ]:


# put test data in test df
df_test = pd.read_csv('twitter_data_cleaned_test.csv')
print(df_test[['text','airline_sentiment']])
df_test.head()


# In[ ]:


# add a column with the predictions on the test set

df_test['sentiment_pred'] = df_test['text'].apply(lambda row:str(tweet_model.predict(row)[0]))


# Let us now calculate the accuracy which is a popular evaluation metric in order to compare with many other approaches.

# In[ ]:


# print the accuracy against the test set
print("Accuracy: {}".format(accuracy_score(df_test['airline_sentiment'],df_test[
    'sentiment_pred'])))


# The result is quite convincing.

# Let's see how our model predicts!

# In[ ]:


import matplotlib.pyplot as plt
# Taken from the scikit-learn documentation
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


# plot the confusion matrix for the test set
plot_confusion_matrix(df_test['airline_sentiment'],df_test['sentiment_pred'],
                      classes=['negative','neutral','positive'])
plt.show()


# Woohooo! That's awesome!

# # References
# - This Notebook is inspired primarily from the [github repository](https://github.com/rajs96/ULMFiT-Twitter-US-Airline-Sentiment/blob/master/ulmfit_results.ipynb).
# - ["Universal Language Model Fine-tuning for Text Classification"](https://arxiv.org/abs/1801.06146) 
# - ["A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis"](https://link.springer.com/chapter/10.1007/978-3-319-67008-9_31)
# - ["A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay"](https://arxiv.org/abs/1803.09820)
