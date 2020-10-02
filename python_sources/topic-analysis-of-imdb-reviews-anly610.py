#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on Movie Reviews

# In this notebook we are going to analyze reviews or the most prominent topics.
# The reviews have positive and negative topics and we will try to identify which reviews are positive and negative.

# In[ ]:


import pandas as pd
import numpy as np
import re
import os
from IPython.display import HTML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn.decomposition import PCA

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words
from nltk.corpus import wordnet 
allEnglishWords = words.words() + [w for w in wordnet.words()]
allEnglishWords = np.unique([x.lower() for x in allEnglishWords])

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


# ## There are 5 parts in this notebook
# ## Part 1 - Importing the dataset
# ## Part 2 - Cleaning and preprocessing the dataset
# ## Part 3 - Creating different wordclouds
# ## Part 4 - Topic detection
# ## Part 5 - Sentiment Analysis

# # Part 1 - Importing the dataset
# First, we need to import the data.
# We create a new pandas dataframe with all the reviews as seperate rows.

# In[ ]:


path = "/kaggle/input/aclimdb/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]


# In[ ]:


positiveReviews, negativeReviews, testReviews = [], [], []
for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())
for tfile in testFiles:
    with open(path+"test/"+tfile, encoding="latin1") as f:
        testReviews.append(f.read())


# In[ ]:


reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),
    pd.DataFrame({"review":testReviews, "label":-1, "file":testFiles})
], ignore_index=True).sample(frac=1, random_state=1)
reviews.head()


# In[ ]:


reviews.tail()


# With everything centralized in 1 dataframe, we now perform train, validation and test set splits.

# In[ ]:


reviews = reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)
valid = reviews[reviews.label!=-1].drop(train.index)
test = reviews[reviews.label==-1]


# In[ ]:


print(train.shape)
print(valid.shape)
print(test.shape)


# In[ ]:


HTML(train.review.iloc[0])


# # Part 2 - Data Preprocessing
# The next step is data preprocessing. The following class behaves like your typical SKLearn vectorizer.
# 
# It can perform the following operations.
# * Discard non alpha-numeric characters
# * Set everything to lower case
# * Stems all words using PorterStemmer, and change the stems back to the most occurring existent word.
# * Discard non-Egnlish words (not by default).

# In[ ]:


class Preprocessor(object):
    ''' Preprocess data for NLP tasks. '''

    def __init__(self, alpha=True, lower=True, stemmer=True, english=False):
        self.alpha = alpha
        self.lower = lower
        self.stemmer = stemmer
        self.english = english
        
        self.uniqueWords = None
        self.uniqueStems = None
        
    def fit(self, texts):
        texts = self._doAlways(texts)

        allwords = pd.DataFrame({"word": np.concatenate(texts.apply(lambda x: x.split()).values)})
        self.uniqueWords = allwords.groupby(["word"]).size().rename("count").reset_index()
        self.uniqueWords = self.uniqueWords[self.uniqueWords["count"]>1]
        if self.stemmer:
            self.uniqueWords["stem"] = self.uniqueWords.word.apply(lambda x: PorterStemmer().stem(x)).values
            self.uniqueWords.sort_values(["stem", "count"], inplace=True, ascending=False)
            self.uniqueStems = self.uniqueWords.groupby("stem").first()
        
        #if self.english: self.words["english"] = np.in1d(self.words["mode"], allEnglishWords)
        print("Fitted.")
            
    def transform(self, texts):
        texts = self._doAlways(texts)
        if self.stemmer:
            allwords = np.concatenate(texts.apply(lambda x: x.split()).values)
            uniqueWords = pd.DataFrame(index=np.unique(allwords))
            uniqueWords["stem"] = pd.Series(uniqueWords.index).apply(lambda x: PorterStemmer().stem(x)).values
            uniqueWords["mode"] = uniqueWords.stem.apply(lambda x: self.uniqueStems.loc[x, "word"] if x in self.uniqueStems.index else "")
            texts = texts.apply(lambda x: " ".join([uniqueWords.loc[y, "mode"] for y in x.split()]))
        #if self.english: texts = self.words.apply(lambda x: " ".join([y for y in x.split() if self.words.loc[y,"english"]]))
        print("Transformed.")
        return(texts)

    def fit_transform(self, texts):
        texts = self._doAlways(texts)
        self.fit(texts)
        texts = self.transform(texts)
        return(texts)
    
    def _doAlways(self, texts):
        # Remove parts between <>'s
        texts = texts.apply(lambda x: re.sub('<.*?>', ' ', x))
        # Keep letters and digits only.
        if self.alpha: texts = texts.apply(lambda x: re.sub('[^a-zA-Z0-9 ]+', ' ', x))
        # Set everything to lower case
        if self.lower: texts = texts.apply(lambda x: x.lower())
        return texts  


# In[ ]:


train.head()


# In[ ]:


train.shape
train.info()


# ## Cleaning the dataset of all non english words

# In[ ]:


preprocess = Preprocessor(alpha=True, lower=True, stemmer=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'trainX = preprocess.fit_transform(train.review)\nvalidX = preprocess.transform(valid.review)')


# In[ ]:


trainX.head()


# In[ ]:


trainX.shape


# ## Appending other columns to the cleaned dataset

# In[ ]:


trainX2 = pd.concat([trainX, train[['label', 'file']]], axis = 1, sort = False) 


# In[ ]:


trainX2.head()


# In[ ]:


print(preprocess.uniqueWords.shape)
preprocess.uniqueWords[preprocess.uniqueWords.word.str.contains("disappoint")]


# In[ ]:


print(preprocess.uniqueStems.shape)
preprocess.uniqueStems[preprocess.uniqueStems.word.str.contains("disappoint")]


# # Part 3 - Creating different wordclouds
# Creating a word cloud for all words

# In[ ]:


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


text = " ".join(review for review in trainX2['review'])
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc", "br", 'movie', 'film', 'one', 'make', 'even'])


# In[ ]:


wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)


# In[ ]:


import matplotlib.pyplot as plt
#% matplotlib inline


# In[ ]:


plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Word Cloud for positive reviews

# In[ ]:


positive = trainX2[trainX2['label'] == 1]
positive.head()


# In[ ]:


positive.shape[0]
print('There are ' + str(positive.shape[0]) + ' positive reviews')


# In[ ]:


text = " ".join(review for review in positive['review'])
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


stopwords.update(["Nan","Negative","etc", "br", 'film', 'movie', 'one'])
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Word Cloud for negative reviews

# In[ ]:


negative = trainX2[trainX2['label'] == 0]
negative.head()


# In[ ]:


print('There are ' + str(negative.shape[0]) + ' negative reviews')


# In[ ]:


text = " ".join(review for review in negative['review'])
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc", "br", 'film', 'movie', 'one', 'even'])
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Most frequently used words in the trainX set

# In[ ]:


all_words = trainX2['review'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = all_words.index.values[2:50],
            y = all_words.values[2:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 (Uncleaned) Word frequencies in the training dataset'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# # Part 4 - Topic detection

# Selecting 90 words from the first review.

# In[ ]:


first_text = negative.review.values[1]
print(first_text)
print("="*90)
print(first_text.split(" "))


# In[ ]:


first_text_list = nltk.word_tokenize(first_text)
print(first_text_list)


# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)


# In[ ]:


first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stopwords]
print(first_text_list_cleaned)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(first_text_list), len(first_text_list_cleaned)))


# In[ ]:


stemmer = nltk.stem.PorterStemmer()


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)


# In[ ]:


lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# ## Topic modelling and word cloud for the negative reviews

# In[ ]:


# Storing the entire training text in a list
text = list(negative.review.values)
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)


# In[ ]:


lda.fit(tf)


# In[ ]:


n_top_words = 40
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# In[ ]:


first_topic = lda.components_[0]
second_topic = lda.components_[1]
third_topic = lda.components_[2]
fourth_topic = lda.components_[3]
fifth_topic = lda.components_[4]
seventh_topic = lda.components_[6]


# In[ ]:


first_topic.shape


# In[ ]:


first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]
fourth_topic_words = [tf_feature_names[i] for i in fourth_topic.argsort()[:-50 - 1 :-1]]
fifth_topic_words = [tf_feature_names[i] for i in fifth_topic.argsort()[:-50 - 1 :-1]]
seventh_topic_words = [tf_feature_names[i] for i in seventh_topic.argsort()[:-50 - 1 :-1]]


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc", "br", 'film', 'movie', 'one', 'johnny', 'cage', 'good'])
firstcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(" ".join(seventh_topic_words))
plt.figure(figsize=(10,8))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()


# ## Topic modelling and word cloud for the positive reviews

# In[ ]:


# Storing the entire training text in a list
text = list(positive.review.values)
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)


# In[ ]:


lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)


# In[ ]:


lda.fit(tf)


# In[ ]:


n_top_words = 40
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# In[ ]:


first_topic = lda.components_[0]
second_topic = lda.components_[1]
third_topic = lda.components_[2]
fourth_topic = lda.components_[3]
fifth_topic = lda.components_[4]
tenth_topic = lda.components_[9]


# In[ ]:


first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]
fourth_topic_words = [tf_feature_names[i] for i in fourth_topic.argsort()[:-50 - 1 :-1]]
fifth_topic_words = [tf_feature_names[i] for i in fifth_topic.argsort()[:-50 - 1 :-1]]
tenth_topic_words = [tf_feature_names[i] for i in tenth_topic.argsort()[:-50 - 1 :-1]]


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc", "br", 'film', 'movie', 'one'])
firstcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(" ".join(tenth_topic_words))
plt.figure(figsize=(10,8))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()


# # Part 5 - Sentiment Analysis
# In the next part we will analyze the sentiment of the reviews.

# In[ ]:


tfidf = TfidfVectorizer(min_df=2, max_features=10000, stop_words=stopwords) #, ngram_range=(1,3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'trainX = tfidf.fit_transform(trainX).toarray()\nvalidX = tfidf.transform(validX).toarray()')


# In[ ]:


print(trainX.shape)
print(validX.shape)


# In[ ]:


trainY = train.label
validY = valid.label


# In[ ]:


print(trainX.shape, trainY.shape)
print(validX.shape, validY.shape)


# ## Part 5a - Feature Selection
# Next, we take the 10k dimensional tfidf's as input, and keep the 2000 dimensions that correlate the most with our sentiment target. The corresponding words - see below - make sense.

# In[ ]:


from scipy.stats.stats import pearsonr


# In[ ]:


getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])
correlations = getCorrelation(np.arange(trainX.shape[1]))
print(correlations)


# In[ ]:


allIndeces = np.argsort(-correlations)
bestIndeces = allIndeces[np.concatenate([np.arange(1000), np.arange(-1000, 0)])]


# In[ ]:


vocabulary = np.array(tfidf.get_feature_names())
print(vocabulary[bestIndeces][:15])
print(vocabulary[bestIndeces][-15:])


# In[ ]:


trainX = trainX[:,bestIndeces]
validX = validX[:,bestIndeces]


# In[ ]:


print(trainX.shape, trainY.shape)
print(validX.shape, validY.shape)


# ## Part 5b - Defining Model Architecture
# We choose a very simple dense network with 6 layers, performing binary classification.

# In[ ]:


DROPOUT = 0.5
ACTIVATION = "tanh"

model = Sequential([    
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/4), activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(100, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(20, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(5, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid'),
])


# In[ ]:


model.compile(optimizer=optimizers.Adam(0.00005), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# ## Part 5c - Model Training
# Training the model using the TrainX dataset.

# In[ ]:


EPOCHS = 30
BATCHSIZE = 1500


# In[ ]:


model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validX, validY))


# In[ ]:


x = np.arange(EPOCHS)
history = model.history.history

data = [
    go.Scatter(x=x, y=history["acc"], name="Train Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["val_acc"], name="Valid Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["loss"], name="Train Loss", marker=dict(size=5)),
    go.Scatter(x=x, y=history["val_loss"], name="Valid Loss", marker=dict(size=5))
]
layout = go.Layout(
    title="Model Training Evolution", font=dict(family='Palatino'), xaxis=dict(title='Epoch', dtick=1),
    yaxis1=dict(title="Loss", domain=[0, 0.45]), yaxis2=dict(title="Accuracy", domain=[0.55, 1]),
)
py.iplot(go.Figure(data=data, layout=layout), show_link=False)


# ## Part 5d -  Model Evaluation
# Adding the probabilities and predictions columns to the original train and validation dataframes.

# In[ ]:


train["probability"] = model.predict(trainX)
train["prediction"] = train.probability-0.5>0
train["truth"] = train.label==1
train.tail()


# In[ ]:


print(model.evaluate(trainX, trainY))
print((train.truth==train.prediction).mean())


# In[ ]:


valid["probability"] = model.predict(validX)
valid["prediction"] = valid.probability-0.5>0
valid["truth"] = valid.label==1
valid.tail()


# In[ ]:


print(model.evaluate(validX, validY))
print((valid.truth==valid.prediction).mean())


# ### Creating confusion matrices to check the accuracy of our model on the validation set

# In[ ]:


trainCross = train.groupby(["prediction", "truth"]).size().unstack()
trainCross


# In[ ]:


validCross = valid.groupby(["prediction", "truth"]).size().unstack()
validCross


# In[ ]:


truepositives = valid[(valid.truth==True)&(valid.truth==valid.prediction)]
print(len(truepositives), "true positives.")
truepositives.sort_values("probability", ascending=False).head(3)


# In[ ]:


truepositives.review.iloc[1]


# In[ ]:


truenegatives = valid[(valid.truth==False)&(valid.truth==valid.prediction)]
print(len(truenegatives), "true negatives.")
truenegatives.sort_values("probability", ascending=True).head(3)


# In[ ]:


truenegatives.review.iloc[1]


# In[ ]:


falsepositives = valid[(valid.truth==True)&(valid.truth!=valid.prediction)]
print(len(falsepositives), "false positives.")
falsepositives.sort_values("probability", ascending=True).head(3)


# In[ ]:


falsepositives.review.iloc[1]


# In[ ]:


falsenegatives = valid[(valid.truth==False)&(valid.truth!=valid.prediction)]
print(len(falsenegatives), "false negatives.")
falsenegatives.sort_values("probability", ascending=False).head(3)


# In[ ]:


falsenegatives.review.iloc[1]


# This is the review that got predicted as positive most certainly - while being labeled as negative. However, we can easily recognize it as a poorly labeled sample.

# In[ ]:


HTML(valid.loc[22148].review)


# ## With this we conclude the Topic Modelling and Sentiment Analysis of the Movie Review dataset.
