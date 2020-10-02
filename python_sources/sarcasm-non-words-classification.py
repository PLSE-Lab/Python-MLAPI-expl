#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pylab as pl


# In[ ]:


#Disabling warnings
warnings.simplefilter("ignore")


# In[ ]:


#Importing data
data=pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True)


# In[ ]:


#Displaying data columns
data.columns


# In[ ]:


#Displaying shape & description
print(data.shape)
print(data.describe())


# In[ ]:


#Peek at data
data.head(10)


# In[ ]:


#Dropping column
data.drop(columns=['article_link'], inplace=True)


# In[ ]:


#Checking the labels distribution
labels = data['is_sarcastic']
print(labels.value_counts())


# In[ ]:


#Using regular expressions for filtering unwanted content 
processed = data['headline'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', ' ')
processed = processed.str.replace(r'\d+',' ')
processed = processed.str.lower()


# In[ ]:


#Removing stop words from reviews
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


# In[ ]:


#Creating Dictionary of words
all_words = []
for txt in processed:
    words = word_tokenize(txt)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)


# In[ ]:


#Printing the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))


# In[ ]:


#Using 5000 words as features
word_features = list(all_words.keys())[:5000]


# In[ ]:


#The find_features function will determine which of the 5000 featured words are contained in the headings
def find_features(headings):
    words = word_tokenize(headings)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# In[ ]:


#Unifying headings with their respective labels
headings = list(zip(processed, data['is_sarcastic']))
headings[0]


# In[ ]:


#Defining a seed for reproducibility and shuffling
seed = 1
np.random.seed = seed
np.random.shuffle(headings)


# In[ ]:


#Calling find_features function for each review
featuresets = [(find_features(text), label) for (text, label) in headings]


# In[ ]:


#Splitting the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
print("Training set:", len(training))
print("Testing set:", len(testing))


# In[ ]:


#Defining Naive Bayes model for training
model = MultinomialNB()


# In[ ]:


#Training the model on the training data and calculating accuracy
nltk_model = SklearnClassifier(model)
nltk_model.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Naive Bayes Accuracy: {}".format(accuracy))


# In[ ]:


#Listing the predicted labels for testing dataset
txt_features, labels = list(zip(*testing))
prediction = nltk_model.classify_many(txt_features)


# In[ ]:


#Printing classification report & Confusion matrix
print(classification_report(labels, prediction))
df = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['Sarcasm','Not Sarcasm']],
    columns = [['predicted', 'predicted'], ['Sarcasm','Not Sarcasm']])
print(df)


# In[ ]:


#Creating Bag of words - Sarcasm/Non for wordcloud
sarc=[];
no_sarc=[];
unifyy = list(zip(prediction, txt_features, labels))
for p, t, l in unifyy:
    for key, value in t.items():
        if value==True and l==p==0:
            no_sarc.append(key)
            break
        elif value==True and l==p==1:
            sarc.append(key)
            break
print("Sarcasm Words:", sarc)
print("Not Sarcasm Words:", no_sarc)


# **Generating wordcloud for Sarcasm/Non**

# In[ ]:


pl.figure(figsize =(10,10))
sarcastic = str(sarc)
wordCloud = WordCloud(background_color="white").generate(sarcastic)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Sarcasm - Words', fontsize=15)
plt.show()


# In[ ]:


pl.figure(figsize =(10,10))
not_sarcastic = str(no_sarc)
wordCloud = WordCloud(background_color="white").generate(not_sarcastic)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Not Sarcasm - Words', fontsize=15)
plt.show()


# 
