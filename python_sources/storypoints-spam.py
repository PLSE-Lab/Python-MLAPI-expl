#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir('../input/storypointsfull'))


# In[ ]:


# reading the data
dataset_file = 'appceleratorstudio'
data = pd.read_csv("../input/storypointsfull/{}.csv".format(dataset_file))

# getting the shape
data.shape


# In[ ]:


data.head()


# In[ ]:


# deleting the unnamed columns 

data = data.drop(['issuekey'], axis = 1)

# getting the shape of new data
data.shape


# In[ ]:


data.describe()


# In[ ]:


# adding a column to represent the length of the tweet

data['lenTitle'] = data['title'].str.len()
data['lenDescription'] = data['description'].str.len()

data.head(10)


# In[ ]:


# relation between title length and storypoints

plt.rcParams['figure.figsize'] = (10, 7)
sns.boxenplot(x = data['storypoint'], y = data['lenTitle'])
plt.title('Relation between Story Points and Title Length', fontsize = 20)
plt.show()


# In[ ]:


# relation between description length and storypoints

plt.rcParams['figure.figsize'] = (10, 7)
sns.boxenplot(x = data['storypoint'], y = data['lenDescription'])
plt.title('Relation between Story Points and Description Length', fontsize = 20)
plt.show()


# In[ ]:


# checking the most common words in the whole dataset - title

from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['title']))

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Most Common words in the dataset', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)


# In[ ]:


# checking the most common words in the whole dataset - description

from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['description']))

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Most Common words in the dataset', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)


# In[ ]:


df = pd.DataFrame(dict(x=np.random.poisson(4, 500)))
ax = sns.barplot(x="storypoint", y="storypoint", data=data, estimator=lambda x: len(x) / len(data) * 100)
ax.set(ylabel="Percent")


# In[ ]:


#Title words frequency
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()
words = cv.fit_transform(data.title)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')
plt.title("Most Frequently Occuring Words - Title - Top 30")


# In[ ]:


data.head()


# In[ ]:


#Description words frequency
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()
words = cv.fit_transform(data['description'].values.astype('U'))

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')
plt.title("Most Frequently Occuring Words - Description - Top 30")


# In[ ]:


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpusTitle = []

for i in range(0, 2919):  
  
  review = re.sub('[^a-zA-Z]', ' ', data['title'][i])
  
  review = review.lower()
  review = review.split()  
  ps = PorterStemmer()
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  
  # joining them back with space
  review = ' '.join(review)
  
  corpusTitle.append(review)

print(corpusTitle)


# In[ ]:


corpusDesc = []

for i in range(0, 2919):  
  
  review = re.sub('[^a-zA-Z]', ' ', data['description'].values.astype('U')[i])
  
  review = review.lower()
  review = review.split()  
  ps = PorterStemmer()
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  
  # joining them back with space
  review = ' '.join(review)
  
  corpusDesc.append(review)

print(corpusDesc)


# In[ ]:


data.head(10)


# In[ ]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_title = cv.fit_transform(corpusTitle).toarray()
y_title = data.iloc[:, 2]
print(x_title)
print(x_title.shape)
print(y_title.shape)


# In[ ]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_desc = cv.fit_transform(corpusDesc).toarray()
y_desc = data.iloc[:, 2]
print(y_desc)
print(x_desc)
print(x_desc.shape)
print(y_desc.shape)


# In[ ]:


#Title
# splitting the training data into train and valid sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_title, y_title, test_size = 0.25, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix
#cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot = True)


# In[ ]:


#Description
# splitting the training data into train and valid sets

from sklearn.model_selection import train_test_split

x_train_desc, x_test_desc, y_train_desc, y_test_desc = train_test_split(x_desc, y_desc, test_size = 0.25, random_state = 42)

print(x_train_desc.shape)
print(x_test_desc.shape)
print(y_train_desc.shape)
print(y_test_desc.shape)


# In[ ]:


# standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_desc = sc.fit_transform(x_train_desc)
x_test_desc = sc.transform(x_test_desc)


# In[ ]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(x_train_desc, y_train_desc)

y_pred = model.predict(x_test_desc)

print("Training Accuracy :", model.score(x_train_desc, y_train_desc))
print("Testing Accuracy :", model.score(x_test_desc, y_test_desc))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix
#cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot = True)

