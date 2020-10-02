#!/usr/bin/env python
# coding: utf-8

# # <center> Text Classification with womens-ecommerce-clothing-reviews data set</center>

# I am very glad to welcome you all to my notebook, In this notebook I'm going to work on a text classification problem. The problem is described as **'Given Review about clothing on E-commerce predict whether the custoomer will recommed it to her friends or not'...**<br/><br/>
# I need to mention one thing I'm new to Machine Learning with text and I have used very easy and effective approaches to solve this problem. I have done some data analysis, data visualizations then finally build both machine learning and deep learning models.
# It's time to jump on the process but before that I will mention the Workflow:<br/><br/>
# **1)Loading the data<br/>
# 2)Handling Missing Values<br/>
# 3)Cleaning the data<br/>
# 4)Data Analysis and Visualization<br/>
# 5)Handling MultiColinearity<br/>
# 6)Tokenisation+stemming+corpus creation<br/>
# 7)Buidling ML model using Bag of words<br/>
# 8)Building ML model using Tf-Idf Vectoriztion<br/>
# 9)Deep Learning Model with Embeddings<br/>
# 10)Checking the model with new example**

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading the Data

# In[ ]:


data = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv',index_col =[0])


# In[ ]:


data.head(2)


# In[ ]:


data.shape


# # Checking For Missing Values and Handling it

# In[ ]:


data.isnull().sum()/len(data)*100


# In[ ]:


data.info()


# In[ ]:


data.drop(labels =['Clothing ID','Title'],axis = 1,inplace = True) #Dropping unwanted columns


# In[ ]:


data[data['Review Text'].isnull()]


# In[ ]:


data = data[~data['Review Text'].isnull()]  #Dropping columns which don't have any review


# In[ ]:


data.shape


# # Data Analysis and Visualization

# In[ ]:


import plotly.express as px


# In[ ]:


px.histogram(data, x = 'Age')


# In[ ]:


px.histogram(data, x = data['Rating'])


# In[ ]:


px.histogram(data, x = data['Class Name'])


# In[ ]:


px.scatter(data, x="Age", y="Positive Feedback Count", facet_row="Recommended IND", facet_col="Rating",trendline="ols",category_orders={"Rating": [1,2,3,4,5],'Recommended IND':[0,1]})


# In[ ]:


px.violin(data, x="Age", y="Department Name", orientation="h", color="Recommended IND")


# In[ ]:


px.box(data, x="Age", y="Division Name", orientation="h",color = 'Recommended IND')


# # Cleaning the Text Data

# In[ ]:


err1 = data['Review Text'].str.extractall("(&amp)")
err2 = data['Review Text'].str.extractall("(\xa0)")


# In[ ]:


print('with &amp',len(err1[~err1.isna()]))
print('with (\xa0)',len(err2[~err2.isna()]))


# In[ ]:


data['Review Text'] = data['Review Text'].str.replace('(&amp)','')
data['Review Text'] = data['Review Text'].str.replace('(\xa0)','')


# In[ ]:


err1 = data['Review Text'].str.extractall("(&amp)")
print('with &amp',len(err1[~err1.isna()]))
err2 = data['Review Text'].str.extractall("(\xa0)")
print('with (\xa0)',len(err2[~err2.isna()]))


# In[ ]:


get_ipython().system('pip install TextBlob')
from textblob import *


# In[ ]:


data['polarity'] = data['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity)


# In[ ]:


data['polarity']


# In[ ]:


px.histogram(data, x = 'polarity')


# In[ ]:


px.box(data, y="polarity", x="Department Name", orientation="v",color = 'Recommended IND')


# In[ ]:


data['review_len'] = data['Review Text'].astype(str).apply(len)


# In[ ]:


px.histogram(data, x = 'review_len')


# In[ ]:


data['token_count'] = data['Review Text'].apply(lambda x: len(str(x).split()))


# In[ ]:


px.histogram(data, x = 'token_count')


# # Reviews with Positive Polarity

# In[ ]:


sam = data.loc[data.polarity == 1,['Review Text']].sample(3).values


# In[ ]:


for i in sam:
    print(i[0])


# # Reviews with Neutral Polarity

# In[ ]:


sam = data.loc[data.polarity == 0.5,['Review Text']].sample(3).values


# In[ ]:


for i in sam:
    print(i[0])


# # Reviews with Negative Polarity

# In[ ]:


sam = data.loc[data.polarity < 0,['Review Text']].sample(3).values


# In[ ]:


for i in sam:
    print(i[0])


# In[ ]:


negative = (len(data.loc[data.polarity <0,['Review Text']].values)/len(data))*100
positive = (len(data.loc[data.polarity >0.5,['Review Text']].values)/len(data))*100
neutral  = len(data.loc[data.polarity >0 ,['Review Text']].values) - len(data.loc[data.polarity >0.5 ,['Review Text']].values)
neutral = neutral/len(data)*100


# # Pie-Chart about Polarity

# In[ ]:


from matplotlib import pyplot as plt 
plt.figure(figsize =(10, 7)) 
plt.pie([positive,negative,neutral], labels = ['Positive','Negative','Neutral']) 


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


def top_n_ngram(corpus,n = None,ngram = 1):
    vec = CountVectorizer(stop_words = 'english',ngram_range=(ngram,ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus) #Have the count of  all the words for each review
    sum_words = bag_of_words.sum(axis =0) #Calculates the count of all the word in the whole review
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,key = lambda x:x[1],reverse = True)
    return words_freq[:n]


# # Visualizing Top 20 Unigrams

# In[ ]:


common_words = top_n_ngram(data['Review Text'], 20,1)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 unigrams in review after removing stop words')


# # Visualizing Top 20 Bigrams

# In[ ]:


common_words = top_n_ngram(data['Review Text'], 20,2)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 bigrams in review after removing stop words')


# # Visualizing Top 20 Trigrams

# In[ ]:


common_words = top_n_ngram(data['Review Text'], 20,3)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 trigrams in review after removing stop words')


# # Visualizing Top 20 Part-of-Speech

# In[ ]:


blob= TextBlob(str(data['Review Text']))
pos = pd.DataFrame(blob.tags,columns =['word','pos'])
pos1 = pos.pos.value_counts()[:20]
plt.figure(figsize = (10,5))
pos1.plot(kind='bar',title ='Top 20 Part-of-speech taggings')


# In[ ]:


y = data['Recommended IND']


# In[ ]:


X = data.drop(columns = 'Recommended IND')


# # Correlation HeatMap

# In[ ]:


import seaborn as sns
sns.heatmap(X.corr(),annot =True)


# # Handling Multi-Colinearity

# In[ ]:


set1 =set()
cor = X.corr()
for i in cor.columns:
    for j in cor.columns:
        if cor[i][j]>0.8 and i!=j:
            set1.add(i)
print(set1)


# In[ ]:


X = X.drop(labels = ['token_count'],axis = 1)


# In[ ]:


X.corr()


# In[ ]:


class1 =[]
for i in X.polarity:
    if float(i)>=0.0:
        class1.append(1)
    elif float(i)<0.0:
        class1.append(0)
X['sentiment'] = class1


# # Statistical Description of Data

# In[ ]:


X.groupby(X['sentiment']).describe().T


# # Model Building

# In[ ]:


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


corpus =[]


# In[ ]:


X.index = np.arange(len(X))


# # RE + Tokenizing + Stemming + Corpus Creation

# In[ ]:


for i in range(len(X)):
    review = re.sub('[^a-zA-z]',' ',X['Review Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review =[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)


# # Bag of Words Technique

# ![image.png](attachment:image.png)

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer as CV
cv  = CV(max_features = 3000,ngram_range=(1,1))
X_cv = cv.fit_transform(corpus).toarray()
y = y.values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size = 0.20, random_state = 0)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)


# In[ ]:


acc


# # Term Frequency- Inverse Document Frequency Technique

# ![image.png](attachment:image.png)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer as TV
tv  = TV(ngram_range =(1,1),max_features = 3000)
X_tv = tv.fit_transform(corpus).toarray()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_tv, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)


# In[ ]:


acc


# # Deep Learning Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer(num_words = 3000)
tokenizer.fit_on_texts(corpus)


# In[ ]:


sequences = tokenizer.texts_to_sequences(corpus)
padded = pad_sequences(sequences, padding='post')


# In[ ]:


word_index = tokenizer.word_index
count = 0
for i,j in word_index.items():
    if count == 11:
        break
    print(i,j)
    count = count+1


# In[ ]:


embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(3000, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


# In[ ]:


num_epochs = 10

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.fit(padded,y,epochs= num_epochs)


# CHECKING NEW EXAMPLE

# In[ ]:


sample_string = "I Will tell my friends for sure"
sample = tokenizer.texts_to_sequences(sample_string)
padded_sample = pad_sequences(sample, padding='post')


# In[ ]:


padded_sample.T


# In[ ]:


model.predict(padded_sample.T)


# **That's all for this notebook....See you soon!!!**<br/>
# *If you like my work upvote it!!!!!*
