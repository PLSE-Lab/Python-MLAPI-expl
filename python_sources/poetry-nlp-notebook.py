#!/usr/bin/env python
# coding: utf-8

# **Poetry Classification Notebook**

# I've came across this dataset as I was looking for renaissance paintings to use in GAN, and seeing there are no kernels on it, I thought I might just dive in. 
# There are five columns, the poetry itself, the type, author, age of it. First I'll do exploratory data analysis and preprocessing, then I'll classify the author and the age of the poetries using decision trees.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import gc
import re


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Dense, Bidirectional


# **Importing the dataset**

# In[ ]:


df_poetry=pd.read_csv("../input/poetry-analysis-with-machine-learning/all.csv", sep=",")
df_poetry.head()


# First I'll do exploratory data analysis, then classify the poetries in age, type and author. Let's see the list of authors, types and ages.

# In[ ]:


df_poetry.rename(columns={"poem name":"poem_name"}, inplace=True)


# In[ ]:


df_poetry.age.unique()


# There are three types of poetry.

# In[ ]:


df_poetry.type.unique()


# Let's see the list of authors.

# In[ ]:


df_poetry.author.unique()


# **Removing special characters from the content column, leaving the spaces for tokenization**

# In[ ]:


def remove_special_chars(text, remove_digits=True):
    text=re.sub('[^a-zA-Z.\d\s]', '',text)
    return text
df_poetry.content=df_poetry.content.apply(remove_special_chars)


# Importing the list of stopwords, I have gathered the below gist def remove_stopwords from another notebook.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_poetry.age=le.fit_transform(df_poetry.age)
df_poetry


# In[ ]:


df_poetry.drop(columns=["author", "poem_name","type"])


# In[ ]:


from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(num_words=1009)
tokenizer.fit_on_texts(df_poetry.content)
sequences=tokenizer.texts_to_sequences(df_poetry.content)
tokenized=tokenizer.texts_to_matrix(df_poetry.content)
word_index=tokenizer.word_index
print("Found %s unique tokens."%len(word_index))


# In[ ]:


tokenized


# In[ ]:


tokenized.shape


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df_poetry.content)
# 

# In[ ]:


X=tokenized
Y=df_poetry.age


# In[ ]:


tokenized.shape


# In[ ]:


df_poetry.age.shape


# In[ ]:


X_train, X_test, y_train, y_test =train_test_split(X,Y,test_size=0.2)


# In[ ]:


X_train=tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=300)
X_test=tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=300)


# In[ ]:


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='my_log_dir')


# In[ ]:


max_features=10
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model = tf.keras.Sequential([Embedding(input_dim=100, output_dim=128),
                            LSTM(128,activation='relu', dropout=0.05, return_sequences=True),
                            LSTM(128, activation="relu",dropout=0.05,recurrent_dropout=0.01, return_sequences=True),
                            LSTM(64, activation="relu",dropout=0.01,recurrent_dropout=0.01, return_sequences=True),
                            LSTM(32, activation="relu",dropout=0.01,recurrent_dropout=0.01),
                            Dense(2, activation="relu"),
                            Dense(1, activation="sigmoid")])
opt=tf.keras.optimizers.RMSprop()
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])


# In[ ]:


model.fit(X_train, y_train.values, epochs=100, batch_size=40, validation_split=0.1, callbacks=[callback, tensorboard])


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


df_poetry=pd.read_csv("../input/poetry-analysis-with-machine-learning/all.csv", sep=",")
df_poetry.head()


# **Most used 20 words**

# In[ ]:


import plotly.graph_objects as go
from plotly.offline import iplot
words = df_poetry['content'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = words.index.values[2:20],
            y = words.values[2:20],
            marker= dict(colorscale='RdBu',
                         color = words.values[2:40]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Most used words excluding stopwords'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')


# **Creating word cloud from most used 100 words**

# In[ ]:


import matplotlib.pyplot as plt
def word_cloud(content, title):
    wc = WordCloud(background_color='white', max_words=200,
                  stopwords=STOPWORDS, max_font_size=50)
    wc.generate(" ".join(content))
    plt.figure(figsize=(16, 13))
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap='Pastel2', random_state=42), alpha=0.98)
    plt.axis('off')


# In[ ]:


word_cloud(df_poetry.content, "Word Cloud")


# I expect strong correlation between label encoded features.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_poetry.type=le.fit_transform(df_poetry.type)
df_poetry.age=le.fit_transform(df_poetry.age)
df_poetry.author=le.fit_transform(df_poetry.author)


# In[ ]:


corr = df_poetry.corr()
corr


# Heat map between label encoded features.

# In[ ]:


sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# Categorical plot to explain distribution of type and authors of poetry through the ages. It'd be better if the ages were given in years instead of two categories.

# In[ ]:


sns.catplot(x="age", y="author",hue="type", data=df_poetry);


# First I'll separate the dataset for training and test, then I'll vectorize both sets with TFIDF and Count Vectorizer, and then apply decision tree for classification.

# In[ ]:


y=df_poetry['author']
x=df_poetry["content"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)
print(X_train)

Trying to predict the author of the poem from the content. Used Count Vectorizer and Decision Tree Classifier with entropy.
# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)


# In[ ]:


vectest.shape


# In[ ]:


y_train.shape


# In[ ]:


dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)


# In[ ]:


accuracy= accuracy_score(preddt,y_test)
print(accuracy)


# Trying to predict the age of the poem from the content. Used Count Vectorizer and Decision Tree Classifier with entropy.

# In[ ]:


y=df_poetry['age']
x=df_poetry["content"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)


# In[ ]:


vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)


# In[ ]:


dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)


# In[ ]:


accuracy= accuracy_score(preddt,y_test)
print(accuracy)


# Trying to predict authors from rest of the features this time, I don't expect too much of an improvement. Used Tfidf vectorizer and decision tree with gini index as split criterion.

# In[ ]:


y=df_poetry['author']
X=df_poetry.loc[:, df_poetry.columns!="author"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)


# In[ ]:


vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)


# In[ ]:


dtclassifier=DecisionTreeClassifier(criterion="gini", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)


# In[ ]:


accuracy= accuracy_score(preddt,y_test)
print(accuracy)

