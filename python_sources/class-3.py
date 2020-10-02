#!/usr/bin/env python
# coding: utf-8

# ## Class 3 - deep learning

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

i = 10
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        if i > 10:
            break
        i += 1

# Any results you write to the current directory are saved as output.


# ## Preparing the experiment
# 
# Let's load and derive data from previous classes and train a model.

# In[ ]:


data = pd.read_csv('/kaggle/input/mytitanic/train.csv')
numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
numeric_data = data[numeric_columns]
labels = data['Survived']


# In[ ]:


from sklearn.preprocessing import LabelEncoder

numeric_data['Age'] = data['Age'].fillna(numeric_data['Age'].median())
numeric_data['Sex'] = LabelEncoder().fit_transform(data['Sex'])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(numeric_data, labels, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5, min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0, random_state=20)
rfc.fit(X_train, y_train)
X_train_preds_rfc = rfc.predict(X_train)
X_test_preds_rfc = rfc.predict(X_test)

# End
print(accuracy_score(y_train, X_train_preds_rfc))
print(accuracy_score(y_test, X_test_preds_rfc))


# ## Data preparation
# 
# Do neural networks need standardized data? Why? Does RandomForest need it? Why?
# 
# **0: Optional: Standardize data**
# 
# You can use `StandardScaler` from `sklearn.preprocessing`

# In[ ]:


# Your code here

X_train = 
X_test = 
# End


# ## Training first neural network
# 
# **1: Plese use `keras` library to define and compile Neural Network.**
# 
# What is the neural network build with?
# 
# 1. Layers with hidden units
# 2. Activation functions
# 3. Loss function
# 4. Way of upgrading the weights (so-called optimizer), f.e. 'sgd' or 'adam'
# 5. Learning rate for the optimizer
# 6. Batch size
# 7. Number of epochs
# 
# You may need some regularization also. You can choose dropout, L-2 penalty or batch-normalization too.
# 
# Note that a casual feed-forward layer in `keras` is called `Dense`
# 
# **Please name the variable with model `model`**
# 
# **Please add argument `metrics=['accuracy']` to `compile` method**

# In[ ]:


# Your code here

model = 

# End

model.summary()


# **2: Train the network to maximize test score. **
# 
# a) Try to get at least the score of Random Forest
# 
# b) For `fit()` method, use argument `validation_data`, pass `(X_test, y_test)` tuple
# 
# **Watchout! When you want to run `fit()` again, you have create `model` variable again. Otherwise it's continuing the next epochs of fitting

# In[ ]:


# Your code here

# End


# Plot learning history using the function below:

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

def plot_learning_history(model, epochs_num):
    x = np.arange(epochs_num)
    history = model.history.history

    hist = [
        go.Scatter(x=x, y=history["accuracy"], name="Train Accuracy", marker=dict(size=5), yaxis='y2'),
        go.Scatter(x=x, y=history["val_accuracy"], name="Valid Accuracy", marker=dict(size=5), yaxis='y2'),
        go.Scatter(x=x, y=history["loss"], name="Train Loss", marker=dict(size=5)),
        go.Scatter(x=x, y=history["val_loss"], name="Valid Loss", marker=dict(size=5))
    ]
    layout = go.Layout(
        title="Model Training Evolution", font=dict(family='Palatino'), xaxis=dict(title='Epoch', dtick=1),
        yaxis1=dict(title="Loss", domain=[0, 0.45]), yaxis2=dict(title="Accuracy", domain=[0.55, 1]),
    )
    py.iplot(go.Figure(data=hist, layout=layout), show_link=False)


# In[ ]:


# Your code here

# End


# Use `predict_classes` method to predict classes for train and test data.

# In[ ]:


# Your code here

X_train_preds_nn = 
X_test_preds_nn = 

# End

print(accuracy_score(y_train, X_train_preds_nn))
print(accuracy_score(y_test, X_test_preds_nn))


# Why history shows different accuracy than `accuracy_score`?

# ** 3: Think about what to do to maximize the training score. Create a model that will try to overfit training data as strong as possible. Get at least 85% training score.**
# 
# You don't have to write the code again, you can use the code above if you want to.

# ## Real task introduction: sentiment analysis
# 
# Sentiment analysis is the automated process that uses AI to identify positive and negative opinions from the text. Sentiment analysis may be used for getting insights from social media comments, responses from surveys or product reviews, and making data-driven decisions.
# 
# We can load the data from the attached files

# In[ ]:


path = "/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]
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
reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),
    pd.DataFrame({"review":testReviews, "label":-1, "file":testFiles})
], ignore_index=True).sample(frac=1, random_state=1)


# In[ ]:


reviews.head()


# Example review:

# In[ ]:


print(reviews.iloc[0]['review'])


# ## TFIDF features
# 
# TFIDF features ([here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) or [here](http://datameetsmedia.com/bag-of-words-tf-idf-explained/)) were explained during the lecture. We can use it to represent the text and input it to the neural network.
# 
# Note that Bag-of-words methods are high-dimensional [sparse](https://en.wikipedia.org/wiki/Sparse_matrix) vectors. Therefore please perform the following steps:
# 
# **4: Implement steps:**
# 
# a) Use `TfidfVectorizer` from sklearn to extract features. Extract features from reviews to `X` variable
# 
# b) Check the dimension (number of features) of the resulting vectors
# 
# c) Reduce the number of features using `SelectKBest` end some algorithm f.e. `f_classif`. Leave 10k features.
# 
# d) Verify whether your new dimensions is equal to 10k
# 
# Packages: `sklearn.feature_extraction.text` and `sklearn.feature_selection`
# 
# 4 a)
# 

# In[ ]:


# Your code here

# End


# 
# 4 b)

# In[ ]:


X.shape


# 4 c)

# In[ ]:


# Your code here

# End


# 4 d)

# In[ ]:


X.shape


# ** 5: Optional: Print vocabulary of your `TfidfVectorizer` to understand this representation **
# 
# You can see that a given element in vector represent different word. 

# In[ ]:


# Your code here

# End


# We now prepare splits:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, reviews['label'].values, test_size=0.2, random_state=42)


# Verify the splits:

# In[ ]:


print(X_train.shape)
print(X_test.shape)


# **6: Define and compile the neural network for sentiment analysis. **
# 

# In[ ]:


# Your code here

model = 

# End

model.summary()


# **7: Train the neural network for sentiment analysis **
# 
# a) You should get over 90% accuracy
# 
# b) For `fit()` method, use argument `validation_data`, pass `(X_test, y_test)` tuple

# In[ ]:


# Your code here

# End


# Plot learning history using the `plot_learning_history()` function

# In[ ]:


# Your code here

# End


# ** 8: Test your model on your sentence. **
# 
# To get the probability for input vector, you should use `model.predict()`. Note that you have to use your trained vectorizer, features selector, and model.

# In[ ]:


unseen_sentence = ["this movie very good"]

# Your code here

probability = 

# End

print(probability)
print("Positive!") if probability > 0.5 else print("Negative!")


# ## Word embeddings
# 
# [Word embeddings](https://en.wikipedia.org/wiki/Word_embedding) are dense high-dimensional (but low dimensional compared to tf-idf) vectors that place words with similar semantics in the same area of the hyperspace. 
# 
# Note that pre-trained word-embeddings are attached in the Notebook. 
# 
# **9: Read and run the following code carefully **
#  
# We're going to train two models - one with randomly initialized embeddings, one with pre-trained embeddings. 

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers

MAX_LENGTH = 220
MAX_WORDS = 10000
EMBEDDING_DIM = 300

# Extract tokens from reviews
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(reviews['review'])

# Transform words to indices
sequences = tokenizer.texts_to_sequences(reviews['review'])
word_index = tokenizer.word_index

print(f'Found {len(word_index)} unique tokens.' )

# Pad shorter sentences with 0 and cut longer sentences
data = pad_sequences(sequences, maxlen=MAX_LENGTH)

# Get labels
labels = np.array(reviews['label'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[ ]:


print(f'Shape of Data tensor is {data.shape}')
print(f'Shape of Labels tensor is {labels.shape}')


# Now we train a model that is able to process sequential data. It is called LSTM and is a type of RNN (Recurrent neural networks). It's beyond the scope of the laboratory class, but you'll learn about it during the lecture.
# 
# We define the Embedding layer that is a lookup table for word embeddings. We initialize the embedding layer (weights) with trained word embeddings.

# In[ ]:


from gensim.models import KeyedVectors

def build_matrix(word_index, path):
    embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((MAX_WORDS + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]:
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
        if i == MAX_WORDS:
            break
    return embedding_matrix


# In[ ]:


embeddings_path = "/kaggle/input/gensim-embeddings-dataset/crawl-300d-2M.gensim"

embedding_matrix = build_matrix(word_index, embeddings_path)


# We prevent embeddings from training, so we can use it as is. 

# In[ ]:


model = models.Sequential()
model.add(layers.Embedding(MAX_WORDS + 1, EMBEDDING_DIM, input_length=MAX_LENGTH))
model.add(layers.SpatialDropout1D(0.4))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()


# In[ ]:


model.layers[0].set_weights([embedding_matrix])


# In[ ]:


from keras import optimizers

EPOCHS=3

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=256,
                    validation_data=(X_test, y_test))


# In[ ]:


plot_learning_history(model, EPOCHS)

