#!/usr/bin/env python
# coding: utf-8

# The code in this post can be found at my [Github](https://github.com/sanjay-raghu) repository. If you are also interested in trying out the code
# I have also written a blog on my [website](https://sanjay-raghu.github.io/Sentiment-Analysis-Using-LSTM/) you can visit there.   
# 
# 
# 
# **Sentiment Analysis:**<br> 
# The process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral. In common ML words its just a classification problem. 
# 
# **What is class imbalance:**<br>
# It is the problem in machine learning where the total number of a class of data (positive) is far less than the total number of another class of data (negative). This problem is extremely common in practice and can be observed in various disciplines including fraud detection, anomaly detection, medical diagnosis, oil spillage detection, facial recognition, etc.
# 
# If you want to know more about class imbalance problem, [here](http://www.chioka.in/class-imbalance-problem/) is a link of a great blog post
# 
# **Solving class imbalanced data:**<br>
# I am using the two most effective ways to mitigate this:<br>
# - Up sampling 
# - Using class weighted loss function
# 
# **Dataset**<br>
# First GOP Debate Twitter Sentiment
# About this Dataset
# This data originally came from [Crowdflower's Data for Everyone library ](http://www.crowdflower.com/data-for-everyone).
# 
# > As the original source says,
# > We looked through tens of thousands of tweets about the early August GOP debate in Ohio and asked contributors to do both
# > sentiment analysis and data categorization. Contributors were asked if the tweet was relevant, which candidate was mentioned,
# > what subject was mentioned, and then what the sentiment was for a given tweet. We've removed the non-relevant messages from
# > the uploaded dataset.
# 
# **Details about model**<br>
#  - model contains 3 layers (Embedding, LSTM, Dense with softmax).
#  - Up-sampling is used to balance the data of minority class.
#  - Loss function with different class weight in keras to further reduce class imbalance.
# 

# ## Lets start coding
# ### Importing useful packages
# Lets first import all libraries. Please make sure that you have these libraries installed.   
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# ### Data Preprocessing
# - reading the data
# - kepping only neccessary columns
# - droping "Neutral" sentiment data
# 

# In[ ]:


data = pd.read_csv('../input/Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]
data = data[data.sentiment != "Neutral"]


# Let See the few lines of the data

# In[ ]:


data.head()


# > A few things to notice here
# - "RT @..." in start of every tweet
# - a lot of special characters <br>
# > We have to remove all this noise also lets convert text into lower case.
# 

# In[ ]:


data['text'] = data['text'].apply(lambda x: x.lower())
# removing special chars
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
data['text'] = data['text'].str.replace('rt','')
data.head()


# This looks better.<br>
# Lets pre-process the data so that we can use it to train the model
# - Tokenize
# - Padding (to make all sequence of same lengths)
# - Converting sentiments into numerical data(One-hot form)
# - train test split
# 

# In[ ]:


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ### Defining model
# Next, I compose the LSTM Network. Note that **embed_dim**, **lstm_out**, **batch_size**, **droupout_x** variables are hyper parameters, their values are somehow intuitive, can be and must be played with in order to achieve good results. Please also note that I am using softmax as activation function. The reason is that our Network is using categorical crossentropy, and softmax is just the right activation method for that.

# In[ ]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# ### Let's train the model
# Here we train the Network. We should run much more than 15 epoch, but I would have to wait forever (run more epochs later), so it is 15 for now. you will see progress bar (if you want to shut it up use verbose = 0)
# 
# 

# In[ ]:


batch_size = 128
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1)


# ### Let evaluate the model
# 

# In[ ]:


Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))
print(classification_report(df_test.true, df_test.pred))


# > It is clear that finding negative tweets (**class 0**) goes very well (**recall 0.92**) for the Network but deciding whether is positive (**class 1**) is not really (**recall 0.52**). My educated guess here is that the positive training set is dramatically smaller than the negative, hence the "bad" results for positive tweets.
# ## Solving data imbalance problem
# 
# **Up-sample Minority Class**
# 
# Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal. There are several heuristics for doing so, but the most common way is to simply re-sample with replacement.
# 
# It's important that we separate test set before up-sampling because after up-sampling there will be multiple copies of same data point and if we do train test split after up-sampling the test set will not be completely unseen.
# 

# In[ ]:


# Separate majority and minority classes
data_majority = data[data['sentiment'] == 'Negative']
data_minority = data[data['sentiment'] == 'Positive']

bias = data_minority.shape[0]/data_majority.shape[0]
# lets split train/test data first then 
train = pd.concat([data_majority.sample(frac=0.8,random_state=200),
         data_minority.sample(frac=0.8,random_state=200)])
test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8,random_state=200).index),
        data_minority.drop(data_minority.sample(frac=0.8,random_state=200).index)])

train = shuffle(train)
test = shuffle(test)


# In[ ]:


print('positive data in training:',(train.sentiment == 'Positive').sum())
print('negative data in training:',(train.sentiment == 'Negative').sum())
print('positive data in test:',(test.sentiment == 'Positive').sum())
print('negative data in test:',(test.sentiment == 'Negative').sum())


# Now Lets do up-sampling

# In[ ]:


# Separate majority and minority classes in training data for upsampling 
data_majority = train[train['sentiment'] == 'Negative']
data_minority = train[train['sentiment'] == 'Positive']

print("majority class before upsample:",data_majority.shape)
print("minority class before upsample:",data_minority.shape)

# Upsample minority class
data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
 
# Display new class counts
print("After upsampling\n",data_upsampled.sentiment.value_counts(),sep = "")

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values) # training with whole data

X_train = tokenizer.texts_to_sequences(data_upsampled['text'].values)
X_train = pad_sequences(X_train,maxlen=29)
Y_train = pd.get_dummies(data_upsampled['sentiment']).values
print('x_train shape:',X_train.shape)

X_test = tokenizer.texts_to_sequences(test['text'].values)
X_test = pad_sequences(X_test,maxlen=29)
Y_test = pd.get_dummies(test['sentiment']).values
print("x_test shape", X_test.shape)


# In[ ]:


# model
embed_dim = 128
lstm_out = 192

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# Lets define class weights as a dictionary, I have defined weight of majority class to be 1 and of minority class to be a multiple of 1/bias
# 

# In[ ]:


batch_size = 128
# also adding weights
class_weights = {0: 1 ,
                1: 1.6/bias }
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1,
          class_weight=class_weights)


# In[ ]:


Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))
print(classification_report(df_test.true, df_test.pred))


# So the class imbalance is reduced significantly recall value for positive tweets (Class 1) improved from 0.54 to 0.74. It is always not possible to reduce it completely. 
# You may also noticed that the recall value for Negative tweets also decreased from 0.90 to 0.79  but this can be improved using training model to more epochs and tuning the hyper-parameters.
# 

# In[ ]:


# running model to few more epochs
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1,
          class_weight=class_weights)
Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))
print(classification_report(df_test.true, df_test.pred))


# In[ ]:


twt = ['keep up the good work']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=29, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")


# 1. ## Tuning the hyper-parameters using gridsearch

# In[ ]:


# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


# # Function to create model, required for KerasClassifier
# def create_model(dropout_rate = 0.0):
#     # create model
#     embed_dim = 128
#     lstm_out = 192
#     model = Sequential()
#     model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
#     model.add(SpatialDropout1D(dropout_rate))
#     model.add(LSTM(lstm_out, dropout=dropout_rate, recurrent_dropout=dropout_rate))
#     model.add(Dense(2,activation='softmax'))
#     model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# #     print(model.summary())
#     return model
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

# model = KerasClassifier(build_fn=create_model, verbose=2,epochs=30, batch_size=128)
# # define the grid search parameters
# # batch_size = [128]
# # epochs = [5]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # class_weight = [{0: 1, 1: 1/bias},{0: 1, 1: 1.2/bias},{0: 1, 1: 1.5/bias},{0: 1, 1: 1.8/bias}]
# param_grid = dict(dropout_rate = dropout_rate)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, Y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

