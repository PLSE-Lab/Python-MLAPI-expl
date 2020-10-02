#!/usr/bin/env python
# coding: utf-8

# **In this kernel we will try to predict sentiment with Logistic Regression and again with LSTM to compare results.
# **I hope you enjoy it:) 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import string
import matplotlib.pyplot as plt;

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Tweets.csv')
df.describe()


# # first: Logistic Regression Model

# ### **1- Text Clean Up**

# In[ ]:


stop_words = stopwords.words('english') 


# In[ ]:


def clean_text(txt):
    
    """
    removing all hashtags , punctuations, stop_words  and links, also stemming words 
    """
    txt = txt.lower()
    txt = re.sub(r"(@\S+)", "", txt)  # remove hashtags
    txt = txt.translate(str.maketrans('', '', string.punctuation)) # remove punctuations 
    txt = re.sub(r"(http\S+|http)", "", txt) # remove links 
    txt = ' '.join([PorterStemmer().stem(word=word) for word in txt.split(" ") if word not in stop_words ]) # stem & remove stop words
    txt = ''.join([i for i in txt if not i.isdigit()]).strip() # remove digits ()
    return txt


# Tryout our newly defined function 

# In[ ]:


print('Original Text : ',df['text'][3])  
print('Processed Text : ',clean_text(df['text'][3]))


# Now apply our function to the dataset, 
# also we need to encode target variable (airline_sentiment) to be 0 whenever a tweet is negative and 1 otherwise.

# In[ ]:


df['sent_encoded'] = df['airline_sentiment'].apply(lambda x:0 if x =='negative' else 1)
df['cleaned_text'] = df['text'].apply(clean_text)  


# Create Train text splits.
# for a model to run we need to:
# 
# 1- tokenize Text
# 
# 2- encode every word as a feature
# 
# 3- represent word accurencies in a text as a count.( done by **CountVectorizer** )

# In[ ]:


def train_test_data():   
    y = df['sent_encoded']   # define target and feature column
    X = df['cleaned_text']
     
    text_train, text_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)    # do the split
    vect = CountVectorizer(min_df=5, ngram_range=(1, 4)) # create Count vectorizer.
    X_train = vect.fit(text_train).transform(text_train) # transform text_train  into a vector 
    X_test = vect.transform(text_test) 
    feature_names = vect.get_feature_names() # to return all words used in vectorizer
  
    return X_train, X_test, y_train, y_test, feature_names


# lets try the newly created function 

# In[ ]:


X_train, X_test, y_train, y_test, feature_names = train_test_data()


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# ### **2- Train The Model**

# In[ ]:


lgstc = LogisticRegressionCV(class_weight={1:0.515,0:0.485})
lgstc.fit(X_train, y_train)


# our model of selection is Logistic Regression reason is "simplicity". as it's always a good idea to start with the simplest model possible then see if you need to complicate it. 
# << Start small , think big >>

# In[ ]:





# ### **3- Test model performance **

# to test the model performance , we create a function to print out all results, to test overfitting we must compare test performance to train performance.
# 

# In[ ]:


def print_model_performance(model,X_train,X_test,y_train,y_test):
    training_sample = model.predict(X_train)
    testing_sample = model.predict(X_test)
    print('training ')
    #print(classification_report(training_sample, y_train))  #uncomment if you want to see full report 
    print('train accuracy ',accuracy_score(training_sample, y_train))
    print('train precision_score ',precision_score(training_sample, y_train)) 
    print('train recall score',recall_score(training_sample, y_train)) 
    
    print('\n testing  ')
    print(classification_report(testing_sample, y_test))   #uncomment if you want to see full report 
    print('test average accuracy ',accuracy_score(testing_sample, y_test))
    print('test average precision_score ',precision_score(testing_sample, y_test)) 
    print('test average recall score',recall_score(testing_sample, y_test)) 
    print('test AUC ',roc_auc_score(testing_sample, y_test))
    
    print(confusion_matrix(testing_sample, y_test))


# In[ ]:


print_model_performance(lgstc, X_train, X_test, y_train, y_test)


# from the confusion matrix bellow , we see that model is better at detecting negative tweets, reason is because we have more negative samples than positive. 
# 

# # Deep Learning (LSTM ) : 

# lets try to "complicate things" a bit. 
# now we will create a deep learning model ( RNN-lstm )  to increase our 82% accuracy.

# 

# 

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


# ### **2- define the global variables: **

# In[ ]:


max_fatures = 2000 # maximum number of words to use
embed_dim = 120 # embidding dimention
lstm_out = 190 # lstm size
batch_size = 32 # batch size
validation_size = 1500 # validation set size


# I came out with these values by intuition :) also trial and error. 

# ### **3- get X and y**

# In[ ]:


def get_X_y(feature, target):
    data = df.copy() # create a copy so we dont mess up our old dataframe
    
    data = data[[feature,target]] # cut down all dataframe to only features and target variables 
    
    data = data.dropna(subset=[feature]) # make sure there is no (NA) values as it will not help predictions 
    
    data[feature] = data[feature].apply(lambda x: x.lower()) # convert text to lower case
    data[feature] = data[feature].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # remove all digits
    
    tokenizer = Tokenizer(num_words=max_fatures, split=' ') # create our tokenizer here we used NLTK as it's a preferable package for developers using deep learning 
    tokenizer.fit_on_texts(data[feature].values) 
    
    X = tokenizer.texts_to_sequences(data[feature].values) # here is the main trick! we convert:  'hi im x' to something like [2, 12, 53] where 2 , 12 , 34 ()
    X = pad_sequences(X)    
    Y = pd.get_dummies(data[target]).values
    return X, Y


#  here is the main trick! we convert:  'hi im x' to something like [2, 12, 53] by using text_to_sequences functions  ( where 2, 12, 53) are indexes of words hi, m , x in **order** . 
#  here we sense the power of LSTM , as order of words can mean different thing 
#  

# In[ ]:


X, Y = get_X_y('text', 'sent_encoded')


# lets try it out : 

# In[ ]:


print(df['text'][3]) # third tweet
print(X[[3]]) # third tweet to sequence


# ### 4- create the model : 
# we will create an LSTM model with a sigmoid activation. you might want to use softmax instead of sigmoid . but from my personal experience sigmoid performs better in binary situations. 
# but you are free try out on your own and see what fits you more :)

# In[ ]:


def get_trained_model(X, Y):
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(rate=0.9))
    model.add(LSTM(lstm_out))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    
    model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    return model, X_validate, Y_validate, X_test, Y_test


# as you can see I used dropout rate of 90% , how did i reach that number?? answer is trial and error :).

# In[ ]:


model, X_validate, Y_validate, X_test, Y_test = get_trained_model(X,Y)


# We have a model with 85% accuracy , which seems fine ( at least we improved a bit !). and all thats left is to validate model to see it's actual performance: 
# 

# In[ ]:


def validate_model(model, X_validate, Y_validate, X_test, Y_test):
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    all_predictions = []
    for x in range(len(X_validate)):
    
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]), batch_size=1, verbose=2)[0]
        all_predictions.append(result)
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
    
        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")
    
    all_currects = neg_correct + pos_correct
    all_samples = neg_cnt + pos_cnt
    
   
    y_predict = np.asarray(all_predictions)
    y_actual = np.asarray(Y_validate)
    
    y_actual = np.argmax(y_actual, axis=1)
    y_predict = np.argmax(y_predict, axis=1)
    
  
    cm = confusion_matrix(y_predict, y_actual)	
    print('AUC : ',roc_auc_score(y_predict, y_actual))
    #print("Average Accuracy : ", all_currects/all_samples*100, "%")
    return cm.ravel()


# although this function might look scary . but actually not so much is happeing , all im doing is checking negative and positive accuracy. and return confusion matrix to compare it with the model above. 
# 
# note: no need to praise me for this function it's a straight up copy and past from keras tutorials. ( with some of my touches) 

# 

# ### **5- model validation **

# In[ ]:


tn, fp, fn, tp = validate_model(model, X_validate, Y_validate, X_test, Y_test)


# In[ ]:


print('average accuracy : ', (tp + tn) / (tn+fp+fn+tp))
print('average precision : ', (tp) / (fp+tp))
print('average Recall : ', (tp) / (tp+fn))


# Second model is better, and there is a big room for improvement. maybe try adding more hidden layers. or even do more text preprocessing. 
