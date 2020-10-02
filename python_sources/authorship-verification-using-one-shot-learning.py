#!/usr/bin/env python
# coding: utf-8

# ## Intorduction
# 
# In this kernel i'm trying to make twitter authorship verfication using Siamese Manhattan LSTM which will try to minimize the distance between tweets vectors 
# 
# ### Siamense network and the architecture :
# Siamese networks are networks that have two or more identical sub-networks in them.
# ![Network](https://serving.photos.photobox.com/03977968b016b46991be02af338e296cc2254b72f5fc9f4225a9b0b11ab01e683e7f26bf.jpg)
# 
# ### The Architecture 
# we have 2 input layers of words to integers lists they pass through Embedding layer to learn embedding weights (better than using pretrained one) then we have a shared LSTM layer between the two inputs then there is a distance function that calculate the distance between the inputs weights vector and last there is a Dense layer with sigmoid activation to predict the probability of each class

# In[ ]:


import os
import re
import string
import pandas as pd
import numpy as np
import random


from time import time
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint


# **The data**  consist of 2 columns (user,text) in the next sections i will reshape this data so that i will make another data frame have pairs of tweets 
# positive matching where 2 tweets comes from same author and negtive matching where 2 tweets come from different authors i will make this process randomly so i will choose random user and then choose 9 random tweets from that user then i will choose random tweet from the rest of users and make pairs i'm not going to make balanced classes here wher we have positive pairs = negative pairs but i will make them closely.

# In[ ]:


dataset = pd.read_excel('../input/train.xlsx')
test=pd.read_excel('../input/test.xlsx')
dataset.head()


# when i collected the data is sampled them to make sure that i have balanced number of tweets per user.

# In[ ]:


dataset.groupby('user').count()


# In[ ]:


len(dataset['user'].unique())


# We have 13 different user in our data

# here i will generate 200K pairs of the data so that the model can have enough data to train on

# In[ ]:


index=range(0,1)
df=pd.DataFrame(index=index)


# i chose to make 20 pairs for each tweet 10 positive pairs and 1 negtive 50:50 as the paper i used mentioned in the reference 

# In[ ]:


first = True
while (len(df)<200000):
    for user in dataset.user.unique():
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        df1 = dataset[dataset['user'] == user][0:20].reset_index(drop=True)
        df2 = dataset[dataset['user'] == user][-10:].reset_index(drop=True)
        df3 = dataset[dataset['user'] != user][-10:].reset_index(drop=True)
        df3 = pd.concat([df2,df3],ignore_index=True)
        if first :
            df = pd.concat([df1,df3],axis = 1)
            first = False
        else:
            df = pd.concat([df,pd.concat([df1,df3],axis = 1)])


# In[ ]:


df.columns = ['user1','text1','user2','text2']
df['is_same'] = 0
df.loc[df['user1']==df['user2'],'is_same'] = 1
df.loc[df['user1']!=df['user2'],'is_same'] = 0


# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)


# Now i got 200K pairs of tweets

# I will clean the data a little bit 

# In[ ]:


def text_to_word_list(tweet):
    ''' Pre process and convert texts to a list of words '''
    tweet = str(tweet)
    tweet = tweet.lower()
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'[_"\-;%()|.,+&=*%]', '', tweet)
    tweet = re.sub(r'\.', ' . ', tweet)
    tweet = re.sub(r'\!', ' !', tweet)
    tweet = re.sub(r'\?', ' ?', tweet)
    tweet = re.sub(r'\,', ' ,', tweet)
    tweet = re.sub(r':', ' : ', tweet)
    tweet = re.sub(r'#', ' # ', tweet)
    tweet = re.sub(r'@', ' @ ', tweet)
    tweet = re.sub(r'd .c .', 'd.c.', tweet)
    tweet = re.sub(r'u .s .', 'd.c.', tweet)
    tweet = re.sub(r' amp ', ' and ', tweet)
    tweet = re.sub(r'pm', ' pm ', tweet)
    tweet = re.sub(r'news', ' news ', tweet)
    tweet = re.sub(r' . . . ', ' ', tweet)
    tweet = re.sub(r' .  .  . ', ' ', tweet)
    tweet = re.sub(r' ! ! ', ' ! ', tweet)
    tweet = re.sub(r'&amp', 'and', tweet)
    tweet = tweet.split()

    return tweet


# here i will encode the data to integers represent the number of word in the global vocabulary

# In[ ]:


vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
tweets_cols = ['text1', 'text2']
def preprocess(df):
    for index, row in df.iterrows():
        for tweet in tweets_cols:
                integers = []
                for word in text_to_word_list(row[tweet]):
                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        integers.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        integers.append(vocabulary[word])

                df.set_value(index, tweet, integers)
    return df


# In[ ]:


df=preprocess(df)


# In[ ]:


df.head()


# In[ ]:


len(vocabulary)


# In[ ]:


max_length = max(df.text1.map(lambda x: len(x)).max(),
                     df.text2.map(lambda x: len(x)).max())


# to make the padding i will need to get the max tweet length 

# In[ ]:


max_length


# prepare the data for the model 

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(df[tweets_cols], df['is_same'], test_size=0.2)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train, test_size=0.1)


# In[ ]:


test = df[df.index.isin(X_test.index)]


# In[ ]:


X_train = {'left': X_train.text1, 'right': X_train.text2}
X_validation = {'left': X_validation.text1, 'right': X_validation.text2}
X_test = {'left': X_test.text1, 'right': X_test.text2}


# In[ ]:


Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values


# Making the padding process so that each vector have the same length of words 

# In[ ]:


def making_pad(data):
    for key,val in data.items():
        data[key]=pad_sequences(data[key],maxlen=max_length)
    return data


# In[ ]:


X_train = making_pad(X_train)
X_validation = making_pad(X_validation)
X_test = making_pad(X_test)


# In[ ]:


assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# In[ ]:


X_train['left'].shape


# Now the network in here i made some differences first i'm adding embedding layer so the the network can learn it's own weights without needing to use another wordtovec
# then i added in the last dense layer to make a probability predection on pairs similarity
# also i used adam optimizer 

# In[ ]:


from keras import optimizers
from keras.layers import merge ,Dense,Concatenate
n_hidden = 20
batch_size = 512
n_epoch = 10

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


left_input = Input(shape=(max_length,), dtype='int32')
right_input = Input(shape=(max_length,), dtype='int32')

embedding_layer = Embedding(len(vocabulary)+1, 100,input_length=max_length)

encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

shared_lstm = LSTM(n_hidden)
left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)
# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0],1))([left_output,right_output])
prediction = Dense(1,activation='sigmoid')(malstm_distance)
# Pack it all up into a model
model = Model([left_input, right_input],outputs=prediction)

# Adadelta optimizer, with gradient clipping by norm
# optimizer = optimizers.Adam(clipnorm=gradient_clipping_norm)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


# Start training
training_start_time = time()

history=model.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In the next sections i'm making the same preprocess for testing data to pass it to the model and check the result .
# 
# i'm testing on 1300 pair of tweets each user with the other users and him self.

# In[ ]:


preds = model.predict([X_test['left'], X_test['right']],batch_size=256)


# In[ ]:


preds[preds>=0.5]=1
preds[preds<0.5]=0


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, preds))


# ### testing the classfication per user accuracy by comparing multiple tweets to multiple users and count how many got the right user

# In[ ]:


index=range(0,1)
dt=pd.DataFrame(index=index)


# In[ ]:


first = True
while (len(dt)<1000):
    for user in test.user1.unique():
        test = test.sample(frac=1).reset_index(drop=True)
        tweet1 = [test[(test['user1']==user)&(test['user2']==user)][0:1]['text1'].values[0]]*13
        tweet2 = [test[(test['user1']==user)&(test['user2']==user)][0:1]['text2'].values[0]]+test[test['user1'] != user][-12:]['text2'].values.tolist()
        if first :
            dt = pd.DataFrame({'tweet1':tweet1,'tweet2':tweet2})
            first = False
        else:
            dt = pd.concat([dt,pd.DataFrame({'tweet1':tweet1,'tweet2':tweet2})])


# In[ ]:


x_test = {'left': dt.tweet1, 'right':dt.tweet2}
x_test=making_pad(x_test)


# In[ ]:


correct = 0
incorrect = 0
for i in range(0,len(dt),13):
    prob = model.predict([x_test['left'][i:i+13],x_test['right'][i:i+13]])
    index=np.argmax(prob)
    if (index==0):
        correct+=1
    else :
        incorrect+=1
    


# In[ ]:


print((100.0*(correct /( correct+incorrect))))


# ### References 
# [keras one shot learning](https://sorenbouma.github.io/blog/oneshot/)
# 
# [similar article with quora questions](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
# 
# [paper using the same method on twitter data](https://lct-master.org/getfile.php?id=3079&n=1&dt=TH&ft=pdf&type=TH)
# 
# [paper talking about Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
# 

# 
