import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential


from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"]
val_X = val_df["question_text"]
test_X = test_df["question_text"]


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

# Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)



# def batch_generator(X, y, batch_size, shuffle):
#     number_of_batches = X.shape[0]/batch_size
#     counter = 0
#     sample_index = np.arange(X.shape[0])
#     if shuffle:
#         np.random.shuffle(sample_index)
#     while True:
#         batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
#         X_batch = X[batch_index,:].toarray()
#         y_batch = y[batch_index]
#         counter += 1
#         yield X_batch, y_batch
#         if (counter == number_of_batches):
#             if shuffle:
#                 np.random.shuffle(sample_index)
#             counter = 0

#sklearn
# vectorizer = CountVectorizer(binary=True)
# vectorizer.fit(test_X)
# train_X=vectorizer.transform(train_X)
# val_X=vectorizer.transform(val_X)
# test_X=vectorizer.transform(test_X)
# feature_size=train_X.shape[1]
# Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


#model.add([Dense(512, input_shape=(feature_size,),activation='softmax'),Dense(2,activation='relu')])
#model.add(Activation('softmax'))

# batch_size = 32
# epochs = 2

# model = Sequential()
# model.add(Dense(1, input_shape=(100,)))
# model.add(Activation('relu'))

model=Sequential([Embedding(max_features,embed_size,input_length=150),Dense(256,activation="relu"),Flatten(),Dense(1,activation="softmax")])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
print(model.summary())

# def train_model(model,train_X,train_y):
#     size=train_X.shape[0]
#     index=0
#     counter=0
#     while True:
        
#         slot=50000
#         if((index+5000)>=size):
#             slot=size-index
#         temp_X=train_X[index:index+slot]
#         temp_y=train_y[index:index+slot]
#         model.fit(temp_X, temp_y, batch_size=32, epochs=1,   verbose=1,  validation_split=0.03)
#         print("counter=> "+str(counter)+ " index=>"+str(index) )
#         index=index+slot
#         if(index>=size):
#             break
#     return
# train_model(model,train_X,train_y)
# history = model.fit(train_X, train_y, batch_size=32, epochs=2,   verbose=1,  validation_split=0.1)
#model.fit_generator(generator=batch_generator(train_X, train_y, 32, True),nb_epoch=2,samples_per_epoch=train_X.shape[0])
model.fit(train_X, train_y, batch_size=32, epochs=1,   verbose=1,  validation_split=0.03)
score = model.evaluate(val_X,val_y, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
predict=model.predict_classes(test_X)
predict1=[]
for i in range(len( predict)):
    predict1.append(int(predict[i][0]))


result=pd.DataFrame({"qid":test_df["qid"],"prediction":predict1})
result.to_csv("submission.csv",index=False)
result.to_csv("submission1.csv",index=False)
model.save("model.h5")
result=pd.DataFrame({"qid":test_df["qid"],"prediction":predict})
result.to_csv("submission2.csv",index=False)