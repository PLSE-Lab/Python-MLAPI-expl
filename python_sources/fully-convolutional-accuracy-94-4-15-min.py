#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
from tqdm import tqdm
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
import bz2
from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


def splitReviewsLabels(lines):
    reviews = []
    labels = []
    for review in tqdm(lines):
        rev = reviewToX(review)
        label = reviewToY(review)
        reviews.append(rev[:512])
        labels.append(label)
    return reviews, labels


# In[ ]:


def reviewToY(review):
    return [1,0] if review.split(' ')[0] == '__label__1' else [0,1] 


# In[ ]:


def reviewToX(review):
    review = review.split(' ', 1)[1][:-1].lower()
    review = re.sub('\d','0',review)
    if 'www.' in review or 'http:' in review or 'https:' in review or '.com' in review:
        review = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", review)
    return review


# In[ ]:


train_file = bz2.BZ2File('../input/train.ft.txt.bz2')
test_file = bz2.BZ2File('../input/test.ft.txt.bz2')


# In[ ]:


train_lines = train_file.readlines()
test_lines = test_file.readlines()


# In[ ]:


train_lines = [x.decode('utf-8') for x in train_lines]
test_lines = [x.decode('utf-8') for x in test_lines]


# In[ ]:


# Load from the file
reviews_train, y_train = splitReviewsLabels(train_lines)
reviews_test, y_test = splitReviewsLabels(test_lines)


# In[ ]:


reviews_train, y_train = shuffle(reviews_train, y_train)
reviews_test, y_test = shuffle(reviews_test, y_test)


# In[ ]:


y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


max_features = 8192
maxlen = 128
embed_size = 64


# In[ ]:


tokenizer = Tokenizer(num_words=max_features)


# In[ ]:


tokenizer.fit_on_texts(reviews_train)


# In[ ]:


token_train = tokenizer.texts_to_sequences(reviews_train)
token_test = tokenizer.texts_to_sequences(reviews_test)


# In[ ]:


x_train = pad_sequences(token_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(token_test, maxlen=maxlen, padding='post')


# In[ ]:


input = Input(shape=(maxlen,))
net = Embedding(max_features, embed_size)(input)
net = Dropout(0.2)(net)
net = BatchNormalization()(net)

net = Conv1D(32, 7, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Conv1D(32, 3, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Conv1D(32, 3, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Conv1D(32, 3, padding='same', activation='relu')(net)
net1 = BatchNormalization()(net)

net = Conv1D(2, 1)(net)
net = GlobalAveragePooling1D()(net)
output = Activation('softmax')(net)
model = Model(inputs = input, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


# In[ ]:


model.fit(x_train, y_train, batch_size=2048, epochs=5, validation_split=0.1)


# In[ ]:


model.evaluate (x_test, y_test)


# In[ ]:




