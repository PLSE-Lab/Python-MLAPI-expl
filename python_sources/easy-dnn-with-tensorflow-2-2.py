#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# aside from the pandas and numpy we are gioing to use TensorFlow and its childs. So we do some imports here:

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # Tokenizing preprations
# Now, in order to tokenize our texts, convert them to sequences and pad/truncate them into equal sizes we are defining some parameters as the following.
# **max_length** determines the size of sequences. if a sequence size is less than 40, by padding we are going to add some 0's so that the number of elements in the sequence sum up to 40. And if it is more than 40, we are going to delete some elements of the sequence. The value **'post'** means while we are padding, the 0s will be added to the end of the sequence not to the begining of it; or while we are truncating the elements will be deleted from the end.
# 
# the **OOV** stands for **out of vocabulary**. we are going to tokenize the first 3000 words (chosen arbitrarily), and any new word that we encounter afterwards is categorized as OOV.****

# In[ ]:


oov_tok = '<OOV>'
pad_conf = 'post'
trunc_conf= 'post'
max_length= 40
# stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# # Stop Words
# Some words do not contribute to the meaning of the sentence but they are used too often; we don't want to spend our precious time and computation power for them. we call them stop words, and we remove them from the our body of text. there are different sources for stop words, one of the famous ones is provided by the NLTK library which we can download as follows. 

# In[ ]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# Now, we are going to read the data and show the top 5 rows.
# then we separate and refine our body of texts by removing special characters (for example '\n') and converting to lowercase format.
# we also remove the stopwords.

# In[ ]:


train_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
train_data.head()


# In[ ]:


sentences= []
counter=0
for comment in train_data.comment_text:
    sentence= comment.replace('\n', ' the ')
    sentence= comment.replace('\\', ' the ')
    sentence= comment.replace("\'", ' the ')
    sentence= comment.replace('\"', ' the ')
    sentence= comment.replace('\a', ' the ')
    sentence= comment.replace('\b', ' the ')
    sentence= comment.replace('\f', ' the ')
    sentence= comment.replace('\r', ' the ')
    sentence= comment.replace('\t', ' the ')
    sentence= sentence.lower()
    sentence= ' '.join([word for word in sentence.split(' ') if word not in stop_words])
    sentences.append(sentence)


# # Sentences to Sequences
# We get our tokenizer object to some use now; defining the vocab_size we fit the tokenizer on our edited corpus, we get sequences out of it and we then use ***pad_sequences*** to make them equal in size. The padded sequences will have a length of max_length which we have defined earlier to be 40.

# In[ ]:


vocab_size= 3000
tokenizer = Tokenizer(num_words=vocab_size, oov_token= oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# len(word_index)


# In[ ]:


sequences= tokenizer.texts_to_sequences(sentences)
sequences = pad_sequences(sequences, maxlen=max_length, padding=pad_conf, truncating=trunc_conf)

# average length is 39.1345
# I choos 40 as the maxlen for truncating

# This is how I calculated the average length
# counter=0
# counter2=0
# for seq in sequences:
#     counter+=len(seq)
#     counter2 +=1
# print('average length is : ' + str(counter/counter2))


# # The DNN part
# 
# Before creating our NN model, we might want to split our training data into train and validation sets. we keep 80% of the samples for training and the remaining 20% for validation.
# we get the 6 lables, and turn them to numpy arrays so we can feed them to the NN later on.

# In[ ]:


#  train and validation split:
# print(len(sequences))
split = round(0.8 * len(sequences))
valid_seq= sequences[split:]
train_seq= sequences[:split]
# print(len(train_seq))


# In[ ]:


train_labels= train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].head(split)
valid_labels=  train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].tail(len(sequences)-split)

train_labels= np.array(train_labels).astype(np.uint8)
valid_labels= np.array(valid_labels).astype(np.uint8)


# In[ ]:


# we can always delete what we don't need to make some room in the RAM
del train_data
del sequences


# # The model
# It is how we define the model, the embeding with a dimension of 32, following with a convolution layer and a 1D max pooling one after that.
# next, is a LSTM layer to consider the previous words (context) in the meaning.
# and finally, a dense layer with 6 nodes (1 node per label) to determine the output.
# Since the multilable problem means that a sample can have multiple lables, we choose sigmoid over softmax (the sum of output can easily be greater than 1 in this case) and we used binarry_crossentropy loss function.

# In[ ]:


# num_epochs = 10
embeding_dim= 32
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embeding_dim, input_length= max_length ),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(6, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(train_seq, train_labels, epochs=4, validation_data=(valid_seq, valid_labels), verbose=2)

print("Training Complete")


# # Prediction
# After training the model, we take the test data and we go through the same preprocessing of training data all over again. The one difference is that we are now using the pre-trained tokenizer on the new body of text. 
# After that we can use the **model.predict(.)** method to get the final results.

# In[ ]:


test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_labels= pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
test_labels = test_labels[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


# In[ ]:


# indexNames = test_labels[ test_labels['toxic'] == -1 ].index
# test_labels.drop(indexNames , inplace=True)
# test_data.drop(indexNames, inplace=True)

sentences= []
for comment in test_data.comment_text:
    sentence= comment.replace('\n', ' the ')
    sentence= comment.replace('\\', ' the ')
    sentence= comment.replace("\'", ' the ')
    sentence= comment.replace('\"', ' the ')
    sentence= comment.replace('\a', ' the ')
    sentence= comment.replace('\b', ' the ')
    sentence= comment.replace('\f', ' the ')
    sentence= comment.replace('\r', ' the ')
    sentence= comment.replace('\t', ' the ')
    sentence= sentence.lower()
    sentence= ' '.join([word for word in sentence.split(' ') if word not in stop_words])
    sentences.append(sentence)
    
test_sequences= tokenizer.texts_to_sequences(sentences)
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding=pad_conf, truncating=trunc_conf)


# In[ ]:


test_labels=np.array(test_labels).astype(np.uint8)


# In[ ]:


predictions = model.predict(test_sequences)


# In[ ]:


predictions= pd.DataFrame(predictions, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])


# In[ ]:


predictions


# In[ ]:


test_id= pd.Series(test_data['id'].values, name='id')
results= pd.concat([test_id, predictions], axis=1)


# In[ ]:


results.head(5)


# In[ ]:


len(test_data['id'])


# In[ ]:


results.to_csv('results.csv', index=False)


# In[ ]:




