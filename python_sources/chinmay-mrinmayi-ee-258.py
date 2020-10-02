#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
df2 = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')


# In[ ]:


# extracting the number of examples of each class
sincere_questions = df[df['target'] == 0].shape[0]
insincere_questions = df[df['target'] == 1].shape[0]


# In[ ]:


import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# bar plot of the 3 classes
plt.bar(10,sincere_questions,3, label="sincere")
plt.bar(15,insincere_questions,3, label="insincere")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Proportion of examples')
plt.show()


# In[ ]:


print("The number of sincere questions is: ", sincere_questions)
print("The number of insincere questions is: ", insincere_questions)


# In[ ]:


import numpy as np
import re
import nltk
#nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# In[ ]:


df["question_text"] = normalize_corpus(df["question_text"])


# In[ ]:


df2["question_text"] = normalize_corpus(df2["question_text"])


# In[ ]:


#nltk.download('punkt')
df["question_text"] = df["question_text"].apply(nltk.word_tokenize)
print ("series.apply to train questions")


# In[ ]:


df2["question_text"] = df2["question_text"].apply(nltk.word_tokenize)
print ("series.apply to test questions")


# In[ ]:


from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
df['question_text_tokenized_stemmed']=df['question_text'].apply(lambda x : [porter_stemmer.stem(y) for y in x])


# In[ ]:


df2['question_text_tokenized_stemmed']=df2['question_text'].apply(lambda x : [porter_stemmer.stem(y) for y in x])


# In[ ]:


#nltk.download('wordnet')
df['question_text_tokenized_lemmatized']=df['question_text_tokenized_stemmed'].apply(lambda x : [WordNetLemmatizer().lemmatize(y) for y in x])


# In[ ]:


df2['question_text_tokenized_lemmatized']=df2['question_text_tokenized_stemmed'].apply(lambda x : [WordNetLemmatizer().lemmatize(y) for y in x])


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#import numpy as np
maxlen = 12
training_samples = 848980 #65%
validation_samples = 457142
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.question_text_tokenized_lemmatized.values)
sequences = tokenizer.texts_to_sequences(df.question_text_tokenized_lemmatized.values)


# In[ ]:


tokenizer.fit_on_texts(df2.question_text_tokenized_lemmatized.values)
test_sequences = tokenizer.texts_to_sequences(df2.question_text_tokenized_lemmatized.values)


# In[ ]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(df.target)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# In[ ]:


x_test = pad_sequences(test_sequences, maxlen=maxlen)


# In[ ]:


import os
glove_dir = '../input/glove6b100dtxt'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# In[ ]:


from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model1 = Sequential()
model1.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model1.add(LSTM(25))
model1.add(Dense(1, activation='sigmoid'))

model1.layers[0].set_weights([embedding_matrix])
model1.layers[0].trainable = False

model1.summary()


# In[ ]:


from keras import backend as K
def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


# In[ ]:


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


model1.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc', f1, precision, recall])
history = model1.fit(x_train, y_train,
epochs=1,
batch_size=100,
validation_data=(x_val, y_val))
model1.save_weights('processed_and_trained1.h5')


# In[ ]:


y_predicted = model1.predict_classes(x_test)


# In[ ]:


print(y_predicted)


# In[ ]:


submit = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
#del submit['question_text']
submit.rename(index=str, columns={"question_text": "target"})
submit.question_text = y_predicted
submit.to_csv("sample_submission.csv", index=False)


# In[ ]:





# In[ ]:




