#!/usr/bin/env python
# coding: utf-8

# # Spam or Ham ?
# * We will see two approaches to vectorize text - word embedding and TF-IDF
# * We will transfer learning from the GloVe word embeddings dataset.
# * We use recall to evaluate the model because the dataset is highly unbalanced.
# * We also see the False Positives because we do not want genuine SMS's to be classified as spam.
# * TF-IDF works supremely better than word embeddings because the texts have a lot of shorthand and misspelt words.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

from keras.models import Model
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


# In[ ]:


embeddings_index = dict()
f = open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


raw_data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')
raw_data.head()


# In[ ]:


raw_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
le = LabelEncoder()
raw_data.v1 = le.fit_transform(raw_data.v1)
raw_data.head()


# # Preprocessing

# We have an unbalanced dataset. Only 13.41% of the data is spam. So, let's use recall as a metric to evaluate our model.

# In[ ]:


num_spam = raw_data.v1.sum()
num_ham = len(raw_data) - num_spam

plt.pie([num_ham, num_spam],labels=["Ham", "Spam"],explode=(0,0.2),autopct='%1.2f%%',startangle=45)
plt.show()


# We are going to try lemmatization and stopword removal. However, conventional processing techniques are not going to work well with a SMS corpus. The texts have a lot of shortened words and abbreviations. Ideally, we have to implement a customized normalization of text.

# In[ ]:


total_stopwords = set([word.replace("'",'') for word in stopwords.words('english')])

def preprocess_text(text):
    text = text.lower()
    text = text.replace("'",'')
    text = re.sub('[^a-zA-Z]',' ',text)
    words = text.split()
    words = [lemma.lemmatize(word) for word in words if (word not in total_stopwords) and (len(word)>1)] # Remove stop words
    text = " ".join(words)
    return text


# In[ ]:


raw_data.v2 = raw_data.v2.apply(preprocess_text)


# Taking a look at the most common words in spam SMS. We also observe a lot of shorthand which is going to mislead our classifiers.

# In[ ]:


wordcloud = WordCloud(height=2000, width=2000, stopwords=set(stopwords.words('english')), background_color='white')
wordcloud = wordcloud.generate(' '.join(raw_data[raw_data.v1==1].v2))
plt.imshow(wordcloud)
plt.title("Most common words in spam SMS")
plt.axis('off')
plt.show()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(raw_data.v2, raw_data.v1, test_size=0.15, stratify=raw_data.v1)


# # Word Embedding approach
# We use a pre-trained GloVe embedding because we have a very small training set. This will serve as a starting point for our trainable embedding layer.

# In[ ]:


max_words = x_train.apply(lambda str: len(str.split())).max()


# In[ ]:


tok = Tokenizer()
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_words, padding='post')
vocab_size = len(tok.word_index) + 1


# In[ ]:


embedding_matrix = np.zeros((vocab_size, 50))
for word, i in tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 50, input_length=max_words, weights=[embedding_matrix], trainable=True))
model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Bidirectional(LSTM(20)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Dense(64, activation='relu'))
model_glove.add(Dense(64, activation='relu'))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model_glove.fit(sequences_matrix, y_train, epochs = 10)


# In[ ]:


test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_words)


# In[ ]:


y_pred = model_glove.predict(test_sequences_matrix)


# In[ ]:


pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(thresholds, pr[1:])
plt.plot(thresholds, rc[1:])
plt.show()
crossover_index = np.max(np.where(pr <= rc))
crossover_cutoff = thresholds[crossover_index]
crossover_recall = rc[crossover_index]

print(classification_report(y_test, y_pred > crossover_cutoff))

m_confusion_test = confusion_matrix(y_test, y_pred > crossover_cutoff)
display(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))

print("Terrible! This model misses all the spam SMS.")


# # TF-IDF approach

# In[ ]:


vectorizer = TfidfVectorizer()
vectorizer.fit(x_train)


# In[ ]:


x_train_vec = vectorizer.transform(x_train).toarray()


# In[ ]:


model_tdif = svm.SVC(gamma='scale')
model_tdif.fit(x_train_vec, y_train)


# In[ ]:


x_test_vec = vectorizer.transform(x_test).toarray()


# In[ ]:


y_pred_tdif = model_tdif.predict(x_test_vec)


# In[ ]:


print(classification_report(y_test, y_pred_tdif))

m_confusion_test = confusion_matrix(y_test, y_pred_tdif)
display(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))

print("This model misclassifies {c} genuine SMS as spam and misses only {d} SPAM.".format(c = m_confusion_test[0,1], d = m_confusion_test[1,0]))


# With a significantly higher recall value and fewer false positives, the TF-IDF based vectorization is a more superior choice as compared to word embeddings. Since the corpus has a lot of shorthand and mispelt words, the word embeddings rendered do not hold any analogical meanings (the main reason why we use embeddings).
