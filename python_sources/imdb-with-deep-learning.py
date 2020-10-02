#!/usr/bin/env python
# coding: utf-8

# **IMDB Movie Review is a dataset for binary sentiment classification containing 25,000 highly polar movie reviews for training, and 25,000 for testing. We will train a neural network to accomplish the objective of sentiment classification.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
from statistics import median

from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
count = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#         pass
        if count > 20 :
            break
        count += 1

plt.style.use("fivethirtyeight")


# # Preparing Train & Test Dataframes 
# **Each train and test sample is provided as a text file contaning movie review as english text. 
# Lets extract each file text and create a data frame.**
#    

# In[ ]:


imdb_dir = '/kaggle/input/raw-imdb-dataset/aclImdb'
train_dir = os.path.join(imdb_dir,'train')
test_dir = os.path.join(imdb_dir,'test')
train_labels = []
train_texts = []

test_labels = []
test_texts = []


# In[ ]:


for label_type in ['pos','neg']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname))
            train_texts.append(f.read())
            f.close()
            if label_type == 'neg':
                train_labels.append(0)
            else:
                train_labels.append(1)


# In[ ]:


for label_type in ['pos','neg']:
    dir_name = os.path.join(test_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname))
            test_texts.append(f.read())
            f.close()
            if label_type == 'neg':
                test_labels.append(0)
            else:
                test_labels.append(1)


# In[ ]:


print(f'Length of train texts is {len(train_texts)}')
print(f'Length of train labels id {len(train_labels)}')
print(f'Length of test texts is {len(test_texts)}')
print(f'Length of test labels is {len(test_labels )}')


# In[ ]:


texts_df = pd.DataFrame({'texts': train_texts,
                        'labels':train_labels})


# In[ ]:


texts_df['word counts'] = texts_df['texts'].apply(lambda x: len(x.split()))


# In[ ]:


texts_df.head()


# # Exploratory Data Analysis

# In[ ]:


median_word_count = median(texts_df['word counts'])
plt.figure(figsize=(12,6))
plt.hist(texts_df['word counts'],edgecolor='black')
plt.title("Words Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency/occurrence")

color = '#fc4f30'

plt.axvline(median_word_count,color=color,label="Median Word Count")
plt.legend()
plt.tight_layout()


# In[ ]:


positive = texts_df[texts_df['labels']==1]['texts']
negative = texts_df[texts_df['labels']==0]['texts']


# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',
                      stopwords=stopwords,
                      max_words=200,
                      max_font_size=40, 
                      random_state=42,
                      collocations=False
                      ).generate(str(positive))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[ ]:


wordcloud = WordCloud(background_color='black',
                      stopwords=stopwords,
                      max_words=200,
                      max_font_size=40, 
                      random_state=42,
                      collocations=False
                      ).generate(str(negative))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# # Feature Engineering 

# In[ ]:


MAX_LENGTH = 1000
MAX_WORDS = 20000
EMBENDING_DIM = 100


# In[ ]:


my_stop_words = ENGLISH_STOP_WORDS.union(["br","movie","film"])
def remove_stopword(text):
    """
    Removes StopWords
    """
    return " ".join([word for word in text.lower().split() if word not in my_stop_words])


# In[ ]:


train_texts = texts_df['texts'].apply(remove_stopword)


# In[ ]:


tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)


# In[ ]:


word_index = tokenizer.word_index


# In[ ]:


print(f'Found {len(word_index)} unique tokens.' )


# In[ ]:


word_list = []
count_list = []
for key,val in word_index.items():
    if val in range(1,21):
        word_list.append(key)
        count_list.append(tokenizer.word_counts[key])


# In[ ]:


word_list.reverse()
count_list.reverse()
plt.figure(figsize=(12,10))
plt.barh(word_list,count_list)
plt.title("Top 20 Words")
plt.xlabel("Counts")
plt.ylabel("Words")
plt.tight_layout()


# In[ ]:


data = pad_sequences(sequences,maxlen=MAX_LENGTH)
labels = np.array(train_labels)


# In[ ]:


print(f'Shape of Data tensor is {data.shape}')
print(f'Shape of Labels tensor is {labels.shape}')


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(data,labels,test_size=0.2,random_state=42)


# In[ ]:


glove_dir = "/kaggle/input/globe6bzip/glove.6B.100d.txt"

embedding_index = {}
f = open(glove_dir)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

print(f'Found {len(embedding_index)} word vectors')


# In[ ]:


embedding_index['go'][:10]


# In[ ]:


embedding_matrix = np.zeros((MAX_WORDS,EMBENDING_DIM))


# In[ ]:


for word, i in word_index.items():
    if i < MAX_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_df = pd.DataFrame(embedding_matrix)


# In[ ]:


embedding_df.shape


# # Model Training

# In[ ]:


def create_model():
    model = models.Sequential()
    model.add(layers.Embedding(MAX_WORDS,EMBENDING_DIM, input_length=MAX_LENGTH))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['acc'])
    return model


# In[ ]:


model = create_model()
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


# In[ ]:


history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


# In[ ]:


val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
train_acc = history.history['acc']


# In[ ]:


plt.figure(figsize=(12,8))
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


final_model = create_model()
final_model.layers[0].set_weights([embedding_matrix])
final_model.layers[0].trainable = False
final_model.fit(data, labels,
                epochs=2,
                batch_size=32)


# # Predictions

# In[ ]:


test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences,maxlen=MAX_LENGTH)
test_labels = np.array(test_labels)


# In[ ]:


predictions = final_model.predict(test_data)
pred_proba = final_model.predict_proba(test_data)


# In[ ]:


pred_labels  = (predictions>0.5)


# # Performance Evaluation

# In[ ]:


mat = confusion_matrix(pred_labels, test_labels)
plt.figure(figsize=(4, 4))
sns.set()
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=np.unique(test_labels),
            yticklabels=np.unique(test_labels))
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[ ]:


print(classification_report(test_labels,pred_labels))


# In[ ]:


precisions, recalls, thresholds = precision_recall_curve(test_labels,pred_proba)
plt.figure(figsize=(12,6))
plt.plot(thresholds, precisions[:-1],label='Precision')
plt.plot(thresholds, recalls[:-1],label="Precision/Recall")
plt.xlabel("Threshold")
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.tight_layout()


# In[ ]:


fpr,tpr, thresholds = roc_curve(test_labels,pred_proba)
plt.figure(figsize=(12,6))
plt.plot(fpr,tpr, label=None)
plt.plot([0,1],[0,1])
plt.axis([0,1,0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()


# In[ ]:




