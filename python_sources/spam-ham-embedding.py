# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
import gensim.downloader as api
import os
import shutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
# Download data
def download_and_read(url):
    local_file = url.split('/')[-1]
    p = tf.keras.utils.get_file(local_file, url, 
                               extract=True, cache_dir='.')
    labels, texts = [], []
    local_file = os.path.join("datasets", "SMSSpamCollection")
    with open(local_file, "r") as f:
        for line in f:
            label, text = line.strip().split('\t')
            labels.append(1 if label=="spam" else 0)
            texts.append(text)
    return texts, labels
            

# %% [code]
def build_embedding_matrix(sequences, word2idx, embedding_dim, embedding_file):
    if os.path.exists(embedding_file):
        E = np.load(embedding_file)
    else:
        vocab_size = len(word2idx)
        E = np.zeros((vocab_size, embedding_dim))
        word_vectors = api.load(EMBEDDING_MODEL)
        for word, idx in word2idx.items():
            try:
                E[idx] = word_vectors.word_vec(word)
            except KeyError:
                pass
        np.save(embedding_file, E)
    return E


# %% [code]
class SpamClassifierModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, input_length, num_filters, kernel_size, output_size,
                run_mode, embedding_weights, **kwargs):
        super(SpamClassifierModel, self).__init__(**kwargs)
        if run_mode=="SCRATCH":
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                       input_length = input_length,
                                                       trainable=True)
        elif run_mode=="VECTORIZER":
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                       input_length = input_length,
                                                       weights=[embedding_weights],
                                                       trainable=False)
        else:
            self.embedding=tf.keras.layres.Embedding(vocab_size, embedding_size,
                                                     input_length = input_length,
                                                    weights=[embedding_weights],
                                                    trainable=True)
        self.conv = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size,
                                          activation="relu")
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(output_size, activation='softmax')
        
    
    def call(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense(x)
        return x
        

# %% [code]
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
texts, labels = download_and_read(DATASET_URL)

# %% [code]
# Tokenize and Pad Text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)
num_records = len(text_sequences)
max_seqlen  = len(text_sequences[0])
print("{:d} sentences, max_seqlen {:d}".format(num_records, max_seqlen))

# %% [code]
# Labels
NUM_CLASSES = len(set(labels))
cat_labels = tf.keras.utils.to_categorical(labels, num_classes = NUM_CLASSES)

# %% [code]
# Vocabulary
word2idx = tokenizer.word_index
idx2word = {v:k for k,v in word2idx.items()}
word2idx["PAD"] = 0
idx2word[0] ="PAD"
vocab_size = len(word2idx)
print("Vocab size {:d}".format(vocab_size))

# %% [code]
# Dataset
dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))
dataset = dataset.shuffle(10000)
test_size = num_records //4
val_size = (num_records - test_size) // 10
test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(val_size+test_size)


# %% [code]
BATCH_SIZE = 128
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

# %% [code]
# Build embedding matrix
api.info().keys()

# %% [code]
EMBEDDING_DIM = 300
DATA_DIR = "/kaggle/working/datasets"
EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")
EMBEDDING_MODEL = "glove-wiki-gigaword-300"
E = build_embedding_matrix(text_sequences, word2idx, EMBEDDING_DIM, EMBEDDING_NUMPY_FILE)
print("Embedding matrix:", E.shape)

# %% [code]
# model definition
conv_num_filters = 256
conv_kernel_size = 3
run_mode = "SCRATCH"
model = SpamClassifierModel(vocab_size, EMBEDDING_DIM, max_seqlen, conv_num_filters, conv_kernel_size, NUM_CLASSES, run_mode, E)
model.build(input_shape=(None, max_seqlen))

# %% [code]
# compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# %% [code]
NUM_EPOCHS = 3
# data distribution is 4827 ham and 747 spam (total 5574), which
# works out to approx 87% ham and 13% spam, so we take reciprocals
# and this works out to being each spam (1) item as being
# approximately 8 times as important as each ham (0) message.
CLASS_WEIGHTS = { 0: 1, 1: 8 }
tf.random.set_seed(42)



# train model
model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=val_dataset,class_weight=CLASS_WEIGHTS)
# evaluate against test set
labels, predictions = [], []
for Xtest, Ytest in test_dataset:
    Ytest_ = model.predict_on_batch(Xtest)
    ytest = np.argmax(Ytest, axis=1)
    ytest_ = np.argmax(Ytest_, axis=1)
    labels.extend(ytest.tolist())
    predictions.extend(ytest.tolist())

# %% [code]
print("test accuracy: {:.3f}".format(accuracy_score(labels,predictions)))
print("confusion matrix")
print(confusion_matrix(labels, predictions))