#!/usr/bin/env python
# coding: utf-8

# # **CONTEXT**
# 
# This is a protein data set retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB).
# 
# The PDB archive is a repository of atomic coordinates and other information describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to each other in the molecule. They then deposit this information, which is then annotated and publicly released into the archive by the wwPDB.
# 
# The constantly-growing PDB is a reflection of the research that is happening in laboratories across the world. This can make it both exciting and challenging to use the database in research and education. Structures are available for many of the proteins and nucleic acids involved in the central processes of life, so you can go to the PDB archive to find structures for ribosomes, oncogenes, drug targets, and even whole viruses. However, it can be a challenge to find the information that you need, since the PDB archives so many different structures. You will often find multiple structures for a given molecule, or partial structures, or structures that have been modified or inactivated from their native form.

# # **WORK DONE**
# This notebooks shows how to classify protein families soley based on their sequence of aminoacids. This work is based on the current success of deep learning models in natural language processing (NLP) and assumes the proteins sequences can be viewed as as a language. Please note, that there are notable search engines such as BLAST for this task.

# In[ ]:


# Importing Libraries and DATA
import numpy as np
import pandas as pd
df_dup = pd.read_csv('../input/pdb_data_no_dups.csv')
df_seq = pd.read_csv('../input/pdb_data_seq.csv')


# In[ ]:


# Merge the two Data set together
df = df_dup.merge(df_seq,how='inner',on='structureId')

# Drop rows with missing labels
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]

# select proteins
df = df[df.macromoleculeType_x == 'Protein']
df.reset_index()
df.head()


# # Visualize and preprocess dataset
# To build a model with appropriate number of instances per class. Let us only focus on the ten most common classes.

# In[ ]:


from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import matplotlib.pyplot as plt
from collections import Counter

cnt = Counter(df.classification)
# select only 10 most common classes!
top_classes = 10
tmp = np.array([[c[0], c[1]] for c in cnt.most_common()[:top_classes]])
[classes, counts] = tmp[:,0], tmp[:,1].astype(int)

N = sum(counts)
plt.bar(range(len(classes)), counts/float(N))
plt.xticks(range(len(classes)), classes, rotation='vertical')
#plt.xlabel('protein class')
plt.ylabel('frequency')
plt.show()


# # Transform labels

# In[ ]:


df = df[[c in classes for c in df.classification]]
seqs = df.sequence.values

# Transform labels into one-hot
lb = LabelBinarizer()
Y = lb.fit_transform(df.classification)

lengths = [len(s) for s in seqs]
plt.hist(lengths, bins=100, normed=True)
plt.xlabel('sequence length')
plt.ylabel('frequency')
plt.show()


# # Further preprocessing of sequences with keras
# 1. ** Tokenizer**: translates every character of the sequence into a number
# 2. **pad_sequences:** ensures that every sequence has the same length (max_length). 
# 3. **train_test_split:** from sklearn splits the data into training and testing samples.

# In[ ]:


from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

max_length = 512

#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
#represent input data as word rank number sequences
X = tokenizer.texts_to_sequences(seqs)
X = sequence.pad_sequences(X, maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.7)
X_train.shape, X_test.shape


# # Let's build up the keras model and get it on fire

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

embedding_dim = 16

# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.index_docs)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(top_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

print(model.summary())


# In[ ]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=lb.classes_, normalize=True)
plt.show()


# In[ ]:




