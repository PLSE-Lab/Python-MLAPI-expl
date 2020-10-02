#!/usr/bin/env python
# coding: utf-8

# # Import Dataset, drop NaN's, select Proteins
# This notebooks shows how to recogniuze protein families soley based on the sequence of aminoacids. Please note, that there are notable search engines such as BLAST for this task.
# 
# ## Preprocessing and visualization of dataset
# preprocessing of the data:
# 1. merge on *structureId*
# 2. drop rows without labels
# 3. drop rows without sequence
# 4. select proteins
# 
# **Ideally: **For comparison I also decided to focus only on those classes where the number of instances is greater than 1000 (as in [this kernel of Akil](https://www.kaggle.com/abharg16/predicting-protein-classification/code)) which corresponds to the 43 most common classes. 
# 
# **But:** one hour on 4 CPU is not sufficient for such big datasets, instead only 10 most common classes are considered.
# 
# ## Important disclaimer:
# As there are 

# In[ ]:


import os

os.listdir()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Merge the two Data set together
df1 = pd.read_csv('../input/protein-data-set/pdb_data_no_dups.csv')
df2 = pd.read_csv('../input/protein-data-set/pdb_data_seq.csv')
df = df1.merge(df2, how='inner', on='structureId')
# Drop rows with missing labels
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]
# select proteins
df = df[df.macromoleculeType_x == 'Protein']
df.reset_index()
df.shape


# In[ ]:


import innvestigate


# # Disclaimer
# ### As there are multiple chain_ids per structure_id, train test split has to be done on unique structure_ids in order to avoid redunancy bias!

# In[ ]:


df.structureId.value_counts()[:30].plot(kind='bar')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from collections import Counter

cnt = Counter(df.classification)
top_classes = 42
# sort classes
sorted_classes = cnt.most_common()[:top_classes]
classes = [c[0] for c in sorted_classes]
counts = [c[1] for c in sorted_classes]
print("at least " + str(counts[-1]) + " instances per class")

# apply to dataframe
print(str(df.shape[0]) + " instances before")
df = df[[c in classes for c in df.classification]]
print(str(df.shape[0]) + " instances after")

seqs = df.sequence.values
lengths = [len(s) for s in seqs]

# visualize
fig, axarr = plt.subplots(1,2, figsize=(20,5))
axarr[0].bar(range(len(classes)), counts)
plt.sca(axarr[0])
plt.xticks(range(len(classes)), classes, rotation='vertical')
axarr[0].set_ylabel('frequency')

axarr[1].hist(lengths, bins=100, normed=False)
axarr[1].set_xlabel('sequence length')
axarr[1].set_ylabel('# sequences')
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer

# Transform labels to one-hot
lb = LabelBinarizer()
Y = lb.fit_transform(df.classification)
inv_dic = {i:c for i,c in enumerate(lb.classes_)}

seqs = df.sequence.values
lengths = np.array([len(s) for s in seqs])

plt.figure(figsize=(20,10))
x = []
ys = []
for y in np.unique(Y.argmax(axis=1)):
    sel = lengths[Y.argmax(axis=1) == y]
    x.append(sel)
    ys.append(str(inv_dic[y]))
plt.hist(x, bins=100, label=ys, histtype='bar', stacked=True)
plt.legend()
plt.show()


# In[ ]:


from keras.preprocessing.text import Tokenizer

#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
tokenizer.word_index


# In[ ]:


from keras.utils import to_categorical
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split

# maximum length of sequence, everything afterwards is discarded!
max_length = 1024

def unsupervised_generator(dataframe, seq_tokenizer, batch_size):
    while True:
        ridxs = np.random.choice(len(dataframe), batch_size)
        X = seq_tokenizer.texts_to_sequences(dataframe.iloc[ridxs].sequence.values)
        X = sequence.pad_sequences(X, maxlen=max_length)
        X = to_categorical(X)
        yield X, X
        
def supervised_generator(dataframe, seq_tokenizer, label_tokenizer, batch_size):
    while True:
        ridxs = np.random.choice(len(dataframe), batch_size)
        X = seq_tokenizer.texts_to_sequences(dataframe.iloc[ridxs].sequence.values)
        X = sequence.pad_sequences(X, maxlen=max_length)
        X = to_categorical(X, len(seq_tokenizer.word_index)+1)
        Y = label_tokenizer.transform(dataframe.iloc[ridxs].classification.values)
        yield X, Y

train_df, test_df = train_test_split(df, test_size=.1)
train_df.shape, test_df.shape, df.shape


# # THIS IS WRONG! INSTEAD DO FOLLOWING:

# In[ ]:


ids = np.array(list(set(df.structureId)))
train_ids, test_ids = train_test_split(ids, test_size=.1)
train_df = df[df.structureId.isin(train_ids)]
test_df = df[df.structureId.isin(test_ids)]
train_ids.shape, test_ids.shape, train_df.shape, test_df.shape, df.shape


# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, Dropout

model = Sequential()
model.add(Convolution1D(1024, 3, strides=2, activation='relu', padding='same', input_shape=(max_length, len(tokenizer.word_index) + 1)))
#model.add(Convolution1D(512, 3, strides=2, activation='relu', padding='same'))
#model.add(Convolution1D(512, 3, activation='relu', padding='same'))
model.add(Convolution1D(512, 3, strides=2, activation='relu', padding='same'))
#model.add(Convolution1D(128, 3, activation='relu', padding='same'))
model.add(Convolution1D(256, 3, strides=2, activation='relu', padding='same'))
#model.add(Convolution1D(64, 3, activation='relu', padding='same'))
model.add(Convolution1D(128, 3, strides=2, activation='relu', padding='same'))
model.add(Convolution1D(64, 3, strides=2, activation='relu', padding='same'))
#model.add(Convolution1D(32, 3, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dropout(.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(top_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
print(model.summary())


# > * 

# In[ ]:


batch_size=32
tr_gen = supervised_generator(train_df, tokenizer, lb, batch_size=batch_size)
te_gen = supervised_generator(test_df, tokenizer, lb, batch_size=batch_size)

model.fit_generator(
    tr_gen,
    validation_data=te_gen,
    steps_per_epoch=len(train_df)//batch_size,
    validation_steps =len(test_df)//batch_size,
    epochs=10
    )


# In[ ]:


model.save('fucking_cnn.h5')


# In[ ]:


import os

os.listdir()


# In[ ]:


import innvestigate


# # Evaluate model
# Evaluation is only done based on accuracy and a confusion matrix which is already implemented in sklearn.
# 

# In[ ]:


#X_train = tokenizer.texts_to_sequences(train_df.sequence.values)
#X_train = sequence.pad_sequences(X_train, maxlen=max_length)
#X_train = to_categorical(X_train, len(tokenizer.word_index)+1)
#y_train = lb.transform(train_df.classification.values)

X_test = tokenizer.texts_to_sequences(test_df.sequence.values)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
X_test = to_categorical(X_test, len(tokenizer.word_index)+1)
y_test = lb.transform(test_df.classification.values)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools

#train_scores = model.predict(X_train)
test_scores = model.predict(X_test)
#train_pred = np.argmax(train_scores, axis=1)
test_pred = np.argmax(test_scores, axis=1)
#Y_train = np.argmax(y_train, axis=1)
Y_test = np.argmax(y_test, axis=1)

#print("train-acc = " + str(accuracy_score(Y_train, train_pred)))
print("test-acc = " + str(accuracy_score(Y_test, test_pred)))

# Compute confusion matrix
cm = confusion_matrix(Y_test, test_pred)

# Plot normalized confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(lb.classes_))
plt.xticks(tick_marks, lb.classes_, rotation=90)
plt.yticks(tick_marks, lb.classes_)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(Y_test, test_pred, target_names=lb.classes_))


# In[ ]:




