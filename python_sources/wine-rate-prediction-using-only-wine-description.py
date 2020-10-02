import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import pickle

# simplify the model
def transform_points_simplified(points):
    if points < 84:
        # under average wines
        return 1  
    elif points >= 84 and points < 88:
        # average wines
        return 2
    elif points >= 88 and points < 92:
        # good wines
        return 3
    elif points >= 92 and points < 96:
        # very good wines
        return 4
    else:
        # excellent wines
        return 5

MAX_NUM_WORDS = 100000
MAX_SEQUENCE_LENGTH = 300

# read wine data from cvs file 
df_wine1 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", usecols = ['description', 'points'])
df_wine1["lens"] = df_wine1.apply(lambda row: len(row.description), axis = 1)
# cleaning dataset
# remove rows with any missing values
df_wine1 = df_wine1.dropna()
# remove duplicates rows
df_wine1 = df_wine1.drop_duplicates('description')
# remove outliers
df_wine1 = df_wine1[np.abs(df_wine1.lens - df_wine1.lens.mean()) <= (3 * df_wine1.lens.std())]
df_wine1 = df_wine1.assign(points_simplified = df_wine1['points'].apply(transform_points_simplified))
# transform descriptions into lower case
df_wine1['description'] = df_wine1['description'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
# remove punctuation
df_wine1['description'] = df_wine1['description'].str.replace('[^\w\s]', '')
# remove of stop words
stop_words = set(stopwords.words('english'))
df_wine1['description'] = df_wine1['description'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
#freq = pd.Series(''.join(df_wine1['description']).split()).value_counts()[-10:]

description = df_wine1['description'].values
points = df_wine1['points_simplified'].values

# split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(description, points, test_size = 0.1)

# prepare tokenizer
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS) # text tokenization utility class
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
# integer encode the documents
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# pad documents to a max length of the document
X_train = pad_sequences(X_train, maxlen = MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen = MAX_SEQUENCE_LENGTH)

# convert y lable
y_train = y_train - 1
y_test = y_test - 1
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
# save the preprocessed training and testing datasets
np.save('X_train', X_train)
np.save('y_train', y_train)
np.save('X_test', X_test)
np.save('y_test', y_test)

# load the whole embedding into memory
embedding_index = dict()
f = open("../input/glove6b100dtxt/glove.6B.100d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embedding_index))

# create a weight matrix for words in training text
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# define model
embedding_vector_length = 100
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, weights = [embedding_matrix], input_length = MAX_SEQUENCE_LENGTH, trainable = False))
#model.add(CuDNNLSTM(50, recurrent_regularizer = l2(0.09)))
model.add(LSTM(256, dropout = 0.25, recurrent_dropout = 0.25))
model.add(Dense(5, activation='softmax'))
rms = optimizers.RMSprop(lr = 0.0009)
model.compile(optimizer = rms, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

# fit and save history
hist = model.fit(X_train, y_train, epochs = 5, validation_data = (X_test, y_test), batch_size = 64, callbacks = callbacks_list)
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)

# save model
model.save('my_model.h5')


        




