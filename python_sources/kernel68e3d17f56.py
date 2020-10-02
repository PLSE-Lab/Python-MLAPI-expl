# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D,SpatialDropout1D
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Concatenate, Input, Dense

#IMPORT DATA
train = pd.read_csv('/kaggle/input/radix-challenge/train.csv')
test = pd.read_csv('/kaggle/input/radix-challenge/test.csv')

#BINARIZE GENRE LABEL
nltk.download('punkt')
nltk.download('stopwords')
genres_label = []
for g in train['genres'].tolist():
    genres_label.append(tuple(word_tokenize(g)))
    
genres = []
for g in train['genres'].tolist():
    genres = genres + word_tokenize(g)
genres = np.asarray(list(set(genres)))

mlb = MultiLabelBinarizer()
mlb.fit_transform(genres.reshape(-1, 1))
train_label= mlb.transform(genres_label)

#TEXT PREPROCESSING
nltk.download('wordnet')
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
def text_processing(doc):
    doc = word_tokenize(doc.translate(str.maketrans('', '', string.punctuation)))
    doc = [w for w in doc if not w in stop_words]
    doc = [lemmatizer.lemmatize(w) for w in doc]
    doc = [ps.stem(w) for w in doc]
    return doc

train_synopsis = train['synopsis'].tolist()
train_synopsis = [text_processing(doc) for doc in train_synopsis]
test_synopsis = test['synopsis'].tolist()
test_synopsis = [text_processing(doc) for doc in test_synopsis]

#BUILS DOC2VEC MODEL
VECTOR_SIZE=250
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_synopsis)]
model = Doc2Vec(documents, vector_size=VECTOR_SIZE, window=10, min_count=1, workers=4, epochs = 600)

train_doc2vec = []
for doc in train_synopsis:
    train_doc2vec.append(model.infer_vector(doc))

test_doc2vec = []
for doc in test_synopsis:
    test_doc2vec.append(model.infer_vector(doc))
    
#BUILD LSTM MODEL

# Convolution
kernel_size = 3
filters = 180
pool_size = 4

# LSTM
epochs = 500
batch_size = 50
TIMESTEPS = 1

# First model
input_1 = Input((TIMESTEPS,VECTOR_SIZE))
input_2 = Input((TIMESTEPS,VECTOR_SIZE))
cnn_1 = Conv1D(filters,3,strides=3,padding='same',input_shape=(None,VECTOR_SIZE),
             activation='sigmoid')(input_1)
maxpool_1 = MaxPooling1D(pool_size=pool_size, padding='same')(cnn_1)

# Second model
cnn_2 = Conv1D(filters,3,strides=1,padding='same',input_shape=(None,VECTOR_SIZE),
             activation='sigmoid')(input_2)
maxpool_2 = MaxPooling1D(pool_size=pool_size, padding='same')(cnn_2)

# Concatenate both
merged = Concatenate()([maxpool_1, maxpool_2])
lstm = LSTM(100,dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(merged)
output_layer = Dense(19,activation='sigmoid')(lstm)

model = Model(input=[input_1,input_2], output=output_layer)
sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['categorical_accuracy'])

history_submission = model.fit([X_train,X_train], np.asarray(train_label), epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',patience=3,min_delta=0.0001)])

#SUBMISSION FUNCTION
test_id = test['movie_id'].tolist()
test_genres = []

def index2label(index):
    encoder = np.zeros((1,19))
    encoder[0][index]=1
    label = mlb.inverse_transform(encoder)[0][0]
    return label
def submissionCSV(test_mat):
    label_list=[]
    for i in range(len(test_id)):
        test_input=test_mat[i].reshape(1, test_mat[i].shape[0],test_mat[i].shape[1])
        pred_proba = model.predict([test_input,test_input])[0]
        pred_5 = sorted(range(len(pred_proba)), reverse = True, key=lambda k: pred_proba[k])
        labels = index2label(pred_5[0])+' '+index2label(pred_5[1])+' '+index2label(pred_5[2])+' '+index2label(pred_5[3])+' '+index2label(pred_5[4])
        label_list.append(labels)
    submission = pd.DataFrame(
        {'movie_id': test_id,
         'predicted_genres': label_list})
    return submission

submission = submissionCSV(X_test)
submission.head()
#submission.to_csv(r'submission.csv',index=False)