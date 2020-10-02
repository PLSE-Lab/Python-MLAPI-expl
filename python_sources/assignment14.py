#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install tensorflow==1.14


# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import sklearn.metrics as sklm
from numpy import array,asarray,zeros
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Flatten,Embedding,LSTM,concatenate,Dropout,Lambda,Reshape,BatchNormalization
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
import random
random.seed(11)


# In[ ]:


tf.__version__


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


data  = pd.read_csv('../input/lstmdata/preprocessed_data.csv')
#data = data.head(2000)


# In[ ]:


from sklearn.metrics import roc_auc_score
def costom_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.000


# In[ ]:


def auc(y_true, y_pred):
    return tf.py_func(costom_auc,(y_true, y_pred), tf.double)


# In[ ]:


def document_padding (train,cv,test):
	import pickle
	total_text = train

	t = Tokenizer()
	t.fit_on_texts(total_text)
	vocab_size = len(t.word_index) + 1
	# integer encode the documents
	encoded_docs = t.texts_to_sequences(total_text)
	max_length = 500
	padded_docs_train = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

	
	# load the whole embedding into memory
	embeddings_index = dict()
	with open('../input/lstmdata1/glove_vectors', 'rb') as f:
			embeddings_index = pickle.load(f)

	print('Loaded %s word vectors.' % len(embeddings_index))
	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, 300))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	#------------------------------------ CV -----------------------------------------
	encoded_docs = t.texts_to_sequences(cv)
	padded_docs_cv = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

	#----------------------------------- Test -----------------------------------------------

	encoded_docs = t.texts_to_sequences(test)
	padded_docs_test = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
 
	return (padded_docs_train,padded_docs_cv,padded_docs_test,embedding_matrix,vocab_size)


# In[ ]:


from sklearn.model_selection import train_test_split

_x,test_x,_y,test_y = train_test_split(data,data.project_is_approved, test_size = 0.2, stratify = data.project_is_approved)
train_x,cv_x,train_y,cv_y = train_test_split(_x,_y,test_size = 0.2,stratify = _y)


# In[ ]:


from keras.utils import to_categorical
train_y = to_categorical(train_y)
cv_y = to_categorical(cv_y)
test_y = to_categorical(test_y)


# In[ ]:


# enc = OneHotEncoder()
# train_y = enc.fit_transform(train_y.to_numpy().reshape(-1,1))
# cv_y = enc.transform(cv_y.to_numpy().reshape(-1,1))
# test_y = enc.transform(test_y.to_numpy().reshape(-1,1))


# In[ ]:


for i in [train_x,cv_x,test_x,train_y,cv_y,test_y]:
    print(i.shape)


# In[ ]:


padded_docs_train,padded_docs_cv,padded_docs_test,embedding_matrix,vocab_size = document_padding (train_x.essay,cv_x.essay,test_x.essay)


# In[ ]:


# LSTM layer for text
max_length = 500
text_input = Input(shape = (max_length,),name = ('text_input'))
text_embedded_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=500, trainable=False)(text_input)
text_lstm_layer = LSTM(32,return_sequences=True)(text_embedded_layer)


# In[ ]:


def combine_numrical_input (layer):
    return K.variable(np.hstack((layer[0],layer[1])))


# In[ ]:


price_input = Input(shape=(1,), name=('price_input'))
no_projects = Input(shape=(1,), name=('no_projects'))
numrical_input = concatenate([price_input,no_projects])
numrical_output = Dense(8)(numrical_input)


# In[ ]:


pd.options.mode.chained_assignment = None

encoder = LabelEncoder()
for i in ['school_state', 'teacher_prefix', 'project_grade_category','clean_categories', 'clean_subcategories']:
    encoder.fit(pd.concat([train_x[i],test_x[i],cv_x[i]],ignore_index=True))
    train_x[i] = encoder.transform(train_x[i])
    test_x[i] = encoder.transform(test_x[i])
    cv_x[i] = encoder.transform(cv_x[i])


# In[ ]:


categorical_embedding = []
categorical_input = {}
for i in ['school_state', 'teacher_prefix', 'project_grade_category','clean_categories', 'clean_subcategories']:
                          categorical_input[i] = Input(shape=(1,), dtype='int32', name=('input_{0}'.format(i)))
                          no_of_unique_cat  = 400
                          embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
                          embedding_size = int(embedding_size)
                          vocab  = no_of_unique_cat+1
                          categorical_embedding.append(Embedding(vocab ,embedding_size, input_length = 1)(categorical_input[i]))


# In[ ]:


def flattening_and_concating (layer_map):
    #Flattening and concating layers
    flatten_layers = []
    for embedded_layer in layer_map:
        flatten_layers.append(Flatten()(embedded_layer))
    return concatenate(flatten_layers)


# In[ ]:


text_lstm_layer


# In[ ]:


layer = Lambda(flattening_and_concating)
flatten1 = layer(categorical_embedding)
flatten2 = Flatten()(text_lstm_layer)
#flatten3 = Flatten()(numrical_output)
everything_is_flatten_now = concatenate([flatten1,flatten2,numrical_output])


# In[ ]:


categorical_embedding


# In[ ]:


# Dense layers
do = Dropout(0.2)(everything_is_flatten_now)
dense1 = Dense(128,activation = 'relu')(do)
do1 = Dropout(0.2)(dense1)
#dense1_1 = BatchNormalization()(do1)
dense2 = Dense(64,activation = 'relu')(do1)
do2 = Dropout(0.2)(do1)
#dense1_2 = BatchNormalization()(dense2)
dense3 = Dense(48,activation = 'relu')(do2)
do3 = Dropout(0.2)(dense3)
output = Dense(2, activation='softmax',name='output')(do3)


# In[ ]:


input_a = [text_input,price_input,no_projects]
input_a.extend(list(categorical_input.values()))
input_a


# In[ ]:


validation_set_x = {'price_input':np.array(cv_x.price),'no_projects':np.array(cv_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(cv_x.school_state),
 'input_teacher_prefix':np.array(cv_x.teacher_prefix),'input_project_grade_category':np.array(cv_x.project_grade_category),'input_clean_categories':np.array(cv_x.clean_categories),'input_clean_subcategories':np.array(cv_x.clean_subcategories),'text_input':padded_docs_cv}
validation_set_y = {'output' : cv_y}
train_set_x = {'price_input':np.array(train_x.price),'no_projects':np.array(train_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(train_x.school_state),
 'input_teacher_prefix':np.array(train_x.teacher_prefix),'input_project_grade_category':np.array(train_x.project_grade_category),'input_clean_categories':np.array(train_x.clean_categories),'input_clean_subcategories':np.array(train_x.clean_subcategories),'text_input':padded_docs_train}
train_set_y = {'output' : train_y}
test_set_x = {'price_input':np.array(test_x.price),'no_projects':np.array(test_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(test_x.school_state),
 'input_teacher_prefix':np.array(test_x.teacher_prefix),'input_project_grade_category':np.array(test_x.project_grade_category),'input_clean_categories':np.array(test_x.clean_categories),'input_clean_subcategories':np.array(test_x.clean_subcategories),'text_input':padded_docs_test}
test_set_y = {'output' : test_y}


# In[ ]:


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


model = Model(inputs=input_a, outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[auc])


# In[ ]:


model.summary()


# In[ ]:


import os,datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# In[ ]:


# model.fit(train_set_x,train_y,epochs=12,validation_data = (validation_set_x,cv_y),callbacks=[checkpoint,tensorboard_callback])


# In[ ]:


# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[auc])


# In[ ]:


model.evaluate(test_set_x, test_set_y)


# In[ ]:


os.listdir()


# In[ ]:


# Load the extension and start TensorBoard

get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ##Modle 2

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_essay = vectorizer.fit_transform(train_x.essay)


# In[ ]:


from tqdm import tqdm
word_dict = {}
for line in tqdm(train_x.essay):
    for word in line.split():
        if word not in word_dict:
              word_dict[word] = 1
print('\nNumber of unique words : ',len(word_dict))


# In[ ]:


x = vectorizer.idf_
x.sort()
x


# In[ ]:


dict_id = vectorizer.vocabulary_
idf_list = vectorizer.idf_
train_rectified_essay = []
cv_rectified_essay = []
test_rectified_essay = []
a = 0
b = 100
for line in tqdm(train_x.essay):
    tmp = []
    for word in line.split():
        if word in dict_id:
            word_id = dict_id[word]
            idf_value = idf_list[word_id]
        if  idf_value > a and idf_value < b:
            tmp.append(word)
    train_rectified_essay.append(' '.join(tmp))

#------------------------------------------- CV ------------------------------------------
for line in tqdm(cv_x.essay):
    tmp = []
    for word in line.split():
        if word in dict_id:
            word_id = dict_id[word]
            idf_value = idf_list[word_id]
        if  idf_value > a and idf_value < b:
            tmp.append(word)
    cv_rectified_essay.append(' '.join(tmp))

#---------------------------------------- Test -------------------------------------------------
for line in tqdm(test_x.essay):
    tmp = []
    for word in line.split():
        if word in dict_id:
            word_id = dict_id[word]
            idf_value = idf_list[word_id]
    if  idf_value > a and idf_value < b:
            tmp.append(word)
    test_rectified_essay.append(' '.join(tmp))


# In[ ]:


word_dict = {}
for line in tqdm(train_rectified_essay):
    for word in line.split():
        if word not in word_dict:
            word_dict[word] = 1
print('\nNumber of unique words : ',len(word_dict))


# In[ ]:


padded_docs_train,padded_docs_cv,padded_docs_test,embedding_matrix,vocab_size = document_padding(train_rectified_essay,cv_rectified_essay,test_rectified_essay)


# In[ ]:


validation_set_x = {'price_input':np.array(cv_x.price),'no_projects':np.array(cv_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(cv_x.school_state),
 'input_teacher_prefix':np.array(cv_x.teacher_prefix),'input_project_grade_category':np.array(cv_x.project_grade_category),'input_clean_categories':np.array(cv_x.clean_categories),'input_clean_subcategories':np.array(cv_x.clean_subcategories),'text_input':padded_docs_cv}
validation_set_y = {'output' : cv_y}
train_set_x = {'price_input':np.array(train_x.price),'no_projects':np.array(train_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(train_x.school_state),
 'input_teacher_prefix':np.array(train_x.teacher_prefix),'input_project_grade_category':np.array(train_x.project_grade_category),'input_clean_categories':np.array(train_x.clean_categories),'input_clean_subcategories':np.array(train_x.clean_subcategories),'text_input':padded_docs_train}
train_set_y = {'output' : train_y}
test_set_x = {'price_input':np.array(test_x.price),'no_projects':np.array(test_x.teacher_number_of_previously_posted_projects),'input_school_state':np.array(test_x.school_state),
 'input_teacher_prefix':np.array(test_x.teacher_prefix),'input_project_grade_category':np.array(test_x.project_grade_category),'input_clean_categories':np.array(test_x.clean_categories),'input_clean_subcategories':np.array(test_x.clean_subcategories),'text_input':padded_docs_test}
test_set_y = {'output' : test_y}


# In[ ]:


filepath="Model1.best.hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


model1 = Model(inputs=input_a, outputs=[output])
model1.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=[auc])


# In[ ]:


model1.summary()


# In[ ]:


logdir = os.path.join("logs1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback1 = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# In[ ]:


# model1.fit(train_set_x,train_y,epochs=3,validation_data = (validation_set_x,cv_y),callbacks=[checkpoint1,tensorboard_callback1])


# In[ ]:


# load weights
model1.load_weights("Model1.best.hdf5")
# Compile model (required to make predictions)
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[auc])


# In[ ]:


model1.evaluate(test_set_x, test_set_y)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs1')


# ##Model 3

# In[ ]:


from sklearn.model_selection import train_test_split

_x,test_x,_y,test_y = train_test_split(data,data.project_is_approved, test_size = 0.2, stratify = data.project_is_approved)
train_x,cv_x,train_y,cv_y = train_test_split(_x,_y,test_size = 0.2,stratify = _y)


# In[ ]:


from keras.utils import to_categorical
train_y = to_categorical(train_y)
cv_y = to_categorical(cv_y)
test_y = to_categorical(test_y)


# In[ ]:


# enc = OneHotEncoder()
# train_y = enc.fit_transform(train_y.to_numpy().reshape(-1,1))
# cv_y = enc.transform(cv_y.to_numpy().reshape(-1,1))
# test_y = enc.transform(test_y.to_numpy().reshape(-1,1))


# In[ ]:


padded_docs_train,padded_docs_cv,padded_docs_test,embedding_matrix,vocab_size = document_padding (train_x.essay,cv_x.essay,test_x.essay)


# In[ ]:


# LSTM layer for text
max_length = 500
text_input_model3 = Input(shape = (max_length,),name = ('text_input_model3'))
text_embedded_layer_model3 = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=500, trainable=False)(text_input_model3)
text_lstm_layer_model3 = LSTM(32,return_sequences=True)(text_embedded_layer_model3)
#text_lstm_layer_model3 = Reshape((32,1))(text_lstm_layer_model3)


# In[ ]:


text_lstm_layer_model3


# In[ ]:


#Other inputs
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
tmp_train = []
tmp_cv = []
tmp_test = []
for i in ['school_state', 'teacher_prefix', 'project_grade_category','clean_categories', 'clean_subcategories']:
    tmp_train.append(enc.fit_transform(train_x[[i]]))
    tmp_cv.append(enc.transform(cv_x[[i]]))
    tmp_test.append(enc.transform(test_x[[i]]))


# In[ ]:


from scipy.sparse import hstack
other_data_train = hstack((tmp_train))
other_data_cv = hstack((tmp_cv))
other_data_test = hstack((tmp_test))


# In[ ]:


categorical_input_model3 = Input(batch_shape=(None,other_data_train.shape[1],1), name=('categorical_input_model3'))
price_input_m3 = Input(batch_shape=(None,1,1), name=('price_input_m3'))
no_projects_m3 = Input(batch_shape=(None,1,1), name=('no_projects_m3'))
other_input = concatenate([price_input_m3,no_projects_m3],axis=1)


# In[ ]:


#other_input = Flatten()(other_input)


# In[ ]:


input_a_m3 = [text_input_model3,categorical_input_model3,price_input_m3,no_projects_m3]


# In[ ]:


cnn1 = Conv1D(6, 2, activation='relu')(other_input)
cnn2 = Conv1D(10, 1, activation='relu')(cnn1)


# In[ ]:


def flattening_and_concating_model3 (layers):
    flatten_layers = []
    flatten_layers.append(Flatten()(layers[0]))
    flatten_layers.append(Flatten()(layers[1]))
    return concatenate(flatten_layers)


# In[ ]:


flatten1 = Flatten()(cnn2)
flatten2 = Flatten()(text_lstm_layer_model3)


# In[ ]:


#layer_model3 = Lambda(flattening_and_concating_model3)
everything_is_flatten_now_model3 = concatenate(list([flatten1,flatten2]))


# In[ ]:


everything_is_flatten_now_model3


# In[ ]:


# Dense layers
do_m3 = Dropout(0.2)(everything_is_flatten_now_model3)
dense1_m3 = Dense(128,activation = 'relu')(do_m3)
dense1_1_m3 = Dropout(0.2)(dense1_m3)
dense2_m3 = Dense(64,activation = 'relu')(dense1_1_m3)
dense2_1_m3 = Dropout(0.2)(dense2_m3)
dense3_m3 = Dense(32,activation = 'relu')(dense2_1_m3)
do4_m3 = Dropout(0.2)(dense3_m3)
output_m3 = Dense(2, activation='softmax',name='output_m3')(do4_m3)


# In[ ]:


from numpy import newaxis
validation_set_x_m3 = {'price_input_m3':np.array(cv_x.price)[...,newaxis,newaxis],'no_projects_m3':np.array(cv_x.teacher_number_of_previously_posted_projects)[...,newaxis,newaxis],'text_input_model3':padded_docs_cv,'categorical_input_model3' : other_data_cv.toarray()[...,newaxis]}
#validation_set_y = {'output_m3' : cv_y.toarray()}
train_set_x_m3 = {'price_input_m3':np.array(train_x.price)[...,newaxis,newaxis],'no_projects_m3':np.array(train_x.teacher_number_of_previously_posted_projects)[...,newaxis,newaxis],'text_input_model3':padded_docs_train,'categorical_input_model3' : other_data_train.toarray()[...,newaxis]}
#train_set_y = {'output_m3' : train_y.toarray()}
test_set_x_m3 = {'price_input_m3':np.array(test_x.price)[...,newaxis,newaxis],'no_projects_m3':np.array(test_x.teacher_number_of_previously_posted_projects)[...,newaxis,newaxis],'text_input_model3':padded_docs_test,'categorical_input_model3' : other_data_test.toarray()[...,newaxis]}
#test_set_y = {'output_m3' : test_y.toarray()}


# In[ ]:


filepath="Model3.best.hdf5"
checkpoint2 = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


model3 = Model(inputs=input_a_m3, outputs=[output_m3])
model3.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=[auc])


# In[ ]:


model3.summary()


# In[ ]:


import os,datetime
logdir = os.path.join("logs2", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# In[ ]:


model3.fit(train_set_x_m3,train_y,epochs=12,validation_data = (validation_set_x_m3,cv_y),callbacks=[checkpoint2,tensorboard_callback2])


# In[ ]:


# load weights
model3.load_weights("Model3.best.hdf5")
# Compile model (required to make predictions)
model3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[auc])


# In[ ]:


model3.evaluate(test_set_x_m3, test_y)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs2')


# In[ ]:


# Please compare all your models using Prettytable library
from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Model", 'Test Loss','Test AUC']

x.add_row(["Model 1",0.42710373645516225, 0.6089656684462287])
x.add_row(["Model 2",1.6488581538081033, 0.5914316826526952])
x.add_row(["Model 3",0.8623193602976592, 0.5837973926053786])

print(x)

