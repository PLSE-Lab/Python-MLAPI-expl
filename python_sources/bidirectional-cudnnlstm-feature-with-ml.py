#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

import keras
import tensorflow as tf
from keras.models import Sequential,load_model,Model
from keras.optimizers import *
from keras.utils import to_categorical
from keras.layers import *
from keras.callbacks import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
sess = tf.Session()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)


# In[ ]:


train_x = pd.read_csv("../input/X_train.csv")
train_y = pd.read_csv("../input/y_train.csv")
test = pd.read_csv("../input/X_test.csv")


# In[ ]:


### column name and shape

print("train_x column name ---- \n",train_x.columns)
print("train_y column name ---- \n",train_y.columns)
print("train_x shape ---- \n",train_x.shape)
print("train_y shape ---- \n",train_y.shape)


# In[ ]:


### train_x head
train_x.head()


# In[ ]:


train_x.shape


# In[ ]:


### train_y head

train_y.head()


# In[ ]:


### check the traget variable
train_y.groupby('surface')['surface'].count()


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(train_y.surface)
train_y['surface'] = le.transform(train_y.surface)


# In[ ]:


train_label = to_categorical(train_y['surface'])
train_label.shape


# In[ ]:


def feature_extraction(raw_frame):
    raw_frame['orientation'] = raw_frame['orientation_X'] + raw_frame['orientation_Y'] + raw_frame['orientation_Z']+ raw_frame['orientation_W']
    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']
    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']
    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']
    raw_frame['velocity_linear_acceleration'] = raw_frame['linear_acceleration'] * raw_frame['angular_velocity']
    return raw_frame


# In[ ]:


train_df = feature_extraction(train_x)
test_df = feature_extraction(test)


# In[ ]:


train_df = train_df.drop(['series_id', 'row_id'], axis=1)
test_df = test_df.drop(['series_id', 'row_id'], axis=1)


# In[ ]:


print("train shape",train_df.shape)
print("test shape", test_df.shape)


# In[ ]:


cols_normalize = train_df.columns
min_max_scaler = preprocessing.MinMaxScaler()
train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=test_df.index)


# In[ ]:


seq_cols = train_df.columns
def gen_sequence(df,num_elements,seq_cols):
    
    data_matrix = df[seq_cols].values
    
    for start, stop in zip(range(0, num_elements + 128,128), range(128, num_elements + 128,128)):
        yield data_matrix[start:stop, :]


# In[ ]:


train_df = list(gen_sequence(train_df,train_df.shape[0],seq_cols))
train_df = np.array(train_df)
test_df = list(gen_sequence(test_df,test_df.shape[0],seq_cols))
test_df = np.array(test_df)


# In[ ]:


print("train shape",train_df.shape)
print("test shape", test_df.shape)


# In[ ]:


train_x,val_x,train_y,val_y = train_test_split(train_df, train_label, test_size = 0.30, random_state=14)
train_x.shape,val_x.shape,train_y.shape,val_y.shape


# In[ ]:


# train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],train_x.shape[2], 1))
# val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1],val_x.shape[2], 1))


# In[ ]:


## https://www.kaggle.com/kabure/titanic-eda-keras-nn-pipelines
## Creating the model
model = Sequential()

# Inputing the first layer with input dimensions
model.add(Dense(16, 
                activation='relu',  
                input_shape=(128, 16),
                kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01)))
#model.add(BatchNormalization())
# Adding an Dropout layer to previne from overfitting
model.add(Dropout(0.50))

#adding second hidden layer 
model.add(Dense(10,
                kernel_initializer='uniform',
                activation='relu', activity_regularizer=regularizers.l1(0.01)))
#model.add(layers.MaxPooling1D())
# Adding another Dropout layer
model.add(Dropout(0.50))
model.add(layers.Flatten())

# adding the output layer that is binary [0,1]
model.add(Dense(9, activation='softmax'))

#Visualizing the model
model.summary()

sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compiling our model
model.compile(optimizer = sgd, 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint('cnn.hdf', save_best_only=True, 
                               monitor='val_loss', mode='min')


# In[ ]:


from keras.models import Sequential,Model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def make_model():
    inp = Input(shape=(128, 16))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
    x = Attention(128)(x)
    # A intermediate full connected (Dense) can help to deal with nonlinears outputs
    x = Dense(64, activation="relu")(x)
    x = Dense(9, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = make_model()
model.summary()


# In[ ]:


# history = model.fit(train_x, train_y,
#                     batch_size=32,
#                     epochs=50,
#                     verbose=1,
#                     validation_data=(val_x, val_y))


# In[ ]:


model_feat = Model(inputs=model.input,outputs=model.get_layer('attention_1').output)


# In[ ]:


train_feature = model_feat.predict(train_x)
val_feature = model_feat.predict(val_x)
test_feature = model_feat.predict(test_df)
train_feature.shape,val_feature.shape


# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

para={'boosting_type': 'gbdt',
 'colsample_bytree': 0.85,
 'learning_rate': 0.1,
 'max_bin': 512,
 'max_depth': -1,
 'metric': 'multi_error',
 'min_child_samples': 8,
 'min_child_weight': 1,
 'min_split_gain': 0.5,
 'nthread': 3,
 'num_class': 9,
 'num_leaves': 31,
 'objective': 'multiclass',
 'reg_alpha': 0.8,
 'reg_lambda': 1.2,
 'scale_pos_weight': 1,
 'subsample': 0.7,
 'subsample_for_bin': 200,
 'subsample_freq': 1}

Classifier = [
    
        LGBMClassifier(),
        LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
        KNeighborsClassifier(),
        SVC(kernel="rbf", C=0.025, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=500,max_depth=20, min_samples_split=5,
                             class_weight='balanced'),
        AdaBoostClassifier(),
        GaussianNB(),
]


# In[ ]:


Accuracy=[]
Model=[]

for classifier in Classifier:
    try:
        
        fit = classifier.fit(train_feature,np.argmax(train_y,axis=1))
        pred = fit.predict(val_feature)
    except Exception:
        fit = classifier.fit(train_feature,np.argmax(train_y,axis=1))
        pred = fit.predict(val_feature)
        
        
    score = accuracy_score(np.argmax(val_y,axis=1), pred)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(score))
    


# In[ ]:





# In[ ]:




