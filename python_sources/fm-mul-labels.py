# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
from keras.layers import * #Input, Embedding, Dense,Flatten, merge,Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
import itertools
import keras
from keras.optimizers import *
from keras.regularizers import l2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


train_data = pd.read_csv("../input/rating.csv")

print(train_data.columns)
train_data = train_data.sample(frac=1.0)
train_data['rating'] = train_data['rating']
feat_cols = []
cat_cols =  ['userId','movieId']


from keras.utils import plot_model
def KerasFM(max_features,K1=12,K2=8,solver=Adam(lr=0.01),l2=0.00,l2_fm = 0.00):
    inputs = []
    flatten_layers1=[]
    fm_layers1 = []

    flatten_layers2=[]
    fm_layers2 = []
    #for c in columns:
    for c in max_features.keys():
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]
        embed1 = Embedding(
                        num_c,
                        K1,
                        input_length=1,
                        name = 'embed1_%s'%c,
			dtype='float32',
			embeddings_regularizer=keras.regularizers.l2(1e-5)
                        )(inputs_c)

        flatten1 = Flatten()(embed1)

        inputs.append(inputs_c)
        flatten_layers1.append(flatten1)
        
        embed2 = Embedding(
                        num_c,
                        K2,
                        input_length=1,
                        name = 'embed2_%s'%c,
                        dtype='float32',
                        embeddings_regularizer=keras.regularizers.l2(1e-5)
                        )(inputs_c)

        flatten2 = Flatten()(embed2)
        flatten_layers2.append(flatten2)

    for emb1,emb2 in itertools.combinations(flatten_layers1, 2):
        dot_layer1 = dot([emb1,emb2],axes=-1,normalize=False)
        fm_layers1.append(dot_layer1)

    for emb1,emb2 in itertools.combinations(flatten_layers2, 2):
        dot_layer2 = dot([emb1,emb2],axes=-1,normalize=True)
        fm_layers2.append(dot_layer2)
    #flatten = BatchNormalization(axis=1)(add((fm_layers)))
    flatten1 = dot_layer1
    flatten2 = dot_layer2
    output1 = Dense(1,activation='linear',name='regression_linear_outputs1')(flatten1)
    output2 = Dense(1,activation='sigmoid',name='binary_sigmoid_outputs2')(flatten2)
    model = Model(input=inputs, output=[output1,output2])
    model.compile(optimizer=solver,loss= ['mae','binary_crossentropy'])
    plot_model(model, to_file='fm_cosine_model.png',show_shapes=True)
    #model.summary()
    return model


max_features = train_data[cat_cols].max() + 1
train_data['fake_label'] = train_data['rating']>=3
label_cols = ['rating','fake_label']
train_len = int(len(train_data)*0.95)
X_train, X_valid = train_data[cat_cols][:train_len], train_data[cat_cols][train_len:]
y_train, y_valid = train_data[label_cols][:train_len], train_data[label_cols][train_len:]

train_input = []
valid_input = []
#test_input = []
#print(test_data)
for col in cat_cols:
    train_input.append(X_train[col])
    valid_input.append(X_valid[col])

train_target = []
valid_target = []
print(label_cols)
for label in label_cols:
    train_target.append(y_train[label])
    valid_target.append(y_valid[label])
ck = keras.callbacks.ModelCheckpoint("best.model", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
es = keras.callbacks.EarlyStopping(monitor='val_outputs1_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model = KerasFM(max_features)
model.fit(train_input, train_target, batch_size=100000,nb_epoch=10,validation_data=(valid_input,valid_target),callbacks=[ck,es],verbose=2)

from sklearn.metrics import roc_auc_score
p_valid = model.predict(valid_input)
auc = roc_auc_score(y_valid['fake_label'],p_valid[1])
print("valid auc is %0.6f"%auc)
from sklearn.metrics import *
mae = mean_absolute_error(y_valid['rating'],p_valid[0])
print("valid mae is %0.6f"%mae)