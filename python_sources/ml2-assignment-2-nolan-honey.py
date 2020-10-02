import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')

target=train_df['cuisine']
train=train_df.drop('cuisine',axis=1)

test=test_df
target.head()
t=Tokenizer()

t.fit_on_texts(train['ingredients'])
train_encoded=t.texts_to_matrix(train['ingredients'],mode='tfidf')
cuisines=train_df['cuisine'].unique()
label2index={cuisine:i for i,cuisine in enumerate(cuisines)}
y=[]

for item in target:
    if item in label2index.keys():
        y.append(label2index[item])
y_encoded=to_categorical(y,20)

def build_model():
    model=Sequential()
    model.add(Dense(128,input_shape=[train_encoded.shape[1], ],activation='relu',name='hidden_1'))
    model.add(Dense(128))
    model.add(Dropout(0.4, name='dropout_1'))

    
    model.add(Dense(20,name='output'))
    
    model.compile(optimizer='adam',
                  loss='categorical_hinge',
                  metrics=['accuracy']
                )
    
    return model
X_train,X_val,y_train,y_val=train_test_split(train_encoded,y_encoded,test_size=0.2,random_state=22)
lasso = Lasso(alpha=0.0004)
model = lasso
model=build_model()
model.summary()
monitor=[
    EarlyStopping(monitor='val_loss',patience=40,verbose=0.005),
    ModelCheckpoint('best-model-0.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
]



model.fit(X_train,y_train,
         validation_data=(X_val,y_val),
         epochs=512,
         callbacks=monitor,
         batch_size=512)

test_encoded=t.texts_to_matrix(test_df['ingredients'],mode='tfidf')
test_encoded.shape
model.load_weights('best-model-0.h5')
y_pred=model.predict(test_encoded).argmax(axis=1)

results=[]

for i in y_pred:
    for k,v in label2index.items():
        if v==i:
            results.append(k)

results[:10]
submission=pd.DataFrame(list(zip(test_df['id'],results)),columns=['id','cuisine'])
submission.to_csv('submission.csv',header=True,index=False)
submission=pd.read_csv('submission.csv')
submission.head()