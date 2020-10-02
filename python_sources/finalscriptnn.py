# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:24:01 2017

@author: Paul
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

def categorical2singleclass(label):
    l = []
    for ll in label:
        if ll[0]>ll[1]:
            l.append(0)
        else:
            l.append(1)
    
    del ll
    return l

def categorical_label(y, classes = 2):
    uniques, id_train=np.unique(y,return_inverse=True)
    return np_utils.to_categorical(id_train,classes)

def replace_non_numeric(df):
    df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
    df["Embarked"] = df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    return df

def preprocess_data(features):
    data = []
    
    for fea in features:
        sample = []
        
        if fea[0] == 0:
            sample.extend([1,0])
        else:
            sample.extend([0,1])
        
        if fea[1] == 1:
            sample.extend([1,0,0])
        elif fea[1] == 2:
            sample.extend([0,1,0])
        else:
            sample.extend([0,0,1])
            
        if fea[2] == 0:
            sample.extend([1,0])
        else:
            sample.extend([0,1])
            
        if fea[3]>0 and fea[3]<=10:
            sample.extend([1,0,0,0,0,0,0,0,0,0])
        elif fea[3]>10 and fea[3]<=20:
            sample.extend([0,1,0,0,0,0,0,0,0,0])
        elif fea[3]>20 and fea[3]<=30:
            sample.extend([0,0,1,0,0,0,0,0,0,0])
        elif fea[3]>30 and fea[3]<=40:
            sample.extend([0,0,0,1,0,0,0,0,0,0])
        elif fea[3]>40 and fea[3]<=50:
            sample.extend([0,0,0,0,1,0,0,0,0,0])
        elif fea[3]>50 and fea[3]<=60:
            sample.extend([0,0,0,0,0,1,0,0,0,0])
        elif fea[3]>60 and fea[3]<=70:
            sample.extend([0,0,0,0,0,0,1,0,0,0])
        elif fea[3]>70 and fea[3]<=80:
            sample.extend([0,0,0,0,0,0,0,1,0,0])
        elif fea[3]>80 and fea[3]<=90:
            sample.extend([0,0,0,0,0,0,0,0,1,0])
        else:
            sample.extend([0,0,0,0,0,0,0,0,0,1])
            
        if fea[4] == 0:
            sample.extend([1,0,0,0,0,0,0,0,0,0])
        elif fea[4] == 1:
            sample.extend([0,1,0,0,0,0,0,0,0,0])
        elif fea[4] == 2:
            sample.extend([0,0,1,0,0,0,0,0,0,0])
        elif fea[4] == 3:
            sample.extend([0,0,0,1,0,0,0,0,0,0])
        elif fea[4] == 4:
            sample.extend([0,0,0,0,1,0,0,0,0,0])
        elif fea[4] == 5:
            sample.extend([0,0,0,0,0,1,0,0,0,0])
        elif fea[4] == 6:
            sample.extend([0,0,0,0,0,0,1,0,0,0])
        elif fea[4] == 7:
            sample.extend([0,0,0,0,0,0,0,1,0,0])
        elif fea[4] == 8:
            sample.extend([0,0,0,0,0,0,0,0,1,0])
        else:
            sample.extend([0,0,0,0,0,0,0,0,0,1])
            
        if fea[5] == 0:
            sample.extend([1,0,0,0,0,0,0,0,0,0])
        elif fea[5] == 1:
            sample.extend([0,1,0,0,0,0,0,0,0,0])
        elif fea[5] == 2:
            sample.extend([0,0,1,0,0,0,0,0,0,0])
        elif fea[5] == 3:
            sample.extend([0,0,0,1,0,0,0,0,0,0])
        elif fea[5] == 4:
            sample.extend([0,0,0,0,1,0,0,0,0,0])
        elif fea[5] == 5:
            sample.extend([0,0,0,0,0,1,0,0,0,0])
        elif fea[5] == 6:
            sample.extend([0,0,0,0,0,0,1,0,0,0])
        elif fea[5] == 7:
            sample.extend([0,0,0,0,0,0,0,1,0,0])
        elif fea[5] == 8:
            sample.extend([0,0,0,0,0,0,0,0,1,0])
        else:
            sample.extend([0,0,0,0,0,0,0,0,0,1])
            
        data.append(sample)
        
    del fea, features, sample
    return np.array(data)

def load_train_data(file_name):
    train_df = replace_non_numeric(pd.read_csv(file_name))
    
    labels = train_df["Survived"].values
    features = train_df[["Sex", "Pclass", 'Embarked', 'Age', 'SibSp', 'Parch']].values    
    data = preprocess_data(features)

    #x_train, x_crosstest, y_train, y_crosstest = cross_validation.train_test_split(data, labels, test_size = 0.1)    
    
    #del train_df, labels, features
    
    Y_train = categorical_label(labels)
    #Y_crosstest = categorical_label(y_crosstest)
    
    #print '~@# TRAIN DATA LOADED #@~'
    return data, Y_train
    
def load_test_data(file_name):
    test_df = replace_non_numeric(pd.read_csv(file_name))
    
    features = test_df[["Sex", "Pclass", 'Embarked', 'Age', 'SibSp', 'Parch']].values
    data = preprocess_data(features)
    
    del test_df, features
    
    #print '~@# TEST DATA LOADED #@~'
    return data
    
def load_model():
    model = Sequential()
    
    model.add(Dense(74, input_dim=37, bias = True, activation='relu'))
    model.add(Dense(148, input_dim=74, bias = True, activation='relu'))
    model.add(Dense(74, input_dim=148, bias = True, activation='relu'))
    model.add(Dense(37, input_dim=74, bias = True, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(2, input_dim=37, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print '~@# MODEL LOADED #@~'
    return model

x_train, Y_train = load_train_data('../input/train.csv')

model = load_model()

batch_size = 32
epochs = 15

model.fit(x_train, Y_train, nb_epoch = epochs, batch_size = batch_size)

x_test = load_test_data('../input/test.csv')

result = pd.Series(categorical2singleclass(model.predict(x_test)))
output = pd.DataFrame(data={"PassengerId":range(892,1310), "Survived":result})
output.to_csv("NNoutput.csv", index=False)