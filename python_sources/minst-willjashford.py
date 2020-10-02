# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#data manipulation packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# model packages
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.utils import to_categorical


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#we have 10 digits, so consider at least 10 models. 
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#initialise the models    
models = {}
for digit in range(10):
    models[digit]  = Sequential()
    models[digit].add(Dense(350, input_shape=(783,),activation='relu'))
    models[digit].add(Dense(175, activation='relu'))
    models[digit].add(Dense(2,activation='softmax'))
    models[digit].compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

#import the data
train_df = pd.read_csv("../input/train.csv")

#split into train and verify (90:10 split)
first_verify_index = math.ceil(0.1*len(train_df.index))
verify_indicies = range(first_verify_index,len(train_df.index))
train_indicies = range(first_verify_index)

verify_df = train_df.iloc[verify_indicies,:]
verify_df = verify_df.reset_index(drop=True)
train_df = train_df.iloc[train_indicies,:]

print(min(train_indicies),max(train_indicies))
print(min(verify_indicies),max(verify_indicies))


#build a scaler for the train set
scaler = StandardScaler()
scaler.fit(train_df.iloc[:,range(1,784)])

#conditionaly train the networks - to do
for digit in models.keys():
    #separate the data into positive and negative
    positive = train_df.loc[train_df['label']==digit,:]
    positive['label'] = 1
    negative = train_df.loc[train_df['label']!=digit,:]
    negative['label'] = 0
    custom_train_df = positive.append(negative)
    
    #shuffle the df
    custom_train_df = custom_train_df.sample(frac=1)
    
    #split into dependent and independent variables
    x = custom_train_df.iloc[:,range(1,784)].values
    #x = scaler.transform(x)
    y= to_categorical(custom_train_df.loc[:,'label'].values)
    print(y.shape)
    print("fitting model for {}".format(digit))
    
    models[digit].fit(x=x,y=y,epochs=3)
    
    x = None
    y = None
    custom_train_df = None
    positive = None
    negative = None

def forecast(data,models,scaler):
    #data = scaler.transform(data)
    prediction_classes_store = {}
    prediction_probabilities_store = {}
    for digit in models.keys():
        prediction_classes = models[digit].predict_classes(data)
        prediction_probabilities = models[digit].predict(data, batch_size=None, verbose=0, steps=None)
        prediction_classes_store[digit] = prediction_classes
        prediction_probabilities_store[digit] = prediction_probabilities
    
    print(prediction_classes_store,prediction_probabilities_store)
    
    resultant_predictions = []
    for row in range(len(prediction_probabilities_store[0])):
        current_max = None
        resultant_prediction = None
        
        for digit in models.keys():
            model_class= prediction_classes_store[digit][row]
            modeL_probability = prediction_probabilities_store[digit][row][1]
            
            if model_class == 1:
                if resultant_prediction is None:
                    resultant_prediction = digit
                    current_max = modeL_probability
                elif modeL_probability > current_max:
                    resultant_prediction = digit
                    current_max = modeL_probability
                else:
                    None
            else:
                None
                
        if resultant_prediction is None:
            for digit in models.keys():
                if resultant_prediction is None:
                    resultant_prediction = digit
                    current_max = modeL_probability
                    
                elif modeL_probability > current_max:
                    resultant_prediction = digit
                    current_max = modeL_probability
                else:
                    None
                    
        resultant_predictions.append(resultant_prediction)
        
            
    return resultant_predictions
        
#apply train scaler to the verify dataset
verify_forecast = forecast(data=verify_df.iloc[:,range(1,784)].values,models=models,scaler=scaler)
num_correct = 0.0
for row in verify_df.index:
    if verify_forecast[row] == verify_df.loc[row,'label']:
        num_correct += 1.0
    else:
        print(verify_forecast[row],verify_df.loc[row,'label'])
print("Correct forecasts: {}".format(num_correct))
verify_score = round(num_correct/float(len(verify_forecast))*100,2)
print("Score {}".format(verify_score))

"""
#import the test set
test_df = pd.read_csv("../input/test.csv")


#forecast the results - to do
submission = forecast(data=test_df.iloc[:,range(1,784)].values,models=models,scaler=scaler)
"""




