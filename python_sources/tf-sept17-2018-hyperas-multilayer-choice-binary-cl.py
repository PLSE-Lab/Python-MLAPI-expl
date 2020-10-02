'''
Dated: Sept17-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for multilayer choice of hyperparameter optimization of keras deepnet models
Results:

Best performing model chosen hyper-parameters:
{'Dense': 4, 'Dense_1': 4, 'Dense_2': 2, 'Dense_3': 1, 'Dense_4': 2, 'alpha': 0.4342110952342011, 'alpha_1': 0.3569902315036171, 'alpha_2': 0.7497067351380373, 'alpha_3': 0.3633435999561372, 'alpha_4': 0.8013684587587554, 'alpha_5': 0.8409681028616584, 'alpha_6': 0.782730904176426, 'alpha_7': 0.1941962780441408, 'alpha_8': 0.10542732763559425, 'alpha_9': 0.3781578397662583, 'batch_size': 3, 'epochs': 4, 'model_choice': 1, 'optimizer': 0}
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_49 (Dense)             (None, 128)               3200      
_________________________________________________________________
leaky_re_lu_36 (LeakyReLU)   (None, 128)               0         
_________________________________________________________________
dropout_36 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_50 (Dense)             (None, 16)                2064      
_________________________________________________________________
leaky_re_lu_37 (LeakyReLU)   (None, 16)                0         
_________________________________________________________________
dropout_37 (Dropout)         (None, 16)                0         
_________________________________________________________________
dense_51 (Dense)             (None, 32)                544       
_________________________________________________________________
leaky_re_lu_38 (LeakyReLU)   (None, 32)                0         
_________________________________________________________________
dropout_38 (Dropout)         (None, 32)                0         
_________________________________________________________________
dense_52 (Dense)             (None, 1)                 33        
=================================================================
Total params: 5,841
Trainable params: 5,841
Non-trainable params: 0

confusion matrix: [[3039  226]
 [ 648  478]]
accuracy: 0.8009565019357777
precision: 0.6789772727272727
recall: 0.42451154529307283
f1score: 0.5224043715846994
cohen_kappa_score: 0.4050137346239342




'''

from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras import optimizers
import time


def data():
    import pandas as pd

    data = pd.read_csv(r'../input/Surgical-deepnet.csv', header=0, sep=',')
    #data = data.drop(['id', 'Unnamed: 32'], axis=1)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data['complication'])
    y = le.transform(data['complication'])

    data = data.drop('complication', axis=1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.30, random_state=777)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=777)

    from sklearn.preprocessing import StandardScaler

    for i in X_train.columns:
        scaler = StandardScaler()
        scaler.fit(X_train[i].values.reshape(-1, 1))
        X_train[i] = scaler.transform(X_train[i].values.reshape(-1, 1))
        X_val[i] = scaler.transform(X_val[i].values.reshape(-1, 1))
        X_test[i] = scaler.transform(X_test[i].values.reshape(-1, 1))

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(X_train, y_train, X_val, y_val):
    from keras import models
    from keras import layers
    import numpy as np

    model = models.Sequential()

    model_choice = {{choice(['one', 'two', 'three'])}}    #comparing two dense layers architecture to three dense layers architecture
    if model_choice == 'one':
        #1st dense
        model.add(layers.Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}, input_shape=(len(data.columns),)))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #2nd dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #output
        model.add(layers.Dense(1, activation='sigmoid'))

    elif model_choice == 'two':
        #1st dense
        model.add(layers.Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}, input_shape=(len(data.columns),)))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #2nd dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #3rd dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #output
        model.add(layers.Dense(1, activation='sigmoid'))

    elif model_choice == 'three':
        #1st dense
        model.add(layers.Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}, input_shape=(len(data.columns),)))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #2nd dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #3rd dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #4th dense
        model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5), np.power(2, 6), np.power(2, 7), np.power(2, 8), np.power(2, 9), np.power(2, 10)])}}))
        model.add(LeakyReLU(alpha={{uniform(0.1, 1)}}))
        model.add(Dropout({{uniform(0.1, 1)}}))
        #output
        model.add(layers.Dense(1, activation='sigmoid'))

    from keras import callbacks
        
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs={{choice([25, 50, 75, 100, 200, 500])}}, batch_size={{choice([16, 32, 64, 128, 256])}}, validation_data=(X_val, y_val), callbacks=[reduce_lr])

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    startTime = time.time()
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())
    X_train, X_val, X_test, y_train, y_val, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save('breast_cancer_model.h5')
    print(best_model.summary())

    #let's make predictions using the best model
    y_pred = best_model.predict(X_test).round().astype(int)

    print("type(y_pred)",type(y_pred))
    print("y_pred.shape",y_pred.shape)
    print(y_pred[35:75])
    print(y_test[35:75])
    
    #testset metrics
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

    # Confusion matrix
    print("confusion matrix:", confusion_matrix(y_test, y_pred))

    # Accuracy 
    print("accuracy:", accuracy_score(y_test, y_pred))

    # Precision 
    print("precision:", precision_score(y_test, y_pred))

    # Recall
    print("recall:",recall_score(y_test, y_pred))

    # F1 score
    print("f1score:", f1_score(y_test,y_pred))

    # Cohen's kappa
    print("cohen_kappa_score:", cohen_kappa_score(y_test, y_pred))
    
    endTime = time.time()
    #print elapsed time in hh:mm:ss format
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))

    print('finished evaluating tf-sept19-2018-ReduceLROnPlateau-hyperas-multilayer-readmitted-binary-sigmoid-two-large-choices-Xhours.py')
    
