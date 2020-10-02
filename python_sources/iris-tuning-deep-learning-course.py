#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:07:07 2019

@author: lucas
"""

import keras
import numpy as np
import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

def criarRede (optimizer, loss, dropout, kernel_initializer, activation, layers):
    classificador = Sequential()
    classificador.add (Dense(units = layers[0], activation = activation, kernel_initializer=kernel_initializer, input_dim = 4))
    classificador.add (Dropout (dropout))
    
    for layer in layers[1:]:
        classificador.add (Dense(units = layer, activation = activation, kernel_initializer=kernel_initializer))
        classificador.add (Dropout (dropout))
        
    classificador.add (Dense(units = 3, activation = 'softmax'))
    
    classificador.compile (optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
    return classificador

iris = load_iris()
x = iris['data']
le = LabelEncoder()
#y = le.transform(iris['target'])
y = iris['target']

classificador = KerasClassifier(build_fn = criarRede)

parametros = {
        'batch_size': [10],
        'epochs': [1000],
        'optimizer': ['adam'],
        'loss': ['categorical_crossentropy'],
        'kernel_initializer': ['random_uniform'],
        'activation': ['relu'],
        'layers': [[8], [10], [12], [15]],
        'dropout': [0.2, 0.3]
}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 10
                          )

grid_search = grid_search.fit (x, y, verbose = 0)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
print ("Melhores parâmetros:")
print (melhores_parametros)
print ("Melhor precisão:")
print (melhor_precisao)