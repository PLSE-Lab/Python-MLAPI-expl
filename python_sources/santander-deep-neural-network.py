# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dados = pd.read_csv('../input/train.csv')

X = dados.iloc[:, 2:].values.astype('float64')
y = dados.iloc[:, 1].values

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    
    clf = Sequential()
    clf.add(Dense(units=neurons,  activation=activation, kernel_initializer=kernel_initializer, input_dim = 200))
    clf.add(Dropout(0.20))
    clf.add(Dense(units=neurons,  activation=activation, kernel_initializer=kernel_initializer))
    clf.add(Dropout(0.20))
    clf.add(Dense(units=neurons,  activation=activation, kernel_initializer=kernel_initializer))
    clf.add(Dropout(0.20))
    clf.add(Dense(units=1,  activation='sigmoid'))
    clf.compile(optimizer= optimizer, loss=loss, metrics=['binary_accuracy'])

    return clf

classificador = KerasClassifier(build_fn=criarRede)

parametros = {'batch_size':[2000, 4000],
              'epochs':[25,50],
              'optimizer':['adam','sgd'],
              'loss':['binary_crossentropy','hinge'],
              'activation':['relu'],
              'kernel_initializer':['random_uniform', 'normal'],
              'neurons':[50,100]}

grid_search = GridSearchCV(estimator= classificador,
                           param_grid= parametros,
                           scoring= 'accuracy',
                           cv= 4)

grid_search = grid_search.fit(X, y)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

#Cria o classificador com os melhores parametros e o treina 
clf_melhor = criarRede(melhores_parametros['optimizer'], melhores_parametros['loss'],
                     melhores_parametros['kernel_initializer'], melhores_parametros['activation'],
                     melhores_parametros['neurons'])

clf_melhor.fit(X, y, batch_size=melhores_parametros['batch_size'], epochs = melhores_parametros['epochs'])

# Realiza a predicao
dados_teste = pd.read_csv('../input/test.csv')
xteste = dados_teste.iloc[:, 1:].values.astype('float64')
predicao = clf_melhor.predict(xteste)


#Faz a avaliação da rede.

limite = 0.7 # Testado de 0.5 a 0.8. 0.7 é um valor razoavel de acerto.
for i in range(len(predicao)):
    if predicao[i] < limite:
        predicao[i] = 0
    else:
        predicao[i] = 1
        
label_teste = pd.read_csv('../input/sample_submission.csv')
yteste = label_teste.iloc[:, 1].values

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(yteste, predicao))
accuracy_score(yteste, predicao)

#salva o arquivo de predição da rede neural.
Santader_DNN = pd.DataFrame(predicao, index=label_teste['ID_code'], columns=['target'])
Santader_DNN.to_csv('predict_santander_DNN.csv')


#salva a estrutura da rede neural em formato json
clf_json = clf_melhor.to_json()

#salva a estrutura da rede json em um arquivo no disco
with open('clf_santander.json', 'w') as json_file:
     json_file.write(clf_json)

#salva os pesos da rede neural em arquivo 
clf_melhor.save_weights('clf_santander_pesos.h5')