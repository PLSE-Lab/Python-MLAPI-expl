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

#uso apenas para o colab
#from google.colab import drive
#drive.mount('/content/drive')

#importando as bibliotecas

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#leitura do dataset de treino do proprio colab
#dataset_treino = np.genfromtxt('/content/drive/My Drive/Colab Notebooks/dataset_treino.csv', delimiter=",")

dataset_treino = np.genfromtxt("../input/dataset_treino.csv", delimiter=",")

#quebrando os dados em X e Y
#X eu retiro o cabeçalho e a ultima coluna que é o Y
#Y eu pego apenas a ultima coluna que é a classe

np.random.seed(4)
X_treino = dataset_treino[1:,1:9]
Y_treino = dataset_treino[1:,9]

#criando o modelo no keras com 8 entradas
#utilizando a função de ativação sigmoid porque quero classificar em duas classes 0 e 1.

model = Sequential()
model.add(Dense(8, activation="relu", input_dim=8, kernel_initializer="normal"))
#model.add(Dense(496, activation="relu", kernel_initializer="normal"))
#model.add(Dense(496, activation="relu", kernel_initializer="normal"))
model.add(Dense(150, activation="relu", kernel_initializer="normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))

#compilacao do modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit do modelo com 50 epocas e um batch_size de 5 com padronização.
scaler = StandardScaler()
model.fit(scaler.fit_transform(X_treino), Y_treino, epochs=50, batch_size=5, verbose=2)

#avalia o modelo com os dados de treino
scores = model.evaluate(scaler.fit_transform(X_treino), Y_treino)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#importa os dados de teste
#dataset_teste = np.genfromtxt('/content/drive/My Drive/Colab Notebooks/dataset_teste.csv', delimiter=",")
dataset_teste = np.genfromtxt("../input/dataset_teste.csv", delimiter=",")
X_teste = dataset_teste[1:, 1:]

#aplicação do modelo no dataset de teste retornando um valor entre 0 e 1.
predictions = model.predict(scaler.fit_transform(X_teste))

#passo por todos os 168 retornos e utilizo a função do python round transformando em inteiro
#o retorno é um valor 0 ou 1 indicando se tem diabetes ou nao
classe = []
for x in predictions:
  rounded = int(round(x[0]))
  classe.append(rounded)
  print(rounded)
  
#exportando os dados para enviar no kaggle

submission = pd.DataFrame()
submission['id'] = dataset_teste[1:, 0].astype('int64')
submission['classe'] = classe
submission

#salvado os dados direto na pasta colab do meu google drive
#submission.to_csv('/content/drive/My Drive/Colab Notebooks/submission1.csv', index=False)

submission.to_csv("submission.csv", index=False)
