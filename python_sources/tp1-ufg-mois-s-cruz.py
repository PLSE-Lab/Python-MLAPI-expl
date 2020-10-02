#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.linear_model import LinearRegression # importa o modelo
from sklearn.model_selection import train_test_split
import sklearn.metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




dataset = pd.read_csv(("../input/auto-mpg.csv" ), usecols = ['mpg' ,'horsepower', 'weight'])

X1 = np.array([dataset["weight"]]).T
X2 = np.array([dataset["horsepower"]]).T
Y = np.array([dataset["mpg"]]).T

#conversões
#Conversão de Libras para Toneladas
x1 = X1 * 0.000453592
#Conversão de mpg para Km/l
y = Y * 0.43

m = y.size

for i in range(m):
    X2i = X2[i][0]
    
    if not X2i.isnumeric():
        
        X2[i][0] = 0
        #print(X2[i][0])
        
X2.astype('float')

X = np.hstack((x1,X2))
ptrain = int(0.7*m)
Xtrain = X [:ptrain,[0,1]]
Ytrain = Y [:ptrain,[0]]

Xtest = X [ptrain:,[0,1]]
Ytest = Y [ptrain:,[0]]
# separa em set de treino e teste
#Xtrain, Ytrain, Xtest, Ytest = train_test_split(Xtrain,Ytrain, test_size=0.7, random_state=42)

regr = LinearRegression() # cria o modelo
regr.fit(Xtrain, Ytrain) # treina o modelo
#Cálculando os parametros 
t0 = regr.intercept_
tn = regr.coef_
print ("t0",t0)
print ("t1" ,tn[0][0])
print("t2", tn[0][1])


#Fazendo a previsão
regr.predict(Xtrain)
Yprevisto = regr.predict(Xtest)

#Mensurando o erro
erro = sklearn.metrics.mean_squared_error(Yprevisto, Ytest)
print ("erro",erro)



#Gráficos
n = dataset
    
fig = plt.figure()
ax = fig.gca(projection='3d')
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Weight + horsepower')
ax.set_ylabel('Mpg')


plt.show()

