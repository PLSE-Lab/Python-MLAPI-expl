#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Bibliotecas
import numpy as np               # algebra linear
import matplotlib.pyplot as plt  # plotar gráficos
import pandas as pd 
import sklearn.linear_model as lr
import sklearn.metrics as skr


# ####################### Banco de Dados #################################### #
# Conjunto de dados de exemplo (dataset)
dataset = pd.read_csv ("../input/auto-mpg.csv", usecols = ['mpg', 'horsepower', 'weight' ])
X1  = np.array ([dataset ['horsepower']]). T
X2  = np.array ([dataset ['weight']]). T
y   = np.array ([dataset['mpg']]). T
m   = y.size

# Converter de Libra para Kg
X2 = X2 * 0.4535923

# Converter de mpg para km / l
y = y * 0.425144

for i in range (m):
    Xi  =   X1 [i][0]
    if not Xi.isnumeric ():
        print(Xi)
        X1[i][0] = 0

X1.astype ('float')

X = X2
# ########################################################################### #

ptrain  =   int (0.7 * m)
XTrain  =   X [:ptrain, [0, 1]]
yTrain  =   y [:ptrain, [0]]

XTeste   =   X [ptrain:, [0, 1]]
yTeste   =   y [ptrain:, [0]]


# #################### PLOTAR DADOS ###########################################
# Dados Originais
plt.scatter(XTeste, yTeste)
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (km/l)")
# ########################################################################### #

# ####################### PREPROCESSAMENTO ################################## #
# Número de Exemplos
m = y.size
########################## REGRESSÂO LINEAR ###################################################
reglinear   =   lr.LinearRegression()
reglinear.fit(XTrain, yTrain)

T0  =   reglinear.intercept_
T1  =   reglinear.coef_

print("t0", t0)
print("t1", t1)

yPrevisto   =   reglinear.predict (XTeste)
erro    =   skr.mean_squared_error (yPrevisto, yTeste)
print(erro)



# #################### GERAR E PLOTAR HIPÓTESE ############################## #

plt.plot(XTeste, yPrevisto, "r")
plt.show()

# ########################################################################## #
