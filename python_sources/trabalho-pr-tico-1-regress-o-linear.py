#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Bibliotecas
import numpy as np               # algebra linear
import matplotlib.pyplot as plt  # plotar gráficos
import pandas as pd              # bibliotecas
import sklearn.linear_model as lr # regressão linear


################



# ####################### Banco de Dados #################################### #
# Conjunto de dados de exemplo (dataset)
dataset = pd.read_csv ("../input/auto-mpg.csv", usecols = ['mpg', 'horsepower', 'weight' ])
X1  = np.array([dataset ['horsepower' ]]). T
X2  = np.array ([dataset['weight']]). T
X   = X2
y   = np.array ([dataset['mpg']]). T
m   = y.size
X0  = np.ones ((m, 1))

###############################################################

# Converter de Libra para Kg
X2 = X2 * 0.4535923

# Converter de mpg para km / l
y = y * 0.425144

maxx    = X2.max()
minx    = X2.min()
mediax  = X2.mean()
X2  =   (X2 - mediax) / (maxx - minx)
X = X2
# ########################################################################### #


# #################### PLOTAR DADOS ###########################################
# Dados Originais
plt.scatter(X, y)
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (km/l)")
# ########################################################################### #

ptrain  =   int (0.7 * m)
XTrain  =   X [:ptrain, [0]]
yTrain  =   y [:ptrain, [0]]

XTeste   =   X [ptrain:, [0]]
yTeste   =   y [ptrain:, [0]]


# ####################### PREPROCESSAMENTO ################################## #
# Número de Exemplos
m = y.size
# ########################################################################### #

# #################### REGRESSAO LINEAR #######################################
# Parâmetros
t0 = 0
t1 = 0
passos = 1000
alfa = 0.1


def custo(X, y, t0, t1):
    errototal   = 0
    for i in range(m):
        xi = X[i]
        yi = y[i]
        hi = t0 + (t1 * xi)
        erro =  (hi - yi) ** 2
        errototal = errototal + erro
    
    J = (1/ (2 * m)) * errototal
    return J


def gradienteDescendente(X, y, t0, t1, alfa, passos):
    # Código Gradiente Descendente
    #
    #
    for p in range(passos):
        J = custo (X ,y ,t0 ,t1)
        print ("Custo rodada ", p, " ", J)
        grad0 = 0
        grad1 = 0
        for i in range(m):
            xi = X[i]
            yi = y[i]
            hi = t0 + (t1 * xi)
            grad0 = grad0 + hi -  yi
            grad1 = grad1 + (hi - yi) * xi
        
        t0 = t0 - alfa * (1 / m)  * grad0
        t1 = t1 - alfa * (1 / m) * grad1
        
   
    return t0, t1


# Execução do algoritmo
t0, t1 = gradienteDescendente(X, y, t0, t1, alfa, passos)

# Imprimindo Theta
print("T0:", t0)
print("T1:", t1)
# ########################################################################### #




# #################### GERAR E PLOTAR HIPÓTESE ############################## #
Xteste = np.linspace(np.min(X), np.max(X), 50).reshape((50, 1))
Yteste = np.zeros(Xteste.shape)

for i in range(Xteste.size):
    xti = Xteste[i]
    hti = t0 + xti * t1
    Yteste[i] = hti

plt.plot(Xteste, Yteste, "r")
plt.show()
# ########################################################################## # 
