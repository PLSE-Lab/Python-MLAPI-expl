#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Bibliotecas
import numpy as np               # algebra linear
import matplotlib.pyplot as plt  # plotar gráficos


# ####################### Banco de Dados #################################### #
# Conjunto de dados de exemplo (dataset)
X = np.array([[0.8],
              [1.2],
              [1.4],
              [1.4],
              [1.6],
              [2.0]])

y = np.array([[15],
              [13],
              [10],
              [12],
              [7.5],
              [7]])
# ########################################################################### #


# #################### PLOTAR DADOS ###########################################
# Dados Originais
plt.scatter(X, y)
plt.xlabel("Peso do veículo (kg)")
plt.ylabel("Autonomia (km/l)")
# ########################################################################### #


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
    # Código Custo
	#
	#
	#
    return J


def gradienteDescendente(X, y, t0, t1, alfa, passos):
	# Código Gradiente Descendente
	#
	#
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
# ########################################################################### #
