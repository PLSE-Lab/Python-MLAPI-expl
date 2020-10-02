########## IMPORTACIÓN DE LAS LIBRERIAS NECESARIAS ##########
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

########## PREPARACIÓN DE LA DATA ##########
data = pd.read_csv('data.csv', index_col=0)

########## ENTENDIMIENTO DEL DATASET ##########
print('Nombres de las columnas:')
print(data.keys(), '\n')

print('Descripción general:')
print(data.describe(),'\n')

print('Número de valores nulos en cada columna:')
print(data.isnull().sum(),'\n')
print('No es necesario hacer limpieza de valores Nan \n')

########## ALGUNOS GRÁFICOS INTERESANTES ##########

X = data[['games_played']]
y = data.kills

plt.scatter(X,y, color='red', label='Kills')
plt.xlabel('Games played')
plt.ylabel('Kills')
plt.legend()
plt.title('R6S')
plt.show()

########## IMPLEMENTACIÓN DE LA REGRESIÓN LINEAL SIMPLE ##########

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo
start = time.time()

lr = linear_model.LinearRegression()

#Entreno al algoritmo

lr.fit(X_train, y_train)

#Realizo la predicción

y_pred = lr.predict(X_test)

#REPRESENTACIÓN DE LOS DATOS
print('\n\n Representación del modelo:\n')

plt.plot(X_test,y_pred, color='darkblue', label='Predicción', linewidth=3)
plt.scatter(X_test,y_test, color='red', label='Real', linewidth=3)
plt.xlabel('Games played')
plt.ylabel('Kills')
plt.legend()
plt.title('R6S')
plt.show()

print('\n\nPendiente (a) de la recta:')
print('a=', lr.coef_)
print('Ordenada en el origen (b):')
print('b=', lr.intercept_)
print('Ecuación de la recta:\n')
print('y = ', lr.coef_ , 'x +', lr.intercept_, '\n')
    
print('Precisión del modelo:')
print(lr.score(X_train, y_train), '\n\n')

print('Tiempo:')
print((time.time()-start), 's')
