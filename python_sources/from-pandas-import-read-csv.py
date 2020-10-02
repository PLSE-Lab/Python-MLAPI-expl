# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

from pandas import read_csv

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# leemos el dataset
iris = read_csv('../input/Iris.csv')
print(iris)

X = iris[['PetalLengthCm', 'PetalWidthCm']] # Tomamos el ancho y longitud del pétalo.
y = iris['Species'].astype('category')     # Para crear luego los gráficos vamos a necesitar categorizar los nombres
y.cat.categories = [0, 1, 2]            # y asignarles un número a cada variedad de planta; en este caso tres.

n_neighbors = 15
h = .02          # step size de la malla
# Creamos los colormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # Creamos una instancia de Neighbors Classifier y hacemos un fit a partir de los
    # datos.
    # Los pesos (weights) determinarán en qué proporción participa cada punto en la
    # asignación del espacio. De manera uniforme o proporcional a la distancia.
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # Creamos una gráfica con las zonas asignadas a cada categoría según el modelo
    # k-nearest neighborgs. Para ello empleamos el meshgrid de Numpy.
    # A cada punto del grid o malla le asignamos una categoría según el modelo knn.
    # La función c_() de Numpy, concatena columnas.
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Ponemos el resultado en un gráfico.
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Representamos también los datos de entrenamiento.
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.xlabel('Petal Width')
    plt.ylabel('Petal Length')
    plt.savefig('iris-knn-{}'.format(weights))
    
