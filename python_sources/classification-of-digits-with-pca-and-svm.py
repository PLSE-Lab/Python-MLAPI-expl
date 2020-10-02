#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


## CARGA DE MODULOS
import numpy as np 
import pandas as pd 
import pylab
import seaborn as sns
sns.set_style("dark")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')
np.random.seed(1)
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
get_ipython().run_line_magic('pylab', 'inline')
from sklearn import decomposition
from sklearn import datasets
import math

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

## OBTENCION DE DATOS
test  = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
print('datos de entrenamiento -> ',train.shape)
print('datos de validacion -> ',test.shape)
train.head()
## SE ASIGNA LA ETIQUETA A UNA MATRIZ
Y_train = train["label"]

## BORRADO COLUMNA "LABEL"
X_train = train.drop(labels = ["label"],axis = 1) 
X_test = test

## GRAFICA DE BARRAS 
g = sns.countplot(Y_train)
Y_train.value_counts()
## GRAFICANDO ALGUNOS NUMEROS
print('Datos de Entrenamiento Etiquetados')
figure(figsize(5,5))
for digit_num in range(0,100):
    subplot(10,10,digit_num+1)
    grid_data = X_train.iloc[digit_num].as_matrix().reshape(28,28) # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])
## NORMALIZANDO LA DATA
N_train = X_train / 255
N_train['label'] = Y_train
## PCA decomposition
pca = decomposition.PCA(n_components=200) #Finds first 200 PCs
pca.fit(N_train.drop('label', axis=1))
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% de varianza')
#plot reaches asymptote at around 50, which is optimal number of PCs to use. 

## PCA decomposition with optimal number of PCs
#decompose train data
pca = decomposition.PCA(n_components=50) #use first 3 PCs (update to 100 later)
pca.fit(N_train.drop('label', axis=1))
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% de varianza')
PCtrain = pd.DataFrame(pca.transform(N_train.drop('label', axis=1)))
PCtrain['label'] = N_train['label']

#decompose test data
PCtest = pd.DataFrame(pca.transform(test))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111)

x =PCtrain[0]
y =PCtrain[1]
z =PCtrain[2]

colors = [int(i % 9) for i in PCtrain['label']]
ax.scatter(x, y, z, c=colors, marker='o', label=colors)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()
from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, Y_train, test_size=0.30, random_state=4)

# With a value of 16 the model can be trained in seconds and still achieve a 0.97229, using 49 components resulted in a smaller score
n_components = 16
t0 = time.time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("PCA hecho en  %0.3fs" % (time.time() - t0))
X_train_pca = pca.transform(X_train)

# see the variance histogram
plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
pca.explained_variance_ratio_.sum()
param_grid = { "C" : [0.1]
              , "gamma" : [0.1]}
rf = SVC()
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train_pca, Y_train)

print(gs.best_score_)
print(gs.best_params_)
bp = gs.best_params_
t0 = time.time()
clf = SVC(C=bp['C'], kernel='rbf', gamma=bp['gamma'])
clf = clf.fit(X_train_pca, Y_train)
print("SVM Hecho en %0.3fs" % (time.time() - t0))

clf.score(pca.transform(X_ts), y_ts)
test = X_test
val = test
pred = clf.predict(pca.transform(val))
# ImageId,Label

val['Label'] = pd.Series(pred)
val['ImageId'] = val.index +1
sub = val[['ImageId','Label']]


from sklearn.neural_network import MLPClassifier
y = PCtrain['label'][0:20000]
X=PCtrain.drop('label', axis=1)[0:20000]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(3500,), random_state=1)
clf.fit(X, y)

from sklearn import  metrics
#accuracy and confusion matrix
predicted = clf.predict(PCtrain.drop('label', axis=1)[20001:42000])
expected = PCtrain['label'][20001:42000]

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

sub.to_csv('resultado.csv', index=False)
sub.head()

