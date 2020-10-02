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

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

cwd = os.getcwd()

data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

data_train.info()
data_test.info()

data_train.head()
data_test.head()


# Analyse des prix des maisons


# On analyse le fichier train, qui contient toutes les informations relatives aux maisons.


# Pour avoir une vision globale, on utilise la fonction describe :

data_train.describe()

# On peut observer notamment que la moyenne du prix des maisons est de $180921, avec un écart type de $79442. Cela signigie qu'il y a une grande disparité des prix.

# Recherchons maintenant la corrélation entre le prix et les autres éléments du data frame.

tabcorr = data_train.corr()
correlation_prix = tabcorr.SalePrice
print(correlation_prix)

# Remarque : on peut suprrimer la ligne SalePrice puisqu'elle n'est pas pertinente. Une valeur est toujours corrélée à elle-même.

correlation_prix = correlation_prix.drop(['SalePrice'], axis=0)
print(correlation_prix)

# On peut aussi la trier par ordre décroissant.

correlation_prix = correlation_prix.sort_values(ascending=False)    
print(correlation_prix)

# Ainsi, on observe que le prix dépend principalement des facteurs suivants : "OverallQual" (qualité des matériaux et finitions de la maison), et du GrLivArea (surface habitable de la maison).
# De plus, la corrélation est positive, donc plus ces valeurs augmentent, et plus le prix de la maison augemente (cela aurait eu un effet contraire si le signe était négatif).

# Pour avoir une idée plus générale, et mieux visualiser les données, on trace ces deux valeurs en fonction du prix des maisons.

plt.figure(figsize=(8,8))
plt.scatter(data_train.OverallQual, data_train.SalePrice)
plt.xlabel('Qualité de la maison')                  
plt.ylabel('Valeur')                        
plt.title('Valeur par la qualité de la maison')        
plt.show

# On a bien un résultat logique : plus la qualité de la maison augmente, plus la maison est chère. 

plt.figure(figsize=(8,8))
plt.scatter(data_train.GrLivArea, data_train.SalePrice)
plt.xlabel('Surface habitable hors sol')                  
plt.ylabel('Valeur')                        
plt.title('Valeur en fonction de la surface habitable hors sol')        
plt.show

# De même, on a un résultat globalement logique : plus cette surface augmente, et plus le prix de la maison en question augmente.
# De plus, on observe deux valeurs aberrantes que l'on supprime pour avoir un modèle pertinent.

max(data_train.GrLivArea)
data_train = data_train[data_train.GrLivArea != 5642]
max(data_train.GrLivArea)
data_train = data_train[data_train.GrLivArea != 4676]


# Prédiction du prix


# Comme les corrélations ne sont pertinentes que pour les colonnes suivantes : OverallQual et GrLivArea. On prédira donc le prix de la maison à partir de ces valeurs. De plus, ce ne sont que des valeurs numériques, ce qui simplifie grandement la prédiction.

Y_train = data_train['SalePrice']
X_train = data_train[['OverallQual','GrLivArea']] 
X_test = data_test[['OverallQual','GrLivArea']] 

X_train.head()

# On vérifie si les dataframes ont des valeurs NaN.

X_train.isnull().values.any()
X_test.isnull().values.any()

#Ce n'est pas le cas, on peut donc établir une prédiction.


# Prédiction par régression linéaire :

lm = LinearRegression()
lm.fit(X_train, Y_train)            
Y_pred = lm.predict(X_test)


# Prédiction par forêts aléatoires :

from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_rf = rf.predict(X_test)


# Pour comparer les modèles utilisées, on les vérifie en les appliquant sur des données que l'on connait en "cachant" les prix.
# Ainsi, on utilise le dataframe Train.csv. On le sépare en deux : un train et un test.

data_train_train = data_train.sample(frac=0.8)          
data_train_test = data_train.drop(data_train_train.index)

Y2_train = data_train_train['SalePrice']
X2_train = data_train_train[['OverallQual','GrLivArea']]
Y2_test = data_train_test['SalePrice']
X2_test = data_train_test[['OverallQual','GrLivArea']]


# On teste le modèle de la régression linéraire.

lm.fit(X2_train, Y2_train)            
Y2_pred = lm.predict(X2_test)         

plt.scatter(Y2_test, Y2_pred)
plt.plot([Y2_test.min(),Y2_test.max()],[Y2_test.min(),Y2_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")

# On peut visualiser l'erreur : 

plt.plot(Y2_test-Y2_pred)

# Calculons la avec l'erreur sur les moindres carrés.

from sklearn import metrics
print(metrics.mean_squared_error(Y2_test, Y2_pred))

# Faisons de même avec la méthode des forêts aléatoires.

rf.fit(X2_train, Y2_train)
Y2_rf = rf.predict(X2_test)

plt.scatter(Y2_test, Y2_rf)
plt.plot([Y2_test.min(),Y2_test.max()],[Y2_test.min(),Y2_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")

# On observe que ce modèle est plus précis. Vérifions le en calculant l'erreur.

plt.plot(Y2_test-Y2_rf)
print(metrics.mean_squared_error(Y2_test, Y2_rf))

# On observe bien que la méthode des forêts aléatoires est plus précise que la méthode de la régression linéaire.