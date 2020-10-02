# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:09:32 2019

@author: daniel.lopez
"""
###############################################################################
# Cargar librarías necesarias
###############################################################################
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

###############################################################################
# Preparación de los datos
###############################################################################
"""
Obtained the full dataset from the following link:
https://weka.8497.n7.nabble.com/file/n23121/credit_fruad.arff
"""

#Cargar datos 
df_org = pd.read_csv('german_credit_data.csv', sep = ';')

#Convertirt arget de good/bad a 1/0
df_org['Type'] = df_org['Type'].replace('good',0)
df_org['Type'] = df_org['Type'].replace('bad',1)

# Transformar otras variables 
df_org['Telephone'] = df_org['Telephone'].replace('none',0)
df_org['Telephone'] = df_org['Telephone'].replace('yes',1)
df_org['foreign worker'] = df_org['foreign worker'].replace('no',0)
df_org['foreign worker'] = df_org['foreign worker'].replace('yes',1)

# Comprobar valoores vacíos
df_org.isnull().sum().max()

#Transformar variables en dummies 
dummies = df_org.select_dtypes(include='object')
dummies_df = pd.get_dummies(dummies)

# Delete these columns from df
df_org = df_org.select_dtypes(exclude=['object'])

# Reescalamos las variables entre 0-1 para que sean comparativas
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_org),columns=df_org.columns)

#Merge dummies df with original df 
df = pd.concat([df_scaled,dummies_df], axis=1)

#Ver cuántos good y bad hay de cada clase
df[df['Type'] == 1].count()
df[df['Type'] == 0].count()
"""
Un balanced dataset
"""
# Visualizar
sns.countplot('Type', data=df)
plt.title('Distribution by class')

# Definición de la correlación entre las variables 
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r',xticklabels=True,yticklabels=True)
plt.title("Matriz de Correlation", fontsize=14)

# Aplicar algoritmos de clusterización para ver patrones
X = df.drop('Type', axis=1)
y = df['Type']

# Algoritmo TSNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# leyenda
azul = mpatches.Patch(color = '#0A0AFF', label = 'Good')
rojo =  mpatches.Patch(color = '#AF0000', label = 'Bad')                     
                      
# t-SNE scatter plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('t-SNE', fontsize=14)
"""
No clear pattern
"""

# Otro método
X_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

plt.scatter(X_pca[:,0], X_pca[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('PCA', fontsize=14)
""" 
No clear pattern
"""
# TruncatedSVD
X_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

plt.scatter(X_svd[:,0], X_svd[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_svd[:,0], X_svd[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('SVD', fontsize=14)
"""
No clear pattern
"""

###############################################################################
# Modelos
###############################################################################

# Separar en test y train

# Hagamos el modelo 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_selection import RFE

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
"""
We chose 20 vars randomly, lets look for the optimum number of vars
"""
n_vars = np.arange(1,len(df.columns))  
resultado = 0
nof=0           
lista_resultados =[]

# Iterar sobre el total de las variables para ver que modelo es el óptimo

for n in range(len(n_vars)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    model = LogisticRegression()
    rfe = RFE(model,n_vars[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    var = model.score(X_test_rfe,y_test)
    lista_resultados.append(var)
    if(var > resultado):
        resultado = var
        nof = n_vars[n]

print('Optimum umber of vars is: %d' %nof)

print("Result with %d variables: %f" % (nof, resultado))

# Buscamos variables a aplicar
cols = list(X.columns)
model = LogisticRegression()

#Iniciamos el RFE (recursive feature elimination)
rfe = RFE(model, 24)             

#Entremos el modelo y vemos las variables a elegir
rfe.fit(X_train,y_train)              
variables_usar = list(X_train.columns[rfe.support_])

# Generamos una nueva matriz de correlaciones con las columnas elegidas
sns.set(style="white")

# Matriz de correlacion
corr = X_train[variables_usar].corr()

sns.heatmap(corr, cmap='coolwarm_r',xticklabels=True,yticklabels=True,annot=True)
plt.title("Matriz de Correlation", fontsize=14)
"""
Some correlations but not too high
"""
# Entranamos el modelo para predecir
model.fit(X_train[variables_usar],y_train)

# Predicciones 
y_pred = model.predict_proba(X_test[variables_usar])[:,1]

"""
Taking into account that if we just randomly selected any user, we would have a 70% chance 
of being correct, therefore every increase in precision is a gain.
"""

# Abría que definir el threshold por ahora he elegido un 0.5
y_predict = np.where(y_pred > 0.5,1,0)

# Matriz de confusiones 
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score

matriz = confusion_matrix(y_test,y_predict)
print(classification_report(y_test, y_predict))

#create a heat map
sns.heatmap(pd.DataFrame(matriz), annot = True, cmap = 'Blues', fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
"""
The model is no very good but it's the best we can do with these variables
"""
##############################################################################
"""
The most harmful situation would be classifing bad credits as good,
therefore let's try to increase our recall.
"""

#Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score

# Entrener el modelo 
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)

#Prdicción en base a los nuevos parámetros
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
matriz_grid = confusion_matrix(y_test,y_pred_acc)

sns.heatmap(pd.DataFrame(matriz_grid), annot = True, cmap = 'Blues', fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
"""
Doesn`t seem to have increased precision a lot.
"""

###############################################################################
# Probamos otros modelos 
###############################################################################

# Los tranformamos en format array para meterlos en los algoritmos 
X_train_mat = X_train[variables_usar].values
X_test_mat = X_test[variables_usar].values
y_train_mat = y_train.values
y_test_mat = y_test.values

# Definimos los algoritmos a utlizar
algoritmos = {
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Definimos la función para sacar la precisión

for key, classifier in algoritmos.items():
    classifier.fit(X_train_mat, y_train_mat)
    training_score = cross_val_score(classifier, X_train_mat, y_train_mat, cv=5)
    print("Algoritmo: ", classifier.__class__.__name__, "Precisión", round(training_score.mean(), 2) * 100)

###############################################################################


# %% [code]
