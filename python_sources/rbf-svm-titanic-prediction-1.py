# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.svm import SVC
import pdb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import  StandardScaler


# Any results you write to the current directory are saved as output.



#Cargar datos
train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
train.head(3)   #ver primeras tres fils para observar datos


#wrangling 
def ArregloData(df_train):
    df_train['Titulo'] = df_train.Name.str.extract('([A-Za-z]+)\.',expand = False)   #De la columna de Nombre obtener el titulo del Nombre

    #Obtener el numero de apariciones por Titulo en los datos y Eliminar las columnas que no aportan mucho al analisis
    df_train.drop(['Name','Cabin','Ticket','PassengerId'], axis = 1, inplace = True); print(pd.crosstab(df_train['Titulo'],df_train['Sex']))
    #pdb.set_trace()

    #sustituir los titulos menos comunes por "Distinto" dejando Mrs, Mr, Master, Miss, Mlle, Mme
    df_train['Titulo'] = df_train['Titulo'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Distinto')
    

    # Remplazar Titulos restantes para señorita, mujeres solteras etc 
    df_train['Titulo'] = df_train['Titulo'].replace(['Mlle','Ms'],'Miss')
    df_train['Titulo'] = df_train['Titulo'].replace('Mme', 'Mrs')

    
    #Cambiar los string values por valores numericos
    df_train.replace({'Titulo':{'Distinto' : 5, 'Master' : 1, 'Miss' : 2, 'Mr' : 3, 'Mrs' : 4}},inplace = True)

    df_train['Titulo'] = df_train['Titulo']

    #Remplazar valores string para la columna embarked  C = Cherbourg, Q = Queenstown, S = Southampton
    df_train.replace({'Embarked':{'C' : 1,'Q' : 2, 'S': 3}},inplace = True)
    #verificar si las columna cuenta con datos nulos
    print(df_train['Embarked'].isnull().sum().sum())
    print(df_train['Embarked'].mode())
    
    #pdb.set_trace()
    
    #Rellenar los datos nulos  con la moda 
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace = True)
    
    #Verificar si hay filas nulas en la columna de Sexo y cambiar valores alfanumericos en la columna de sexo por numericos 
    
    print(df_train['Sex'].isnull().sum().sum())   
    df_train.replace({'Sex': {'male' : 1 , 'female' : 0}},inplace = True)
    
    #pdb.set_trace()
    #Verificar datos en la columnda de edades
    print(df_train['Age'].isnull().sum().sum())
    
    df_train['Age'] = df_train.groupby('Titulo')['Age'].apply(lambda x: x.fillna(x.median()))
    
    df_train['Age'] = df_train['Age'].astype(int)
    
    print(df_train.info())
    
    #pdb.set_trace()

    return df_train



dataset = ArregloData(train)
#obtener matriz de correlacion para observar el comportamiento de las variables 

df_corr = dataset.corr()


#Aqui comienza la aprte de entranamiento


#Radial basis function
X = dataset.drop(['Survived'], axis = 1).values
y = dataset[['Survived']].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None)

#  cross-validation 
c_validation = KFold(n_splits = 6, random_state = None)

Tipo = ["RBF SVM"]
Clase = [SVC(gamma=3, C=1)]
models = []
trained_clases = []
for name, clf in zip(Tipo, Clase):
    scores = []
    for train_indices, test_indices in c_validation.split(X):
        clf.fit(X[train_indices], y[train_indices].ravel())

        scores.append( clf.score(X_test, y_test.ravel()) )
        
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    trained_clases.append(clf)
    models.append((name, min_score, max_score, avg_score))
    
fin_models = pd.DataFrame(models, columns = ['Name', 'Min Score', 'Max Score', 'Mean Score'])
fin_models.sort_values(['Mean Score']).head()


#Procesar los datos del Test con la funcion hecha anteriormente
df_id = df_test.iloc[:,[0]]
dataset_2 = ArregloData(df_test)


dataset_2 = pd.concat([df_id,dataset_2],axis = 1)
dataset_2['Fare'].fillna(dataset_2['Fare'].mode()[0],inplace = True)
ids = dataset_2['PassengerId']
dataset_2.drop(['PassengerId'],axis = 1, inplace = True)
x_2 = dataset_2.values
x_2 = StandardScaler().fit_transform(x_2)
predictions = clf.predict(x_2)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
#output.to_csv('titanic-predictions.csv', index = False)
output.head()