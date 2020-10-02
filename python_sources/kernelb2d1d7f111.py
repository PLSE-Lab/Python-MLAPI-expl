# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Create Thiago Costa
#  NaiveBayes dataset CancerBreast diagnosis

import pandas as pd
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


dados = pd.read_csv("breast-cancer.csv")
print(dados.head())

#Count the number of rows and columns in the data set
print(dados.shape)

#Count the empty (NaN, NAN, na) values in each column
print(dados.isna().sum())

#Drop the column with all missing values (na, NAN, NaN)
#NOTE: This drops the column Unnamed
print(dados.dropna(axis=1))

#Encoding categorical data values (Transforming categorical data/ Strings to integers)

labelencoder = LabelEncoder()
dados.iloc[:,1]= labelencoder.fit_transform(dados.iloc[:,1].values)
print(labelencoder.fit_transform(dados.iloc[:,1].values))

#Notice I started from index  2 to 31, essentially removing the id column & diagnosis

previsores = dados.iloc[:, 2:31].values 
classe = dados.iloc[:, 1].values 

X_treinamento,X_teste,y_treinamento,y_teste = train_test_split(previsores,classe,test_size=0.25,random_state=0)
algoritmo = GaussianNB()
algoritmo.fit(X_treinamento,y_treinamento)
previsao = algoritmo.predict(X_teste)
matrix = confusion_matrix(y_teste,previsao)

taxaacerto = accuracy_score(y_teste,previsao)
print("A taxa de acerto desse modelo foi de:%.2f"%taxaacerto)
taxaerro = 1 - taxaacerto
print("A taxa de erro desse modelo foi de:%0.2f"%taxaerro)