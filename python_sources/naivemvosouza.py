import pandas as pd
import numpy as np
import csv as csv
from sklearn.naive_bayes import GaussianNB

#Iniciando o script
#Base de Treino
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

# Pegando o numero de clases
clases = np.unique(train_df['Pclass'])
numClasses = len(clases)

# Separa os registros em bin, pela classe do passageiro
media_precos = np.zeros(3);
des_padrao = np.zeros(3);

# Eliminando os valores ausentes dos Precos, utilizando a media do bin.
# A divisao dos bins foi por meio da classe dos passageiros, como classe 1,2 e 3.
# Entao a distribuicao normal de cada bin foi atribuida ao registo com valor ausente de seu proprio bin.
for i in range(0, 3):
    media_precos[i] = train_df[train_df.Pclass == i + 1]['Fare'].dropna().median()
    des_padrao[i] = train_df[train_df.Pclass == i + 1]['Fare'].dropna().std()
for i in range(0, 3):
    train_df.loc[(train_df.Pclass == i + 1) & (train_df.Fare == 0) | (train_df.Fare.isnull()), 'Fare'] = abs(np.random.normal(media_precos[i],des_padrao[i],1)[0]).astype(float)

#pega a media das idades
median_age = train_df['Age'].dropna().median()
#substitui as idades menores que um pela media
train_df.loc[(train_df['Age'] < 1) ,'Age'] = median_age
#substitui NaN pela media
train_df['Age'].fillna(median_age,inplace=True)
#arredanda as idades para um valor inteiro
train_df['Age'] = [int(elemento) for elemento in train_df['Age']]

# Discretizacao do sexo
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Remove a coluna de Nome, Cabine, Ticket
train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'SibSp','Parch'], axis=1)


# !!!!!!TEST DATA!!!!!!!!
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Pegando o numero de clases
clases = np.unique(test_df['Pclass'])
numClasses = len(clases)

# Separa os registros em bin, pela classe do passageiro
media_precos = np.zeros(3);
des_padrao = np.zeros(3);

# Eliminando os valores ausentes dos Precos, utilizando a media do bin.
# A divisao dos bins foi por meio da classe dos passageiros, como classe 1,2 e 3.
# Entao a distribuicao normal de cada bin foi atribuida ao registo com valor ausente de seu proprio bin.
for i in range(0, 3):
    media_precos[i] = test_df[test_df.Pclass == i + 1]['Fare'].dropna().median()
    des_padrao[i] = test_df[test_df.Pclass == i + 1]['Fare'].dropna().std()
for i in range(0, 3):
    test_df.loc[(test_df.Pclass == i + 1) & (test_df.Fare == 0) | (test_df.Fare.isnull()), 'Fare'] = abs(np.random.normal(media_precos[i],des_padrao[i],1)[0]).astype(float)

#pega a media das idades
median_age = test_df['Age'].dropna().median()
#substitui as idades menores que um pela media
test_df.loc[(test_df['Age'] < 1) ,'Age'] = median_age
#substitui NaN pela media
test_df['Age'].fillna(median_age,inplace=True)
#arredanda as idades para um valor inteiro
test_df['Age'] = [int(elemento) for elemento in test_df['Age']]

# Discretizacao do sexo
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

ids = test_df['PassengerId'].values
test_df = test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'SibSp','Parch'], axis=1)

train_data = train_df.values
test_data = test_df.values

#------------------NaiveBayes-------------------------#

model = GaussianNB()
model.fit( train_data[0::,1::], train_data[0::,0] )
outputNaive = model.predict(test_data).astype(int)
