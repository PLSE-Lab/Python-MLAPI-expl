import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ** Importando os Dados **

treino = pd.read_csv('../input/titanic/train.csv')
teste = pd.read_csv('../input/titanic/test.csv')

# teste.head()
# treino.head()

# ** Limpando os Dados **

# * Coluna Pclass *

treino.Pclass.isnull().sum()

# * Coluna Name *

def limpando_nome(df):
  df.Name = df.Name.str.split(' ').apply(lambda x:x[1])
  titles = ['Dr.', 'Rev.', 'y', 'Impe,',
        'Planke,', 'Mlle.', 'Major.', 'Col.', 'Gordon,', 'Don.', 'Walle,',
        'Melkebeke,', 'Pelsmaeker,', 'Messemaeker,', 'Capt.', 'Jonkheer.',
        'the', 'Mme.', 'Mulder,', 'Steen,', 'Carlo,', 'der', 'Shawah,',
        'Billiard,', 'Cruyssen,', 'Ms.', 'Velde,' , 'Palmquist,', 'Brito,'  , 'Khalil,'  ]
  df.Name.replace(titles, 'others', inplace=True)
  return df

df = treino.copy()
limpando_nome(df).Name.value_counts()

# * Coluna Sex *

df = treino.copy()
df.Sex.value_counts()

# sns.barplot(data=df, x = 'Sex', y = 'Survived')

# * Coluna Age *

def limpando_idades(df):
  df.Age = df.Age.fillna(df.Age.median())
  df.Age = pd.cut(df.Age, 5)
  return df

# * Coluna SibSp *

df = treino.copy()
df.columns

def limpando_familias(df):
  df['Family'] = df.SibSp + df.Parch
  df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
  return df

# * Limpando Ticket *

def limpando_ticket(df):
  df.drop('Ticket', axis=1, inplace=True)
  return df 

# * Limpando Fare *

def limpando_fare(df):
  df.Fare = df.Fare.fillna(df.Fare.median())
  df.Fare = pd.cut(df.Fare , 5)
  return df

# * Limpando Cabin *

def limapando_cabine(df):
  df.Cabin.fillna('N', inplace=True)
  df.Cabin = df.Cabin.apply(lambda x:x[0])
  return df

# * Limpando Embarked *

def limpando_embarcou (df):
  df.Embarked.fillna('S', inplace=True)
  return df

# *************************************************

def data_frame_limpo(df):
  df = limpando_nome(df)
  df = limpando_idades(df)
  df = limpando_familias(df)
  df = limpando_ticket(df)
  df = limpando_fare(df)
  df = limapando_cabine(df)
  df = limpando_embarcou(df)
  return df

# df = treino.copy() 
# data_frame_limpo(df)


# ** Pré-processamento de dados **

treino_limpo =  data_frame_limpo(treino)
teste_limpo =  data_frame_limpo(teste)

print('------------------ Data Frame Treino ------------------ ')
treino_limpo.head()
print('------------------ Data Frame Treino ------------------ ')
teste_limpo.head()



from sklearn import preprocessing

def lebels_encoded(df_treino, df_teste):
  Features = ['Name', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked']
  d = pd.concat([df_treino[Features], df_teste[Features]])

  for F in Features:
    le = preprocessing.LabelEncoder()
    le = le.fit(d[F])
    df_treino[F] = le.transform(df_treino[F])
    df_teste[F] = le.transform(df_teste[F])
  return df_treino, df_teste

treino, teste = lebels_encoded(treino_limpo, teste_limpo)

print('------------------ Sklearn Treino ------------------')
treino.head()
print('------------------ Sklearn Teste ------------------')
teste.head()

# ** Machine Learning - Model **

X = treino.drop(['PassengerId','Survived'], axis=1)
Y = treino.Survived

X.head()

teste_ml = teste.drop('PassengerId', axis=1)

teste_ml.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_treino, Y_treino)
rf.score(X_treino, Y_treino)

rf.fit(X,Y)

y = rf.predict(teste_ml)

teste['Survived'] = y 
teste.head()

df = teste[['PassengerId', 'Survived']]
df.head()
df.to_csv('Titanic.csv', index=False)




