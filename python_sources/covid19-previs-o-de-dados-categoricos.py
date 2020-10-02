# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:11:09 2020

@author: Fernando Henrique
"""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import confusion_matrix, accuracy_score

# Entrada
arquivo = "/kaggle/input/covid19/dataset.xlsx"

# Dado original
df = pd.read_excel( arquivo, unidecode='ascii', sep='\s+', index_col = 0 )

# Falta de dados
# Para lidar com dados que estão faltando (que são muitos) selecionei uma faixa de atributos que minimiza essa lacuna
miss = df.isnull().sum().sort_values()
miss_percent = ( df.isnull().sum() / df.isnull().count() ).sort_values()
missing_data = pd.concat( [ miss, miss_percent ], axis = 1, keys = [ 'Total', 'Percent' ] )
f, ax = plt.subplots( figsize = ( 30, 12 ) )
plt.xticks( rotation = '90' )
sns.barplot( x = missing_data.index, y = missing_data[ 'Percent' ] )
plt.xlabel( 'Atributos', fontsize = 15 )
plt.ylabel( 'Porcentagem de valores inexistentes', fontsize = 15 )
plt.title( 'Porcentagem de valores inexistentes por atributos no geral', fontsize = 15 )

# 5 primeiros atributos não faltam dados, sendo 4 delas targets.
# 17 atributos seguintes com aproximadamente 76% de perdas
missing_data.head(23)

# Diminuindo o dataset, para que contenha, por paciente, no minimo 22 atributos
df.dropna( thresh = 22, inplace = True )

# A porcentagem de perda no máximo 13% para 22 atributos 
miss = df.isnull().sum().sort_values()
miss_percent = ( df.isnull().sum() / df.isnull().count() ).sort_values()
missing_data = pd.concat( [ miss, miss_percent ], axis = 1, keys = [ 'Total', 'Percent' ] )
f, ax = plt.subplots( figsize = ( 30, 12 ) )
plt.xticks( rotation = '90' )
sns.barplot( x = missing_data.index, y = missing_data[ 'Percent' ] )
plt.xlabel( 'Atributos', fontsize = 15 )
plt.ylabel( 'Porcentagem de valores inexistentes', fontsize = 15 )
plt.title( 'Porcentagem de valores inexistentes por atributos no geral', fontsize = 15 )
plt.savefig('Porcentagem_missing_values.png', dpi=300)

# Os melhores atributos
good_features = missing_data.head(22) 

# Pré processamento 
# Aqui utilizamos os bons atributos, bons em questão de não faltar dados, criando um dataset e retirando as colunas que não precisamos 
df_good = df[ good_features.index.to_list() ].dropna()

df_good = df_good.drop( ['Patient addmited to regular ward (1=yes, 0=no)', 
                   'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                   'Patient addmited to intensive care unit (1=yes, 0=no)' ], axis = 1 )

# Tranformando dados categoricos
# Transformar os dados categoricos em numericos
replace_map = { 'not_detected' : 1, 'negative' : 1, 'detected' : 2, 'positive' : 2 }

df_good.replace( replace_map, inplace = True )

# Desbalanceado
df_good['SARS-Cov-2 exam result'].value_counts()

# Separate majority and minority classes
df_majority = df_good[ df_good[ 'SARS-Cov-2 exam result' ] == 1 ]
df_minority = df_good[ df_good[ 'SARS-Cov-2 exam result' ] == 2 ]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace = True,     # sample with replacement
                                 n_samples = 1240,    # to match majority class
                                 random_state = 223) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled['SARS-Cov-2 exam result'].value_counts()

df_y = df_upsampled[ 'SARS-Cov-2 exam result' ]

df_x = df_upsampled.drop( [ 'SARS-Cov-2 exam result' ], axis = 1 )

#Amostra estratificada
X_train, X_test, y_train, y_test = train_test_split( df_x, df_y, test_size = 0.25, stratify = df_y )

# Random Forest
fig = plt.gcf()
rf = ConfusionMatrix( RandomForestClassifier( n_estimators = 10 ) )
rf.fit(X_train, y_train) # Classifica RF
prediction_rf = rf.predict(X_test) # Demora muito
confusao_rf = confusion_matrix(y_test, prediction_rf)
taxa_acerto_rf = accuracy_score(y_test, prediction_rf)
taxa_erro_rf = 1 - taxa_acerto_rf
score_rf = rf.score(X_test, y_test) #mostra a porcentagem de acerto para os dados de teste
rf.poof()

fig.savefig('Matrix_confusao_Covid19.png', dpi=300)

PFI_rf = pd.DataFrame(index = df_x.columns, columns = ["Importancia"] , data = list(rf.feature_importances_)).sort_values(by=['Importancia'], ascending=False)

f, ax = plt.subplots( figsize = ( 30, 12 ) )
plt.xticks( rotation = '90' )
sns.barplot( x = PFI_rf.index, y = PFI_rf[ "Importancia" ] )
plt.xlabel( 'Atributos', fontsize = 15 )
plt.ylabel( 'Porcentagem da importância dos atributos', fontsize = 15 )
plt.title( 'Porcentagem da importância dos atributos num classificador RF', fontsize = 15 )
plt.show()
plt.savefig('PFI_RF.png', dpi=300)

PFI_rf["Importancia"].sort_values(ascending=False)[:10].sum()

############################################################################################

df_x_PFI = df_x.drop( PFI_rf["Importancia"].sort_values(ascending=False)[:10].index.to_list(), axis = 1 )

#Amostra estratificada
X_train, X_test, y_train, y_test = train_test_split( df_x_PFI, df_y, test_size = 0.25, stratify = df_y )

# Random Forest
fig = plt.gcf()
rf = ConfusionMatrix( RandomForestClassifier( n_estimators = 100 ) )
rf.fit(X_train, y_train) # Classifica RF
prediction_rf = rf.predict(X_test) # Demora muito
confusao_rf = confusion_matrix(y_test, prediction_rf)
taxa_acerto_rf = accuracy_score(y_test, prediction_rf)
taxa_erro_rf = 1 - taxa_acerto_rf
score_rf = rf.score(X_test, y_test) #mostra a porcentagem de acerto para os dados de teste
rf.poof()

fig.savefig('Matrix_confusao_Covid19_PFI.png', dpi=300)













