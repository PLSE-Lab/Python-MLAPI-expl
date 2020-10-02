#!/usr/bin/env python
# coding: utf-8

# # Visualizacion de Datos con Python

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm 
import re as re
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Cargando el dataset

# In[ ]:


df = pd.read_csv('../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv', header = 0)


# In[ ]:


df.info()
# EL dataset esta incompleto asi que veamos que podemos hacer


# In[ ]:


# Sustituimos todo valor nan en la columna Clasiffier a valores 0 
df.Age  = df.Age.fillna('No Cateogorice')


# In[ ]:


# Transformamos la columna Rotten Tomatoes
def transform(df):
    Rating = df['Rotten Tomatoes'].str.split('%' ,expand = True)
    Rating = Rating[0]
    df = df.drop('Rotten Tomatoes', axis = 1)
    df.insert(loc=5 ,column='Rotten Tomatoes', value=Rating)
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(0)
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(int)
    return df


# In[ ]:


# Ahora aplicamos la funcion a nuestro DataFrame
df = transform(df)


# In[ ]:


df


# In[ ]:


# Transformamos la columna IMDb a porcentaje
df['IMDb'] = df['IMDb']*10


# In[ ]:


# Sustituimos los valores Nan en IMDb
df['IMDb'] = df['IMDb'].fillna(0)


# # Comenzamos a Visualizar los Datos

# # Veamos que servicio posee mas contenido

# In[ ]:


df.columns


# In[ ]:


features = ['Year', 'Age', 'IMDb', 'Rotten Tomatoes',
       'Netflix', 'Hulu', 'Prime Video', 'Disney+']
features = df[features]


# In[ ]:


N = features.Netflix.value_counts()
N = N[1]
H = features.Hulu.value_counts()
H = H[1]
P = features['Prime Video'].value_counts()
P = P[1]
D = features['Disney+'].value_counts()
D = D[1]


# In[ ]:


print(N) 
print('+'*40)
print(H)
print('+'*40)
print(P)
print('+'*40)
print(D)


# In[ ]:


# Crearemos una tabla para graficarla mejor
Dictionary = {'Service Name':('Netflix', 'Hulu', 'Prime Video', 'Disney+'), 'tv shows on the platform': (N,H,P,D)}
a = pd.DataFrame(Dictionary)


# In[ ]:


sns.barplot(x='Service Name',y='tv shows on the platform', data=a)


# ## Ahora veamos que Servicio te ofrece programas mas actuales

# ## Tomaremos como actuales este lustro del 2015 al 2020

# In[ ]:


Actuality = features.Year >=2015
Actuality = features[Actuality]


# In[ ]:


#Comparamos los datos perdidos
features.shape , Actuality.shape


# In[ ]:


# Ahora creamos una tabla para hacer el grafico
NA = Actuality.Netflix.value_counts()
NA = NA[1]
HA = Actuality.Hulu.value_counts()
HA = HA[1]
PA = Actuality['Prime Video'].value_counts()
PA = PA[1]
DA = Actuality['Disney+'].value_counts()
DA = DA[1]


# In[ ]:


print(NA) 
print('+'*40)
print(HA)
print('+'*40)
print(PA)
print('+'*40)
print(DA)


# In[ ]:


Dictionary = {'Service Name':('Netflix', 'Hulu', 'Prime Video', 'Disney+'), 'tv shows Actuality': (NA,HA,PA,DA)}
actuality = pd.DataFrame(Dictionary)


# In[ ]:


sns.barplot(x='Service Name', y='tv shows Actuality', data=actuality)b


# ## Ahora veamos quien te ofrece contenido de calidad

# In[ ]:


features['Global Rating'] = (features['IMDb'] + features['Rotten Tomatoes'])/2


# In[ ]:


# Aplicamos un filtro para las mejores peliculas
Best = features['Global Rating'] >= 80
Best = features[Best]


# In[ ]:


Best.shape


# In[ ]:


BN = Best.Netflix.value_counts()
BN = BN[1]
BH = Best.Hulu.value_counts()
BH = BH[1]
BP = Best['Prime Video'].value_counts()
BP = BP[1]
BD = Best['Disney+'].value_counts()
BD =BD[1]


# In[ ]:


print(BN) 
print('+'*40)
print(BH)
print('+'*40)
print(BP)
print('+'*40)
print(BD)


# In[ ]:


# Creamos una tabla
Bestd = {'Service Name':('Netflix', 'Hulu', 'Prime Video', 'Disney+'), 'Most qualified': (BN,BH,BP,BD)}
Bestd = pd.DataFrame(Bestd)


# In[ ]:


sns.barplot(x='Service Name', y = 'Most qualified', data=Bestd)


# ## Year con mas media de calificacion

# In[ ]:


# Veamos cual es el year que obtuvo mas media en calificacione
GoldYear = features['Global Rating'] > 0 
GoldYear = features[GoldYear]


# In[ ]:


GoldYear


# In[ ]:


GoldYear[['Year','Global Rating']].groupby('Year', as_index=False).mean()


# In[ ]:


sns.relplot(x='Year', y='Global Rating',kind='line', data=GoldYear)


# # Veamos que contenido podemos encontrar en cada plataforma

# ## Netflix

# In[ ]:


sns.countplot(x='Netflix', hue='Age', data=features)


# In[ ]:


features[['Netflix','Age']].groupby('Age', as_index = False).mean()


# In[ ]:


sns.barplot(x='Netflix',y='Age',data=features)


# ## Hulu

# In[ ]:


sns.countplot(x='Hulu',hue='Age', data=features)


# In[ ]:


features[['Hulu','Age']].groupby('Age', as_index=False).mean()


# In[ ]:


sns.barplot(x='Hulu',y='Age',data=features)


# ## Prime Video

# In[ ]:


sns.countplot(x='Prime Video',hue='Age', data = features)


# In[ ]:


features[['Prime Video','Age']].groupby('Age', as_index=False).mean()


# In[ ]:


sns.barplot(x='Prime Video', y='Age', data = features)


# ## Disney+

# In[ ]:


sns.countplot(x='Disney+',hue='Age', data = features)


# In[ ]:


features[['Disney+','Age']].groupby('Age', as_index=False).mean()


# In[ ]:


sns.barplot(x='Disney+', y='Age', data = features)

