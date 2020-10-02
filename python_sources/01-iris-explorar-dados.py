#!/usr/bin/env python
# coding: utf-8

# ### Importando pacotes necessarios
# * numpy - manipulacao de arrays e outros
# * pandas - biblioteca com ferramentas para analise de dados e manipulacao de estruturas de dados
# * matplotlib - biblioteca para geracao de graficos
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ### Carga de dados
# **Carregando dados de arquivo csv com Pandas. Criando um dataframe para manipular dados**
# 

# In[ ]:


#carregar dados iris
file = '../input/iris-train.csv' # caminho absoluto do arquivo
dataFrame = pd.read_csv(file, delimiter = ',', index_col='Id')


# ### Exploracao inicial (conhecendo os dados carregados)

# In[ ]:


# Apresentar os primeiros 10 registros do dataframe

dataFrame.head(n=10)


# In[ ]:


# Apresentando estatisticas sumarizadas do dataframe, excluindo valores nulos
# count, media, desvio padrao, min, max
dataFrame.describe()


# In[ ]:


# qtde. de linhas e colunas do dataframe
dataFrame.shape


# In[ ]:


# apresenta colunas e respectivos tipos de dados
dataFrame.info()


# In[ ]:


# nome das colunas do data frame
dataFrame.columns


# In[ ]:


# quantidade de null de cada coluna
pd.isna(dataFrame).sum()


# In[ ]:


# tipo das colunas do dataFrame
dataFrame.dtypes


# In[ ]:


dataFrame['Species'].value_counts()


# In[ ]:


# existem colunas com dados nulos?
dataFrame[dataFrame.columns[dataFrame.isnull().any()]].isnull().sum()

#data[data.columns[data.isnull().any()]].isnull().sum()


# ### Exploracao inicial(visualizacao estatistica grafica)

# In[ ]:


# gerar grafico de dispersao simples
dataFrame.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[ ]:


# grafico de dispersao + histograma
import seaborn as sns
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=dataFrame, size=5, )


# In[ ]:


# grafico de dispersao por especie
sns.FacetGrid(dataFrame, hue="Species", height=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


# grafico de regressao linear (comprimento / largura das sepalas)
sns.regplot(x="SepalLengthCm", y="SepalWidthCm", data=dataFrame);


# In[ ]:


# grafico de regressao linear por especie (comprimento / largura das sepalas)
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=dataFrame);


# In[ ]:


# grafico de regressao linear (comprimento / largura das petalas)
sns.regplot(x="PetalLengthCm", y="PetalWidthCm", data=dataFrame);


# In[ ]:


# grafico de regressao linear por especie (comprimento / largura das petalas)
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", hue="Species", data=dataFrame);


# In[ ]:


# grafico de mapa de calor a partir da matriz correlacional do dataFrame
sns.heatmap(dataFrame.corr(), annot=True)

