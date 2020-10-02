#!/usr/bin/env python
# coding: utf-8

# In[95]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

print(10)
# Any results you write to the current directory are saved as output.


# In[ ]:





# In[96]:


df = pd.read_csv('../input/LibreOffice Writer.csv')


# In[97]:


print("ODIO")


# In[98]:


df.columns = ['dataHora', 'nome', 'p1', 'p2', 'p3','p4','p5','p6', 'p7','p8','p9','p10']
mx = df.as_matrix()
print(mx)


# In[99]:


def limparCelula(x):
    try:
        x = int(x[0])
        return x
    except:
        return x


# In[100]:


def limparTabela():
    
    for x in range(df.shape[1] - 2):
        i = 'p' + str(x + 1)
        print(i)
        df[i] = df[i].apply(limparCelula)


# In[101]:


limparTabela()
df


# In[102]:


def calcularResposta(resposta):
    if resposta % 2 == 0:
        return 5 - resposta
    return resposta - 1


# In[103]:


def estatisticas(col):
    return [col.min(), col.max(), col.mean(), col.std()]


# In[104]:


def inverterNota():
    if nota == 5:
        nota = 1
    elif nota == 4:
        nota = 2
    elif nota == 2:
        nota = 4
    elif nota == 1:
        nota = 5


# In[107]:


linha = 0
coluna = 2
soma = 0
total = 0
respostas  = []
while(linha <= 8):
    if(coluna >= df.columns.size):
        linha = linha+1
        print(respostas)
        respostas.append(soma * 2.5)
        total = total + soma*2.5
        soma = 0
        coluna = 2
        print("\n")
        continue
    try:
        mx[linha][coluna] = mx[linha][coluna][0]
    except:
        coluna = coluna+1
        continue
 
    soma = soma + int(mx[linha][coluna])
    coluna = coluna+1
    
print("Resultado Final: " + str(total / 8))

    


# In[ ]:



i =1;
while(i < 10):
    print(df['p'+str(i)])
    i = +1


# In[ ]:


print(df)
print(mx[0][11])

