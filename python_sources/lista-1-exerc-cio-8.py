#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[7]:


def operacoesMedia(lista):
    dados = {}
    dados['media'] = round(np.mean(lista), 2)
    
    distanciaMedia = abs(lista[0] - dados['media'])
    
    for numero in lista:
        if  abs(numero - dados['media']) < distanciaMedia:
            dados['proximoDaMedia'] = numero
            distanciaMedia = abs(numero - dados['media'])
    
    return dados


# In[9]:


a = [1,4,5,6,7,3,5,4,50]
operacoesMedia(a)

