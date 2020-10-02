#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import warnings
    
# Source: 
#http://www.tesourotransparente.gov.br/ckan/dataset/lista-de-orgaos-do-siafi/resource/d82d88e8-07a0-486e-a5c8-6c096f98d0d6

df_Orgaos = pd.read_csv('../input/Brazil_Gov_Depts.csv', engine='python', error_bad_lines=False)


# In[ ]:


get_ipython().system('pip install requests')


# In[ ]:


df_Orgaos.columns = ['CD', 'DESC', 'CD2', 'IND', 'DESC2', 'IND2', 'OrgaoPai']

df_Orgaos


# In[ ]:


import sys
import json
import codecs
import urllib.request
import urllib.parse
import requests

# don't forget pip install requests!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

api_base_url = 'http://www.transparencia.gov.br/api-de-dados/despesas/por-orgao?ano=2018&orgao=20202&pagina=1'

api_response = requests.get(api_base_url)
response = json.loads(api_response.text)
#print(response.status_code)

result = pd.DataFrame(response)
result


# In[ ]:


N = len(df_Orgaos)
N


# In[ ]:


df_Orgaos.head(3)


# In[ ]:


# PART 01 - GENERATE FILE WITH VALID AGENCIES:

import re

Lista = []
api_base_urlFim = '';
parm1 = '&pagina=1'
i = 1
ANO = 2013
N = len(df_Orgaos)

for i in range(N):
    print ('Registro: ', i, ' / df_Orgaos[CD][i]: ', df_Orgaos['CD'][i])
    link1 = 'http://www.transparencia.gov.br/api-de-dados/despesas/por-orgao?ano=%s' %ANO
    link1 = link1 +'&orgao=%s'
    link1 = link1 %df_Orgaos['CD'][i]
    link2 = link1 + '%s' %parm1
    print(link2)
    api_response = requests.get(link2);
    response = json.loads(api_response.text);
    if response != []:
        print ('Resposta : ', response);
        Lista.append(response);
        i = i +1;
        
ano = list()
orgao = list()
codigoOrgao = list()
orgaoSuperior = list()
codigoOrgaoSuperior = list()
empenhado = list()
liquidado = list()
pago = list()


# In[ ]:


x=0

for i in Lista:
    print ('Ocorrencia: ', x, 'Linha: ', i)
    Error = str(i)
    Error = Error[2:7]
    if Error != 'Error':
        ano.append(i[0]['ano'])
        codigoOrgao.append(i[0]['codigoOrgao'])
        codigoOrgaoSuperior.append(i[0]['codigoOrgaoSuperior'])
        empenhado.append(i[0]['empenhado'])
        liquidado.append(i[0]['liquidado'])
        orgao.append(i[0]['orgao'])
        orgaoSuperior.append(i[0]['orgaoSuperior'])
        pago.append(i[0]['pago'])
    x = x+1


# In[ ]:


Listafim = pd.DataFrame({'Year':ano,              
              'Gov_agency_Cod':codigoOrgao,
              'Sup_Gov_agency_Cod':codigoOrgaoSuperior,
              'Planned':empenhado,
              'Released':liquidado,
              'Gov_agency_Name':orgao,
              'Sup_Gov_agency_Name':orgaoSuperior,              
              'Paid_out':pago})

ANOTXT = str(ANO)
Filename = 'Brazil_Gov_Exp_' + ANOTXT + '.csv'
Listafim.to_csv(Filename, index = False)


# In[ ]:


Listafim = pd.read_csv(Filename, engine='python', error_bad_lines=False, thousands='.', decimal =',')

Listafim['Paid_out'] = Listafim['Paid_out'].astype(int)
Listafim['Paid_out'] = Listafim['Paid_out'] / 1000
Listafim['Paid_out'] = Listafim['Paid_out'].astype(int)

ListSum = pd.DataFrame(Listafim.groupby(['Sup_Gov_agency_Cod', 'Sup_Gov_agency_Name'], as_index=False)['Paid_out'].sum())

#from pandas_datareader import fred
import seaborn as sns
import matplotlib as mpl

Graph_SGAC = pd.DataFrame()
Graph_SGAC['Sup_Gov_agency_Cod'] = ListSum['Sup_Gov_agency_Cod']
Graph_SGAC['Sup_Gov_agency_Name'] = ListSum['Sup_Gov_agency_Name']
Graph_SGAC['Paid_out'] = ListSum['Paid_out']

sns.factorplot(y='Paid_out', x='Sup_Gov_agency_Cod', data=Graph_SGAC, hue='Sup_Gov_agency_Name', size=5, aspect=2)

ListSum = ListSum.sort_values(by=['Paid_out'], ascending = False)
ListSum
y = 1

print ('# # # 2014 # # # NUMBER IN MILIONS!!!')
N = len(ListSum)
for i in range(N):
    #Print = ListSum['Sup_Gov_agency_Cod'][i]
    Print = ListSum.iloc[i:y]
    NAME = Print['Sup_Gov_agency_Name'].to_string(index=False)
    TOT = Print['Paid_out'].to_string(index=False); TOT = int(TOT); TOT = float(TOT); TOT = ('{:7,.2f}'.format(TOT))
    COD = Print['Sup_Gov_agency_Cod'].to_string(index=False); COD = int(COD);
    print ('##', i, ' - Cod: ', COD,  ' - Ag. Name: ', NAME, ' - Value: ', TOT)
    y=y+1
    
ANOTXT = str(ANO)
Filename = 'Brazil_Gov_Exp_Sum_' + ANOTXT + '.csv'
ListSum.to_csv(Filename, index = False)

