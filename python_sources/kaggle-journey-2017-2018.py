#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import folium 
from folium import plugins
from highcharts import Highchart
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
df_2018=pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
df_2018.columns=df_2018.iloc[0]
df_2018=df_2018.drop([0])


# In[ ]:


print('Total respondents in 2018:',df_2018.shape[0],'with a growth of:',(df_2018.shape[0]-df_2017.shape[0])/df_2017.shape[0]*100,'%')


# In[ ]:


lol1=df_2017[(df_2017['CurrentJobTitleSelect']=='Data Scientist')&(df_2017['StudentStatus']!='Yes')].Country.value_counts()[:20].to_frame()
lol2=df_2018[(df_2018['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Data Scientist')&(df_2018['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice']!='I am a student')]['In which country do you currently reside?'].value_counts()[:20].to_frame()
coun_10=lol1.merge(lol2,left_index=True,right_index=True,how='outer')
coun_10.columns=['2017','2018']
H = Highchart(width=800, height=500)

options = {

    'title': {
        'text': 'Where do new users come from (2017-2018)'
    },
    'xAxis': {
        'categories': list(coun_10.index),
        'title': {
            'text': None
        }
    },
    'yAxis': {
        'min': 0,
        'title': {
            'text': 'Respondents'
        },
        'labels': {
            'overflow': 'justify'
        }
    },
    'legend': {
        'layout': 'vertical',
        'align': 'right',
        'verticalAlign': 'top',
        'x': -80,
        'y': 20,
        'floating': True,
        'borderWidth': 1,
        'shadow': True
    },
    'labels': {
        'items': [{
            'html': "<b>Clutter is your enemy!</b>",
            'style': {
                'color':'red',
                'font-family': 'Comic Sans MS',
                'font-size':'300%',
                'left': '200px',
                'top': '100px',
                'width': '750px'
            }
        }]
    },
    'credits': {
        'enabled': False
    },
    'plotOptions': {
        'bar': {
            'dataLabels': {
                'enabled': True
            }
        }
    }
}

H.set_dict_options(options)




data1 = list(coun_10['2017'])
data2 = list(coun_10['2018'])
H.add_data_set(data1, 'bar', '2017')
H.add_data_set(data2, 'bar', '2018')

H


# Where do new users come from?

# In[ ]:


coun_10


# In[ ]:




