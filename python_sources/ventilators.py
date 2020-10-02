#!/usr/bin/env python
# coding: utf-8

# # What is upper limit value of Positive pressure ventilators for patients with Pulmonary Edema

# In[ ]:


import numpy as np 
import pandas as pd 
from IPython.display import HTML


def make_clickable(url, title):
    return '<a href="{}">{}</a>'.format(url,title)


ventilators = pd.read_csv('/kaggle/input/tables-for-tayab/ventilators.csv',header=None)[3:40]
ventilators.drop(ventilators.columns[[11,12,13,14,15,16,17,18,19]], axis=1, inplace=True)
ventilators.columns = ['Date','Study','Study Link','Journal','Study Type','Sample Size','Therapeutic method(s) utilized/assessed','Severity of Disease','Pressure Setting Range (Identify Max First)','Primary Endpoint(s) of Study','Clinical Improvement (Y/N)']
df = ventilators

list_of_columns = ['Date','Study','Journal','Study Type','Sample Size','Therapeutic method(s) utilized/assessed','Severity of Disease','Pressure Setting Range (Identify Max First)','Primary Endpoint(s) of Study','Clinical Improvement (Y/N)']
df['Study'] = df.apply(lambda x: make_clickable(x['Study Link'], x['Study']), axis=1) # Add in link
df = df[list_of_columns]
HTML(df.to_html(escape=False))


# In[ ]:


df.to_csv('/kaggle/working/ventilators.csv')

