#!/usr/bin/env python
# coding: utf-8

# # To what extent do various procedures (Intubation, extubation, aerosol therapies, HFNC, NIV) cause aerosolization (need metric ie ppm) in COVID 19 patients, and what is the risk of infection of health care worker of the various procedures?

# In[ ]:


import numpy as np 
import pandas as pd 
from IPython.display import HTML


def make_clickable(url, title):
    return '<a href="{}">{}</a>'.format(url,title)


aerosolization = pd.read_csv('/kaggle/input/tables-for-tayab/aerosolization.csv',header=None)[2:100]
aerosolization.drop(aerosolization.columns[[10,11,12,13,14]], axis=1, inplace=True)
aerosolization.columns = ['Date','Study','Study Link','Journal','Study Type','Type of Ventilation','Aerolization Risk','Infection Risk','Results','Measure of Evidence']
df = aerosolization

list_of_columns = ['Date','Study','Journal','Study Type','Type of Ventilation','Aerolization Risk','Infection Risk','Results','Measure of Evidence']
df['Study'] = df.apply(lambda x: make_clickable(x['Study Link'], x['Study']), axis=1) # Add in link
df = df[list_of_columns]
HTML(df.to_html(escape=False))


# In[ ]:


df.to_csv('/kaggle/working/aerosolization.csv')

