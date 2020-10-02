#!/usr/bin/env python
# coding: utf-8

# This notebook is an example of how to access a radionuclide database to retrieve desintegration spectra for selected isotope

# With this code we can add missing packages, in this case we need numerical units from pip. We need internet connection activated in the kernel settings.

# In[ ]:


get_ipython().system('pip install numericalunits')


# We now check if we have access to the database. In this case is a local sqlite3 database.

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


# Here we can make all the required imports and initialize plotly

# In[ ]:


import sklearn.preprocessing as preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, BayesianRidge, TheilSenRegressor)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
import numericalunits

numericalunits.reset_units('SI')
#MeV = 1.602e-19 * 1e6
MeV = numericalunits.MeV
#mGy_MBqs = 1e-9
mGy = numericalunits.mJ/numericalunits.kg
MBqs = 1e6 
mGy_MBqs = mGy/MBqs

print(__version__) # requires version >= 1.9.0
init_notebook_mode(connected=True)


# We declare helper database class, this is a container for one full element

# In[ ]:


import json
particles = ['alpha', 'beta-', 'beta+', 'gamma', 'X', 'neutron',
             'auger', 'IE', 'alpha recoil', 'anihilation',
             'fission', 'betaD', 'b-spectra']

class Isotope:
    def __init__(self, name = '', half_life = 0.0, time_unit = ''):
        self.name = name
        self.half_life = half_life
        self.time_unit = time_unit
        self.emissions={}
        for particle in particles:
            self.emissions[particle] = []
            
    def __repr__(self):
        return json.dumps(self.__dict__)
    
    def getEmissionYield(self, particle='beta-'):
        if particle in particles:
            r=0.
            for em in self.emissions[particle]:
                r+=em[1]
            return r
        else:
            return NAN
            


# Connect to database and retrieve one isotope. In this case is sqlite3 database but connection object can be anything, just adapt to relevant database type.

# In[ ]:


Isotopes = {}
isot = 'Y-90'
import sqlite3

with sqlite3.connect("../input/icrp107.db3") as con:
    cursor = con.cursor()
    sql= "SELECT * FROM isotopes"
    cursor.execute(sql)

    for row in cursor.fetchall():
        Isotopes[row[0]]=Isotope(name=row[1], half_life=row[2], time_unit=row[3])
    
    sql= "SELECT * FROM particles"
    cursor.execute(sql)
    particles = {}
    for row in cursor.fetchall():
        particles[row[1]]={'id':row[0], 'name':row[1]}
    
    sql= "SELECT * FROM valores WHERE isotope = {} AND particle = {}"
    emissions = {}
    isotope = [k for k in Isotopes.keys() if Isotopes[k].name==isot]
    isotope=isotope[0]
    for particle in particles.values():
        tpart = particle['id']
        tsql=sql.format(isotope, tpart)
        cursor.execute(tsql)
        for row in cursor.fetchall():
            Isotopes[isotope].emissions[particle['name']].append({'energy':row[1], 'yield':row[2]})
            


# plot the b- spectra of selected isotope, in this example Y-90

# In[ ]:


x=[]
y=[]
for em in Isotopes[isotope].emissions["b-spectra"]:
    x.append(em["energy"])
    y.append(em["yield"])
    
scat = [go.Scatter(x = x, y = y, mode = 'lines', line = dict(width=1), name = 'b- spectra of '+ Isotopes[isotope].name)]
layout_scat = dict(title = 'B- spectra of '+ Isotopes[isotope].name,
                   xaxis = dict(title = 'energy (MeV)'),
                   yaxis = dict(title = 'Yield'),
              )
fig = dict(data=scat, layout=layout_scat)
iplot(fig)


# In[ ]:




