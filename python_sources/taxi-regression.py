#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
from scipy.ndimage import interpolation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def unzip_file(filepath):
    zf = ZipFile(filepath, 'r')
    zf.extractall()
    zf.close()
unzip_file('/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/train.csv.zip')


# In[ ]:


train = pd.read_csv("/kaggle/working/train.csv")


# In[ ]:


def quantifier(array,taille):

    if(len(array)<taille):#interpolation
        #indexes = np.linspace(0, 3.0, num=taille)
        #for i in range(taille):
        z = taille / len(array)
        new_array = interpolation.zoom(array,z)
    elif(len(array)>taille):#quatification
        indexes = np.linspace(0, len(array)-1, num=taille,dtype = int)
        #print(indexes)
        new_array = array[indexes]
    else:
        new_array = array
    
    return new_array


# In[ ]:


trajects = train["POLYLINE"]
new_trajects = pd.DataFrame()
#print(trajects.size)
#minn = 23
#maxx = 23
#l=0
Min_cord_Number = 25
Traject_lenght = 50
for traject in trajects:
    traject_x = list()
    traject_y = list()
    for cor in traject[2:-2].split('],['):
        try:
            traject_x.append(float(cor.split(',')[0]))
            traject_y.append(float(cor.split(',')[1]))
        except ValueError:
            pass
    
    if(len(traject_x) >= Min_cord_Number):
        
        traject_xx = np.array(traject_x[:-1])
        destination_x = traject_x[-1]
        traject_x_quan = quantifier(traject_xx,Traject_lenght)
        
        traject_yy = np.array(traject_y[:-1])
        destination_y = traject_y[-1]
        traject_y_quan = quantifier(traject_yy,Traject_lenght)
        
        tmp_df = pd.DataFrame([[traject_x_quan,traject_y_quan,(destination_x,destination_y)]])
        new_trajects = new_trajects.append(tmp_df, ignore_index=True)

print(new_trajects.shape)
new_trajects.to_csv("newlines", encoding='utf-8')

#print(train["POLYLINE"].head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




