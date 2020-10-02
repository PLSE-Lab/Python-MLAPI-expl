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
"""
Created on Sun Jul 21 12:21:19 2019

@author:Amine Bouslimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('../input/API_TUN_DS2_en_csv_v2_41567.csv',sep=',',header=[0],skiprows=4)
#converting the csv into data 
df=pd.DataFrame(data)
df.columns = df.columns.str.strip()
df_top=df.head()
print(df_top)
#select the rw containing the population of males between20-24 
#df.loc[df.Indicator Name == "Survival to age 65, male (% of cohort)"]
thing = np.array(df.iloc[902])
year=[]
for i in range(1960,2019):
    year.append(i)
    
print(type(year))
years=np.array(year)
print(type(years))
things=thing[5:]
plt.plot(things,years,linestyle=None)
plt.xlabel("Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)")
plt.ylabel("years")
plt.show()
