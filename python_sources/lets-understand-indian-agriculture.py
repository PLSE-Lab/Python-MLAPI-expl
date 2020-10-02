#!/usr/bin/env python
# coding: utf-8

# 
#  Agriculture is most important sector in India. 47% of India's GDP is contributed by Agricuture and most importantly Most of Indias workforce is working in this sector. Although It is important sector, People working in this sector has very low GDP per capita. Most of people are poor and uneducated. Many goverments has tried to change this but ground situation is still the same. Modi Goverment has made commitment to double the income of farmers by 2022. 
# 
# 
# There are many problems in this sector. Many researcher and scholar has suggested many solutions. But most of those solutions are not applied on ground level. 
# 
# 
# 
# *In this kernel We will try to understand how Indian Agriculture look like and We will try to define what are its problems based on data we have. In the subsequent Kernels We will design a model that can be implemented in this sector. *
# 
# 
# Now first lets take a look at Indian Agriculture.
# 
# 

# We have one dataset of Indian Economy. This is world bank dataset. Now lets take a look what this dataset tells about Indian Agriculture.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

agri=pd.read_csv('../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv')
agri.head()

# Any results you write to the current directory are saved as output.


# In[ ]:


x=agri['Years']
y=agri['EMP_AGR']
z=agri['GDP_AGR']


# In[ ]:


agri.describe().T


# This dataset having economic records of India of last 26 years.

# Lets take a look how Agri GDP has changed from 1991 to 2016. We will also plot a graph of how Agri Employment has changed from 1991 to 2016.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from scipy import stats

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
slope


# In[ ]:


import seaborn as sns
sns.regplot(x,y,label="Agriculture Employment").set_title("Agriculture Employment")


# In[ ]:



sns.regplot(x,z,label="Agriculture GDP").set_title("Agriculture GDP")


# In[ ]:


from scipy import stats

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,z)
slope


# From the above results we can see that **Slope of Agriculture Employment vs Years is more as compared with slope of Agriculture GDP vs Years**
# 
# Which tells us that Employment has gone down rapidly in the sector but GDP has not gone equipropertionally with Employment. **Which results in many seasonal employed people** Which lowered the GDP per Capita and poor standard of Life.

# Now We have another dataset. Now We understand Outlook of Agriculture. Now lets look at its economy more closely and try to understand How this econmy works in perspective of production and Crops.

# In[ ]:


data1=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (1).csv')
data1.head()


# In[ ]:


data1=pd.DataFrame(data=data1)
data1.describe().T


# Now We have find the crops for which we have data.

# In[ ]:


data1.Crop.unique()


# Once now I got names of Crops and I have data of Cost of Cultivation (/Hectare), Cost of Production (/Quintal) and Yield (Quintal/ Hectare). Now My aim is to Plot a graph of all the Crops and understand iits variation in lines. 
# 
# 
# First lets plot a graphs for All Crops

# In[ ]:


Arhar=data1[0:5]
COTTON=data1[6:10]
GRAM=data1[10:15]
GROUNDNUT=data1[15:20]
MAIZE=data1[20:25]
MOONG=data1[25:30]
PADDY=data1[30:36]
RAPESEEDAndMUSTARD=data1[35:40]
SUGARCANE=data1[40:45]
WHEAT=data1[46:49]


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))

x=Arhar['State']
y=Arhar['Cost of Cultivation (`/Hectare) A2+FL']
z=Arhar['Cost of Cultivation (`/Hectare) C2']
a=Arhar['Cost of Production (`/Quintal) C2']
b=Arhar['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("Arhar Production Variation", fontsize=16, fontweight='bold')
plt.show()


plt.figure(figsize=(10,8))
x=COTTON['State']
y=COTTON['Cost of Cultivation (`/Hectare) A2+FL']
z=COTTON['Cost of Cultivation (`/Hectare) C2']
a=COTTON['Cost of Production (`/Quintal) C2']
b=COTTON['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("COTTON Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=GRAM['State']
y=GRAM['Cost of Cultivation (`/Hectare) A2+FL']
z=GRAM['Cost of Cultivation (`/Hectare) C2']
a=GRAM['Cost of Production (`/Quintal) C2']
b=GRAM['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("GRAM Production Variation", fontsize=16, fontweight='bold')
plt.show()




plt.figure(figsize=(10,8))
x=GROUNDNUT['State']
y=GROUNDNUT['Cost of Cultivation (`/Hectare) A2+FL']
z=GROUNDNUT['Cost of Cultivation (`/Hectare) C2']
a=GROUNDNUT['Cost of Production (`/Quintal) C2']
b=GROUNDNUT['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("GROUNDNUT Production Variation", fontsize=16, fontweight='bold')
plt.show()




plt.figure(figsize=(10,8))
x=MAIZE['State']
y=MAIZE['Cost of Cultivation (`/Hectare) A2+FL']
z=MAIZE['Cost of Cultivation (`/Hectare) C2']
a=MAIZE['Cost of Production (`/Quintal) C2']
b=MAIZE['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("MAIZE Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=MOONG['State']
y=MOONG['Cost of Cultivation (`/Hectare) A2+FL']
z=MOONG['Cost of Cultivation (`/Hectare) C2']
a=MOONG['Cost of Production (`/Quintal) C2']
b=MOONG['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("MOONG Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=PADDY['State']
y=PADDY['Cost of Cultivation (`/Hectare) A2+FL']
z=PADDY['Cost of Cultivation (`/Hectare) C2']
a=PADDY['Cost of Production (`/Quintal) C2']
b=PADDY['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("PADDY Production Variation", fontsize=16, fontweight='bold')
plt.show()


plt.figure(figsize=(10,8))
x=RAPESEEDAndMUSTARD['State']
y=RAPESEEDAndMUSTARD['Cost of Cultivation (`/Hectare) A2+FL']
z=RAPESEEDAndMUSTARD['Cost of Cultivation (`/Hectare) C2']
a=RAPESEEDAndMUSTARD['Cost of Production (`/Quintal) C2']
b=RAPESEEDAndMUSTARD['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("RAPESEED And MUSTARD Production Variation", fontsize=16, fontweight='bold')
plt.show()

plt.figure(figsize=(10,8))
x=SUGARCANE['State']
y=SUGARCANE['Cost of Cultivation (`/Hectare) A2+FL']
z=SUGARCANE['Cost of Cultivation (`/Hectare) C2']
a=SUGARCANE['Cost of Production (`/Quintal) C2']
b=SUGARCANE['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("SUGARCANE Production Variation", fontsize=16, fontweight='bold')
plt.show()

plt.figure(figsize=(10,8))
x=WHEAT['State']
y=WHEAT['Cost of Cultivation (`/Hectare) A2+FL']
z=WHEAT['Cost of Cultivation (`/Hectare) C2']
a=WHEAT['Cost of Production (`/Quintal) C2']
b=WHEAT['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("WHEAT Production Variation", fontsize=16, fontweight='bold')
plt.show()


# To understand what has happened in those days, We need lots of data of climate, rain, GDP growth and many more.
# 
# Crop production depends on the availability of arable land and is affected in particular by yields, macroeconomic uncertainty, as well as consumption patterns; it also has a great incidence on agricultural commodities' prices. The importance of crop production is related to harvested areas, returns per hectare (yields) and quantities produced.
# 
# Now We have limited data So we can understand only the variation in Graph But To understand reasons Why graphs looks like this, We need lots of data. We will do it in the upcoming Kernels.

# In[ ]:


import seaborn as sns
sns.pairplot(data=data1,kind="reg")


# 

# Now We had another dataset. 

# In[ ]:


data2=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (2).csv')
data2.head()


# We had dataset having data 

# In[ ]:


data2.describe().T


# In[ ]:


Data2=data2.describe().T


# In[ ]:


dat=Data2.loc[:,"mean"]
dat.plot(kind='bar')


# In[ ]:


data2.head()


# In[ ]:


sns.pairplot(data=data2,kind="reg",y_vars=('Production 2006-07','Production 2007-08','Production 2008-09','Production 2009-10','Production 2010-11'),x_vars=('Area 2006-07','Area 2007-08','Area 2008-09','Area 2009-10','Area 2010-11'))


# In[ ]:


data3=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (3).csv')
data3.groupby('Crop')


# In[ ]:


data4=pd.read_csv('../input/agricuture-crops-production-in-india/datafile.csv')
data4


# In[ ]:


data5=pd.read_csv('../input/agricuture-crops-production-in-india/produce.csv')


# In[ ]:


data5.T

