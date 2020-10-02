#!/usr/bin/env python
# coding: utf-8

# **Overview**
# 
# This is my first shared kernel. I used e-stat's API which has a lot of data related to restraunts. Its entirely in Japanese, so if you don't know how to run an XML file through a translator it may be difficult to utilize.
# 
# From this API I made maps of Japan showing different types of restrauntsby prefecture. I was surprised to see how this industry differs so much regionally!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import geopandas
import os
print(os.listdir("../input"))
tree = ET.parse('../input/japan-restaurant-database/getStatsData.xml')
file = open('../input/japan-restaurant-database/getStatsData.xml')
root = tree.getroot()


# I am thinking of making other kernels based on this one in the future. So I created a legend for all variables even the ones held as constants in this kernel.

# In[ ]:


# I made some dictionaries that have the corrisponding codes for XML queries. Some will not be referenced in this program.
l1 = {}; l2 = {}; l3 = {}; l4 = {}
p=[l1,l2,l3,l4]
atr=['tab','cat01','cat02','area']
for k in range(4):
    for i in root.findall(".//CLASS_OBJ[@id='{}']//".format(atr[k])):
        p[k][i.attrib.get('name')]=i.attrib.get('code')
        
# Here I printed out a list of restraunt type here and will use their codes for the querrying.
print(l3)
prefectureNames = list(l4.keys())


# Here is a function I created to make querries to pull values and make a map

# In[ ]:


#I didn't use 'area' for my querry as that will be variable
def createMap(code):
    code = str(code)
    rest = [i for i in root.findall(".//VALUE[@tab='811'][@cat01='000'][@cat02='{}']".format(code))]
    u=[int(i.text) for i in rest]
    df2 = pd.read_csv('../input/populations/pop2.csv')
    df3 = pd.read_csv('../input/cnterdfk/CTD.csv')
    pops = df2['pop'].tolist()
    popsint = [int(i.replace(',', '')) for i in pops]
    final = pd.DataFrame({"Population" : popsint[1:],
                          "nam_ja" : prefectureNames[1:],
                          "Value" : u[1:]})
    final['PerCapita'] = (final.Value/final.Population)
    final.set_index('nam_ja', inplace=True)
    Japan = geopandas.read_file('../input/japanmap/japan.geojson')
    Japan.set_index('nam_ja', inplace=True)
    Japan['per capita'] = final.PerCapita
    Japan.plot(column = 'per capita', figsize = (20,20), cmap = 'viridis')


# # Okonomiyaki

# In[ ]:


createMap(17940)


# # Chinese Restaurants

# In[ ]:


createMap(17760)


# # Hamburger Restraunts

# In[ ]:


createMap(17930)


# # Bars, Cabaret Clubs, and Nightclubs

# In[ ]:


createMap(17880)


# # Izakaya Pubs, Beer Halls

# In[ ]:


createMap(17860)


# # Udon/Soba Shops

# In[ ]:


createMap(17820)


# # Ramen Shops

# In[ ]:


createMap(17770)


# # Cafe/Teahouse

# In[ ]:


createMap(17910)

