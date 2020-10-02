#!/usr/bin/env python
# coding: utf-8

# ## NYS Waterbody Classifications

# ### From New York State Open Data

# <br>

# ### <font color=#5184d6> Kaggle</font>: https://www.kaggle.com/new-york-state/nys-waterbody-classifications

# <br>
# Author: Jagadeesh Kotra (hello@jagadeesh.me)
# 

# <br>

# <img src="https://i.imgur.com/r4OoKOW.png" alt="Lake Champlain" title="Lake Champlain" />

# In[ ]:


from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../input/waterbody-classifications.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# **We have 4(Four) types of water bodies in this dataset:**
# -  Estuary
# -  Ponds
# -  Shoreline
# -  Streams
# 

# In[ ]:


water_type = df.groupby('Waterbody Type')
for typ,group in water_type:
    print(typ)


# In[ ]:


#Let's find out how many...

Estuary = df[df['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Ponds = df[df['Waterbody Type'] == 'Ponds']['Waterbody Type'].count()
Shoreline = df[df['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Streams = df[df['Waterbody Type'] == 'Streams']['Waterbody Type'].count()

print('Estuary:',Estuary)
print('Ponds:',Ponds)
print('Shoreline:',Shoreline)
print('Streams:',Streams)


# In[ ]:


##Lets visualize it with a bar chart.

plt.figure(figsize=(15,8))  #setting the size
sns.barplot(x=['Streams','Ponds','Estuary','Shoreline'],y=[Streams,Ponds,Estuary,Shoreline])


# In[ ]:


#Max Segment Miles
df[df['Segment Miles'] == df['Segment Miles'].max()]


# In[ ]:


#Min Segment Miles
df[df['Segment Miles'] == df['Segment Miles'].min()]


# <br>
# #### <u>Per-Basin Analysis</u>

# In[ ]:


basin = df.groupby('Basin')
for basin_n,basin_g in basin:
    print(basin_n)


#  <br>
#  -- Oops,There is a typo,2nd and 3rd basin's are actually the same = Atlantic Ocean/Long Island Sound
# 
#     Lets fix it!

# In[ ]:


df = df.replace(to_replace='Atlantic Ocean/Long Island Soun',value='Atlantic Ocean/Long Island Sound')


# In[ ]:


#Lets try again!

basin = df.groupby('Basin')
for basin_n,basin_g in basin:
    print(basin_n)


# #### We have 17 basin's!

# ## Basin Aalysis - Allegheny River

# In[ ]:


AR = basin.get_group('Allegheny River')

Estuary = AR[AR['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Ponds = AR[AR['Waterbody Type'] == 'Ponds']['Waterbody Type'].count()
Shoreline = AR[AR['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Streams = AR[AR['Waterbody Type'] == 'Streams']['Waterbody Type'].count()

print('Estuary:',Estuary)
print('Ponds:',Ponds)
print('Shoreline:',Shoreline)
print('Streams:',Streams)


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(x=['Streams','Ponds','Estuary','Shoreline'],y=[Streams,Ponds,Estuary,Shoreline])


# <br>
# ## <u>Water Quality Analysis</u>

# <img src="https://i.imgur.com/badGrTD.png" alt="Water Quality" title="Water Quality" />

# In[ ]:


waterquality = df.groupby('Water Quality Class')

#Fresh.
b1 = waterquality.get_group('A')['Name'].count()
b2 = waterquality.get_group('AA')['Name'].count()
b3 = waterquality.get_group('A-S')['Name'].count()
b4 = waterquality.get_group('AA-S')['Name'].count()
b5 = waterquality.get_group('B')['Name'].count()
b6 = waterquality.get_group('C')['Name'].count()
b7 = waterquality.get_group('D')['Name'].count()

#Saline
s1 = waterquality.get_group('SA')['Name'].count()
s2 = waterquality.get_group('SB')['Name'].count()
s3 = waterquality.get_group('SC')['Name'].count()
s4 = waterquality.get_group('I')['Name'].count()
s5 = waterquality.get_group('SD')['Name'].count()

print(b1,b2,b3,b4,b5,b6,b7)
print(s1,s2,s3,s4,s5)

sum_of_fresh = b1+b2+b3+b4+b5+b6+b7
sum_of_saline = s1+s2+s3+s4+s5

plt.figure(figsize=(15,8))
sns.barplot(x=['Fresh','Saline'],y=[sum_of_fresh,sum_of_saline],palette="rocket")


# Hope you got a good first impression's about the dataset ;) keep plotting!
