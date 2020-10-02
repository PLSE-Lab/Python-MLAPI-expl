#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/Pokemon/Pokemon.csv')


# In[ ]:


df.columns = df.columns.str.upper().str.replace('_', '')
df.head()
df = df.set_index('NAME')


# I changed all the attributes to upper case for simplicity and defined columns. we set index as NAME

# In[ ]:


df.index = df.index.str.replace(".*(?=Mega)", "")
df=df.drop(['#'],axis=1)
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True)


# This code is to clear the data set, some values in type 2 are missing so we fill them with null values and clear all the words before "mega" under "type" attribute. we also drop "#" column because it is unnessisary for us.

# In[ ]:


data=df
data.drop(["TYPE 1", "TYPE 2","TOTAL","GENERATION","LEGENDARY"], axis = 1, inplace = True)
Attributes =list(data)
AttNo = len(Attributes)
name = input("Enter name of first pokemon : ") 
print(name)
values = data.loc[name].tolist()
values += values [:1]

angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]
name2 = input("Enter name of second pokemon: ") 
print(name2)
values2 = data.loc[name2].tolist()
values2 += values2 [:1]

angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles2 += angles2 [:1]

ax = plt.subplot(111, polar=True)

plt.xticks(angles[:-1],Attributes)

ax.plot(angles,values)
ax.fill(angles, values, 'blue', alpha=0.1)

ax.plot(angles2,values2)
ax.fill(angles2, values2, 'red', alpha=0.1)

#Rather than use a title, individual text points are added
plt.figtext(0.2,0.9,name,color="red")
plt.figtext(0.2,0.85,"vs")
plt.figtext(0.2,0.8,name2,color="blue")
plt.show()


# this code asks the user to input two valid names of pokemon, the different attributes such as HP,ATTACK,DEFENSE,SPECIAL DEFENSE, SPECIAL ATTACK and SPEED are compared among the two pokemon using a radial plot.
