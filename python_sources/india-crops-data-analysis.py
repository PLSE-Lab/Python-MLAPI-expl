#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


crops_prod_data = pd.read_csv("../input/datafile.csv",index_col='Crop')
print(crops_prod_data.info())


# In[ ]:


print(crops_prod_data)


# This dataset  contains production growth data for different crops in India from 2004-05 to 2011-12

# In[ ]:


crops_prod_data = crops_prod_data.dropna()
print(crops_prod_data)


# In[ ]:


# plots for crops
crops_prod_data.loc['Rice',:].plot()
crops_prod_data.loc['Milk',:].plot()
crops_prod_data.loc['All Agriculture',:].plot()
plt.legend(loc='upper left')


# You can see how Rice production has decreased in comparision to all agriculture

# In[ ]:


print(crops_prod_data.index)


# In[ ]:


print(crops_prod_data.T)


# In[ ]:


fig_size = plt.rcParams["figure.figsize"]
 
print("Current size:", fig_size)
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
crops_prod_data.T.plot(subplots=True,layout=(3,4))
plt.xticks(rotation=45)
plt.tight_layout()


# You can see from plots. Milk has exponential increase in production from 2008-09 to 2010-11. Rice, wheat, vegetables and pulses had steep decline in production from 2010-11 to 2011-12. Overall for all agriculture the production has increased from 2004-05 to 2011-12.

# In[ ]:


cultivation_data = pd.read_csv("../input/datafile (1).csv")
print(cultivation_data.info())


# In[ ]:


print(cultivation_data.head())
print(cultivation_data.columns)


# In[ ]:


cultivation_data['Per Hectare Cost Price'] = cultivation_data['Cost of Production (`/Quintal) C2'] * cultivation_data['Yield (Quintal/ Hectare) ']
cultivation_data['Cost of cultivation per hectare'] = cultivation_data['Cost of Cultivation (`/Hectare) A2+FL'] + cultivation_data['Cost of Cultivation (`/Hectare) C2']
cultivation_data['Yield in Kg per hectare'] = cultivation_data['Yield (Quintal/ Hectare) '] * 100
print(cultivation_data.head())


# In[ ]:


print(cultivation_data.T.head())


# In[ ]:


print(cultivation_data.Crop.value_counts())


# In[ ]:


print(cultivation_data.columns)
columns = ['Crop','State','Yield (Quintal/ Hectare) ']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Yield (Quintal/ Hectare) ')
table = table.fillna(0)
table.plot(kind='bar',stacked=True,colormap='Paired')
plt.ylabel('Yield (Quintal/ Hectare)')


# This bar chart provides information of yield of different crops. As yield of different crops vary a lot, sugarcane yield (Quintal/ Hectare) is much more as compared to other crops so it makes this plot much less informative. We will make this plot state wise to make it more informative.

# In[ ]:


table.T.plot(kind='bar',stacked=True)
plt.ylabel('Yield (Quintal/ Hectare) ')
plt.legend(loc='best')


# The above plot makes more sense than the previous plot. Now you can see Andra Pradesh has data on maximum number of crops and Tamil Nadu has maximum yield of Sugarcane.

# In[ ]:


columns = ['Crop','State','Yield in Kg per hectare']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Yield in Kg per hectare')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)
labels = []
for j in table.T.columns:
    for i in table.T.index:
        label = round((int(table.T.loc[i][j])),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2.,y + height/2.,label, ha='center',va='center')

plt.ylabel('Yield (Kg/ Hectare)')
plt.legend(loc='best')


# In this plot we added values of yield as kg/Hectare. This information helps us to compare yield of different crops across different states. Due to large number of values, the plot is not formatted well.

# In[ ]:


columns = ['Crop','State','Per Hectare Cost Price']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Per Hectare Cost Price')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)
labels = []
for j in table.T.columns:
    for i in table.T.index:
        label = round((int(table.T.loc[i][j])/10000),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2.,y + height/2.,label, ha='center',va='center')

plt.ylabel('Per Hectare Cost Price')


# The above plot helps us to compare per hectare cost of different crops across different states. For formatting reasons, cost values are added as factors (actual value/10000 rounded to one digit). You can clearly see Uttar pradesh has lowest per hectare cost for sugarcane but it has lowest yield as well (infering from previous plot).

# In[ ]:


columns = ['Crop','State','Cost of cultivation per hectare']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Cost of cultivation per hectare')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)

labels = []
for j in table.T.columns:
    for i in table.T.index:
        label = round((int(table.T.loc[i][j])/10000),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
#    if width > 0:
    x = rect.get_x()
    y = rect.get_y()
    height = rect.get_height()
    ax.text(x + width/2.,y + height/2.,label,ha='center', va='center')
    
plt.ylabel('Cost of cultivation per hectare')


# The above plot helps us to compare cost of cultivation of different crops across different states. For formatting reasons, cost values are added as factors (actual value/10000 rounded to one digit). You can clearly see Karnataka has lowest cost of cultivation for Groundnut.

# In[ ]:


crop_data = pd.read_csv("../input/datafile (2).csv")
print(crop_data.info())


# In[ ]:


print(crop_data.columns)


# As you can see Crop column name is having lot of space so it needs to be formatted

# In[ ]:


crop_data['Crop'] = crop_data['Crop             ']
del crop_data['Crop             ']
print(crop_data.columns)


# As there is data available on three different attributes (Production, Area and Yield). We divide our dataframe into three different dataframes, setting Crop column as index in each one.

# In[ ]:


columns = ['Crop','Production 2006-07', 'Production 2007-08', 'Production 2008-09','Production 2009-10', 'Production 2010-11']
columns2 = ['Crop','Area 2006-07', 'Area 2007-08', 'Area 2008-09','Area 2009-10', 'Area 2010-11']
columns3 = ['Crop','Yield 2006-07', 'Yield 2007-08', 'Yield 2008-09','Yield 2009-10', 'Yield 2010-11']
production = crop_data[columns]
area = crop_data[columns2]
yd = crop_data[columns3]
print(production.head())
production.index = production['Crop']
area.index = area['Crop']
yd.index = yd['Crop']
del production['Crop']
del area['Crop']
del yd['Crop']


# In[ ]:


print(production.T.columns)


# In[ ]:


fig_size = plt.rcParams["figure.figsize"]
 
# Prints: current size
print("Current size:", fig_size)
fig_size[0] = 18
fig_size[1] = 18
plt.rcParams["figure.figsize"] = fig_size
i=0
ax = production.T.plot(subplots=True,layout=(11,5),color='red',label='Production',legend=False)
for a in ax.flat:
    a.set_title(production.index[i])
    i += 1
ax1 = area.T.plot(subplots=True,layout=(11,5),ax=ax, linestyle=':',marker='.',color='blue',legend=False)
ax2 = yd.T.plot(subplots=True,layout=(11,5),ax=ax, linestyle='--',marker="*",color='green',legend=False)
labels = ['Production','Area','Yield']
plt.xticks(np.arange(5),('2006-07','2007-08','2008-09','2009-10','2010-11'))
plt.legend(labels=labels,loc='upper center', bbox_to_anchor=(-0.5, -0.5),  shadow=True, ncol=3, fontsize = 'xx-large')

plt.tight_layout()


# 
