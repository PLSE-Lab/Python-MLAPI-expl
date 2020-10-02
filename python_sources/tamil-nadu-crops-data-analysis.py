#!/usr/bin/env python
# coding: utf-8

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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read csv file as pandas dataframe
data = pd.read_csv("../input/tn.csv")
# checking contents of file
data.head()


# In[ ]:


# information of dataframe
data.info()


# We can see Production column has null values. Now check the different columns for unique values

# In[ ]:


data['Crop_Year'].value_counts().sort_values()


# In[ ]:


data['State_Name'].value_counts()


# As we can see State_Name value is Tamil Nadu for whole column so this column is redundant so we can delete this column

# In[ ]:


del data['State_Name']


# In[ ]:


# checking dataframe info after deletion of column
data.info()


# Question 1 : How many unique district data is available in the dataset ?

# In[ ]:


data.District_Name.nunique()


# Question 2 : How many unique crops data is available in the dataset ?

# In[ ]:


data['Crop'].str.upper().nunique()


# Question 3 : How many unique seasons data is available in the dataset ?

# In[ ]:


data['Season'].value_counts()


# Now let's look at each crop is cultivated in how many districts ?

# In[ ]:


# Get current size of figure
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: current size
print("Current size:", fig_size)
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 20
plt.rcParams["figure.figsize"] = fig_size
data.groupby(['Crop'])['District_Name'].nunique().sort_values().plot(kind='barh')
plt.xlabel("Number of districts")


# As you can see majority of crops are not grown in all districts. Now let's check which crop has max production ?

# In[ ]:


data.Production.max()


# In[ ]:


data[data.Production == 1250800000.0]


# Coimbatore has the maximum production for coconut in 2011 but it has huge area sown as well. let's check how Coimbatore performes in terms of Yield (Production per hectare). Here area unit is considered as Hectare.

# In[ ]:


Coconut_data_2011 = data[(data.Crop == 'Coconut ') & (data.Crop_Year == 2011)][['District_Name','Area','Production']]
Coconut_data_2011['Production per hectare'] = Coconut_data_2011['Production']/Coconut_data_2011['Area']
Coconut_data_2011['Production per hectare'].plot(kind = 'bar')
plt.xticks( np.arange(31), (Coconut_data_2011['District_Name']) )
plt.title("Coconut production per hectare districtwise in Tamil Nadu")
#plt.xticks(Coconut_data_2011['District_Name'])


# Thanjavur has max. yield of coconut in 2011. let's check the min. production of a crop.

# In[ ]:


data.Production.min()


# In[ ]:


len(data[data.Production == 0.0])


# We have 880 instances where production is zero. May be crop is spoiled in these cases. let's check average production ?

# In[ ]:


data.Production.mean()


# In[ ]:


data[data.Production >= 910330.4]['Crop'].unique()


# Only three crops has production greater than average production

# In[ ]:


data[data.Production <= 910330.4]['Crop'].nunique()


# Let's explore production of next high produce crop: Sugarcane 

# In[ ]:


ax = data[data.Crop == 'Sugarcane'].groupby('Crop_Year')['Production'].sum().div(100).plot()
data[data.Crop == 'Sugarcane'].groupby('Crop_Year')['Area'].sum().plot(ax=ax)
plt.xlabel("Year")
plt.ylabel("Production/100")
plt.legend(loc='best')
plt.title("Production of Sugarcane in Tamil Nadu in relation to area sown from 1997 to 2013")
#plt.ylabel("Sugarcane Production")


# According to data Sugarcane production levels has decreased since 1997. Let's explore another high produce crop : Tapioca using subplots.

# In[ ]:


Tapioca_data = data[data.Crop == 'Tapioca'][['District_Name','Crop_Year','Production']]
ax = Tapioca_data.pivot(index='District_Name', columns='Crop_Year', values='Production').T.plot(kind='bar',subplots=True,layout=(8,4),legend=False)


# These subplots gives us the power to analyse a crop data across different districts. Maximum districts shows a decline in  Tapioca production.
