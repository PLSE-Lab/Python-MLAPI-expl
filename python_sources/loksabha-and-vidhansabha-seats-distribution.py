#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


numpy_list = np.array(os.listdir("../input"))
numpy_list


# In[3]:


my_list = np.arange(10)
my_list


# In[4]:


my_list2 = np.arange(1,11)
my_list2


# In[5]:


my_list3 = np.arange(1,11,2)
my_list3


# In[6]:


lin_arr = np.linspace(1, 3, 15)
lin_arr


# In[7]:


#my_series = pd.Series(numpy_list,lin_arr)
index_list = np.arange(1,3)
my_series = pd.Series(numpy_list,index_list)     # even if you don't pass index_list, pandas will create index starting from 0
my_series


# In[8]:


my_series[1]


# In[9]:


my_series[3] = "change_series_value"
my_series


# In[10]:


my_dataframe = pd.DataFrame(numpy_list,index_list)     # even if you don't pass index_list, pandas will create index starting from 0
my_dataframe  # each column here represents a panda Series, Hence, it is safe to say a DataFrame is a collection of Series sharing the same index.


# In[11]:


my_dataframe[0][1]      #[column][row]


# In[12]:


#csv = pd.read_csv(my_dataframe[0][1])
type(my_dataframe[0])


# In[13]:


type(my_dataframe[0][1])


# In[14]:


new_dataframe = pd.DataFrame(['lucknow','bhopal','mumbai','hyd','bangalore'],['UP','MP','Maharashtra','Telangana','Karnataka'],['capital'])  
new_dataframe


# In[15]:


new_series = {'Capital' : pd.Series(['lucknow','bhopal','mumbai','hyd','bangalore']
                                    ,['UP','MP','Maharashtra','Telangana','Karnataka'])}
type(new_series)


# In[16]:


new_dataframe2 = pd.DataFrame(new_series)  
new_dataframe2


# In[17]:


new_dataframe2['country'] = pd.Series(['India','India','India','India'],['UP','MP','Maharashtra','Telangana'])
pd.DataFrame(new_dataframe2)           #adding new column to the dataframe


# In[18]:


new_dataframe2.hist


# In[19]:


new_dataframe2['population'] = pd.Series([199812341,72626809,112374333,35003674,61095297],['UP','MP','Maharashtra','Telangana','Karnataka'])
pd.DataFrame(new_dataframe2)


# In[20]:


new_dataframe2.reset_index(inplace = True)              #reset_index deosn't permanently reset the index. For permanently, use reset_index(inplace=true)


# In[21]:


new_dataframe2


# In[22]:



new_dataframe2.set_index('index')        #this way you can set any column to work as index
new_dataframe2.columns


# In[23]:


new_dataframe2.columns[0]


# In[24]:


new_dataframe2.reset_index(inplace=True)
new_dataframe2.columns = ['level_0','State','Capital','Country','Population']
new_dataframe2


# In[25]:


new_dataframe2["State"].value_counts()      


# In[26]:


new_dataframe2.describe()


# In[27]:


#help.describe
new_dataframe2.hist()
#plt.show()


# In[28]:


new_dataframe2['NoOfLoksabhaConstituency'] = pd.Series([80,29,48,17,28],[0,1,2,3,4])
pd.DataFrame(new_dataframe2)      


# In[29]:


new_dataframe2.hist()


# In[30]:


#new_dataframe2.hist("Population")    wrong syntax
new_dataframe2["Population"].hist()


# In[31]:


new_dataframe2.corr()


# In[32]:


new_dataframe2['NoOfVidhansabhaConstituency'] = pd.Series([403,230,288,119,224])
new_dataframe2


# In[33]:


new_dataframe2.hist()


# In[34]:


new_dataframe2.corr()


# In[35]:


new_dataframe2['Country'].dropna(inplace = True)


# In[36]:


new_dataframe2


# In[37]:


new_dataframe2.loc[4]


# In[38]:


new_dataframe2.iloc[4]        #can get using index too


# In[39]:


#new_dataframe2.loc[5] = pd.Series()      #to add rows
for i in range(8):
    if i<5:
        continue
    new_dataframe2.loc[i] = pd.Series()    
new_dataframe2


# In[40]:


new_dataframe2.loc[1].loc['State']


# In[41]:


new_dataframe2.loc[1].loc['State']


# In[42]:


new_dataframe2.groupby('Country').mean()     #groupby group rows based on the 'Country' column and call the aggregate function .mean()on it


# In[43]:


new_dataframe2.groupby('Country').count()      #Using the count() method, we can get the number of times an item occurs in a DataFrame.


# In[44]:


new_dataframe2.groupby('Country').describe() 


# The .describe() method is used to get an overview of what a DataFrame looks like. It gives us a summary of each of the DataFrame index.

# In[45]:


new_dataframe2.plot()


# In[46]:


new_dataframe2.plot(kind='scatter', x=6,y='Population')


# In[47]:


new_dataframe2.corr()


# In[48]:


new_dataframe2.plot.area(alpha=0.1)


# In[49]:


new_dataframe2['Population'].plot(kind='kde')


# In[50]:


new_dataframe2['NoOfLoksabhaConstituency'].plot(kind='kde')


# In[51]:


new_dataframe2['NoOfVidhansabhaConstituency'].plot(kind='kde')


# In[52]:


new_dataframe2


# In[53]:


new_dataframe2['areaOfStates'] = pd.Series([243290,308350,307713,112077,191791])
new_dataframe2


# In[54]:


new_dataframe2.corr()


# In[55]:


new_dataframe2.loc[0].loc['Population']


# In[56]:


areaPlusPopulationList = np.arange(1,7)
for i in range(6):
    areaPlusPopulationSeries = pd.Series(areaPlusPopulationList)
    areaPlusPopulationSeries[i] = new_dataframe2.loc[i].loc['Population'] + new_dataframe2.loc[i].loc['areaOfStates']
new_dataframe2['areaPlusPopulation'] = pd.Series(areaPlusPopulationSeries)

new_dataframe2


# In[57]:


new_dataframe2.corr()


# Very imp stuff comes in front of us here(though this may be the case of Overfitting due to less data), when we see the correlation of  **NoOfLoksabhaConstituency**, we find that it's best correlated with population but when it comes to **NoOfVidhansabhaConstituency**, then it's best correlated with **areaPlusPopulation** which is area and population combined (Although the difference is very minimal). 

# Let's add more data to analyse this further.

# In[58]:


new_dataframe2


# In[59]:


seriesOfState = pd.Series(['Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh','Goa','Gujarat','Haryana',
                          'Himachal Pradesh','Jammu and Kashmir','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',
                          'Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu',
                           'Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal','NCT of Delhi','Puducherry'])
seriesOfNoOfLoksabhaConstituency = pd.Series([25,2,14,40,11,2,26,10,4,6,14,28,20,29,48,2,2,1,1,21,13,25,1,39,17,2,80,5,42,
                                        7,1])
seriesOfNoOfVidhansabhaConstituency = pd.Series([175,60,126,243,90,40,182,90,68,87,81,224,140,230,288,60,60,40,60,
                                                147,117,200,32,234,119,60,403,70,294,70,30])
seriesOfAreaOfState = pd.Series([160205,83743,78438,94165,135191,3702,196024,44212,55673,101387,79714,191791,38863,308350,
                                307713,22327,22429,21081,16579,155707,50362,342238,7096,130058,112077,10486,243290,53483,
                                88752,1483,492])
seriesOfPopulationOfState = pd.Series([49577103,1383727,31205576,104099452,25545198,1458545,60439692,25351462,6864602,
                                      12541302,32988134,61095297,33406061,72626809,112374333,2570390,2966889,1097206,1978502,
                                      41974218,27743338,68548437,610577,72147030,35003674,3673917,199812341,10086292,91276115,
                                      16787941,1247953])


# In[60]:


for i in range(31):
    new_dataframe2.loc[i] = pd.Series()    


# In[61]:


new_dataframe2['State'] = seriesOfState
new_dataframe2['Population'] = seriesOfPopulationOfState
new_dataframe2['NoOfLoksabhaConstituency'] = seriesOfNoOfLoksabhaConstituency
new_dataframe2['NoOfVidhansabhaConstituency'] = seriesOfNoOfVidhansabhaConstituency
new_dataframe2['areaOfStates'] = seriesOfAreaOfState


# In[62]:


new_dataframe2


# In[63]:


areaPlusPopulationList = np.arange(0,31)
for i in range(31):
    areaPlusPopulationSeries = pd.Series(areaPlusPopulationList)
    areaPlusPopulationSeries[i] = new_dataframe2.loc[i].loc['Population'] + new_dataframe2.loc[i].loc['areaOfStates']
new_dataframe2['areaPlusPopulation'] = pd.Series(areaPlusPopulationSeries)


# In[64]:


new_dataframe2


# In[65]:


new_dataframe2.corr()


# Now this data concludes what we had said earlier, **NoOfLoksabhaConstituency** is best correlated with **population** while **NoOfVidhansabhaConstituency** is best correlated with **areaPlusPopulation** if we take consider only **population** and **areaPlusPopulation** but here is another thing, **NoOfVidhansabhaConstituency** is ever more correlated with **NoOfLoksabhaConstituency**, it means as no of **loksabha seats** increased in a state, **vidhansabha seats** increases too. 
