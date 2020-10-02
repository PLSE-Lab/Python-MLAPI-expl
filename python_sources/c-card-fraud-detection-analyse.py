#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


CreditData = pd.read_csv("../input/creditcard.csv")


# In[ ]:


CreditData.describe()


# In[ ]:


CreditData.shape


# In[ ]:


print(CreditData.Amount.value_counts(dropna = False)) # Nan values comes too. But here there is no Nan value.


# In[ ]:


CreditData.boxplot(column = 'Time', by = 'Class')


# In[ ]:


CreditHead = CreditData.head()


# In[ ]:


MeltedCredit = pd.melt(frame = CreditHead, id_vars="Time", value_vars = ["Class","Amount"])
MeltedCredit


# In[ ]:


# MeltedCredit.pivot(index = 'Class', columns = 'variable',values='value')


# In[ ]:


# First and last 5 values.
CreditTail = CreditData.tail()
conc_data_row = pd.concat([CreditHead,CreditTail],axis= 0 , ignore_index = True )
conc_data_row


# In[ ]:


Credit1 = CreditData["Amount"].head()
Credit2 = CreditData["Class"].head()
conc_data_col = pd.concat([Credit1,Credit2],axis= 1 )
conc_data_col


# In[ ]:


CreditData.dtypes


# In[ ]:


CreditData["V1"] = CreditData["V1"].astype("category")
CreditData.V1.dtype
CreditData.V2 = CreditData.V2.astype("int")
CreditData.V2.dtype
# CreditData.V2


# In[ ]:


#CreditData.info() #There aren t any Nan values.
CreditData.V1[0:20] = np.nan
CreditData3 = CreditData.copy()
#np.isnan(CreditData.V1[0]) # is 0. value nan ? True
#so now we have a nan value 
CreditData2 = CreditData.copy()
#CreditData2Head = CreditData2.head()
CreditData2.V1.value_counts(dropna=False) #we have 20 nan values
CreditData2.V1.dropna(inplace = True)
assert CreditData2.V1.notnull().all() # there is nothing return so we drop nan values.
CreditData2.V1.head(10)


# In[ ]:


V1 = [1.35, 1.47]
Time = [0 , 1]
Class = [0, 1]
Amount = [1456 , 1453]
list_label = ["V1","Time","Class","Amount"]
list_column = [Time, V1, Class, Amount ]
zipped = list(zip(list_label,list_column))
MakeDict = dict(zipped)
SampleCredit = pd.DataFrame(MakeDict)
# So now we've made a basic dataframe from list to dict and from there to dataframe
SampleCredit["New_Feature"] = ["First","Second"]
SampleCredit


# In[ ]:


CreditData4 = CreditData.head(50).loc[:,["Time","Amount"]]
CreditData4.plot()


# In[ ]:


CreditData4.plot(subplots= True)
plt.show()


# In[ ]:


CreditData4.plot(kind = "scatter", x = "Amount", y= "Time")
plt.show()


# In[ ]:


CreditData4.plot(kind = "hist", y= "Amount",bins = 50,range = (0,50))
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows = 2 , ncols= 1)
CreditData4.plot(kind = "hist",color = "black",y = "Amount",bins = 50, range=(0,50),normed = True, ax = axes[0])
CreditData4.plot(kind = "hist",y = "Amount",bins = 50,color = "green", range=(0,50),normed = True, ax = axes[1],cumulative = True)
plt.savefig("graphh.jpg")
plt.show()


# In[ ]:


date_list = ["07.11.2018","08.11.2018","09.11.2018","10.11.2018","11.11.2018"]
date_time_obj = pd.to_datetime(date_list)
CreditHead["Date"] = date_time_obj
CreditHead = CreditHead.set_index("Date")
print(CreditHead.loc["2018-07-11":"2018-09-11"])


# In[ ]:


CreditHead.resample("A").mean()


# In[ ]:


CreditHead.resample("M").mean()


# In[ ]:


CreditHead.resample("M").first().interpolate("linear") # fills empty sections.


# In[ ]:


Credit_Data = pd.read_csv("../input/creditcard.csv")
Credit_Data[["Time","Amount"]]
# Let's find greatest value of amount according as time
Filter1 = ((Credit_Data.Amount > 250) & (Credit_Data.Time < 40))
Credit_Data[Filter1]


# In[ ]:


Credit_Data.Time[Credit_Data.Amount<2] #Time values that amounts' less than 2 


# In[ ]:


def div(n):
    return n*(1.5)
Credit_Data.V1.apply(div)

#Credit_Data.V1.apply(lambda x : x*1.5)


# In[ ]:


Credit_Data["V1+V2"] = Credit_Data.V1 + Credit_Data.V2
Credit_Data.head()


# In[ ]:


#print(Credit_Data.index.name)
Credit_Data.index.name = "Index name"
CreditData4.index = range(100,150,1)
CreditData4


# In[ ]:


#we will make two index 
Credit_Data = Credit_Data.set_index(["V1","V2"])
Credit_Data.head()


# In[ ]:


Credit4Drop = pd.read_csv("../input/creditcard.csv")
DroppedCredit = Credit4Drop.drop(columns=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"])
DroppedCredit


# In[ ]:


DroppedCredit.unstack(level=0)


# In[ ]:


DroppedCredit.groupby("Time").mean()


# In[ ]:


DroppedCredit.groupby("Time").Amount.max()
#max(DroppedCredit.Amount)


# In[ ]:




