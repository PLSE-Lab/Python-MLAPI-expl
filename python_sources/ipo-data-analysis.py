#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("/kaggle/input/financial-ipo-data/IPODataFull.csv",encoding="ISO-8859-1")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


fig,ax=plt.subplots()

plt.bar(df['dayOfWeek'].value_counts().index,df['dayOfWeek'].value_counts().values)
ax.set_xticklabels(['','Monday','Tuesday','Wednesday','Thursday','Friday'])
plt.xlabel("Days of Week")
plt.ylabel("Frequency")
plt.title("What day stocks go public on")
plt.show()


# In[ ]:


plt.bar(df['Year'].value_counts().index,df['Year'].value_counts().values)
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title("Amount of each stock in a year")
plt.show()


# Heat map is drwan to understand various correalations between the various parameters

# In[ ]:


fig=sns.heatmap(df[['DaysBetterThanSP', 'daysProfit', 'daysProfitGrouped',
       'exactDiffernce', 'Year', 'Month', 'Day', 'dayOfWeek', 'closeDay0','usableCEOAge', 'usableCEOGender', 'usablePresidentAge',
       'usablePresidentGender','FoundingDateGrouped', 'yearDifferenceGrouped',
       'Profitable', 'Safe', 'HomeRunDay', 'HomeRun']].corr(),
                annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Amounts of Stocks that Tripled in the first year itslef

# In[ ]:


fig,ax=plt.subplots()

plt.bar(df['HomeRun'].value_counts().index,df['HomeRun'].value_counts().values)
ax.set_xticklabels(['','','No-Triple','','','','Triple'])
plt.xlabel("Home Run Status")
plt.ylabel("Frequency")
plt.title("Amounts of stocks that tripled in value in the first year")
plt.show()


# Major industries to contribute to the Tripled the stocks 

# In[ ]:


plt.figure(figsize=(14,8))
df.groupby('Industry')         .size()         .sort_values(ascending = False)         .iloc[:5]         .plot.pie(explode=[0,0,0.1,0,0],autopct='%1.1f%%',shadow=True)
plt.ioff()


# Profit in grouped after IPO is processed

# In[ ]:


df.daysProfitGrouped.value_counts(normalize=True).sort_index().plot.bar()
plt.grid()
plt.title('daysProfitGrouped')
plt.xlabel('daysProfitGrouped')
plt.ylabel('Fraction')


# In[ ]:




