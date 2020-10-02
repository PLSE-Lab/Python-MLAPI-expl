#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:


#insert dataset
df=pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding='ISO-8859-1')
df.head(10)


# In[ ]:


month=df["month"].unique().tolist()
month


# Merubah menjadi bahasa inggris dengan menggunakan label encoder

# In[ ]:


bulan=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
for i in range(0, len(bulan)):
    df['month'][df['month']==month[i]]=bulan[i]
df.head()


# In[ ]:


#Check Missing Values
df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


print(df.groupby(['state']).count())
state_plot=sns.countplot(x='state',data=df)
state_plot.set_xticklabels(state_plot.get_xticklabels(), rotation = 70)


# 1. RIO mencatat rekor tertinggi kebakaran hutan
# 2. Padaiba dan Mato Grosso mencatat rekor kebakaran hutan tertinggi kedua dan ketiga.

# In[ ]:


essential_data=df.groupby(by=['year','state','month']).sum().reset_index()
essential_data


# In[ ]:


from matplotlib.pyplot import MaxNLocator,FuncFormatter
plt.figure(figsize=(10,5))
plot=sns.lineplot(data=essential_data,x='year',y='number',markers=True)
plot.xaxis.set_major_locator(plt.MaxNLocator(19))
plot.set_xlim(1998,2017)


# 1. 2003 menunjukkan jumlah tertinggi kebakaran hutan.
# 2. 2008 dan 2001 menunjukkan jumlah kebakaran hutan yang serupa.
# 3. Tingkat kebakaran hutan meningkat selama tahun 1998 hingga 2003, 2008-      2009, 2011-2012, 2013-2018
# 4. Tingkat kebakaran hutan semakin menurun selama tahun 2003 hingga 2008,      2009-2011, 2012-2013

# Shall we split the dataset into increasing and decreasing number of forest fires and try to find some inner meanings from them ?

# In[ ]:


year_number_data=essential_data[['year','number']]
year_number_data[year_number_data['year']==1998]


# Tingkat kebakaran hutan meningkat selama tahun 1998 hingga 2003, 2008-2009, 2011-2012, 2013-2017
# 
# 
# Tingkat kebakaran hutan berkurang selama tahun 2003 hingga 2008, 2009-2011, 2012-2013

# In[ ]:


increasing_list = [1998, 1999, 2000, 2001, 2002, 2008, 2011, 2013, 2014, 2015]
decreasing_list = [2003, 2004, 2005, 2006, 2007, 2009, 2010, 2012, 2016]

increasing_dataframe = pd.DataFrame()
for i in increasing_list:
    df = year_number_data[year_number_data['year'] == i]
    increasing_dataframe = increasing_dataframe.append([df])
increasing_dataframe.head()


# In[ ]:


decreasing_dataframe = pd.DataFrame()
for i in decreasing_list:
    df1 = year_number_data[year_number_data['year'] == i]
    decreasing_dataframe = decreasing_dataframe.append([df1])
decreasing_dataframe.head()


# In[ ]:


plt.figure(figsize=(10,5))
plot=sns.lineplot(data=increasing_dataframe,
                 x='year',
                 y='number',
                 lw=1,
                 err_style="bars",
                 ci=100)
plot=sns.lineplot(data=decreasing_dataframe,
                 x='year',
                 y='number',
                 lw=1,
                 err_style="bars",
                 ci=100)
plot.xaxis.set_major_locator(plt.MaxNLocator(19))
plot.set_xlim(1998,2017)


# Kami membagi tahun-tahun menjadi peningkatan jumlah kebakaran dan mengurangi jumlah kebakaran. Kita dapat menyimpulkan bahwa jumlah kebakaran semakin meningkat selama beberapa dekade terakhir (bahkan jika kita mempertimbangkan penurunan jumlah kebakaran selama beberapa tahun, itu tidak menang atas peningkatan jumlah kebakaran untuk tahun-tahun yang tersisa). Ada kebutuhan serius untuk mempertimbangkan menyelamatkan hutan Brasil.
