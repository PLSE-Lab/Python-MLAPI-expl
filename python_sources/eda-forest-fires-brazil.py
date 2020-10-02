#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df= pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
df.isna().sum() # find out the number of missing values


# In[ ]:


df.head(10) 


# In[ ]:


# convert last column date to datetime type
df['date']= pd.to_datetime(df['date'])


# In[ ]:


# convert all month names from latin to english
latin_months= df['month'].unique().tolist()
eng_month= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_dict= dict(zip(latin_months, eng_month))
df.replace(month_dict, inplace=True)


# In[ ]:


df['year']= df['year'].astype('str') # converting year values from int to string
df.dtypes


# In[ ]:


df.head()


# In[ ]:


def c_plotting(df, xaxis, yaxis, title_str, xlabel_str, ylabel_str, fsize, fcolor, pcolor, xrot, yrot):
    plt.figure(figsize= (20, 8))
    plt.title(title_str, fontsize = fsize, color= fcolor)
    plt.xlabel(xlabel= xlabel_str, fontsize = 15, color= fcolor)
    plt.ylabel(ylabel= ylabel_str, fontsize = 15, color= fcolor)
    plt.yticks(rotation= yrot, color= fcolor)
    plt.xticks(rotation= xrot, color= fcolor)
    sns.lineplot(data= df, x= xaxis, y= yaxis, estimator= 'sum', color= pcolor, ci= None)
    plt.show()


# In[ ]:


tdf= df.groupby(by= ['year', 'month']).sum()
c_plotting(tdf, df['year'], df['number'], 'Forest fires in Brazil over the years (1998-2017)', 'Year', 'Fire Counts', 15, 'Red', 'Blue', 0, 0)


# <b> Highest number of fires took place in 2003, 2009, 2012 & 2016.</b> **Let us examine the states next.**

# In[ ]:


def d_plotting(df, xaxis, yaxis, title_str, xlabel_str, ylabel_str, fsize, fcolor, pcolor, xrot, yrot, order_list):
    plt.figure(figsize= (20, 8))
    plt.title(title_str, fontsize = fsize, color= fcolor)
    plt.xlabel(xlabel= xlabel_str, fontsize = 15, color= fcolor)
    plt.ylabel(ylabel= ylabel_str, fontsize = 15, color= fcolor)
    plt.yticks(rotation= yrot, color= fcolor)
    plt.xticks(rotation= xrot, color= fcolor)
    sns.barplot(data= df, x= xaxis, y= yaxis, color= pcolor, order= order_list)
    plt.show()


# In[ ]:


order_list= df['state'].unique().tolist()
d_plotting(tdf, df['state'], df['number'], 'Forest fire distribution in Brazilian states ', 'Year', 'Fire Counts', 15, 'Red', 'Green', 90, 0, order_list)


# <p>**Fires in Matto Grosso have been abnormally high. Some other states with higher number of forest fires are:**</p>
# * Bahia, Goias, Paraiba, Rio, Sao Paolo
# **<p>We shall examine these above states using a pivot chart.**</p>
# <p>**Next: Examining the number of forest fires over a single year in term of months.**</p>
# 

# In[ ]:


order_list= df['month'].unique().tolist()
d_plotting(tdf, df['month'], df['number'], 'Forest fires in Brazilian states over the months', 'Year', 'Fire Counts', 15, 'Yellow', 'Green', 90, 0, order_list)


# **Most forest fires take place in the year end months: From July to Dec.**
# <p> It is clear from above figure that mean of number of forest fires in the second half is greater than the mean of number of forest fires in the first half.</p>
# <p>Also lowest numbers of forest fires in the second half is almost equal to the highest numbers in the first half. </p>
# <p> **Thus: Forest fires are more probable to occur in the second half of the year.** </p>
# *Examining the states of Mato Grosso, Bahia, Goias, Paraiba, Rio, Sao Paolo in more detail next:*

# In[ ]:


df.groupby(by=['state'])['number'].sum().sort_values(ascending= False).head(6)


# In[ ]:


top_states= ['Mato Grosso', 'Bahia', 'Goias', 'Paraiba', 'Rio', 'Sao Paolo']
top_df= df[df['state'].isin(top_states)].groupby(by=['year', 'month', 'state']).sum().reset_index()

plt.figure(figsize= (20, 8))
plt.title('Highest forest fire numbers in Brazillian states in Years', fontsize = 15, color= 'Yellow')
plt.xlabel('Months', fontsize = 15, color= 'Red')
plt.ylabel('Counts', fontsize = 15, color= 'Red')
plt.yticks(rotation= 0, color= 'Red')
plt.xticks(rotation= 0, color= 'Red')
sns.lineplot(lw= 5, data= top_df, x= top_df['year'], y= top_df['number'], estimator= 'sum', color= 'Blue', ci= None, hue= top_df['state'])
plt.show()


# **Number of forest fires in Mato Grosso has always been greater than the other states. The difference is most visible in 2009. In recent times also, numbers for Mato Grosso is seeing upward trend. States other than Mato Grosso & Rio have decreasing number of forest fires in recent times.**

# In[ ]:


plt.figure(figsize= (20, 8))
plt.title('Highest forest fire numbers in Brazillian states in Months', fontsize = 15, color= 'Red')
plt.xlabel('Months', fontsize = 15, color= 'Red')
plt.ylabel('Counts', fontsize = 15, color= 'Red')
plt.yticks(rotation= 0, color= 'Red')
plt.xticks(rotation= 0, color= 'Red')
sns.lineplot(lw= 3, data= top_df, x= top_df['month'], y= top_df['number'], estimator= 'sum', color= 'Blue', ci= None, hue= top_df['state'])
plt.show()


# <b>A strange phenomenon is observed in Jul, Aug & Nov. Number of forest fires suddenly increase in all these states compared to the previous  months. </b>

# In[ ]:




