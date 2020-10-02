#!/usr/bin/env python
# coding: utf-8

# # TEMPERATURE DATA SET FOR 1901-2016

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv("../input/indian_temp.csv")
print(df)


# # Slice of 1901-1920 Temperature Data Set

# In[ ]:


df1 = pd.read_excel("../input/indian_temp.xlsx")
df1
df2 = df1.iloc[0:20,0:13]
df2


# In[ ]:


df2.shape


# In[ ]:


df2


# # Details of the Dataframe 1901-1920

# In[ ]:


df2.describe()


# # Table Of Monthly Temperatures for 20 years 

# ## JAN

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JAN',data=df2)


# We analyse the data frame to produce desirable plots for the month of JAN for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1907 has the maximum temperature in this season and year 1918 has the minimum temperature in this season.

# ## FEB

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='FEB',data=df2)


# We analyse the data frame to produce desirable plots for the month of FEB for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1912 has the maximum temperature in this season and year 1905 has the minimum temperature in this season.

# ## MAR

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAR',data=df2)


# We analyse the data frame to produce desirable plots for the month of MAR for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1916 has the maximum temperature in this season and year 1905 has the minimum temperature in this season.

# ## APR

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='APR',data=df2)


# We analyse the data frame to produce desirable plots for the month of APR for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1908 has the maximum temperature in this season and year 1905 has the minimum temperature in this season.

# ## MAY

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAY',data=df2)


# We analyse the data frame to produce desirable plots for the month of MAY for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1915 has the maximum temperature in this season and year 1917 has the minimum temperature in this season.

# ## JUN

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUN',data=df2)


# We analyse the data frame to produce desirable plots for the month of JUN for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1905 has the maximum temperature in this season and year 1917 has the minimum temperature in this season.

# ## JUL

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUL',data=df2)


# We analyse the data frame to produce desirable plots for the month of JUL for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1903 has the maximum temperature in this season and year 1909 has the minimum temperature in this season.

# ## AUG

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='AUG',data=df2)


# We analyse the data frame to produce desirable plots for the month of AUG for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1905 has the maximum temperature in this season and year 1907 has the minimum temperature in this season.

# ## SEP

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='SEP',data=df2)


# We analyse the data frame to produce desirable plots for the month of SEP for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1915 has the maximum temperature in this season and year 1909 has the minimum temperature in this season.

# ## OCT

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='OCT',data=df2)


# We analyse the data frame to produce desirable plots for the month of OCT for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1905 has the maximum temperature in this season and year 1917 has the minimum temperature in this season.

# ## NOV

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='NOV',data=df2)


# We analyse the data frame to produce desirable plots for the month of NOV for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1915 has the maximum temperature in this season and year 1910 has the minimum temperature in this season.

# ## DEC

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='DEC',data=df2)


# We analyse the data frame to produce desirable plots for the month of DEC for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1906 has the maximum temperature in this season and year 1910 has the minimum temperature in this month.

# # Table Of Seasonal Temperatures for 20 years 

# In[ ]:


df3 = df1[['YEAR','JAN-FEB','MAR-MAY','JUN-SEP','OCT-DEC']]
df3 = df3.iloc[0:20,]
df3


# We split the data set into set of 20 years and stored this into a new dataframe. Now we analyse this new data frame to produce desirable plots.

# ## JAN-FEB

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JAN-FEB',data=df3)


# We analyse the data frame to produce desirable plots for season JAN-FEB for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1912 has the maximum temperature in this season and year 1905 has the minimum temperature in this season.

# ## MAR-MAY
# 

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAR-MAY',data=df3)


# We analyse the data frame to produce desirable plots for season MAR-MAY for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1916 has the maximum temperature in this season and year 1907 has the minimum temperature in this season.

# ## JUN-SEP

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUN-SEP',data=df3)


# We analyse the data frame to produce desirable plots for season JUN-SEP for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1915 has the maximum temperature in this season and year 1909 has the minimum temperature in this season.

# ## OCT-DEC

# In[ ]:


plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='OCT-DEC',data=df3)


# We analyse the data frame to produce desirable plots for season OCT-DEC for 20 years.Seaborn barplot to represent this data for better visualisation. We conclude that year 1915 has the maximum temperature in this season and year 1917 has the minimum temperature in this season.

# # Correlation between Temperatures

# In[ ]:


df1.corr()


# In[ ]:


df4 = df1[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']]
df4


# # Slice of 1901-1920 temperature data set

# In[ ]:


df4 = df4.head(20)
df4


# # Correlation matrix for JAN-DEC and annual

# In[ ]:


df5=df4.corr()
df5


# # Heatmap for correlation matrix

# In[ ]:


plt.figure(figsize=(10,8))
sb.heatmap(df5,annot=True)


# We determine the correlation between the temperatures of different months in a year and we find that AUG-OCT has the maximum correlation with value is equals to 0.72 and FEB-OCT has minimum correlation with value is equals to -0.7.

# In[ ]:




