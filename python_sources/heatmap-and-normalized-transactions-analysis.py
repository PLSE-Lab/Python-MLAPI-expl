#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head(10)


# # Extracting Year, Month, Day, Hour, and Minute to Seprarate columns

# In[ ]:


df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day
df['hour'] = pd.DatetimeIndex(df['Time']).hour
df['minute'] = pd.DatetimeIndex(df['Time']).minute
df.head()


# # Normalization Process

# In[ ]:


df_Normalized = df.groupby(['Transaction', 'year', 'month', 'day', 'hour', 'minute'])['Item'].count().reset_index()
df_Normalized.head(20)


# ### Total Number of transactions provided

# In[ ]:


df_Normalized.count()


# In[ ]:


df_Normalized.tail()


# ## Graphs based on Months and number of Transactions

# In[ ]:


import seaborn as sns
sns.countplot(data=df_Normalized, x="month")


# ### Comparing it with the number of items graphs

# In[ ]:


sns.countplot(data=df, x="month")


# Monstly number of items depict number of transactions per month except for the Fabruary where more items are bought but in lesser transactions.
# 
# We will be doing same comparisons for other graphs as well

# ## Transaction by Hour on any given day

# In[ ]:


sns.countplot(data=df_Normalized, x="hour")


# In[ ]:


sns.countplot(data=df, x="hour")


# #### Result: Transactions done between 1 and 3 pm has most items

# # Heatmaps

# ### Between Items and Hours

# #### Data Preparation

# In[ ]:


df_ItemHour = df.groupby(['hour', 'Item'])['Transaction'].count().reset_index()
df_ItemHour.head(10)


# In[ ]:


df_ItemHour_Pivot = df_ItemHour.pivot(index='Item', columns='hour', values='Transaction')
df_ItemHour_Pivot = df_ItemHour_Pivot.fillna(0)
df_ItemHour_Pivot.head(10)


# In[ ]:


import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,80))
sns.heatmap(df_ItemHour_Pivot, vmin=5, vmax=200, yticklabels=True, ax=ax, cmap="YlGnBu")


# #### Since in the above graph we can see most of the items are not sold frequently, lets focus on items that are frequent in transactions

# In[ ]:


df_ItemHour_Focused = df_ItemHour_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
df_ItemHour_Focused


# In[ ]:


fig,ax=plt.subplots(figsize=(10,20))
sns.heatmap(df_ItemHour_Focused, vmin=3, vmax=200, yticklabels=True, ax=ax, cmap="YlGnBu")


# ### Heat Map Between Items and Day of Month****

# #### Data Preparation 

# In[ ]:


df_ItemDay = df.groupby(['day', 'Item'])['Transaction'].count().reset_index()
df_ItemDay.head(10)


# In[ ]:


df_ItemDay_Pivot = df_ItemDay.pivot(index='Item', columns='day', values='Transaction')
df_ItemDay_Pivot = df_ItemDay_Pivot.fillna(0)
df_ItemDay_Focused = df_ItemDay_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
df_ItemDay_Focused


# #### Graph

# In[ ]:


fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df_ItemDay_Focused, vmin=5, vmax=150, yticklabels=True, ax=ax, cmap="YlGnBu")


# ### Heatmap between Day of Week and Items

# #### Data Preparation

# #### First Step is to make Extract Day of Week from Date Column 

# In[ ]:


df['my_dates'] = pd.to_datetime(df['Date'])
df['day_of_week'] = df['my_dates'].dt.weekday_name
df.head(10)


# In[ ]:


sns.countplot(data=df, x="day_of_week")


# #### Make Pivots for Heatmap

# In[ ]:


df_ItemDayOfWeek = df.groupby(['day_of_week', 'Item'])['Transaction'].count().reset_index()
df_ItemDayOfWeek.head(10)


# In[ ]:


df_ItemDayOfWeek_Pivot = df_ItemDayOfWeek.pivot(index='Item', columns='day_of_week', values='Transaction')
df_ItemDayOfWeek_Pivot = df_ItemDayOfWeek_Pivot.fillna(0)
df_ItemDayOfWeek_Focused = df_ItemDayOfWeek_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
column_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
df_ItemDayOfWeek_Focused = df_ItemDayOfWeek_Focused.reindex(column_order, axis=1)
df_ItemDayOfWeek_Focused


# In[ ]:


fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df_ItemDayOfWeek_Focused, vmin=5, vmax=250, yticklabels=True, ax=ax, cmap="YlGnBu")


# In[ ]:




