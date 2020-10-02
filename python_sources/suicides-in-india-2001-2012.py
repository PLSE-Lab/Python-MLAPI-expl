#!/usr/bin/env python
# coding: utf-8

# # **A complete Exploratory data analysis on the India's suicide data from 2001 - 2012**

# In[ ]:


import ipywidgets as widgets
from IPython.display import display
from ipywidgets import widgets, interactive
from ipywidgets import interact


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas_profiling
import statsmodels


# In[ ]:


df=pd.read_csv("../input/Suicides in India 2001-2012.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:


df.info(memory_usage='deep')


# In[ ]:


df.memory_usage(deep=True)


# In[ ]:


df.memory_usage(deep=True).sum()


# In[ ]:


df['State'] = df['State'].astype('category')
df['Type_code'] = df['Type_code'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Type'] = df['Type'].astype('category')
df['Age_group'] = df['Age_group'].astype('category')


# In[ ]:


df.dtypes


# In[ ]:


df['State'].head()


# In[ ]:


df.State.cat.codes.head()


# In[ ]:


df.memory_usage(deep = True)


# In[ ]:


df.info('deep')


# In[ ]:


df.memory_usage(deep = True).sum()


# ## Lets take a look at the summary of the Data

# In[ ]:


pandas_profiling.ProfileReport(df)


# Unfortunately the profile report is not being displayed in my Notebook due to some problem. but it would work in other Notebook.

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df_copy = df.copy()


# In[ ]:


df.plot()


# In[ ]:


df['Total'].plot()


# In[ ]:


df['Total'].max()


# In[ ]:


df[df["Total"]==63343]


# In[ ]:


df["State"].unique()


# In[ ]:


df_all_ind=df[df['State']=="Total (All India)"]
df_all_ind.head()


# In[ ]:


df_all_ind["Total"].sum()


# In[ ]:


df['Type_code'].unique()


# In[ ]:


df['Type'].unique()


# In[ ]:


df['Type'].nunique()


# In[ ]:


df.head()


# In[ ]:


df =df[df['Total']>0]


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['Type_code'].value_counts()


# In[ ]:


plt.title('Type Code')
df['Type_code'].value_counts().plot(kind='bar')


# In[ ]:


df['Type'].value_counts(normalize=True)


# In[ ]:


plt.title('Type of suicides')
df['Type'].value_counts(normalize=False).plot(kind='bar',figsize=(20,12))


# In[ ]:


df['Age_group'].value_counts()


# In[ ]:


plt.title('Age group')
df['Age_group'].value_counts().plot(kind='bar')


# In[ ]:


df['Gender'].value_counts(normalize=True)


# In[ ]:


plt.title('Male and Female')
df['Gender'].value_counts().plot(kind='bar')


# In[ ]:





# In[ ]:


df.head()


# In[ ]:





# In[ ]:


df_year_total = df.groupby('Year')['Total'].sum()


# In[ ]:


df_year_total = pd.DataFrame(df_year_total)


# In[ ]:


df_year_total.reset_index(inplace=True)


# In[ ]:


df_year_total.head()


# ### Total no of suicides by year from 2001 - 2012

# In[ ]:


fig =px.bar(df_year_total,x='Year',y='Total',labels={'Total':'No of suicides: '}, height=400,color='Year')
fig.update_layout(title_text='Total no of suicides by year from 2001 - 2012')
fig.show()


# In[ ]:





# In[ ]:


df_grnder = df.groupby('Gender')['Total'].sum()
df_grnder=pd.DataFrame(df_grnder)
df_grnder.reset_index(inplace=True)
df_grnder.head()


# ### Suicides of Male and Female from 2001 - 2012

# In[ ]:


fig=px.bar(df_grnder,x='Gender',y='Total',labels={'Total':'No of suicides: '},height=400,color='Gender')
fig.update_layout(title_text='Suicides of Male and Female from 2001 - 2012')
fig.show()


# In[ ]:


df_type=df.groupby(['Type','Gender'])['Total'].sum()
df_type=pd.DataFrame(df_type)
df_type.reset_index(inplace=True)
df_type.sort_values(by="Total" , inplace=True ,ascending=False)

df_type.head()


# ### List of types and total no of suicides

# In[ ]:


fig=px.bar(df_type,x='Type',y='Total',labels={'Total':'No of suicides: '},height=1000,width=1800,color='Gender',barmode='group')
fig.update_layout(title_text = 'List of types and total no of suicides')
fig.show()


# ### List of top 7 types of suicide's 

# In[ ]:


fig=px.bar(df_type[:7],x='Type',y='Total',labels={'Total':'No of suicides: '},height=600,width=1500,color='Gender',barmode='group')
fig.update_layout(title_text = 'List of top 7 types of suicides')
fig.show()


# ### States and Their Total no of suicide

# In[ ]:


df_state = df.groupby('State')['Total'].sum()
df_state = pd.DataFrame(df_state)
df_state.reset_index(inplace= True)
df_state.sort_values(by='Total',inplace=True,ascending = False)
df_state.head()


# In[ ]:


df_state.drop(df_state.index[[0,1]],inplace=True)


# In[ ]:


df_state.head()


# ### List of states and their total no of suicides

# In[ ]:


fig = px.bar(df_state,x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'List of states and their total no of suicides ')
fig.show()


# ### List of Top 7 States and their total no of suicides

# In[ ]:


fig = px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'List of Top 7 States and their total no of suicides')
fig.show()


# In[ ]:


df_state.sort_values(by='Total',inplace=True,ascending=True)
df_state.head()


# ### States with lowest number of suicides from 2001-2012

# In[ ]:


fig =px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'States with lowest number of suicides from 2001-2012')
fig.show()


# In[ ]:


df.head()


# In[ ]:


df_state_year = df.groupby(['State','Year','Gender'])['Total'].sum()
df_state_year = pd.DataFrame(df_state_year)
df_state_year.reset_index(inplace = True)
df_state_year.head(5)


# In[ ]:


df_Island_Andhra=df_state_year[df_state_year.State.isin(['A & N Islands','Andhra Pradesh'])]
df_Island_Andhra.head()


# ### Suicides in Andhra And A & N island over year's

# In[ ]:


fig=px.bar(df_Island_Andhra,x='Year',y='Total',facet_col='State',color='State')
fig.update_layout(title_text = 'Suicides in Andhra And A & N island over years')
fig.show()


# ### Suecides in Kerala and Tamil Nadu by types

# In[ ]:


df.head()


# In[ ]:


df_state_and_type=df.groupby(['State','Type'])['Total'].sum()
df_state_and_type = pd.DataFrame(df_state_and_type)
df_state_and_type.sort_values(by='Total',inplace=True,ascending =False)
df_state_and_type.reset_index(inplace=True)
df_state_and_type


# In[ ]:


df_state_and_type['State'].unique()


# In[ ]:


df_Kl_TN = df_state_and_type[df_state_and_type.State.isin(['Kerala','Tamil Nadu'])]
df_Kl_TN


# In[ ]:


df_Kl_TN.Type.unique()


# In[ ]:


df_Kl_TN=df_Kl_TN[df_Kl_TN['Total']>500]


# In[ ]:


df_Kl_TN


# In[ ]:


df_Kl_TN = df_Kl_TN[df_Kl_TN['Total']>20000]
df_Kl_TN.head()


# In[ ]:


plt.figure(figsize=(20,10))
g=sns.barplot('Type',y='Total',data=df_Kl_TN)
g.set_xticklabels(g.get_xticklabels(), rotation=90,horizontalalignment='right')
plt.show()


# In[ ]:


type(g)


# In[ ]:


df.head()


# ### List of type and the total no of suicides

# In[ ]:


df_Type_suicide = df.groupby(['Year','Type_code','Type','Gender'])['Total'].sum()
df_Type_suicide = pd.DataFrame(df_Type_suicide)
df_Type_suicide.reset_index(inplace=True)
df_Type_suicide.head()


# In[ ]:


fig = px.bar(df_Type_suicide, x='Type', y='Total',labels={'Total':'suicides'}, height=700,width = 1400)
fig.update_layout(title_text = 'List of type and the total no of suicides')
fig.show()


# In[ ]:


df.head()


# In[ ]:


df_st_yr_tc = df.groupby(['State','Year','Type_code','Gender'])['Total'].sum()
df_st_yr_tc=pd.DataFrame(df_st_yr_tc)
df_st_yr_tc.reset_index(inplace=True)
df_st_yr_tc.head()


# In[ ]:


df_sytc_TN_AP_KL = df_st_yr_tc[df_st_yr_tc.State.isin(['Tamil Nadu','Andhra Pradesh','Kerala'])]
df_sytc_TN_AP_KL.head()


# In[ ]:


df_sytc_TN_AP_KL.State.unique()


# In[ ]:


df_st_yr_tc['Gender'].unique()


# ### Total no of suicide in a specific type code and year

# In[ ]:


fig=px.bar(df_st_yr_tc,x='Year',y='Total',facet_col='Type_code',height=400,width=1200,color='Gender',barmode = 'group')
fig.update_layout(title_text = 'Total no of suicide in a specific type code and year')
fig.show()


# In[ ]:


df_state_year['State'].unique()


# In[ ]:


df_state_year = df_state_year[df_state_year.State != 'Total (All India)']
df_state_year = df_state_year[df_state_year.State != 'Total (States)']


# In[ ]:


df_state_year['State'].unique()


# In[ ]:


df_state_year.sort_values(by='Total',inplace=True,ascending=True)


# ### Total no of suicides in different States by year and gender

# In[ ]:


fig=px.bar(df_state_year,x='State',y='Total',facet_col='Year',height=700,width=13000,color='Gender',barmode='group')
fig.update_layout(title_text = 'Total no of suicides in different States by year and gender')
fig.show()


# In[ ]:


df.head()


# In[ ]:



df.sort_values(by='Total',ascending=False)


# In[ ]:


df.isna().sum()


# In[ ]:


df[(df['Year']==2012)&(df['State']=='Tamil Nadu')&(df['Type']=='Love Affairs')].sum()


# In[ ]:


df.pivot_table(index='State',columns='Type',values='Total',aggfunc='sum',margins=True).head()


# # Conclution

# ### List of states and their total no of suicides

# In[ ]:


fig = px.bar(df_state,x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'List of states and their total no of suicides ')
fig.show()


# ### List of Top 7 States and their total no of suicides
# 

# In[ ]:


df_state.sort_values(by='Total',inplace=True,ascending=False)
df_state.head()


# In[ ]:


fig = px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'List of top 7 states and their total no of suicides ')

fig.show()


# #### Maharashtra, West Bengal, Tamil Nadu, Andhra Pradesh, Karnataka, Kerala and Madhyapradesh are the seven states with the higest no of suicides from 2001 - 2012

# ### List of states with lowest no of suicides

# In[ ]:


df_state.sort_values(by='Total',inplace=True,ascending=True)
df_state.head()


# In[ ]:


fig =px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')
fig.show()


# #### Lakshadweep, Daman & Diu, Nagaland, Manipur, D & N Haveli, Mizoram, Chandigar are the seven states with the lowest no of suicides from 2001 - 2012

# ## Total no of suicides by year from 2001 - 2012

# In[ ]:


fig =px.bar(df_year_total,x='Year',y='Total',labels={'Total':'No of suicides: '}, height=400,color='Total')
fig.update_layout(title_text = 'Total no of suicides by year from 2001 - 2012')

fig.show()


# #### Higest no of suicides occured in the year 2011 which is appoximately (1219499)

# ### Suecides in states by genderand years

# In[ ]:


fig=px.bar(df_state_year,x='State',y='Total',facet_col='Year',height=700,width=13000,color='Gender',barmode='group')
fig.update_layout(title_text = 'Suecides in states by genderand years')
fig.show()


# #### We can clearly see that " Male (Boy's) " are the most who suecides in every state

# In[ ]:


df.head()


# In[ ]:





# ### Total no of suicide by state, year, type_code, gender

# In[ ]:


df_state_year_typec_gender= df.groupby(['State','Year','Type_code','Gender'])['Total'].sum()
df_state_year_typec_gender = pd.DataFrame(df_state_year_typec_gender)
df_state_year_typec_gender.reset_index(inplace = True)
df_state_year_typec_gender


# In[ ]:


df_state_year_typec_gender['State'].unique()


# In[ ]:


df_state_year_typec_gender = df_state_year_typec_gender[df_state_year_typec_gender.State != 'Total (All India)']
df_state_year_typec_gender = df_state_year_typec_gender[df_state_year_typec_gender.State != 'Total (States)']


# In[ ]:


df_state_year_typec_gender['State'].unique()


# In[ ]:


fig = px.bar(df_state_year_typec_gender,x='Year',y='Total',facet_col='Type_code',color='Gender',barmode='group',
             height=400,width=1600,hover_name='State')
fig.show()


# # Final Conclution

# <li>We can see that top 3 States with most no of suicides are Maharashtra, West Bengal and Tamil Nadu.</li>
# <li>And the most no of suicides occured in the year's 2010, 2011, 2012.</li>
# <li> Overall, Men were the most who committed suicide than Women.</li>
# <li> Social and Education status are the top most reason for committing suicide.</li>
# 

# ### Yet to be analysed.

# In[ ]:




