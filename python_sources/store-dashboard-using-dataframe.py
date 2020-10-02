#!/usr/bin/env python
# coding: utf-8

# #### This Notebook is helpful to create simple dashboard using Pandas DataFrame & Functions, Without using  plot. Function Groupby,  is used extensively. Data source is created manually using numpy. This notebook generates sales summary and stastics for different attributes like Store level, Gender, HML etc.
# ![image.png](attachment:image.png)

# 1. Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns # Visualization
import warnings
warnings.filterwarnings('ignore')
#Print Multiple line output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# 2. Create Data Set

# In[ ]:


data=pd.DataFrame()

data['CUST_ID']=np.arange(1,10001)
data['STORE']=np.random.randint(1,15,10000)
data['SALES']=np.random.randint(10,100,10000)
data['UNITS']=np.random.randint(5,15,10000)
data['GP']=data['SALES']*(30/100) # Gross Profit
data['TRANSACTIONS']=np.random.randint(1,5,10000)
data['HML']=np.random.randint(0,3,10000)
data['AGE']=np.random.randint(10,100,10000)
data['GENDER']=np.random.randint(0,2,10000)
data['BV']=data['SALES']/data['TRANSACTIONS'] # Basket Value
data['BS']=data['UNITS']/data['TRANSACTIONS'] # Basket Size
data['CUST_TYPE']=np.random.randint(0,2,10000) # Existing Or New


# In[ ]:


#view data set
data.head()


# In[ ]:


#Sort by store id and reset the index
data.sort_values('STORE',ascending=True, inplace=True)
data.reset_index(drop=True,inplace=True)
data.head()


# <p>**Preprocessing which involves converting values in dashboard readable format**
#    <br>
#    <br>#Cust_Id - 1 : Cust1, 2 : Cust2..
#    <br>#HML - 2 : high, 1 : Medium , 0 : Low
#    <br>#Gender - 0 : M, 1 : F 
#    <br>#Cust_Type - 1 : Existing, 0 : New</p>

# In[ ]:


#chnage name of customer id, gender, store no, hml and cutomer type
x=[]
for i in data['CUST_ID']:
    x.append("Cust"+str(i))
data['CUST_ID']=x

x=[]
for i in data['STORE']:
    x.append("Str"+str(i))
data['STORE']=x

z=[]
for i in data['GENDER']:
    if i==0:
        z.append('M')
    else:
        z.append('F')
data['GENDER']=z

z=[]
for i in data['HML']:
    if i==0:
        z.append('Low')
    elif i==1:
        z.append('Medium')
    else:
        z.append('High')
data['HML']=z

z=[]
for i in data['CUST_TYPE']:
    if i==0:
        z.append('New')
    else:
        z.append('Existing')
data['CUST_TYPE']=z


# In[ ]:


data.head() # all set for dashboard creation


# In[ ]:


#Store level overall sales summary 
Store_Summary=pd.DataFrame(data['STORE'].unique(),columns=['STORE'])
for i in data.describe().columns:
    if i != 'AGE':
        df = pd.DataFrame(data.groupby('STORE').sum()[i], columns=[i])
        Store_Summary= pd.merge(Store_Summary,df,on='STORE',how='inner')
Store_Summary.set_index('STORE',inplace=True)
Store_Summary=Store_Summary[['SALES','UNITS','TRANSACTIONS','BV','BS','GP']]


# In[ ]:


def highlight_header(x):
    y= ['background-color: LightSkyBlue' if v>=40000 else '' for v in list(x)]
    return y


# In[ ]:


Store_Summary.style.apply(highlight_header)


# *  There are  stores where sales are high

# In[ ]:


# function to get mean, median, mode, IQR(1st quantile 3rd quantile) and range on Sales attribute for a given value
def stats(df,col):
    df = pd.DataFrame(data.groupby(col).describe()['SALES']).rename(columns={'25%':'FIRST_QUANTILE',
                                          '50%':'MEDIAN',
                                          '75%':'THIRD_QUANTILE',
                                          'std': 'STD',
                                           'mean':'MEAN'}) 
    df['MODE']=pd.DataFrame(data.groupby(col).SALES.apply(lambda x: x.mode()[0]))
    df['IQR']=df.apply(lambda x: x['THIRD_QUANTILE'] - x['FIRST_QUANTILE'], axis=1)
    df['RANGE']=df.apply(lambda x: str(x['min'])+ ' - '+ str(x['max']), axis=1)
    df=df[['MEAN','MEDIAN','MODE','STD','IQR','RANGE']]
    return df


# In[ ]:


#Store level Sales statistics  
Store_Stats=pd.DataFrame()
Store_Stats = stats(Store_Stats,'STORE')


# In[ ]:


Store_Stats


# In[ ]:


#Customer Type wise sales Summary
Type_Sales_Summary= pd.DataFrame(data.groupby('CUST_TYPE')['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())


# In[ ]:


Type_Sales_Summary


# In[ ]:


#Customer Type wise sales stats Summary
Cust_Type_Stats=pd.DataFrame()
Cust_Type_Stats = stats(Store_Stats,'CUST_TYPE')


# In[ ]:


Cust_Type_Stats


# In[ ]:


#Age group wise sales Summary
#To split data by age group , we can use pd.cut() functions
Age=pd.cut(data['AGE'],[0,18,24,35,50,101])
Age_Sales_Summary= pd.DataFrame(data.groupby(Age)['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())


# In[ ]:


Age_Sales_Summary.reset_index(inplace=True)


# In[ ]:


Age_Sales_Summary


# In[ ]:


# Function to format Age data
def format_age(df,col):
    x=[]
    for i in df[col]:
        if str(i)[1]=='0':
            x.append(str('0-18'))
        if str(i)[1]=='1':
            x.append(str('18-24'))
        if str(i)[1]=='2':
            x.append(str('24-35'))
        if str(i)[1]=='3':
            x.append(str('35-50'))
        if str(i)[1]=='5':
            x.append(str('50+'))
    df[col]=x
    return df


# In[ ]:


Age_Sales_Summary = format_age(Age_Sales_Summary,'AGE')


# In[ ]:


Age_Sales_Summary.set_index('AGE',inplace=True)


# In[ ]:


Age_Sales_Summary


# In[ ]:


# Age wise sales statistics
age_sales_stats =pd.DataFrame()
age_sales_stats = stats(age_sales_stats,Age) # Age is groups of Age


# In[ ]:


age_sales_stats.reset_index(inplace=True)


# In[ ]:


age_sales_stats = format_age(age_sales_stats,'AGE')


# In[ ]:


age_sales_stats.set_index('AGE',inplace=True)


# In[ ]:


age_sales_stats


# In[ ]:


#HML wise sales Summary
HML_Sales_Summary= pd.DataFrame(data.groupby('HML')['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())


# In[ ]:


HML_Sales_Summary


# In[ ]:


# HML wise sales statistics
hml_sales_stats =pd.DataFrame()
hml_sales_stats = stats(hml_sales_stats,'HML') 
hml_sales_stats


# In[ ]:


#GENDER wise sales Summary
gender_Sales_Summary= pd.DataFrame(data.groupby('GENDER')['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())
gender_Sales_Summary


# In[ ]:


# Gender wise sales statistics
gender_sales_stats =pd.DataFrame()
gender_sales_stats = stats(gender_sales_stats,'GENDER') 
gender_sales_stats


# In[ ]:


#Sales Summary by gender, and age group
new_data=data[data['CUST_TYPE']=='Existing']
new_age=pd.cut(new_data['AGE'],[0,18,24,35,50,101])
df1=pd.DataFrame(data.groupby(['GENDER',Age])['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())
df2=pd.DataFrame(pd.DataFrame(new_data.groupby(['GENDER',new_age])['CUST_TYPE'].count()))
df2=df2.rename(columns={'CUST_TYPE' : 'EXISTING_CUSTOMER'})
summary=pd.merge(df1,df2,on=['GENDER','AGE'],how='inner')
final_summary=(summary.unstack(level=0).T).swaplevel(0,1).sort_index(level=0)
final_summary.apply(np.floor)


# In[ ]:


#Sales Summary by Customer Type, and age group
summary1=pd.DataFrame(data.groupby(['CUST_TYPE',Age])['SALES','UNITS','TRANSACTIONS','BV','BS','GP'].sum())
final_summary1=(summary1.unstack(level=0).T).swaplevel(0,1).sort_index(level=0)
final_summary1.apply(np.floor)

