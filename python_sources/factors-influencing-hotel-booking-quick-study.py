#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Demand
# ![](https://www.indiatourpackage.nl/images/Hotel%20Booking.jpg)
# 

# ## Methodology 
# - Data Decription
# - Feature Engineering
#      - Find and Impute Missing Values and Outliers
#      - Feature transformation
# - Data Visualization
#     - using Pie Chart
#     - using Bar Chart

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.head()


# # 1.0 Data 
# ### 1.1. Basic Data Description

# In[ ]:


print('Shape of Dataset=',df.shape)
df.describe(include='all')


# - The dataset have **<font color='green'>32 variables (Continuous and Categorical)</font>** with one identified dependent variable (categorical), which is 'is_cancelled.
# - The dataset **<font color='green'>contains Missing values & Outliers</font>**.
# - Few columns such as 'reservation_status_date' needs **<font color='green'>Data Transformation</font>**.

# ### 1.2. Change Column Names
# Columns name needs modification for the sake of convinience and representability. <br>
# Note: *However, it is not mandatory and will vary from person to person*

# In[ ]:


df.columns = ['Hotel', 'Canceled', 'LeadTime', 'ArrivingYear', 'ArrivingMonth', 'ArrivingWeek','ArrivingDate', 'WeekendStay',
              'WeekStay', 'Adults', 'Children', 'Babies', 'Meal','Country', 'Segment', 'DistChannel','RepeatGuest', 'PrevCancel',
              'PrevBook', 'BookRoomType','AssignRoomType', 'ChangeBooking', 'DepositType', 'Agent','Company', 'WaitingDays', 
              'CustomerType', 'Adress','ParkSpace', 'SpecialRequest','Reservation', 'ReservationDate']


# ### 1.3. Identify Categorical and Continuous Variables

# In[ ]:


def get_cat_con_var(df):
    unique_list = pd.DataFrame([[i,len(df[i].unique())] for i in df.columns])
    unique_list.columns = ['name','uniques']

    universe = set(df.columns)
    cat_var = set(unique_list.name[(unique_list.uniques<=12)      | 
                                   (unique_list.name=='Country')  | 
                                   (unique_list.name=='Agent')                                     
                                  ])
    con_var = universe - cat_var
    
    return cat_var, con_var 


cat_var, con_var = get_cat_con_var(df)

print("Continuous Variables (",len(con_var),")\n",con_var,'\n\n'
      "Categorical Variables(",len(cat_var),")\n",cat_var)


# - 13 continuous and 19 categorical variables are identified, based on number of unique values i.e. **<font color='green'>factor levels</font>**.
# - The number of countries is greater than 12 **<font color='green'>(factor level threshold)</font>**, but should be assigned to categorical variables based on knowledge. 

# # 2.0 Feature Engineering

# ## 2.1 Missing Values
# ### 2.1.1 Find Missing Values
# 

# In[ ]:


missing_col_list = df.columns[df.isna().sum()>0]
print('Missing data columns =',missing_col_list)
t = pd.DataFrame([[i,df[i].unique(),df[i].isna().sum()] for i in missing_col_list])
t.columns = ['name','unique','missing']
t   


# - The number of missing values found is 4 & 488 for features 'Children' & 'Country' respectively, both of which are categorical.
# - Features Agent and Company acts as primary keys, however, contains missing values.

# ### 2.1.2 Impute Missing Values

# In[ ]:


df.loc[df.Children.isna(),'Children'] = 0

df.loc[df.Country.isna(),'Country'] = 'NAA'

# agent and country are ID, cannot be imputed. Impute available/unavailable.
df.loc[df.Agent>0,'Agent']      = 1
df.loc[df.Agent.isna(),'Agent'] = 0

df.loc[df.Company>0,'Company']      = 1
df.loc[df.Company.isna(),'Company'] = 0

print('Remaining Missing Values = ',df.isna().sum().sum())


# - Feature 'Children' is imputed by **<font color='green'>Statistical Transformation</font>** using mode = 0.
# - Country is categorical variable, but accurate imputation is not possible. Thus, missing values are defined as **<font color='green'>New Level</font>** which is 'NAA'. 
# - Agent and Company acts as primary keys, thus imputation is not possible. Therefore, available values are marked as 1 where as unavailable values are maked with 0. Thus, **<font color='green'>Feature Transformation</font>** is performed. 

# ## 2.2 Find Outliers
# ### 2.2.1. Find Outliers (for Categorical Variables)

# In[ ]:


def print_unique_values(cols):
    for i in cols:
        print(i,df[i].unique())
        
print_unique_values(cat_var)


# - 3 Categorical Features seems to have outlier values
#     - Babies   : 9, 10
#     - Parkspace: 8
#     - Children : 10

# ### 2.2.2. Impute Outliers (for Categorical Variables)

# In[ ]:


df.loc[df.Babies    > 8,'Babies']    = 0
df.loc[df.ParkSpace > 5,'ParkSpace'] = 0
df.loc[df.Children  > 8,'Children']  = 0

df[con_var].describe()


# No outliers seem to exist, by examining all Categorical Variables. 

# ### 2.2.3. Find Outliers (for Continuous Variables)

# In[ ]:


df[con_var].describe()


# - List of Continuous Variables with Outliers identified through **<font color='green'>Summary Statistics</font>** are:
#     - LeadTime
#     - WaitingDays
#     - WeekendStay
#     - Adults
#     - PrevBook
#     - PrevCancel
#     - WeekStay
# - List of Continuous Variables which should be Categorical
#     - PrevBook
#     - PrevCancel

# ### 2.2.4. Impute Outliers (for Continuous Variable)

# In[ ]:


df.loc[df.LeadTime      > 500,'LeadTime'     ]=500
df.loc[df.WaitingDays   >   0,'WaitingDays'  ]=  1
df.loc[df.WeekendStay   >=  5,'WeekendStay'  ]=  5
df.loc[df.Adults        >   4,'Adults'       ]=  4
df.loc[df.PrevBook      >   0,'PrevBook'     ]=  1
df.loc[df.PrevCancel    >   0,'PrevCancel'   ]=  1
df.loc[df.WeekStay      >  10,'WeekStay'     ]= 10
df.loc[df.ChangeBooking >   5,'ChangeBooking']=  5

cat_var = set(list(cat_var) + ['PrevBook','PrevCancel'])
con_var = set(df.columns) - cat_var

df[con_var].describe()


# ## 3.0 Visualization

# ### 3.1. Understand Correlation
# - The dataset containts 32 variables, where one or more variables may be a function of another variable(s)
# - The 2D visualization helps to identify variable(s) which have high influence on another variable(s).

# In[ ]:


cor_mat = df.corr()
fig, ax = plt.subplots(figsize=(16,6))
sns.heatmap(cor_mat,ax=ax,cmap="YlGnBu",linewidths=0.1)


# - **<font color='green'>Target Variable</font>** (Cancelled) is influence by LeadTime, PrevCancel, Agent, WaitingDays, Address. It implies that, hotel booking is most likely to be cancelled,
#     - if *lead-time* (i.e. difference between booking and arrival) increases, which may be due to availablity of alternatives
#     - if the customer have *previously cancelled booking* a few times, which implies that customer may cancel again
#     - if the *booking confirmation time is longer*, i.e. when difference between time to book and get confirmation is higher, which increases the uncertainty and thus probability of cancellation
#     - based on *travel agency*, as some agencies tend to perform bookings based on seasonality which are withdrawn later
# 
# 
# - **<font color='green'>Other Highly Related Features:</font>**
#     - *PrevBook, RepeatGuest, Company*: Customers who have booked previously is like to book again, preferably through same company / entity that the customers have already associated
#     - *LeadTime and PrevCancel*: Higher lead-time influences cancellation, thus an increase in Previous Cancellations
#     - *ArrivingYear and Address*: Some destinations can have higher influx of people due to seasonality or certain event 
#     - *WeekendStay and WeekStay*: It is expected that people who stays on weekdays is likely to overstay on weekends
#     - *Adress, Adults and Chilren*: Certain location are preferred by indivual and family

# ### 3.2. Pre-processing for Visualization
# #### 3.2.1. Select best order of visualization
# - Categorical Variables with **atmost 5 levels** can be presented in **Pie-Chart** clearly
# - Categorical variables with **more than 5 levels** can be presented in **Bar-Chart** properly

# In[ ]:


pie_list = list()
bar_list = list()
line_list = list()

for i in cat_var:
    if len(df[i].unique())<=5:
        pie_list.append(i)
    elif len(df[i].unique())<=12:
        bar_list.append(i)
    else:
        line_list.append(i)
        
print('Features with 5 levels   \n',pie_list,'\n\n',
      'Features with 5-10 levels\n',bar_list,'\n\n',
      'Features with >10 levels \n',line_list)


# #### 3.2.1. User-defined function to find count number of records in each category

# In[ ]:


def get_pie_label_values(df,col):
    temp = pd.DataFrame([[i,df[df[col]==i].shape[0]] for i in df[col].unique()])    
    return temp[0],temp[1] 


# #### 3.2.2. LeadTime Grouping
# LeadTime have wide range [0-500], thus forming category using class limits will help in identifying trend and perform analysis.

# In[ ]:


def put_into_bucket(df,col,bucket):    
    diff = int(max(df[col])/bucket)
    for i in range(bucket):    
        df.loc[(df[col] > diff*(i)) & (df[col] <= diff*(i+1)),col] = i+1
    df.loc[df[col]==0,col] = 1
    return df

df = put_into_bucket(df,'LeadTime',bucket=5)


# #### 3.2.3. Reservation Date Extraction (into Date, Month, Year)
# - Extraction of date, month, year can provide additional information such as,
#     - Impact of monthly / annual expenditure
#     - Impact of seasonality   

# In[ ]:


# Extraction
new = df['ReservationDate'].str.split('-', n = 2, expand = True) 
df['YearReserve' ]= new[0] 
df['MonthReserve']= new[1] 
df['DateReserve' ]= new[2] 
df.drop(columns=['ReservationDate'],inplace=True)


# ### 3.3. Visualization of Categorical Features
# #### 3.3.1. Part - I (Using Pie Chart)

# In[ ]:


n_row = 3
n_col = 5
fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}],
                                           [{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}],
                                           [{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}]],                                           
                   subplot_titles=pie_list,
                   horizontal_spacing = 0.03, vertical_spacing = 0.08)

row = 1
col = 1
x_adr = 0.082
y_adr = 0.85
x_diff = 0.21 # increasing order
y_diff = 0.845 - 0.485 # decreasing order
ls = list()
for i in pie_list:
    labels, values = get_pie_label_values(df,i)    
    fig.add_trace(go.Pie(labels=labels, values=values, name=i),row,col)      # Design Pie Charts          
    ls.append(dict(text=str('<b>'+i+'</b>'), x=x_adr, y=y_adr, font_size=10, showarrow=False)) # Get position of text in Pie-Holes    
    col+=1                                                                   # Get Grid Details
    x_adr+=x_diff
    if col > n_col:
        col =1
        row+=1
        x_adr = 0.082
        y_adr-= y_diff
    
fig.update_traces(hole=0.65, hoverinfo="label+percent+name")    
fig.update_layout(title_text="Visualizing Categorical Variables using <b>Pie charts</b> : (<i>With or less than 5 levels</i>)",
                  annotations=ls,
                  width=1200,height=650,
                  showlegend=False)
fig.show()


# **<font color='green'>The Hotels:</font>**
# - In the dataset, **most hotels are City Hotels** represented by 66.4% and remaining hotels are resorts by 33.6%
# 
# - **82% bookings are from Travel/Tour Operator(s)**, with increasing trendvisitors at a significant rate i.e. 18.4% to 47.5% from 2015 to 2016. However, the percentage dropped to 34.1% in 2017. 
# 
# 
# 
# **<font color='green'>The Bookings:</font>**
# - **Almost 2/3 of hotel booking gets cancelled**, therefore the probability of hotel booking cancellation is much higher. 
# 
# - **87.6% cancellation is observed with 'no deposit'** followed by 12.2% non-refundable deposit. This is beacuse, customers is not willing to deposit any amount which is non-refundable, resulting in high ratio of 'no deposit' booking. 
# 
# 
# 
# **<font color='green'>The Customers:</font>**
# - It is observed that 94.6% of customers who have booked hotel have not cancelled any bookings earlier. This looks positive on hotel business. However, as **97% of the customers are new**, it is cannot be denied that **hotels do not get repeat visitors**, observed as 3.19%. 
# 
# - **Most customers are Transient**, who visit without kids which is accounted by 99.2% without babies, 92.8% without children. Although, feature babies and children seems redundant in presence of one another, it however is not. 
# 
# - Most customers have certain request, such as **bed in breakfast as highest preference**. However, as most hotels are City Hotels, **most customers do not give priority to parking space**. It is because, cities usually have well transportaion connectivities.

# #### 3.3.2. Part - II (Using Bar Chart)

# In[ ]:


n_row = 1
n_col = 5
fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'}, {'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}]],                                                   
                   subplot_titles=bar_list,
                   horizontal_spacing = 0.03, vertical_spacing = 0.13)

row = 1
col = 1
for i in bar_list:
    labels, values = get_pie_label_values(df,i)
    #print(labels, values)
    fig.add_trace(go.Bar(y=values),row=row, col=col)
    
    col+=1
    if col > n_col:
        col =1
        row+=1    
    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])
    
fig.update_layout(title_text="Visualizing Categorical Variables using <b>Bar charts</b>: (<i>Within 5 - 10 levels</i>)",
                  width=1200,height=500,showlegend=False)
fig.show()


# **<font color='green'>Room Preference:</font>**
# - Room code 1, which have the highest booking requests are assigned, followed by Room code 2, 3, and 5. 
# 
# 
# **<font color='green'>Effect of Seasonality</font>**
# - **Hotel Bookings drops significantly during summer** i.e. for months May to August. The highest hotel bookings are observed during winter, and especially beginning of new year.
# 
# 
# **<font color='green'>Booking Channels:</font>**
# - Mostly **Online TA perform Hotel Booking**, followed by Offline TA/TO and certain Groups, as shown in Segment.
# 
# 
# **<font color='green'>Requests:</font>**
# - **Most customers do not place any special request**. However, few customers do regarding bed types, meal preference, etc as observed earlier.

# #### 3.3.3. Part - III (Using Bar Chart)

# In[ ]:


ls=list()
for i in line_list:
    for j in df[i].unique():
        ls.append([j,df[df[i]==j].shape[0],i])
ls = pd.DataFrame(ls)
ls.columns = ['column','counts','feature']

ls.sort_values(by='counts',ascending=False,inplace=True)
fig = px.bar(ls[1:50],x='column',y='counts',color='counts',facet_col='feature')
fig.update_layout(title_text="Visualizing Categorical Variables using <b>Bar charts</b> : (<i>More than 10 levels</i>, Top 50 Countries)",
                  width=1150,height=400,showlegend=False)
fig.show()


# **<font color='green'>Geographically Attractive:</font>**
# - Top 50 countries are shown in the chart. It can be observed that **United Kingdom receives maximum number of visitors**, followed by France, Spain, Germany, and others. 
# 
# - **Most countries are from Europe** with few from American Continent.

# ### 3.4. Visualization of Continuous Features
# #### 3.4.1. Part - I (Using Bar Chart)

# In[ ]:


list_01 = ['Adults', 'WaitingDays', 'LeadTime', 'ChangeBooking', 
            'WeekStay', 'WeekendStay','YearReserve','MonthReserve']

n_row = 2
n_col = 4
fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}],
                                                   [{'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}]],               
                   subplot_titles=list_01,
                   horizontal_spacing = 0.03, vertical_spacing = 0.23)

row = 1
col = 1
for i in list_01:
    labels, values = get_pie_label_values(df,i)
    #print(labels, values)
    fig.add_trace(go.Bar(x=labels,y=values),row=row, col=col)
    
    col+=1
    if col > n_col:
        col =1
        row+=1    
    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])
    
fig.update_layout(title_text="Visualizing Continuous Variables using <b>Bar charts</b>",width=1200,height=500,showlegend=False)
fig.show()


# **<font color='green'>Age on record:</font>**
# - Maximum number of adult customer seems to be 0. It is because, most booking are done by online and offline TA, where people book hotels to visit with thier family members which can be children
# 
# 
# **<font color='green'>Benefit of Lead Time:</font>**
# - Hotel booking gets confirmed on the same day as the booking. Little comfirmation delay of 2-4 days, which is the lead time is used by the customers to alter thier booking details. 
# 
# **<font color='green'>Weekdays and Weekends:</font>**
# - Number of weeday stay is much higher than weekend stays. It is simply use, weekdays:weekend  is 5:2. Therefore, most bookings tend to fall on weekdays. 
# 
# **<font color='green'>Effect of seasonality:</font>**
# - Number of booking increases significantly in 2016-17 when compared to year 2015, as observed earlier. Similar trend is observed for month feature where most booking are perform just before winter. 

# #### 3.4.2. Part - II (Using Bar Chart)

# In[ ]:


list_02 = ['DateReserve','ArrivingDate','ArrivingWeek']

n_row = 3
n_col = 1
fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'}],
                                                   [{'type':'bar'}],
                                                   [{'type':'bar'}]],                                                                                    
                   subplot_titles=list_02,
                   horizontal_spacing = 0.03, vertical_spacing = 0.08)

row = 1
col = 1
for i in list_02:
    labels, values = get_pie_label_values(df,i)
    print(i,'=',min(values))
    values = values - min(values)        
    fig.add_trace(go.Bar(x=labels,y=values),row=row, col=col)
    
    col+=1
    if col > n_col:
        col =1
        row+=1    
    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])
    
fig.update_layout(title_text="Visualizing Continuous Variables using <b>Bar charts</b>",width=1200,height=700,showlegend=False)
fig.show()


# - Most reservation happens on the second half of the month. It may acts as an association with successive months operations.
# - Most customers visit on the first half of the month, in correspondence to the booking performed the previous month. 
# - Most arrival occurs first half of the year when compared to second half of the year.
# 

# ### This Notebook is still in process. Please provide feedback and suggestions to improve the Notebook.
