#!/usr/bin/env python
# coding: utf-8

# # Suicide Analysis in India
# 
# In this notebook we will try to understand what might be the different reasons due to which people committed suicide in India (using the dataset "Suicides in India"). Almost 11,89,068 people committed suicide in 2012 alone, it is quite important to understand why they commit suicide and try to mitigate.
# 

# In[ ]:


# import lib
import numpy as np #for math operations
import pandas as pd#for data manipulation
import plotly.express as px#for better visualization
import plotly.io as pio

# read dataset
data = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
data.tail(10)


# # Dataset Information

# In[ ]:


data.info()


# # Check Missing & Null Values
# 
# 

# In[ ]:


data.isna().sum()


# # People committed suicide from 2001-2012

# In[ ]:


print("Total cases from 2001-12: \n",data.groupby("Year")["Total"].sum())
data.groupby("Year")["Total"].sum().plot(kind="line",marker="o",title="People Commited Suicide From 2001-2012")


# # States Present Inside Dataset
# 
# This step is for merging states with same name and remove redundency.

# In[ ]:


data["State"].value_counts()


# Remove rows with value as Total (States), Total (All India) or Total (Uts)

# In[ ]:


data = data[(data["State"]!="Total (States)")&(data["State"]!="Total (Uts)")&(data["State"]!="Total (All India)") ]


# # Which Gender with Highest number of suicide?
# 
#  Males are commiting more sucides in comaprision to females 

# In[ ]:


filter_gender = pd.DataFrame(data.groupby("Gender")["Total"].sum()).reset_index()
px.bar(filter_gender,x="Gender", y="Total",color="Gender")


# # States with Higher Suicide cases
# 
# 
# 1. Maharashtra<br>
# 2. West Bengal<br>
# 3. Tamil Nadu<br>
# 4. Andhra Pradesh<br>

# In[ ]:


pio.templates.default = "plotly_dark"
filter_state = pd.DataFrame(data.groupby(["State"])["Total"].sum()).reset_index()
px.bar(filter_state,x = 'State', y = 'Total',color="State")


# # Number of cases changing over time 
# Changing Rate of sucides over time 

# In[ ]:


grouped_year = data.groupby(["Year","Gender"])["Total"].sum()
grouped_year = pd.DataFrame(grouped_year).reset_index()
# grouped_year
px.line(grouped_year,x="Year", y="Total", color="Gender")


# # Number of cases based on the reasons they committed suicide
# 
# 

# In[ ]:


filter_type_code = pd.DataFrame(data.groupby(["Type_code","Year"])["Total"].sum()).reset_index()
filter_type_code
px.bar(filter_type_code,x="Type_code", y="Total",color="Year")


# # Which social issues causes more suicides?
# 
# It is clear that **married people** are more Suicides.<br>
# 
# Which makes sense because marriage issues may cause conflict between the couple and as a result they might be prone to commit suicide.

# In[ ]:


filter_social_status = pd.DataFrame(data[data["Type_code"]=="Social_Status"].groupby(["Type","Gender"])["Total"].sum()).reset_index()
px.bar(filter_social_status,x="Type", y="Total",color="Gender")


# # Education status of people who committed suicides
# people with low education are  commiting more suicide.<br>
# 
# People with Diploma and Graduate tend to commit least no. of suicide

# In[ ]:


filter_social_status = pd.DataFrame(data[data["Type_code"]=="Education_Status"].groupby(["Type","Gender"])["Total"].sum()).reset_index()
fig = px.bar(filter_social_status,x="Type", y="Total",color="Gender")
fig.show(rotation=90)


# # Profession of the people who committed suicides
# 
# **Farmers** and **housewives** have commited more suicide compared to others.
# 
# This makes sense because most of the Indian farmers have debt and their life depends on the yield of their crops, if the yield is not good then they will not be able to clear their debt and in the worst case they might commit suicide.
# 
# > Global warming, monsoon delay, drought etc can lead to bad yield.
# 
# Housewives might have issues in their marriage which this might be a reason for such a high number of cases.
# > Domestic violence, dowry, gender discrimination, etc might be some of the reasons for housewives to commit suicide.

# In[ ]:


filter_social_status = pd.DataFrame(data[data["Type_code"]=="Professional_Profile"].groupby(["Type","Gender"])["Total"].sum()).reset_index()
fig2 = px.bar(filter_social_status,x="Type", y="Total",color="Gender")
fig2.show(rotation=90)


# # Which age group people have commited most Suicides?
# 
# From the below visualization it is clear that youngsters (15-29 age) and middle age (30-44) tend to commit the maximum number of suicides.
# 
# It can be due to several reasons like:
# * unemployment
# * academic stress
# * bad friend circle
# * farmers (since they have to be young and strong enough to do farming)
# * addictions

# In[ ]:


# age group 0-100+ encapsulates all the remaining age groups, hence it would make sense to drop it
import matplotlib.pyplot as plt #for visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_palette("BrBG")
filter_age = data[data["Age_group"]!="0-100+"]
sns.catplot(x="Age_group", y="Total", kind="bar", data=filter_age,height=8.27, aspect=11.7/8.27);


# # Conclusion
# 
# * Males tend to commit more suicides compared to Females in India
# * Highest no. of suicide cases occur in Maharashtra, West Bengal, and Tamil Nadu, Andhra Pradesh.
# * Male might commit more suicide compared to females in the future if this trend continues.
# * People who commit suicide are mostly:
#     * Married
#     * Farmers and housewives
#     * Youngsters (15-29 age) and middle age (30-44)
