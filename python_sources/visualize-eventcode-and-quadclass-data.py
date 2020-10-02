#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# eventcode1 file contains results of a Hive query that counted the number of eventcodes by country (for selected 6 countries) 

# In[ ]:


eventcode_file="../input/eventcode1.csv"
df_code=pd.read_csv(eventcode_file)


# In[ ]:


df_code.head()


# In[ ]:


# number of rows
len (df_code)


# In[ ]:


df_code.dtypes


# In[ ]:


#rename the count column
df_code.rename(columns={'_c2':'count'},inplace=True)


# In[ ]:


df_code.tail(5)


# In[ ]:


#total number of events by country
totevents=df_code.groupby('actor1countrycode')['count'].sum()
print (totevents)


# In[ ]:


totevents.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of events')
plt.title('Number of events by Country')


# In[ ]:


#divide up large dataframe into separate ones for each country
NZL=df_code[df_code['actor1countrycode']=='NZL']
AUS=df_code[df_code['actor1countrycode']=='AUS']
BEL=df_code[df_code['actor1countrycode']=='BEL']
JPN=df_code[df_code['actor1countrycode']=='JPN']
IND=df_code[df_code['actor1countrycode']=='IND']
FRA=df_code[df_code['actor1countrycode']=='FRA']


# In[ ]:


#sort from highest to lowest count of each eventcode and create a dataframe for each country's top 5 eventcodes
NZL1=NZL.sort_values('count',ascending=False)
NZLtop_e = NZL1.head()
AUS1=AUS.sort_values('count',ascending=False)
AUStop_e = AUS1.head()
BEL1=BEL.sort_values('count',ascending=False)
BELtop_e= BEL1.head()
JPN1=JPN.sort_values('count',ascending=False)
JPNtop_e = JPN1.head()
IND1=IND.sort_values('count',ascending=False)
INDtop_e = IND1.head()
FRA1=FRA.sort_values('count',ascending=False)
FRAtop_e = FRA1.head()


# Plot of each country's top 5 event codes by count

# In[ ]:


NZLtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - New Zealand')

AUStop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Australia')

BELtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Belgium')

JPNtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - Japan')

INDtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - India')

FRAtop_e.plot('eventcode','count',kind='bar')
plt.xlabel('Event Code')
plt.ylabel('Count')
plt.title('Count of events - France')


# Load up the quadclass count file which contains results of a Hive query that counted the number of events in each quad class by country 

# In[ ]:


quadclass_file="../input/quadclass_count.csv"
df_quad=pd.read_csv(quadclass_file)


# In[ ]:


df_quad.head()


# rename the count column

# In[ ]:


df_quad.rename(columns={'_c2':'count'},inplace=True)


# break up the large dataframe into separate ones by country

# In[ ]:


aus1=df_quad[df_quad['actor1countrycode']=='AUS']
bel1=df_quad[df_quad['actor1countrycode']=='BEL']
fra1=df_quad[df_quad['actor1countrycode']=='FRA']
ind1=df_quad[df_quad['actor1countrycode']=='IND']
jpn1=df_quad[df_quad['actor1countrycode']=='JPN']
nzl1=df_quad[df_quad['actor1countrycode']=='NZL']


# visualize the quad class data by country

# In[ ]:


aus1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Australia')


# In[ ]:


bel1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Belgium')


# In[ ]:


fra1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - France')


# In[ ]:


ind1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - India')


# In[ ]:


jpn1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - Japan')


# In[ ]:


nzl1.plot('quadclass','count',kind='bar')
plt.xlabel('QuadClass')
plt.ylabel('Number of Events')
plt.title('Number of events by QuadClass - New Zealand')

