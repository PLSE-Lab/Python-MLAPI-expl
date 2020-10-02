#!/usr/bin/env python
# coding: utf-8

# # Analysis of Hate Crime in India from 2001-2012

# Maintain a Zoom Level on the WebPage to around 70 as it will be better in Understanding the plots as more amount of data could be seen in one frame. Thank You!

# **What is a hate crime?**
# 
# A hate crime is when someone commits a crime against you because of your disability, gender identity, race, sexual orientation, religion, or any other perceived difference.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading CSV files 
# 1. cbd stands for Crime by District
# 2. cbdr stands for Crime by Distrct Rt
# 3. cbs stands for Crime by State
# 4. cbsr stands for Crime by State Rt

# In[ ]:


cbd = pd.read_csv("../input/crimeanalysis/crime_by_district.csv")
cbdr = pd.read_csv("../input/crimeanalysis/crime_by_district_rt.csv")
cbs = pd.read_csv("../input/crimeanalysis/crime_by_state.csv")
cbsr = pd.read_csv("../input/crimeanalysis/crime_by_state_rt.csv")


# # DATA PREPROCESSING

# Shape gives the number of rows and number of coloumns in the dataframe.

# In[ ]:


cbd.head()


# In[ ]:


cbdr.head()


# In[ ]:


cbd.shape


# In[ ]:


cbdr.shape


# **'Total1' is used to store values which are not common in 'CBD' and 'CBDR'**

# In[ ]:


total1 = pd.concat([cbd,cbdr]).drop_duplicates(keep = False)


# **I have added a new column to the DataFrame 'Total1' so that all the crimes comes under one column. This will be useful later on.**

# In[ ]:


total1['Total Atrocities'] = total1['Murder'] +total1['Assault on women']+total1['Kidnapping and Abduction']+total1['Dacoity']+total1['Robbery']+total1['Arson']+total1['Hurt']+total1['Prevention of atrocities (POA) Act']+total1['Protection of Civil Rights (PCR) Act']+total1['Other Crimes Against SCs']
total1.head()


# In[ ]:


cbs.shape


# In[ ]:


cbsr.shape


# **Similarly, the crime for states duplicates are removed and different values are stored in 'Total' DataFrame.**

# In[ ]:


total = pd.concat([cbs,cbsr]).drop_duplicates(keep = False)


# In[ ]:


total.shape


# This shows that the different values between 'CBS' and "CBSR" was the total value of number of crimes of States and UTs in a Year. Then these values are added to the total Sum for Hate Crime in the particular years for State and UTs.

# In[ ]:


total.head(20)


# In[ ]:


total.tail(20)


# **As Hate Crime by UT and State are summed to form 'TOTAL (ALL-INDIA)' row for hate crimes, these are regarded and the rest I have dropped from the DataFrame.**

# In[ ]:


total.drop(total[total['STATE/UT'] == 'TOTAL (UTs)'].index , inplace = True) 
total.drop(total[total['STATE/UT'] == 'TOTAL (STATES)'].index , inplace = True) 


# 'DataFrame'.isnull().sum() is used to show that if there is any Null value present in the DataFrame.

# In[ ]:


cbdr.isnull().sum()


# In[ ]:


cbsr.isnull().sum()


# In[ ]:


total.isnull().sum()


# In[ ]:


cbdr['Total Atrocities'] = cbdr['Murder'] +cbdr['Assault on women']+cbdr['Kidnapping and Abduction']+cbdr['Dacoity']+cbdr['Robbery']+cbdr['Arson']+cbdr['Hurt']+cbdr['Prevention of atrocities (POA) Act']+cbdr['Protection of Civil Rights (PCR) Act']+cbdr['Other Crimes Against SCs']


# # EXPLORATORY DATA ANAYLSIS ON THE BASIS OF DISTRICTS

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')


# The groupby above shows the highest number of murders done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Majority of the 10 highest spots for murder goes to Uttar Pradesh State for a good range of years.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)
s.head(10).style.background_gradient(cmap='Purples')


# The groupby above shows the highest number of Assault on women done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Here Assault is highest in Rajasthan with 4 spots taken by Rajasthan in the top 10.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')


# The groupby above shows the highest number of Kidnapping and Abduction done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Uttar Pradesh takes all the 10 spots for highest number of Crimes in this case which is a high number from the year 2008 and so on.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')


# The groupby above shows the highest number of Dacoity cases in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# This is a fairly low number and as the years have progressed the cases have decreased constantly.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)
s.head(10).style.background_gradient(cmap='Oranges')


# The groupby above shows the highest number of Robberies done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# This has also decreased fairly as years have progressed.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)
s.head(10).style.background_gradient(cmap='RdPu')


# The groupby above shows the highest number of Arson cases in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Bihar had the highest number of cases for two consecutive years in the same district.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')


# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)
s.head(10).style.background_gradient(cmap='Purples')


# The groupby above shows the highest number of Atrocities cases in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Bihar and Rajasthan are the states which are most affected here at a high number.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')


# The groupby above shows the highest number of Civil rights infringement done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Most of the cases are from Andhra Pradesh and Tamil Nadu in the initial years which have decreased over the years.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')


# The groupby above shows the highest number of Crimes against SCs done in the STATE/UT and DISTRICT column for all the years in the Dataset.
# 
# Tamil Nadu had consecutively the highest number of cases for the initial years which then dissipated later on. Rajasthan then took the lead for highest number of cases consecutively for 4 years.

# In[ ]:


s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')


# Tamil Nadu had the highest Hate Crime in the initial years which decreased a lot in the progressive years. Bihar and Rajasthan then had the highest number of cases followed by Andhra Pradesh.

# In[ ]:


sns.catplot(x='Year', y='Murder', data=cbdr,height = 5, aspect = 4)


# This plot shows Murder for successive years and on average 10 murders are done for every state per year for hate crimes. The highest was 24 for the year 2006.

# In[ ]:


sns.catplot(x='Year', y='Assault on women', data=cbdr,height = 5, aspect = 4)


# This plot shows that on average 20 Assaults are made per state on women every year and the highest was recorded for 2013 rising upto 35+.

# In[ ]:


sns.catplot(x='Year', y='Kidnapping and Abduction', data=cbdr,height = 5, aspect = 4)


# This plot shows kidnapping and abduction for all the districts. On average, there were around 10 for each year for each state. The 2008-2011 bracket showed the highest number of cases.

# In[ ]:


sns.catplot(x='Year', y='Dacoity', data=cbdr,height = 5, aspect = 4)


# This plot shows dacoity which is is high initially and it decreased over the years.

# In[ ]:


sns.catplot(x='Year', y='Robbery', data=cbdr,height = 5, aspect = 4)


# This plot shows cases of robbery which was high initially. It decreased quite a lot as the years progressed.

# In[ ]:


sns.catplot(x='Year', y='Arson', data=cbdr,height = 5, aspect = 4)


# This shows cases of Arson which was highest for 2009 and for rest of the years, it maintains a constant rate which is around 6.

# In[ ]:


sns.catplot(x='Year', y='Hurt', data=cbdr,height = 5, aspect = 4)


# This shows that many states have high number of hurt cases with the maximum people in the range of 0-50 and some of the cases reach upto 250. This shows that a huge number of people suffer in this bracket.

# In[ ]:


sns.catplot(x='Year', y='Prevention of atrocities (POA) Act', data=cbdr,height = 5, aspect = 4)


# This shows that a huge number of people are attacked which is around 0-100 for each state and they are shileded under this act.

# In[ ]:


sns.catplot(x='Year', y='Protection of Civil Rights (PCR) Act', data=cbdr,height = 5, aspect = 4)


# The cases for PCR was high initially and it decrased as the years progressed with an average for states at around 0-20.

# In[ ]:


sns.catplot(x='Year', y='Other Crimes Against SCs', data=cbdr,height = 5, aspect = 4)


# Hate crimes against SCs are high initially and they decreased as the years progressed. The crimes are almost at a mean of 200.

# In[ ]:


sns.catplot(x='Year', y='Total Atrocities', data=cbdr,height = 5, aspect = 4)


# The total atrocities for each state rise up to almost 400+ and they remain constant throughout the years.

# # EXPLORATORY DATA ANALYSIS ON THE BASIS OF STATE

# This adds total atrocities in the state and makes a new coloumn added to the CBSR dataset.

# In[ ]:


cbsr['Total Atrocities'] = cbsr['Murder'] +cbsr['Assault on women']+cbsr['Kidnapping and Abduction']+cbsr['Dacoity']+cbsr['Robbery']+cbsr['Arson']+cbsr['Hurt']+cbsr['Prevention of atrocities (POA) Act']+cbsr['Protection of Civil Rights (PCR) Act']+cbsr['Other Crimes Against SCs']
cbsr.head()


# In[ ]:


sns.relplot(x ='Total Atrocities', y ='Year', col = 'STATE/UT', data = cbsr, height=3 ,col_wrap = 9)


# **This shows the states and UTs over the successive years and the total number of cases rising in them.**
# 
# 1. States and UTs having a constant rate are - Arunanchal Pradesh, Assam, Goa, Gujarat, Haryana, Himachal Pradesh, Jammu & Kashmir, Kerala,Manipur,Meghalaya, Mizroam, Nagaland, Punjab, Sikkim, Tripura, Uttrakhand, West Bengal, A&N Islands, Chandigarh, D& N Haveli, Daman & Diu, Delhi, Lakshwadeep and Puducherry
# 
# 2. States and UTs showing flucatuations are Andhra Pradesh, Bihar, Karnataka, Jharkhand, Madhya Pradesh, Maharashtra, Odisha, Rajasthan, Tamil Nadu, Uttar Pradesh.
# 
# The worst cases are prevalent in Uttar Pradesh followed by Madhya Pradesh and Rajasthan. 

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')


# The highest number of murders are in Uttar Pradesh for the successive years.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)
s.head(10).style.background_gradient(cmap='Purples')


# Assault on women are having high cases in the state of Madhya Pradesh and Uttar Pradesh.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')


# Kidnapping and Abduction are all prevalent in Uttar Pradesh for the successive years.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')


# Dacoity is prevalent in Maharasthra which is highest during 2008-2010.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)
s.head(10).style.background_gradient(cmap='Oranges')


# Robbery had high cases in 2001 and then it went down to almost half of the original number. It has then successively fallen.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)
s.head(10).style.background_gradient(cmap='RdPu')


# Arson cases were high initially which then decreased over the years. However, Uttar Pradesh state had quite a number for the years.

# In[ ]:


s = cbsr.groupby(['STATE/UT','Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')


# Cases of attack have been most in Madhy Pradesh which is high for a lot of the years.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)
s.head(10).style.background_gradient(cmap='Purples')


# Prevention of attrocities were highest in initial years which decreased over the progressive years but in 2012 it almost reached the same number as 2001.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')


# PCR was high initially for Andhra Pradesh and it decreased over the years showing a good effect of authorities.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')


# Discriminatory crimes have maintained a constant rate over the years and has not dropped much.

# In[ ]:


s= cbsr.groupby(['STATE/UT','Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')


# Total number of Hate Crime are maximum in Uttar Pradesh which hasn't gone down a bit over the years.

# In[ ]:


x = cbsr['Year']
y = cbsr['Total Atrocities']


# In[ ]:


sns.axes_style('white')
sns.jointplot(x=x, y=y, kind = 'hex', color = 'green')


# This shows Total Crimes by states which has a high value of 300 for almost each state in a year.
# 
# The black Hexagon shows the mean for that particular year and the green fainted hexagon means the values the state reach upto.

# In[ ]:


f, ax = plt.subplots(figsize=(6,6))
cmap = sns.cubehelix_palette(as_cmap = True, dark=0,light = 1,reverse=True)
sns.kdeplot(x,y,cmap=cmap, n_levels = 60, shade= True)


# This shows a mean of around 300 for atrocities for each state in a year.

# # EXPLORATORY DATA ANALYSIS ON THE BASIS OF TOTAL DATA COMPRISING OF STATES+UTs
# 

# This shows total crimes for each year which is added in the form of column 'Total Atrocities' into the 'Total' DataFrame.

# In[ ]:


total['Total Atrocities'] = total['Murder'] +total['Assault on women']+total['Kidnapping and Abduction']+total['Dacoity']+total['Robbery']+total['Arson']+total['Hurt']+total['Prevention of atrocities (POA) Act']+total['Protection of Civil Rights (PCR) Act']+total['Other Crimes Against SCs']
total.head(15)


# In[ ]:


s= total.groupby(['Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)
s.head(15).style.background_gradient(cmap='Reds')


# This shows highest number of murders for 2001 and 2002 which then decreases as years progresses.

# In[ ]:


s= total.groupby(['Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)
s.head(15).style.background_gradient(cmap='Blues')


# This shows that the assault on women have increasing recently and were less in the initial years.

# In[ ]:


s= total.groupby(['Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)
s.head(12).style.background_gradient(cmap='Purples')


# This shows kindnapping and abduction has increased over the successive years.

# In[ ]:


s= total.groupby(['Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)
s.head(15).style.background_gradient(cmap='Greens')


# This shows that dacoity reached a high around the middle years and were less on each end.

# In[ ]:


s= total.groupby(['Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)
s.head(15).style.background_gradient(cmap='Oranges')


# This shows that robbery was high for initial years and decreased a lot as years progressed.

# In[ ]:


s= total.groupby(['Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)
s.head(15).style.background_gradient(cmap='RdPu')


# This shows cases of arson decreasing over the years. They were high initially.

# In[ ]:


s = total.groupby(['Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)
s.head(15).style.background_gradient(cmap='Greys')


# This shows the attack were very high in initial years. Over the successive years, they have gone down but not a huge rate.

# In[ ]:


s= total.groupby(['Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)
s.head(15).style.background_gradient(cmap='Purples')


# This shows POA hgh for 2001 and then it goes on decreasing until it becomes high again from 2010.

# In[ ]:


s= total.groupby(['Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)
s.head(15).style.background_gradient(cmap='Greens')


# PCR was very high in the initial years but decreased to a small number as the years progressed.

# In[ ]:


s= total.groupby(['Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)
s.head(15).style.background_gradient(cmap='Blues')


# Discriminatory crimes have been high for almost all years but in the recent years, they have gone much up.

# In[ ]:


s= total.groupby(['Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)
s.head(15).style.background_gradient(cmap='Greys')


# The total number of crimes were high initially and then they decreased. After 2010, these hate crimes have attained a high rate again.

# In[ ]:


sns.catplot(x='Year', y='Murder', data=total ,height = 5, aspect = 4,kind = 'bar')


# This shows an average of 700 murders are done in Hate Crimes in a year.

# In[ ]:


sns.catplot(x='Year', y='Assault on women', data=total ,height = 5, aspect = 4,kind = 'bar')


# This shows the assault on women have increased over the years.

# In[ ]:


sns.catplot(x='Year', y='Kidnapping and Abduction', data=total ,height = 5, aspect = 4,kind = 'bar')


# Kidnapping and Abduction have also increased over the years reaching an all time high for 2011.

# In[ ]:


sns.catplot(x='Year', y='Dacoity', data=total ,height = 5, aspect = 4,kind = 'bar')


# Dacoity reached a low number in the median years but rose up sharply. For the last quantile of years, it has gone down.

# In[ ]:


sns.catplot(x='Year', y='Robbery', data=total ,height = 5, aspect = 4,kind = 'bar')


# This shows robbery was very high initially and it went to a lower number over the years.

# In[ ]:


sns.catplot(x='Year', y='Arson', data=total ,height = 5, aspect = 4,kind = 'bar')


# Cases of arson were also high during the initial years with successive years making it low.

# In[ ]:


sns.catplot(x='Year', y='Hurt', data=total ,height = 5, aspect = 4,kind = 'bar')


# POA has been high for almost all the years and hasn't gone down much for even a single year. This shows this has maintained almost a constant rate for the years.

# In[ ]:


sns.catplot(x='Year', y='Prevention of atrocities (POA) Act', data=total ,height = 5, aspect = 4,kind = 'bar')


# POA has fairly a high number already and it has a mean of about 9000 which is high. Initially, it was high and for the middle quantile it decreased. For the last quantile, it again increased.

# In[ ]:


sns.catplot(x='Year', y='Protection of Civil Rights (PCR) Act', data=total ,height = 5, aspect = 4,kind = 'bar')


# This shows that over the years PCR has decreased to a very small number showing better effect in this case.

# In[ ]:


sns.catplot(x='Year', y='Other Crimes Against SCs', data=total ,height = 5, aspect = 4,kind = 'bar')


# Discriminatory attacks have increased successively over the years and they are the highest magnitude of crimes done in India.

# In[ ]:


sns.catplot(x='Year', y='Total Atrocities', data=total ,height = 5, aspect = 4,kind = 'bar')


# **This shows that almost 30000+ Hate Crimes have been done in India from 2001-2012. The crimes went down for a bit but has maintained a constant rate. This shows that a lot of work has to be done in bringing hate crimes to a low number.**
