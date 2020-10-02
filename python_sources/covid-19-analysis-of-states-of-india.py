#!/usr/bin/env python
# coding: utf-8

# #                     CORONAVIRUS IN INDIA
# 
# the coronavirus in india is spreading like a wildfire and its not stopping.So what is india doing to fight against it in this piece i will try to analyze weather the locdown imposed worked and is it the write time to unlock

# So in this piece i will analyze the top 5 states in india with the most coronavirus cases
# 

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("..//input//covid19-in-india//covid_19_india.csv")
df.groupby(["Date"], sort=False)['ConfirmedIndianNational','ConfirmedForeignNational']
df_date=df[df['Date']=='06/07/20']
df_date_top=df_date[(df_date['Confirmed'] > 25000) ]


# In[ ]:


def con(str):
    df = pd.read_csv("covid_19_india.csv")
    df_delhi = df[(df['State/UnionTerritory']==str)]
    df_1=df_delhi[(df_delhi['Date']=='23/03/20')]
    df_2=df_delhi[(df_delhi['Date']=='15/04/20')]
    df_3=df_delhi[(df_delhi['Date']=='04/05/20')]
    df_4=df_delhi[(df_delhi['Date']=='18/05/20')]
    df_5=df_delhi[(df_delhi['Date']=='08/06/20')]
    print(df_1,df_2,df_3,df_4,df_5)


# In[ ]:


def test(str):
    df_test = pd.read_csv("StatewiseTestingDetails.csv")
    df_delhi_test=df_test[(df_test['State']==str)]
    df_1=df_delhi_test[(df_delhi_test['Date']=='2020-03-23')]
    df_2=df_delhi_test[(df_delhi_test['Date']=='2020-04-15')]
    df_3=df_delhi_test[(df_delhi_test['Date']=='2020-05-04')]
    df_4=df_delhi_test[(df_delhi_test['Date']=='2020-05-18')]    
    df_5=df_delhi_test[(df_delhi_test['Date']=='2020-06-08')] 
    print(df_1,df_2,df_3,df_4,df_5)


# In[ ]:


plt.figure()
ax=plt.gca()
plt.bar(df_date_top['State/UnionTerritory'],df_date_top['Confirmed'] , width = 0.3,alpha = 0.5,color='grey')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('States')
plt.ylabel('Confirmed Cases')
plt.title('States V/S Confirmed Cases')
plt.rc('ytick', labelsize=7)  


# In[ ]:


df = pd.read_csv("..//input//covid19-in-india//covid_19_india.csv")
df.groupby(["Date"], sort=False)['ConfirmedIndianNational','ConfirmedForeignNational']
df_date=df[df['Date']=='06/07/20']
df_date.sort_values('Confirmed' , ascending=False).head(5)


# As we can see the top 5 are Delhi,Gujarat,Maharashtra,Tamil NAdu and Uttar pradesh in no particular order
# 

# In[ ]:


df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
total_samples_country=df_test[(df_test['Date']=='2020-07-05')]
plt.figure()
plt.pie('TotalSamples',labels= 'State', autopct='%1.1f%%'#,colors=my_colors
        ,data=total_samples_country.sort_values('TotalSamples', ascending=False).head(10))
plt.axis('equal')
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Percentage of testing in top 10 States\n', size=15)
plt.show()


# In[ ]:


total_samples_country.sort_values('TotalSamples', ascending=False).head(10)


# The above table shows the 10 states with the highest number of testing samples in descending order with the pie chart showing the same.Now we will see various aspects of the pandemic and how it is affecting the the top 5 states .

#  # **Delhi**

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Delhi')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi['Date'],df_delhi['Confirmed'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Dates V/S Confirmed Cases')
ax.annotate('Lockdown1', xy =('23/03/20',29), xytext=('23/03/20',15000),ha='center',
                arrowprops = dict(facecolor ='Green',shrink = 0.05),) 
ax.annotate('Lockdown 2', xy =('15/04/20',1561), xytext=('15/04/20',25000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('04/05/20',4549), xytext=('04/05/20',35000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('18/05/20',10054), xytext=('18/05/20',45000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('08/06/20',27654), xytext=('08/06/20',5000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 


# As we can see the cases have been rising eversince the first lockdown and event though the lockdowns helped in keeping the rate of increase in check,after the unlock as we can see in the graph the rate has increased a lot 

# In[ ]:


df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_delhi_test=df_test[(df_test['State']=='Delhi')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi_test['Date'],df_delhi_test['TotalSamples'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Total Samples')
plt.title('Dates V/S Total Samples')
ax.annotate('Lockdown 2', xy =('2020-04-15',16605), xytext=('2020-04-15',55000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('2020-05-04',64108), xytext=('2020-05-04',15000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('2020-05-18',139727), xytext=('2020-05-18',185000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('2020-06-08',255615), xytext=('2020-06-08',200000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 


# The testing has also seen a great amount of increase in Delhi  but compared to the cases ,it is a bit underwhelming and also a fact to look at is the testing started around lockdown 2.the rate as we can see has a constant rate of increase ulike the cases graph but we can observe after the unlock the rate at which testing is done has seen a massive increase 

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Delhi')]
df_dt=df_delhi[(df_delhi['Date']=='06/07/20')]
labels = 'Cured', 'Death', 'Unknown'
sizes = [71.7, 3.0, 25.3]
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# the above is the pie chart for the percentage cured,unkown and deaths patients

# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_bed[(df_bed['State/UT']=='Delhi')]


# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_bed[(df_bed['State / Union Territory']=='Delhi')]


# the above two table can provide a comparision between number of beds to the population

# # GUJARAT
# ****

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Gujarat')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi['Date'],df_delhi['Confirmed'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Dates V/S Confirmed Cases')
ax.annotate('Lockdown1', xy =('23/03/20',29), xytext=('23/03/20',5000),ha='center',
                arrowprops = dict(facecolor ='Green',shrink = 0.05),) 
ax.annotate('Lockdown 2', xy =('15/04/20',695), xytext=('15/04/20',10000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('04/05/20',5428), xytext=('04/05/20',15000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('18/05/20',11379), xytext=('18/05/20',20000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('08/06/20',20070), xytext=('08/06/20',10000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 


df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_delhi_test=df_test[(df_test['State']=='Gujarat')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi_test['Date'],df_delhi_test['TotalSamples'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Total Samples')
plt.title('Dates V/S Total Samples')
ax.annotate('Lockdown 2', xy =('2020-04-15',19197), xytext=('2020-04-15',55000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('2020-05-04',84648), xytext=('2020-05-04',15000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('2020-05-18',148824), xytext=('2020-05-18',185000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('2020-06-08',256289), xytext=('2020-06-08',200000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 

df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Gujarat')]
df_dt=df_delhi[(df_delhi['Date']=='06/07/20')]


labels = 'Cured', 'Death', 'Unknown'
sizes = [71.8,0.05,28.15]
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')


# This is Gujarat's comparsion of their testing and confirmed cases

# In[ ]:


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# With 71.8 percent death rate gujarat is doing a good job

# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_bed[(df_bed['State/UT']=='Gujarat')]


# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_bed[(df_bed['State / Union Territory']=='Gujarat')]


# Gujarat comparison between number of beds and the population

# # TAMIL NADU

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Tamil Nadu')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi['Date'],df_delhi['Confirmed'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Dates V/S Confirmed Cases')
ax.annotate('Lockdown1', xy =('23/03/20',9), xytext=('23/03/20',9000),ha='center',
                arrowprops = dict(facecolor ='Green',shrink = 0.05),) 
ax.annotate('Lockdown 2', xy =('15/04/20',1204), xytext=('15/04/20',10000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('04/05/20',3023), xytext=('04/05/20',15000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('18/05/20',11224), xytext=('18/05/20',20000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('08/06/20',31667), xytext=('08/06/20',10000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 

df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_delhi_test=df_test[(df_test['State']=='Tamil Nadu')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi_test['Date'],df_delhi_test['TotalSamples'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Total Samples')
plt.title('Dates V/S Total Samples')
ax.annotate('Lockdown 2', xy =('2020-04-15', 21994), xytext=('2020-04-15',150000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('2020-05-04',162970), xytext=('2020-05-04',5000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('2020-05-18',337841), xytext=('2020-05-18',185000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('2020-06-08',607952), xytext=('2020-06-08',200000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 





# This is Tamil Nadu's comparsion of their testing and confirmed cases

# In[ ]:


labels = 'Cured', 'Death', 'Unknown'
sizes = [56.4,1.3,42.3]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# with cured being only 56.4% it can be said Tamil nadu is doing a poor job 

# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_bed[(df_bed['State/UT']=='Tamil Nadu')]


# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_bed[(df_bed['State / Union Territory']=='Tamil Nadu')]


# This is Tamil Nadu's comaprision between the number of beds to the pouplation 

# # MAHARASHTRA

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Maharashtra')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi['Date'],df_delhi['Confirmed'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Dates V/S Confirmed Cases')
ax.annotate('Lockdown1', xy =('23/03/20', 74 ), xytext=('23/03/20',45000),ha='center',
                arrowprops = dict(facecolor ='Green',shrink = 0.05),) 
ax.annotate('Lockdown 2', xy =('15/04/20',2687), xytext=('15/04/20',50000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('04/05/20',12974), xytext=('04/05/20',55000),ha='center', 
            
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('18/05/20',33053), xytext=('18/05/20',65000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('08/06/20',85975), xytext=('08/06/20',15000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 



df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_delhi_test=df_test[(df_test['State']=='Maharashtra')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi_test['Date'],df_delhi_test['TotalSamples'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Total Samples')
plt.title('Dates V/S Total Samples')
ax.annotate('Lockdown 2', xy =('2020-04-15', 45142), xytext=('2020-04-15',150000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('2020-05-04',168374), xytext=('2020-05-04',5000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('2020-05-18',282437), xytext=('2020-05-18',185000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('2020-06-08',565290), xytext=('2020-06-08',200000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 





# This is Maharashtra's comparision between confirmed cases and the total samples.Maharashtra is with the highest number of corona cases .above we can see that their rate of testing is incrasing at a good rate but so is the confirmed cases 

# In[ ]:


labels = 'Cured', 'Death', 'Unknown'
sizes = [54.1,4.2,41.7]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 


# with only 54.1% cured,Maharshtra is doing a poor job 

# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_bed[(df_bed['State/UT']=='Maharashtra')]


# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_bed[(df_bed['State / Union Territory']=='Maharashtra')]


# Maharshtra's ratio of number of beds to urban population must be more as it is the worst affected state 

# # UTTAR PRADESH

# In[ ]:


df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_delhi = df[(df['State/UnionTerritory']=='Uttar Pradesh')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi['Date'],df_delhi['Confirmed'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Dates V/S Confirmed Cases')
ax.annotate('Lockdown1', xy =('23/03/20', 31 ), xytext=('23/03/20',4500),ha='center',
                arrowprops = dict(facecolor ='Green',shrink = 0.05),) 
ax.annotate('Lockdown 2', xy =('15/04/20',735), xytext=('15/04/20',5000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('04/05/20',2742), xytext=('04/05/20',5500),ha='center', 
            
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('18/05/20',4259), xytext=('18/05/20',6500),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('08/06/20',10536), xytext=('08/06/20',7000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 



df_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_delhi_test=df_test[(df_test['State']=='Uttar Pradesh')]
plt.figure(figsize=(8,8))
plt.plot(df_delhi_test['Date'],df_delhi_test['TotalSamples'],'-r')
plt.rc("xtick",labelsize=9)
plt.rc("ytick",labelsize=10)
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Total Samples')
plt.title('Dates V/S Total Samples')
ax.annotate('Lockdown 2', xy =('2020-04-15', 19506), xytext=('2020-04-15',150000),ha='center',
                arrowprops = dict(facecolor ='Blue',shrink = 0.05),) 
ax.annotate('Lockdown 3', xy =('2020-05-04',98300), xytext=('2020-05-04',5000),ha='center', 
                arrowprops = dict(facecolor ='Yellow',shrink = 0.05),) 
ax.annotate('Lockdown 4', xy =('2020-05-18',176479), xytext=('2020-05-18',255000),ha='center',
                arrowprops = dict(facecolor ='Black',shrink = 0.05),) 
ax.annotate('Unlock 1.0', xy =('2020-06-08',380723), xytext=('2020-06-08',200000),ha='center',
                arrowprops = dict(facecolor ='orange',shrink = 0.05),) 





# above is the comaprision of Uttar Pradesh's total samples to the confirmed case

# In[ ]:


labels = 'Cured', 'Death', 'Unknown'
sizes = [67.7,2.8,29.5]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 


# with 67.7% uttar pradesh is doing a resonable job 

# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_bed[(df_bed['State/UT']=='Uttar Pradesh')]


# In[ ]:


df_bed = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_bed[(df_bed['State / Union Territory']=='Uttar Pradesh')]


# Above is the comparision between the beds and population of Uttar Pradesh

# # CONCLUSION

# So from the above graphs i can conclude that while lockdown was benefitial in curbing thr rate of increase ,it was not as usefull as it was in other countries.And with the unlock the above graphs clearly show that the rate has skyrockted and not showing any signs of stopping.
# 
# And also with the increasing cases,we will need to increase the number of beds considerably beacuse even with the current number our beds to patient ratio is not enough and with the number increasing it will be a must.
