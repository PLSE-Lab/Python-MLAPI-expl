#!/usr/bin/env python
# coding: utf-8

# ##    ## THIS NOTEBOOK USES DATA UNTIL 9 AUGUST 2020. CHECK  THE PREVIOUS VERSIONS FOR EARLIER DATES. SINCE IT USES DAILY UPDATED DATA, IF YOU RUN THIS NOTEBOOK LATER YOU WILL HAVE THE LATEST POSSIBLE GRAPHS. ONLY THE COMMENTS CAN CHANGE WITH THE UPDATED DATA.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


df.head(50)


# In[ ]:


df.tail(50)


# In[ ]:


df = df.rename(columns={'Province/State': 'Province','Country/Region':'Country','ObservationDate':'Date','Confirmed':'Cases'})


# In[ ]:


df.describe().T


# ### Preprocessing
# ##### Renaming Mainland China as China, formatting date,dropping Sno and zero confirmed observations

# In[ ]:


df.Country.replace({'Mainland China': 'China'}, inplace=True)
df['Date'] = df['Date'].apply(pd.to_datetime)
df.drop(['SNo'],axis=1,inplace=True)
df.drop(list(df[df.Cases==0.0].index), axis = 0, inplace=True)


# ##### Missing values

# In[ ]:


df.isnull().sum()


# #### Examining missing provinve data 

# In[ ]:


dfmissing=df[df['Province'].isnull()]['Country'].value_counts().to_frame(name='Missing_Province_count')
dfcountry=df['Country'].value_counts().to_frame(name='Country_count')
mergedDf = dfcountry.merge(dfmissing, left_index=True, right_index=True)


# In[ ]:


mergedDf[mergedDf.Missing_Province_count>1]


# ##### Assigning country names for missing provinces as there are almost no province data for those countries 
# ##### (Excluding Australia for which it is set the most frequent province)

# In[ ]:


df[df.Country=='Australia']=df[df.Country=='Australia'].fillna(df[df.Country=='Australia']["Province"].mode()[0])

df["Province"]=df["Province"].fillna(df["Country"]);


# ### Countries in Descending order with respect to number of cases

# In[ ]:


df.sort_values(by='Cases', ascending=False)['Country'].unique()


# In[ ]:


df[df.Country=='Others']


# ##### Assigning the cruise ship to Japan as it was mostly situated in Japan

# In[ ]:


df.Country.replace({'Others': 'Japan'}, inplace=True)


# In[ ]:


countries=df.sort_values(by='Cases', ascending=False)['Country'].unique()


# ## Visualisations

# In[ ]:


print("\nNumber of countries affected by virus: ",len(countries))


# ### Descriptives in descending order with respect to maximum cases in all data for the first 40 country

# In[ ]:


df.groupby("Country").aggregate(['mean', np.std,max,'count']).sort_values([('Cases','max')], ascending=False).head(40)


# ### Descriptives in descending order with respect to maximum deaths in all data for the first 40 country

# In[ ]:


df.groupby("Country").aggregate(['mean', np.std,max,'count']).sort_values([('Deaths','max')], ascending=False).head(40)


# In[ ]:


dftotal=df.groupby(['Date','Country'])['Cases','Deaths','Recovered'].sum()
dftotal.reset_index(inplace=True)  
dftotal['Death_rate']=dftotal['Deaths']/dftotal['Cases']
dftotal['Recovery_rate']=dftotal['Recovered']/dftotal['Cases']


# In[ ]:


dftotal[dftotal.Country=='US']


# ## First 10 Countries with respect to highest number of Log (Coronavirus Cases)

# In[ ]:


dftotal['log(Cases)']=np.log(dftotal.Cases)


# The growth of the confirmed case rates seem to stabilize only in Italy,Spain, France and UK. There is a significant accelation in the case groeth rates in  India,  Chile and Brazil. Despite some drop in their accelations, the cases in Russia,Iran and US seem to be far from stabilization.

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="log(Cases)", hue='Country',linewidth=6,  data=dftotal[dftotal.Country.isin(countries[0:10])]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=4, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# ## Cases after 15/03/2020

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="log(Cases)", hue='Country',linewidth=6,  data=dftotal[(dftotal.Country.isin(countries[0:10]))&(dftotal.Date>'2020-04-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=4, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# ##  Log(Cases) for the next 20 countries 

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="log(Cases)", hue='Country',linewidth=6,  data=dftotal[dftotal.Country.isin(countries[10:30])]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=2, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# ## Cases after 15/04/2020 for the 11-20th countries 

# Excluding China, Germany and Belgium, rapid rise of Coronavirus continues in all other 7 countries. Only in Turkey and Qatar there are some drops in the accelaration of the growth rate but they are also still far from stabilization.

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="log(Cases)", hue='Country',linewidth=6,  data=dftotal[(dftotal.Country.isin(countries[10:20]))&(dftotal.Date>'2020-04-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=4, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# ## Cases after 15/04/2020 for the 20-30th countries

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="log(Cases)", hue='Country',linewidth=6,  data=dftotal[(dftotal.Country.isin(countries[20:30]))&(dftotal.Date>'2020-04-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=4, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# In[ ]:





# ## Deaths rates for the top 10 counries according to their number of cases

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="Death_rate", hue='Country', linewidth=6, data=dftotal[(dftotal.Country.isin(countries[0:10]))&(dftotal.Date>'2020-03-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=2, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# ### Recovery rates after 15/03/2020

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="Recovery_rate", hue='Country',linewidth=6, data=dftotal[(dftotal.Country.isin(countries[0:10]))&(dftotal.Date>'2020-03-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=2, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


#  ## Death rates for the next 20 countries after 15/03/2020

# In[ ]:


plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
sns.set_style("whitegrid")
sns.lineplot(x="Date", y="Death_rate", hue='Country',linewidth=6,  data=dftotal[(dftotal.Country.isin(countries[10:29]))&(dftotal.Date>'2020-03-15')]);
plt.xticks(rotation=45,ha='right');
sns.set(rc={'figure.figsize':(14,14)})
leg = plt.legend(fontsize='x-large',loc=2, facecolor='white', )
# set the linewidth of each legend object
for i in leg.legendHandles:
    i.set_linewidth(10.0)


# In[ ]:




