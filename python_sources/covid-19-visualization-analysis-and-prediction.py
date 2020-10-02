#!/usr/bin/env python
# coding: utf-8

# Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans. The virus identified as the cause of the recent outbreak is being referred to as 2019-nCoV.
# 
# A virus that was first reported in the Chinese city of Wuhan, has now spread to more than a dozen countries across the world, sparking an unprecedented health and economic crisis which later declared as Pandemic by WHO due to high rate spreads throughout the world. Currently (on date 14 April 2020), this leads to more than a 1.9 million infected cases and 1.2 million Deaths across the globe. Pandemic is spreading all over the world.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Import Libraries

# In[ ]:


# Importing Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt,datetime
from datetime import timedelta
import folium

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ### **Symptoms of Corona Virus**

# In[ ]:


symptoms={'symptom':['Fever',
                'Dry cough',
                'Fatigue',
                'Sputum production',
                'Shortness of breath',
                'Muscle pain',
                'Sore throat',
                'Headache',
                'Chills',
                'Nausea or vomiting',
                'Nasal congestion',
                'Diarrhoea',
                'Haemoptysis',
                'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms


# In[ ]:


plt.figure(figsize=(12,8))
plt.title('Symptoms of Coronavirus',fontsize=20)    
plt.pie(symptoms['percentage'],autopct='%1.1f%%')
plt.legend(symptoms['symptom'],bbox_to_anchor=(1.45, 0.8),loc="upper center")

centre_circle = plt.Circle((0,0),0.50,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show() 


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

text=symptoms['symptom'].to_list()
plt.figure(figsize=(10,6))
wordcloud=WordCloud(max_words=200,background_color='white',).generate(str(text))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# ### Reading Datasets

# In[ ]:


data=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
confirmed_data=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_data=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recovered_data=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
data.head()


# In[ ]:


#getting a summary of the columns

print("Size/Shape of the dataset: ",data.shape)
print("Checking for null values:\n",data.isnull().sum())
print("Checking Data-type of each column:\n",data.dtypes)


# ### Data Exploration/Analysis

# In[ ]:


# Changing Datatypes of 'Confirmed', 'Deaths' and 'Recovered'
data[['Confirmed','Deaths','Recovered']]=data[['Confirmed','Deaths','Recovered']].astype(int)

# Renaming Columns name
data.rename(columns={'Country/Region':'Country','Province/State':'State','ObservationDate':'Observation Date'},inplace=True)
#data.tail()

# Creating column of Active Case
data['Active']=data['Confirmed']-(data['Recovered']+data['Deaths'])
#data.tail()

# Replacing 'Mainland China' with 'China'
data['Country']=data['Country'].replace('Mainland China','China')

# #converting 'Date' column to datetime
data['Observation Date']=pd.to_datetime(data['Observation Date'], format='%m/%d/%Y')
data['Date1'] = pd.to_datetime('2020/01/22')
data['Days']= data['Observation Date'] - data['Date1']
data['Days'] = data['Days'].astype(str).str[:2]
data['Days'] = data[['Days']].apply(pd.to_numeric)
data['Observation Date']=pd.to_datetime(data['Observation Date'],unit='ns').dt.date
data.drop('Date1',inplace=True, axis=1)
data.tail()


# ### Description of Data

# In[ ]:


data.describe()


# ### Correlation Analysis

# In[ ]:


data.corr().style.background_gradient(cmap='Reds')


# In[ ]:


sns.heatmap(data.corr(),annot=True)


# **There is no strong correlation between any of the variables except for Confirmed, Deaths and Recovered variables**

# ### Total impacted countries across the world 
# 
# **A total of 220 countries have been impacted uptill now.**

# In[ ]:


#listing all the countries where the virus has spread to
countries=data['Country'].unique().tolist()
print('Total countries affected by virus : ',len(countries))
print('\ncountries:', countries)


# ### Latest data of Corona Virus
# 
# **Total number of confirmed cases, deaths reported, revoveries and active cases all across the world**

# In[ ]:


world_data = data.groupby('Observation Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
world = world_data[world_data['Observation Date']==max(world_data['Observation Date'])].reset_index(drop=True)
world['Global Moratality'] = world['Deaths']/world['Confirmed']
world['Deaths per 100 Confirmed Cases'] = world['Global Moratality']*100
world.style.background_gradient(cmap='Pastel2')


# ## Countrywise Analysis
# **US now leads with highest confirmed cases all over the world follwing Spain, Italy, France, Germany and UK. This scenario is in total contrast to the initial days when China accounted for nearly 99% of the cases. China moved to 7th place with respect to the number of confirmed cases as UK takes over**

# In[ ]:


Data=data[data['Observation Date']==max(data['Observation Date'])].reset_index()
data_country_wise=Data.groupby(['Country'])[["Confirmed","Recovered","Deaths","Active"]].sum().reset_index().sort_values('Confirmed',ascending=False).reset_index(drop=True)
data_country_wise['Mortality Rate']=round((data_country_wise['Deaths']/data_country_wise['Confirmed'])*100,2)
data_country_wise['Recovery Rate']=round((data_country_wise['Recovered']/data_country_wise['Confirmed'])*100,2)
data_country_wise['Confirmed Case Rate']=round((data_country_wise['Confirmed']/data_country_wise['Confirmed'].sum())*100,2)
#data_country_wise.style.background_gradient(cmap='Oranges')

data_country_wise.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate"])                        .background_gradient(cmap='Oranges',subset=["Recovery Rate"])                        .background_gradient(cmap='Greens',subset=["Confirmed Case Rate"])                      


# ### Corona virus cases: Confirmed, Deaths, Recovered and Active 

# In[ ]:


most_country=data_country_wise.groupby('Country')["Confirmed","Recovered","Deaths","Active"].sum()
#most_country.head()

plt.figure(figsize=(15,35))

plt.subplot(4,1,1)
most_country[most_country['Confirmed'] > 10000]['Confirmed'].sort_values(ascending=False).plot(kind='pie',autopct='%1.1f%%')
plt.title("Countries Confirmed more than 10000 Cases", fontsize=15)
plt.xlabel('Country');
plt.ylabel('No. of Cases');
plt.xticks(rotation='30')

plt.subplot(4,1,2)

most_country[most_country['Deaths'] > 1000]['Deaths'].sort_values(ascending=False).plot(kind='pie',autopct='%1.1f%%')
plt.title("Countries Deaths more than 1000 Cases", fontsize=15)
plt.xlabel('Country');
plt.ylabel('No. of Cases');
plt.xticks(rotation='30')

plt.subplot(4,1,3)

most_country[most_country['Active'] > 10000]['Active'].sort_values(ascending=False).plot(kind='pie',autopct='%1.1f%%')
plt.title("Countries Active more than 10000 Cases", fontsize=15)
plt.xlabel('Countries');
plt.ylabel('No. of Cases');
plt.xticks(rotation='30')

plt.subplot(4,1,4)

most_country[most_country['Recovered'] > 5000]['Recovered'].sort_values(ascending=False).plot(kind='pie',autopct='%1.1f%%')
plt.title("Countries Recovered more than 5000 Cases", fontsize=15)
plt.xlabel('Countries');
plt.ylabel('No. of Cases');
plt.xticks(rotation='30')


plt.show()


# ### Top 10 Infected Countries

# In[ ]:


top10=data_country_wise.head(10)
X=top10.Country
Y=top10.Confirmed
plt.figure(figsize=(15,9))
sns.barplot(X,Y,order=X,palette='RdBu').set_title('Most 10 Infected Country')
plt.xticks(rotation=90,fontsize=12)

for i, v in enumerate(top10['Confirmed']):
    plt.text(i-.25, v,
              top10['Confirmed'][i], 
              fontsize=12 )


# ### Top 10 infected countries: Confirmed vs Deaths vs Recovered 

# In[ ]:


f, ax = plt.subplots(figsize=(10,8))
bar1=sns.barplot(x="Confirmed",y="Country",data=top10,
            label="Confirmed", color="darkcyan")


bar2=sns.barplot(x="Recovered", y="Country", data=top10,
            label="Recovered", color="gold")


bar3=sns.barplot(x="Deaths", y="Country", data=top10,
            label="Deaths", color="darkred")

plt.xlabel('No of Cases', fontsize=15)
plt.ylabel('Country',fontsize=15)
plt.title('Confirmed vs Recovery vs Deaths ',fontsize=20)
ax.legend(loc=4, ncol = 1)
plt.show()


# ### Mortality Rate Analysis Country wise 

# In[ ]:


mortality_10=data_country_wise.sort_values(by='Deaths',ascending=False).reset_index(drop=True).head(10)
plt.figure(figsize=(8,5))
plt.style.use('ggplot') # ggplot for grid
ars=mortality_10.sort_values('Mortality Rate',ascending=True).head(10)
ax=ars.plot(kind='barh',x='Country',y='Mortality Rate',color='salmon',title='World wide Mortality rate',figsize=(10,6))

plt.show()


# ### Datewise Analysis 

# In[ ]:


main_cols=["Confirmed","Recovered","Deaths","Active"]


data_date_wise=data.groupby(['Observation Date'])[main_cols].sum().reset_index(drop=None)
data_date_wise[['Confirmed','Deaths','Recovered']]=data_date_wise[['Confirmed','Deaths','Recovered']].astype(int)
data_date_wise['daily_cases']=data_date_wise.Confirmed.diff()
data_date_wise['daily_deaths']=data_date_wise.Deaths.diff()
data_date_wise['daily_recoveries']=data_date_wise.Recovered.diff()
data_date_wise['daily_cases']=data_date_wise['daily_cases'].replace(np.nan, 0, regex=True)
data_date_wise['daily_deaths']=data_date_wise['daily_deaths'].replace(np.nan, 0, regex=True)
data_date_wise['daily_recoveries']=data_date_wise['daily_recoveries'].replace(np.nan, 0, regex=True)
data_date_wise['daily_cases']=data_date_wise['daily_cases'].astype(int)
data_date_wise['daily_deaths']=data_date_wise['daily_cases'].astype(int)
data_date_wise['daily_recoveries']=data_date_wise['daily_cases'].astype(int)
data_date_wise.nlargest(5,'Confirmed')


# ### Statstics Analysis across the World

# In[ ]:


plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(data_date_wise.index, world_data['Confirmed'],color='b')
plt.plot(data_date_wise.index, world_data['Deaths'],color='r')
plt.plot(data_date_wise.index, world_data['Recovered'],color='g')
plt.plot(data_date_wise.index, world_data['Active'],color='y')
plt.bar(data_date_wise.index,world_data['Confirmed'],alpha=0.2,color='c')
plt.title('Statistics of Worlds Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('No. of Cases', size=20)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['Confirmed','Deaths','Recovered','Active Cases'])

plt.subplot(122)
plt.plot(data_date_wise.index,np.log10(world_data['Confirmed']))
plt.plot(data_date_wise.index, np.log10(world_data['Deaths']))
plt.plot(data_date_wise.index, np.log10(world_data['Recovered']))
plt.title('Coronavirus Cases Over Time on Log Scale', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('No.of Cases (Logerithmic scale)', size=20)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['Confirmed','Deaths','Recovered'])

plt.show()


# ### Daily Increases in Confirmed Cases and Deaths Worldwide

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.bar(data_date_wise['Observation Date'],data_date_wise['daily_cases'])
plt.title('World Daily Increases in Confirmed Cases ', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)

plt.subplot(212)
plt.bar(data_date_wise['Observation Date'],data_date_wise['daily_deaths'])
plt.title('World Daily Increases in Deaths Cases ', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


# ### Daily Increase: Confirmed vs Deaths vs Recovered

# In[ ]:


plt.figure(figsize=(15,6))
plt.plot(data_date_wise["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",marker='o')
plt.plot(data_date_wise["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",marker='*')
plt.plot(data_date_wise["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",marker='^')
plt.xlabel("Timestamp")
plt.ylabel("Daily Increment")
plt.title("Daily increase in different Types of Cases Worldwide")
plt.xticks(rotation=90)
plt.legend()

print("Average increase in number of Confirmed Cases every day: ",np.round(data_date_wise["Confirmed"].diff().fillna(0).mean()))
print("Average increase in number of Recovered Cases every day: ",np.round(data_date_wise["Recovered"].diff().fillna(0).mean()))
print("Average increase in number of Deaths Cases every day: ",np.round(data_date_wise["Deaths"].diff().fillna(0).mean()))


# ### Growth Factor of different Types of Cases Worldwide

# In[ ]:


daily_increase_confirm=[]
daily_increase_recovered=[]
daily_increase_deaths=[]
for i in range(data_date_wise.shape[0]-1):
    daily_increase_confirm.append(((data_date_wise["Confirmed"].iloc[i+1]/data_date_wise["Confirmed"].iloc[i])))
    daily_increase_recovered.append(((data_date_wise["Recovered"].iloc[i+1]/data_date_wise["Recovered"].iloc[i])))
    daily_increase_deaths.append(((data_date_wise["Deaths"].iloc[i+1]/data_date_wise["Deaths"].iloc[i])))
daily_increase_confirm.insert(0,1)
daily_increase_recovered.insert(0,1)
daily_increase_deaths.insert(0,1)

plt.figure(figsize=(15,5))
plt.plot(data_date_wise.index,daily_increase_confirm,label="Growth Factor Confiremd Cases",marker='o')
plt.plot(data_date_wise.index,daily_increase_recovered,label="Growth Factor Recovered Cases",marker='*')
plt.plot(data_date_wise.index,daily_increase_deaths,label="Growth Factor Death Cases",marker='^')
plt.xlabel("Timestamp")
plt.ylabel("Growth Factor")
plt.title("Growth Factor of different Types of Cases Worldwide")
plt.axhline(1,linestyle='--',color='black',label="Baseline")
plt.xticks(rotation=90)
plt.legend()


# ### Mortality Rate and Recovery Rate across the World Over Time

# In[ ]:


# Calculate Mortality rate and Recovery rate

mortality_rate=(data_date_wise.Deaths/data_date_wise.Confirmed)*100
recovery_rate=(data_date_wise.Recovered/data_date_wise.Confirmed)*100

mean_mortality_rate = np.mean(mortality_rate)
mean_recovery_rate=np.mean(recovery_rate)

plt.figure(figsize=(15, 7))

plt.subplot(121)
plt.plot(data_date_wise['Observation Date'], mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=20)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 10})
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Mortality Rate', size=15)
plt.xticks(size=12)
plt.yticks(size=12)

plt.subplot(122)
plt.plot(data_date_wise['Observation Date'], recovery_rate, color='blue')
plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')
plt.title('Recovery Rate of Coronavirus Over Time', size=20)
plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 10})
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Recovery Rate', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


# ### Confirmed vs Deaths

# In[ ]:


plt.plot(data_date_wise['Confirmed'],data_date_wise['Deaths'])
#plt.scatter(data_date_wise['Confirmed'],data_date_wise['Deaths'])
plt.xlabel("Confirmed cases")
plt.ylabel("Deaths")
plt.title("Confirmed vs Deaths")
plt.show()


# ### Confirmed vs Recoveries

# In[ ]:


plt.plot(data_date_wise['Recovered'],data_date_wise['Deaths'])
#plt.scatter(data_date_wise['Confirmed'],data_date_wise['Deaths'])
plt.xlabel("Confirmed cases")
plt.ylabel("Deaths")
plt.title("Confirmed vs Recoveries")
plt.show()


# ## DateWise Analysis

# In[ ]:


date_country_wise=data.groupby(['Observation Date','Country'])[main_cols].sum().reset_index(drop=None)
date_country_wise[['Confirmed','Deaths','Recovered']]=date_country_wise[['Confirmed','Deaths','Recovered']].astype(int)
date_country_wise['Confirmed']=date_country_wise['Confirmed'].replace(np.nan, 0, regex=True)
date_country_wise['Deaths']=date_country_wise['Deaths'].replace(np.nan, 0, regex=True)
date_country_wise['Recovered']=date_country_wise['Recovered'].replace(np.nan, 0, regex=True)

#Rest of the world
rest_of_the_world=date_country_wise.loc[date_country_wise['Country']!='China']
rest_of_the_world_n=rest_of_the_world.groupby(['Observation Date'])[main_cols].sum().reset_index()
rest_of_the_world_n['daily_cases']=rest_of_the_world_n.Confirmed.diff()
rest_of_the_world_n['daily_deaths']=rest_of_the_world_n.Deaths.diff()
rest_of_the_world_n['daily_recoveries']=rest_of_the_world_n.Recovered.diff()

# China 
china=date_country_wise[date_country_wise['Country']=='China']
china_n=china.groupby(['Observation Date'])[main_cols].sum().reset_index()
china_n['daily_cases']=china_n.Confirmed.diff()
china_n['daily_deaths']=china_n.Deaths.diff()
china_n['daily_recoveries']=china_n.Recovered.diff()


# ### Comparison of Confirmed Cases between China and Rest of the World 

# In[ ]:


plt.figure(figsize=(8,6))
plt.bar('China', china_n['Confirmed'])
plt.bar('Rest of the world', rest_of_the_world_n['Confirmed'])
plt.title('No of Coronavirus Confirmed Cases', size=20)
plt.xlabel('Country', fontsize=15)
plt.ylabel('No of Case',fontsize=15)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# ### US: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


us=date_country_wise[date_country_wise['Country']=='US']
us_n=us.groupby(['Observation Date'])[main_cols].sum().reset_index()
us_n['daily_cases']=us_n.Confirmed.diff()
us_n['daily_deaths']=us_n.Deaths.diff()
us_n['daily_recoveries']=us_n.Recovered.diff()
#us_n.tail()

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(us_n.index,us_n.Confirmed,color='purple')
bar2=plt.plot(us_n.index,us_n.Active,color='orange')
bar3=plt.plot(us_n.index,us_n.Recovered,color='g')
bar4=plt.plot(us_n.index,us_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('US Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### Italy: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


italy=date_country_wise[date_country_wise['Country']=='Italy']
italy_n=italy.groupby(['Observation Date'])[main_cols].sum().reset_index()
italy_n['daily_cases']=italy_n.Confirmed.diff()
italy_n['daily_deaths']=italy_n.Deaths.diff()
italy_n['daily_recoveries']=italy_n.Recovered.diff()

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(italy_n.index,italy_n.Confirmed,color='purple')
bar2=plt.plot(italy_n.index,italy_n.Active,color='orange')
bar3=plt.plot(italy_n.index,italy_n.Recovered,color='g')
bar4=plt.plot(italy_n.index,italy_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('Italy Corona Virus Cases - Confirmed, Active, Recovered and Deaths ',fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### Spain: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


spain=date_country_wise[date_country_wise['Country']=='Spain']
spain_n=spain.groupby(['Observation Date'])[main_cols].sum().reset_index()
spain_n['daily_cases']=spain_n.Confirmed.diff()
spain_n['daily_deaths']=spain_n.Deaths.diff()
spain_n['daily_recoveries']=spain_n.Recovered.diff()
spain_n.nlargest(5,'Confirmed')

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(spain_n.index,spain_n.Confirmed,color='purple')
bar2=plt.plot(spain_n.index,spain_n.Active,color='orange')
bar3=plt.plot(spain_n.index,spain_n.Recovered,color='g')
bar4=plt.plot(spain_n.index,spain_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('Spain Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### Germany: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


germany=date_country_wise[date_country_wise['Country']=='Germany']
germany_n=germany.groupby(['Observation Date'])[main_cols].sum().reset_index()
germany_n['daily_cases']=germany_n.Confirmed.diff()
germany_n['daily_deaths']=germany_n.Deaths.diff()
germany_n['daily_recoveries']=germany_n.Recovered.diff()
germany_n.nlargest(5,'Confirmed')

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(germany_n.index,germany_n.Confirmed,color='purple')
bar2=plt.plot(germany_n.index,germany_n.Active,color='orange')
bar3=plt.plot(germany_n.index,germany_n.Recovered,color='g')
bar4=plt.plot(germany_n.index,germany_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('Germany Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### France: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


france=date_country_wise[date_country_wise['Country']=='France']
france_n=france.groupby(['Observation Date'])[main_cols].sum().reset_index()
france_n['daily_cases']=france_n.Confirmed.diff()
france_n['daily_deaths']=france_n.Deaths.diff()
france_n['daily_recoveries']=france_n.Recovered.diff()
france_n.nlargest(5,'Confirmed')

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(france_n.index,france_n.Confirmed,color='purple')
bar2=plt.plot(france_n.index,france_n.Active,color='orange')
bar3=plt.plot(france_n.index,france_n.Recovered,color='g')
bar4=plt.plot(france_n.index,france_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('France Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### UK: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


uk=date_country_wise[date_country_wise['Country']=='UK']
uk_n=uk.groupby(['Observation Date'])[main_cols].sum().reset_index()
uk_n['daily_cases']=uk_n.Confirmed.diff()
uk_n['daily_deaths']=uk_n.Deaths.diff()
uk_n['daily_recoveries']=uk_n.Recovered.diff()
uk_n.nlargest(5,'Confirmed')

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(uk_n.index,uk_n.Confirmed,color='purple')
bar2=plt.plot(uk_n.index,uk_n.Active,color='orange')
bar3=plt.plot(uk_n.index,uk_n.Recovered,color='g')
bar4=plt.plot(uk_n.index,uk_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title(' UK Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### China: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(china_n.index,china_n.Confirmed,color='purple')
bar2=plt.plot(china_n.index,china_n.Active,color='y')
bar3=plt.plot(china_n.index,china_n.Recovered,color='g')
bar4=plt.plot(china_n.index,china_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('China Corona Virus Cases - Confirmed, Active, Recovered and Deaths ',fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### India: Confirmed vs Recovered vs Deaths and Active cases

# In[ ]:


india=date_country_wise[date_country_wise['Country']=='India']
india_n=india.groupby(['Observation Date'])[main_cols].sum().reset_index()
india_n['daily_cases']=india_n.Confirmed.diff()
india_n['daily_deaths']=india_n.Deaths.diff()
india_n['daily_recoveries']=india_n.Recovered.diff()
india_n.nlargest(5,'Confirmed')

f, ax = plt.subplots(figsize=(10, 8))
bar1=plt.plot(india_n.index,india_n.Confirmed,color='Purple')
bar2=plt.plot(india_n.index,india_n.Active,color='orange')
bar3=plt.plot(india_n.index,india_n.Recovered,color='g')
bar4=plt.plot(india_n.index,india_n.Deaths,color='r')
ax.legend(['Confirmed','Active','Recovered','Deaths'],loc='best')
plt.xlabel("Since 22/1",fontsize=15)
plt.ylabel("No of Cases",fontsize=15)
plt.title('India Corona Virus Cases - Confirmed, Active, Recovered and Deaths ', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ### Daily Increases in Confirmed Cases and Deaths in Different Countries Over Time

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.plot(india_n.index,india_n['daily_cases'])
plt.plot(china_n.index,china_n['daily_cases'])
plt.plot(us_n.index,us_n['daily_cases'])
plt.plot(spain_n.index,spain_n['daily_cases'])
plt.plot(italy_n.index,italy_n['daily_cases'])
plt.plot(germany_n.index,germany_n['daily_cases'])
plt.plot(france_n.index,france_n['daily_cases'])
plt.plot(uk_n.index,uk_n['daily_cases'])
plt.title('Daily Increases in Confirmed Cases ', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['India','China','US','Spain','Italy','Germany','France','UK'])

plt.subplot(212)
plt.plot(india_n.index,india_n['daily_deaths'])
plt.plot(china_n.index,china_n['daily_deaths'])
plt.plot(us_n.index,us_n['daily_deaths'])
plt.plot(spain_n.index,spain_n['daily_deaths'])
plt.plot(italy_n.index,italy_n['daily_deaths'])
plt.plot(germany_n.index,germany_n['daily_deaths'])
plt.plot(france_n.index,france_n['daily_deaths'])
plt.plot(uk_n.index,uk_n['daily_deaths'])
plt.title('Daily Increases in Deaths Cases ', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['India','China','US','Spain','Italy','Germany','France','Uk'])
plt.show()


# ### Statstics Analysis (US vs China vs Italy vs Spain vs Germany vs France vs UK vs India)

# In[ ]:


plt.figure(figsize=(10,25))

plt.subplot(3,1,1)
plt.plot(india_n.index,india_n['Confirmed'],label='India',color='green')
plt.plot(china_n.index,china_n['Confirmed'],label='China',color='red')
plt.plot(italy_n.index,italy_n['Confirmed'],label='Italy',color='brown')
plt.plot(spain_n.index,spain_n['Confirmed'],label='Spain',color='black')
plt.plot(germany_n.index,germany_n['Confirmed'],label='Germany',color='blue')
plt.plot(france_n.index,france_n['Confirmed'],label='Germany',color='orange')
plt.plot(uk_n.index,uk_n['Confirmed'],label='Germany',color='purple')
plt.plot(us_n.index,us_n['Confirmed'],label='US',color='magenta')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.title('Comparison between Confirmed Cases',fontsize=20)
plt.legend(loc='best')
plt.xticks(rotation=45)

plt.subplot(3,1,2)
plt.plot(india_n.index,india_n['Recovered'],label='India',color='green')
plt.plot(china_n.index,china_n['Recovered'],label='China',color='red')
plt.plot(italy_n.index,italy_n['Recovered'],label='Italy',color='brown')
plt.plot(spain_n.index,spain_n['Recovered'],label='Spain',color='black')
plt.plot(germany_n.index,germany_n['Recovered'],label='Germany',color='blue')
plt.plot(france_n.index,france_n['Recovered'],label='Germany',color='orange')
plt.plot(uk_n.index,uk_n['Recovered'],label='Germany',color='purple')
plt.plot(us_n.index,us_n['Recovered'],label='US',color='magenta')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.title('Comparison between Recovered Cases',fontsize=20)
plt.legend(loc='best')
plt.xticks(rotation=45)

plt.subplot(3,1,3)
plt.plot(india_n.index,india_n['Deaths'],label='India',color='green')
plt.plot(china_n.index,china_n['Deaths'],label='China',color='red')
plt.plot(italy_n.index,italy_n['Deaths'],label='Italy',color='brown')
plt.plot(spain_n.index,spain_n['Deaths'],label='Spain',color='black')
plt.plot(germany_n.index,germany_n['Deaths'],label='Germany',color='blue')
plt.plot(france_n.index,france_n['Deaths'],label='Germany',color='orange')
plt.plot(uk_n.index,uk_n['Deaths'],label='Germany',color='purple')
plt.plot(us_n.index,us_n['Deaths'],label='US',color='magenta')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.title('Comparison between Deaths Cases',fontsize=20)
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


italy_n['Death Rate in Italy'] = ((italy_n['Deaths']/italy_n['Confirmed'])*100)
germany_n['Death Rate in Germany'] = (germany_n['Deaths']/germany_n['Confirmed'])*100
spain_n['Death Rate in Spain'] = (spain_n['Deaths']/spain_n['Confirmed'])*100
us_n['Death Rate in US'] = (us_n['Deaths']/us_n['Confirmed'])*100
france_n['Death Rate in France'] = (france_n['Deaths']/france_n['Confirmed'])*100
uk_n['Death Rate in UK'] = (uk_n['Deaths']/uk_n['Confirmed'])*100
india_n['Death Rate in India'] = (india_n['Deaths']/india_n['Confirmed'])*100
china_n['Death Rate in China'] = (china_n['Deaths']/china_n['Confirmed'])*100
rest_of_the_world_n['Death Rate in Outside China'] = (rest_of_the_world_n['Deaths']/rest_of_the_world_n['Confirmed'])*100

#Recoveries
italy_n['Recovery Rate in Italy'] = ((italy_n['Recovered']/italy_n['Confirmed'])*100)
germany_n['Recovery Rate in Germany'] = (germany_n['Recovered']/germany_n['Confirmed'])*100
spain_n['Recovery Rate in Spain'] = (spain_n['Recovered']/spain_n['Confirmed'])*100
us_n['Recovery Rate in US'] = (us_n['Recovered']/us_n['Confirmed'])*100
france_n['Recovery Rate in France'] = (france_n['Recovered']/france_n['Confirmed'])*100
uk_n['Recovery Rate in UK'] = (uk_n['Recovered']/uk_n['Confirmed'])*100
india_n['Recovery Rate in India'] = (india_n['Recovered']/india_n['Confirmed'])*100
china_n['Recovery Rate in China'] = (china_n['Recovered']/china_n['Confirmed'])*100
rest_of_the_world_n['Recovery Rate in Outside China'] = (rest_of_the_world_n['Recovered']/rest_of_the_world_n['Confirmed'])*100


# ### Death Rate in deifferent Countries Over Time

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.plot(india_n.index,india_n['Death Rate in India'])
plt.plot(china_n.index,china_n['Death Rate in China'])
plt.plot(us_n.index,us_n['Death Rate in US'])
plt.plot(spain_n.index,spain_n['Death Rate in Spain'])
plt.plot(italy_n.index,italy_n['Death Rate in Italy'])
plt.plot(germany_n.index,germany_n['Death Rate in Germany'])
plt.plot(france_n.index,france_n['Death Rate in France'])
plt.plot(uk_n.index,uk_n['Death Rate in UK'])
plt.title('Death Rate in deifferent Countries ', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['India','China','US','Spain','Italy','Germany','France','UK'])


# ### Recovery Rate in deifferent Countries Over Time

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.plot(india_n.index,india_n['Recovery Rate in India'])
plt.plot(china_n.index,china_n['Recovery Rate in China'])
plt.plot(us_n.index,us_n['Recovery Rate in US'])
plt.plot(spain_n.index,spain_n['Recovery Rate in Spain'])
plt.plot(italy_n.index,italy_n['Recovery Rate in Italy'])
plt.plot(germany_n.index,germany_n['Recovery Rate in Germany'])
plt.plot(france_n.index,france_n['Recovery Rate in France'])
plt.plot(uk_n.index,uk_n['Recovery Rate in UK'])
plt.title('Recovery Rate in deifferent Countries', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['India','China','US','Spain','Italy','Germany','France','UK'])


# ### Recovery Rate in China and Rest of the World Over Time

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.plot(rest_of_the_world_n.index,rest_of_the_world_n['Death Rate in Outside China'])
plt.plot(china_n.index,china_n['Death Rate in China'])
plt.title('Recovery Rate : Chian vs Outside China', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['Outside of China','China'])


# ### Death Rate in China and Rest of the World Over Time

# In[ ]:


plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.plot(rest_of_the_world_n.index,rest_of_the_world_n['Death Rate in Outside China'])
plt.plot(china_n.index,china_n['Death Rate in China'])
plt.title('Daeth Rate : Chian vs Outside China', size=20)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('No. of Cases', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(['Outside of China','China'])


# ### Future Forecasting using Machine Learning

# In[ ]:


date=np.array(data_date_wise["Observation Date"]).reshape(-1,1)

since_21 = np.array([i for i in range(len(date))]).reshape(-1, 1)
confirmed_world= np.array(data_date_wise["Confirmed"]).reshape(-1, 1)
deaths_world = np.array(data_date_wise["Deaths"]).reshape(-1, 1)
recovered_world = np.array(data_date_wise["Recovered"]).reshape(-1, 1)


# In[ ]:


days_in_future = 10
future_forcast = np.array([i for i in range(len(date)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

adjusted_dates=adjusted_dates.reshape(1, -1)[0]
adjusted_dates


# In[ ]:


# Creating future forcast dates
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# #### Splitting Data for Confirmed Cases

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(since_21, confirmed_world,test_size=0.2, random_state=0) 


# ### Polynomial Regression

# In[ ]:


# Transform our data for polynomial regression
poly_reg=PolynomialFeatures(degree=4)
poly_X_train = poly_reg.fit_transform(X_train)
poly_X_test = poly_reg.fit_transform(X_test)
poly_future_forcast = poly_reg.fit_transform(future_forcast)


# In[ ]:


# polynomial regression

linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train, y_train)
test_linear_pred = linear_model.predict(poly_X_test)
linear_pred = linear_model.predict(poly_future_forcast)


# In[ ]:


print('MAE:', mean_absolute_error(test_linear_pred, y_test))
print('MSE:',mean_squared_error(test_linear_pred, y_test))
print('R2 Score :',r2_score(y_test,test_linear_pred ))


# #### Future Prediction for Confirmed Cases

# In[ ]:


linear_pred = linear_pred.reshape(1,-1)[0]
print('Polynomial Regression future predictions:')
set(zip(future_forcast_dates[-10:], np.round(linear_pred[-10:])))


# In[ ]:


plt.figure(figsize=(8, 8))

plt.plot(adjusted_dates,confirmed_world)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='green')
plt.title('Coronavirus Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('No of Cases', size=20)
plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 20})
plt.xticks(size=12)
plt.yticks(size=12)

plt.show()


# #### Spliting Data for Deaths

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(since_21, deaths_world,test_size=0.2, random_state=0)


# ### Polynomial Regression

# In[ ]:


# Transform our data for polynomial regression
poly_reg=PolynomialFeatures(degree=4)
poly_X_train = poly_reg.fit_transform(X_train)
poly_X_test = poly_reg.fit_transform(X_test)
poly_future_forcast = poly_reg.fit_transform(future_forcast)


# In[ ]:


# polynomial regression

linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train, y_train)
test_linear_pred = linear_model.predict(poly_X_test)
linear_pred = linear_model.predict(poly_future_forcast)


# In[ ]:


print('MAE:', mean_absolute_error(test_linear_pred, y_test))
print('MSE:',mean_squared_error(test_linear_pred, y_test))
print('R2 Score :',r2_score(y_test,test_linear_pred ))


# #### Future prediction for Deaths

# In[ ]:


linear_pred = linear_pred.reshape(1,-1)[0]
print('Polynomial Regression future predictions:')
set(zip(future_forcast_dates[-10:], np.round(linear_pred[-10:])))


# In[ ]:


plt.figure(figsize=(8, 8))

plt.plot(adjusted_dates,deaths_world)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='green')
plt.title('Coronavirus Deaths Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('No of Cases', size=20)
plt.legend(['Deaths Cases', 'Polynomial Regression Predictions'], prop={'size': 20})
plt.xticks(size=12)
plt.yticks(size=12)

plt.show()


# ### Map Visualization

# In[ ]:


# Confirmed cases
conf= confirmed_data.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed")
conf['Date'] = pd.to_datetime(conf['Date']).dt.date
conf.rename(columns={'Country/Region':'Country'},inplace=True)

# Recovered Cases
recover= recovered_data.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Recovered")
recover['Date'] = pd.to_datetime(recover['Date']).dt.date
recover.rename(columns={'Country/Region':'Country'},inplace=True)

# Deaths
death= deaths_data.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Deaths")
death['Date'] = pd.to_datetime(death['Date']).dt.date
death.rename(columns={'Country/Region':'Country'},inplace=True)

covid=conf.merge(death).merge(recover)
#print("Size/Shape of the dataset: ",covid.shape)
#print("Checking for null values:\n",covid.isnull().sum())
#print("Checking Data-type of each column:\n",covid.dtypes)


# ### Map Visualization : Confirmed Cases

# In[ ]:


world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(covid['Lat'], covid['Long'], covid['Confirmed'], covid['Country']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
world_map


# ### Map Visualization : Death Cases

# In[ ]:


world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(covid['Lat'], covid['Long'], covid['Deaths'], covid['Country']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Deaths Cases</strong>: ' + str(value) + '<br>'),
                        color='green',
                        
                        fill_color='green',
                        fill_opacity=0.3 ).add_to(world_map)
world_map


# ### Map Visualization : Recovered Cases

# In[ ]:


world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(covid['Lat'], covid['Long'], covid['Recovered'], covid['Country']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Recovered Cases</strong>: ' + str(value) + '<br>'),
                        color='blue',
                        
                        fill_color='blue',
                        fill_opacity=0.5 ).add_to(world_map)
world_map


# **Conclusion:**
# 
# **A large number of infected people in China have recovered and as of now China as very few active cases. COVID-19 doesn't have very high mortality rate as we can see which the most positive take away. Also the healthily growing Recovery Rate implies the disease is curable. The only matter of concern is the exponential growth rate of infection.**
# 
# **Countries like US, Italy, Spain, Germany,France and UK are facing some serious trouble in containing the disease showing how deadly the negligence can lead to. The need of the hour is to perform COVID-19 pandemic controlling practices like Testing, Contact Tracing and Quarantine with a speed greater than the speed of disease spread at each country level.**
# 
# 
# **Prevention:**
# 
# **To avoid the critical situation people are suggested to do following things**
# 
# **Avoid contact with people who are sick. Avoid touching your eyes, nose, and mouth. Stay home when you are sick. Cover your cough or sneeze with a tissue, then throw the tissue in the trash. Clean and disinfect frequently touched objects and surfaces using a regular household Wash your hands often with soap and water, especially after going to the bathroom; before eating; and after blowing your nose, coughing, or sneezing. If soap and water are not readily available, use an alcohol-based hand sanitizer.**

# In[ ]:




