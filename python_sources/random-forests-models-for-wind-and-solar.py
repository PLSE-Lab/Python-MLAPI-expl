#!/usr/bin/env python
# coding: utf-8

# In[725]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
solar= pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRES_PVGIS_TSh_CF_n2_19862015.csv")
solar_Country= pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRESPV_TSh_CF_Country_19862015.csv")
wind=pd.read_csv("../input/30-years-of-european-wind-generation/TS.CF.N2.30yr.csv")
wind_Country=pd.read_csv("../input/30-years-of-european-wind-generation/EMHIRESPV_TSh_CF_Country_19862015.csv")
# Any results you write to the current directory are saved as output.


# ### Solar and Wind Hourly Power Capacity
# The datasets contain hourly power generation data for NUTS 2 regions in Europe. The objective is to develop an algorithm for the prediction of the power generation using factors like economic status, renewable energy interest and the non-renewable power generation. 

# In[726]:


solar.head()
t=pd.DataFrame(solar.sum())
get_ipython().run_line_magic('matplotlib', 'inline')


# Subsets the data for the last 15 years instead of all 30 years.
# Column names are the countries and the rows are the production per hour.

# In[727]:


print(solar_Country.shape)
solar_Country= solar_Country[solar_Country.index<len(solar_Country.index)-(15*24*365)]
print(solar_Country.shape)
solar_Country.head()


# This dataset  is by the individual stations and their production per hour.

# In[728]:


print(solar.shape)
solar= solar[solar.index<len(solar.index)-(15*24*365)]
print(solar.shape)
solar.head()


# The following dataset is for the last 15 years of wind energy by station.

# In[729]:


print(wind.shape)
wind= wind[wind.index<len(wind.index)-(15*24*365)]
print(wind.shape)
wind.head()


# ## Mean and Maximum Power Generation Capacity

# The following table shows the mean and the max power based on the 15 years of production by the countries in Europe. The data for Cyprus is missing. 

# In[730]:


Stats= pd.DataFrame()
Stats['country']=solar_Country.columns
ref=pd.DataFrame(solar_Country.mean())
ref=ref.round(28)
Stats['mean']=ref[0].unique()
ref=pd.DataFrame(solar_Country.max())
ref=ref.round(28)
Stats['max']=ref[0].unique()
Stats.sort_values('mean', ascending=False)


# The following scatter shows the countries in terms of their mean and maximum power capacity. As seen, Cyprus has not generated any renewable power for the last 15 years. On the other hand PT has generated the greatest mean power over the last 15 years. 

# In[731]:


sns.lmplot('max','mean',data=Stats,fit_reg=False,hue='country',size=8)
plt.show()


# The following figure displays the mean and max distribution of the power production. As it can be seen, the maximum power capacity per hour distribution is further right to the mean capacity being used.

# In[732]:


fig=plt.figure(figsize=(15,5))
ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3= fig.add_subplot(1,3,3)
ax1.set_title("Mean  Capacity Distribution")
ax2.set_title("Maximum Capacity Distribution")
ax1.hist(Stats['mean'],color='lightblue')
ax2.hist(Stats['max'], color='lightblue')
ax3.hist(Stats['mean'])
ax3.hist(Stats['max'])
#ax3.hist()


# The same distribution is seen in the boxplots. In both boxplots, Cyprus shows up as an outlier under the distribution. The distribution for the maximum has greater range than the mean distribution. The mean power produced per hours is localized to a small range for the countries in Europe. 

# In[733]:



fig=plt.figure(figsize=(10,5),edgecolor='orange')
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2, sharey=ax1)
ax1.boxplot(Stats['mean'],patch_artist=True)
ax2.boxplot(Stats['max'],patch_artist=True,vert=True)
plt.tight_layout()


# In[734]:


wind_by_mean=wind.describe().T.sort_values('mean',ascending= False)
solar_by_mean=solar.describe().T.sort_values('mean',ascending=False)


# ### Mean Power Distribution for Wind and Solar

# The following histogram shows the distribution of the mean wind and mean solar capacity. The wind capacity is maximum is greater than the solar capacity.

# In[735]:


wind_by_mean=wind_by_mean.dropna()
solar_by_mean=solar_by_mean.dropna()
fig, axes= plt.subplots(1,2,figsize=(15,5))
axes[0].set_title("Mean Wind Capacity")
axes[0].hist(wind_by_mean['mean'], color='lightblue')
axes[1].set_title("Mean Solar Capacity")
axes[1].hist(solar_by_mean['mean'][1:], color='lightblue')


# In[736]:


print("Number of years of solar observations:",round(solar['time_step'][solar.shape[0]-1]/(24*365),1))
print("Number of years of wind observations:",round(wind.index[wind.shape[0]-1]/(365*24),1))


# In[737]:


#solar=solar[solar['time_step']<24*365]


# In[738]:


Inactive_hours=[]
for item in solar:
    t=solar[solar[item]==0.0]
    Inactive_hours.append(len(t))


# In[739]:


Inactive_hoursW=[]
for item in wind:
    t=wind[wind[item]!=0.0]
    Inactive_hoursW.append(len(t))


# The following table shows the number of hours and years that the Wind stations in Europe were active.

# In[740]:


Hours_NonZeroWind=pd.DataFrame(Inactive_hoursW[1:])
Hours_NonZeroWind['Years']=Hours_NonZeroWind[0]/(24*365)
Hours_NonZeroWind['Stations']=wind.columns[1:]
Hours_NonZeroWind=Hours_NonZeroWind.sort_values(0)
Hours_NonZeroWind.describe()


# The station that produced the least was AT21.
# The station that produced the most was BG32.

# In[741]:


print(Hours_NonZeroWind[Hours_NonZeroWind['Years']==Hours_NonZeroWind['Years'].min()])
print(Hours_NonZeroWind[Hours_NonZeroWind['Years']==Hours_NonZeroWind['Years'].max()])
Active_hours=[]
for item in solar:
    t=solar[solar[item]!=0.0]
    Active_hours.append(t.shape[0])
Hours_NonZero=pd.DataFrame(Active_hours[1:])
Hours_NonZero['Years']=Hours_NonZero[0]/(24*365)
Hours_NonZero['Stations']=solar.columns[1:]
Hours_NonZero=Hours_NonZero.sort_values(0)


# As seen in the table below, the stations have been predominantly more active between the levels of 12.788 and 15.019. 

# In[742]:


fig, axes= plt.subplots(1,2,figsize=(15,5))
#Hours_NonZeroWind=Hours_NonZeroWind[Hours_NonZeroWind['Years']!=Hours_NonZeroWind['Years'].min()]
Hours_NonZeroWind['YearDist']=pd.cut(Hours_NonZeroWind['Years'],5)
c=Hours_NonZeroWind.groupby('YearDist').count()

print("Wind Distribution")
sns.barplot(c.index,c['Stations'], ax=axes[0])
print(c)
Hours_NonZero=Hours_NonZero[Hours_NonZero['Years']!=Hours_NonZero['Years'].min()]
Hours_NonZero['YearDist']=pd.cut(Hours_NonZero['Years'],3)
c=Hours_NonZero.groupby('YearDist').count()

print("Solar Distribution")
print(c)


axes[0].set_title('Wind')
axes[1].set_title('Solar')
sns.barplot(c.index,c['Stations'], ax=axes[1])


# Compared to the Wind maximum of 15 years, the solar maximum is only approximately 7 years.

# In[743]:


Hours_NonZero.describe()


# In[744]:


Hours_NonZero.sort_values('Stations',ascending=False)[:10]


# The station that produced the least was SE33.
# The station that produced the most was ES61.

# In[745]:


print(Hours_NonZero[Hours_NonZero['Years']==Hours_NonZero['Years'].min()])
print(Hours_NonZero[Hours_NonZero['Years']==Hours_NonZero['Years'].max()])


# In[746]:


item3=""
Power=[]

solar[solar.columns[1]].sum()+solar[solar.columns[2]].sum()
for item in range(1,261):
    if(item3!=solar.columns[item][0:2]):
        Power.append(item)
        item3= solar.columns[item][0:2]
count=0
num_stations_solar=[]
for item in Power[1:]:
    num_stations_solar.append(item-count)
    count=item
num_stations_solar.append(261-224)
Power.append(260)
power_amount=[]
solar_mean_year=[]
start=1
for item in Power:
    end= item
    count=0
    for item in range(start, end):
        count=count+solar[solar.columns[item]].mean()
    start=item
    power_amount.append(count)
len(power_amount)
power_amount=power_amount[1:]
power_amount=power_amount[:21]
name=[]
solar_year=[]
num=0
for x in range(0, len(Power)-1):
    count=0
    for item2 in range(24*365, len(solar), 8771):
        if(num==131554):
            num=0
        s=solar.loc[num:item2]
        num=item2
        for item in (s.columns[Power[x]:Power[x+1]]):
            count=count+s[item].sum()
        name.append(s.columns[Power[x]])
        solar_year.append(count)


# In[747]:


import numpy as np
solar_mean_byYear= pd.DataFrame()
solar_mean_byYear['CountryCode']= name
solar_mean_byYear['Solar_Power']=solar_year


# This shows the solar stations per country and the power for that country.

# In[748]:


Power=Power[:21]
num_stations_df= pd.DataFrame({ 'Country':solar.columns[Power],'Number_of_Stations':num_stations_solar,'Power_per_Country':power_amount})
CountryNamesSolar=['Austria','Belgium','Bulgaria','Czech Republic','Germany','Spain','Finland','France','Greece','Hungary','Switzerland','Ireland','Italy','Netherlands','Norway','Poland','Portugal','Romania','Sweden','Slovakia','United Kingdom']
solar_mean_byYear['Country or Area']= np.repeat(CountryNamesSolar, 15)
num_stations_df['Regions']=CountryNamesSolar


# The following table shows the number of stations  per region as well as the mean power for the past 15 years.

# In[749]:


num_stations_df.iloc[:,1:]


# In[750]:


windT=wind.sum().T
windT=windT.reindex(sorted(wind.columns))
#windT
item3=""
Power=[]
for item in range(len(windT.index)):
    if(item3!=windT.index[item][0:2]):
        Power.append(item)
        item3= windT.index[item][0:2]
count=0
num_stations_wind=[]
for item in Power[1:]:
    num_stations_wind.append(item-count)
    count=item
num_stations_wind.append(255-221)
Power.append(254)
power_amount=[]
start=1
for item in Power:
    end= item
    count=0
    for item in range(start, end):
        count=count+wind[windT.index[item]].mean()
    start=item
    power_amount.append(count)
len(power_amount)
power_amount=power_amount[1:]
#power_amount=power_amount[:21]


# In[751]:


wind=wind[sorted(wind.columns)]
wind.head()
name=[]
wind_year=[]
num=0
for x in range(0, len(Power)-1):
    count=0
    for item2 in range(24*365, len(wind), 8771):
        if(num==131554):
            num=0
        s=wind.loc[num:item2]
        num=item2
        for item in (s.columns[Power[x]:Power[x+1]]):
            count=count+s[item].sum()
        name.append(s.columns[Power[x]])
        wind_year.append(count)


# In[752]:


wind_mean_byYear= pd.DataFrame()
wind_mean_byYear['CountryCode']= name
wind_mean_byYear['Wind_Power']=wind_year


# In[753]:


Power=Power[:24]
num_Wstations_df= pd.DataFrame({ 'Country':windT.index[Power],'Number_of_Stations':num_stations_wind,'Power_per_Country':power_amount})
#print(num_Wstations_df)


# The following dataset displays the number of stations and mean power produced for European countries. 24 countries have wind capacity whereas only 21 countries solar capacity data is available as seen in the table above. This data was missing in the original dataset. The regions listed are aggregates of the NUTS 2 regional levels with the same code.

# In[754]:


CountryNamesWind=['Austria','Belgium','Bulgaria','Switzerland','Czech Republic','Germany','Denmark','Spain','Finland','France','Greece','Croatia','Hungary','Ireland','Italy','Netherlands','Norway','Poland','Portugal','Romania','Sweden','Slovenia','Slovakia','United Kingdom']
num_Wstations_df['Regions']=CountryNamesWind
wind_mean_byYear['Country or Area']=np.repeat(CountryNamesWind, 15)
num_Wstations_df


# In[755]:


n1=(num_stations_df['Regions'])
n2=(num_Wstations_df['Regions'])
unique_countries=[]
for item in n2:
    if item in n1:
        unique_countries.append(item)
    else: unique_countries.append(item)


# In[756]:


combustible= pd.read_csv("../input/un-data/UNdata_Export_20180503_190432666.csv")
hydro= pd.read_csv("../input/un-data/UNdata_Export_20180503_191509360.csv")
economic=pd.read_csv("../input/un-data/UNdata_Export_20180503_192018346.csv")


# In[757]:


print(combustible.shape)
combustible.head()
CountriesDiscard=[]
for item in combustible['Country or Area'].unique():
    if item in unique_countries:
        print ("Hoorah")
    else: CountriesDiscard.append(item)


# In[758]:


E_combustible= combustible
for item in CountriesDiscard:
    E_combustible= E_combustible[E_combustible['Country or Area']!=item]
E_hydro= hydro
E_economic= economic


# In[759]:


for item in hydro['Country or Area'].unique():
    c=0
    if item in unique_countries:
        c=c+1
    else: CountriesDiscard.append(item)
for item in CountriesDiscard:
    E_hydro= E_hydro[E_hydro['Country or Area']!=item]
for item in economic['Country or Area'].unique():
    c=0
    if item in unique_countries:
        c=c+1
    else: CountriesDiscard.append(item)
for item in CountriesDiscard:
    E_economic= E_economic[E_economic['Country or Area']!=item]


# In[760]:


E_combustible.head()


# In[761]:


E_hydro.head()


# In[762]:


E_economic.head()


# In[763]:


E_combustible= E_combustible[E_combustible['Year']>2000]
E_combustible['Year'].unique()
E_hydro['Year']=pd.to_numeric(E_hydro['Year'])
E_hydro= E_hydro[E_hydro['Year']>2000]
E_hydro['Year'].unique()
E_economic['Year']=pd.to_numeric(E_economic['Year'])
E_economic= E_economic[E_economic['Year']>2000]
E_economic['Year'].unique()


# The following bar graph show the distribution of the GNI per capita of the European countries. For the analysis of the solar capacity and wind capacity as predictors of each other, the other factors of interest are the GNI per capita, Hydroelectric capacity and Electricity produced from Combustion.

# In[764]:


plt.subplots(figsize=(10,10))
sns.barplot(y=E_economic['Country or Area'], x=E_economic['Value'])
E_economic_byYear=E_economic.groupby(["Country or Area", "Year"]).mean()


# The plot shows the distribution of hydroelectic power (MW) produced by each of the countries. Norway uses the greatest amount of electricity produced from hydropower. 

# In[765]:


plt.subplots(figsize=(10,10))
sns.barplot(y=E_hydro['Country or Area'],x= E_hydro['Quantity'])
E_hydro_byYear=E_hydro.groupby(["Country or Area", "Year"]).mean()


# The mean combustible energy produced by country.

# In[766]:


plt.subplots(figsize=(10,10))
sns.barplot(y=E_combustible['Country or Area'], x=E_combustible['Quantity'])
E_combustible_byYear=E_combustible.groupby(["Country or Area", "Year"]).mean()


# In[767]:


#g=sns.FacetGrid(E_hydro, row='Country or Area')
#g.map(plt.scatter, "Year","Quantity" )


# In[768]:


Final_combustible=E_combustible.groupby('Country or Area').mean()
Final_combustible['Regions']=Final_combustible.index
Final_hydro= E_hydro.groupby('Country or Area').mean()
Final_hydro['Regions']= Final_hydro.index
Final_economic= E_economic.groupby('Country or Area').mean()
Final_economic['Regions']=Final_economic.index
Final_hydro['Quantity'].nunique()


# In[769]:


final_df=pd.DataFrame()
final_df['Regions']= sorted(unique_countries)
#final_df['hydro']= Final_hydro['Quantity']
#final_df['combustible']= Final_combustible['Quantity'].unique()
#final_df['economic']=Final_economic['Value'].unique()
j1=pd.merge(final_df,Final_combustible, how='left', on=['Regions'])
j2=pd.merge(j1,Final_hydro, how='left', on=['Regions'])
j3=pd.merge(j2, Final_economic, how='left', on=['Regions'])
j4= pd.merge(j3, num_stations_df, how='left', on=['Regions'])
j5= pd.merge(j4, num_Wstations_df, how='left', on=['Regions'])
#j5
j5=j5[['Regions','Quantity_x','Quantity_y','Value','Number_of_Stations_x','Power_per_Country_x','Number_of_Stations_y','Power_per_Country_y']]
j5['division']=0
import random
for item in range(len(j5)):
    j5['division'][item]=random.randint(0,1)
#j5['Power_per_Country_x']=pd.to_numeric(j5['Power_per_Country_x'])
#j5[j5.isnull()==True]
j5.fillna(0, inplace=True)
j5['Wind_Division']=pd.cut(j5['Power_per_Country_y'],2)
j5['Solar_Division']=pd.cut(j5['Power_per_Country_x'],2)
j5['Wind_Class']='Lower'
j5['Solar_Class']='Lower'
j5['Wind_Class'][j5['Power_per_Country_y']>.19]='Greater'
j5['Solar_Class'][j5['Power_per_Country_x']>.10]="Greater"
j5.columns=['Regions','Combustible','HydroElectric','Economic_measure','Number_of_Solar_stations','Solar_generated_Power','Number_of_wind_Stations','Wind_Generated_Power','division','Wind_cut','Solar_cut','Wind_Class','Solar_Class']


# In[770]:


j5.head()


# ## Random Forest: Predictors for Solar and Wind (15 Years)

# ## Wind Power Analysis

# In[771]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_jobs=2, random_state=1)
j6=j5[j5['division']==0]
features=j6[j6.columns[1:6]]
clf.fit(features, j6[j6.columns[11]])
clf.predict(features)


# ### Observed Values vs. Predicted Values

# In[772]:


pd.crosstab(j5['Wind_Class'], clf.predict(j5[j5.columns[1:6]]))


# ### Importance of Predictors

# In[773]:


#clf.predict_proba(features)
list(zip(features,clf.feature_importances_))


# ## Solar Power Analysis

# In[774]:


clf= RandomForestClassifier(n_jobs=2, random_state=1)
features=j5[j5.columns[[1,2,3,7]]]
clf.fit(features, j5[j5.columns[12]])
clf.predict(features)


# ### Observed Values vs. Predicted Values

# In[775]:


pd.crosstab(j5['Solar_Class'], clf.predict(j5[j5.columns[[1,2,3,7]]]))


# ### Importance of Predictors

# In[776]:


list(zip(features,clf.feature_importances_))


# In[777]:


final_df_byYear=pd.DataFrame()
import numpy as np
final_df_byYear['Country or Area']= np.repeat(unique_countries, 15)
final_df_byYear['Year']=sorted(E_combustible['Year'].unique())*len(unique_countries)
E_combustible_byYear.reset_index(inplace=True)
E_hydro_byYear.reset_index(inplace=True)
E_economic_byYear.reset_index(inplace=True)


# ## Description Table by Year

# In[778]:


j1=pd.merge(final_df_byYear,E_combustible_byYear, how='left', on=['Country or Area','Year'])
j2=pd.merge(j1,E_hydro_byYear, how='left', on=['Country or Area','Year'])
j3=pd.merge(j2,E_economic_byYear, how='left', on=['Country or Area','Year'])
j3= j3[['Country or Area','Year','Quantity_x','Quantity_y','Value']]
j3
#j4= pd.merge(j3, num_stations_df, how='left', on=['Regions'])
#j5= pd.merge(j4, num_Wstations_df, how='left', on=['Regions'])
solar_mean_byYear['Year']=sorted(E_combustible['Year'].unique())*solar_mean_byYear['Country or Area'].nunique()
wind_mean_byYear['Year']=sorted(E_combustible['Year'].unique())* wind_mean_byYear['Country or Area'].nunique()
j4=pd.merge(j3,solar_mean_byYear, how='left', on=['Country or Area','Year'])
j5=pd.merge(j4,wind_mean_byYear, how='left', on=['Country or Area','Year'])
j5.columns=['Country or Area','Year','Combustible','Hydro','Economic','x','Solar_power','y','Wind_power']
j5['Solar_power']=j5['Solar_power']
j5['Wind_power']=j5['Wind_power']
j5.describe()


# In[779]:


fig, axes= plt.subplots(1,2,figsize=(20,5),sharey='row')
sns.barplot(j5['Year'],j5['Solar_power'], ax=axes[0])
sns.barplot(j5['Year'],j5['Wind_power'], ax=axes[1])


# In[ ]:





# In[780]:


print(j5.describe().index)
MOG=j5.describe()
initial=[7,6,5,4]
MOG.iloc[1]
j5['factorial_solar']=7
j5['factorial_wind']=7
for item in initial:
    j5['factorial_solar'][j5['Solar_power']<=MOG.iloc[item]['Solar_power']]=item
    j5['factorial_wind'][j5['Wind_power']<=MOG.iloc[item]['Wind_power']]=item


# In[781]:


j5.head()


# ## Random Forest: Year Inclusive

# In[782]:


clf= RandomForestClassifier(n_jobs=2, random_state=1)
j5.fillna(0, inplace=True)
features=j5[j5.columns[[2,3,4,6]]]
clf.fit(features, j5[j5.columns[10]])
clf.predict(features)


# ### Observed Values vs. Predicted

# In[783]:


print("4 is 0% to 25%")
print("5 is 25% to 50%")
print("6 is 50% to 75%")
print("7 is 75% to 100%")
pd.crosstab(j5['factorial_wind'], clf.predict(j5[j5.columns[[2,3,4,6]]]))


# ### Predictor Importance

# In[784]:


list(zip(features,clf.feature_importances_))


# ## Solar Analysis

# In[785]:


clf= RandomForestClassifier(n_jobs=2, random_state=1)
j5.fillna(0, inplace=True)
features=j5[j5.columns[[2,3,4,8]]]
clf.fit(features, j5[j5.columns[9]])
clf.predict(features)


# ### Observed Values vs. Predicted

# In[786]:


print("4 is 0% to 25%")
print("5 is 25% to 50%")
print("6 is 50% to 75%")
print("7 is 75% to 100%")
pd.crosstab(j5['factorial_solar'], clf.predict(j5[j5.columns[[2,3,4,6]]]))


# The model does not accurately predict the solar capacity of the countries. 

# ### Predictor Importance

# In[787]:


list(zip(features,clf.feature_importances_))


# The wind is better predicted by the models both as a snapshot and by year. However, solar power prediction has other porssible factors of influence. The snapshot is a little more accurate..

# In[788]:


fig, axes= plt.subplots(1,2,figsize=(20,7),sharey='row')
sns.swarmplot(x='Year',y='Solar_power',data=j5,ax=axes[0])
axes[0].set_title('Solar By Year')
axes[1].set_title('Wind By Year')
sns.swarmplot(x='Year',y='Wind_power',data=j5, ax=axes[1])


# In[789]:


fig, ax=plt.subplots(1,1,figsize=(30,10))
sns.pointplot(x='Country or Area',y='Solar_power',data=j5,color='green')
sns.pointplot(x='Country or Area',y='Wind_power',data=j5, color='blue')
ax.set_title('Solar and Wind By Country')


# In[790]:


fig, ax= plt.subplots(1,1,figsize=(20,5))
sns.pointplot(x='factorial_solar',y='Solar_power',data=j5)
sns.pointplot(x='factorial_wind', y='Wind_power',data=j5)


# In[ ]:




