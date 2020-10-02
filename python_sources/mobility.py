#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import DataFrame 
import numpy as np


# In this analysis, we have chosen 4 countries that seem to have handled the pandemic well, and 4 other countries that are still unable to see a dip in the number of cases.
# The 4 countries doing well are - Finland, Australia, New Zealand and Taiwan.
# These countries have not only observed very few confirmed cases, but have also been able to contain the virus and stop the spread.
# The 4 countries that are in a worse off position are- USA, Italy, Spain and France
# These countries not only have a rising number of cases, but also rising deaths. 
# From previous analysis, we see that social distancing does have a role in stopping the spread. We decided to analyse the data from google mobility reports, which uses the location history of users across the werld to track their movement across different categories of places such as retail and recreation, groceries and pharmacies, parks, transit stations, workplaces, and residential.
# We try to see if there is a relation between the ability of a country to have handled the pandemic well, and the mobility of its people.
# 

# We first import the dataset and store it in a dataframe. 
# We then drop empty columns
# We then query data from the specific countries we want to analyse.

# In[ ]:


mobility=DataFrame(pd.read_csv('../input/globalmobilityreports/GlobalMobilityReports.csv'))
mobility.head()
mobility.drop(['sub_region_1','sub_region_2'],axis=1)
finland=mobility[mobility['country_region_code']=='FI']
finland=finland.reset_index().drop('index',axis=1)
# print(finland)
NewZ=mobility[mobility['country_region']=='New Zealand']
NewZ=NewZ.reset_index().drop('index',axis=1)
# print(NewZ)
Aus=mobility[mobility['country_region']=='Australia']
Aus=Aus.reset_index().drop('index',axis=1)
# print(Aus)
Tai=mobility[mobility['country_region']=='Taiwan']
Tai=Tai.reset_index().drop('index',axis=1)
# print(Tai)
lowest=['Finland','New Zealand','Australia','Taiwan']
highest=['USA','Italy','Spain','China']

us=mobility[mobility['country_region']=='United States']
us=us.reset_index().drop('index',axis=1)
# print(us)

italy=mobility[mobility['country_region']=='Italy']
italy=italy.reset_index().drop('index',axis=1)
# print(italy)

spain=mobility[mobility['country_region']=='Spain']
spain=spain.reset_index().drop('index',axis=1)
# print(spain)

fran=mobility[mobility['country_region']=='France']
fran=fran.reset_index().drop('index',axis=1)
# print(fran)


# We now gather dates from the dates column for which these mobility observations have been made for each country in this analysis. We store the dates in a list.

# In[ ]:


# print(mobility['date'])


findates=list(finland['date'])

newzdates=list(NewZ['date'])

ausdates=list(Aus['date'])

taidates=list(Tai['date'])

usdates=list(us['date'])

itdates=list(italy['date'])

spdates=list(spain['date'])

frdates=list(fran['date'])


# We then gather data from retail and recreation mobility and store it in a list. We then plot this mobility versus time, i.e dates. 

# In[ ]:


#Retail and Recreation
from matplotlib import pyplot as plt
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['retail_and_recreation_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['retail_and_recreation_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['retail_and_recreation_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['retail_and_recreation_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('retail_and_recreation_percent_change_from_baseline')
plt.title('Retail mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['retail_and_recreation_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['retail_and_recreation_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['retail_and_recreation_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['retail_and_recreation_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('retail_and_recreation_percent_change_from_baseline')
plt.title('Retail and recreation mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# From these graphs, we can infer the following:
# 1. retail and recreation mobility-
# Countries least affected seem to have maintained a low mobility of 20 percent from the baseline since the beginning of february.
# There seems to be a steady decline and the mobility towards these places has reduced to almost nil recently. All these countries seem to follow a similar curve.
# The countries worst affected have high mobilities to these places from the beginning, and dont seem to reduce much. Though some cases the mobility has reduced by 50 percent, it is gradual and there still seems to be 100 percent mobility above the baseline, even in recent times.
# We can see that France, Italy and Spain have reduced their mobility recently by almost 200 percent, but the US still seems to have higher mobility, which explains the  rapidly increasing number of cases till date.

# Gathering grocery and pharmacy mobility and plotting graphs

# In[ ]:


# grocery_and_pharmacy_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['grocery_and_pharmacy_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['grocery_and_pharmacy_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['grocery_and_pharmacy_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['grocery_and_pharmacy_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('grocery_and_pharmacy_percent_change_from_baseline')
plt.title('Grocery and Pharmacy mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['grocery_and_pharmacy_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['grocery_and_pharmacy_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['grocery_and_pharmacy_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['grocery_and_pharmacy_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('grocery_and_pharmacy_percent_change_from_baseline')
plt.title('Grocery and Pharmacy mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# 2. grocery and pharmacy mobility
# 
# There is a low mobility in all the least affected countries right from the beginning. There is a slight peak in between, for New Zealand and australia, which might be explained by people deciding to buy essentials before the complete lockdown phase is initiated. It then remains quite low, with people probably going only if necessary. New Zealand sees a huge dip in mobility almost by 80 percent from baseline, which might be explained by the government mandating home deliveries of groceries and medicines, if required, after implementing of phase 4- complete lockdown.
# AMong the badly affected countries, while Spain, France and Italy, do see a slight dip in mobility, people in the US still seem to have widely varying mobilities, with most of it 50 percent above the baseline.
# 

# Gathering parks mobility and plotting graphs

# In[ ]:


# parks_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['parks_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['parks_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['parks_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['parks_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('parks_percent_change_from_baseline')
plt.title('Parks mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['parks_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['parks_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['parks_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['parks_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('parks_percent_change_from_baseline')
plt.title('Park mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# 3. parks mobility-
# All of the least affected countries seem to have consistently maintained low and reduced mobility to parks, except Finland. This has a rise in mobility,probably due to people wanting to go outside, yet maintain distance from other people. This probably did not affect the cases as much due to the large number of urban parks that can still function as spaces for people to visit, while adhering to social distancing norms. 
# 
# The US continues to see people going to parks more and more, probably as staying at home all day may add to wanting to go outdoors. Without strict lockdown measures, large numbers of people are able to move around and this number is extremely high. The other countries see a reduction in mobility to parks.

# Gathering transit stations mobility and plotting graphs

# In[ ]:


# transit_stations_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['transit_stations_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['transit_stations_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['transit_stations_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['transit_stations_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('transit_stations_percent_change_from_baseline')
plt.title('Transit stations mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['transit_stations_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['transit_stations_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['transit_stations_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['transit_stations_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('transit_stations_percent_change_from_baseline')
plt.title('Transit Stations mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# 4. transit stations mobility
# The countries least affected implemented travel bans almost immediately, sometimes even before the first cases were reported. They also implemented intra country transit bans unless absolutely necessary. New Zealand allows transport only after testing and making sure the person does not test positive, has had no contact with other positively tested individuals, and no recent travel history.
# 
# These countries do see a reduction in transit, except in the US again. We see France, Italy and Spain have also stopped people from traveling to different places. But the US still has steady 50 percent above the baseline transit mobility.

# Gathering workplace mobility and plotting graphs

# In[ ]:


# workplaces_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['workplaces_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['workplaces_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['workplaces_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['workplaces_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('workplaces_percent_change_from_baseline')
plt.title('Workplace mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['workplaces_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['workplaces_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['workplaces_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['workplaces_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('workplaces_percent_change_from_baseline')
plt.title('Workplace mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# 5. workplace mobility
# While Finland seems to have reduced workplace mobility right from the start, the other countries seem to have reduced it gradually. But the mobility is still low and only about 20 percent above the baseline in the initial stages as well.
# 
# All the worst hit countries, do not seemt to reduce mobility until mid march to the end of march. This is when the cases increased drastically in all these countries. While France, Italy and SPain have reduced workplace mobility by 60 percent from the baseline, the US still seems to show very little change.

# Gathering residential mobility and plotting graphs

# In[ ]:


# residential_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['residential_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['residential_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['residential_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['residential_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('residential_percent_change_from_baseline')
plt.title('Residential mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['residential_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['residential_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['residential_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['residential_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('residential_percent_change_from_baseline')
plt.title('Residential mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()

6. residential mobility
Residential mobility has increased over time for all countries with people being encouraged to stay at home.

# From all this, we can conclude, that social distancing is the key factor that determines the ability for a country to reduce the number of Covid-19 cases. In the countries least affected, there has been a rapid change in the behaviour of people with mobility reducing by 90 percent over all categories. It is important to acrue from this visualisation that countries which had initially mandated social distancing and followed through with strict implementation, have the least number of cases and a nearly flatlining growth rate of cases. The worst affected countries seem to have had late and relaxed implementation, which helps form a strong correlation to number of cases in the country. Overall, the reduction of mobility,even for essential services , has to be reduced drastically in order to contain the spread of Covid-19.

# In[ ]:




