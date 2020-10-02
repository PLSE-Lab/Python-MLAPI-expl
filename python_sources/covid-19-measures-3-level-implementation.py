#!/usr/bin/env python
# coding: utf-8

# # COVID_19 3 Level Measure Comparision done for The most affected and least affected countries.
# How is the implementation of existing strategies affecting the rates of COVID-19 infection?
# Task Details
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management. - COVID19 Uncover Challenge
# 
# Implementation of the Problem
# The General Question which we require to study in this notebook is how the implementation of existing strategies affecting the rate of COVID-19 Infection. For the sake of this notebook we proceed with the following analogies ans steps.
# 
# Checking which countries are somewhat successful in controlling the rate of COVID-19 Spreads.
# Figuring out the various measures adopted by that very countries to understand how did it affected spreads.
# Checking the countries that displayed a much higher COVID-19 Infections and the growth is exponential as of now.
# Figuring out where the country lacked in implementation tools.
# Finally we can compare how the country that controlled the COVID-19 Proceeded with the community measures than that to the countries that currently exhibit a near to exponential growth of the COVID-19 Infections.

# In[ ]:


import pandas as pd
from pandas import DataFrame
import numpy as np


# **This data set contains data that have been collected from public sources. The column "Date" corresponds to the date of application (if not found, date of announcement is provided). The column "Region" specifies if measures were implemented at sub-national level and precise the region. Intervention measures are classified following a four-level coding strategy. Level 1 (L1) shows the top-level categories of intervention measures in our classification scheme, including (1) Case identification, contact tracing and related measures, (2) Environmental measures, (3) Healthcare and public health capacity, (4) Resource allocation, (5) Risk communication, (6) Social distancing, (7) Travel restriction, (8) Returning to normal life. Level 2 (L2) presents subcategories of measures for each L1 category. Level 3 (L3) presents subcategories of measures for each L2-category measure or specific detail about the corresponding L2-category measure.**

# In[ ]:


mobility=DataFrame(pd.read_csv(r'../input/tagged-covid19-measures/covid_measures.csv',encoding='latin-1'))
mobility.head()


# # Extraction of data from least affected countries like New Zealand, Austria, Finland etc. and least affected countris like UK, Japan, Canada etc.

# In[ ]:


finland=mobility[mobility['iso3c']=='FIN']
finland=finland.reset_index().drop('index',axis=1)
#print(finland)


# In[ ]:


NewZ=mobility[mobility['Country']=='New Zealand']
NewZ=NewZ.reset_index().drop('index',axis=1)
# print(NewZ)
Aus=mobility[mobility['Country']=='Austria']
Aus=Aus.reset_index().drop('index',axis=1)
# print(Aus)
Tai=mobility[mobility['Country']=='Taiwan']
Tai=Tai.reset_index().drop('index',axis=1)
# print(Tai)
Sw=mobility[mobility['Country']=='Sweden']
Sw=Sw.reset_index().drop('index',axis=1)

Sk=mobility[mobility['Country']=='South Korea']
Sk=Sk.reset_index().drop('index',axis=1)

Mx=mobility[mobility['Country']=='Mexico']
Mx=Mx.reset_index().drop('index',axis=1)

Sg=mobility[mobility['Country']=='Singapore']
Sg=Sg.reset_index().drop('index',axis=1)

Nw=mobility[mobility['Country']=='Norway']
Nw=Nw.reset_index().drop('index',axis=1)

Cz=mobility[mobility['Country']=='Czech Republic']
Cz=Cz.reset_index().drop('index',axis=1)

lowest=['Finland','New Zealand','Austria','Taiwan','Sweden','South Korea','Mexico','Singapore','Norway','Czech Republic']
highest=['Uk','France','Canada','Japan','Germany','Switzerland','Belgium','Netherlands','Portugal','India']

uk=mobility[mobility['Country']=='United Kingdom']
uk=uk.reset_index().drop('index',axis=1)
# print(uk)

France=mobility[mobility['Country']=='France (metropole)']
France=France.reset_index().drop('index',axis=1)
# print(France)

Canada=mobility[mobility['Country']=='Canada']
Canada=Canada.reset_index().drop('index',axis=1)
# print(Canada)

Japan=mobility[mobility['Country']=='Japan']
Japan=Japan.reset_index().drop('index',axis=1)
#print(Japan)
Ge=mobility[mobility['Country']=='Germany']
Ge=Ge.reset_index().drop('index',axis=1)

St=mobility[mobility['Country']=='Switzerland']
St=St.reset_index().drop('index',axis=1)

Be=mobility[mobility['Country']=='Belgium']
Be=Be.reset_index().drop('index',axis=1)

Nt=mobility[mobility['Country']=='Netherlands']
Nt=Nt.reset_index().drop('index',axis=1)

Pt=mobility[mobility['Country']=='Portugal']
Pt=Pt.reset_index().drop('index',axis=1)

Id=mobility[mobility['Country']=='India']
Id=Id.reset_index().drop('index',axis=1)


# # Extraction of Dates from the countries for performing Time-Series Forcasting

# In[ ]:


# print(mobility['date'])


findates=list(finland['Date'])

newzdates=list(NewZ['Date'])

ausdates=list(Aus['Date'])

taidates=list(Tai['Date'])

swdates=list(Sw['Date'])

skdates=list(Sk['Date'])

mxdates=list(Mx['Date'])

sgdates=list(Sg['Date'])

nwdates=list(Nw['Date'])

czdates=list(Cz['Date'])

ukdates=list(uk['Date'])

frdates=list(France['Date'])

cndates=list(Canada['Date'])

jpdates=list(Japan['Date'])

gedates=list(Ge['Date'])

stdates=list(St['Date'])

bedates=list(Be['Date'])

ntdates=list(Nt['Date'])

ptdates=list(Pt['Date'])

iddates=list(Id['Date'])


# # Tags Used for annotation

# In[ ]:


tags=DataFrame(pd.read_csv(r'../input/tagged-covid19-measures/tags.csv',encoding='latin-1'))
tags


# # Plotting of Graphs: Tagged Measures Vs Dates for most affected and least countries for Level 1 measure

# In[ ]:


#Retail and Recreation
from matplotlib import pyplot as plt
plt.style.use('seaborn')
plt.plot_date(findates,list(map(int,  list(finland['Measure_L1']))),label='Finland')
plt.plot_date(newzdates,list(map(int,  list(NewZ['Measure_L1']))),label='NewZealand')
plt.plot_date(ausdates,list(map(int,  list(Aus['Measure_L1']))),label='Austria')
plt.plot_date(taidates,list(map(int,  list(Tai['Measure_L1']))),label='Taiwan')
plt.plot_date(swdates,list(map(int,  list(Sw['Measure_L1']))),label='Sweden')
plt.plot_date(skdates,list(map(int,  list(Sk['Measure_L1']))),label='South Korea')
plt.plot_date(mxdates,list(map(int,  list(Mx['Measure_L1']))),label='Mexico')
plt.plot_date(sgdates,list(map(int,  list(Sg['Measure_L1']))),label='Singapore')
plt.plot_date(nwdates,list(map(int,  list(Nw['Measure_L1']))),label='Norway')
plt.plot_date(czdates,list(map(int,  list(Cz['Measure_L1']))),label='Czech Republic')
plt.xlabel('Date')
plt.ylabel('Measure_L1')
plt.title('COVID19 Least Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()

plt.plot_date(ukdates,list(map(int,  list(uk['Measure_L1']))),label='UK')
plt.plot_date(frdates,list(map(int,  list(France['Measure_L1']))),label='France')
plt.plot_date(cndates,list(map(int,  list(Canada['Measure_L1']))),label='Canada')
plt.plot_date(jpdates,list(map(int,  list(Japan['Measure_L1']))),label='Japan')
plt.plot_date(gedates,list(map(int,  list(Ge['Measure_L1']))),label='Germany')
plt.plot_date(stdates,list(map(int,  list(St['Measure_L1']))),label='Switzerland')
plt.plot_date(bedates,list(map(int,  list(Be['Measure_L1']))),label='Belgium')
plt.plot_date(ntdates,list(map(int,  list(Nt['Measure_L1']))),label='Netherlands')
plt.plot_date(ptdates,list(map(int,  list(Pt['Measure_L1']))),label='Portugal')
plt.plot_date(iddates,list(map(int,  list(Id['Measure_L1']))),label='India')
plt.xlabel('Date')
plt.ylabel('Measure_L1')
plt.title('COVID19 Most Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()

# plt.plot_date((mobility['Date']),list(map(int,  list(mobility['Measure_L3']))))
# plt.show()


# **In the above graph, many observations about the L1 measures taken is seen.The least affected countries are seen to implement measures like Social Distancing(1), Travel Restricition(2), Case Identification(3) etc..in earlier dates itself(a dense plot is seen). Where as when we see the plot of Most affected countries, the measures taken initially is sparse comparitively and later becomes dense as the date progresses. **

# ----------------------------------------------------------------------------------------------------------------------------------------------

# # Plotting of Graphs: Tagged Measures Vs Dates for most affected and least countries for Level 2 measures

# In[ ]:


#Retail and Recreation
from matplotlib import pyplot as plt
plt.style.use('seaborn')
plt.plot_date(findates,list(map(int,  list(finland['Measure_L2']))),label='Finland')
plt.plot_date(newzdates,list(map(int,  list(NewZ['Measure_L2']))),label='NewZealand')
plt.plot_date(ausdates,list(map(int,  list(Aus['Measure_L2']))),label='Austria')
plt.plot_date(taidates,list(map(int,  list(Tai['Measure_L2']))),label='Taiwan')
plt.plot_date(swdates,list(map(int,  list(Sw['Measure_L2']))),label='Sweden')
plt.plot_date(skdates,list(map(int,  list(Sk['Measure_L2']))),label='South Korea')
plt.plot_date(mxdates,list(map(int,  list(Mx['Measure_L2']))),label='Mexico')
plt.plot_date(sgdates,list(map(int,  list(Sg['Measure_L2']))),label='Singapore')
plt.plot_date(nwdates,list(map(int,  list(Nw['Measure_L2']))),label='Norway')
plt.plot_date(czdates,list(map(int,  list(Cz['Measure_L2']))),label='Czech Republic')
plt.xlabel('Date')
plt.ylabel('Measure_L2')
plt.title('COVID19 Least Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()

plt.plot_date(ukdates,list(map(int,  list(uk['Measure_L2']))),label='UK')
plt.plot_date(frdates,list(map(int,  list(France['Measure_L2']))),label='France')
plt.plot_date(cndates,list(map(int,  list(Canada['Measure_L2']))),label='Canada')
plt.plot_date(jpdates,list(map(int,  list(Japan['Measure_L2']))),label='Japan')
plt.plot_date(gedates,list(map(int,  list(Ge['Measure_L2']))),label='Germany')
plt.plot_date(stdates,list(map(int,  list(St['Measure_L2']))),label='Switzerland')
plt.plot_date(bedates,list(map(int,  list(Be['Measure_L2']))),label='Belgium')
plt.plot_date(ntdates,list(map(int,  list(Nt['Measure_L2']))),label='Netherlands')
plt.plot_date(ptdates,list(map(int,  list(Pt['Measure_L2']))),label='Portugal')
plt.plot_date(iddates,list(map(int,  list(Id['Measure_L2']))),label='India')
plt.xlabel('Date')
plt.ylabel('Measure_L2')
plt.title('COVID19 Most Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()


# **In the above graph, many observations about the L2 measures taken is seen.The least affected countries are seen to implement measures like Risk Communication(4), Travel Restricition(2), Healthcare(5) etc..in earlier dates itself(a dense plot is seen). Where as when we see the plot of Most affected countries, the measures like Travel Restriction has been taken in earlier dates whereas other measures are neglected, however later becomes dense as the date progresses. **

# ----------------------------------------------------------------------------------------------------------------------------------------------

# # Plotting of Graphs: Tagged Measures Vs Dates for most affected and least countries for Level 3 measures

# In[ ]:


# parks_percent_change_from_baseline
plt.style.use('seaborn')
#Retail and Recreation
from matplotlib import pyplot as plt
plt.style.use('seaborn')
plt.plot_date(findates,list(map(int,  list(finland['Measure_L3']))),label='Finland')
plt.plot_date(newzdates,list(map(int,  list(NewZ['Measure_L3']))),label='NewZealand')
plt.plot_date(ausdates,list(map(int,  list(Aus['Measure_L3']))),label='Austria')
plt.plot_date(taidates,list(map(int,  list(Tai['Measure_L3']))),label='Taiwan')
plt.plot_date(swdates,list(map(int,  list(Sw['Measure_L3']))),label='Sweden')
plt.plot_date(skdates,list(map(int,  list(Sk['Measure_L3']))),label='South Korea')
plt.plot_date(mxdates,list(map(int,  list(Mx['Measure_L3']))),label='Mexico')
plt.plot_date(sgdates,list(map(int,  list(Sg['Measure_L3']))),label='Singapore')
plt.plot_date(nwdates,list(map(int,  list(Nw['Measure_L3']))),label='Norway')
plt.plot_date(czdates,list(map(int,  list(Cz['Measure_L3']))),label='Czech Republic')
plt.xlabel('Date')
plt.ylabel('Measure_L3')
plt.title('COVID19 Least Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()

plt.plot_date(ukdates,list(map(int,  list(uk['Measure_L3']))),label='UK')
plt.plot_date(frdates,list(map(int,  list(France['Measure_L3']))),label='France')
plt.plot_date(cndates,list(map(int,  list(Canada['Measure_L3']))),label='Canada')
plt.plot_date(jpdates,list(map(int,  list(Japan['Measure_L3']))),label='Japan')
plt.plot_date(gedates,list(map(int,  list(Ge['Measure_L3']))),label='Germany')
plt.plot_date(stdates,list(map(int,  list(St['Measure_L3']))),label='Switzerland')
plt.plot_date(bedates,list(map(int,  list(Be['Measure_L3']))),label='Belgium')
plt.plot_date(ntdates,list(map(int,  list(Nt['Measure_L3']))),label='Netherlands')
plt.plot_date(ptdates,list(map(int,  list(Pt['Measure_L3']))),label='Portugal')
plt.plot_date(iddates,list(map(int,  list(Id['Measure_L3']))),label='India')
plt.xlabel('Date')
plt.ylabel('Measure_L3')
plt.title('COVID19 most Affected Countries')
plt.tight_layout()
plt.legend()
plt.show()


# **In the above graph, the same observations about the L3 measures is seen. The density of measures taken initially by least affected countries is more especially for Case Identification(3),Healthcare(5), closure of educational instituions(6) etc.. The same trend is observed in Most affected countires who fail to implement these measures in the initial dates.**

# **Thus from the above data analysis we can easily conclude that the least affected countries have taken up certain measures in the initial days itself which has reduced the risk of contracting COVID19:**
# * Social distancing
# * Travel restriction
# * Case identification	
# * Healthcare and public health capacity
# * Closure of educational institutions
# * Environmental cleaning and disinfection
# 
# **The most affected countires have however failed to implement the same in intital days paving way to a lot of COVID19 cases and deaths eventually.**
# 
