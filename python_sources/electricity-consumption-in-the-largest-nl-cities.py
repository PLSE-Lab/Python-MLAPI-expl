#!/usr/bin/env python
# coding: utf-8

# # Electricity Consumption Summary
# 

# [1. Introduction](#Intro)<br>
# &emsp; [1.1 Data and Libraries](#Data)<br>
# [2. Analysis](#Analysis)<br>
# &emsp;[2.1. Annual electricity consumption in the largest 10 dutch municipalities](#A21)<br>
# &emsp;[2.2. Number of connections in Eindhoven](#A22)<br>
# &emsp;[2.3. Number of connections vs Annual consumption of electricity](#A23)<br>
# &emsp;[2.4. Renewable resources](#A24)<br>
# &emsp;[2.5. Net Managers](#A25)<br>
# &emsp;[2.6. Low tarifs](#A26)<br>
# [3. Final Considerations](#conclusions)

# <a id="Intro"></a>
# # 1. Introduction
# 

# ![](https://images.pexels.com/photos/159397/solar-panel-array-power-sun-electricity-159397.jpeg?auto=compress&cs=tinysrgb&h=325&w=470)

# **The Netherlands** is known by being one of the countries that want to [ban petrol and diesel cars](https://www.theguardian.com/technology/2016/apr/18/netherlands-parliament-electric-car-petrol-diesel-ban-by-2025) in the coming few years. For this reason, we may expect a growth of electric vehicles as a sustainable transport alternative. Concerning the topics **electricity** and **Netherlands**, in this kernel, we analyze the energy consumption for the ten largest **dutch** municipalities listed on [wikipedia](https://en.wikipedia.org/wiki/Netherlands). We aim at enlightening questions such as:
# 
# **(a)** how the percentage of the net consumption of electricity changed along the years? 
# 
# **(b)** did more people save energy by using *renewable resources*, like solar panels? 
# 
# By investigating these points we may hypothesize the upcoming consumption of electricity, in particular, the importance of renewable energies to this matter. 

# <a id="Data"></a>
# ## 1.1. Data and Libraries
# ### Setting up the libraries and reading the data needed for to perform the analysis:

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html 
from scipy import stats
import re
import os


# The data we are interested in is divided in 29 csv files located inside a folder called *Electricity*. First, we read all these files inside the *Electricity* folder into a dictionary to acess them easier later.

# In[ ]:


df_electricity = dict()
for file in os.listdir("../input/dutch-energy/Electricity"):
    company = file.split('_')[0]
    year = re.findall('2+[0-9]+',file.split('_')[2])[0]
    df_electricity[company+year] = pd.read_csv("../input/dutch-energy/Electricity/"+file)


# Below, we find the 29 **keys** that will be used to refer to the *electricity* csv files. Each key is named after the net manager its data come from and its respective year. 

# In[ ]:


df_electricity.keys(),len(df_electricity)


# We may take one of these files, e.g. **enexis2010**, and explore its features/columns programmatically. This give us some insight about the dataset.

# In[ ]:


df_electricity['enexis2010'].info()


# <a id="Analysis"></a>
# # 2. Analysis

# <a id="A21"></a>
# ## 2.1. Annual electricity consumption in the 10 largest dutch municipalities

# Let us start the analysis by evaluating some features of the **10 largest municipalities** in the Netherlands listed on [wikipedia](https://en.wikipedia.org/wiki/Netherlands).

# In[ ]:


largest_municipalities = ['AMSTERDAM','ROTTERDAM',"'S-GRAVENHAGE",'UTRECHT',                          'EINDHOVEN','TILBURG','ALMERE','GRONINGEN','BREDA','NIJMEGEN']


# Note: The name for the city **Den Haag** has been replaced by **'s Gravenhage**, which is the dutch name as pointed out in the comments. 

# Now, we build a dataframe by gathering some information related to these municipalities. As we are interested in the annual consumption of electricity per city, then we group the data by city and select from each csv file the column **.annual_consume**. Since each csv file represents a different year, it is important to keep track of this information as well.

# In[ ]:


col1,col2,col3,col4 = [],[],[],[]
for net_manager in df_electricity.keys():
    for city in df_electricity[net_manager].groupby('city').sum().index.values:
        if city in largest_municipalities:
            value = df_electricity[net_manager].groupby('city').sum().annual_consume[city]   
            col1.append(re.findall('^[a-z]+',net_manager)[0])
            col2.append(re.findall('[0-9]+',net_manager)[0])
            col3.append(city)
            col4.append(value)
            
d={'net_manager': col1,
    'year': col2,
    'city': col3,
    'annual_consume': col4}

table = pd.DataFrame.from_dict(data=d)


# In[ ]:


table_yearcity = table.groupby(['year','city']).sum()['annual_consume'].unstack(level=1)
table_yearcity


# In[ ]:


fig = plt.figure(figsize=(12,6))
for city in largest_municipalities:
    try:
        plt.plot(table_yearcity[city].index.values,                 table_yearcity[city].values, label=city, marker='o')
        plt.legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=16)
    except:
        pass
plt.title("Annual consumption of Electricity",{'size':'18'})
plt.ylabel('Annual Consume (kWh)',{'size':'16'})
plt.xlabel('Year',{'size':'16'})
plt.grid(True,color='whitesmoke')
plt.tick_params(axis='both', labelsize=15)

caption = '''The consumption of electricity in kWh per year from 2009 to 2018. Each city in the 10 largest 
dutch municipalities list is represented by a different color. The dots indicate the exact annual consume 
for each city per year, and they are connected by straight lines to better infere the data behavior along 
the years. Notice that in 2009 some of the cities have missing data, therefore no record is shown.'''
fig.text(.5,- .17, caption, ha='center',fontsize=16)

plt.show()


# The plot above shows the annual electricity consumption the 10 largest municipalities from 2009 to 2018. We see that Rotterdam is experiencing a decrease in the annual consumption since 2012, while cities such as Utrecht, Nijmegen, Breda, Tilburg and Groningen do not seem to change significantly. On the other hand, the data for Eindhoven shows a huge increase in the annual consumption from 2016 to 2018. Almere seems to follow the same trend as Rotterdam, descreasing the consumption little by little each year, and Amsterdam has a descrease in 2015 but it goes back to an almost constant line.
# 
# To understand better some of this behavior we may also look at other features in the dataset. For example, was there an increase of connections in Eindhoven that led to this change from 2016 to 2018? And for the cities that experienced a reduction in the annual consumption, does it related to the use of renewable resources? For this second question we can check the percentage of the net consumption of electricity. If lower, this indicates that more energy returned to the grid. 

# <a id="A22"></a>
# ## 2.2. Number of Connections in Eindhoven

# In[ ]:


fig = plt.figure(figsize=(12,6))
for net_manager in sorted(df_electricity.keys()):
    if (df_electricity[net_manager].city=='EINDHOVEN').sum() !=0:
        n_conn_street = df_electricity[net_manager][df_electricity[net_manager].city=='EINDHOVEN']        .groupby('street').num_connections.sum()
        n_streets = len(n_conn_street)
        n_conn = df_electricity[net_manager][df_electricity[net_manager].city=='EINDHOVEN']        .num_connections.sum()
        year = re.findall('[0-9]+',net_manager)[0]        
        plt.barh(year,width=n_conn,label=n_streets)
        plt.legend(title="Number of Streets",loc=0,fontsize=13)

        #print(year,n_conn,n_streets)
        
plt.xlabel('Number of Connections',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Number of Connections per Year in Eindhoven',{'size':'18'})
plt.grid(True,color='whitesmoke')

plt.tick_params(axis='both', labelsize=13)
caption = '''The number of connections per year in Eindhoven. These connections are grouped by ranges of zipcodes,
thus collecting data from different streets. Here, the variable year is treated as a category, which is represented
by different colors. For each year, we also provide the number of distinct streets within the ranges of zipcodes,
as shown by the box inside the plot. For example, the number of connections in 2010 (blue) are from records 
of three distinct streets.'''
fig.text(.5,- .17, caption, ha='center',fontsize=14)

plt.show()


# The most prominent change observed in the previous graph is the increase of the annual consumption of electricity in Eindhoven from 2016 to 2018. One of the apparent causes for this behavior is the growth of connections in the city. However, the difference between the number of streets listed between the years of 2017 and 2018 and the previous years is so expressive that it is most likely that we lack data from 2016 backwards. 

# <a id="A23"></a>
# ## 2.3. Number of Connections vs. Annual Consumption of Electricity

# As shown in Section 2.2, the number of connections in Eindhoven increased along the years, and at the same time its annual consume. Now, we look at the overall growth for all the 10 largest dutch municipalities in order to understand if there is any correlation between these two variables.
# 
# In Section 2.1, we created a table with the data of the annual consume for each city and year, i.e. *table_yearcity*. Here, we will build a table containing the information about the number of connections for each city and year. 

# In[ ]:


col1,col2,col3 = [],[],[]
        
for i in df_electricity.keys():
    counts = df_electricity[i].groupby('city').num_connections.sum().values
    cities = df_electricity[i].groupby('city').num_connections.sum().index
    year = re.findall('[0-9]+',i)[0]
    for city,count in zip(cities,counts):
        if city in largest_municipalities:
            col1.append(year)
            col2.append(city)
            col3.append(count)
            
d={ 'year': col1,
    'city': col2,
    'n_connections': col3}

table_n_conn = pd.DataFrame.from_dict(data=d)


# In[ ]:


table_n_connections = table_n_conn.groupby(['year','city']).sum()['n_connections'].unstack(0)
table_n_connections


# In the table above we gathered the data about the number of connections for each of the 10 largest dutch municipalities per year. Because we wish to analyze the overall correlation between the number of connections and the annual consume, we aggregate these values by year.

# In[ ]:


year_conn_cons =pd.DataFrame({'year':table_n_connections.sum().index,              'n_connections':table_n_connections.sum().values,              'annual_consume':table_yearcity.T.sum().values})
year_conn_cons


# In[ ]:


fig = plt.figure(figsize=(12,6))
plt.grid(color='whitesmoke')

xi=year_conn_cons.n_connections
y=year_conn_cons.annual_consume

slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = slope*xi+intercept
print('slope:',slope,'intercept:',intercept)

for i,j in zip(year_conn_cons.n_connections,year_conn_cons.annual_consume):
    plt.scatter(i,j,label='year')
    plt.legend(title='Year',labels=list(range(2009,2019,1)),loc=4,fontsize=14)

plt.plot(xi, line,color='black',linestyle='--')

plt.xlabel('Number of Connections',{'size':'16'})
plt.ylabel('Annual Consume',{'size':'16'})
plt.title('Total Number of Connections vs Total Annual Consume Per Year',{'size':'18'})
plt.tick_params(axis='both', labelsize=13)

caption = '''Total annual consume of electricity per year vs the total number of connections per year. 
Each different color (dots) labels a different year. The dashed line represents a linear regression for 
these data points, where the slope and the intercept are ~95.61 and ~77821380.46, respectively.'''
fig.text(.5,- .10, caption, ha='center',fontsize=14)

plt.show()


# Although an overall behavior suggests that the annual consumption of electricity increases as we increase the number of connections, we cannot make any strong statement, since we only have 10 data points and in the period 2012-2015 the increasing trend is broken. There should be other variables that are affecting this growth. 
# 

# <a id="A24"></a>
# ## 2.4.  Renewable resources

# Another interesting aspect of the electricity consumption is the delivery percentage. For every year (and city), there is a distribution of these delivery percentages, where 100% means that no energy was given back to the grid. Let us assume that if more electricity has returned to the grid, then people are using different resources. Below we analyze the percentage of energy that has been delivered every year from 2009 to 2018.

# In[ ]:


data_not100,data_100,list_year = [],[],[]

for net_manager in sorted(df_electricity.keys()):
    for city in largest_municipalities:
        try:
            year = re.findall('[0-9]+',net_manager)[0]
            data = df_electricity[net_manager].groupby(['city','delivery_perc'])            .count()['net_manager'].unstack(level=0)[city]
            list_year.append(year)
            data_not100.append(data[data.index!=100].sum())
            data_100.append(data[data.index==100].values[0])
        except:
            pass


# In[ ]:


data_plot = pd.DataFrame({"year":list_year,"100":data_100,"Not 100":data_not100})
data_plot[data_plot.year=='2010']


# To briefly explain the data we have selected, let us take the year 2010 above as an example. Each row indicates an observation related to one of the cities in the 10 largest municipalities list. The values for columns 100 and Not 100 are **counts** of the number of percentages equal to 100% or different from it. For the first row (index=0), its respective city has no observations in which there has been a delivery percentage different from 100%, and 11 observations with a delivery percentage of 100%. Notice that there are more rows than cities considered here, and this happens because some cities are present in the data of more than one net manager (enexis, stedin, liander) and we read each file individually to extract the information. 

# In[ ]:


sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(x="100", y="year", kind="box", orient="h", height=6, aspect=1.5,data=data_plot)
plt.xlabel('100',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Counting the electricity delivery % along the years',{'size':'18'});


# In[ ]:


sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(x="Not 100", y="year", kind="box", orient="h", height=6,aspect=1.5, data=data_plot)
plt.xlabel('Not 100',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Counting the electricity delivery % along the years',{'size':'18'});


# In the plots above we split the data for the largest municipalities into two categories: the counts of percentage that is either **100%** or below (i.e. **Not 100%**). In this manner, we can investigate if there a trend towards lowering the delivery of energy, meaning that more people are using renewable energies like solar panels.
# 
# Here, we decide to use a boxplot, which measures the spread of the data, to compare the electricity delivery percentage distribution along the years. The first plot shows that from 2009 to 2018, there has been a significant decrease of the maximum value (also identifying some outliers, i.e. the individual diamonds records). The mean of the distribution has moved towards left, indicating that the distribution has changed its shape from right to left skewed. In 2018, 75% of the data falls below the median of 2009.  In the second plot, an opposite behavior happens. There has been an increase in both maximum and mininum values, and in the distribution's range. 

# In[ ]:


d2018 = data_plot[data_plot.year=='2018']
sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(data=d2018[['100','Not 100']],kind="box", orient="h", height=3.2, aspect=3)
plt.xlabel('Count',{'size':'16'})
plt.title('Comparing the electricity delivery % in 2018',{'size':'16'});


# <a id="A25"></a>
# ## 2.5. Net managers
# 
# In this section, we investigate the differences between the net managers Enexis, Liander and Stedin. First, we check how many cities each one of them covers, and how this number changes from 2009 to 2018. 
# 

# In[ ]:


dict_enexis, dict_liander, dict_stedin = dict(),dict(),dict() 
dict_enexis_cities, dict_liander_cities, dict_stedin_cities = dict(),dict(),dict() 

for net_manager in sorted(df_electricity.keys()):
    city_names = list(df_electricity[net_manager].city.value_counts().index)
    city_values = list(df_electricity[net_manager].city.value_counts().values)
    year = re.findall('[0-9]+',net_manager)[0]
    net_manager_name = re.findall('^[a-z]+',net_manager)[0]
    #print(year,net_manager_name,len(city_names))
    if net_manager_name == 'enexis':
        dict_enexis[year]=len(city_names)
        dict_enexis_cities[year]=city_names
    elif net_manager_name == 'liander':
        dict_liander[year]=len(city_names)
        dict_liander_cities[year]=city_names
    else:
        dict_stedin[year]=len(city_names)
        dict_stedin_cities[year]=city_names


# In[ ]:


df_nm = pd.DataFrame.from_dict({'enexis': dict_enexis,'enexis_cities': dict_enexis_cities,                        'liander': dict_liander,'liander_cities': dict_liander_cities,                        'stedin': dict_stedin,'stedin_cities': dict_stedin_cities})


# In[ ]:


data_net_managers =pd.DataFrame({'enexis': dict_enexis, 'liander': dict_liander,'stedin': dict_stedin})

rescaled_dnm = (data_net_managers-data_net_managers.min())/(data_net_managers.max()-data_net_managers.min())

tb1 = data_net_managers.style.set_table_attributes("style='display:inline'").set_caption('Original Data')
tb2= rescaled_dnm.style.set_table_attributes("style='display:inline'").set_caption('Rescaled Data')

display_html(tb1._repr_html_() + tb2._repr_html_(), raw=True)


# In[ ]:


fig = plt.figure(figsize=(12,6))

plt.scatter(y=rescaled_dnm.enexis,x=data_net_managers.index, marker='d',label='Enexis')
plt.scatter(y=rescaled_dnm.liander,x=data_net_managers.index,label='Liander', marker='8')
plt.scatter(y=rescaled_dnm.stedin,x=data_net_managers.index,label='Stedin', marker='x')

plt.ylabel('Relative # of cities covered',{'size':'16'})
plt.xlabel('Year',{'size':'16'})
plt.title('Relative number of Cities covered per Year by Net Manager',{'size':'18'})
plt.legend(loc=0,fontsize=13)

plt.tick_params(axis='both', labelsize=13)
caption = '''The rescaled data concerning the number of cities covered by each net manager
(Enexis, Liander and Stedin) per year. Here, 0.0 and 1.0 represent the minimum and maximum 
number of cities, respectively. There is no data from Enexis in 2009. '''
fig.text(.5,- .10, caption, ha='center',fontsize=14)

plt.show()


# Above I plot the relative number of cities covered per year by each Net Manager. Because the number of cities covered by both Enexis and Liander are comparable in size while Stedin is not, it is easier to visualize how the three of them changed altogether by plotting a rescaled feature, where zero and one mean the mininum and maximum value, respectively, they ever had. We observe an increase in Stedin from 2009 to 2013 and then a sudden reduction followed by another increase. Importantly, we cannot judge how significant was the change but only the behavior along the years. If one looks at the tables before, it is easy to see that for Stedin the changes are a few cities more or less. Similarly, for Enexis and Liander, the changes do not represent a huge increase/descrease. However, we may ask if the change from 2012 to 2013 means that some cities previously covered by Stedin passed to either Enexis or Liander. Does this exchanging behavior mean that cities have changed their net manager? 

# Enexis cities did not change from 2012 to 2013, same list:

# In[ ]:


np.array(df_nm[df_nm.index=='2012'].enexis_cities[0].sort(reverse=False)) ==np.array(df_nm[df_nm.index=='2013'].enexis_cities[0].sort(reverse=False))


# Although the numbers indicate that Liander city counts changed from 1106 to 1105, this net manager had 5 cities from 2012 to 2013 replaced by 4 new ones, and effectively a reduction of 1 city count.

# In[ ]:


for city in df_nm[df_nm.index=='2012'].liander_cities[0]:
    if city not in df_nm[df_nm.index=='2013'].liander_cities[0]:
        print('2012',city)
for city in df_nm[df_nm.index=='2013'].liander_cities[0]:
    if city not in df_nm[df_nm.index=='2012'].liander_cities[0]:
        print('2013',city)


# Stedin city counts changed from 267 to 269 by replacing 2 cities with 4 new ones.

# In[ ]:


for city in df_nm[df_nm.index=='2012'].stedin_cities[0]:
    if city not in df_nm[df_nm.index=='2013'].stedin_cities[0]:
        print('2012',city)        
for city in df_nm[df_nm.index=='2013'].stedin_cities[0]:
    if city not in df_nm[df_nm.index=='2012'].stedin_cities[0]:
        print('2013',city)


# This result means that the net managers that have increase the city counts not necessarily got to cover the places occupied by one of these three net managers before. In the case of 2012 to 2013 actually none of the cities that were covered before by either Stedin or Liander were occupied after by one of the three. Unless Enexis has already those cities covered then it could have increase the frequency of the cities, since we only count them by name. However this isnt the case, as one may check. Another question one may ask is which net manager is covering now these cities? (this ends up out of the dataset scope) 

# <a id="A26"></a>
# ## 2.6. Low tarifs
# 
# The last feature we wish to cover in this analysis is the low tarif hours. These are hours from 10 p.m. to 7 a.m. on weekdays, and the whole day on weekends. In this section, we aim at understading if people have saved more money by making use of the low tarif hours. 

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize= (15,3))

for i,j in zip(range(3),['liander','stedin','enexis']):
    axs[i].hist(df_electricity[j+'2018'].annual_consume_lowtarif_perc,alpha=0.5,label='2018')
    axs[i].axvline(x=df_electricity[j+'2018'].annual_consume_lowtarif_perc.describe()['mean'])
    axs[i].hist(df_electricity[j+'2010'].annual_consume_lowtarif_perc,alpha=0.5,label='2010')
    axs[i].axvline(x=df_electricity[j+'2010'].annual_consume_lowtarif_perc.describe()['mean'],               color='orange',linestyle='--')
    axs[i].set_title(j.capitalize(),{'size':'18'})

axs[1].set_xlabel('% of consume during the low tarif hours',{'size':'16'})
axs[0].set_ylabel('Frequency',{'size':'14'})
plt.legend(loc=0)

caption = '''...'''
fig.text(.5,- .17, caption, ha='center',fontsize=14)

plt.show()


# To understand whether people are saving more money, we compare the annual consumption at low tarif hours between the years 2010 (orange) and 2018 (blue). The vertical lines (orange and blue) indicate the mean value of the distributions in 2010 and 2018. We observe that for all net managers there has been an increase of the mean percentage of consume during low tarif hours. This may suggest that people are more concious on how they can save money by using some higher energy consuming equipments (such as washing machines) during low tarif hours. 
# 

# <a id="conclusions"></a>
# # 3. Final considerations

# work in progress...

# In[ ]:




