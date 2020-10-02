#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Let's import all data 

# The shape of the population dataframe is (264,61). It shows the population for each country from 1960 to 2016

# In[ ]:


population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')


# ## **Population**

# In[ ]:


temp_pop= population
temp_pop.drop(columns=['Country Name','Country Code', 'Indicator Name', 'Indicator Code'],axis =1, inplace=True)


# In[ ]:


pop_sum=temp_pop.sum()
pop_sum=pd.DataFrame(pop_sum).reset_index()
pop_sum.columns= ['Year','Total Population']
# type(pop_sum)
pop_sum.head()
pop_sum.shape


# In[ ]:


#  Stack overflow method 

plt.figure(figsize=(25,10))
plt.plot(pop_sum['Year'], pop_sum['Total Population'])
plt.title('Global Population from 1960 to 2016')
plt.xticks(np.arange(1960,2017))

plt.show()


# **The above graph shows the increase of the total population as the years are going by . The x axis is supposed to go from 1960 to 2016. Right now it starts at 0 which is representing 1960 and goes upto 2016 representing 50 something. I will work on reseting this axes tomorrow**

# ## Let's do likewise for fertility 

# In[ ]:


temp_fert= fertility
temp_fert.head()
temp_fert.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_fert =temp_fert.dropna()


# In[ ]:


# new_fert.isnull().values.any() # no empty values 
new_fert.head()


# In[ ]:


fert_sum = new_fert.mean()
fert_sum = pd.DataFrame(fert_sum).reset_index()
fert_sum.columns=['Year', 'Fertility']
fert_sum.describe()


# In[ ]:


# fert_sum.plot()
plt.figure(figsize=(25,10))
plt.plot(fert_sum['Year'], fert_sum['Fertility'])
plt.xticks(np.arange(1960,2017))
plt.title('Fertility from 1960 to 2016')
plt.show()


# **The above graph shows that over the 50 something years, the  mean fertility rate across the world has fallen, pretty drastically to almost half of it was in the 1960 and a little after that **
# 

# **Life Expectancy**

# In[ ]:


df_life = life
# df_life.head()
df_life.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_life = df_life.dropna()
new_life.head()


# In[ ]:


new_life.isnull().values.any()# no empty values 
life_mean =  new_life.mean()
life_mean = pd.DataFrame(life_mean).reset_index()
life_mean.columns= ['Year', 'Life expectancy']
# life_mean.plot()
life_mean.head()


# In[ ]:


plt.figure(figsize=(25,10))
plt.plot(life_mean['Year'], life_mean['Life expectancy'])
plt.xticks(np.arange(1960,2017))
plt.title('Life Expectancy from 1960 to 2016')
plt.show()


# **The above graph shows that there is an increase in the mean life expectancy across the world in the last 50 or so years. The life expectancy has gone up by almost 20 years. **

# ## Formalize EDA
# *Don't really need to worry about this. This was an excercise of just formalizing the process. If you did the earlier steps, then this following code cell is redundant *
# 

# In[ ]:


def make_df(df,value_name):
    
    # First off, I will drop the useless columns 
    df.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
    
    df.dropna()
    
#     while df.isnunll().values.isany()== True:
#           df.dropna()
#     else:
#           continue
        
    if value_name == 'Population':
        df_stat =df.sum()
    else:
        df_stat =df.mean()
    
    df = pd.DataFrame(df_stat).reset_index()
    
    if 'Population' in value_name:
        df.columns= ['Year', 'Population']
    elif 'Fertility' in value_name:
        df.columns = ['Year', 'Fertility']
    else:
        df.columns = ['Year', 'Life Expectancy']
    
    return df
    
    


# ## This following cell is to make sure that the formalized EDA works and it does!!!

# In[ ]:


population = pd.read_csv('../input/country_population.csv')
pop_df = make_df(population, 'Population')
pop_df.plot()
# pop_df.head()


# **Fertility dataFrame**

# In[ ]:


fert_sum.describe()


# **DataFrame containing the mean of life expectancy**

# In[ ]:


life_mean.describe()


# **Data Frame containig the sum of population in the world in that year** 

# In[ ]:


pop_sum.describe()


# # Let's create a new dataFrame containing the data for life expectancy, total population and fertility rate in each year 

# In[ ]:


# BUILD A MASSIVE DATA FRAME CONTAING THE DATA FOR ALL THREE CRITERIA
# world_data = pd.merge(pd.merge(pop_sum, life_mean, on='Year'), fert_sum, on='Year')
test = pd.merge(pop_sum,life_mean, on='Year')
world_data = pd.merge(test, fert_sum, on='Year')
world_data.columns


# **The columns are too far apart in numbering. The population is in millions and fertility is in single digits with decimals.**
# 
# 
# **So let's scale them. **

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
world_data[['Total Population', 'Life expectancy', 'Fertility']] = scaler.fit_transform(world_data[['Total Population', 'Life expectancy', 'Fertility']])
world_data.head()


# **Let's plot the massive dataFrame **

# In[ ]:


# world_data.plot(x="Year", y=["Total Population","Life expectancy","Fertility"], figsize = (10,10) )


# # So obviously the most interesting thing here is that The Total population is increasing from 1960 to 2016 consistently. However, the fertility rate is decreasing consistently at the same time.
# 
# ## <font color='red'> So the question I will be tackling will be see how the world population is still increasing while fertility is so clearly decreasing </font>

# In[ ]:


population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')
    


# In[ ]:


population = pd.read_csv('../input/country_population.csv')
population.head()
pop_year = population.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
pop_year= pop_year.T
# population.head()
pop_year.head()

pop_year.shape
# pop_year.isnull().values.any()
t = pop_year.isnull().apply(sum, axis=0)
t

# for col in pop_year:
#     if t:
#         pop_year.fillna(pop_year.mean())

pop_year = pop_year.fillna(pop_year.mean())        
# pop_year.isnull().values.any() # The answer is false
pop_year.shape
# pop_year


# In[ ]:


print(fertility.T.shape[1]-fert_year.shape[1],'countries have been dropped from the dataset because there were atleast 40 or more missing NaN values')


# ## <font color ='purple'>The following code cell was interesting to clean up. There were abotu 5 to 7 countries that had over 50 entries in each column that were missing. Since the data spans for only 56 years, any country that had more than410 missing values I dropped, them and any other column with less than 40 missing values, I replace the missing values of that column with the mean of the remaining values in that columns. As a result of this, I had 20 columns or countries are removed from the fertility dataframe.  </font>

# In[ ]:


fertility = pd.read_csv('../input/fertility_rate.csv')

fertility=fertility.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
fert_year = fertility
# # fert_year.reset_index()
# t = fert_year.isnull().apply(sum, axis=0)
# t

# for col in fert_year:
#     if t[col]>=40:
#         del fert_year[col]
# #         fert_year.fillna(fert_year[col].mean)
#     else:
#         fert_year.fillna(fert_year[col].mean)
        
fert_year= fert_year.dropna()
fert_year


# In[ ]:


fert_temp = fert_year

fert_temp= fert_temp.dropna()
# fert_temp=fert_temp.drop('mean of year', axis =1)
fert_temp


# In[ ]:


country_names = pd.DataFrame(fert_temp['Country Name'])
country_names
fert_ignore =pd.DataFrame(fert_temp.iloc[:,:]) #Ignores the names of the country
# fert_ignore.columns = range(fert_ignore.shape[1])
fert_ignore
if 'mean of year' in fert_ignore.columns:
    fert_ignore =fert_ignore.drop('mean of year',axis =1)
fert_ignore.T
fert_mean = fert_ignore.mean(axis =0)
type(fert_mean)
fert_mean
# print(fert_mean.shape)
# print(fert_ignore.T.shape)
# # #Need to add the mean column to DataFrame as well. I will add it at the very begining
idx= 0
fert_add = fert_ignore.T
fert_add.insert(loc=idx, column='mean of year', value=fert_mean)
fert_add.columns[0]
# fert_add.columns[1:] #= np.arange(1960,2017)
fert_add = fert_add.T
# len(fert_add.T.columns)
# pd.DataFrame(fert_add['1960']).reset_index().drop('index',axis =1)
# column_interest= fert_add.columns[41:]
fert_add.iloc[1]


# ## <font color=Green>We are now going to seperate the columns where the fertility value if greater than the mean fertility value across the globe for that year. I am taking the years 2000 and onwars because y2k and what not . Any country with higher than 3.0 fertility rate stays </font>

# In[ ]:


column_drop = fert_add.columns[1:41] #1960 to 1999
column_interest= fert_add.columns[41:] #2000 to 2016
mil_fert = fert_add.drop(column_drop,axis=1) #fertility data since 2000

mil_fert.iloc[:,1:]#.mean().mean()
mil_filter = mil_fert[mil_fert[column_interest]>= 3.0]
# mil_temp = mil_filter[]
count_try= mil_filter.drop('Country Name', axis=1).dropna()
count_try


# <font color=purple> **The above  cell shows that there are 72 countries that have a higher than mean/average fertility rate from 2000 to 2016.  72 is too many, so i will try to limit it down even more**</font>

# In[ ]:


more_filter = count_try[count_try[column_interest]>=5.0]
more_filter= more_filter.dropna()
more_filter


# <font color = green>**Okay so by increasing the threshold of fertility rate to 5.0, we have dropped the countries from 72 to 15. let's see if we can limit it down to less than 10 **</font>

# In[ ]:


filter_six = more_filter[more_filter[column_interest]>=5.5]
filter_six = filter_six.dropna()
filter_six


# ## <font color =green >******Okay, so now we are talking. That is ten countries who had higher than 5.5 fertility rate from 2000 to 2016 when the mean for all the countries in the same period was around 3.0. The fertility rate for these 10 countries was almost twice the global rate. **</font>

# <font color = Purple>**Now we are going to put the names of those countries back into the above data frame **</font>

# In[ ]:


country_indices=filter_six.index.values
country_indices
type(country_indices)


# In[ ]:


insert_names = country_names[country_names.index.isin(country_indices)]
insert_names['Country Name']


# In[ ]:


idx= 0
country_fert = filter_six
country_fert.insert(loc=idx, column='Country Name', value=insert_names)
country_fert
# fert_add.columns[0]


# ## <font color=red>The above data Frame show the 10 countries with the highest fertility rate in the world from the year 2010 to 2016.   </font> 

# In[ ]:


country_fert.plot(figsize= (10,10), title= '10 countries with highest fertility rate 2000-2016')


# ## Let's compare it to the graph of the world fertility in the same time
# 

# In[ ]:


fertility[column_interest].dropna().plot(kind='line',figsize=(10,10), title = 'World fertility 2000-2016')


# ## If we compare the fertility graphs for the 10 countries we selected compare it the general graph for the rest of the world in 2000 to 2016 frame, we can see that our data for the top 10 countries explains the max peaks for the world. 
# ## Now, lets plot the graphs of populations for the 10 countries vs the rest of the world
# 

# In[ ]:


population[column_interest].plot(figsize=(10,10), title='World population 2000-2016')


# ## Now for the population of the top 10 most fertile countries in the world

# In[ ]:


population = population = pd.read_csv('../input/country_population.csv')
population= population.drop(['Country Code','Indicator Name','Indicator Code'], axis=1)
population['2016'].sum()
world_16 = 78856789486.0

country_pop= population[column_interest].iloc[country_indices]
country_pop['2016'].sum()
ten_16= 414258372.0
country_pop
#population of the 10 countries in 2016 was 414258372.0

idx= 0
country_pop.insert(loc=idx, column='Country Name', value=insert_names)
country_pop.plot(figsize= (10,10), title = 'population of 10 mos fertile countries')


# ## The population of the top 10 most fertile countries in 2016 as a percentage of the world population in 2016

# In[ ]:


percent = np.multiply(np.divide(ten_16,world_16),100)
print('The top 10 most fertile countries in 2016, form ', percent,'%  of the world population in 2016')
# percent 


# ## **Top 10 most fertile countries in the world from the year 2000 to 2016 are listed in no particular order,**
# 
# * Angola
# * Burundi
# * Congo, Dem. Rep.
# * Mali
# * Niger
# * Nigeria
# * Somalia
# * Chad
# * Timor-Leste
# * Uganda

# # <font color= red>Confession time: I had an earlier pre-conceived notion that the most fertile countries would probably have a significant contrinution to the world population. However, after this exploratory analysis, they only form 0.525% of the world population in the year 2016</font>
# 
# # <font color =purple>Another thing worth noticing, is that while the global trend is for the fertility rate is declining, there are pockets of countries, where the fertlity rates are 2 to almost 3 times that of the global mean fertility rate  </font>
