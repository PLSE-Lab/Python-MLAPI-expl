#!/usr/bin/env python
# coding: utf-8

# # Suicide rates

# ___All the outcomes and understandings are written in <font color= green> GREEN</font>___

# In[ ]:


#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. Loading and Cleaning Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loading the data set
df = pd.read_csv("../input/master.csv")
df


# ## 2. Inspect the dataframe
# This helps to give a good idea of the dataframes.

# In[ ]:


# The .info() code gives almost the entire information that needs to be inspected, so let's start from there
df.info()


# In[ ]:


#To get the idea of how the table looks like we can use .head() or .tail() command
df.head()


# In[ ]:


df.tail()


# In[ ]:


# The .shape code gives the no. of rows and columns
df.shape


# In[ ]:


#To get an idea of the numeric values, use .describe()
df.describe()


# ## 3. Cleaning the dataframe

# <font color= green>___From the above .info() it is visible that therethere is only one null value column___</font>

# In[ ]:


df = df.drop(columns = 'HDI for year')
df.info()


# In[ ]:


df.isnull().sum()


# <font color= green>___It's clear that all the null values are gone and we can see if the columns are fir for the analysis___</font>

# In[ ]:


df


# <font color = green>___Changes to be done:<br>
# -Make age bins<br>
# -Remove country-year<br>
# -Change column headings<br>
# -Change the type of 'gdp for year'___</font>

# In[ ]:


df['age'] = df['age'].str.rstrip(' years')
df = df.drop(['country-year'], axis= 1)
df = df.rename(columns={' gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capital'})
df['gdp_for_year'] = df['gdp_for_year'].apply(lambda x: float(x.split()[0].replace(',', '')))
df


# <font color = green>___We will be following EDA (Exploratory Data Analysis) from here.___</font>

# # 4. Univariate Analysis
#  <br />Countplot() will be used in this analysis since we are dealing with categorical value and that gives us the count as the value. <br /> 

#  ##  4.1  'Country' (Unordered Categorical) 

# In[ ]:


# Compaing the different countries
plt.figure(figsize=(20,10))
country = sns.countplot(df['country'],order = df['country'].value_counts().index)
country.tick_params(axis='x', rotation=90)
plt.show()


# <font color = green>___Some of them have same levels of count meanwhile some of the countries have very less count. This means that the data collected has a better idea of a some countries while others, not so much.___</font>

#  ##  4.2  'Years' (Ordered Categorical) 

# In[ ]:


# Compaing the different years
plt.figure(figsize=(20,10))
year = sns.countplot(df['year'],order = df['year'].value_counts().index)
year.tick_params(axis='x', rotation=90)
plt.show()


# <font color = green>___It's clear that there is a gradual decrease in the count through the years but in 2016 the depth is a lot___</font>

#  ##  4.3  'Generation' (Unordered Categorical) 

# In[ ]:


# Compaing the different generations
generation = sns.countplot(df['generation'],order= ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
generation.tick_params(axis='x', rotation=90)
plt.show()


# <font color = green>___Each generations have different levels as visible___</font>

# # 5. Segmented Univariate Analysis
#  We will take the above Variables against the others: <br />
# Countplot() will be used in this analysis since we are dealing with categorical value and that gives us the count as the value. <br /> Multiple colors are given in this to easily differentiate between the different hues since it has importance.

#  ##  5.1  Grouping by 'Suicide Number' 

#  ###  5.1.1 'Country'

# In[ ]:


suicides_no_country = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='country')
suicides_no_country


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(suicides_no_country)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___The number of suicides are extreme in few countries while very little in others.___</font>

#  ###  5.1.2 'Year'

# In[ ]:


suicides_no_year = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='year')
suicides_no_year


# In[ ]:


plt.plot(suicides_no_year)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___The number of suicide was less till 1990s then it increased to a large extend and has had a quick drop in 2016___</font>

#  ###  5.1.3 'Sex'

# In[ ]:


suicides_no_sex = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='sex')
suicides_no_sex


# In[ ]:


plt.plot(suicides_no_sex)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___Men seems to be falling in trap of suicide as compared to a female___</font>

#  ###  5.1.4 'Age'

# In[ ]:


suicides_no_age = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='age')
suicides_no_age


# In[ ]:


plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],suicides_no_age)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___There is a major peak at 25-34 and a huge depth at 35-54 in the number of suicides___</font>

#  ###  5.1.5 'Generation'

# In[ ]:


suicides_no_generation = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='generation')
suicides_no_generation


# In[ ]:


plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],suicides_no_generation)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In number of suicides:<br>- Peak: 'GI generation','Boomers','Generation Z'<br>- Depths: 'Generation X', 'Silent'___</font>

#  ##  5.2  Grouping by 'Population'  

#  ###  5.2.1 'Country'

# In[ ]:


population_country = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='country')
population_country


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(population_country)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In population:<br>- Peak: 'Brazil','Japan','United States'<br>- Depths: Most countries other than these___</font>

#  ###  5.2.2 'Year'

# In[ ]:


population_year = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='year')
population_year


# In[ ]:


plt.plot(population_year)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In population:<br>- Peak: '2010','2002','2000'<br>- Depths: '1985','2016'___</font>

#  ###  5.2.3 'Sex'

# In[ ]:


population_sex = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='sex')
population_sex


# In[ ]:


plt.plot(population_sex)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In population:<br>Females more than males___</font>

#  ###  5.2.4 'Age'

# In[ ]:


population_age = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='age')
population_age


# In[ ]:


plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],population_age)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In population:<br>- Peak: '25-34'<br>- Depths: '75+'___</font>

#  ###  5.2.5 'Generation'

# In[ ]:


population_generation = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='generation')
population_generation


# In[ ]:


plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],population_generation)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In population:<br>- Peak: 'Boomers','GI generation','Millenials'<br>- Depths: 'Silent','Generation X'___</font>

#  ##  5.3  Grouping by 'suicides/100k population' 

#  ###  5.3.1 'Country'

# In[ ]:


suicides_100k_pop_country = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='country')
suicides_100k_pop_country


# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(suicides_100k_pop_country)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In suicides/100k population:<br>- Peak: 'Russian Federation','Hungary','Lithuania'<br>- Depths: There are many countries___</font>

#  ###  5.3.2 'Year'

# In[ ]:


suicides_100k_pop_year = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='year')
suicides_100k_pop_year


# In[ ]:


plt.plot(suicides_100k_pop_year)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In suicides/100k population:<br>- Peak: '1995','1999','2002'<br>- Depths: '2016','1986','1987'___</font>

#  ###  5.3.3 'Sex'

# In[ ]:


suicides_100k_pop_sex = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='sex')
suicides_100k_pop_sex


# In[ ]:


plt.plot(suicides_100k_pop_sex)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In suicides/100k population:<br>Males are more than female___</font>

#  ###  5.3.4 'Age'

# In[ ]:


suicides_100k_pop_age = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='age')
suicides_100k_pop_age


# In[ ]:


plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],suicides_100k_pop_age)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In suicides/100k population:<br>- Peak: '75+','55-74','25-34'<br>- Depths: '35-54','5-14'___</font>

#  ###  5.3.5 'Generation'

# In[ ]:


suicides_100k_pop_generation = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='generation')
suicides_100k_pop_generation


# In[ ]:


plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],suicides_100k_pop_generation)
plt.xticks(rotation=90)
plt.show()


# <font color = green>___In suicides/100k population:<br>- Peak: 'Generation Z','G.I. Generation'<br>- Depths: 'Generation X','Millenials'___</font>

# # Step 6 : Bivariate Analysis <br />
#  We will take the Variables against each other for this step to understand how much each of them are correlated.<br/>Two categorical variables are taken against a numeric variable.<br/> Heatmap() is used to do the following plots as they will give a good view when we relate two variables with each other. <br/> The colour scheme will help us to know the major changes since they are gradient in these heat maps

# <font color = green>___Since the gdp columns are repeated a new df is mad with only unique values.___</font>

# In[ ]:


df_gdp = pd.DataFrame(data = df.gdp_for_year)
df_gdp['gdp_per_capital'] = df['gdp_per_capital']
df_gdp['country'] = df['country']
df_gdp['year'] = df['year']
df_gdp['sex'] = df['sex']
df_gdp['age'] = df['age']
df_gdp['generation'] = df['generation']
df_gdp = df_gdp.drop_duplicates(keep='first')
df_gdp


#  ##  6.1 'Country' & 'Year' with 'Suicide no.'

# In[ ]:


country_year_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'year' ,aggfunc='sum')
country_year_suicides_no.fillna(0, inplace=True)
country_year_suicides_no


# In[ ]:


plt.figure(figsize=(10,50))
sns.heatmap(country_year_suicides_no,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In Number of suicides:<br>- In Unites States there is an increase till 2015 and then it reduced very well in 2016<br>-In Russian Federation and Japan the maximum are at 1996 and 2000 and it has decreased very well in 2016<br>- France Germany and Brazil has a slight increase till 2015 and very little in 2016. <br>- In the rest the rates are very low___</font>

#  ##  6.2 'Country' & 'Year' with 'Population'

# In[ ]:


country_year_population = pd.pivot_table(df, values= 'population',index='country', columns = 'year' ,aggfunc='sum')
country_year_population.fillna(0, inplace=True)
country_year_population 


# In[ ]:


plt.figure(figsize=(10,50))
sns.heatmap(country_year_population ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In Population:<br>- In Unites States, Mexico and Brazil there is an increase till 2015 and then it reduced very well in 2016<br>-In Russian Federation and Japan It's stable till 2015 and it has decreased very well in 2016<br>- France, Germany, UK and Italy is stable till 2015 and very little in 2016. <br>- In the rest the rates are very low and stable___</font>

#  ##  6.3 'Country' & 'Year' with 'Suicides/100k population'

# In[ ]:


country_year_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'year' ,aggfunc='sum')
country_year_suicides_100k_pop.fillna(0, inplace=True)
country_year_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(10,50))
sns.heatmap(country_year_suicides_100k_pop,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>- In Ukraine, Uruguay, Srilanka, Slovakia, Serbia, Russian Federation, New Zeland, Lithunia, Latvia, Kazakhstan, Japan, Hungary, Guyana, Estonia, Croatia, Belarus there is an increase till 2002 and then it reduced very well after that.<br>-In Austia, Belgium, Bulgaria, Croatia, Cuba, Estonia, Finland, France, Germany, Hungary, Latvia reduce gradually.___</font>

#  ##  6.4 'Country' & 'Year' with 'gdp_for_year'

# In[ ]:


country_year_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'year' ,aggfunc='sum')
country_year_gdp_for_year.fillna(0, inplace=True)
country_year_gdp_for_year 


# In[ ]:


plt.figure(figsize=(10,50))
sns.heatmap(country_year_gdp_for_year ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for year':<br>- In Japan, Germany, France there is a increases till 2002 and then it decreases after that.<br>- In Unites States, United Kingdom, Spain, Mexico, Canada, Brazil, Australia increases gradually till 2015 and decreases in 2016___</font>

#  ##  6.5 'Country' & 'Year' with 'gdp_for_capital'

# In[ ]:


country_year_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'year' ,aggfunc='sum')
country_year_gdp_per_capital.fillna(0, inplace=True)
country_year_gdp_per_capital


# In[ ]:


plt.figure(figsize=(10,50))
sns.heatmap(country_year_gdp_per_capital ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for year':<br>- In United States, Unites Kingdom, Switzerland, Singapore, Qatar, Norway, Luxembourg, Japan, Germany, Finland, Denmark, Belgium, Australia increases gradually till 2015 and a quick decrease in 2016<br>- Rest all have stayed same with just few changes___</font>

#  ##  6.6 'Country' & 'Sex' with 'Suicides_no'

# In[ ]:


country_sex_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'sex' ,aggfunc='sum')
country_sex_suicides_no.fillna(0, inplace=True)
country_sex_suicides_no


# In[ ]:


plt.figure(figsize=(2,50))
sns.heatmap(country_sex_suicides_no,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Number of suicides':<br>- In United States, Ukraine, Thailand, Russian Federation, Japan male is more than female.<br>- In rest they are almost similar___</font>

#  ##  6.7 'Country' & 'Sex with 'Population'

# In[ ]:


country_sex_population = pd.pivot_table(df, values= 'population',index='country', columns = 'sex' ,aggfunc='sum')
country_sex_population.fillna(0, inplace=True)
country_sex_population 


# In[ ]:


plt.figure(figsize=(2,50))
sns.heatmap(country_sex_population ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Population':<br>- There isn't much difference between the two genders___</font>

#  ##  6.8 'Country' & 'Sex' with 'Suicides/100k population'

# In[ ]:


country_sex_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'sex' ,aggfunc='sum')
country_sex_suicides_100k_pop.fillna(0, inplace=True)
country_sex_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(2,50))
sns.heatmap(country_sex_suicides_100k_pop,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>- In Argentina, Austalia, Austria, Belarus, Finland, France, Hungary, Kazakhstan, Lithuania, Republic of Korea, Russian Federation, Ukraine male is more than female.<br>- In rest they are almost similar___</font>

#  ##  6.9 'Country' & 'Sex' with 'gdp_for_year'

# In[ ]:


country_sex_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'sex' ,aggfunc='sum')
country_sex_gdp_for_year.fillna(0, inplace=True)
country_sex_gdp_for_year 


# In[ ]:


plt.figure(figsize=(2,50))
sns.heatmap(country_sex_gdp_for_year ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for year':<br>There is no difference between the genders___</font>

#  ##  6.10 'Country' & 'Sex' with 'gdp_for_capital'

# In[ ]:


country_sex_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'sex' ,aggfunc='sum')
country_sex_gdp_per_capital.fillna(0, inplace=True)
country_sex_gdp_per_capital


# In[ ]:


plt.figure(figsize=(2,50))
sns.heatmap(country_sex_gdp_per_capital ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for year':<br>There is no difference between the genders___</font>

#  ##  6.11 'Country' & 'Age' with 'Suicides No.'

# In[ ]:


country_age_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'age' ,aggfunc='sum')
country_age_suicides_no.fillna(0, inplace=True)
country_age_suicides_no


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Number of Suicides':<br>- In United States, United Kingdom, Ukraine, Russian Federation, Japan increases till 25-34 then decreases.<br>- In rest they are almost similar___</font>

#  ##  6.12 'Country' & 'Age' with 'Population'

# In[ ]:


country_age_population = pd.pivot_table(df, values= 'population',index='country', columns = 'age' ,aggfunc='sum')
country_age_population.fillna(0, inplace=True)
country_age_population 


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Population':<br>- In United States, Russian Federation, Mexico, Japan, Brazil increases till 25-34 then decreases.<br>- In rest they are almost similar___</font>

#  ##  6.13 'Country' & 'Age' with 'Suicides/100k population'

# In[ ]:


country_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'age' ,aggfunc='sum')
country_age_suicides_100k_pop.fillna(0, inplace=True)
country_age_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>- It gradually increases in all the countires___</font>

#  ##  6.16 'Country' & 'Generation' with 'Suicide No.'

# In[ ]:


country_generation_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'generation' ,aggfunc='sum')
country_generation_suicides_no.fillna(0, inplace=True)
country_generation_suicides_no


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Number of Suicide':<br>- Peak is among the GI generation for all the countries<br>- The lowest is in Generation X for all the countries___</font> 

#  ##  6.17 'Country' & 'Generation' with 'Population'

# In[ ]:


country_generation_population = pd.pivot_table(df, values= 'population',index='country', columns = 'generation' ,aggfunc='sum')
country_generation_population.fillna(0, inplace=True)
country_generation_population 


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Population':<br>- Peak is among the GI generation and Boomers for all the countries<br>- The lowest is in Generation X and silent for all the countries___</font> 

#  ##  6.18 'Country' & 'Generation' with 'suicides/100k population'

# In[ ]:


country_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'generation' ,aggfunc='sum')
country_generation_suicides_100k_pop.fillna(0, inplace=True)
country_generation_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'suicides/100k population':<br>- Peak is among the generation Z and silent for all the countries<br>- The lowest is in Generation X for all the countries___</font> 

#  ##  6.19 'Country' & 'Generation' with 'gdp_for_year'

# In[ ]:


country_generation_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'generation' ,aggfunc='sum')
country_generation_gdp_for_year.fillna(0, inplace=True)
country_generation_gdp_for_year 


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_generation_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'gdp for year':<br>- Generation Z, Millenials have the most nad rest the least in the others in few countries <br>- In rest of the countries it's almost similar through out all generations___</font> 

#  ##  6.20 'Country' & 'Generation' with 'gdp_for_capital'

# In[ ]:


country_generation_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'generation' ,aggfunc='sum')
country_generation_gdp_per_capital.fillna(0, inplace=True)
country_generation_gdp_per_capital


# In[ ]:


plt.figure(figsize=(6,50))
sns.heatmap(country_generation_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'gdp for capital':<br>- In countries like United States, United Kingdom, Switzerland, SingaporeQatar, Norway, Luxembourg - Generation Z, Millenials have the most nad rest the least in the others in few countries <br>- In rest of the countries it's almost similar through out all generations___</font> 

#  ##  6.21 'Year' & 'Sex' with 'Suicide No.'

# In[ ]:


year_sex_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='year', columns = 'sex' ,aggfunc='sum')
year_sex_suicides_no.fillna(0, inplace=True)
year_sex_suicides_no


# In[ ]:


plt.figure(figsize=(4,10))
sns.heatmap(year_sex_suicides_no,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Number of suicides':<br>- Male suicide has always been high always.<br>- There is a quick reduction in 2016___</font> 

#  ##  6.22 'Year' & 'Sex' with 'Population'

# In[ ]:


year_sex_population = pd.pivot_table(df, values= 'population',index='year', columns = 'sex' ,aggfunc='sum')
year_sex_population.fillna(0, inplace=True)
year_sex_population 


# In[ ]:


plt.figure(figsize=(4,10))
sns.heatmap(year_sex_population ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Population':<br>- Female is more than male<br>- There is a quick reduction in 2016___</font> 

#  ##  6.23 'Year' & 'Sex' with 'Suicides/100k population'

# In[ ]:


year_sex_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'sex' ,aggfunc='sum')
year_sex_suicides_100k_pop.fillna(0, inplace=True)
year_sex_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(4,10))
sns.heatmap(year_sex_suicides_100k_pop,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>- Male is very high compared to females<br>- There is a quick reduction in 2016___</font> 

#  ##  6.24 'Year' & 'Sex' with 'gdp_for_year'

# In[ ]:


year_sex_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'sex' ,aggfunc='sum')
year_sex_gdp_for_year.fillna(0, inplace=True)
year_sex_gdp_for_year 


# In[ ]:


plt.figure(figsize=(4,10))
sns.heatmap(year_sex_gdp_for_year ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for year':<br>-It's the same for male and female___</font> 

#  ##  6.25 'Year' & 'Sex' with 'gdp_for_capital'

# In[ ]:


year_sex_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'sex' ,aggfunc='sum')
year_sex_gdp_per_capital.fillna(0, inplace=True)
year_sex_gdp_per_capital


# In[ ]:


plt.figure(figsize=(4,10))
sns.heatmap(year_sex_gdp_per_capital ,cmap= sns.cubehelix_palette(200))
plt.show()


# <font color = green>___In 'gdp for capital':<br>-It's the same for male and female___</font> 

#  ##  6.26 'Year' & 'Age' with 'Suicides No.'

# In[ ]:


year_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'year', columns = 'age' ,aggfunc='sum')
year_age_suicides_no.fillna(0, inplace=True)
year_age_suicides_no


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Number of Suicide':<br>- Peak is at 25-34 for all the years<br>- The lowest is in 35-54 in all the years___</font> 

#  ##  6.27 'Year' & 'Age' with 'Population'

# In[ ]:


year_age_population = pd.pivot_table(df, values= 'population',index='year', columns = 'age' ,aggfunc='sum')
year_age_population.fillna(0, inplace=True)
year_age_population 


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Population':<br>- Peak is at 25-34 in all the years<br>- The lowest is in 75+ in all the years___</font> 

#  ##  6.28 'Year' & 'Age' with 'Suicides/100k population'

# In[ ]:


year_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'age' ,aggfunc='sum')
year_age_suicides_100k_pop.fillna(0, inplace=True)
year_age_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>- Peak is at 75+ in all the years<br>- The lowest is in 35-54 in all the years___</font> 

#  ##  6.29 'Year' & 'Age' with 'gdp_for_year'

# In[ ]:


year_age_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'age' ,aggfunc='sum')
year_age_gdp_for_year.fillna(0, inplace=True)
year_age_gdp_for_year 


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_age_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'gdp for year':<br>- It's almost the same through out the ages every year.___</font> 

#  ##  6.30 'Year' & 'Age' with 'gdp_for_capital'

# In[ ]:


year_age_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'age' ,aggfunc='sum')
year_age_gdp_per_capital.fillna(0, inplace=True)
year_age_gdp_per_capital


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_age_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'gdp for capital':<br>- It's almost the same through out the ages every year.___</font> 

#  ##  6.31 'Year' & 'Generation' with 'Suicides No.'

# In[ ]:


year_generation_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'year', columns = 'generation' ,aggfunc='sum')
year_generation_suicides_no.fillna(0, inplace=True)
year_generation_suicides_no


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Number of Suicides':<br>- Generation Z, Millenials, Boomers, GI Generation have the most and rest the least in every year.<br>- In 2016 they are all similar___</font> 

#  ##  6.32 'Year' & 'Generation' with 'Population'

# In[ ]:


year_generation_population = pd.pivot_table(df, values= 'population',index='year', columns = 'generation' ,aggfunc='sum')
year_generation_population.fillna(0, inplace=True)
year_generation_population 


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Population':<br>- GI Generation, Boomers, Millenials have the most and the others least in almost every year<br>- In 2016 they are all similar___</font> 

#  ##  6.33 'Year' & 'Generation' with 'Suicides/100k population'

# In[ ]:


year_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'generation' ,aggfunc='sum')
year_generation_suicides_100k_pop.fillna(0, inplace=True)
year_generation_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Suicides/100k population':<br>-very few in Generation X<br>- From 1985 to 2000 there were a lot of suicides from the silent generation. <br>- From 20001 to 2010 there is a huge amount in Generation Z which then gradually reduces___</font> 

#  ##  6.34 'Year' & 'Generation' with 'gdp_for_year'

# In[ ]:


year_generation_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'generation' ,aggfunc='sum')
year_generation_gdp_for_year.fillna(0, inplace=True)
year_generation_gdp_for_year 


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_generation_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'gdp for year':<br>-very less in Silent<br>- From 1985 to 2010 among boomers kept on increasing then <br>- From 2001 to 2010 there is an increase gradually for Generation Z which then gradually reduces<br>- From 2011-2015 there is a huge spike among the millenials___</font> 

#  ##  6.35 'Year' & 'Generation' with 'gdp_for_capital'

# In[ ]:


year_generation_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'generation' ,aggfunc='sum')
year_generation_gdp_per_capital.fillna(0, inplace=True)
year_generation_gdp_per_capital


# In[ ]:


plt.figure(figsize=(6,10))
sns.heatmap(year_generation_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'gdp for capital':<br>-very less in Silent <br>- From 2001 to 2010 there is an increase gradually for Generation Z which then gradually reduces<br>- From 2011-2015 there is a huge spike among the millenials and later on it gradually decreases<br>- in 2010 Boomers have a huge spike although rest of the years there isn't much difference___</font> 

#  ##  6.36 'Sex' & 'Age' with 'Suicides No.'

# In[ ]:


sex_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'sex', columns = 'age' ,aggfunc='sum')
sex_age_suicides_no.fillna(0, inplace=True)
sex_age_suicides_no


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Number of Suicides':<br>-Male has always been more than females<br>- There is a huge spike among 25-34 maeles___</font> 

#  ##  6.37 'Sex' & 'Age' with 'Population'

# In[ ]:


sex_age_population = pd.pivot_table(df, values= 'population',index='sex', columns = 'age' ,aggfunc='sum')
sex_age_population.fillna(0, inplace=True)
sex_age_population 


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'Population ':<br>-It is almost same in all ages among the genders<br>- There is a huge spike among 25-34 and a huge reduction among 75+___</font> 

#  ##  6.38 'Sex' & 'Age' with 'suicides/100k Population'

# In[ ]:


sex_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='sex', columns = 'age' ,aggfunc='sum')
sex_age_suicides_100k_pop.fillna(0, inplace=True)
sex_age_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])
plt.show()


# <font color = green>___In 'suicides/100k Population':<br>-Male is more than female in all the cases except 35-54___</font> 

#  ##  6.39 'Sex' & 'Generation' with 'Suicide No.'

# In[ ]:


sex_generation_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'sex', columns = 'generation' ,aggfunc='sum')
sex_generation_suicides_no.fillna(0, inplace=True)
sex_generation_suicides_no


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Number of Suicide':<br>- Number of GI Generation, Boomers, Generation Z male is very much higher than females<br>- In all the rest it's almost similar___</font> 

#  ##  6.40 'Sex' & 'Generation' with 'Population'

# In[ ]:


sex_generation_population = pd.pivot_table(df, values= 'population',index='sex', columns = 'generation' ,aggfunc='sum')
sex_generation_population.fillna(0, inplace=True)
sex_generation_population 


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Population':<br>- Number of Generation X female is higher than male<br>- In all the rest it's almost similar___</font> 

#  ##  6.41 'Sex' & 'Generation' with 'Suicides/100k Population'

# In[ ]:


sex_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='sex', columns = 'generation' ,aggfunc='sum')
sex_generation_suicides_100k_pop.fillna(0, inplace=True)
sex_generation_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(5,2))
sns.heatmap(sex_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Suicides/100k Population':<br>- Males are more than females in all except Generation X.___</font> 

#  ##  6.42 'Age' & 'Generation' with 'Suicides No.'

# In[ ]:


generation_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'generation', columns = 'age' ,aggfunc='sum')
generation_age_suicides_no.fillna(0, inplace=True)
generation_age_suicides_no


# In[ ]:


plt.figure(figsize=(7,3))
sns.heatmap(generation_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Number of Suicides':<br>- GI generation and 25-34 and Generation Z and 55-74 has maximum.<br>- Rest are almost similar___</font>

#  ##  6.43 'Age' & 'Generation' with 'Population'

# In[ ]:


generation_age_population = pd.pivot_table(df, values= 'population',index='generation', columns = 'age' ,aggfunc='sum')
generation_age_population.fillna(0, inplace=True)
generation_age_population 


# In[ ]:


plt.figure(figsize=(7,3))
sns.heatmap(generation_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Population':<br>- GI generation and 25-34 and Generation Z and 55-74 has maximum.<br>- Rest are almost similar___</font>

#  ##  6.44 'Age' & 'Generation' with 'Suicides/100k Population'

# In[ ]:


generation_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='generation', columns = 'age' ,aggfunc='sum')
generation_age_suicides_100k_pop.fillna(0, inplace=True)
generation_age_suicides_100k_pop


# In[ ]:


plt.figure(figsize=(7,3))
sns.heatmap(generation_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.show()


# <font color = green>___In 'Suicides/100k Population':<br>- GI generation and 25-34, Generation Z and 55-74 and 75+ and Silent and 75+ has maximum.<br>- Rest are almost similar___</font>
