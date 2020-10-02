#!/usr/bin/env python
# coding: utf-8

# # 1.Loading the libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 2.Importing the data

# In[ ]:


suicides=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')


# # 3.Understanding the data

# In[ ]:


suicides.head()


# In[ ]:


suicides.generation.unique() # For finding how many generations there are


# In[ ]:


suicides.columns


# So the columns are:
# 1. country 
# 2. year
# 3. sex 
# 4. age 
# 5. suicides_no = the number of males/ females with age in an interval that committed suicide in the specified year & country. For example, the first row says that 21 males between 15- 24 years from Albania committed suicide in 1987.
# 6. population = total males / females with age in an interval in the specified year & country. For example, the first row says that in Albania in 1987 there were 312900 of males with age between 15-24 years.
# 7. suicides/100k pop= the number of suicides reported to 100,000 people. The formula is : suicides_no * 100,000/population
# 8. country-year
# 9. HDI for year (Human Development Index)= statistic composite index of life expectancy, education, and per capita income indicators, which are used to rank countries into four tiers of human development. A country scores a higher HDI when the lifespan is higher, the education level is higher, and the gross national income (GNI) per capita is higher.
# 10. gdp_for_year (Gross Domestic Product)= measures the value of economic activity within a country. Strictly defined, GDP is the sum of the market values, or prices, of all final goods and services produced in an economy during a period of time (in this case, one year).
# 11. gdp_per_capita= it's a measure of a country's economic output that accounts for its number of people. It divides the country's gross domestic product by its total population. That makes it the best measurement of a country's standard of living. It tells you how prosperous a country is.
# 12. generation: There are 6 generations in our data (I will write the periods for each generation):
#       1. G.I. Generation (or Greatest Generation): 1910-1924 
#       2. Silent: 1925-1945
#       3. Boomers: 1946-1964 
#       4. Generation X:  1965-1979
#       5. Millenials: 1980-1994
#       6. Generation Z: 1995-2012
#            
#  

# # 4.Cleaning the data

# In[ ]:


suicides.shape 


# In[ ]:


suicides.info()


# So there are 27820 entries and 12 features. 
# The type of GDP/year should be int, not string.

# In[ ]:


suicides[' gdp_for_year ($) ']=suicides[' gdp_for_year ($) '].str.replace(',','').astype(int) 
# Because I can't convert the string with the commas in it, I need to remove them and then converting the string.


# In[ ]:


suicides.head()


# In[ ]:


suicides.isnull().any() # Verifying if there are any null values.


# In[ ]:


suicides.isnull().sum() # Counting all the null entries, adding 1 tot the sum everytime it's true that the value is null.


# There are way too many null values for HDI, and also it's not a very important feature for our future analysis. So we're gonna eliminate it.

# In[ ]:


del suicides['HDI for year']


# In[ ]:


suicides.columns # It was successfully deleted.


# How many countries are studied in this dataset?

# In[ ]:


len(suicides.country.unique()) 
# 101 countries.


# I'm gonna change some names of columns for an easier work when I'm gonna use them.

# In[ ]:


suicides.rename(columns={'suicides/100k pop':'suicides/100k',' gdp_for_year ($) ':'gdp/year','gdp_per_capita ($)':'gdp/capita'},inplace=True)
# We already know that the currency is $.


# # 5.Vizualising the data

# In[ ]:


suicides.drop('year',axis=1).describe()


# **Number of suicides per country **

# In[ ]:


suicides.groupby('country').suicides_no.sum().sort_values(ascending=False)
# The country with the most suicides is Russia, followed by USA and Japan
# The last 2 countries have 0 suicides, so we don't need them in our analysis about the number of suicides 


# In[ ]:


suicides=suicides[(suicides.country!='Saint Kitts and Nevis') & (suicides.country!='Dominica')]


# In[ ]:


total_suicides=pd.DataFrame(suicides.groupby('country').suicides_no.sum().sort_values(ascending=False))
# I'm gonna save the results to a DataFrame so I can plot them


# In[ ]:


total_suicides=total_suicides.reset_index()
# I don't want countries as index


# In[ ]:


total_suicides.head()


# In[ ]:


plt.figure(figsize=(10,20))
sns.barplot(y='country',x='suicides_no',data=total_suicides)
plt.title('Number of suicides aprox. 1985-2016') # It's aprox because the years for each country aren't quite the same
plt.ylabel('Countries')
plt.xlabel('Number of suicides')


# I will sectionate the figure because the countries with a small number of suicides aren't visible.

# In[ ]:


plt.figure(figsize=(10,20))
sns.barplot(y=total_suicides[total_suicides.suicides_no<7000].country,
            x=total_suicides[total_suicides.suicides_no<7000].suicides_no,data=total_suicides)
plt.title('Number of suicides aprox. 1985-2016')
plt.ylabel('Countries')
plt.xlabel('Number of suicides')

# I chosed the suicides_no < 7000 just by looking at the figure and trying multiple times another values,
# finding this number appropiate.


# **The evolution of number of suicides globally**

# In[ ]:


from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
output_notebook()

suicides_globally=suicides.groupby('year').sum().suicides_no.values
years=suicides.year.sort_values(ascending=True).unique()
years

p1 = figure(plot_height=400, plot_width=900, title='Evolution of suicides over the world per year', tools='pan,box_zoom')

p1.line(x=years, y=suicides_globally, line_width=2, color='aquamarine')
p1.circle(x=years, y=suicides_globally, size=5, color='green')
show(p1)
    


# We observe that in 2016 the number of suicides is very low, so let's check the number of countries studied  this year.

# In[ ]:


len(suicides[suicides.year==2016].country.unique())
#There are only 16 countries studied this year, so we don't need the value in our analysis.


# In[ ]:


suicides_globally=suicides[suicides.year!=2016].groupby('year').sum().suicides_no.values
years=suicides[suicides.year!=2016].year.sort_values(ascending=True).unique()
p1 = figure(plot_height=400, plot_width=900, title='Evolution of suicides over the world per year', tools='pan,box_zoom')

p1.line(x=years, y=suicides_globally, line_width=2, color='aquamarine')
p1.circle(x=years, y=suicides_globally, size=5, color='green')
show(p1)


# There is a very big increasing of the number of suicides from the year 1988; the peak is in the 1998-2003, then there is a slow decreasing.

# **Number of suicides by gender**

# In[ ]:


suicides_gender=pd.DataFrame(suicides.groupby(['country','sex']).suicides_no.sum())
suicides_gender=suicides_gender.reset_index()
suicides_gender.head()


# In[ ]:


plt.figure(figsize=(10,25))
sns.barplot(x=suicides_gender[suicides_gender.suicides_no > 7000].suicides_no,
            y=suicides_gender[suicides_gender.suicides_no > 7000].country,data=suicides_gender,hue='sex')


# In[ ]:


plt.figure(figsize=(10,25))
sns.barplot(x=suicides_gender[suicides_gender.suicides_no < 7000].suicides_no,
            y=suicides_gender[suicides_gender.suicides_no < 7000].country,data=suicides_gender,hue='sex')


# In[ ]:


suicides.groupby('sex').suicides_no.sum()


# In[ ]:


suicides.groupby('sex').suicides_no.sum().male/suicides.groupby('sex').suicides_no.sum().female * 100


# Men are 3 times more likely to committ suicide than women.

# **Evolution of number of suicides by gender globally**

# In[ ]:


genders=suicides.groupby(['year','sex']).sum().suicides_no
female=genders.loc[:2015,'female'].values
male=genders.loc[:2015,'male'].values


p2=figure(plot_height=500,plot_width=900,title='Evolution of number of suicides by gender globally',tools='pan, box_zoom')
p2.circle(x=years,y=male,color='purple',size=5)
p2.line(x=years,y=male,color='purple',legend='male',line_width=2)
p2.circle(x=years,y=female,color='orange',size=5)
p2.line(x=years,y=female,color='orange',legend='female',line_width=2)
show(p2)


# **Number of suicides by age**

# In[ ]:


suicides.groupby('age').sum().suicides_no


# In[ ]:


plt.figure(figsize=(10,5))
suicides_age=suicides.groupby('age').sum().suicides_no.values
age=suicides.groupby('age').sum().reset_index().age.values
sns.barplot(x=age,y=suicides_age)
plt.title('Number of suicides by age')


# People with age between 35-54 year comitted suicide the most.

# **Evolution of number of suicides by age globally**

# In[ ]:


suicides_age=suicides.groupby(['year','age']).sum().suicides_no
suicides_age


# In[ ]:


age15_24=suicides_age.loc[:2015,'15-24 years'].values
age25_34=suicides_age.loc[:2015,'25-34 years'].values
age35_54=suicides_age.loc[:2015,'35-54 years'].values
age5_14=suicides_age.loc[:2015,'5-14 years'].values
age_above75=suicides_age.loc[:2015,'75+ years'].values


# In[ ]:


p3=figure(plot_height=500,plot_width=1000,title='Evolution of number of suicides by age globally',tools='pan,box_zoom')
p3.circle(x=years,y=age35_54,color='darkorchid',size=5)
p3.circle(x=years,y=age25_34,color='turquoise',size=5)
p3.circle(x=years,y=age15_24,color='limegreen',size=5)
p3.circle(x=years,y=age5_14,color='tomato',size=5)
p3.circle(x=years,y=age_above75,color='blue',size=5)
p3.line(x=years,y=age35_54,color='darkorchid',line_width=2,legend='35-54 years')
p3.line(x=years,y=age25_34,color='turquoise',line_width=2,legend='25-34 years')
p3.line(x=years,y=age15_24,color='limegreen',line_width=2,legend='15-24 years')
p3.line(x=years,y=age5_14,color='tomato',line_width=2,legend='5-14 years')
p3.line(x=years,y=age_above75,color='blue',line_width=2,legend='75+ years')
show(p3)


# **Number of suicides by generation**

# In[ ]:


suicides.groupby('generation').sum().suicides_no


# In[ ]:


plt.figure(figsize=(10,5))
generations=suicides.groupby('generation').sum().suicides_no.reset_index().generation
suicides_gen=suicides.groupby('generation').sum().suicides_no.values
sns.barplot(x=generations,y=suicides_gen)
plt.xlabel('Generation')
plt.title('Number of suicides by generation')


# The Boomers comitted suicide the most.

# Now let's check the correlation between values.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(suicides.corr(),annot=True,cmap="BuPu")


# We can also get some economic conclusions from our data.
# The correlations that we are interested in are :
# 1. gdp/year - population: it's logical that if there are more people in a country GDP will be bigger; 0.71 it's a high positive correlation
# 2. suicides_no - population: more people => more suicides
# 3. gdp/year - suicides: big salaries => more people => more suicides

# In[ ]:


sns.pairplot(suicides.drop('year',axis=1),hue='sex',palette='bright')


# **Identifying countries by population and number of suicides in 2015 with Bokeh**

# In[ ]:


gdp_year=suicides[suicides.year==2015]['gdp/year'].drop_duplicates().values
pop=suicides[suicides.year==2015].groupby('country').sum().population.values
no_suicides=suicides[suicides.year==2015].groupby('country').sum().suicides_no.values
ctry=suicides[suicides.year==2015].groupby('country').sum().reset_index().country.values

s=pd.DataFrame()
s['country']=ctry
s['pop']=pop
s['no_suicides']=no_suicides
s['gdp_year']=gdp_year

s.head()


# In[ ]:


from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
source=ColumnDataSource(s)
p4=figure(plot_height=500,plot_width=900,title='Identifying countries by population, number of suicides and GDP in 2015',tools='pan,box_zoom,wheel_zoom,reset')
p4.diamond(x='pop',y='no_suicides',size=10,color='green',source=source)
p4.add_tools(HoverTool(tooltips=[('population','@pop'),('number suicides','@no_suicides'),('country','@country'),('GDP','@gdp_year')]))
p4.xaxis.axis_label='Population'
p4.yaxis.axis_label='Number of suicides'
p4.xaxis.axis_label_standoff = 30 # Distance of the xaxis label from the xaxis 
show(p4)

