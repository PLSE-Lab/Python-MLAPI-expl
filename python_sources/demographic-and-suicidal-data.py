#!/usr/bin/env python
# coding: utf-8

# # Demographic details

# ![](http://www.dicasecuriosidades.net/wp-content/uploads/2017/02/Global-Population.jpg)

# If I have to categorize the whole data, I would categorize it into 3 part:
# 1. Demographic details.
# 2. Economic details.
# 3. Suicide details.
# I have tried to do the analysis of each part.

# First we will import all the libraries.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Then we will load the dataset.

# In[ ]:


df=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
data=pd.DataFrame(df)
data.head(6)


# Then we will check the shape of the dataset, that is what is the number of rows and columns in the dataset.

# In[ ]:


data.shape


# Now we will check for data types of each variable.

# In[ ]:


data.info()


# Then we will check whether our data has any missing values or not.

# In[ ]:


data.isnull().sum()


# We won't work with the original dataset for safety, we will make a copy of it, and work on it.

# In[ ]:


data_copy=data.copy()


# In[ ]:


data_copy.describe()


# Also we will convert year column into categorical variable.

# In[ ]:


data_copy["year"]=data_copy["year"].astype("str")


# Now we will check whether it is converted or not.

# In[ ]:


data_copy.info()


# Here we will drop two variables from our analysis, first one will be, country-year, since it is the combination of the variables country and year, or we can say it is simply the concatenation of the two, and next we will remove the variable HDI for year, and we will remove this variable because it has many missing values. 

# In[ ]:


data_copy.drop(columns=["country-year","HDI for year"],inplace=True)


# # First we will do univariate analysis using visualization.

# In[ ]:


px.histogram(data_copy,x="country",title="No. of times each country appear in the dataset",color="country",width=1200,height=600)


# In the below plot we are trying to see what is the frequency of different years in the whole dataset, also the year is in categorical format in our dataset, and we will let it be in that format only.

# In[ ]:


descending_order=data_copy["year"].value_counts().sort_values(ascending=False).index
sns.set(style="whitegrid")
graph=sns.catplot(x="year",data=data_copy,kind="count",height=5,aspect=3.2,order=descending_order,palette="Blues_d")
graph.set_xticklabels(rotation=90)


# In[ ]:


px.histogram(data_copy,x="sex",title="Frequency of male and Female in the whole dataset",color="sex",width=1000)


# In[ ]:


px.histogram(data_copy,x="age",title="Frequency of each age category in the dataset",color="age",width=1000)


# Limiting x axis was done intentionally, because most of the values are concentrated within 500 suicides.

# In[ ]:


fig=px.histogram(data_copy,x="suicides_no",nbins=100,width=1000,marginal="box",title="Distribution of suicides numbers using histogram and box plot")
fig.update_xaxes(range=[0,5000])


# Here also we need to limit the x-axis because most of the values are concentrated in the range upto 5 million.

# In[ ]:


fig2=px.histogram(data_copy,x="population",title="Distribution of Population in the whole dataset",width=1000,marginal="box")
fig2.update_xaxes(range=[0,6000000])


# In[ ]:


fig3=px.histogram(data_copy,x="suicides/100k pop",title="Distribution of the varaible suicides/100k pop",width=1000,marginal="box")
fig3.update_xaxes(range=[0,100])


# In[ ]:


px.histogram(data_copy,x="generation",title="Frequency of the variable generation in the dataset", color="generation",width=1000)


# # **Now we will start Multivariate Analysis.**

# # Now we will see how both male and female population is changing over the years for different age groups for different countries.

# **We will start with Japan.**

# In[ ]:


px.bar(data_copy.query("country=='Japan'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,
      title="Population change over the years for different age category in Japan")


# Same we will do for Italy. So we can clearly see from the below bar graph, how Italy's population has changed over the years, also we can notice that population of younger age group people are decreasing, whereas older age group population is on the rise.

# In[ ]:


px.bar(data_copy.query("country=='Italy'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,
      title="Population change over the years for different age category in Italy")


# Now we will do for United States, and we will not do for all the countries since there are in total 101 countries, so doing for each country won't be possible, I am doing it for the countries of my choice.

# In[ ]:


px.bar(data_copy.query("country=='United States'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,
      title="Population change over the years for different age category in United States")


# Now we will do the same for Spain.

# In[ ]:


px.bar(data_copy.query("country=='Spain'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,
      title="Population change over the years for different age category in Spain")


# # Now we will do the same for number of suicides for different countries for different age group, we will start with Japan

# In[ ]:


px.line(data_copy.query("country=='Japan'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="Total number of suicides in Japan for different age category over the years")


# Now we will check for Italy.

# In[ ]:


px.line(data_copy.query("country=='Italy'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="Total number of suicides in Italy for different age category over the years")


# Now we will do for United States.

# In[ ]:


px.line(data_copy.query("country=='United States'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="Total number of suicides in United States for different age category over the years")


# And finally we will do for Spain, we can keep on doing this for all the countries, but since there are 101 countries, so it is not possible, I did for countries of my own choice.

# In[ ]:


px.line(data_copy.query("country=='Spain'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="Total number of suicides in Spain for different age category over the years")


# # Now we will check how suicides/100k population is changing over the years for different countries and we will look for both the genders.

# In[ ]:


px.line(data_copy.query("country=='Japan'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="'Suicides per 100k population' in Japan for different age group over the years")


# In[ ]:


px.line(data_copy.query("country=='Italy'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="'Suicides per 100k population' in Italy for different age group over the years")


# In[ ]:


px.line(data_copy.query("country=='United States'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="'Suicides per 100k population' in United States for different age group over the years")


# In[ ]:


px.line(data_copy.query("country=='Spain'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,
        title="'Suicides per 100k population' in Spain for different age group over the years")


# **Now we will see how over the years GDP per Capita for countries like Japan, Italy, United States, Spain and Colombia has changed over the years or what is the trend for these countries.**

# In[ ]:


px.line(data_copy.query("country==['Japan','Italy','United States','Spain','Colombia']"),x="year",y="gdp_per_capita ($)",width=1000,height=560,color="country",
       title="Comparison of 'GDP per capita' of Colombia, Italy, Japan, Spain and United States for different years")


# Now we will do the same for variable GDP.

# In[ ]:


px.line(data_copy.query("country==['Japan','Italy','United States','Spain','Colombia']"),x="year",y=" gdp_for_year ($) ",width=1000,height=560,color="country",
       title="Comparison of 'GDP' of Colombia, Italy, Japan, Spain and United States for different years")


# That's all for now. Thank you.
