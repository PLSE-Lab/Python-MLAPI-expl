#!/usr/bin/env python
# coding: utf-8

# # Kiva Data Exploration

# 1. [Introduction](#1)
# 2. [Data Overview](#2)
# 3. [Borrower Demographics](#3)
# 4. [Loan Distribution](#4)
# 5. [Feature Correlation](#5)
# 6. [Conclusion](#6)

# <a id="1"></a> <br>
# ## Introduction

# Kiva is a non-profit that lets people lend money to low-income students and entrepreneurs around the world. Lenders can help to fund a requested loan for projects such as buying supplies for a store, starting a farm, or installing solar panels. The borrowers can request as much as they want, and lenders can contribute as much as they would like to this loan. 
# 
# This notebook tackles [Kiva's Data Science for Good Kaggle Challenge](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding/home) (which is already closed). They provide a dataset with information from over 600,000 loans from the past two years, including the loan amount, data, location, etc. The goal is to "help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loan". 

# <a id="2"></a> <br>
# ## Data Overview

# First, let's import the necessary libraries and datasets. We are using the provided Kiva data along with additional external datasets for more insight. 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import collections
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))


# In[ ]:


df_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df_mpi = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
mpi_national = pd.read_csv("../input/mpi/MPI_national.csv")
world_countries = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")
country_stats = pd.read_csv("../input/additional-kiva-snapshot/country_stats.csv")


# Let's take a look at the loan data!

# In[ ]:


df_loans.head()


# In[ ]:


df_loans.shape


# Number of recorded loans is **671,205**. 

# In[ ]:


df_loans.describe()


# In[ ]:


df_loans.isna().sum().sort_values(ascending=False)


# In[ ]:


df_mpi.head()


# <a id="3"></a> <br>
# ## Borrower Demographics

# In[ ]:


genders = np.array(df_loans['borrower_genders'])

genders_updated = []

for gender in genders:
    if(type(gender)==str):
        gender = gender.replace(',', '').replace("'", '').replace("[", '').replace("]", '').split(' ')
        for x in gender:
            genders_updated.append(x)

borrower_genders = collections.Counter(genders_updated)

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
the_grid = GridSpec(3, 1)

plt.figure(figsize=(7,7))
plt.pie(borrower_genders.values(), labels=borrower_genders.keys(), autopct='%1.1f%%',)
plt.title('Borrower genders', fontsize=16)


plt.show()


# After parsing the data in the "borrower_genders" column, we can see that the **majority of borrowers are female**. Unfortunately, this is the only information we receive about the borrower (apart from their location). It would be useful to have more information such as income, health, family size etc., for measuring poverty level. 

# <a id="4"></a> <br>
# ## Loan Distribution

# ### Loan amount distribution 

# In[ ]:


fig = plt.figure(figsize=(14,8))
plt.xticks(np.arange(0, max(df_loans['loan_amount'])+1, 5000), rotation = 45)
g = sns.distplot(df_loans['loan_amount'], norm_hist=False)
g.plot()


# Most loans are under \$5,000, but the range of this plot shows that there are some outliers. What / where are these loans for?

# In[ ]:


df_loans[df_loans['loan_amount'] > 50000]


# It looks like there is only one loan greater than $50,000 - $100,000 for an Agriculture-related project in Haiti. A little research showed that this loan was for a Haitian becuty company called [Kreyol Essence](https://kreyolessence.com/). The [loan](https://www.kiva.org/lend/722883) was used to help create 300 new jobs in rural Haiti for their new castor oil brand. This involved hiring farmers to grow the castor plants, and hiring women to extract the castor oil.

# In[ ]:


df_loans['loan_amount'].sum()


# The total loan amount requested in this dataset is **\$565,421,150!** However in total Kiva has given over \$1,000,000,000 in loans! 

# ### Loan Distribution by Sector

# In[ ]:


sectors = df_loans['sector'].value_counts().reset_index()
sectors = sectors.rename(columns={'index': 'sector', 'sector': 'Loan Count'})

sectors_funded = df_loans.groupby('sector').sum()['loan_amount'].reset_index().sort_values('loan_amount', ascending=False)


# In[ ]:


sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

sectors_funded.plot.bar(x='sector', y='loan_amount',color='plum',ax=ax,width=width, position=0)
sectors.plot.bar(x='sector', y='Loan Count', color='mediumaquamarine', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loan Amount and Loan Count by Sector',fontsize=16)


# This plot shows the total amount lent for individual sectors along with the number of loans for each sector. **Agriculture, Food, and Retail** are the top 3 sectors for both loan amount and number of loans.  

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(x='loan_amount',y='sector', data=df_loans, showfliers=False)
plt.title('Loan amount by sector', fontsize=16)


# In[ ]:


df = df_loans[df_loans['country'].isin(['Philippines', 'Kenya', 'El Salvador', 'Cambodia', 'Pakistan', 'Peru'])]
g = sns.FacetGrid(df, col="country", col_wrap=2)
g.set_xticklabels(rotation=90)
g.map(sns.countplot, 'sector')


# In[ ]:


plt.figure(figsize=(12,8))
activities = df_loans['activity'].value_counts().head(40)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.title("Loan Count by Activity (top 40)", fontsize=16)
plt.xlabel("Loan Count", fontsize=15)
plt.ylabel("Activity", fontsize=15)


# Here, we see the loan count by Activity. The top activity is Farming, which aligns with the top sector (Agriculture). Next we see General Store, again likely explaining the next highest sector (Food), or the third (Retail). 

# ### Loan Distribution by Country

# In[ ]:


countries = df_loans['country'].value_counts().reset_index()
countries = countries.rename(columns={'index': 'country', 'country': 'Loan Count'})

countries_total_funded = df_loans.groupby('country').sum()['loan_amount'].reset_index().sort_values('loan_amount', ascending=False)
countries_total_funded = countries_total_funded.rename(columns={'loan_amount':'Total Amount Loaned'})


# In[ ]:


countries_merged = countries.merge(countries_total_funded, on='country')


# In[ ]:


sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

countries_merged.head(30).plot.bar(x='country', y='Total Amount Loaned',color='plum',ax=ax,width=width, position=0)
countries_merged.head(30).plot.bar(x='country', y='Loan Count', color='mediumaquamarine', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loan Amount and Loan Count by Country (top 30)',fontsize=16)


# **Phillippines, Kenya and El Salvador** are the top three countries for number of loans. The top three for amount funded are **Philippenes, Kenya, and the United States**. This is interesting because the USA only ranks 26th in terms of number of loans. They must be getting loans of higher value. 

# In[ ]:


top_countries = countries_total_funded.head(20)['country']

loan_range=df_loans[(df_loans['country'].isin(top_countries))]
plt.figure(figsize=(12,8))
ax = sns.boxplot(x='loan_amount',y='country', data=loan_range, showfliers=False)
plt.title('Loan amount by country', fontsize=16)


# Boxplot of loan amount distribution with outliers removed. It seems like the US has the highest range of loan amounts by far. Maybe it's because wealthy donors from the USA are more likely to give money to a project they are familiar with from their own lives. 

# In[ ]:


df_loans['region_country'] = df_loans['region'] + ', ' + df_loans['country']

plt.figure(figsize=(12,8))
regions = df_loans['region_country'].value_counts().head(30)
sns.barplot(y=regions.index, x=regions.values, alpha=0.6)
plt.title("Loan Count by Region (top 30)", fontsize=16)
plt.xlabel("Loan Count", fontsize=15)
plt.ylabel("Region", fontsize=15)


# ### Average Loan Amount by Country

# In[ ]:


country_avg_loan = df_loans.groupby('country')['loan_amount'].mean().reset_index()
country_avg_loan = country_avg_loan.rename(columns={'loan_amount': 'avg_loan_amount'}).sort_values('avg_loan_amount', ascending=False)
country_avg_loan.head()


# Top countries for average loan amount are **Cote D'Ivoire and Mauritania**. Note that these countries have only received one loan each. 

# In[ ]:


df_loans.groupby('country').agg({'loan_amount': 'mean'}).sort_values('loan_amount', ascending=False).tail(10)


# In[ ]:


top_countries_df = df_loans[df_loans['country'].isin(top_countries)]
top_countries_df.groupby('country').agg({'loan_amount': 'mean'}).sort_values('loan_amount', ascending=False)


# The United States has the highest average loan amount among the top loan countries. Phillipenes has the lowest, but are still the highest receivers due to loan frequency

# ### Loan term length and loan amount

# In[ ]:


plt.figure(figsize=(12,8))
plt.title('Term in Months vs. Loan Amount', fontsize=16)
sns.scatterplot(x=df_loans['loan_amount'], y=df_loans['term_in_months'])

# df_loans[df_loans['loan_amount'] < 4000].sample(100).plot.scatter(x='loan_amount', y='term_in_months', title='Term in Months vs Loan Amount')


# Doesn't seem like there's a correlation - not sure what term in months means. 

# In[ ]:


region_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
region_themes[region_themes['region'] == 'Kaduna']


# In[ ]:


region_themes['number'].sum()


# ### Loans vs MPI

# MPI stands for **Multidimensional Poverty Index**. MPI is a poverty indicator calculated across 10 indicators within 3 dimensions of poverty - Health (nutrition and child mortality), Education (years of schooling and school attendance), and Living Standards (cooking fuel, sanitation, drinking water, electricity, housing, and assets). If someone is experiencing at least three of the indicators, they are classified as MPI poor, and their poverty depends on the number of deprivations. Calculating poverty across a variety of dimensions gives us a more comprehensive picture than just using one indicator such as income. 

# In[ ]:


countries_mpi = countries_total_funded.merge(mpi_national, left_on='country', right_on='Country')
countries_mpi = countries_mpi.sort_values('MPI Urban', ascending=False)
countries_mpi = countries_mpi.drop(['Country'], axis=1)


# In[ ]:


data = [ dict(
        type = 'choropleth',
        locationmode='country names',
        locations=countries_mpi['country'],
        z=countries_mpi['MPI Rural']
      ) ]

layout = dict(
    title = 'Rural MPI'
)

fig = dict( data=data, layout=layout )
iplot( fig )


# In[ ]:


data = [ dict(
        type = 'choropleth',
        locationmode='country names',
        locations=countries_mpi['country'],
        z=countries_mpi['MPI Urban']
      ) ]

layout = dict(
    title = 'Urban MPI'
)

fig = dict( data=data, layout=layout )
iplot( fig )


# Can see that usually the Urban MPI is higher than the Rural MPI - expected - in more densely populated areas there might be better access to necessary resources such as water and shelter. 

# In[ ]:


data = [ dict(
        type = 'scattergeo',
        locationmode = 'ISO-3',
        lon = df_mpi['lon'],
        lat = df_mpi['lat'],
        text = df_mpi['region'] + ', ' + df_mpi['country'] + ': ' + df_mpi['MPI'].astype(str),
        mode = 'markers',
        marker = dict(
            size = 7,
            opacity = 0.8,
            color = df_mpi['MPI'],
            symbol = 'square',
            cmin = 0,
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'MPI by Region',
        colorbar = True,
        geo = dict(
            scope='world',
            #projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
fig = dict( data=data, layout=layout )


iplot( fig, validate=False, filename='d3-loan-map' )


# Note that there are some mistakes in the lat/lon data. Apparently East China is actually an Iowan town called Waterloo. As expected, many of the regions with the lowest MPI are in Sub-Saharan Africa. 

# In[ ]:


sns.scatterplot(x=countries_mpi['MPI Rural'], y=countries_mpi['Total Amount Loaned'])


# It seems like there isn't a high correlation between MPI Rural and Total Amount Loaned. This seems strange - shouldn't countries with lower MPI receive more money if the goal is to lift people out of poverty? Well, here are a few reasons this might be occuring.
# * Countries in with extremely high MPIs might be conflict zones, preventing Kiva or Kiva partners from working with borrowers in that country
# * Somebody living in extreme poverty might not be in the position to request a loan. Without access to basic necessities such as food and water, it could be difficult to come up with a process for repaying a loan
# * It's possible that some of these countries have a very low population. Maybe it's better to lend to countries with more people living under the poverty line - "Liftin One, Lifting Many". So let's take a look at population. 

# ### Amount Loaned vs Population

# In[ ]:


# country_stats.head()
country_stats = country_stats.rename(columns={'country_name': 'country'})
country_stats = country_stats.merge(countries_total_funded, on = 'country')


# In[ ]:


country_stats[country_stats['country'] == 'Afghanistan']


# In[ ]:


x = country_stats[country_stats['population'] < 1000000000]
sns.scatterplot(x=x['population'], y=x['Total Amount Loaned'])


# In[ ]:


country_stats['Loan Amount per Capita'] = country_stats['Total Amount Loaned'] / country_stats['population']
country_stats = country_stats.sort_values('Loan Amount per Capita', ascending = False)


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(y=country_stats.head(20)['Loan Amount per Capita'], x=country_stats.head(20)['country'], alpha=0.6)
plt.title("Loan amount per Capita", fontsize=16)
plt.xlabel("Amount per Capita", fontsize=15)
plt.ylabel("Country", fontsize=15)
plt.xticks(rotation = 45)


# Population below poverty line vs. amount per capita

# In[ ]:


country_stats['number_below_poverty_line'] = country_stats['population_below_poverty_line'] * 0.01 * country_stats['population']
country_stats


# In[ ]:


plt.figure(figsize=(12,8))
country_stats = country_stats.sort_values('number_below_poverty_line', ascending=False)
sns.barplot(y=country_stats.head(20)['number_below_poverty_line'], x=country_stats.head(20)['country'], alpha=0.6)
plt.title("Citizens Below Poverty Line", fontsize=16)
plt.xlabel("Country", fontsize=15)
plt.ylabel("Citizens Below Poverty Line", fontsize=15)
plt.xticks(rotation = 45)


# In[ ]:


sns.set()
fig = plt.figure(figsize=(15,8)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .4

country_stats.head(30).plot.bar(x='country', y='number_below_poverty_line',color='plum',ax=ax,width=width, position=0,alpha=0.7)
country_stats.head(30).plot.bar(x='country', y='Total Amount Loaned', color='mediumaquamarine', ax=ax2,width = width,position=1,alpha=0.7)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Total Loan Amount')
ax2.set_ylabel('Number of Loans')

ax.set_title('Loans per capita and people under poverty line',fontsize=16)


# Interestingly, total amount loaned doesn't seem to relate to the number of people living under the poverty line. India has the most under the poverty line by a wide margin, but does not have the highest amount loaned. Depends on severity of poverty. People choose where to lend their money.. 

# <a id="6"></a> <br>
# ## Feature Correlation

# In[ ]:


fig = plt.figure(figsize=(10,8))

sns.heatmap(df_loans.corr(), annot=True)


# * Loan amount and lender count are highly correlated. This makes sense - a larger loan takes more lenders to fulfill. 
# * Term in months and loam amount are slightly correlated. A larger loan amount likely takes longer to repay.
# * Loan amount and funded amount are highly correlated, but not equal. This implies that some loan requests are not completely funded. 

# ## Better Welfare Model 

# After doing an analysis of the data, a suggestion for a better model would be adding information about a borrower's financial situation to the welfare estimation, rather than just using MPI. Although MPI is a good indicator, income is also important in estimating welfare. 

# 

# 

# 

# 

# 
