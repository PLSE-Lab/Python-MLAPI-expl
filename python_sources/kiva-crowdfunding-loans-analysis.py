#!/usr/bin/env python
# coding: utf-8

# # Kiva Crowdfunding Loans Analysis
# 
# ### Table of contents:
# * [First look at the data set](#first)
# * [Looking deeper](#second)
#     * [How loans are distributed](#sec-1)
#     * [How money is distributed](#sec-2)
# * [Global analyses](#third)
# * [Country specific analyses](#fourth)
#     * [Philippines](#four-1)
# 
# Note: This notebook is under construction. More analyses will be added in the next weeks.

# ## First look at the data set <a class="anchor" id="first"></a>
# 
# We start by loading and examining the data provided in the challenge.

# In[202]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import plotly
from plotly.graph_objs import Scatter, Figure, Layout

plotly.offline.init_notebook_mode(connected=True)

print(os.listdir("../input/"))

loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
data_regions = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
themes_regions = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

hdi = pd.read_csv("../input/human-development/human_development.csv")
mpi = pd.read_csv("../input/human-development/multidimensional_poverty.csv")


# In[4]:


# look at the data
print(loans.shape)
loans.sample(3)


# In[5]:


#checking for missing data
loans.isnull().sum()


# In[6]:


print(data_regions.shape)
data_regions.sample(5)


# In[7]:


data_regions.isnull().sum()


# In[8]:


print(themes.shape)
themes.sample(5)


# In[9]:


themes.isnull().sum()


# In[10]:


print(themes_regions.shape)
themes_regions.sample(5)


# In[11]:


themes_regions.isnull().sum()


# In[12]:


print(hdi.shape)
hdi.sample(5)


# In[ ]:


print(mpi.shape)
mpi.sample(5)


# ## Looking deeper <a class="anchor" id="second"></a>
# 
# Here we will look at some simple data distributions, which will give more insight about the data we have.
# 
# Note: the Kiva website says that in most cases, even if a loan is not funded, the person still receives the money. So, here, due to the very large amount of data, I will consider that every loan asked was received.

# ### How loans are distributed <a class="anchor" id="sec-1"></a>
# 
# Let's start by looking at how the quantity of loans is distributed according to different parameters.

# The first plot shows the distribution of loans according to different sectors of economy. By far, the most common are agriculture, food and retail.

# In[57]:


# select only the loans that were funded
funded_loans = loans[loans['funded_time'].isnull()==False]
funded_loans.isnull().sum()


# In[50]:


plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['sector'].where(loans['funded_time'].isnull()==False), order=loans['sector'].value_counts().index)
plt.title("Sectors which received loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Sector', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# Within these sectors, there are many activities. The next graph explores how the loans are distributed in relation to them. Again, by far, we see most loans being given to farming and general stores, going back to the top three sectors we had seen before. In general, that suggests people who receive loans are mostly in need of food and/or basic income.

# In[14]:


plt.figure(figsize=(14,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['activity'], order=loans['activity'].value_counts().index)
plt.title("Activities which received loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Activities', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# Now a more detailed look at how the sectors are divided into activities and how the loans are distributed among them.

# In[15]:


plt.figure(figsize=(14,40))
plt.subplot(821)
agric_act = loans['activity'].where(loans['sector'] == "Agriculture")
sns.countplot(y=agric_act, order=agric_act.value_counts().iloc[0:10].index)
plt.title("Agriculture", fontsize=16)
plt.subplot(822)
food_act = loans['activity'].where(loans['sector'] == "Food")
sns.countplot(y=food_act, order=food_act.value_counts().iloc[0:10].index)
plt.title("Food", fontsize=16)
plt.subplot(823)
retl_act = loans['activity'].where(loans['sector'] == "Retail")
sns.countplot(y=retl_act, order=retl_act.value_counts().iloc[0:10].index)
plt.title("Retail", fontsize=16)
plt.subplot(824)
serv_act = loans['activity'].where(loans['sector'] == "Services")
sns.countplot(y=serv_act, order=serv_act.value_counts().iloc[0:10].index)
plt.title("Services", fontsize=16)
plt.subplot(825)
pruse_act = loans['activity'].where(loans['sector'] == "Personal Use")
sns.countplot(y=pruse_act, order=pruse_act.value_counts().iloc[0:10].index)
plt.title("Personal Use", fontsize=16)
plt.subplot(826)
house_act = loans['activity'].where(loans['sector'] == "Housing")
sns.countplot(y=house_act, order=house_act.value_counts().iloc[0:10].index)
plt.title("Housing", fontsize=16)
plt.subplot(827)
clth_act = loans['activity'].where(loans['sector'] == "Clothing")
sns.countplot(y=clth_act, order=clth_act.value_counts().iloc[0:10].index)
plt.title("Clothing", fontsize=16)
plt.subplot(828)
edu_act = loans['activity'].where(loans['sector'] == "Education")
sns.countplot(y=edu_act, order=edu_act.value_counts().iloc[0:10].index)
plt.title("Education", fontsize=16)
plt.subplot(829)
trans_act = loans['activity'].where(loans['sector'] == "Transportation")
sns.countplot(y=trans_act, order=trans_act.value_counts().iloc[0:10].index)
plt.title("Transportation", fontsize=16)
plt.subplot(8, 2, 10)
art_act = loans['activity'].where(loans['sector'] == "Arts")
sns.countplot(y=art_act, order=art_act.value_counts().iloc[0:10].index)
plt.title("Arts", fontsize=16)
plt.subplot(8, 2, 11)
hlth_act = loans['activity'].where(loans['sector'] == "Health")
sns.countplot(y=hlth_act, order=hlth_act.value_counts().iloc[0:10].index)
plt.title("Health", fontsize=16)
plt.subplot(8, 2, 12)
ctrn_act = loans['activity'].where(loans['sector'] == "Construction")
sns.countplot(y=ctrn_act, order=ctrn_act.value_counts().iloc[0:10].index)
plt.title("Construction", fontsize=16)
plt.subplot(8, 2, 13)
mnft_act = loans['activity'].where(loans['sector'] == "Manufacturing")
sns.countplot(y=mnft_act, order=mnft_act.value_counts().iloc[0:10].index)
plt.title("Manufacturing", fontsize=16)
plt.subplot(8, 2, 14)
etmt_act = loans['activity'].where(loans['sector'] == "Entertainment")
sns.countplot(y=etmt_act, order=etmt_act.value_counts().iloc[0:10].index)
plt.title("Entertainment", fontsize=16)
plt.subplot(8, 2, 15)
wlsl_act = loans['activity'].where(loans['sector'] == "Wholesale")
sns.countplot(y=wlsl_act, order=wlsl_act.value_counts().iloc[0:10].index)
plt.title("Wholesale", fontsize=16)


# Now let's look at the duration of these loans. Most of them seem to be for a 14-month or 8-month term. Very few have a duration of more than 2 years.

# In[18]:


plt.figure(figsize=(10,8))
sns.distplot(loans['term_in_months'], bins=80)
plt.title("Loan term in months", fontsize=20)
plt.xlabel('Number of months', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# The next graph shows the top 20 most common currencies of the loans. The Philippine Piso is the most common currency (also the country with most loans), followed bu the U. S. dollar and the Kenyan Shilling (second country with the most loans).

# In[19]:


plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['currency'], order=loans['currency'].value_counts().iloc[0:20].index)
plt.title("Top 20 most common currencies for loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Currency', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# The next plots show the distribution of the amount of money disbursed to the borrowers and the quantity of people who helped fund each loan. As expected, the plots are fairly similar in shape, since smaller value loans will tend to need less people to fund them, so these two quantities are expected to be correlated. Also, these plots show that most loans were for smaller amounts, below 10 thousand dollars, with a few reaching much higher amounts. In average, around 20 people were needed to fund each loan.

# In[20]:


plt.figure(figsize=(10,8))
sns.distplot(loans['loan_amount'], bins=80)
plt.title("Amount of money in the loans", fontsize=20)
plt.xlabel('Money (USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# In[21]:


plt.figure(figsize=(10,8))
sns.distplot(loans['lender_count'], bins=80)
plt.title("Amount of people who helped fund a loan", fontsize=20)
plt.xlabel('People', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
avg = loans['lender_count'].sum() / len(loans)
print("Average amount of people per loan: ", avg)


# Just making sure that the assumption made before (i.e., the number of lenders is correlated to the value of the loan), here is the scatter plot of these two quantities. There is a very clear positive correlation between the two quantities. The Pearson correlation matrix is calculated below and shows a high positive coefficient, as expected.

# In[22]:


plt.figure(figsize=(10,8))
plt.scatter(x=loans['lender_count'], y=loans['loan_amount'])
plt.title("Correlation between loan amount and people funding them", fontsize=20)
plt.xlabel('Number of lenders', fontsize=18)
plt.ylabel('Loan amount', fontsize=18)
plt.xticks(fontsize=12)
plt.show()

print("Pearson correlation:\n",np.corrcoef(loans['lender_count'], y=loans['loan_amount']))


# Also, we can look at how the loans are distributed over time. First the quantity of loans per year, which shows a increase every year. Even though the data only goes until July/2017, the quantity of loans in that year is already way over half of the year before, so, if the trend maintains, we should see an increase in 2017 as well.

# In[23]:


loans['date'] = pd.to_datetime(loans['date'], format = "%Y-%m-%d")


# In[24]:


plt.figure(figsize=(8,6))
sns.countplot(loans['date'].dt.year)
plt.title("Loans over the years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# The distribution of loans per month shows most of them in the first half of the year, but that is due to the fact we have data for 4 years in that range and only 3 years in the second semester. 

# In[25]:


plt.figure(figsize=(8,6))
sns.countplot(loans['date'].dt.month)
plt.title("Quantity of loans per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# Interestingly, the distribution of quantity of loans over the days of the month shows that most loans are posted at the second half of the month.

# In[26]:


plt.figure(figsize=(10,6))
sns.countplot(loans['date'].dt.day)
plt.title("Quantity of loans per day of the month", fontsize=20)
plt.xlabel('Day', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# ### How is the money distributed <a class="anchor" id="sec-2"></a>
# 
# Before I looked at how the loans are distributed, but now I want to know how the money itself is distributed in relation to some variables. If all loans had the same values, of course, we would expect to see the same distributions here, but, since that is not the case, maybe we can see some interesting patterns by looking at this data. In order to make the comparisson with the previous analyses easier, I will order the bar plots showing the distributions in the same order as the ones used in the loans.
# 
# To start, I will look at how the money is distributed per sector. In this plot, sectors in the left side receive a larger quantity of loans than the ones on their right side. The results show that sectors which received more loans did not necessarily receive more money.

# In[27]:


sectors = loans['sector'].unique()
money_sec = []
loan_sec = []

for sec in sectors:
    money_sec.append(loans['loan_amount'].where(loans['sector']==sec).sum())
    loan_sec.append((loans['sector']==sec).sum())

df_sector = pd.DataFrame([sectors, money_sec, loan_sec]).T


# In[28]:


plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_sector[1], y=df_sector[0], order=loans['sector'].value_counts().index)
plt.title("Distribution of money per sectors", fontsize=20)
plt.ylabel('Sectors', fontsize=18)
plt.xlabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()


# The previous result suggest that loans in different sectors received very different amounts of money. We can better see this variation by looking at the average amount of money per loan in each sector.

# In[29]:


plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_sector[1]/df_sector[2], y=df_sector[0], order=loans['sector'].value_counts().index)
plt.title("Average amount of money per loan", fontsize=20)
plt.ylabel('Sectors', fontsize=18)
plt.xlabel('Average money per loan (USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()


# Thus, even though agriculture, food and retail received the most loans, the borrowers in the sectors of entertainment and wholesale received, in average, a much larger amount of money.

# Now looking at how the money is distributed accross activities (again sorted in the same order as presented for the loans), we can see that, again, the activities with most loans are not, necessarily, the ones that received the most money. 

# In[30]:


activities = loans['activity'].unique()
money_act = []
loan_act = []

for act in activities:
    money_act.append(loans['loan_amount'].where(loans['activity']==act).sum())
    loan_act.append((loans['activity']==act).sum())

df_activity = pd.DataFrame([activities, money_act, loan_act]).T


# In[31]:


plt.figure(figsize=(10,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_activity[1], y=df_activity[0], order=loans['activity'].value_counts().index)
plt.title("Distribution of money per activity", fontsize=20)
plt.ylabel('Activities', fontsize=18)
plt.xlabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()


# We can again calculate the average amount of money spent in each activity to see which types of loans are receiving the largest investiments. Again, we can see that the loans with the largest average funding are not always the most common loans.

# In[32]:


plt.figure(figsize=(10,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_activity[1]/df_activity[2], y=df_activity[0], order=loans['activity'].value_counts().index)
plt.title("Average amount of money per activity", fontsize=20)
plt.ylabel('Activities', fontsize=18)
plt.xlabel('Average money per loan (USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()


# We can also look at how the money is distributed over time, as seen in the next plots. These distributions, however, are very similar to the ones seen for the qua

# In[33]:


years = loans['date'].dt.year.unique()
months = loans['date'].dt.month.unique()
days = loans['date'].dt.day.unique()
money_year = []
loan_year = []
money_month = []
loan_month = []
money_day = []
loan_day = []

for year in years:
    money_year.append(loans['loan_amount'].where(loans['date'].dt.year==year).sum())
    loan_year.append((loans['date'].dt.year==year).sum())
    
for month in months:
    money_month.append(loans['loan_amount'].where(loans['date'].dt.month==month).sum())
    loan_month.append((loans['date'].dt.month==month).sum())
    
for day in days:
    money_day.append(loans['loan_amount'].where(loans['date'].dt.day==day).sum())
    loan_day.append((loans['date'].dt.day==day).sum())

df_year = pd.DataFrame([years, money_year, loan_year]).T
df_month = pd.DataFrame([months, money_month, loan_month]).T
df_day = pd.DataFrame([days, money_day, loan_day]).T


# In[34]:


plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_year[0], y=df_year[1])
plt.title("Money distribution per year", fontsize=20)
plt.xlabel('Years', fontsize=18)
plt.ylabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# In[35]:


plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_month[0], y=df_month[1])
plt.title("Money distribution per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Money (x10^7 USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# In[36]:


plt.figure(figsize=(14,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_day[0], y=df_day[1])
plt.title("Money distribution per day of month", fontsize=20)
plt.xlabel('Day of month', fontsize=18)
plt.ylabel('Money (x10^7 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()


# ## Global analyses<a class="anchor" id="third"></a>
# 
# The goal in this section is to examine the distribution of loans and money around the globe and how this is related to poverty and human development measures. 

# First, we look at how the quantity of loans are distributed globally. The Philippines seems to be the country receives the largest amount of loans, followed by Kenya.

# In[47]:


count_country = loans['country'].value_counts().dropna()
codes = loans['country_code'].unique()
countries = loans['country'].unique()
money_ctry = []

for c in countries:
    money_ctry.append(loans['funded_amount'].where(loans['country']==c).sum())
    
dataMap = pd.DataFrame([codes, countries, count_country[countries], money_ctry]).T


# In[46]:


data = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[2],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Loans per country'),
      ) ]

layout = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot(fig, validate=False)


# Now, we look at how much money, in total, each country received. It is obvious, from the map, that the countries with more loans do not, necessarily receive the most money.

# In[48]:


data2 = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[3],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money per country'),
      ) ]

layout2 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig2 = dict( data=data2, layout=layout2 )
plotly.offline.iplot(fig2, validate=False)


# To see the difference in the value of loans per country, we can make the same plot, but now, showing the average value of each loan.

# In[60]:


data3 = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[3]/dataMap[2],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Average amount per loan'),
      ) ]

layout3 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig3 = dict( data=data3, layout=layout3 )
plotly.offline.iplot(fig3, validate=False)


# This last plot shows that, although the Philippines and Kenya receive the most amount of loans, these loans are not among the ones with the highest value. The highest value was for the single loan given in Cote D'Ivoire, which received 50,000 USD to invest in agriculture, as seen below.

# In[58]:


loans[loans['country']=='Cote D\'Ivoire']


# Now we can compare these maps with the maps showing some development indicators, starting with the Human Development Index (HDI).

# In[61]:


data4 = [ dict(
        type = 'choropleth',
        locations = hdi['Country'],
        z = hdi['Human Development Index (HDI)'],
        text = hdi['Country'],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'HDI'),
      ) ]

layout4 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig4 = dict(data=data4, layout=layout4)
plotly.offline.iplot(fig4, validate=False)


# From this last map, we can see that most of the countries with the highest HDI have not received a single loan from Kiva, showing that Kiva tends to reach people from regions with worse living conditions in a global scale. The question is, though, if the quantity of loans is correlated to the HDI.

# In[212]:


hdi_temp = []
leb_temp = []
mye_temp = []
gni_temp = []
mpi_temp = []

for c in dataMap[1]:
    h = hdi[hdi['Country']==c]
    m = mpi[mpi['Country']==c]
    hdi_temp.append(h['Human Development Index (HDI)'].values)
    leb_temp.append(h['Life Expectancy at Birth'].values)
    mye_temp.append(h['Mean Years of Education'].values)
    gni_temp.append(h['Gross National Income (GNI) per Capita'].values)
    mpi_temp.append(m['Multidimensional Poverty Index (MPI, 2010)'].values)

hdi_kiva = []
leb_kiva = []
mye_kiva = []
gni_kiva = []
mpi_kiva = []
c_hdi_kiva = []
c_mpi_kiva = []

for i in range(0, len(dataMap[1])):
    if hdi_temp[i].size:
        hdi_kiva.append(hdi_temp[i][0])
        leb_kiva.append(leb_temp[i][0])
        mye_kiva.append(mye_temp[i][0])
        gni_kiva.append(gni_temp[i][0])
        c_hdi_kiva.append(dataMap[2][i])
    if mpi_temp[i].size:
        mpi_kiva.append(mpi_temp[i][0])
        c_mpi_kiva.append(dataMap[2][i])

df_hdi = pd.DataFrame([c_hdi_kiva, hdi_kiva, leb_kiva, mye_kiva, gni_kiva, c_mpi_kiva, mpi_kiva]).T


# In[153]:


plt.figure(figsize=(8,6))
plt.scatter(x=df_hdi[0], y=df_hdi[1])
plt.title("Correlation between quantity of loans per country and its HDI", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('HDI', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# Although, it is possible to observe that most of the loans are for countries with HDI below 0.7, which means they are considered countries medium or low human development, the number os loans does not increase as the HDI descreases. Thus, there is no correlation between the two quantities. Since the HDI is calculated based on metrics of health, standard of living and education, these results are not at all surprising. The people who would probably need this resource the most will not have access to computers/internet and, maybe, will not even know Kiva exists.

# Let's now see the correlation plots between the quantity of Kiva loans and other indices, which reveal information on human welfare.

# In[167]:


plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[0], y=df_hdi[2])
plt.title("Life Expectancy at Birth", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Life Expectancy at Birth (years)', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[0], y=df_hdi[2])
plt.title("Life Expectancy at Birth - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Life Expectancy at Birth (years)', fontsize=18)
plt.axis([-2500, 45000, 48, 85])
plt.xticks(fontsize=12)
plt.show()


# In[208]:


plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[0], y=df_hdi[3])
plt.title("Mean Years of Education", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Mean Years of Education (years)', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[0], y=df_hdi[3])
plt.title("Mean Years of Education - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Mean Years of Education (years)', fontsize=18)
plt.axis([-2500, 45000, 1, 14])
plt.xticks(fontsize=12)
plt.show()


# In[216]:


plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[5], y=df_hdi[6])
plt.title("Multidimensional Poverty Index", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Multidimensional Poverty Index', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[5], y=df_hdi[6])
plt.title("Multidimensional Poverty Index - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Multidimensional Poverty Index', fontsize=18)
plt.axis([-2500, 45000, -0.05, 0.6])
plt.xticks(fontsize=12)
plt.show()


# Again, none of the indicators show a correlation to the observed quantity of loans, although, again, we see most of the loans going to countries with comparatively high MPI (i.e., where more people suffer from multidimensional poverty).
# 
# Therefore, the majority of loans were given to people from countries with a low HDI and a high MPI, which suggest people with a high level of poverty. However, the number of borrowers does not increase as the HDI decreases or the MPI increases, probably due to difficulties caused by their own level of welfare (e.g., low education levels, no access to computers/internet, among other factors).

# ## Country specific analyses<a class="anchor" id="fourth"></a>
# 
# Having a general knowlegde about the loans performed in the past years, now we can have a look at specifics for different countries and their different regions. This section will give preference to those countries where most of the loans go to, starting with the Philippines.

# ### Philippines<a class="anchor" id="four-1"></a>
# 
# The Philippines in a country in the Southeast Asia with a population of 103.3 million people (2016, World Bank). It has the 34th largest economy in the world and their principal exports include semiconductors and electronic products, transport equipment, garments, copper products, petroleum products, coconut oil, and fruits. The currency is the Philippine peso (PHP). People in the Philippines have received 160,441 loans.

# In[37]:


# separate the Kiva data from the Philippines
phil = loans[loans['country'] == 'Philippines']
print(phil.shape)
phil.sample(3)


# Now let's see how the people in the Philippines are using their loans, in terms of sectors of the economy.

# In[38]:


plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=phil['sector'], order=phil['sector'].value_counts().index)
plt.title("Sectors which received loans in the Philippines", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Sector', fontsize=18)
plt.xticks(fontsize=12)
plt.show()


# Most of the loans are going toward retail, food and agriculture, in this order. Together, these three sectors make up almost 83% of all the loans. We can look in more detail which are the main activities being developed in these sectors.

# In[39]:


plt.figure(figsize=(14,6))
plt.subplot(131)
retl_phil = phil['activity'].where(phil['sector'] == "Retail")
sns.countplot(y=retl_phil, order=retl_phil.value_counts().iloc[0:10].index)
plt.title("Retail", fontsize=16)
plt.subplot(132)
food_phil = phil['activity'].where(phil['sector'] == "Food")
sns.countplot(y=food_phil, order=food_phil.value_counts().iloc[0:10].index)
plt.title("Food", fontsize=16)
plt.subplot(133)
agric_phil = phil['activity'].where(phil['sector'] == "Agriculture")
sns.countplot(y=agric_phil, order=agric_phil.value_counts().iloc[0:10].index)
plt.title("Agriculture", fontsize=16)
plt.show()


# So, in retail, the main activity, by far is general store. In food, we have fish selling, fishing and food production/sales as the main activities. And in agriculture, the most loans are for pigs and farming.

# In[ ]:




