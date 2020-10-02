#!/usr/bin/env python
# coding: utf-8

# *There are a lot of kernels which delve into detailed step-by-step analysis. In this kernel, I will be solely focusing on the insights derived from my analyses. Do share your views on the insights that I share. Thanks. *

# **6 Key Insights from Kiva dataset:**
# 
# 1. Percentage of funding is very low for Technology, Communications and Cleaning Services activities. Why is that so? 
# 
# 2. USD is the currency in which most of the countries' loans are recorded (it is used in 64 countries). 15.7% of the loans are in USD currency. 
# 
# 3. Is there any difference in the way loans in USD and other currencies treated? 
# 
#        a. Except Health 'sector', the funded amount as a percentage of loan amount is higher for Non USD currency loans.
#     
#        b. Loans in USD currency got better funding in the following activities: Aquaculture, Electrician, Music Discs & Tapes.
#        
#        c. Loans in Non USD currency got better funding in the following activities: Technology, Entertainment and Recycling.
#             
# 4. 73% of the loans were received by female only borrowers - a good thing by Kiva. 
# 
# 5. Men got longer repayment terms for the loans than Women. 
# 
# 6. Male Borrower Ratio is very high (> 80%) in Mauritania, Belize, Suriname and Nigeria. 
# 
# Check out the visualizations below for details on these in.

# **Datasets provided:**
# 1. Loans - 671,205 rows x 20 columns - Contains the list of loans, amount, reason, location, partner, timelines, term, repayment details, etc.
# 2. Locations - 2,772 rows x 9 columns - Containg the list of locations, their coordinates and MPI (Multidimensional Poverty Index - an index comprising of health, education and standard of living dimensions published by UN, http://hdr.undp.org/en/content/multidimensional-poverty-index-mpi)
# 3. Themes - 779,092 rows x 4 columns - TO BE UNDERSTOOD
# 4. Themes by region - 15,736 rows x 21 columns - TO BE UNDERSTOOD
# 
# **Data specific observations:**
# 1. In 'Loans', 'country' Namibia has no corresponding 'country_code'. Probably someone typed 'Nan' instead of 'Nam'. :)

# In[1]:


# Import required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Read data
folder = "../input/"
loans = pd.read_csv(folder + "kiva_loans.csv")
locations = pd.read_csv(folder + "kiva_mpi_region_locations.csv")
themes = pd.read_csv(folder + "loan_theme_ids.csv")
themes_by_region = pd.read_csv(folder + "loan_themes_by_region.csv")

# Function to create USA currency flag
def USD_flag(currency):
    if currency == "USD":
        return 1
    else:
        return 0

# Function to classify gender of the group of borrowers
def classify_genders(x):
    if x==0:
        return "Only Females"
    elif x==1:
        return "Only Males"
    elif x==0.5:
        return "Equal Males and Females"
    elif x<0.5:
        return "More Females"
    elif x>0.5:
        return "More Males"
    
# Initial data processing - features for analysis
loans['percentage_funding'] = loans['funded_amount'] * 100 / loans['loan_amount']
loans['USD'] = loans['currency'].apply(lambda x: USD_flag(x))
#loans.dropna(subset=['borrower_genders'])
loans['borrower_genders'] = loans['borrower_genders'].astype(str)
loans['male_borrowers'] = loans['borrower_genders'].apply(lambda x: len(re.findall(r'\bmale', x)))
loans['female_borrowers'] = loans['borrower_genders'].apply(lambda x: len(re.findall(r'\bfemale', x)))
loans['borrowers_count'] = loans['male_borrowers'] + loans['female_borrowers']
loans['male_borrower_ratio'] = loans['male_borrowers'] / loans['borrowers_count']
loans['gender_class'] = loans['male_borrower_ratio'].apply(lambda x: classify_genders(x))


sectors = loans['sector'].unique()
activities = loans['activity'].unique()


# **Insight #1: Percentage of funding is very low for Technology, Communications and Cleaning Services 'activities' - all 3 in Services 'sector'. Funded amount as a percentage of loan amount stands at 59.1% as against 93.4% for all other 'activities'. Why is that so? **
# 
# Even for Entertainment, it's less than 80%

# In[2]:


df_temp = loans.groupby('activity')['percentage_funding'].describe().sort_values('mean').head(10).reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x="mean", y="activity", data=df_temp, palette=sns.color_palette("YlOrRd_r", 10), alpha=0.6)
plt.title("Activity wise - average % funding for loans", fontsize=16)
plt.xlabel("Avg. % funding", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show()


# In[ ]:


funding_low_sectors = loans[(loans['activity']=='Technology') | (loans['activity']=='Communications') | (loans['activity']=='Cleaning Services')]['funded_amount'].sum()
loan_low_sectors = loans[(loans['activity']=='Technology') | (loans['activity']=='Communications') | (loans['activity']=='Cleaning Services')]['loan_amount'].sum()

funding_high_sectors = loans[(loans['activity']!='Technology') & (loans['activity']!='Communications') & (loans['activity']!='Cleaning Services')]['funded_amount'].sum()
loan_high_sectors = loans[(loans['activity']!='Technology') & (loans['activity']!='Communications') & (loans['activity']!='Cleaning Services')]['loan_amount'].sum()

print("Percentage of funding for Tech, Comm and Cleaning services activities: " + str(round(funding_low_sectors * 100 / loan_low_sectors, 2)) + "%")
print("Percentage of funding for other activities: " + str(round(funding_high_sectors * 100 / loan_high_sectors, 2)) + "%")


# **Insight #2: USD is the currency in which most of the countries' loans are recorded (it is used in 64 countries). The next highest is XOF (5), XAF (2), ILF (2) and JOD (2). 15.7% of the loans are in USD currency. USD loans = 105,494 and Non USD loans = 565,711.**

# In[3]:


df_temp = loans.groupby('currency')['country'].nunique().sort_values(ascending=False).reset_index().head()

plt.figure(figsize=(6,3))
sns.barplot(x="country", y="currency", data=df_temp, palette=sns.color_palette("Spectral", 5), alpha=0.6)
plt.title("Number of Countries using a currency for loans", fontsize=16)
plt.xlabel("Number of countries", fontsize=16)
plt.ylabel("Currency", fontsize=16)
plt.show()


# In[ ]:


print("Number of loans using USD as currency: " + str(loans[loans['USD']==1]['USD'].count()))
print("Percentage of overall loans using USD as currency: " + str(round(loans[loans['USD']==1]['USD'].count() * 100 / loans['USD'].count(),2)) + "%")


# **Insight #3: Is there any difference in the way loans in USD and other currencies treated? **

# *3.a. Except Health 'sector', the funded amount as a percentage of loan amount is higher for Non USD currency loans. The difference is highest in Entertainment at 39.6 percentage points. *

# In[4]:


df_sector = pd.DataFrame(columns=['sector', 'USD_fund', 'USD_loan', 'Non_USD_fund', 'Non_USD_loan'])

for sector in sectors:
    USD_fund = loans[(loans['sector']==sector) & (loans['USD']==1)]['funded_amount'].sum()
    USD_loan = loans[(loans['sector']==sector) & (loans['USD']==1)]['loan_amount'].sum()
    
    Non_USD_fund = loans[(loans['sector']==sector) & (loans['USD']==0)]['funded_amount'].sum()
    Non_USD_loan = loans[(loans['sector']==sector) & (loans['USD']==0)]['loan_amount'].sum()
    
    df_sector.loc[len(df_sector)]=[sector, USD_fund, USD_loan, Non_USD_fund, Non_USD_loan] 
    
df_sector['USD_fund_percent'] = round(df_sector['USD_fund'] * 100 / df_sector['USD_loan'],2)
df_sector['Non_USD_fund_percent'] = round(df_sector['Non_USD_fund'] * 100 / df_sector['Non_USD_loan'], 2)
df_sector['Difference'] = df_sector['Non_USD_fund_percent'] - df_sector['USD_fund_percent']

df_sector.sort_values('Difference')

plt.figure(figsize=(8,6))
sns.barplot(x="Difference", y="sector", data=df_sector.sort_values('Difference'), palette=sns.color_palette("BrBG", 15), alpha=0.6)
plt.title("Difference in funding for loans in USD and non-USD currencies", fontsize=16)
plt.xlabel("Difference in % points", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.show()


# *3.b. Digging into 'activity' level gives a detailed picture. *
#     
# Loans in USD currency got better funding in the following activities: 
# 1. Aquaculture - 14.4 percentage points, 
# 2. Electrician - 12.8 percentage points & 
# 3. Music Discs & Tapes - 10.3 percentage points. 
# 
# Loans in Non USD currency got better funding in the following activities: 
# 1. Technology - 48.1 percentage points, 
# 2. Entertainment - 44.4 percentage points & 
# 3. Recycling - 39.4 percentage points.

# In[5]:


df_activities = pd.DataFrame(columns=['activity', 'USD_fund', 'USD_loan', 'Non_USD_fund', 'Non_USD_loan'])

for activity in activities:
    USD_fund = loans[(loans['activity']==activity) & (loans['USD']==1)]['funded_amount'].sum()
    USD_loan = loans[(loans['activity']==activity) & (loans['USD']==1)]['loan_amount'].sum()
    
    Non_USD_fund = loans[(loans['activity']==activity) & (loans['USD']==0)]['funded_amount'].sum()
    Non_USD_loan = loans[(loans['activity']==activity) & (loans['USD']==0)]['loan_amount'].sum()
    
    df_activities.loc[len(df_activities)]=[activity, USD_fund, USD_loan, Non_USD_fund, Non_USD_loan] 
    
df_activities['USD_fund_percent'] = round(df_activities['USD_fund'] * 100 / df_activities['USD_loan'], 2)
df_activities['Non_USD_fund_percent'] = round(df_activities['Non_USD_fund'] * 100 / df_activities['Non_USD_loan'], 2)
df_activities['Difference'] = df_activities['Non_USD_fund_percent'] - df_activities['USD_fund_percent']

plt.figure(figsize=(6,4))
sns.barplot(x="Difference", 
            y="activity", 
            data=df_activities.sort_values('Difference').head(10), 
            palette=sns.color_palette("YlGnBu", 10),
            alpha = 0.6)
plt.title("Difference in funding for loans in USD and non-USD currencies", fontsize=16)
plt.xlabel("Difference in % points", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show()


# In[6]:


plt.figure(figsize=(6,4))
sns.barplot(x="Difference", 
            y="activity", 
            data=df_activities.sort_values('Difference', ascending=False).head(10), 
            palette=sns.color_palette("YlGnBu_r", 10),
            alpha = 0.6)
plt.title("Difference in funding for loans in USD and non-USD currencies", fontsize=16)
plt.xlabel("Difference in % points", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show()


# **Insight #4. Only 6% of the loans were received by a combination of male and female borrowers. 73% of the loans were received by female only borrowers - a good thing by Kiva. 83.6% of the loans are received by single (just one in number, not their marital status ;)) borrowers. Only 4.7% of the loans are provided by single lenders. Among the loans sought by single borrowers, 3/4th were received by females.**

# In[8]:


female_borrowers_only_loans = round(
    100 * loans[loans['male_borrower_ratio']==0]['male_borrower_ratio'].count() / loans['male_borrower_ratio'].count(), 2)

male_borrowers_only_loans = round(
    100 * loans[loans['male_borrower_ratio']==1]['male_borrower_ratio'].count() / loans['male_borrower_ratio'].count(), 2)

male_female_borrowers_loans = round(
    100 - female_borrowers_only_loans - male_borrowers_only_loans, 2)

one_borrowers = round(
    100 * loans[loans['borrowers_count']==1]['borrowers_count'].count() / loans['borrowers_count'].count(), 2)

one_lenders = round(
    100 * loans[loans['lender_count']==1]['lender_count'].count() / loans['lender_count'].count(), 2)

one_female_borrowers = round(
    100 * loans[(loans['female_borrowers']==1) & (loans['borrowers_count']==1)]['female_borrowers'].count() 
    / loans[loans['borrowers_count']==1]['female_borrowers'].count(), 2)

print("% of loans with only female borrowers: " + str(female_borrowers_only_loans) + "%")
print("% of loans with only male borrowers: " + str(male_borrowers_only_loans) + "%")
print("% of loans with both male and female borrowers: " + str(male_female_borrowers_loans) + "%")
print("% of loans with only one borrower: " + str(one_borrowers) + "%")
print("% of loans with only one female borrower: " + str(one_female_borrowers) + "%")
print("% of loans with only one lender: " + str(one_lenders))


# **Insight #5. Men got longer repayment terms for the loans than Women.**

# *Only male borrowers have a higher term_in_months distribution as seen in the below graph.*

# In[10]:


plt.figure(figsize=(10,6))
sns.lvplot(x="gender_class", 
           y="term_in_months", 
           data=loans[loans['term_in_months']<=36], 
           palette=sns.color_palette("PiYG", 5))
plt.title("Distribution of term_in_months vs borrower gender", fontsize=16)
plt.xlabel("Borrower gender classes", fontsize=16)
plt.ylabel("Term in months", fontsize=16)
plt.show()


# **Insight #6: Male Borrower Ratio is very high (> 80%) in Mauritania, Belize, Suriname and Nigeria. **

# In[ ]:


df_country = loans.groupby(['country', 'country_code'])['male_borrower_ratio'].mean().reset_index()

data = [dict(
    type = 'choropleth',
    locations = df_country['country'],
    locationmode = 'country names',
    z = df_country['male_borrower_ratio'],
    text = df_country['country'],
    colorscale = [[0,"rgb(159,51,51)"],
                  [0.2,"rgb(221,66,66)"],
                  [0.5,"rgb	(249,217,217)"],
                  [0.8,"rgb(188,240,255)"],
                  [1,"rgb(26,94,118)"]],
    autocolorscale = False,
    reversescale = True,
    marker = dict(
        line = dict (
            color = 'rgb(180,180,180)',
            width = 0.5
        ) ),
    colorbar = dict(
        autotick = False,
        tickprefix = '',
        title = 'Mean Male Borrower Ratio'),
) ]

layout = dict(
    title = 'Male Borrower Ratio across countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout )
py.iplot(fig, validate=False, filename='d3-world-map' )


# **TO BE CONTINUED**
