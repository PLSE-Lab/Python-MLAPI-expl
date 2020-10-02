#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# This notebook has some of the basic Exploratory Data Analysis graphs and visualizations done using datasets provided by Kiva as part of "Data Science for Good: Kiva Crowdfunding. Being a begineer, i have tried gain some useful insights from these datasets.  I look forward to learn and improve along the way and would really appreciate all your comments and suggestions. 
# 
# In the first section of the notebook, i will focus on EDA of the 4 datasets provided by Kiva and in the next section, i use few of the available external datasets to gain more insights about MPI

# **Sections**
# 1. Kiva - Exploratory Data Analysis
#     *       Dataset Description
#     *     Average Funded Loan Amount
#     *     Distribution of loan by Country
#     *     Distribution of loan by Rregion and Country)
#     *     Top activities by sector
#     *     Top 50 Poverty regions by MPI
#     *     Mean Loan/Funded amount by MPI
#     *      Funded amount by world region
#     *      Loan Theme Count by (Theme Type and Country)
#     *     Funded Amount by Loan Theme Type
#     *     Loan count by Field Partner
#     *     Mean Loan amount by Field Partner
#     *    Borrower Gender Distribution
#     *    Funded Loan amount distribution by Gender
#     *    Lender Count Distribution by  Borrower Gender

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
import os
print(os.listdir("../input"))


# **1. Kiva - Exploratory Data Analysis**
# The purpose of this section is to understand more about the datasets provided by Kiva through EDA and visualizations.

# **1. 1.Dataset Description**

# In[ ]:


kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
print (kiva_loans_df.head(5))
kiva_mpi_region_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
print (kiva_mpi_region_locations_df.head(5))
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
print (loan_theme_ids_df.head(5))
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
print (loan_themes_by_region_df.head(5))


# **1.2.Average Funded Loan Amount**

# In[ ]:


#  Average funded amount by country
mean_funded_amt_reg = kiva_loans_df.groupby('country').agg({'funded_amount':'mean'}).reset_index()
mean_funded_amt_reg.columns = ['country','funded_amount_mean']
mean_funded_amt_reg = mean_funded_amt_reg.sort_values('funded_amount_mean',ascending=False)
print (mean_funded_amt_reg.head())

data = [ dict(
        type = 'choropleth',
        locations = mean_funded_amt_reg['country'],
        locationmode = 'country names',
        z = mean_funded_amt_reg['funded_amount_mean'],
        text = mean_funded_amt_reg['country'],
        colorscale='Earth',
        reversescale=False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Average Funded Loan Amount'),
      ) ]

layout = dict(
    title = 'Average Funded Loan Amount by Country (US Dollars)',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        showlakes = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot(fig, validate=False)


# As seen from the World map above, Most of the kiva funded countries have average funded amount well below 10K US dollars, except for **Cote D'Ivoire** whose mean funded loan amount is at 50K US dollars. 

# **1.3.Distribution of loan by Country**

# In[ ]:


## Top 25 countries with most number of loans
loan_by_country = kiva_loans_df.groupby('country')['country'].count().sort_values(ascending=False).head(25)
#print (loan_by_country)
plt.figure(figsize=(16,8))
sns.barplot(x=loan_by_country.index,y=loan_by_country.values,palette="BuGn_d")
plt.xticks(rotation="vertical")
plt.xlabel("Countries with most number of borrowers",fontsize=20)
plt.ylabel("Number of active loans",fontsize=18)
plt.title("Top 25 countries with most number of active loans", fontsize=22)
plt.show()


# Philippines leads the chart with more number of borrowers and has close to 160K active loans, followed by Kenya and El Salvador and most of the other countries in the top 25 has lower than 20K loans.

# **1.4.Distribution of loan by(Region and Country)**

# In[ ]:


## Top 25 regions with most number of loans sanctioned
loan_by_con_reg = kiva_loans_df.groupby('country')['region'].value_counts().nlargest(25)     
plt.figure(figsize=(16,8))
sns.barplot(x=loan_by_con_reg.index,y=loan_by_con_reg.values,palette="RdBu_r")
plt.xticks(rotation="vertical")
plt.xlabel("Regions with most number of borrowers",fontsize = 16)
plt.ylabel("Number of loans",fontsize=14)
plt.title("Top 25 Regions with most number of active loans",fontsize = 20)
plt.show()


# The bar chart above shows top 25 regions with most number of active loans sorted by  country. Kaduna, Nigeria has about 10000 active loans, followed Lahore and Rawalpindi,both these regions are part of Pakistan. Most of the other top regions are in Phillippines.

# 

# **1.5.Top activities by sector**

# In[ ]:


## Let's find out the activities that got more loans approved and plot them by sector
activity_type_by_sector = kiva_loans_df.groupby(['sector','activity']).size().sort_values(ascending=False).reset_index(name='total_count')
activity_type_by_sector = activity_type_by_sector.groupby('sector').nth((0,1)).reset_index()
plt.figure(figsize=(16,10))
sns.barplot(x="activity",y="total_count",data=activity_type_by_sector)
plt.xticks(rotation="vertical")
plt.xlabel("Activites with most number of Borrowers",fontsize = 16)
plt.ylabel("Number of loans",fontsize=14)
plt.title("Top Two Actvities By Sector vs Number Of Loans",fontsize = 20)
plt.show()


# The top 2 dominant activites by sector is plotted in the above bar plot. Farming and Agriculture are the two dominant activities by sector "Agriculture". General Stores has the next highest number of loans, followed by Personal housing expenses and Food Production/sales.

#  Let's do some analysis on kiva_loans_df and kiva_mpi_region_locations datasets. The join will be based on country and region columns from both the datasets since these values are the only available information common to both the datasets.
#  
#  kiva_loans and kiva_mpi_region_locations datasets will be pre-processed  to drop any null values from the relevant columns and the merge operation will be done on the pre-processed datasets

# In[ ]:


kiva_loans = kiva_loans_df.filter(['country','region','funded_amount','loan_amount','activity','sector','borrower_genders', 'repayment_interval'])
kiva_loans = kiva_loans.dropna(subset=['country','region'])          
#print (kiva_loans.shape)

kiva_mpi_region_locations = kiva_mpi_region_locations_df[['country','region','world_region','MPI','lat','lon']]
kiva_mpi_region_locations = kiva_mpi_region_locations.dropna()                         
#print (kiva_mpi_region_locations.shape)
mpi_values = pd.merge(kiva_mpi_region_locations,kiva_loans,how="left")
#print (mpi_values.shape)


# Surprisingly, Many active loans are missing MPI  and only a subset of loan data got assigned to corresponding MPI region. Hence, the dataset obtained after the merge has only about 51K records and the following analysis will be done based on the available data

# **1.6. Top 50 Poorest Regions by MPI**

# In[ ]:


# Distribution of top 50 poorest regions by MPI
mpi_values_df = mpi_values.sort_values(by=['MPI'],ascending=False).head(50)
data = [ dict(
        type = 'scattergeo',
        lon = mpi_values_df['lon'],
        lat = mpi_values_df['lat'],
        text = mpi_values_df['MPI'].astype('str') + ' ' + mpi_values_df.region + ' ' + mpi_values_df.country,
        mode = 'markers',
        marker = dict(
            size = mpi_values_df['MPI']/0.04,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            cmin = 0,
            color = mpi_values_df['MPI'],
            cmax = mpi_values_df['MPI'].max(),
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'Top 50 Poverty Regions by MPI',
        colorbar = True,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
        ),
    )

fig1= dict( data=data, layout=layout)
iplot(fig1, validate=False)


# We can observe from the above graph that most of the Poorest regions are in the African countries and come under "Sub-Sahara" world region which is one of the poorest sub-national regions per OPHI.

# **1.7. Mean Loan/funded amount by MPI**

# In[ ]:


# Let's see the distribution of average loan/funded_amount by MPI
mpi_values_amount = mpi_values.groupby('MPI',as_index=False)['loan_amount','funded_amount'].mean()
mpi_values_amount = mpi_values_amount.dropna(subset=['loan_amount','funded_amount'],how="all")
#print (mpi_values_amount.shape)
fig,ax = plt.subplots(figsize=(15,8))
lns1 = ax.plot('MPI','loan_amount',data=mpi_values_amount,label="Loan amount",color="Blue")
ax2 = ax.twinx()
lns2 = ax2.plot('MPI','funded_amount',data=mpi_values_amount,label = "Funded amount",color="Green")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.set_title("Loan/Funded Amount by MPI")
ax.set_xlabel("MPI")
ax.set_ylabel("Loan Amount (in US dollars)")
ax.tick_params(axis='y',labelcolor="Blue")
ax.set_ylim(100,10000)
ax2.set_ylabel("Funded Loan Amount (in  US dollars)")
ax2.tick_params(axis='y',labelcolor="Green")
ax2.set_ylim(100,10500)
plt.show()


# The above graph loan amount/funded amount by MPI is definitely not in line with our expectations.  Loan/Funded amount distributed to moderate/low API regions is significantly larger than the loan amount distributed to high MPI regions. The loan amount seems to flatten out for high MPI values (0.3 and above).

# **1.8.Funded amount by world region**

# In[ ]:


## Let's plot the total amount funded for each world region
fund_by_world_region = mpi_values.groupby('world_region')['funded_amount'].sum()
plt.figure(figsize=(15,8))
sns.barplot(x = fund_by_world_region.index, y=fund_by_world_region.values,log=True)
#plt.xticks(rotation="vertical")
plt.xlabel("World Region",fontsize=20)
plt.ylabel("Funded loan amount(US dollars)",fontsize=20)
plt.title("Funded loan amount by world-region",fontsize=22)
plt.show()


# Bar chart above depicts the Total funded loan amount by world-region. It appears from the chart that world-region "Latin America and Carribean" stands first with most funds allocated followed Sub-sahara Africa and East Asia and the pacific, South Asia being the least funded region stands last.

# In[ ]:


# Join/Merge Loan,loan_theme,loan_theme_region datasets for further Analysis
kiva_loans_df.rename(columns={'partner_id':'Partner ID'},inplace=True)
loan_themes  = kiva_loans_df.merge(loan_theme_ids_df,how='left').merge(loan_themes_by_region_df,on=['Loan Theme ID','Partner ID','country','region'])
print (loan_themes.columns)


# **1.9.Loan Theme Count by (Theme Type and Country)**

# In[ ]:


theme_country = loan_themes.groupby(['country'])['Loan Theme Type_x'].value_counts().sort_values(ascending=False)
theme_country = theme_country.unstack(level=0)
theme_country = theme_country.unstack().dropna(how="any")
theme_country = theme_country.sort_values(ascending=False).reset_index(name="loan_theme_count")
theme_country = theme_country.rename(columns={'Loan Theme Type_x': 'Loan_Theme_Type'})
theme_country = theme_country.head(20)
print (theme_country)
theme_country.iplot(kind="bar",barmode="stack",title="Loan Theme Count by Theme Type and Country")


# This bar plot shows the top 20 ** loan count** sorted by "loan theme type" and "country".  Most of the loans accquired falls under the loan theme category "General" . and Borrowers from different countries  have got more loans sanctioned under this category

# **1.10. Funded Amount by Loan Theme Type**

# In[ ]:


# Lets analyse the distribution of average funded_amount and funded_amount count by by loan theme type
amt_theme = loan_themes.groupby('Loan Theme Type_x').agg({'funded_amount':['mean','count']}).reset_index()
amt_theme.columns = ['Loan Theme Type_x','funded_amount_mean','funded_amount_count']
#print (amt_theme)
amt_theme = amt_theme.sort_values(['funded_amount_mean','funded_amount_count'],ascending=[False,False])
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111) 
ax2 = ax.twinx() 
width = 0.4
amt_theme = amt_theme.set_index('Loan Theme Type_x')
amt_theme.head(20).funded_amount_mean.plot(kind='bar',color='dodgerblue',width=width,ax=ax,position=0)
amt_theme.head(20).funded_amount_count.plot(kind='bar',color='goldenrod',width=width,ax=ax2,position=1)
ax.grid(None, axis=1)
ax2.grid(None)
ax.set_xlabel("Loan Theme Type")
ax.set_ylabel('Funded Amount(mean)')
ax2.set_ylabel('Funded Amount(count)')
ax.set_title("Mean funded amount and count by Loan Theme Type")
plt.show()


# The most funded "Loan Theme Type" per the above bart chart is "Business in a box" with the average funded amount 50K US dollars, followed by "Equipment Purchase". Most of the other loan theme types haven't received as much in funding and the average funded amount for those loan theme types is lower than 10K US dollars. In terms of number of loans received by "Loan Theme Type",  "Women without Poverty" theme type has received more than 2000 loans with "SME" following it with more than 500 loans. Most of the other loan theme types does not have any significant number of loans  sanctioned, except for few as seen in the chart.

# **1.11.Loan count by Field Partner**

# In[ ]:


# Analyze loan count by field partner name
loancount_by_fpartner = loan_themes.groupby(['Field Partner Name'])['Field Partner Name'].count().sort_values(ascending=False).head(20)
plt.figure(figsize=(15,8))
pal = sns.color_palette("Oranges", len(loancount_by_fpartner))
sns.barplot(x=loancount_by_fpartner.index,y=loancount_by_fpartner.values,palette=np.array(pal[::-1]))
plt.xticks(rotation="vertical")
plt.xlabel("Field Partner Name",fontsize=20)
plt.ylabel("Loan Count",fontsize=20)
plt.title("Loan count by Field Partners",fontsize=22)
plt.show()


# The Bar chart depicts the top 20 Field Partners with most number of loans. "Negros Women for Tomorrow Foundation" is the top most in the list with over 100K loans disbursed through it,followed by "CreditCambo" and "Juhundi Kilmo". 

# **1.12. Mean Loan amount by Field Partner**

# In[ ]:


# Lowest 10 average loan amount by field partner name
mloan_fpartner = loan_themes.groupby('Field Partner Name')['loan_amount'].mean().sort_values(ascending=True)
print (mloan_fpartner.head(10))
mloan_fpartner.head(15).iplot(kind='bar',yTitle="Average Loan Amount(US Dollars)",title="Average Loan amount by Field Partner Name")


# We see that average Minimum loan amount by field partner(s) doesnt seem to vary greatly. most of the loans amounts vary between 160 to 260 US dollars.

# **1.13.Borrower Gender Distribution**

# In[ ]:


# Clean the borrower_genders column to replace list of male/female values to 'group'
mask = ((loan_themes.borrower_genders!= 'female') &
                                  (loan_themes.borrower_genders != 'male') & (loan_themes.borrower_genders != 'NaN'))
loan_themes.loc[mask,'borrower_genders'] = 'group'
print (loan_themes.borrower_genders.unique())
bgenders = loan_themes.borrower_genders.value_counts()
labels = bgenders.index
values = bgenders.values
trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)
layout = go.Layout(
    title="Borrowers' genders"
)
data = trace
fig = go.Figure(data=[data], layout=layout)
iplot(fig,validate=False)


# From the piechart, we see that majority of the loan borrowers are female, contributing to more than **60**% of the active loans, followed by male and group. 15% of the loans are approved to groups which maybe male only or female only or [male,female] 

# **1.14.Funded Loan amount distribution by Gender**

# In[ ]:


# Funded loan amount by gender/type
plt.figure(figsize=(9,6))
g = sns.violinplot(x="borrower_genders",y="funded_amount",data=loan_themes,order=['male','female','group'])
plt.xlabel("")
plt.ylabel("Funded Loan Amount",fontsize=12)
plt.title("Funded loan amount by borrower type/gender",fontsize=15)
plt.show()


# The boxplot element shows the median funded amount  loaned to female(s)  is comparatively lower than male and group. The shape of the distribution(on the upper end and wide in the middle) indicates the funded amount of the group is higly concentrated around the median

# **1.15.Lender Count Distribution by  Borrower Gender**

# In[ ]:


# plot distribution of lender count by Gender/Type
loan_themes['lender_count_lval'] = np.log(loan_themes['lender_count'] + 1)
(sns
  .FacetGrid(loan_themes, 
             hue='borrower_genders', 
             size=6, aspect=2)
  .map(sns.kdeplot, 'lender_count_lval', shade=True,cut=0)
 .add_legend()
)
plt.xlabel("Lender count")
plt.show()


# The Kernel density plot shows the distribution of lender count by Borrower gender. mean Lender_count for each gender and group varies at different levels. Lender_count distribution for female(s) has greater variation in the lower tail than the other two

# *** This Kernel is a work in Progress. Being Updated with External Datasets ***
