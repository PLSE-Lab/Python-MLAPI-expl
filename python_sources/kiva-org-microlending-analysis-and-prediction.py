#!/usr/bin/env python
# coding: utf-8

# <img src="https://www-kiva-org.global.ssl.fastly.net/cms/kiva_logo_2.png" alt="drawing" width="250px;"/>
# 
# <font size="1">(Kiva org logo accessed from Kiva.org website - copyright 2019)$</font>
# # Kiva.org Microlending Analysis
# 
# 
# ---
# 
# 
# ## Abstract: 
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. In this project, we intend to build models to estimate the poverty levels of residents in the regions where Kiva has active loans.
# 
# We analyze correlations between funded amounts and the sector/activity, country and the region's [Multidimensional Poverty Index (MPI)](http://hdr.undp.org/en/2019-MPI), a measure of poverty in a region.
# 
# We then predict funded amounts given parameters such as MPI, sector etc and also classify payment interval patterns. We also briefly attempt to analyze the data for bias in lending.
# 
# ---
# 
# ## Authors: 
# 
# Ishaan Malhi - UC Berkeley, M.Eng in EECS
# 
# Ziqian Qin - UC Berkeley, M.Eng. in EECS
# 
# ---
# 
# Submitted for the fulfillment of the Data 200 Graduate Student Project - 
# 
# Option 1. Link: http://www.ds100.org/fa19/gradproject/
# 
# Dataset: https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding
# 
# Data provided by Kiva.org and Kaggle.com licensed under the CC0: Public Domain License.

# 
# ## Tasks
# 
# 1. [x] Frame a question of your choice that can be addressed by identifying, collecting, and analyzing relevant data.
# 2. [x] Describe and obtain the data.
# 3. [x] Perform exploratory data analysis (EDA) and include in your report at least two (but probably many more) data visualizations.
# 4. [x] Describe any data cleaning or transformations that you perform and why they are motivated by your EDA.
# 5. [x] Apply relevant inference or prediction methods (e.g., linear regression, logistic regression, or classification and regression trees), including, if appropriate, feature engineering and regularization. Use cross-validation or test data as appropriate for model selection and evaluation. Make sure to carefully describe the methods you are using and why they are appropriate for the question to be answered.
# 6. [x] Summarize and interpret your results (including visualization). Provide an evaluation of your approach and discuss any limitations of the methods you used.
# 7. [x] Describe any surprising discoveries that you made and future work.
# 

# # Introduction
# 
# ## Kiva.org and Microlending
# > Kiva (commonly known by its domain name, Kiva.org) is a **501(c)(3) non-profit organization** that allows **people to lend money via the Internet to low-income entrepreneurs and students** in 77 countries. Kiva's mission is "to expand financial access to help underserved communities thrive."
# 
# > Kiva **relies on a network of field partners to administer the loans on the ground**. These field partners can be **microfinance institutions, social impact businesses, schools or non-profit organizations**. Kiva includes personal stories of each person who needs a loan so that their lenders can connect with their entrepreneurs on a human level.
# 
# <font size="1">https://en.wikipedia.org/wiki/Kiva_(organization)</font>
# 
# ### 1. Framing the Question
# 
# #### Objective
# 
# The dataset obtained is from Kiva.org's [Kaggle Competition in 2018](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding). The aim of the competition was provide Kiva.org with a better understanding of their impact on the localities they serve.
# 
# The original Kaggle competition does not provide any specific prediction problem and leaves the analysis open ended. In this project, we design our own objectives to understand patterns in the microlending space.
# 
# Some of our key objectives are:
# 
# 1. Run Exploratory Data Analysis (EDA) to find patterns or characteristics in the data to help Kiva.org get insights around lending patterns.
# 2. Try to predict the poverty levels of regions given loan amounts and sectors requiring loans.
# 3. Predict funded amounts, sectors and poverty levels of an area via supervised learning approaches.
# 4. **Explore bias in lending**: We explore if certain parts of society have a higher loan approval rate than others. This is an open ended analysis and our experiments try to show the **negative effects** of machine learning algorithms in a FinTech space if not analyzed for bias.
# 
# 

# In[ ]:


## Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px
pyo.init_notebook_mode()

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
titleid=0


# ### 2. Obtaining Data & Data Description
# 
# We will use the dataset provided by Kiva.
# 
# We may use additional dataset to help make predictions. The data is downloaded from the [Kaggle Competition](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding). We then upload it to Google Drive as a zip file, mount the drive contents and extract the contents of the zip file into the local Collab runtime.

# In[ ]:


# #@title Enter dataset file path in drive
# #@markdown Enter the full data path (e.g: /content/drive/My Drive/DS 200/)

# mount_point = './'  #@param {type: "string"}
# data_dir_path = './'  #@param {type: "string"}
# is_zip_file = True  #@param {type: "boolean"}
# file_name = 'data-science-for-good-kiva-crowdfunding.zip'  #@param {type: "string"}
# #@markdown ---


# In[ ]:


# # load data provided by Kiva
# dir_path = 'data-science-for-good-kiva-crowdfunding/' # local path
# if use_colab:
#     from google.colab import drive
#     drive.mount(mount_point)


# In[ ]:


# if is_zip_file:
#   import zipfile
#   zip_ref = zipfile.ZipFile(f'{data_dir_path}/{file_name}', 'r')
#   zip_ref.extractall(dir_path)
#   zip_ref.close()


# In[ ]:


dir_path = '../input/data-science-for-good-kiva-crowdfunding/'


# In[ ]:


loans = pd.read_csv(dir_path + 'kiva_loans.csv')
mpi_region_location = pd.read_csv(dir_path + 'kiva_mpi_region_locations.csv')
theme_ids = pd.read_csv(dir_path + 'loan_theme_ids.csv')
themes_by_region = pd.read_csv(dir_path + 'loan_themes_by_region.csv')


# Let's have a look at the data.
# 
# First let's see what we have about loans.

# In[ ]:


loans.head(3)


# In[ ]:


loans.shape


# In[ ]:


loans.describe()


# In[ ]:


loans.columns


# The meaning of each column:
# 
# - id: Unique ID for loan
# - funded_amount: Dollar value of loan funded on Kiva.org
# - loan_amount: Total dollar amount of loan
# - activity: Loan activity type
# - sector: Sector of loan activity as shown to lenders
# - use: text description of how loan will be used
# - country_code: 2-letter Country ISO Code
# - country: country name
# - region: name of location within country
# - currency: currency in which loan is disbursed
# - partner_id: Unique ID for field partners
# - posted_time: date and time when loan was posted on kiva.org
# - disbursed_time: date and time that the borrower received the loan
# - funded_time: date and time at which loan was fully funded on kiva.org
# - terminmonths: number of months over which loan was scheduled to be paid back
# - lender_count: number of lenders contributing to loan
# - tags: tags visible to lenders describing loan type
# - borrower_genders: gender of borrower(s)
# - repayment_interval: frequency at which lenders are scheduled to receive installments
# - date: date on which loan was posted

# Descriptions of ohter tables can be acquired from  https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding. We don't put them here.

# # 3. Exploratory Data Analysis

# Let's focus on loans for now.

# Apparently loans have different distributions in different countries. Figure 1 shows the top K(=20) countries with largest total amount of funded loans.

# In[ ]:


plt.figure(figsize=(15,10))
topK = 20
loan_amount_country = loans.groupby('country').sum()['funded_amount'].sort_values(ascending=False)
sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));
plt.ylabel('Country')
plt.xlabel('Total Funded Amount (USD 10M)')
titleid += 1
plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount');


# Phillipines and Kenya are the top funded countries. This corroborates Kiva.org's real life funding operations, that occure majoratarily in these countries. The top countries also seem to be largely developing countries.
# 
# Figure 2 shows the average amount of funded loans by country.

# In[ ]:


plt.figure(figsize=(15,10))
loan_amount_country = loans.groupby('country').mean()['funded_amount'].sort_values(ascending=False)
sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));

plt.ylabel('Country')
plt.xlabel('Average Funded Amount (USD 10M)')
titleid += 1
plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount');


# Cote D'Ivoire has extremely high average amount of funded loans.

# In[ ]:


loans.query(r'country == "Cote D\'Ivoire"')


# There is only one record, thus we redo the last plot by eliminating countries with less than 5 records.
# 
# Figure 3 shows the result.

# In[ ]:


plt.figure(figsize=(15,10))
useful_countries = list(loans.groupby('country').count().query('id >= 5').index)
loan_amount_country = loans.groupby('country').mean()['funded_amount'][useful_countries].sort_values(ascending=False)
sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));

plt.ylabel('Country')
plt.xlabel('Average Funded Amount (USD 10M)')
titleid += 1
plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount (Without Outliers)');


# We would like to see if the average amount of funded loans is related to the poverty level in these countries.
# 
# Figure 4 shows the points of amount of funded loands and MPIs, with the linear regression result.

# In[ ]:


topK = 40
mpis_country = mpi_region_location.groupby('country').mean()['MPI'].dropna()
countries = loan_amount_country.head(topK)
common_countries = mpis_country.index.intersection(countries.index)
df = pd.DataFrame({'mpis': mpis_country[common_countries], 'amount': countries[common_countries]} )
sns.lmplot(x='mpis', y='amount', data=df)
titleid += 1
plt.title(f'Figure {titleid}: Relationship between funded_amount and MPI');


# In general, countries with higher MPI tend to have lower average amounts of loans.
# 
# The relationship is not linear though.

# Questions we explore in the subsequent sections:
# 
# 1. What is the general trend in funded loans? How fast is Kiva.org adopted in countries?
# 
# 2. How large is a loan based on a sector? Do certain sectors like agriculture or small business get higher loans? Do they ask for higher loans?
# 
# 3. What is the percentage of irregular payments in countries, and how is it correlated to sector, country, activity and loan amount?
# 
# 4. Geo-Vizualization: Is there a certain clustering on loan amounts for neighboring countries? Do countries with similar MPI scores that are in different geographical regions get different loan amounts?
# 
# 5. Which countries have the strongest peer to peer networks?

# ## 1. What is the general trend in funded loans? How fast is Kiva.org adopted in countries?

# In[ ]:


## Let's take the top 20 countries with total amount of funding
top_countries_list = list(loans.groupby('country').sum()['funded_amount'].sort_values(ascending=False).head(10).index)
top_country_loans = loans.query(f'country in {top_countries_list}').copy()
top_country_loans.head()


# In[ ]:


# We seen some NaN values in the region and tags columns
for col in top_country_loans.columns:
  print(f'There are {len(top_country_loans[top_country_loans[col].isnull()])} records with NaN values in {col}.')


# Since funded time has a large number of NaN values, let's use posted_time to analyze Kiva.org adoption.
# Later, we will clean this data and fill or drop specific null columns.
# 
# [Jump to Null value cleaning](#scrollTo=eaN0yeXpeko5&line=3&uniqifier=1)
# 

# In[ ]:


import datetime
top_country_loans.loc[:, 'posted_time'] = pd.to_datetime(top_country_loans.posted_time)
## Lets add the year column separately to track growth over time.
## Parse datetime and set the day to the 1st of every month and club amounts together for easier plotting
top_country_loans.loc[:, 'posted_month_year'] = top_country_loans.posted_time.apply(lambda x: datetime.datetime(x.year, x.month, 1)) 

loan_data = top_country_loans.groupby(['posted_month_year', 'country']).sum()['funded_amount'].reset_index().sort_values(by='posted_month_year')
loan_data


# In[ ]:


fig = go.Figure()

for index, country in enumerate(loan_data.country.unique()):
  query = f"country == '{country}'"
  fig.add_trace(
      go.Scatter(
          y=loan_data.query(query)['funded_amount'], 
          x=loan_data.query(query)['posted_month_year'], 
          text=country,
          mode='lines',
          name=country,
          visible="legendonly" if index <= 4 else True
      )
  )

fig.update_layout(
      width=1200,
      height=1000,
      autosize=True,
      template="plotly_white",
  )

titleid += 1
fig.update_layout(
    title=f"Figure {titleid}: Funded Amount in top 10 countries over time",
    xaxis_rangeslider_visible=True,
    xaxis_title="Year",
    yaxis_title="Loan Amount in USD"
)

fig.show()


# According to Figure 5, Kenya seems to have shown high growth in the amount that was funded in the first few months of launch.
# 
# An odd occurence is that we see a sudden downward spike in July 2017 for all countries, let's check what these values are.

# In[ ]:


top_country_loans[top_country_loans.posted_time.dt.year == 2017][top_country_loans.posted_time.dt.month == 7]


# We see that the funded amount in July 2017 is pretty low, some have zero values in `funded_amount`. There also seem to be many NaN values for funded_amount, gender etc. Since this is the tip, it's highly likely that the data is incomplete.

# ## How large is a loan based on a sector? Do certain sectors like agriculture or small business get higher loans? Do they ask for higher loans?

# In[ ]:


sector_loans = loans.groupby('sector').sum()[['funded_amount', 'loan_amount']].reset_index()
sector_loans


# In[ ]:


from plotly.subplots import make_subplots

fig = go.Figure(data=[go.Pie(labels=sector_loans.sector, values=sector_loans.funded_amount)])
    
titleid += 1
fig.update_layout(dict(
    title=f'Figure {titleid}: Funded Loan Amount per Sector',
    width=800,
    height=800
))

fig.show()


# From Figure 6 we see that Agriculture, Food and Retails are the largest sectors that get funded on Kiva.org. Let's see if these sectors are different across countries.

# In[ ]:


top_country_sector_loans = top_country_loans.groupby(['sector', 'country']).sum()['funded_amount'].reset_index()
top_country_sector_loans


# In[ ]:


fig = go.Figure(
    data=[
          go.Pie(
              labels=top_country_sector_loans.query("country == 'Paraguay'").sector, 
              values=top_country_sector_loans.query("country == 'Paraguay'").funded_amount
              )
          ]
        )

titleid += 1  
fig.update_layout(dict(
    title=f'Figure {titleid}: Funded Loan Amount per Sector',
    width=800,
    height=800
))

fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            buttons=list([
                dict(
                    args=[
                        {
                            "labels": [top_country_sector_loans.query(f"country == '{country}'").sector], 
                            "values": [top_country_sector_loans.query(f"country == '{country}'").funded_amount],
                            "title": [
                                f'Funded Loan Amount per Sector for {country}'
                            ]
                        }],
                    label=country,
                    method="restyle"
                ) for country in list(top_country_sector_loans.country.unique())
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.8,
            yanchor="top"
        ),
    ]
)

fig.show()


# From Figure 7 we see that most developing nations have Agriculture, Retail and Food as the most funded sectors. Thus, the sector can be a good indicator of predicting the likelihood of getting a microloan and/or the funded amount of the loan.

# ## What is the percentage of irregular payments in countries, and how is it correlated to sector, country, activity and loan amount?

# Here we seek to analyze the distribution of irregular payments. Certain parameters like country, activity, sector and loan amount will help us understand the likelihood of getting a loan based on whether the loan payment type is irregular.
# 
# In later sections, we classify a loan's payment type (regular, irregular) based on these parameters. Before classifying we analyze if there's sufficient distribution of irregular payment types across these parameters.

# In[ ]:


loans.query("repayment_interval == 'irregular'").head(2)


# In[ ]:


repayment_data = loans.groupby(['repayment_interval', 'country', 'sector', 'activity']).sum()['funded_amount'].reset_index()
repayment_data


# In[ ]:


fig = go.Figure(
    data=go.Splom(
        dimensions=[
                    dict(
                        label=label,
                         values=list(repayment_data[label])
                        ) for label in list(repayment_data.columns)
                    ]
))

titleid += 1
fig.update_layout(
    title=f'Figure {titleid}: Loan Repayment Interval',
    dragmode='select',
    width=1200,
    height=1200,
    hovermode='closest',
)

fig.show()


# In Figure 8, we see at the top right graph an interesting correlation between `funded_amount` and `repayment_interval`. Let's see this graph up close per sector and per country.

# In[ ]:


# for col in list(repayment_data.iloc[:, 1:].columns:
payment_interval_dist = repayment_data.groupby(['country', 'repayment_interval']).count()['sector'].reset_index()
payment_interval_dist


# In[ ]:


fig = go.Figure(
    data=[
          go.Pie(
              labels=payment_interval_dist.repayment_interval, 
              values=payment_interval_dist.sector
              )
          ]
        )
    
titleid += 1
fig.update_layout(dict(
    title=f'Figure {titleid}: Distribution of Repayment Intervals for Each Country',
    width=800,
    height=800
))

fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            buttons=list([
                dict(
                    args=[
                        {
                            "labels": [payment_interval_dist.query(f'country == "{country}"').repayment_interval], 
                            "values": [payment_interval_dist.query(f'country == "{country}"').sector],
                            "title": [
                                f'Funded Loan Amount per Repayment Interval for {country}'
                            ]
                        }],
                    label=country,
                    method="restyle"
                ) for country in list(payment_interval_dist.country.unique())
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.8,
            yanchor="top"
        ),
    ]
)

fig.show()


# From Figure 9 we see that countries such as Rwanda has a majority of loans under irregular repayment intervals. We will investigate during Principal Component Analysis (PCA) to see if `repayment_interval` plays a role in deciding the `funded_amount`.

# In[ ]:


# for col in list(repayment_data.iloc[:, 1:].columns:
payment_interval_dist_sector = repayment_data.groupby(['sector', 'repayment_interval']).count()['country'].reset_index()
payment_interval_dist_sector.head(10)


# In[ ]:


fig = go.Figure(
    data=[
          go.Pie(
              labels=payment_interval_dist_sector.repayment_interval, 
              values=payment_interval_dist_sector.country
              )
          ]
        )
    
titleid += 1
fig.update_layout(dict(
    title=f'Figure {titleid}: Distribution of Repayment Intervals for Each Sector',
    width=800,
    height=800
))

fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            buttons=list([
                dict(
                    args=[
                        {
                            "labels": [payment_interval_dist_sector.query(f'sector == "{sector}"').repayment_interval], 
                            "values": [payment_interval_dist_sector.query(f'sector == "{sector}"').country],
                            "title": [
                                f'Funded Loan Amount per Repayment Interval Type for {sector}'
                            ]
                        }],
                    label=sector,
                    method="restyle"
                ) for sector in list(payment_interval_dist_sector.sector.unique())
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.8,
            yanchor="top"
        ),
    ]
)

fig.show()


# From Figure 10 we see that most sectors have a majority repayment interval under monthly payments. **This is surprising even for sectors like Agriculture** where a microloan would be expected to be paid after a long time, or paid irregularly due to the seasonal aspect of Agriculture, especially in developing countries.

# ## Geo-Vizualization: Is there a certain clustering on loan amounts for neighboring countries? Do countries with similar MPI scores that are in different geographical regions get different loan amounts?

# In[ ]:


loan_amount_per_country = loans.groupby(['country', 'country_code']).agg(['sum', 'count'])['funded_amount'].reset_index().set_index('country')
loan_amount_per_country['sum'] =loan_amount_per_country['sum'].astype(int)
loan_amount_per_country


# In[ ]:


# # The country codes are not in ISO Alpha format, let's set these.

iso_list = mpi_region_location.groupby(['ISO', 'country']).mean()['MPI'].reset_index().set_index('country')

loan_amount_per_country['country_code'] = iso_list['ISO']
## Let's drop the countries without any country codes
## For the countries with NaN values, set the code to the first three letters.
## Future: maybe scrape wikipedia/un sites to get info. Currently the data is not easily scrapped.

loan_amount_per_country.loc[loan_amount_per_country.country_code.isnull(), 'country_code'] = loan_amount_per_country[loan_amount_per_country.country_code.isnull()].apply(lambda row: row.name[:3].upper(), axis=1)
loan_amount_per_country


# In[ ]:


titleid += 1
fig = px.scatter_geo(loan_amount_per_country.reset_index(), locations="country_code", color="country",
                     hover_name="country", size="sum",
                     projection="natural earth",
                     title=f"Figure {titleid}: Geo-Visualization of Funded Amount")
fig.show()


# From Figure 11 we see certain similarly sized bubbles in Africa (Kenya, Uganda, Rwanda) and parts of south-central America. We also see Phillipines and Cambodia as the largest markets in Southeast Asia. Generally, there is a certain clustering among neighboring countries for the total amount of funded loans for the time period of the dataset. What makes this observation important is that microlending can have a spillover effect into neighboring countries. It would be interesting to see Kiva.org's growth strategies into neighboring countries and if they were largely driven by word of mouth.

# In[ ]:


"""
Let's now analyze the number of loans, the funded amount and the MPI of countries.
"""
loan_amount_per_country['MPI'] = iso_list['MPI']
loan_amount_per_country = loan_amount_per_country.dropna() ## Drop NaN MPI values to not skew the plot

loan_amount_per_country.head(10)


# In[ ]:


# # # ## Analyze class size and cost of attendance
fig = px.scatter(
    loan_amount_per_country.reset_index(), 
    x="MPI",
    y="sum", 
    color="country", 
    hover_name="country",
    size="count"
)

titleid += 1
fig.update_layout(
    height=900,
    width=900,
    xaxis_title="MPI",
    yaxis_title="Total Funded Amount",
    title=f'Figure {titleid}: MPI vs Funded Amount vs Number of Funded Loans'
)
fig.show()


# From Figure 12 we see that countries with similar MPI have a large difference between the total funded amount. Specifically, Cameroon (MPI 0.202, funded_amount=\$875K, count=2230) is not given as many funds as Kenya (MPI = 0.209, sum=\$32.2M, count=75825). There are subtleties here in the data we do not capture such as MPI between regions and microloans per region which could explain this difference between the two countries. It is also possible that Cameroon and Kenya have different peer to peer networks, and Kenya's p2p network is stronger.
# 
# We thus analyze the strength (in terms of lender count) of different countries.

# ## Which countries have the strongest peer to peer networks?
# 
# Hypothesis: The number of lenders for a loan should be a good predictor of the strength of the p2p network. Thus, a stronger p2p network should lead to a higher total funded amount.

# In[ ]:


country_peer_network = loans.groupby(['country', 'country_code']).agg(['sum', 'count'])[['funded_amount', 'lender_count']].reset_index().iloc[:, :5]
country_peer_network.head(3)


# In[ ]:


## Let's squash the column levels
country_peer_network['funded_amount_sum'] = country_peer_network['funded_amount']['sum']
country_peer_network['funded_amount_count'] = country_peer_network['funded_amount']['count']
country_peer_network['lender_count_sum'] = country_peer_network['lender_count']['sum']

country_peer_network.head(10)


# In[ ]:


# # # ## Analyze class size and cost of attendance
fig = px.scatter(
    country_peer_network, 
    x="lender_count_sum",
    y="funded_amount_sum", 
    color="country", 
    hover_name="country",
    size="funded_amount_count"
)

titleid += 1
fig.update_layout(
    height=900,
    width=900,
    xaxis_title="Total Lender Count",
    yaxis_title="Total Funded Amount",
    title=f'Figure {titleid}: Lender Count vs Funded Amount vs Number of Funded Loans'
)
fig.show()


# From Figure 13 We see there is a strong linear correlation between total funded amount and total lender count for nations. **This is somewhat expected and validates our hypothesis**. **As the number of lenders grows, so does the capital/funds available in a peer to peer network, leading to larger amounts**. We see that the radius of the circles also grow, i.e the total count of funded loans also grows as lender count grows.
# 
# What's **interesting to note is that the points on the graph are towards the upper left of what a linear regression line would look like**. This means that countries with lower number of lenders also get higher funded amounts. Therefore, a stronger peer to peer microloan network might not mean more lenders, but lenders willing to invest larger amounts of money.
# 
# **We now see the reason behind the large difference between Kenya and Cameroon's total funded amounts. Cameroon has very few lenders (~29k) while Kenya has ~1M lenders on the Kiva.org network. Even though the two countries have similar MPIs, their funded amounts differ.** Note however, that this could also be due to Kiva.org not pushing into Cameroon as aggresively, differences in cultural norms around lending and various other societal factors that we do not show.
# 
# We will analyze the correlation between lender count and funded amount in the next sections.
# 
# [Jump to Lender Count vs Funded Amount Analysis](#scrollTo=C3_ywSKU-tbb&line=3&uniqifier=1)

# ##### Clustering 
# 
# We have seen above that neighboring countries seem to have similar amounts in funding. We now aim to see if countries can be categorized by the strength of the peer networks, and how stronger peer networks influence loan amounts.

# Q: Do certain countries with similar peer to peer network strength and geographic location have similar funded amounts?
# 
# While strong peer to peer networks are a strong metric to influence funded amounts, here we analyze if countries next to each other have the same strength of peer to peer networks.
# 
# After clustering neighboring countries, we see if the average MPI of a cluster has an influence over the funded amount of the cluster to understand the impact of a microloan over an entire geographical cluster. Analyzing spillover is important to understand the effect of microloans in lifting entire areas out of poverty. 

# In[ ]:


country_loans = pd.merge(
    loans, 
    mpi_region_location, 
    how='inner', 
    on='country'
    )[['country','MPI', 'lat', 'lon', 'lender_count', 'funded_amount']].groupby('country').agg(
    {
      'lat': 'mean', 
      'lon': 'mean', 
      'lender_count': 'sum', 
      'funded_amount': 'sum', 
      'MPI': 'mean'
    }).reset_index()
country_loans = country_loans.dropna()
country_loans.head(2)


# In[ ]:


titleid += 1
fig = px.scatter(
    country_loans, 
    x='lat', 
    y='lon',
    title=f'Figure {titleid}: Region vs total lender count',
    size="lender_count",
    hover_name="country"
)
fig.show()

titleid += 1

fig = px.scatter(
    country_loans, 
    x='lat', 
    y='lon',
    title=f'Figure {titleid}: Region vs total average MPI',
    size="MPI",
    hover_name="country"
)

fig.show()


# Here we see what we saw on the world map before as well - neighboring countries have high lender counts.
# 
# We also see that generally the MPI of neighboring regions is the same. Poorer countries (higher MPI) are clustered together.
# 
# Let's cluster these areas and use the clusters to generate trend lines between funded amount and MPI.

# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(country_loans[['lat', 'lon']])

country_loans['cluster'] = kmeans.labels_


# In[ ]:


from sklearn.metrics import silhouette_score

silhouette_score(country_loans[['lat', 'lon']], kmeans.predict(country_loans[['lat', 'lon']]))


# In[ ]:


## Let's get the optimal number of clusters
geo_labels = ['lat', 'lon']

for n_clusters in range(5, 15):
    labels = KMeans(n_clusters=n_clusters, random_state=10).fit_predict(country_loans[geo_labels])
    silhouette_avg = silhouette_score(country_loans[geo_labels], labels)

    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")


# In[ ]:


## We see that the optimal n_clusters is 12
kmeans = KMeans(n_clusters=12).fit(country_loans[['lat', 'lon']])

country_loans['cluster'] = kmeans.labels_


# In[ ]:


titleid += 1
fig = px.scatter(
    country_loans, 
    x='lat', 
    y='lon',
    title=f'Figure {titleid}: Clustered countries vs total lender count',
    size="lender_count",
    hover_name="country",
    color='cluster'
)
fig.show()


# In[ ]:


## We now analyze the trend between clusters, MPI and funded_amounts
clustered_loans = country_loans.groupby(['cluster']).mean().reset_index()
clustered_loans


# In[ ]:


titleid += 1
px.scatter(
    clustered_loans, 
    x='MPI', 
    y='funded_amount',
    color='cluster', 
    trendline='ols', 
    title=f'Figure {titleid}: Average Funded Amount vs Average MPI',
    labels={
        'MPI': 'Average Multidimensional Poverty Index (MPI)',
        'funded_amount':'Average Funded Amount'
    }
    )


# We see in the above graph that clusters with the same average MPI receive very different funding. Let's see what these countries are. We will look at cluster 2 and 4.

# In[ ]:


country_loans.query('cluster == 2').country


# In[ ]:


country_loans.query('cluster == 4')


# We see above some discrepancies within clusters. In cluster 2, while Pakistan and Tajikistan have widely different MPI values, they both receive similar funded amounts and have comparable lender counts.
# 
# These countries are neighboring and thus could be seeing a spillover of microlending, however the funded amount is small compared to cluster 4.
# 
# Within Cluster 4, Indonseia and Philippines receive large amounts of funding and they have a much lower MPI than Timor-Leste.
# 
# We see a problem with this approach because of outliers. The average MPI values of both these clusters is skewed due to outliers and thus clustering and generating inference based on geographic location is **NOT** a good approach to predicting funded amounts.

# # 4. Data Cleaning & Transformation 
# 
# We now start with predicting various features to understand correlations. We form questions around these predictions to understand the value of a peer to peer network provided by companies like Kiva.org.

# We need to clean the data before we use it to train models.
# 
# ## Handling Null values
# First we will handle null values.

# In[ ]:


display(loans.isnull().sum())
display(mpi_region_location.isnull().sum())


# The simplest way to handle them is to drop any rows with null values. Potentially, we can update the data and fill in the null values. However, for this project we choose to drop null values to prevent zero value skew in financial data. Note that removing null values can also skew our inference, but dropping null values is safer and we can exclude data points (in this case certain countries), thereby excluding certain countries. Excluding certain countries should be safe based on our prior analysis that shows countries geographically close have similar characteristicss with respect to loan amounts and MPI barring a few outliers.

# In[ ]:


loans = loans.dropna()
mpi_region_location = mpi_region_location.dropna()


# ## [Cleaning & Transforming `borrower_gender` column](#scrollTo=6ul_eBCb9moN)

# 1. ## [Principal Component Analysis](#scrollTo=Ns-djwZ51yir&line=4&uniqifier=1)
# 
# 

# ## [One Hot Encoding](#scrollTo=xxJ4rzkgXZGT)

# # 5. Prediction and Inference
# 
# Now that the null values are dropped, we make predictions based on certain functional questions. 

# #### Q1. Is the `funded_amount` a good predictor of the MPI of a region?

# Task: Predict MPI using data in loans using Linear Regression.
# 
# From our EDA, we see that countries with different MPI values can get different funded amounts. In this task we aim to predict the MPI of a region using loan data like `funded_amount`, `term_in_months` and `lender_count`. If loan parameters are a good predictor of the MPI of a region, it means changes in these parameters can have a large influence in a region's MPI, and thus showing that micrloans have a demonstrated impact on the MPI of a region.

# In[ ]:


# data design
mpi_data = mpi_region_location.groupby(['country', 'region']).mean()['MPI']
loans_data = loans.groupby(['country', 'region']).mean().drop(columns=['id', 'partner_id'])
data_index = mpi_data.index.intersection(loans_data.index)


# In[ ]:


# loss function
def l2(pred, true):
    return ((pred - true) ** 2).mean()


# In[ ]:


loans_data


# In[ ]:


model = LinearRegression()

X_0 = loans_data.loc[data_index]
y_0 = mpi_data.loc[data_index]


# In[ ]:


# train
def train_linear_regression(X, y):
    model = LinearRegression()

    kf = KFold(n_splits=5, shuffle=True)
    train_loss = []
    val_loss = []
    for train_id, val_id in kf.split(X):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id]
        y_val = y.iloc[val_id]
        model.fit(X_train, y_train)
        train_loss.append(l2(model.predict(X_train), y_train))
        val_loss.append(l2(model.predict(X_val), y_val))
    print('training loss = {}\nvalidation loss = {}'.format(np.mean(train_loss),np.mean(val_loss)))      


# In[ ]:


# train
train_linear_regression(X_0, y_0)
      
# use all data to train
model.fit(X_0, y_0)
plt.figure(figsize=(14, 14))
for i in range(X_0.shape[1]):
    plt.subplot(2, 2, i + 1)
    plt.xlabel(X_0.columns[i])
    plt.ylabel('residual')
    plt.scatter(X_0.iloc[:, i], y_0 - model.predict(X_0))
titleid += 1
plt.suptitle(f'Figure {titleid}: Residual Plots');


# From Figure 14, we find that the residuals are not symmetric, indicating our linear regression does not perform well on the data.
# 
# The validation loss is also not quite satisfactory. 
# 
# So our data design can be improved. Since there is no obvious patterns in the residual plots, it might be hard to explore a good way to transform the data. Therefore we currently skip this approach.
# 
# Our hypothesis of predicting MPI is invalidated to a certain extent. However, this does not mean microloans do not have an impact, but rather that we need more data points to understand factors that influence the MPI of a region.

# #### Q2. Can we predict the funded amount a region is likely to get, given the mpi and sector?
# 
# #### Task: Predict the amount of loans from mpi and sector:
# 
# This could be helpful for us to know about the regional financial needs in different fields.
# 
# We will first perform simple Linear Regression.

# In[ ]:


# data design
# y: loan_amount
# X: mpi, sector(dummy encoding)
X_1 = loans[['country', 'region', 'loan_amount']]
X_1 = pd.concat([X_1, pd.get_dummies(loans['sector'])], axis=1) # one hot encode sector values
X_1 = X_1.merge(mpi_region_location[['country', 'region', 'MPI']])
y_1 = X_1['loan_amount']
X_1 = X_1.drop(columns=['country', 'region', 'loan_amount'])


# Visualize the data design.

# In[ ]:


data = loans[['country', 'region', 'loan_amount', 'sector']]
data = data.merge(mpi_region_location[['country', 'region', 'MPI']])
plt.figure(figsize=(10, 8))
titleid += 1
plt.title(f'Figure {titleid}: Loan Amount Distribution by Sector and MPI')
sns.scatterplot(x='MPI', y='loan_amount', hue='sector', data=data);


# Figure 15 is messy. We choose the major sectors and remove records with amounts > 10000.

# In[ ]:


plt.figure(figsize=(10, 8))
sectors = data['sector'].value_counts().head(5).index.to_series(name='sector')
titleid += 1
plt.title(f'Figure {titleid}: Loan Amount(<=10000) Distribution by Major Sectors and MPI')
sns.scatterplot(x='MPI', y='loan_amount', hue='sector', data=data.merge(sectors).query('loan_amount <= 10000'));


# There exists some patterns in Figure 16.
# 
# The amount of personal use are usually low (< 2000).
# 
# The agriculture use is less frequent in regions with higher MPI.
# 
# The majority of educational loans are in regions with quite low MPI.

# Perform a Linear Regression on all sectors:

# In[ ]:


train_linear_regression(X_1, y_1)


# Next we will try to use random forest regressor.

# In[ ]:


def train_random_forest_regressor(X, y, **params):
    model = RandomForestRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True)
    train_loss = []
    val_loss = []
    for train_id, val_id in kf.split(X):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id]
        y_val = y.iloc[val_id]
        model.fit(X_train, y_train)
        train_loss.append(l2(model.predict(X_train), y_train))
        val_loss.append(l2(model.predict(X_val), y_val))
    return train_loss, val_loss    


# In[ ]:


train_loss, val_loss = train_random_forest_regressor(X_1, y_1, n_estimators=10)
print('training loss = {}\nvalidation loss = {}'.format(np.mean(train_loss), np.mean(val_loss)))


# Both the training loss and validation loss is smaller.
# 
# We will try to tune the `n_estimators` and `max_depth` and see if the result changes.

# In[ ]:


def heatmap_random_forest_regressor(ns, depths, X, y):
    loss = np.zeros((len(ns), len(depths)))
    for i in range(len(ns)):
        for j in range(len(depths)):
            _, val_loss = train_random_forest_regressor(X, y, n_estimators=ns[i], criterion='mse', max_depth=depths[j])
            loss[i, j] = np.mean(val_loss)
    sns.heatmap(loss, cmap='YlGnBu', xticklabels=depths, yticklabels=ns)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    global titleid
    titleid += 1
    plt.title(f'Figure {titleid}: Random Forest Regressor MSE')
    return loss


# In[ ]:


ns_reg = np.arange(1, 21, 5)
depths_reg = np.arange(1, 61, 5)
loss_forest_0 = heatmap_random_forest_regressor(ns_reg, depths_reg, X_1, y_1)


# Let's adjust the search space do it again.

# In[ ]:


fine_ns_reg = np.arange(5, 41, 5)
fine_depths_reg = np.arange(7, 23, 3)
loss_forest_1 = heatmap_random_forest_regressor(fine_ns_reg, fine_depths_reg, X_1, y_1)


# In[ ]:


loss_forest_1.min()


# Figure 18 shows the result does not vary a lot in the specified range.
# 
# So we can choose `n_estimators=20` and `max_depth=10`. 
# 
# The minimal loss would be around 1700000, which is less than simple linear regression.

# #### Q3. Do certain sectors contribute more to the MPI and amount of loans?
# 
# Task: Can we predict sector using MPI and amount of loans
# 
# Predicting a sector can help microloan providers understand the best sectors to invest in given fixed capital/funds for a given region (via the region's MPI). On paper, this approach ignores the feasibility of an actual sector in a region. Certain regions might not have the infrastructure to support a sector even though the loan data shows the best possible sector.
# 
# First we will use simple Logistic Regression.

# In[ ]:


# data design
X_logi = data[['MPI', 'loan_amount']]
y_logi = data['sector']


# In[ ]:


def train_logistic_regression(X, y):
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    kf = KFold(n_splits=5, shuffle=True)
    train_acc = []
    val_acc = []
    for train_id, val_id in kf.split(X):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id]
        y_val = y.iloc[val_id]
        model.fit(X_train, y_train)
        train_acc.append((model.predict(X_train) == y_train).mean())
        val_acc.append((model.predict(X_val) == y_val).mean())
    print('training accuracy = {}\nvalidation accuracy = {}'.format(np.mean(train_acc),np.mean(val_acc)))


# In[ ]:


train_logistic_regression(X_logi, y_logi)


# Draw the confusion matrix Figure 19:

# In[ ]:


def show_logistic_confusion_matrix(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)
    global titleid
    titleid += 1
    plt.title(f'Figure {titleid}: Confusion Matrix for Logistic Regression')
    sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')

show_logistic_confusion_matrix(X_logi, y_logi)


# Then try to predict the sector using random forest with the same data.

# In[ ]:


def train_random_forest_classifier(X, y, heatmap=False, **params):
    model = RandomForestClassifier(**params)
    kf = KFold(n_splits=5, shuffle=True)
    train_acc = []
    val_acc = []
    for train_id, val_id in kf.split(X):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id]
        y_val = y.iloc[val_id]
        model.fit(X_train, y_train)
        train_acc.append((model.predict(X_train) == y_train).mean())
        val_acc.append((model.predict(X_val) == y_val).mean())
        if heatmap:
            print(model.classes_)
            sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')
        plt.show()
    return train_acc, val_acc


# In[ ]:


train_acc, val_acc = train_random_forest_classifier(X_logi, y_logi, n_estimators=10, criterion='gini')
print('training accuracy = {}\nvalidation accuracy = {}'.format(np.mean(train_acc),np.mean(val_acc)))


# Draw the confusion matrix of random forest classifier:

# In[ ]:


def draw_decisicon_boundary(X, y, model):
    sns_cmap = ListedColormap(np.array(sns.color_palette('Paired'))[:, :])

    xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 30), np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 30))
    Z_string = model.predict(np.c_[xx.ravel(), yy.ravel()])
    categories, Z_int = np.unique(Z_string, return_inverse = True)
    Z_int = Z_int.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z_int, cmap = sns_cmap)
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
    plt.legend(proxy, categories)
    plt.xlabel('MPI')
    plt.ylabel('funded amount')
    global titleid
    titleid += 1
    plt.title(f'Figure {titleid}: Decision Boundaries of Random Forest Classifier');


# In[ ]:


def show_forest_confusion_matrix(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=10, criterion='gini')
    model.fit(X_train, y_train)
    plt.figure(figsize=(10, 15))
    plt.subplot(2,1,1)
    global titleid
    titleid += 1
    plt.title(f'Figure {titleid}: Confusion Matrix for Random Forest Classifier')
    sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')
    plt.subplot(2,1,2)
    draw_decisicon_boundary(X, y, model)

show_forest_confusion_matrix(X_logi, y_logi)


# It seems random forest achieves high accuracy on both the training set and the validation set.
# 
# And the confusion matrices is also better than that of logistic regression.
# 
# From a set of experiments, we find that the decision boundaries are not stable, however the accuracy keeps relatively stable.
# 
# We will tune the parameters of the random forest to see if the result may vary.
# 
# For now, we focus on the number of estimators and the max depth.

# In[ ]:


def heatmap_random_forest_classifier(ns, depths):
    acc = np.zeros((len(ns), len(depths)))
    for i in range(len(ns)):
        for j in range(len(depths)):
            _, val_acc = train_random_forest_classifier(X_logi, y_logi, n_estimators=ns[i], criterion='gini', max_depth=depths[j])
            acc[i, j] = np.mean(val_acc)
    sns.heatmap(acc, cmap='YlGnBu', xticklabels=depths, yticklabels=ns)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    global titleid
    titleid += 1
    plt.title(f'Figure {titleid}: Random Forest Classifier Accuracy')
    return acc


# In[ ]:


ns = np.arange(1, 42, 10)
depths = np.arange(8, 25, 5)
acc_forest_0 = heatmap_random_forest_classifier(ns, depths)


# In[ ]:


acc_forest_0.max()


# In our experiments, we find that the optimal hyperparameters are not quite stable. 
# 
# Considering the time of computation, we will suggest using `max_depth=15` and `n_estimators=25`. 
# 
# In general, the best validation accuracy could be above 0.50, which is better than logistic regression.
# 
# Draw the decision boundaries with above parameters:
# 
# The boundaries seems a little more stable.
# 
# 

# In[ ]:


model = RandomForestClassifier(n_estimators=25, max_depth=15, criterion='gini')
X_train, _ , y_train, _ = train_test_split(X_logi, y_logi, test_size=0.2)
model.fit(X_train, y_train)
plt.figure(figsize=(10, 10))
draw_decisicon_boundary(X_logi, y_logi, model)


# ### Analyzing Bias in Lending
# 
# Bias in consumer lending is a prominent topic of research. We see if there are certain sections of society that get larger loans, or are more likely to get a higher loan using simple loan approval algorithms that companies commonly use.

# In[ ]:


loans.columns


# In[ ]:


## Clean gender column to keep a single gender value, which is the first gender value
loans.borrower_genders = loans.borrower_genders.str.split(", ").apply(lambda x: list(set(x))[0])


# In[ ]:


## We know there was a linear relationship between lender_count and funded_amount, let's analyze the slopes of these
## and split based on certain columns
def split_show_linear(df, separator, use_plotly=False):
    temp_df = df.groupby([separator, 'lender_count']).sum()['funded_amount'].reset_index()
    if use_plotly:
        global titleid
        titleid += 1
        fig = px.scatter(temp_df, x="lender_count", y="funded_amount", facet_col=separator, color=separator, trendline="ols",
                         title=f'Figure {titleid}')
        fig.show()
    else:
        sns.lmplot(x="lender_count", y="funded_amount", hue=separator, data=temp_df)
    
separator='borrower_genders'
split_show_linear(loans, separator, True)


# In[ ]:


#The data seems overplottled for low lender counts, let's group this into two segments
separator='borrower_genders'
split_show_linear(loans.query('lender_count < 200'), separator, True)


# In[ ]:


separator='borrower_genders'
split_show_linear(loans.query('lender_count >= 200'), separator, True)


# In[ ]:


titleid += 1
px.bar(
    loans.groupby(['borrower_genders', 'sector']).sum()['funded_amount'].reset_index(),
    x='borrower_genders',
    y='funded_amount',
    color='sector',
    barmode='group',
    title=f'Figure {titleid}: Gender specific funded amounts per sector'
)


# We see that a large number of women get high funded amounts but for lower lender counts. 
# Kiva.org loans are largely used to fund women led projects (81% of borrowers are women according to [this](https://www.kiva.org/about/impact) report), however this regression shows 
# that these are projects with few lender counts. When lender counts are >= 200 and < 1200, the regression lines
# show that men get higher funded amounts. This doesn't give us any strong evidence of algorithmic bias, it just gives
# us some form of loose correlation. 
# 
# **An important note here is that a machine learning model that predicts funded amount based on other features, that takes the gender column as a feature will fit a curve that will predict a higher funded amount of men for lender counts >= 200 < 1200.** The question of whether to include gender data into a model is thus questionable. A  model can predict higher amounts of women where lender counts <= 200 as well. Moreover, gender could be embedded into the activity column if women perform specific activities that they need funds for. For e.g, in the bar plot above, we see that there are relatively more women in the clothing sector. Does that mean that including a sector information as a feature also introduces bias?
# 
# Further analysis on the region and number of rejected loans would be needed to establish bias. That said, higher lender count leading to higher funded amounts of males might be something to look into. Maybe these patterns are true for certain sectors, regions etc. We leave this question open for future work.
# 

# ## Classifying repayment intervals
# 
# - Repayment intervals (monthly, one time (bullet), irregular etc) for a loan get often determine the chances of getting a loan approved. Since we do not have the loan approval status, we use repayment interval as a loose proxy.
# 
# - If a loan has a high probability of being classified as an irregular repayment type, lenders might not want to fund the loan. For one time payments, the lending amount and period might determine willingness to fund. We don't analyze one time repayment and thus use binary classification to classify monthly (regular) or regular payments and club bullet(one time) payments with irregular payments.

# In[ ]:


## Let's analyze the spread of repayment interval with respect to funded_amounts and sectors
plt.figure(figsize=(20,10));
temp_df = loans.groupby(['repayment_interval', 'sector']).sum()['funded_amount'].reset_index()
titleid += 1
px.bar(
    temp_df, 
    color='sector', 
    y='funded_amount', 
    x='repayment_interval', 
    barmode='group',
    title=f'Figure {titleid}: Loan Repayment Intervals vs Funded Amounts per sector',
)


# We see some distribution between sector and funded_amount when it comes to the distribution of `repayment_interval`.
# Let's run PCA to find the best features to use to predict the `repayment_interval`.
# 
# We pick only the continuous variables, since PCA on categorical variables won't give us any meaningful data.

# ### Data Cleaning

# In[ ]:


## We join two dataframes to get geo location values as well.
mpi_region_location.country = mpi_region_location.country.astype('str')
loans.country = loans.country.astype('str')
full_data = pd.merge(loans, mpi_region_location, on='country', how='left')


# In[ ]:


numerical_features = ['MPI', 'lat', 'lon', 'funded_amount', 'loan_amount', 'term_in_months', 'lender_count']
categorical_features = ['sector', 'activity', 'country', 'region_x', 'currency']

features = numerical_features + categorical_features
features


# ### Train Test Split

# In[ ]:


## Let's split the data into train and test
outcome_variable = 'repayment_interval'
full_data = full_data[features + [outcome_variable]] ## Select only certain starting features
full_data = full_data.dropna()
X_train, X_test, Y_train, Y_test = train_test_split(full_data[features], full_data[outcome_variable])


# ### Principal Component Analysis (PCA)

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
## Clean and transform data
scaler = StandardScaler()
centered_data = scaler.fit_transform(X_train[numerical_features])

pca = PCA()
pca.fit(centered_data)


# In[ ]:


fig, ax = plt.subplots(1, figsize=(6,5))

# plot explained variance as a fraction of the total explained variance
ax.plot(pca.explained_variance_ratio_)

# mark the 1th principal component
ax.axvline(5, c='k', linestyle='--')

ax.set_xlabel('PC index')
ax.set_ylabel('% explained variance')
titleid += 1
ax.set_title(f'Figure {titleid}: Scree plot')
fig.tight_layout()


# Thus 5 principal components should be enough to classify the repayment_interval.
# Let's see what these features are

# In[ ]:


sns.barplot(x=numerical_features, y=pca.components_[0])
titleid += 1
plt.title(f'Figure {titleid}: 1st PC')
plt.xticks(rotation=90);


# Surprisingly MPI has very little variance in the 1st PC, let's see the 2nd PC

# In[ ]:


sns.barplot(x=numerical_features, y=pca.components_[1])
titleid += 1
plt.title(f'Figure {titleid}: 2nd PC')
plt.xticks(rotation=90);


# To see which columns we can remove, let's see the last PC

# In[ ]:


sns.barplot(x=numerical_features, y=pca.components_[-1])
titleid += 1
plt.title(f'Figure {titleid}: last PC')
plt.xticks(rotation=90);


# As expected, funded_amount and loan_amount have similar distributions. Since loan_amount values are sometimes `null`, we pick funded_amount as the feature.

# ### Feature Selection

# In[ ]:


numerical_features.remove('loan_amount')

features = numerical_features + categorical_features
features


# ### One Hot Encoding Categorical Variables
# 
# Let's one hot encode the categorical variables and run classification using logistic regression and random forests.

# In[ ]:


def one_hot_encode(dataframe, categorical_features):
  train = pd.DataFrame()
  for feature in categorical_features:
    train = pd.concat([train, pd.get_dummies(dataframe[categorical_features])], axis=1)

  return train


# In[ ]:


### Note: Collab RAM crashes because one hot encoding region and currency takes a lot of RAM
## let's reduce the variables and remove region, activity and currency
categorical_features = ['country', 'sector']
one_hot_encoded = one_hot_encode(X_train, categorical_features)


# In[ ]:


def merge_category_numerical(dataframe, numerical_features, one_hot_encoded_matrix):
  return pd.concat([dataframe[numerical_features], one_hot_encoded_matrix], axis=1)


# In[ ]:


X_concat = merge_category_numerical(X_train, numerical_features, one_hot_encoded)


# In[ ]:


## We convert the multi class problem into a binary class problem since
## the characteristics for one time and irregular payments would largely be the 
## same.
def merge_classes(Y):
  return Y.apply(lambda x: 'regular' if x == 'monthly' else 'irregular')

Y_train = merge_classes(Y_train)
Y_test = merge_classes(Y_test)


# In[ ]:


#@title Logistic Classification - Set Hyperparameters
#@markdown Set the hyperparameters for the Logistic Classification model

penalty = 'l2' #@param ["l2", "l1"] {allow-input: true}
penalty_weight =   10#@param {type: "number"}
class_weight = 'balanced' #@param ["balanced", "None"] {allow-input: true}
#@markdown ---


# In[ ]:


hyperparameters = {
    'penalty': penalty,
    'C': penalty_weight,
    'class_weight': None if class_weight == 'None' else class_weight
}


# In[ ]:


def get_model(hyperparameters):
  print(f'Setting hyperparameters: {hyperparameters}')
  return LogisticRegression(solver='liblinear', **hyperparameters)


# In[ ]:


def create_sparse_dataframe(dataframe):
  return dataframe.to_sparse()


# In[ ]:


X_concat_sparse = create_sparse_dataframe(X_concat)


# In[ ]:


def train_logistic_classifier_payment_interval(X_train, Y_train, params):
  kf = KFold(n_splits=5, shuffle=False)
  train_acc = []
  val_acc = []
  class_model = get_model(params)
  for train, test in kf.split(X_train):
    class_model.fit(X_train.iloc[train], Y_train.iloc[train])
    train_acc.append(class_model.score(X_train.iloc[train], Y_train.iloc[train]))
    val_acc.append(class_model.score(X_train.iloc[test], Y_train.iloc[test]))

  print(f'validation accuracy: {np.mean(val_acc)}, {np.mean(train_acc)}')
  return class_model


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class_model = train_logistic_classifier_payment_interval(X_concat_sparse, Y_train, hyperparameters)


# In[ ]:


# Run on test set
X_test_concat = merge_category_numerical(X_test, numerical_features, one_hot_encode(X_test, categorical_features))


# In[ ]:


metric = confusion_matrix(Y_test, class_model.predict(X_test_concat))


# In[ ]:


fig = go.Figure(
      data=go.Heatmap(
            z=metric ,
            x=['predicted irregular', 'predicted regular'],
            y=['actual irregular', 'actual regular']
          )
    )
titleid += 1
fig.update_layout(
    title=f"Figure {titleid}: Confusion Matrix for predicted classes"
)
fig.show()


# The confusion matrix shows acceptable classification scores. The matrix also helps understand the relative accuracy between True Positive and True Negative rates, the latter being higher. This could be due to our assumption that one time payments are irregular, hence classifying them as positive.

# #6. Conclusion & Future Work
# 
# ## Summary
# 
# We set out to analyze Kiva.org's lending patterns and understand the distribution of loans across the globe. We analyzed the loan distribution across sectors, countries and activities.
# 
# A significant part of the analysis involved seeing the distribution of loans across countries and to find any geographical clustering.
# 
# Here are our key findings:
# 
# 1. Kenya saw a large amount of growth in the initial months of Kiva.org launching (subject to the fact that the dataset is from the time when Kiva.org launched).
# 
# 2. We see that Agriculture, Food and Retails are the largest sectors that get funded on [Kiva.org](http://kiva.org/)
# 
# 3. We see certain countries (such as Rwanda) that have a majority of loans under irregular repayment intervals. Irregular payments is a deciding factor of the loan amount.
# 
# 4. We see that most sectors have a majority repayment interval under monthly payments. 
# 
# **Interesting Finding**: For sectors like Agriculture where a microloan would be expected to be paid after a long time, or paid irregularly due to the seasonal aspect of Agriculture, especially in developing countries.
# 
# 5. We see a certain clustering among neighboring countries in terms of funded amounts, thus showing a potential spillover effect.
# 
# 6. Countries with similar MPI have a large difference between the total funded amount. This is due to weaker peer to peer networks by way of lower number of lenders.
# 
# 7. We see there is a strong linear correlation between total funded amount and total lender count for nations. Thus lender count is a good predictor of the strength of the microlending network.
# 
# **Interesting Finding**: Countries with lower number of lenders also get higher funded amounts. Therefore, a stronger peer to peer microloan network might not mean more lenders, but lenders willing to invest larger amounts of money.
# 
# 8. We briefly analyze bias to see the distribution of loans for men and women and pose open ended questions for considering bias in machine learning models. We discuss the question of including features if they may contain certain implicit bias.
# 
# 9. Finally, we do a classification task to classify a loan as 'regular' or 'irregular' payment, as a rough proxy for the likelihood of getting a loan. We use feature engineering techniques such as principal component analysis, one hot encoding and data transformation to clean and transform the data. We then use a logistic regression model to train using cross validation and calculate the confusion matrix using a test subset.
# 
# Overall, we go through the entire data science lifecycle in this project and frequently iterate between forming questions, cleaning data, doing EDA and running inference tasks.
# 
# ## Evaluation of our approach
# 
# Our approach involved posing questions, running EDA and drawing inferences based on the EDA. 
# 
# Since there wasn't a prediction problem setup in the dataset, a large part of our EDA was to understand correlations between features and evaluate which features could be converted to outcome variables such that they gave unique insights. Potentially, another approach could be to draw additional external data to see what loans do not get funded.
# 
# Our approach to finding bias in lending was via one specific set of features. Arguably, other features such as the tags assigned to the loan, the payment period etc could also have been used to check for bias.
# 
# ## Methods we used
# 
# In this project, we used a set of supervised and unsupervised learning models. For supervised methods, we used cross-validation to measure their performance.
# 
# Models we used:
# 
# 1. Linear Regression
# 2. Logistic Regression
# 3. Random Forest Classifier
# 4. Random Forest Regressor
# 5. Principle Component Analysis
# 
# For Q1 we used linear regression trying to find the relationship between funded amount and MPI.
# 
# For Q2 we used both linear regression and a random forest regressor. The random forest approach proved to have a better performance.
# 
# Q3 is a classification problem so we started from logistic regression. Then we improve the accuracy by applying random forest classifier. We visualized the results by confusion matrices and decision boundaries.
# 
# When classifying repayment intervals we used PCA to find useful features.
# 
# ## Limitations
# 
# 1. The large amount of `null` values of the data posed a problem. We decided to drop the `null` valued data to prevent zero value bias into the EDA. We found setting null values into the dataset actually skewed the metrics on funded amounts quite a bit.
# 
# 2. Lack of a prediction variable made most of our analysis open ended. We posed questions and ran inferences tasks based on what we thought would be beneficial in a microlending context, however the inference is depdendent on what the microlending company deems important.
# 
# 3. We were limited by a computational environment and we could not one hot encode all activities due to the feature set reaching very large values. A faster approach to this problem would probably involve using parallel processing libraries on a scalable cluster.
# 
# ## Future Work
# 
# Future work can involve the following:
# 
# 1. Analyzing the social bias introduced by including or removing the `gender_borrower` column.
# 2. Analyzing the social bias caused by implicit information embedded into features such as `sector` that have a correlation to the gender of the borrower.
# 3. Using Neural Networks for classification and regressions tasks with better metric scores.
# 4. Using ensemble methods such as Bagging, Boosting for better metric scores.
# 5. Exploring hyperparameter search algorithms such as [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find the most optimal hyperparameters for a model.
# 
# 
# ## Acknowledgements
# 
# We would like to thank the Faculty, Staff and Students of CS200 for the opportunity to work on this project as well as their support and feedback. This work used datasets provided by Kaggle and Kiva.org, for which we are grateful.
# 
# We would also like to thank UC Berkeley for providing us this opportunity.

# In[ ]:




