#!/usr/bin/env python
# coding: utf-8

# <h1>Welcome to the Corrected Kernel of Leonardo.</h1> (**Kindly refrain from upvoting this one rather upvote the original Kernel itself- Thanks for Complying)**
# 
# We will do some analysis trying to understand the Kiva data...
# 
# Kiva is an excellent crowdfunding plataform that helps the poor and financially excluded people around the world. 

# <h2> OBJECTIVES OF THIS EXPLORATION </h2>
# - Understand the distribuitions of loan values
# - Understand the principal sectors that was helped
# - Understand the countrys that receive the loan's
# - Understand the Date's through this loans
# - Understand what type of loan have more lenders
# - And much more... Everything that we can get of information about this dataset

# <h2> About the Dataset</h2>
# 
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. Unlike traditional machine learning competitions with rigid evaluation criteria, participants will develop their own creative approaches to addressing the objective. Instead of making a prediction file as in a supervised machine learning problem, submissions in this challenge will take the form of Python and/or R data analyses using Kernels, Kaggle's hosted Jupyter Notebooks-based workbench.
# 
# Kiva has provided a dataset of loans issued over the last two years, and participants are invited to use this data as well as source external public datasets to help Kiva build models for assessing borrower welfare levels. Participants will write kernels on this dataset to submit as solutions to this objective and five winners will be selected by Kiva judges at the close of the event. In addition, awards will be made to encourage public code and data sharing. With a stronger understanding of their borrowers and their poverty levels, Kiva will be able to better assess and maximize the impact of their work.
# 
# The sections that follow describe in more detail how to participate, win, and use available resources to make a contribution towards helping Kiva better understand and help entrepreneurs around the world.

# 1. <h2>Importing the library's</h2>

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from bokeh.plotting import figure 
from bokeh.io import output_notebook, show


# In[2]:


output_notebook()


# <h2>Loading the data</h2>

# In[3]:


df_kiva = pd.read_csv("../input/kiva_loans.csv")
df_kiva_loc = pd.read_csv("../input/kiva_mpi_region_locations.csv")
df_kiva_themes_ids = pd.read_csv("../input/loan_theme_ids.csv")
df_kiva_themes_region = pd.read_csv("../input/loan_themes_by_region.csv")


# [](http://)<h2>Let's Look At the data</h2>

# In[4]:


df_kiva.shape, df_kiva.nunique()


# In[5]:


df_kiva.describe()


# In[6]:


df_kiva.head()


# In[7]:


df_kiva_loc.head()


# In[8]:


df_kiva_themes_ids.head()


# In[9]:


df_kiva_themes_region.head()


# <h1>Let's start Exploring the Dataset's</h1>

# In[12]:


## Bokeh
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot

source = ColumnDataSource(df_kiva)

options = dict(plot_width=300, plot_height=300,
               tools="pan,wheel_zoom,box_zoom,box_select,lasso_select")

p1 = figure(title="Funded Amount Distribuition", **options)
p1.circle(y = df_kiva[df_kiva["country"] == 'India']['funded_amount'],x = df_kiva[df_kiva["country"] == 'India']['partner_id'], color="blue",legend="India")

p = gridplot([[ p1]], toolbar_location="right")

show(p)


# In[13]:


print("Description of distribuition")
print(df_kiva[['funded_amount','loan_amount']].describe())

g = sns.distplot(np.log(df_kiva['funded_amount'] + 1))
g.set_title("Funded Amount Distribuition", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(222)
g1 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.funded_amount.values))
g1= plt.title("Funded Amount Residual Distribuition", fontsize=15)
g1 = plt.xlabel("")
g1 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplot(223)
g2 = sns.distplot(np.log(df_kiva['loan_amount'] + 1))
g2.set_title("Loan Amount Distribuition", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Dist Frequency", fontsize=12)

plt.subplot(224)
g3 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.loan_amount.values))
g3= plt.title("Loan Amount Residual Distribuition", fontsize=15)
g3 = plt.xlabel("")
g3 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplots_adjust(wspace = 0.3, hspace = 0.3,
                    top = 0.9)


# So We have a normal distribuition to the both the values (except some Outliers)

# Exploring the Lenders_count column

# In[14]:


lenders = df_kiva.lender_count.value_counts();
lenders[:10]


# In[15]:


plt.figure(figsize=(12,10))

plt.subplot(222)
g = sns.distplot(np.log(df_kiva['lender_count'] + 1))

g.set_title("Dist Lenders Log", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(221)
g1 = sns.distplot(df_kiva[df_kiva['lender_count'] < 1000]['lender_count'])

g1.set_title("Dist Lenders", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplot(212)
g2 = sns.barplot(x=lenders.index[:40], y=lenders.values[:40])
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Top 40 most frequent numer of Lenders to the transaction", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)


# We have an interesting distribuition... 
# **You will wither have 1 lender or you will have a great chance to have 4 ~ 12 lenders in the project...**

# In[16]:


months = df_kiva.term_in_months.value_counts()

plt.figure(figsize=(12,10))

plt.subplot(222)
g = sns.distplot(np.log(df_kiva['term_in_months'] + 1))

g.set_title("Term in Months Log", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(221)
g1 = sns.distplot(df_kiva['term_in_months'])

g1.set_title("Term in Months", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplot(212)
g2 = sns.barplot(x=months.index[:40], y=months.values[:40],orient='v')
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("The top 40 Term Frequency", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)


# That's Weird....
# 
# In a "Normal" loan the term almost ever is always either 6 or 12 or 24 months but 
# It's the first time that I see 8 / 14 /[](http://) 20; 
# Very Curious.<br>
# 

# In[17]:


cont = df_kiva['country_code'].value_counts()

plt.figure(figsize=(12,10))

g2 = sns.barplot(x=cont.index[:20], y=cont.values[:20],data=df_kiva, orient='v')
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("The top 10 Country", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)


# PH is Dominating The Field
# Let's bring in Plotly and verify the same with the regions plot

# Lets checkout Sector Repayment 

# In[18]:


cnt_srs = df_kiva.repayment_interval.value_counts()

plt.figure(figsize=(12,10))

g = sns.barplot(x=cnt_srs.index[:5], y=cnt_srs.values[:5],data=df_kiva, orient='v')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Repayment Interval of loans", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)


# In[19]:


df_kiva['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12));


# In[20]:


# Distribution of world regions
fig, ax2 = plt.subplots(figsize=(10,10))
plt.xticks(rotation='vertical')
sns.countplot(x='world_region', data=df_kiva_loc);


# Let's Check How the distribution looks on A World Map....

# In[21]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[22]:


con_df = pd.DataFrame(df_kiva['country'].value_counts()).reset_index()
con_df.columns = ['country', 'num_loans']
con_df = con_df.reset_index().drop('index', axis=1)

#Find out more at https://plot.ly/python/choropleth-maps/
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_loans'],
        text = con_df['country'],
        #colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],
        #colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(220, 83, 67)']],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],\
            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
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
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Number of loans by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans-world-map')


# <h1>Let's look through the Sectors to know them</h1>

# In[23]:


sector_amount = pd.DataFrame(df_kiva.groupby(['sector'])['loan_amount'].mean().sort_values(ascending=False)).reset_index()

plt.figure(figsize=(12,12))

plt.subplot(211)
g = sns.countplot(x='sector', data=df_kiva)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Sector Loan Counts", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)

plt.subplot(212)
g1 = sns.barplot(x='sector',y='loan_amount',data=sector_amount,)
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.set_xlabel('Sector', fontsize=12)
g1.set_ylabel('Average Loan Amount', fontsize=12)
g1.set_title('Loan Amount Mean by sectors ', fontsize=15)

plt.subplots_adjust(wspace = 0.2, hspace = 0.35, top = 0.9)


# Very cool Plot....
# 
# It show us that the highest mean values isn't  the most frequent Sectors.... 
# 
# > Transportation, Arts and Servies have a low frequency but a high mean. 

# <h2>Now Let's look at some values through the sectors.</h2>

# In[24]:


df_kiva['loan_amount_log'] = np.log(df_kiva['loan_amount'])
df_kiva['funded_amount_log'] = np.log(df_kiva['funded_amount'] + 1)
df_kiva['diff_fund'] = df_kiva['loan_amount'] / df_kiva['funded_amount'] 

plt.figure(figsize=(12,14))

plt.subplot(312)
g1 = sns.boxplot(x='sector', y='loan_amount_log',data=df_kiva)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Loan Distribuition by Sectors", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount(log)", fontsize=12)

plt.subplot(311)
g2 = sns.boxplot(x='sector', y='funded_amount_log',data=df_kiva)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Funded Amount(log) by Sectors", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Funded Amount", fontsize=12)

plt.subplot(313)
g3 = sns.boxplot(x='sector', y='term_in_months',data=df_kiva)
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Term Frequency by Sectors", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Term Months", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)


# The values have an equal distribution through all data, being little small to Personal Use, which makes sense....
# 
# The highest Term months is  in Agriculture, Education and Health Fields respectively.....
# 
# 

# <h1> Taking advantage of sectors, let's look at the Acitivities</h1>

# In[25]:


acitivies = df_kiva.activity.value_counts()[:30]
activies_amount = pd.DataFrame(df_kiva.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False)[:30]).reset_index()

plt.figure(figsize=(12,10))

plt.subplot(211)
g = sns.barplot(x=acitivies.index, y=acitivies.values)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("The 30 Highest Frequency Activities", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(212)
g1 = sns.barplot(x='activity',y='loan_amount',data=activies_amount)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel('', fontsize=12)
g1.set_ylabel('Average Loan Amount', fontsize=12)
g1.set_title('The 30 highest Mean Amounts by Activities', fontsize=15)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8, top = 0.9)


# We can see that the activities with highest mean loan amount aren't the same as more frequent one's......
# 
# This is an interesting distribution of Acitivies but it isn't so meaningful... 
# 
# Let's try to further understand this.

# <h1>Now We  will explore the activies by the top 3 sectors....</h1>

# In[26]:


plt.figure(figsize=(12,14))

plt.subplot(311)
g1 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Agriculture'])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Activities by Agriculture Sector", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g2 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Food'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=80)
g2.set_title("Activities by Food Sector", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplot(313)
g3 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Retail'])
g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
g3.set_title("Activiies by Retail Sector", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.7,top = 0.9)


# Looking the activities by Sector is more insightful than just looking through the sectors or activities seperately....

# <h2>Let's plot the distribuition of loan by Repayment Interval to see if they share the same distribuition ......</h2>

# In[27]:


sns.FacetGrid(df_kiva, hue='repayment_interval', size=5, aspect=2).map(sns.kdeplot, 'loan_amount_log', shade=True).add_legend();


# <h2>Let's see how is the distribuition of Lenders over the Repayment Interval looklike.....</h2>

# In[28]:


df_kiva['lender_count_log'] = np.log(df_kiva['lender_count'] + 1)

sns.FacetGrid(df_kiva, hue='repayment_interval', size=5, aspect=2).map(sns.kdeplot, 'lender_count_log', shade=True).add_legend();


# Intesresting behavior of Irregular Payments, there's  a little difference...
# 
# The first peak in lenders distribuition is for the zero values to which I had added 1..

# <h2>Let's take a better look on Sectors and Repayment Intervals in a heatmap of correlation</h2>

# In[29]:


sector_repay = ['sector', 'repayment_interval']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[sector_repay[0]], df_kiva[sector_repay[1]]).style.background_gradient(cmap = cm)


# We have 3 sector's that have a high number of Irregular Repayments.
# 
# Why is this difference there? 
# 
# Maybe just because it's the most frequent ? 

# In[30]:


df_kiva.loc[df_kiva.country == 'The Democratic Republic of the Congo', 'country'] = 'Republic of Congo'
df_kiva.loc[df_kiva.country == 'Saint Vincent and the Grenadines', 'country'] = 'S Vinc e Grenadi'


# <h1>And What's the most frequent country? </h1>

# In[31]:


country = df_kiva.country.value_counts()
country_amount = pd.DataFrame(df_kiva[df_kiva['loan_amount'] < 20000].groupby(['country'])['loan_amount'].mean().sort_values(ascending=False)[:35]).reset_index()

plt.figure(figsize=(10,14))
plt.subplot(311)
g = sns.barplot(x=country.index[:35], y=country.values[:35])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("The 35 most frequent helped countrys", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g3 = sns.barplot(x=country_amount['country'], y=country_amount['loan_amount'])
g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
g3.set_title("The 35 highest Mean's of Loan Amount by Country", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Amonunt(US)", fontsize=12)

plt.subplot(313)
g2 = sns.countplot(x='world_region', data=df_kiva_loc)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("World Regions Distribuition", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)
plt.subplots_adjust(wspace = 0.2, hspace = 0.90,top = 0.9)


# 1. The highest mean value without the filter is for Cote D'Ivoire and it's $50000$ because we have just 1 loan to this country and that to  was lended by 1706 lenders... 

# In[32]:


df_kiva[df_kiva['country'] == "Cote D'Ivoire"]


# The most frequent Regions with more projects is really for poor regions... 
# - One interesting information is that almost all borrowers values means are under $ 10000...

# In[33]:


country_repayment = ['country', 'repayment_interval']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_repayment[0]], df_kiva[country_repayment[1]]).style.background_gradient(cmap = cm)


# On this heatmap correlation above we can see that just Kenya has the highest number of Irregular payments.....

# In[34]:


#To see the result output click on 'Output' 
country_sector = ['country','sector']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_sector[0]], df_kiva[country_sector[1]]).style.background_gradient(cmap = cm)


# We can look a lot of interesting values on this heatmap.
# - Philipines has high number of loan in almost all sectors
# - The USA is the country with the highest number of Entertainment loans
# - Cambodia has the highest number of Loans to Personal Use
# - Paraguay is the country with highest Education Loan request
# - Pakistan has highest loan requests to Art and Whosale
# - Tajikistan has highest requests in Education and Health
# - Kenya and Philipines have high loan requests to Construction
# - Kenya also has high numbers to Services Loans

# <h2>Let's verify the most frequent currency.... </h2>

# In[35]:


currency = df_kiva['currency'].value_counts()

plt.figure(figsize=(10,5))
g = sns.barplot(x=currency.index[:35], y=currency.values[:35])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("The 35 most Frequency Currencies at Platform", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.show()


# - PHP is the currency of Philiphine's
# - USD is the currency of USA's
# - KES is the currency of Kenya's

# <h2>Let's take a look at the Genders</h2>
# 
# - We will start cleaning the column borrower_genders and create a new column with the processed data.. 

# In[36]:


df_kiva.borrower_genders = df_kiva.borrower_genders.astype(str)

df_sex = pd.DataFrame(df_kiva.borrower_genders.str.split(',').tolist())

df_kiva['sex_borrowers'] = df_sex[0]

df_kiva.loc[df_kiva.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan


# In[37]:


sex_mean = pd.DataFrame(df_kiva.groupby(['sex_borrowers'])['loan_amount'].mean().sort_values(ascending=False)).reset_index()


# <h2> Let's  look through the Repayment Intervals column </h2>

# In[38]:


plt.figure(figsize=(10,6))

g = sns.countplot(x='sex_borrowers', data=df_kiva, 
              hue='repayment_interval')
g.set_title("Exploring the Genders by Repayment Interval", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count Distribuition", fontsize=12)


# In[39]:


print("Gender Distribuition")
print(round(df_kiva['sex_borrowers'].value_counts() / len(df_kiva['sex_borrowers'] )* 100),2)

plt.figure(figsize=(12,14))

plt.subplot(321)
g = sns.countplot(x='sex_borrowers', data=df_kiva, 
              order=['male','female'])
g.set_title("Gender Distribuition", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(322)
g1 = sns.barplot(x='sex_borrowers', y='loan_amount', data=sex_mean)
g1.set_title("Mean Loan Amount by Gender ", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Average Loan Amount(US)", fontsize=12)

plt.subplot(313)
g2 = sns.countplot(x='sector',data=df_kiva, 
              hue='sex_borrowers', hue_order=['male','female'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Exploring the Genders by Sectors", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g2 = sns.countplot(x='term_in_months',data=df_kiva[df_kiva['term_in_months'] < 45], 
              hue='sex_borrowers', hue_order=['male','female'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Exploring the Genders by Term in Months", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)


# We have a great difference between the genders, and very meaningfull informations when analysing by Gender.
# 
# Although we have 77% of women in our dataset, the men have the highest mean of Loan Amount....
# 
# On the sectors, we have also a interesting distribution of genders....

# <h2>Let's do some transformation in Dates to reveal the information contained by them... </h2>

# In[40]:


df_kiva['date'] = pd.to_datetime(df_kiva['date'])
df_kiva['funded_time'] = pd.to_datetime(df_kiva['funded_time'])
df_kiva['posted_time'] = pd.to_datetime(df_kiva['posted_time'])

df_kiva['date_month_year'] = df_kiva['date'].dt.to_period("M")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("M")
df_kiva['posted_month_year'] = df_kiva['posted_time'].dt.to_period("M")
df_kiva['date_year'] = df_kiva['date'].dt.to_period("A")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("A")
df_kiva['posted_year'] = df_kiva['posted_time'].dt.to_period("A")


# Yearwise Exploration

# In[41]:


df_kiva['Century'] = df_kiva.date.dt.year
loan = df_kiva.groupby(['country', 'Century'])['loan_amount'].mean().unstack()
loan = loan.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan = loan.fillna(0)
temp = sns.heatmap(loan, cmap='Blues');


# In[42]:


plt.figure(figsize=(10,14))

plt.subplot(311)
g = sns.countplot(x='date_month_year', data=df_kiva)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Month-Year Loan Counting", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g1 = sns.pointplot(x='date_month_year', y='loan_amount', 
                   data=df_kiva, hue='repayment_interval')
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Mean Loan by Month Year", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount", fontsize=12)

plt.subplot(313)
g2 = sns.pointplot(x='date_month_year', y='term_in_months', 
                   data=df_kiva, hue='repayment_interval')
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Term in Months by Month-Year", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Term in Months", fontsize=12)
plt.subplots_adjust(wspace = 0.2, hspace = 0.50,top = 0.9)


# It looks nice and very meaninful....
# - We have mean of $15000 of projects by month
# - The weekly payments was ended in 2015
# - The peak of projects was in 2017-03

# Term_In_Months V.S. Repayment_Interval

# In[43]:


fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(df_kiva['term_in_months'][df_kiva['repayment_interval'] == 'monthly'] , color='b',shade=True, label='monthly')
ax=sns.kdeplot(df_kiva['term_in_months'][df_kiva['repayment_interval'] == 'weekly'] , color='r',shade=True, label='weekly')
ax=sns.kdeplot(df_kiva['term_in_months'][df_kiva['repayment_interval'] == 'irregular'] , color='g',shade=True, label='irregular')
ax=sns.kdeplot(df_kiva['term_in_months'][df_kiva['repayment_interval'] == 'bullet'] , color='y',shade=True, label='bullet')
plt.title('Term in months(Number of months over which loan was scheduled to be paid back) vs Repayment intervals')
ax.set(xlabel='Terms in months', ylabel='Frequency');


# <h1>Quick look through the USE descriptions....</h1>

# In[44]:


from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize = (12,10))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
        
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          width=600, height=300,
                          random_state=42,
                         ).generate(str(df_kiva['use']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION")
plt.axis('off')
plt.show()


# In[45]:


from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize = (12,10))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
        
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          width=600, height=300,
                          random_state=42,
                         ).generate(str(df_kiva['use']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION")
plt.axis('off')


# Field_partner name v.s. Funding count

# In[46]:


#Distribution of Kiva Field Partner Names with funding count
print("Top Kiva Field Partner Names with funding count : ", len(df_kiva_themes_region["Field Partner Name"].unique()))
print(df_kiva_themes_region["Field Partner Name"].value_counts().head(10))
lender = df_kiva_themes_region['Field Partner Name'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9)
plt.xticks(rotation='vertical', fontsize=14)
plt.xlabel('Field Partner Name', fontsize=18)
plt.ylabel('Funding count', fontsize=18)
plt.title("Top Kiva Field Partner Names with funding count", fontsize=25);


# There are total 302 Kiva Field Partner.
# - Out of these, Alalay sa Kaunlaran (ASKI) did the highest number of funding followed by SEF International and Gata Daku Multi-purpose Cooperative (GDMPC).

# > Top 25 loan uses in India (My Country)

# In[54]:


loan_use_in_india = df_kiva['use'][df_kiva['country'] == 'India']
percentages = round(loan_use_in_india.value_counts() / len(loan_use_in_india) * 100, 2)[:25]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top 25 loan uses in India',titlefont= dict(size=20))

fig = go.Figure(data=data)
py.iplot(fig, show_link=False)


# Top Use of Loan in India is to buy a Smokeless Stove...

# In[49]:


plt.figure(figsize = (12,10))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
        
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          width=600, height=300,
                          random_state=42,
                         ).generate(str(df_kiva['use'][df_kiva['country'] == 'India']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION")
plt.axis('off')


# 1. Trying to Use Latitude And Longitudes
# 
# **Can be made better Using Region's Wise MIP's**

# In[50]:


scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        lon = df_kiva_loc['lon'],
        lat = df_kiva_loc['lat'],
        text = df_kiva_loc['MPI'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df_kiva_loc['ISO'],
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'MPI\s',
        colorbar = True,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world' )


# Simple Analysis On India

# In[57]:


df_kiva[df_kiva.country == 'India'].groupby('sector').id.count().sort_values().plot.bar()


# In[60]:


df_kiva[df_kiva.country == 'India'].groupby('region').id.count().sort_values(ascending=False).head(15).plot.bar();


# In[65]:


df_kiva_loc.head()


# In[66]:


from mpl_toolkits.basemap import Basemap
longitudes = list(df_kiva_loc[df_kiva_loc.country == 'India']['lon'])
latitudes = list(df_kiva_loc[df_kiva_loc.country == 'India']['lat'])


# In[72]:


scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        lon = df_kiva_loc[df_kiva_loc.country == 'India']['lon'],
        lat = df_kiva_loc[df_kiva_loc.country == 'India']['lat'],
        text = df_kiva_loc[df_kiva_loc.country == 'India']['MPI'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df_kiva_loc['ISO'],
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'MPI\s',
        colorbar = True,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='India')


# In[ ]:




