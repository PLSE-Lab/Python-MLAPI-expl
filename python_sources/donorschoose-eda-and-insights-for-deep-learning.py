#!/usr/bin/env python
# coding: utf-8

# <h1><center><font color='coral'>Detailed Analysis, Visualization and Key Insights</font></center></h1>

# ### The goal of this analysis is to find some key insights, which can help build a machine learning models for predicting donors. 
# Click [here](https://www.kaggle.com/aamitrai/donorschoose-ml-pair-donors-to-projects/) to refer the Machine Learning Notebook (to predict donors for the project)
# 

# <img src="http://internet.savannah.chatham.k12.ga.us/schools/mes/PublishingImages/DonorsChoose_org_logo.jpg" alt="Drawing" style="width: 500px;">

# ## Table of Contents
# 1 [Load Libraries and Datasets](#Import-Required-Libraries-and-Datasets)<br>
# 2 [Analyze Projects Dataset](#Analyze-Projects-Dataset)<br>
# 3 [Analyze Donors and Donations Datasets](#Analyze-Donors-and-Donations-Datasets)<br>
# 4 [Analyze Teachers and Resources Datasets](#Analysis-of-Teachers-and-Resources)<br>
# 5 [Data Integrity Checks](#Data-Integrity-Checks)<br>
# 6 [Key Insights for ML Solution](#Key-Insights-for-Developing-ML-Solution)<br>
# 

# ### Import Required Libraries and Datasets

# In[1]:


# General libraries
import os
import re
import urllib
from collections import Counter
from itertools import cycle, islice
import warnings

# Data analysis and preparation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import cufflinks as cf
from statsmodels.tsa.seasonal import seasonal_decompose

# Set configuration
warnings.filterwarnings("ignore")
cf.go_offline()


# In[2]:


get_ipython().run_cell_magic('time', '', "input_dir = '../input/io/'\n#input_dir = './io/'\nall_projects = pd.read_csv(input_dir + 'Projects.csv')\nall_donations = pd.read_csv(input_dir + 'Donations.csv')\nall_donors = pd.read_csv(input_dir + 'Donors.csv')\nall_schools = pd.read_csv(input_dir + 'Schools.csv')\nall_resources = pd.read_csv(input_dir + 'Resources.csv')\nall_teachers = pd.read_csv(input_dir + 'Teachers.csv')\n\n# Pull the state codes from another dataset and load in a dataframe\n#filename = './us-state-county-name-codes/states.csv'\nfilename = '../input/us-state-county-name-codes/states.csv'\ndf_statecode = pd.read_csv(filename, delimiter=',')\n\nprint('Done loading files')")


# ### Analyze Projects Dataset

# In[3]:


# Describe dataset
display(all_projects.head(5))
print('Overall {} Rows and {} columns'.format(all_projects.shape[0], all_projects.shape[1]))
display(all_projects.nunique())


# ##### Key Observations
# 1. There are 1110017 rows in project dataset but 1110015 unique projects id. So, there 2 duplicate projects entries
# 2. Projects are spread across 72, 361 schools.
# 3. Project Subject Category Tree & Project Subject Subcategory Tree have categorical data. However, there seem to be multiple entries in these columns. Looking at [DonorsChoose.org](https://www.donorschoose.org/donors/search.html), it makes sense since a classroom request can be associated with multiple categories.
# 4. There are many duplicate project titles, short descriptions and even essays & need statements. This may indicate that teachers are re-using these across classroom requests.

# In[4]:


proj_stat = all_projects['Project Current Status'].value_counts().sort_values().to_frame()
proj_stat['count'] = proj_stat['Project Current Status']
proj_stat['Project Current Status'] = proj_stat.index
proj_stat.iplot(kind='pie', labels='Project Current Status', values='count',
                title = 'Funding Status of Projects',
                pull=.1,
                hole=.4,
                textposition='outside',
                textinfo='value+percent')

proj_type = all_projects['Project Type'].value_counts().sort_values().to_frame()
proj_type['count'] = proj_type['Project Type']
proj_type['Project Type'] = proj_type.index
proj_type.iplot(kind='pie',labels='Project Type',values='count',
                title = 'Types of Projects',
                pull=.1,
                hole=.4,
                textposition='outside',
                textinfo='value+percent',
               )


# ##### Key Observations
# 
# 1. There are more than 41K projects live right now. It would be an important factor in determining targeting strategy. 
# 2. Most of the projects are 'Teacher-Led'. For creating ML model, this would probably not be an important factor.

# In[5]:


# Monthly trends of project funding over the years
proj_posted = all_projects['Project Posted Date'].str.slice(start=0, stop=7)
proj_posted = proj_posted.value_counts().sort_index()

proj_funded = all_projects['Project Fully Funded Date'].str.slice(start=0, stop=7)
proj_funded = proj_funded.value_counts().sort_index()

proj = pd.concat([proj_posted, proj_funded], axis=1)
proj[(proj['Project Posted Date']) < (proj['Project Fully Funded Date'])]

proj.iplot([{'x': proj.index, 'y': proj[col], 'mode': 'line','name': col}
            for col in proj.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Monthly Observerd Volume of Projects Over Time"
          )
# proj.iplot(kind = 'scatter', xTitle='Year and Month the Project was Posted',  yTitle = "Total Count", 
#                 title ="Volume of Monthly Projects Over Time", width=5)

# Let's decomponse the time series
proj.index = pd.to_datetime(proj.index )
decomp_post = seasonal_decompose(proj['Project Posted Date'].to_frame(), model='multiplicative')
decomp_fund = seasonal_decompose(proj['Project Fully Funded Date'].to_frame(), model='multiplicative')
trend = decomp_post.trend.join(decomp_fund.trend)
seasonal = decomp_post.seasonal.join(decomp_fund.seasonal)
resid = decomp_post.resid.join(decomp_fund.resid)

trend.iplot([{'x': trend.index, 'y': trend[col], 'mode': 'line','name': col}
            for col in trend.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Trend of Projects Over Time"
          )
        
seasonal.iplot([{'x': seasonal.index, 'y': seasonal[col], 'mode': 'line','name': col}
            for col in seasonal.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Seasonality of Projects Over Time"
          )
        
resid.iplot([{'x': resid.index, 'y': resid[col], 'mode': 'line','name': col}
            for col in resid.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Randomness in Project Volume Over Time"
          )


# ##### Key Observations
# 
# 1. A lot of classroom requests are placed in Q3 - most probably due to 'back to school season'. We can see a very strong seasonality component in the decomposition.
# 2. The trend of number of classroom requests is ingeneral upwards, although there has been some reduction in requests in 2015 & 2017.
# 3. A large number of projects (50K) got funded in March 2018. Infact it is one of the months when the volume of funding is much more than number of new classroom requests. It would be very intersting to understand the driver behind this surge.
# 4. In general there is a lot of randomness in project funding, over the time

# In[7]:


# Let's analyze the project cost
print('Lowest project cost - {}'.format(all_projects['Project Cost'].min()))
print('Highest project cost - {}'.format(all_projects['Project Cost'].max()))

# The cost of classroom requests vary a lot; between $35 to $255K
# Now split the data in some buckets such that it represents a normal distribution
# This is of course a simulated distribution, but it does provides some perspective on project costs.
custom_bucket = [0, 179, 299, 999, 2500, 100000]
custom_bucket_label = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
proj_cost = pd.cut(all_projects['Project Cost'], custom_bucket, labels=custom_bucket_label)
proj_cost = proj_cost.value_counts().sort_index()

proj_cost.iplot(kind='bar', xTitle = 'Project Cost', yTitle = "Project Count", 
                title = 'Distribution on Project Cost', color='violet')


# In[8]:


# Project Category and Subcategory are stacked columns. A classromm request can span across multiple categories.
# I will start by exploding the columns and then analyze the trend over the years
def stack_attributes(df, target_column, separator=', '):
    df = df.dropna(subset=[target_column])
    df = (df.set_index(df.columns.drop(target_column,1).tolist())
          [target_column].str.split(separator, expand=True)
          .stack().str.strip()
          .reset_index()
          .rename(columns={0:target_column})
          .loc[:, df.columns])
    df = (df.groupby([target_column, 'Project Posted Date'])
          .size()
          .to_frame(name ='Count')
          .reset_index())
    
    return df

def plot_trend(df, target_column, chartType=go.Scatter,
              datecol='Project Posted Date', 
              ytitle='Number of relevant classroom requests'):
    trend = []
    for category in list(df[target_column].unique()):
        temp = chartType(
            x = df[df[target_column]==category][datecol],
            y = df[df[target_column]==category]['Count'],
            name=category
        )
        trend.append(temp)
    
    layout = go.Layout(
        title = 'Trend of ' + target_column,
        xaxis=dict(
            title='Year & Month',
            zeroline=False,
        ),
        yaxis=dict(
            title=ytitle,
        ),
    )
    
    fig = go.Figure(data=trend, layout=layout)
    iplot(fig)
    
proj = all_projects[['Project Subject Category Tree',
                     'Project Subject Subcategory Tree',
                     'Project Resource Category',
                     'Project Grade Level Category',
                     'Project Posted Date']].copy()
proj['Project Posted Date'] = all_projects['Project Posted Date'].str.slice(start=0, stop=4)

proj_cat = stack_attributes (proj, 'Project Subject Category Tree')
proj_sub_cat = stack_attributes (proj, 'Project Subject Subcategory Tree')
proj_res_cat = (proj.groupby(['Project Resource Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
proj_grade_cat = (proj.groupby(['Project Grade Level Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(proj_cat, 'Project Subject Category Tree')
plot_trend(proj_sub_cat, 'Project Subject Subcategory Tree')
plot_trend(proj_res_cat, 'Project Resource Category')
plot_trend(proj_grade_cat, 'Project Grade Level Category', chartType=go.Bar)


# #### Key Observations
# 
# 1. A bluk of classroom requests have cost between 150 - 1500. There are decent number of outliers though. Some projects are as expensive as $255K
# 2. We can see some very useful information from the project category trends. Literacy, Language, Math & Science are the most prominent project categories.
# 3. Computers, tablets, educational games and books are popularly being requested.
# 4. While globally health and fitness awreness is trending, surprisingly there has been a reduction in classroom request related to health, wellness & gym equipments.
# 5. Most of the classroom requests are for grades 'PreK-2' and '3-5'
# 
# 
# ##### I am skipping the detailed analysis of project essays and other free-form text columns, as of now; Some of the other kagglers have already created beautiful wordcloud analysis. As we will see in subsequent analysis that more than 75% of the donors have 'only' donated once. Thus, putting too much emphasis on these texts may lead to a model that may not generalize very well. While these colums provide important insights, I believe the project category, subcategory and resource types attributes express the adequate sentiments to help build a recommendation engine.

# ### Analyze Donors and Donations Datasets
# 
# I'll be analyzing both the datasets in parallel are they are closely related

# In[9]:


# Describe donation dataset
display(all_donations.head(5))
display('Overall {} Rows and {} columns'.format(all_donations.shape[0], all_donations.shape[1]))
display(all_donations.nunique())


# In[10]:


# Describe donor dataset
display(all_donors.head(5))
print('Overall {} Rows and {} columns'.format(all_donors.shape[0], all_donors.shape[1]))
display(all_donors.nunique())


# #### Key Insights
# 
# 1. We don't see any primary key duplicates in Donors and Donations - that's great!!
# 2. There are more than 2M donors; On the overview page, it's mentioned there are more than 3M donors. So, we may be looking at a subset of the data.
# 3. There are 2,024,554 donors in 'Donations' dataset, but '2,122,640' in Donors. So, at least in the data that we have there are Donors who have not made any donation yet.
# 4. There are 901,965 unique projects that has received some donation. As we saw earlier, there are more than 1M projects in 'projects' datasets. Thus, there are a quite a few projects which have not received any funding at all. 
# 

# In[ ]:


# Let's analyze the donation amoung
print('Lowest project cost - {}'.format(all_donations['Donation Amount'].min()))
print('Highest project cost - {}'.format(all_donations['Donation Amount'].max()))

# The Donation amount vary a lot; between $0.01 to $60,000
# Now the goal is to split the data in some buckets such that it represents a normal distribution
# This is of course a simulated distribution, but it will help provide some perspective on donation amount.
custom_bucket = [0, 0.99, 9.99, 99.99, 999.99, 1000000]
custom_bucket_label = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
don_amnt = all_donations[['Donation Amount', 'Donation Received Date']]
don_amnt['Donation Amount'] = pd.cut(don_amnt['Donation Amount'], custom_bucket, labels=custom_bucket_label)
don_amnt['Donation Received Date'] = don_amnt['Donation Received Date'].str.slice(start=0, stop=4)
don_amn = don_amnt['Donation Amount'].value_counts().sort_index()

don_amn.iplot(kind='bar', xTitle = 'Donation Amount', yTitle = 'Donation Count', 
                title = 'Simulated Distribution on Donation Amount')

don_amnt = (don_amnt.groupby(['Donation Amount', 'Donation Received Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(don_amnt, 'Donation Amount', chartType=go.Scatter, datecol='Donation Received Date', 
           ytitle='Number of donations')


# In[ ]:


# Let's find number of donations per project
custom_bucket = [0, 1, 5, 10, 20, 1000000]
custom_bucket_label = ['Single Donor', '1-5 Donors', '6-10 Donors', '11-20 Donors', 'More than 20 Donors']
num_of_don = all_donations['Project ID'].value_counts().to_frame(name='Donation Count').reset_index()
num_of_don['Donation Cnt'] = pd.cut(num_of_don['Donation Count'], custom_bucket, labels=custom_bucket_label)
num_of_don = num_of_don['Donation Cnt'].value_counts().sort_index()

num_of_don.iplot(kind='bar', xTitle = 'Number of Donors', yTitle = 'Number of Projects', 
                title = 'Distribution on Number of Donors and Project Count')


# #### Key Observations
# 
# 1. There are 350 donation below \$1; almost 14K donations equal to or more than \$1K
# 2. Most donors seem to donate between \$10 - \$100
# 3. There are still ~800K donations which are more than \$100 but less than \$1K
# 4. Also ~700K donors donated less than \$10
# 5. Number of donations, in general, have an upward trend. Donationaton in range \$10 - \$100 are incresing significantly
# 6. Most projects have less than 5 Donors

# In[ ]:


# Let find how many time donors are donating to the classrooms
custom_bucket = [0, 1, 5, 10, 100, 1000000]
custom_bucket_label = ['Donated Once', 'Donated 1 - 5 times', 'Donated 6 - 10 time',
                       'Donated 11 & 100 times', 'Donated more than 100 times']
don_repeat = all_donations['Donor ID'].value_counts().to_frame(name='Donation Count')
display ('Maximum Repeat donations by a Donor - {}'.format(don_repeat['Donation Count'].max()))
display ('Minimum Repeat donations by a Donor - {}'.format(don_repeat['Donation Count'].min()))

don_repe = don_repeat.copy()
don_repe = don_repe['Donation Count'].value_counts().to_frame(name='Number of Donors')
don_repe['Number of Donations'] = don_repe.index
don_repe['Number of Donations'] = pd.cut(don_repe['Number of Donations'], 
                                         custom_bucket, labels=custom_bucket_label)
don_repe = {
  'data': [
    {
      'values': don_repe['Number of Donors'],
      'labels': don_repe['Number of Donations'],
      'name': 'Number of Donations',
      'hoverinfo':'name',
      'hole': .4,
      'pull': .01,
      'type': 'pie'
    }],
  'layout': {'title':'Share of number of Donations'}
}
iplot(don_repe, filename='donut')


# #### Key Observations
# 
# 1. ~73% Donors have donated only once. ML Models are usually build by profiling a lost of data from the customers. In this case we would have only a single record to go by, for many of the donors.
# 2. 0.09% of customers have donated more that a 100 times. DonorsChoose parter with many organizations. It would be interesting to find out whether these are such institutional donors.
# 3. 22% of Donors have donated more that once. That is also a significant numbers. A good targetting model would help move more donors in this segment

# In[ ]:


# Lets find out the donor trends across state. 
# Let's start by finding out number of donors in each state
donor_per_state = all_donors.groupby('Donor State').size().to_frame(name='Total Donors in State')

donation_state = don_repeat.copy()
donation_state['Donor ID'] = donation_state.index
donation_state = donation_state.merge(all_donors, how='inner', on='Donor ID')

# This will give the repeat donors now
repeat_donors = donation_state[donation_state['Donation Count'] > 1]

# Let's find Number of donations per state and repeat donations per state 
repeat_donors_cnt = repeat_donors.groupby('Donor State').size().to_frame('Number of Repeat Donors')
repeat_donors_cnt = repeat_donors_cnt.merge(donor_per_state, left_index=True, right_index=True)
repeat_donors_cnt['Percentage of Repeat Donors'] = (repeat_donors_cnt['Number of Repeat Donors'] 
                                                * 100 / repeat_donors_cnt['Total Donors in State'])

(repeat_donors_cnt['Total Donors in State']
     .sort_values(ascending=False)
     .iplot(kind='bar', xTitle = 'States', yTitle = "Number of Donors", 
                title = 'Distribution on Donors Across State', color='Green'))

(repeat_donors_cnt['Percentage of Repeat Donors']
     .sort_values(ascending=False)
     .iplot(kind='bar', xTitle = 'States Cost', yTitle = "Number of Donors", 
                title = 'Distribution Repeat Donors Across States', color='Red'))


# #### Key Observations
# 
# 1. Most of the donations are coming in from California, followed by New York, Texas and Florida
# 2. The percentage of repeat donors are though kind of similar across all the states. South Carolina have highest (28%) repeat donors, followed closely by Oklahoma and California

# In[ ]:


# Let's analyze if the city/state of classroom request impacts the donation
# Get the School state and the State of Donors, associated with classroom requests
don_info = all_donations[['Donor ID', 'Project ID']].copy()
don_info = don_info.merge(all_donors[['Donor ID', 'Donor State']], on='Donor ID', how='inner')
don_info = don_info.merge(all_projects[['Project ID', 'School ID']], on='Project ID', how='inner')
don_info = don_info.merge(all_schools[['School ID', 'School State']], on='School ID', how='inner')
don_info['In State Donation'] = np.where((don_info['School State']) == (don_info['Donor State']), 'Yes', 'No')

in_state = (don_info['In State Donation'].value_counts()
            .sort_values()
            .to_frame(name='Count')
            .reset_index()
           )
in_state['In State Donation'] = in_state.index
in_state.iplot(kind='pie',labels='In State Donation',values='Count',
                title = 'Are the Donors Donating Within the States They Live In?',
                pull=.01,
                hole=.01,
                colorscale='set3',
                textposition='outside',
                textinfo='value+percent')


# Let's see how this trend varies across states
in_stat = (don_info.groupby(['Donor State', 'In State Donation'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
in_stat = in_stat.pivot(index='Donor State', columns='In State Donation', values='Count')
in_stat['In-State Donation Ratio'] = in_stat['Yes'] * 100 / (in_stat['Yes'] + in_stat['No'])

# (in_stat['In-State Donation Ratio']
#      .sort_values(ascending=True)
#      .iplot(kind='bar', xTitle = 'States', yTitle = "Pecentate of donations made within state", 
#             title = 'Percentage of in-state donations across states',
#             colorscale='-ylorrd', theme = 'pearl'))
in_stat['State'] = in_stat.index
temp = in_stat.merge(df_statecode, on='State')
temp.head()

data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = temp['Abbreviation'],
        z = in_stat['In-State Donation Ratio'].astype(float),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict()
        ) ]

layout = dict(
        title = 'Percentage of Donations Within Home State',
        geo = dict(
            scope='usa',
            projection=dict(type='albers usa'))
             )
    
fig = dict(data=data, layout=layout )
iplot(fig, filename='d3-cloropleth-map', validate=False)


# ### Key Observations
# 
# 1. A significant portion of the donors are donating within the state that they live in. Thus location is a cruicial factor in decision making / influencing donors.
# 2. Geographicall North East U.S is somewhat neutral to the fact whether they are donating in-state or out of state
# 3. West cost is predominantly donating to the classroom requests from their own state

# In[ ]:


# Let's see what percentage of the donors are teachers
teachers_in_donors = round((all_donors[all_donors['Donor Is Teacher'] == 'Yes'].shape[0]) 
                      * 100 / all_donors.shape[0])
teachers_in_donors = {"Teachers": teachers_in_donors, "Others": 100 - teachers_in_donors}

tchr_repeat_donors = round((repeat_donors[repeat_donors['Donor Is Teacher'] == 'Yes'].shape[0]) 
                      * 100 / repeat_donors.shape[0])
tchr_repeat_donors = {'Teachers': tchr_repeat_donors, 'Others': 100 - tchr_repeat_donors}


tchr = {
  'data': [
    {
        'values': list(teachers_in_donors.values()),
        'labels': list(teachers_in_donors.keys()),
        'domain': {"x": [0, .48]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(215, 112, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Percentage of Teachers Among Donors',
        'hole': .1,
        'pull': .0,
        'type': 'pie'
    },
    {
        'values': list(tchr_repeat_donors.values()),
        'labels': list(tchr_repeat_donors.keys()),
        'domain': {"x": [.52, 1]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(200, 200, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Percentage of Teachers Among Repeat Donors',
        'hole': .1,
        'pull': .0,
        'type': 'pie'
    }
  ],
    'layout': {'title':'Are Teachers Donating More That Others?'}
}

iplot(tchr, filename='styled_pie_chart')


# #### Key Observations
# 
# 1. 10% of the donors are teachers. Looking at the US Census, I can see that teachers constitute only 1.5% of US adult (above 18) population. Given the fact a much larger ratio of teachers are donating for the cause.
# 
# 2. Even more interesting is the fact that among those who have donated more than once, 24% are teachers

# ### Analysis of Teachers and Resources

# In[ ]:


# Describe teachers dataset
display(all_teachers.head(5))
display('Overall {} Rows and {} columns'.format(all_teachers.shape[0], all_teachers.shape[1]))
display(all_teachers.nunique())


# In[ ]:


# Describe resources dataset
display(all_resources.head(5))
display('Overall {} Rows and {} columns'.format(all_resources.shape[0], all_resources.shape[1]))
display(all_resources.nunique())


# In[ ]:


# Let's find the gender ratio of the teachers
tech_gend = all_teachers[['Teacher Prefix', 'Teacher ID', 'Teacher First Project Posted Date']].copy()
tech_gend['Gender'] = tech_gend['Teacher Prefix']
tech_gend.loc[tech_gend['Gender'] == 'Mrs.', 'Gender'] = 'Female'
tech_gend.loc[tech_gend['Gender'] == 'Ms.', 'Gender'] = 'Female'
tech_gend.loc[tech_gend['Gender'] == 'Mr.', 'Gender'] = 'Male'
tech_gend.loc[tech_gend['Gender'] == 'Teacher', 'Gender'] = 'Neutral'
tech_gend.loc[tech_gend['Gender'] == 'Dr.', 'Gender'] = 'Neutral'
tech_gend.loc[tech_gend['Gender'] == 'Mx.', 'Gender'] = 'Neutral'

gen = tech_gend.groupby('Gender').size().to_frame(name='Count')
gen['Gender'] = gen.index

# Average number of classroom request by gender
tech_gend_proj = tech_gend.merge(all_projects[['Teacher ID', 'Project Cost']], on='Teacher ID')
tech_gend_proj = tech_gend_proj.groupby('Gender').size().to_frame('Count')
tech_gend_proj['Gender'] = tech_gend_proj.index

tchr = {
  'data': [
    {
        'values': list(gen['Count']),
        'labels': list(gen['Gender']),
        'domain': {"x": [0, .48]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(215, 112, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Spread of Teachers Based on Gender',
        'hole': .70,
        'pull': .0,
        'type': 'pie'
    },
    {
        'values': list(tech_gend_proj['Count']),
        'labels': list(tech_gend_proj['Gender']),
        'domain': {"x": [.52, 1]},
        'marker': {'colors': ['rgb(124, 173, 200)', 'rgb(200, 200, 200)']},
        'hoverinfo':'labels+percentage',
        'name': 'Number of Clasroom Requests Based on Gender',
        'hole': .70,
        'pull': .0,
        'type': 'pie'
    }
  ],
    'layout': {'title':"Distribution Based on Gender"}
}
iplot(tchr, filename='styled_pie_chart')

# How many new teachers are joining DonorsChoose each month
tech_start_mnth = tech_gend['Teacher First Project Posted Date'].str.slice(start=0, stop=7)
tech_start_mnth = tech_start_mnth.value_counts().to_frame(name='Count').sort_index()
tech_start_mnth['Date'] = tech_start_mnth.index

trace = go.Scatter(
            x=tech_start_mnth.Date,
            y=tech_start_mnth.Count,
            name = "New Teachers Onboarding Every Month",
            line = dict(color = '#17BECF'),
            opacity = 0.8)

layout = dict(
    title = 'Monthly New Teachers Associating with DonorsChoose Over Time',
    xaxis = dict(
        title='Time Period',
    ),
    yaxis = dict(
        title='Number of Teachers',
    )
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename = 'time-series')


# #### Key Insights
# 
# 1. A huge percentage of female teachers are associated with DonorsChoose's platform.
# 2. Increasing number of teachers are continuosuly joining the DororChoose's platform. The numbers seem to have reduced slightly in past 18 months, though.
# 3. Also, I can see some multiplicative seasonal trends, with a large number of teachers joining in Q2 and Q4

# In[ ]:


display(all_resources['Resource Item Name'].nunique())
display(all_resources['Resource Item Name'].value_counts().sort_values(ascending=False).head(10))


# ### Data Integrity Checks
# 
# See if all the datasets can be linked together without any issues. It is important to identify such issues upfront, prior to building ML model

# In[ ]:


proj_proj = list(all_projects['Project ID'].unique())
proj_teacher = list(all_projects['Teacher ID'].unique())
proj_school = list(all_projects['School ID'].unique())

dontn_proj = list(all_donations['Project ID'].unique())
dontn_donor =list(all_donations['Donor ID'].unique())

donor_donor = list(all_donors['Donor ID'].unique())
resrcs_proj = list(all_resources['Project ID'].unique())
tech_teacher = list(all_teachers['Teacher ID'].unique())
sch_school = list(all_schools['School ID'].unique())

print('{} projects are there in total'.format(len(set(proj_proj))))
print('{} Teachers have classroom Projects but no records in Teachers dataset'
      .format(len(set(proj_teacher) - set(tech_teacher))))
print('{} Schools have a classroom request but no info in School dataset'
      .format(len(set(proj_school)-set(sch_school))))
print('{} Projects have no Resources associated with it'
      .format(len(set(proj_proj)-set(resrcs_proj))))
print('{} Projects have not received any Donoation'
      .format(len(set(proj_proj)-set(dontn_proj))))
print('{} people have donated to projects but missing in Donor dataset'
      .format(len(set(dontn_donor)-set(donor_donor))))


# #### Key Observations
# 
# There are some inconsistencies between the datasets. 
#  - For 4 teachers, there is no info in 'Teachers' dataset.
#  - Similarly, details are missing for 12 Schools.
#  - Of 1.1M projects, 236K have not received any donation at all. These projects cannot be used for building recommendation engine.
#  - 5.5K Donors do not have any information in Donor dataset. Without info, organization won't be able to target them and thus they should be dropped from analysis.

# ## Key Insights for Developing ML Solution
# 
# Below are the key insights which will be useful for buiding the ML modes. Click [here](https://www.kaggle.com/aamitrai/data-visualization-ml-for-predicting-donors) to take a look at the analysis notebook.
# 
# - 73% customers have donated only once. If organization has some additional data, such as clickstream, it would be a lot usefulfor creating better profile the donors.
# 
# - Given that a such a high number of donors only have a single record, it would not be possible to say with certainity what motivated them to donate in the first place. For this specific reason, I would not be using the essay descriptions and other freeform text; it may lead to overfitting. I would instead rely on the combination of Project category, subcategory, resource category, and donation amount to find high level pattern.
# 
# - 1% customers have donated more than 100 times. Give the nature of operations, there should be a slightly different (more personalized) strategy for high volume donors. Deep Learning can be used to build a much better profile for such 
# 
# - Some of the donors are donating thousands of time. On DonorsChoose website, I can see that they parter with many organizations. Some of these orgs match individual donations on the project. That may be one of the reason for these outliers (cases with thousands of donation), although I may be wrong on this one. Data related to such institutions should be removed before building the models.
# 
# - 22% of the project could not get required funding. In my analysis I could not find any significant pattern that may lead to projects not being funded; though, project cost does have some influence on funding. It would be interesting to see if ML can help provide some insights on probability of the project getting funded.
# 
# - The bluk of classroom requests cost between $150 - $1,500. There are decent number of outliers though. Some projects are as expensive as $255K.
# 
# - Most donors seem to donate between $10 - $100
# 
# - A huge chunk of classroom requests orginate from California. Most of the donors are from California as well.
# 
# - A large percentage of donors prefer to donate within their own state, although, residents in norteastern US seem to have less bias. Thus, location would be an important factor in recommending the project.
# 
# 
# 
# Few features which I'll be skipping deliberately:
# - Project Essays, while informational, may not add a lot of value for ML, based on the reasons explained earlier.
# - Bulk of projects are teacher lead, thus that attribute would not be relevant for analysis.
# - Teacher's Gender should not impact donation
# - Project cost is something that I would probably experiment, but not in first iteration.
# 
# 
# 
# We identified few Data Integrity issues as well
# - There are 2 duplicate 'Projects' in project file
# - There are 5.5K Donors present in 'Donation' dataset, but missing in 'Donor' dataset. If there is no info about donors, it would make sense to drom them from recommendation process.
# 

# <h1><center><font color='green'>Thanks for checking this out :)</font></center></h1>

# In[ ]:




