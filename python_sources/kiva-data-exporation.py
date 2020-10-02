#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Kiva Dataset


# In[11]:


#Load Kiva dataset


# In[12]:


import pandas as pd
import numpy as np


# In[13]:


# Reading kiva loans file
kiva_loan = pd.read_csv("../input/kiva_loans.csv")
kiva_loan.head(n=3)


# In[14]:


kiva_loan.shape[0]


# In[15]:


# Reading Kiva Region Location file.
kiva_region_location = pd.read_csv("../input/kiva_mpi_region_locations.csv")
kiva_region_location.head(n=5)


# In[16]:


# Reading Kiva Loan theme File
kiva_loan_theme = pd.read_csv("../input/loan_theme_ids.csv")
kiva_loan_theme.head(n=3)


# In[17]:


# Reading Kiva Loan Themes by Region File
kiva_loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
kiva_loan_themes_by_region.head(n=3)


# In[18]:


import plotly
import plotly.plotly as py
#import plotly.figure_factory as ff
from plotly.tools import FigureFactory as ff
#plotly.tools.set_credentials_file(username='abhay.rnj', api_key='8MzosR0uht5BE3tfUlhd')
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
from plotly.graph_objs import Scatter,Figure,Layout
init_notebook_mode(connected=True)


# In[19]:


loan_count_by_country_df = pd.DataFrame(kiva_loan['country'].value_counts()).head(n=25).reset_index()
loan_count_by_country_df.columns = ['Country','number_of_loans']
loan_count_by_country_df.head(n=2)


# In[20]:


#scl = [[0.0,'rgb(242,240,248)'],[0.20,'rgb(215,220,225)'],[0.40,'rgb(190,195,198)'],[0.60,'rgb(160,170,180)'],[0.80,'rgb(140,144,150)'],[1.0,'rgb(110,114,120)']]
colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
loan_data = [ dict(
        type='choropleth',
        colorscale = colorscale,
        autocolorscale = False,
        reversescale = True,
        locations = loan_count_by_country_df['Country'],
        z = loan_count_by_country_df['number_of_loans'].astype(float),
        locationmode = 'USA-states',
        text = loan_count_by_country_df['Country'],
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 2
            )
        ),
        colorbar = dict(
            tickprefix = '',
            title = "Number Of Loans"
        )
    ) ]

loan_layout = dict(
    title = 'Loans Instances by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

#fig = dict(data=loan_data,layout=loan_layout)
#py.iplot(fig,validate=False,filename='choropleth-map')
fig = ff.dict(data=loan_data,layout=loan_layout)
py.iplot(fig)


# In[21]:


#Heatmap for number of loans by country


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 15,10
x = loan_count_by_country_df['Country']
y = loan_count_by_country_df['number_of_loans']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='number_of_loans',data=loan_count_by_country_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)


# In[23]:


import plotly.graph_objs as go
init_notebook_mode(connected=True)
#loan_count_by_country_df
trace = go.Pie(labels=loan_count_by_country_df.Country,values=loan_count_by_country_df.number_of_loans)
py.iplot([trace],file_name='Loan Count By Country')

# Post Execution Notes: The pie chart below shows that from countries like Philipines,Kenya,El Salvador,Cambodia highest number of loans are being taken.


# In[24]:


# Number of Loans by Sector


# In[25]:


kiva_loan.columns.values


# In[26]:


# number_of_loans_by_sector = kiva_loan.groupby(['sector'])['id'].count()
# print(number_of_loans_by_sector)
# print("*************")
import plotly.graph_objs as go
import plotly.plotly as py

kiva_loans_count_by_sector = kiva_loan['sector'].value_counts().reset_index()
kiva_loans_count_by_sector.columns = ['Sector','NumberOfLoans']
kiva_loans_count_by_sector.head(n=5)
trace = go.Pie(labels=kiva_loans_count_by_sector.Sector,values=kiva_loans_count_by_sector.NumberOfLoans)
py.iplot([trace],name='LoansBySector')


# In[27]:


# Average Dollar Value of Loan per Country


# In[28]:


average_loan_value_by_country_df = pd.DataFrame(kiva_loan.groupby(['country'])['loan_amount'].mean()).head(n=25).reset_index()
average_loan_value_by_country_df.columns = ['Country','AverageLoanAmount']
average_loan_value_by_country_df = average_loan_value_by_country_df.sort_values(by=['AverageLoanAmount'],ascending=False)
average_loan_value_by_country_df.head(n=5)


# In[29]:


rcParams['figure.figsize'] = 15,10
x = average_loan_value_by_country_df['Country']
y = average_loan_value_by_country_df['AverageLoanAmount']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='AverageLoanAmount',data=average_loan_value_by_country_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)


# In[30]:


# Total Loan Dollar Value Per Country
total_loan_value_by_country_df = pd.DataFrame(kiva_loan.groupby(['country'])['loan_amount'].sum()).head(n=25).reset_index()
total_loan_value_by_country_df.columns = ['Country','TotalLoanAmount']
total_loan_value_by_country_df = total_loan_value_by_country_df.sort_values(by=['TotalLoanAmount'],ascending=False)
total_loan_value_by_country_df.head(n=5)


# In[31]:


rcParams['figure.figsize'] = 15,10
x = total_loan_value_by_country_df['Country']
y = total_loan_value_by_country_df['TotalLoanAmount']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='TotalLoanAmount',data=total_loan_value_by_country_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

