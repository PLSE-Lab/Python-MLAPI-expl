#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# ### Objective of the competition:
# 
#  The objective of this competition is to help Kiva build models for assessing borrower welfare levels.
# 
# ### Objective of the notebook:
# 
# To explore on how Kiva is creating impact in the world using the dataset provided by Kiva,
#  
#  ##### Table of Content
#  
#  
#  
# 
# #### Imports

# In[3]:


import pandas as pd 
from sklearn import preprocessing
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly import tools
import numpy as np 

init_notebook_mode(connected=True)

pd.options.display.max_columns = 999

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(plotly.__version__)


# ## Dataset Exploration
# 
# In this section, Lets look into the given dataset and its features. 

# In[4]:


kiva_loans = pd.read_csv("../input/kiva_loans.csv")


# kiva loans exploration:
# 
# First lets explore into the loans given by kiva

# In[ ]:


table = ff.create_table(kiva_loans.head())
table.layout.width=2500
iplot(table)


# In this section, We will analyze on how 

# In[ ]:


sector_count = kiva_loans['sector'].value_counts().sort_index(ascending=True)
sector_ln_amt = kiva_loans.groupby(['sector'])['loan_amount'].sum().sort_index(ascending=True)

cnt_trace =Scatter(
   y=sector_count.values[::-1],
   x=sector_count.index[::-1],
   name='Loan Count',
   fill='tozeroy',
    
)

amt_trace = Scatter(
   y=sector_ln_amt.values[::-1],
   x=sector_ln_amt.index[::-1],
   name='Loan Amount',
   yaxis='y2',
   fill='tonexty',
     
)

data = [cnt_trace,amt_trace]

layout = Layout(
   title='Sector wise Loan Amount vs Loan Count ',
   width=800,
   height=500,
   yaxis=dict(
       title='Loan Count'
   ),
   yaxis2=dict(
       title='Loan Amount',
       titlefont=dict(
           color='rgb(148, 103, 189)'
       ),
       tickfont=dict(
           color='rgb(148, 103, 189)'
       ),
       overlaying='y',
       side='right'
   )
)


fig = Figure(data=data, layout=layout)
iplot(fig)


# lets look into the sector wise loan count based on gender. 

# In[ ]:


unique_sector = kiva_loans['sector'].unique()

gender_secotr_dist= {}
for sector in unique_sector:
    gender_list = []
    sector_df = kiva_loans[kiva_loans['sector']==sector]
    for r in sector_df['borrower_genders'].values:
        if str(r) != "nan":
            gender_list.extend( [l.strip() for l in r.split(",")] )
    gender_secotr_dist[sector] =  gender_list

kys = list(gender_secotr_dist.keys())

z = len(kys)
colm = 3
row = 5

data = []

x = [] 
y = [] 
for r in range(row):
    for c in range(colm):
        x.append([float(c)/float(colm),float(c+1)/float(colm)])
        y.append([float(r)/float(row),float(r+1)/float(row)])


for i in range(len(kys)):
    value = gender_secotr_dist.get(kys[i])

    pd_series = pd.Series(value).value_counts()

    labels = (np.array(pd_series.index))
    sizes = (np.array((pd_series / pd_series.sum())*100))
    
    data.append({
            'labels': labels,
            'values': sizes,
            'type': 'pie',
            'name': kys[i],
            'domain': {'x': x[i],
                       'y': y[i]},
            "hole": .4,
        })

layout = Layout(
    title='Sector wise Loan count by genders')


fig = Figure(data=data, layout=layout)
iplot(fig)


# with selection filter

# In[5]:


gender_list = []
for r in kiva_loans['borrower_genders'].values:
        if str(r) != "nan":
            gender_list.extend( [l.strip() for l in r.split(",")] )
            
pd_series = pd.Series(gender_list).value_counts()

alllabels = (np.array(pd_series.index))
allsizes = (np.array((pd_series / pd_series.sum())*100))


unique_sector = kiva_loans['sector'].unique()

ulen = list(kiva_loans['sector'].unique())

gender_secotr_dist= {}
for sector in unique_sector:
    gender_list = []
    sector_df = kiva_loans[kiva_loans['sector']==sector]
    for r in sector_df['borrower_genders'].values:
        if str(r) != "nan":
            gender_list.extend( [l.strip() for l in r.split(",")] )
    gender_secotr_dist[sector] =  gender_list
    
    
kys = list(gender_secotr_dist.keys())

data = []

layout = Layout(
    title='Sector wise Loan count by genders',
    updatemenus = [
    {
      'x': -0.05, 
      'y':1, 
      'buttons':[{
          "args": ["visible", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True]], 
          "label":  "All",
           "method":"restyle"             
        }],
    }
    ]
   )

for i in range(len(kys)):
    
    visiblity = [False for j in range(len(ulen)+1)]
    value = gender_secotr_dist.get(kys[i])
    pd_series = pd.Series(value).value_counts()

    labels = (np.array(pd_series.index))
    sizes = (np.array((pd_series / pd_series.sum())*100))
    
    data.append({
            'labels': labels,
            'values': sizes,
            'type': 'pie',
            'name': kys[i],
            "hole": .4,
        })
    
    visiblity[i] = True
    layout['updatemenus'][0]['buttons'].append(
     {
          "args": ["visible", visiblity], 
          "label":  kys[i],
           "method":"restyle"             
        }
    )
    
data.append({
            'labels': alllabels,
            'values': allsizes,
            'type': 'pie',
            'name':  'All',
             "hole": .4,
        })
    
fig = Figure(data=data, layout=layout)
iplot(fig) 


# From above chart, Female has borrowed more loan in all sectors and we can infer that Female has more interest in Arts, Clothing,Food sectors.
# 
# Let's check on how the top loan borrowed country spend their fund in different sectors.

# In[6]:


cntry_amt = kiva_loans.groupby(['country'])['loan_amount'].sum().reset_index()
top_loan_cntry = cntry_amt.sort_values('loan_amount', ascending=False)[:5]
 
top_country_sector_loan = pd.merge(kiva_loans, top_loan_cntry, on='country', how='inner')

top_country_sector_loan_distribution = top_country_sector_loan.groupby(['country','sector'])['loan_amount_x'].sum().reset_index()

top_country_sector_loan_distribution['cty_label'] = top_country_sector_loan_distribution['country']
top_country_sector_loan_distribution['sector_label'] = top_country_sector_loan_distribution['sector']

le = preprocessing.LabelEncoder()
le.fit(top_country_sector_loan_distribution['sector'].values.tolist() + top_country_sector_loan_distribution['country'].values.tolist())
top_country_sector_loan_distribution['country'] = le.transform(top_country_sector_loan_distribution['country'])
top_country_sector_loan_distribution['sector']= le.transform(top_country_sector_loan_distribution['sector'])
lable = le.classes_

data_trace = dict(
    type='sankey',
      domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    valuesuffix = "$",
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(
        color = "black",
        width = 0.5
      ),
     label =  lable
    
    ),
    link = dict(
      source =  top_country_sector_loan_distribution['country'] ,
      target =  top_country_sector_loan_distribution['sector'] ,
      value =  top_country_sector_loan_distribution['loan_amount_x'] ,
      label =  lable 
  ))

layout =  dict(
    title =  "Top 5 country Loan and its sector wise Distribution.",
    font = dict(
      size = 10
    )
)

fig = dict(data=[data_trace], layout=layout)
iplot(fig)


# Philippines borroweed the highed fund and it spend mostly in Agri, Food and Retail. 
# 
# Lets looks into the number of loans each country borrowed and the value.

# In[ ]:


cntry_lount_amt = kiva_loans.groupby(['country'])['loan_amount'].sum().sort_index(ascending=True)
cntry_lount_cnt = kiva_loans['country'].value_counts().sort_index(ascending=True)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ 
       dict(
        type='choropleth',
        locationmode = 'country names',
        autocolorscale = False,
        locations = cntry_lount_amt.index,
        z = cntry_lount_amt.values,
        colorscale = scl,
        name='Country wise Loan amount',
        reversescale = True,
        colorbar = dict(
            tickprefix = '',
            title = 'Number of Loans'),
         marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
        )
       ]

 
layout = dict(
    title = 'Country wise Loan amount distribution',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = Figure(data=data, layout=layout)
iplot( fig, validate=False, )


# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ 
       dict(
        type='choropleth',
         locationmode = 'country names',
        autocolorscale = False,
        locations = cntry_lount_cnt.index,
        z = cntry_lount_cnt.values,
        colorscale = scl,
        name='Country wise Loan amount',
         marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
        
        )
       ]

 

layout = dict(
    title = 'Country wise Loan count distribution',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = Figure(data=data, layout=layout)
iplot( fig)


# above charts shows that some of the countires borrowed less number of loans but the sum of loan amount is high compared to other countries.
# 
# Lets explore on repayment interval based on sector and gender.

# In[ ]:


repayment_interval_count = kiva_loans['repayment_interval'].value_counts()

data = [Bar(
            x=repayment_interval_count.index,
            y=repayment_interval_count.values,
         marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
    )]

layout = Layout(
    title = 'Repayment interval distribution'
)

fig = Figure(data=data,layout = layout) 
iplot(fig)


# Most of the loans have monthly installment period.  Let see how the gender distribution in each interval.

# In[ ]:


repayment_interval = kiva_loans['repayment_interval'].unique()

gender_interval_dist= {}
for interval in repayment_interval:
    gender_list = []
    interval_df = kiva_loans[kiva_loans['repayment_interval']==interval]
    for r in interval_df['borrower_genders'].values:
        if str(r) != "nan":
            gender_list.extend( [l.strip() for l in r.split(",")] )
    gender_interval_dist[interval] =  gender_list

kys = list(gender_interval_dist.keys())


female = [] 
male = [] 
for interval in kys:
    olist_value = gender_interval_dist.get(interval)
    temp_series = pd.Series(olist_value).value_counts()
    female.append(temp_series['female'])
    male.append(temp_series['male'])
    
print(female,male)

data = [Bar(
            x=kys,
            y=female,
        name='Female'
        
    ),
        Bar(
            x=kys,
            y=male,
         name='Male'
    )
       ]

layout = Layout(
    title = 'Repayment interval distribution based on Gender',
     barmode='stack'
)

fig = Figure(data=data,layout = layout) 
iplot(fig)
    


# Below chart explains on how in each sector the repayment interval varies

# In[ ]:


sector_repayment_interval = kiva_loans.groupby(['sector','repayment_interval'])['loan_amount'].count()
sector_repayment_interval = sector_repayment_interval.reset_index() 

sector_repayment_interval.columns = ['sector','repayment_interval','loan_count']

data = []
for inter in sector_repayment_interval['repayment_interval'].unique():
    sector = sector_repayment_interval[sector_repayment_interval['repayment_interval'] == inter]['sector']
    count = sector_repayment_interval[sector_repayment_interval['repayment_interval'] == inter]['loan_count']

    data.append(Bar(
            x=sector,
            y=count,
            name=inter  
    ))

layout = Layout(
    title = 'Repayment interval distribution based on Gender',
     barmode='stack'
)

fig = Figure(data=data,layout = layout) 
iplot(fig)


# Wholesale,Personal use, Housing and Education does not have the Weekly installment. 
# 
# **Yet to add more... Consider for upvote and please leave your comments.**

# 
