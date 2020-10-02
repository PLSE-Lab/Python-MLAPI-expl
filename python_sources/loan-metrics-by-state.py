#!/usr/bin/env python
# coding: utf-8

# <h1> A notebook to visualize loan data state by state. </h1>

# In[ ]:


#https://plot.ly/python/choropleth-maps/
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
import matplotlib.pyplot as plt 
df = pd.read_csv("../input/loan.csv", low_memory=False)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
#print (df.columns)


# <h1> Cell for processing data for Plotly output </h1>

# In[ ]:


State_List = []
for x in df['addr_state']:
    if x not in State_List:
        State_List.append(x)

Loan_Amount = []
Average_Balance = []
Default_Rate = []
Weighted_Rate = []
Average_Income = []
Average_Employment_Length = []
Average_DTI = []
Average_Inq_12 = []
Average_Inq_6 = []

for x in State_List:
    new_df = df[df['addr_state'] == x]
    
    Loan_Sum = sum(new_df['funded_amnt'])
    Loan_Amount.append(Loan_Sum)
    
    Average_Balance.append(Loan_Sum/len(new_df['funded_amnt']))
    
    Defaults = []
    for value in new_df.loan_status:
        if value == 'Default':
            Defaults.append(1)
        if value == 'Charged Off':
            Defaults.append(1)
        if value == 'Late (31-120 days)':
            Defaults.append(1)   
        if value == 'Late (16-30 days)':
            Defaults.append(1)
        if value == 'Does not meet the credit policy. Status:Charged Off':
            Defaults.append(1) 
    Default_R = len(Defaults) / len(new_df.loan_status)  
    Default_Rate.append(Default_R)
    
    new_df['weighted'] = (new_df['int_rate']/100)*new_df['funded_amnt']
    Weighted_Sum = sum(new_df['weighted'])
    Weighted_i_rate = Weighted_Sum / Loan_Sum
    Weighted_Rate.append(Weighted_i_rate)
    
    Income_Average = np.mean(new_df['annual_inc'])
    Average_Income.append(Income_Average)
    

    Employ_Length = []
    for term in new_df.emp_length:
        if term == '10+ years':
            Employ_Length.append(10)
        if term == '< 1 year':
            Employ_Length.append(0.5)    
        if term == '1 year':
            Employ_Length.append(1)
        if term == '3 years':
            Employ_Length.append(3)
        if term == '8 years':
            Employ_Length.append(8)
        if term == '9 years':
            Employ_Length.append(9)    
        if term == '4 years':
            Employ_Length.append(4)
        if term == '5 years':
            Employ_Length.append(5)
        if term == '6 years':
            Employ_Length.append(6)
        if term == '2 years':
            Employ_Length.append(2)    
        if term == '7 years':
            Employ_Length.append(7)
        if term == 'n/a':
            Employ_Length.append(0)  
            
    Average_Employment_Length.append(np.mean(Employ_Length))        
    
    DTI_Average = np.mean(new_df['dti'])
    Average_DTI.append(DTI_Average)
    
    inquiry_average = np.mean(new_df['inq_last_12m'])
    Average_Inq_12.append(inquiry_average)
    
    inquiry_average_6 = np.mean(new_df['inq_last_6mths'])
    Average_Inq_6.append(inquiry_average_6)
    
from collections import OrderedDict
combine_data = OrderedDict([ ('Loan_Funding',Loan_Amount),
                         ('Average_Balance', Average_Balance),
                         ('Default_Rate',  Default_Rate),
                         ('Weighted_Rate', Weighted_Rate),
                         ('Average_Income', Average_Income),
                         ('Average_Employment_Length', Average_Employment_Length),
                         ('Average_DTI', DTI_Average),
                         ('12m_Inquiries', Average_Inq_12),
                         ('6m_Inquiries', Average_Inq_6),   
                         ('code', State_List)])

df_plot = pd.DataFrame.from_dict(combine_data)
df_plot = df_plot.round(decimals=2)
df_plot.head()


# In[ ]:


for col in df_plot.columns:
    df_plot[col] = df_plot[col].astype(str)

    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df_plot['text'] = df_plot['code'] + '<br>' +    'Avg Balance Per Borrower ($ USD): '+df_plot['Average_Balance']+'<br>'+    'Avg Employment Term Per Borrower (Years): '+df_plot['Average_Employment_Length']+'<br>'+    'Avg Annual Income Per Borrower ($ USD): '+df_plot['Average_Income']
    

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_plot['code'],
        z = df_plot['Loan_Funding'], 
        locationmode = 'USA-states',
        text = df_plot['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "$s USD")
        ) ]

layout = dict(
        title = 'Lending Club Portfolio<br> Total Funded By State <br> (Hover over state for other metrics)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# In[ ]:


# from the tutorial, leaving in just for now
for col in df_plot.columns:
    df_plot[col] = df_plot[col].astype(str)

    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df_plot['text'] = df_plot['code'] + '<br>' +    '<br>'+'Weighted Rate: '+df_plot['Weighted_Rate']+'<br>'+    'Inquiries Last 12m: '+df_plot['12m_Inquiries']+'<br>'+    'Inquiries Last 6m: '+df_plot['6m_Inquiries']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = True,
        locations = df_plot['code'],
        z = df_plot['Default_Rate'], #.astype(int),
        locationmode = 'USA-states',
        text = df_plot['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "%")
        ) ]

layout = dict(
        title = 'Lending Club Portfolio<br> Default Rate By State <br> (Hover over state for other metrics)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# <h1> Create A Bar Chart For Default Rate Based on Count of 6 Month Inquiries </h1>

# In[ ]:


#df = pd.read_csv("../input/loan.csv", low_memory=False)
#print (df.columns)
print ("Length of All Data:",len(df.loan_status)) # get total length of data
print ("Nans in Loan Status Column:",df.loan_status.isnull().sum()) # get nans in loan status column
print ("Nans in 12mth inquiry column:",df.inq_last_12m.isnull().sum()) # get nans in inquiry 12 month column
print ("Nans in 6mth inquiry column:",df.inq_last_6mths.isnull().sum()) # get nans in inquiry 6 month column
print ("Nans in last credit date columns:",df.last_credit_pull_d.isnull().sum()) # get nans in last credit pull date column
print ("First data point in last credit pulled column:",df.last_credit_pull_d.iloc[0])
print ("First data point in 6mth inquiry column:",df.inq_last_6mths.iloc[0])
print ("First data point in issue_d  column:",df.issue_d.iloc[0]) # no good, all dec 2011
print ("First data point in 12 mth collection column:",df.collections_12_mths_ex_med.iloc[0])
print (np.mean(df.collections_12_mths_ex_med))
cols_to_keep = ['loan_status','inq_last_6mths','collections_12_mths_ex_med']
new_df = df[cols_to_keep]
new_df.head()


# In[ ]:


print (len(new_df.loan_status))
print (new_df.loan_status.isnull().sum())
print (new_df.inq_last_6mths.isnull().sum())
print (new_df.collections_12_mths_ex_med.isnull().sum())
new_df = new_df.dropna(axis=0)
print (len(new_df.loan_status))


# In[ ]:


new_df['default_binary'] = 0 # dummy columns
for index,row in new_df.iterrows():
        if row['loan_status'] == 'Default':
            new_df.set_value(index, 'default_binary', 1)
        if row['loan_status'] == 'Charged Off':
            new_df.set_value(index, 'default_binary', 1)
        if row['loan_status'] == 'Late (31-120 days)':
            new_df.set_value(index, 'default_binary', 1)  
        if row['loan_status'] == 'Late (16-30 days)':
            new_df.set_value(index, 'default_binary', 1)
        if row['loan_status'] == 'Does not meet the credit policy. Status:Charged Off':
            new_df.set_value(index, 'default_binary', 1)
new_df.head()           


# In[ ]:


inquiry = []
for x in new_df.inq_last_6mths:
    if x not in inquiry:
        if x <= 10.0:
            inquiry.append(x)
        
inquiry.sort()

the_dict = {}        
for x in inquiry:
    dfn = new_df[new_df.inq_last_6mths == x]
    #print (len(dfn.default_binary))
    dfn_d = dfn[dfn.default_binary == 1]
    the_dict[x] = len(dfn_d.default_binary) / len(dfn.default_binary)
print (the_dict)    


# In[ ]:


# combine all inquiries greater than 10
dfn = new_df[new_df.inq_last_6mths >= 10]
dfn_d = dfn[dfn.default_binary == 1]
the_dict[10.0] = len(dfn_d.default_binary) / len(dfn.default_binary)
#the_dict.pop(30.0)# popped out a bunch of data points higher than 10
print (the_dict)


# In[ ]:


plt.bar(range(len(the_dict)), the_dict.values(), align='center')
plt.xticks(range(len(the_dict)), the_dict.keys())
plt.title("Default Rate Based on Count of Inquiries in Last 6mths")
plt.xlabel("Count of Inquiries in Last 6 Months")
plt.ylabel("Default Rate")
plt.show()

