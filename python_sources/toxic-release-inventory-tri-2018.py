#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


tri=pd.read_csv('/kaggle/input/release/TRI_2018_FED.csv')
tri


# * EPA has been collecting Toxic Release Inventory (TRI) data since 1987. Each "Basic" data file accessible from this webpage contains the 100 most-requested data fields from the TRI Reporting Form R and Form A. 
# 
# Quantities of dioxin and dioxin-like compounds are reported in grams, while all other chemicals are reported in pounds.

# There are a couple problems with the data at first glance. Many columns are ID's or tracking information that's not important for this analysis. Let's extract the relevat data to make the dataframe more readable.

# In[ ]:


descriptions=tri.iloc[:,[0,3,4,5,6,7,8,13,15,16,17,18,32,33,36,37,38,40,42,99]]
descriptions


# With 116 columns, this microset will be the descriptions for all the data entries. There is a lot of numerical data to understand and sort through, so let's hold that off until later. Tri is still our original dataset so all of the original data is still in tact.

# In[ ]:


# Python program to Remove all  
# digits and periods from a list of string 
import re 
  
def remove(strings): 
    pattern = '[0-9,.]'
    strings = [re.sub(pattern, '', i) for i in strings] 
    return strings
  
#call method on all column names  
descriptions.columns=remove(descriptions.columns)

#call strip method to get rid of leading and trailing white space
descriptions.columns=descriptions.columns.str.strip()
descriptions


# This looks much better- the numbers in the headings made it confusing and extra to type in.

# Let's try to make a simple pie graph displaying the percent of carcinogenic toxins.

# In[ ]:


descriptions['CARCINOGEN'].value_counts()


# In[ ]:


#create dataframe with amounts and their title (index)
piechart = pd.DataFrame({'Carcinogen': [descriptions['CARCINOGEN'].value_counts()[0],descriptions['CARCINOGEN'].value_counts()[1]]},
                  index=['no', 'yes'])

#use dataframe plot function, specify as pie chart
plot = piechart.plot.pie(y='Carcinogen', figsize=(5, 5),autopct='%1.1f%%')


# Make pie chart showing distrbution of clean air act chemicals.

# In[ ]:


descriptions['CLEAN AIR ACT CHEMICAL'].value_counts()


# In[ ]:


piechart = pd.DataFrame({'Clean Air Act Chemical': 
        [descriptions['CLEAN AIR ACT CHEMICAL']
         .value_counts()[0],descriptions['CLEAN AIR ACT CHEMICAL'].value_counts()[1]]},
                  index=['yes', 'no'])
plot = piechart.plot.pie(y='Clean Air Act Chemical', figsize=(5, 5),autopct='%1.1f%%')


# Which states had the most dumps?

# In[ ]:


#create series of the 5 states that were cited the most 
most_dumps=descriptions['ST'].value_counts()[0:5]

#create dataframe of number of dumps and the respective state
#rather than list data points one by one, can call the indices of the most_dumps series
md = pd.DataFrame({'Number of Releases': [most_dumps[i] for i in range(len(most_dumps))]},
                  index=[most_dumps.index[j] for j in range(len(most_dumps.index))])
plot = md.plot.pie(y='Number of Releases', figsize=(5, 5),autopct='%1.1f%%')


# How about the most total amount of chemicals released?

# In[ ]:


descriptions['UNIT OF MEASURE'].value_counts()


# In[ ]:


#series of all states
states=descriptions['ST'].value_counts().index

#create blank dataframe that we'll add to
totals=pd.DataFrame()

#iterate through every stat
for state in states:
    
    #extract all of state's data from full dataset
    state_df=descriptions[descriptions['ST'].isin([state])]
    state_df=state_df.reset_index()
    
    #set variable to zero before we add the mass of all chemicals released for given state
    weight=0
    
    #iterate through dataframe of state's data
    for release in range(len(state_df)):
        
        #most entries are in pounds, but use if statement to account for the ones in grams
        if state_df['UNIT OF MEASURE'][release]=='Grams':
            weight+=(state_df['TOTAL RELEASES'][release])*0.00220462
        else:
            weight+=state_df['TOTAL RELEASES'][release]
            
    #add to the empty dataframe: name of state and the total amountof chemicals released        
    totals=totals.append({'state': state,'total':weight},ignore_index=True)
totals


# In[ ]:


#get dataframe with top 5 highest amounts
top=totals.nlargest(5,'total').reset_index()


# In[ ]:


md2 = pd.DataFrame({'Total mass released': [top['total'][i] for i in range(len(top))]},
                  index=[top['state'][j] for j in range(len(top))])
plot = md2.plot.pie(y='Total mass released', figsize=(5, 5),autopct='%1.1f%%')


# Interesting- Alaska wasn't even in the top five for the most number of releases, but they have released nearly as many chemicals (by mass) as the next four combined. Let's see if there are any large corporations in Alaska that may be primary contributors to this.

# In[ ]:


descriptions.sort_values(by=['TOTAL RELEASES'],ascending=False).head()


# Red Dog Operations, operating in Alaska, has the two largest releases tracked in the TRI- by far. This explains why Alaska's releases were so high.

# In[ ]:


descriptions[descriptions['FACILITY NAME'].isin(['RED DOG OPERATIONS'])]


# Red dog operations is a zince minng company, which explains why a lot of their dumps are so heavy. Their website acknowledges that they standout on the TRI, and Alaska Department of Environmental Conservation addresses concerns and asserts that there is little to no effect to human health or the envionemt, and that their chemical dumps are barely make the cut to be considered toxic.

# Let's take a look at all of the carcinogenic releases.

# In[ ]:


carcinogens=descriptions[descriptions['CARCINOGEN'].isin(['YES'])].reset_index()

#create series of the 5 states that were cited the most 
most_dumps_carcinogenic=carcinogens['ST'].value_counts()[0:5]

#create dataframe of number of dumps and the respective state
#rather than list data points one by one, can call the indices of the most_dumps series
mdc = pd.DataFrame({'Number of Carcinogenic Releases': 
                   [most_dumps[i] for i in range(len(most_dumps_carcinogenic))]},
    index=[most_dumps.index[j] for j in range(len(most_dumps_carcinogenic.index))])
plot = mdc.plot.pie(y='Number of Carcinogenic Releases', figsize=(5, 5),autopct='%1.1f%%')


# In[ ]:


totals_carcinogenic=pd.DataFrame()

for state in states:
    
    state_df_carcinogenic=carcinogens[carcinogens['ST'].isin([state])]
    state_df_carcinogenic=state_df_carcinogenic.reset_index()
    weight_carcinogenic=0
    
    for release in range(len(state_df_carcinogenic)):
        if state_df_carcinogenic['UNIT OF MEASURE'][release]=='Grams':
            weight_carcinogenic+=(state_df_carcinogenic['TOTAL RELEASES'][release])*0.00220462
        else:
            weight_carcinogenic+=state_df_carcinogenic['TOTAL RELEASES'][release]
            
    totals_carcinogenic=totals_carcinogenic.append({'state': state,'total':weight_carcinogenic},ignore_index=True)

#get data from all other states for comparison
others=carcinogens[-carcinogens['ST'].isin(['TX','IL','LA','IN','MI'])].reset_index()
weight_others=0
for release in range(len(others)):
        if others['UNIT OF MEASURE'][release]=='Grams':
            weight_others+=(others['TOTAL RELEASES'][release])*0.00220462
        else:
            weight_others+=others['TOTAL RELEASES'][release]

others_df=pd.DataFrame({'Total carcinogenic material released':weight_others},index=['others'])            

top_carcinogenic=totals_carcinogenic.nlargest(10,'total').reset_index()
mdc2 = pd.DataFrame({'Total carcinogenic material released': [top_carcinogenic['total'][i] for i in range(len(top_carcinogenic))]},
                  index=[top_carcinogenic['state'][j] for j in range(len(top_carcinogenic))])
mdc2_full=mdc2.append(others_df)

plot=mdc2_full.plot.pie(y='Total carcinogenic material released', figsize=(5, 5),autopct='%1.1f%%')
plot.legend(loc='center right',bbox_to_anchor=(1.5, 0.5))


# Let's change up the analysis and look at the different industry sectors.

# Which sector is the most carcinogenic? Which has released the most carcinogenic materials?

# In[ ]:


sectors=descriptions['INDUSTRY SECTOR'].value_counts().index

sectors_list=[]
for sector in sectors:
    df=descriptions[descriptions['INDUSTRY SECTOR'].isin([sector])]
    sectors_list.append(df)

def get_sector(sector):
    for num in range(len(sectors_list)):
        short=sectors_list[num]['INDUSTRY SECTOR'].value_counts()
        if short.index[0]==sector:
            return sectors_list[num].reset_index()


# In[ ]:


get_sector('Chemicals')


# In[ ]:


by_sector=pd.DataFrame()
for sector in sectors:
    df=get_sector(sector)
    total=0
    for release in range(len(df)):
        if df['UNIT OF MEASURE'][release]=='Grams':
            total+=(df['TOTAL RELEASES'][release])*0.00220462
        else:
            total+=df['TOTAL RELEASES'][release]
    by_sector=by_sector.append({'sector':sector,'Total material released':total},ignore_index=True)


# In[ ]:


top=by_sector.nlargest(7,'Total material released').reset_index()
dfpie=pd.DataFrame({'Total':[top['Total material released'][row] for row in range(len(top))]},
                   index=[top['sector'][row] for row in range(len(top))])
plot=dfpie.plot.pie(y='Total', figsize=(5, 5),autopct='%1.1f%%')
plot.legend(loc='center right',bbox_to_anchor=(1.7, 0.5))


# In[ ]:


others=by_sector[-by_sector['sector'].isin([top['sector'][i]for i in range(len(top))])].reset_index()
weight=0
for release in range(len(others)):
        weight+=others['Total material released'][release]

others_df=pd.DataFrame({'Total':weight},index=['others']) 
dfpie=dfpie.append(others_df)
plot=dfpie.plot.pie(y='Total', figsize=(5, 5),autopct='%1.1f%%')
plot.legend(loc='center right',bbox_to_anchor=(1.7, 0.5))


# Metal mining stands out- what metals contribute most to this?

# In[ ]:


metals=get_sector('Metal Mining')['CHEMICAL'].value_counts().index
df=get_sector('Metal Mining')
metals_df=pd.DataFrame()

for metal in metals:
    metal_df=df[df['CHEMICAL'].isin([metal])].reset_index()
    weight=0
    for release in range(len(metal_df)):
        if metal_df['UNIT OF MEASURE'][release]=='Grams':
            weight+=(metal_df['TOTAL RELEASES'][release])*0.00220462
        else:
            weight+=metal_df['TOTAL RELEASES'][release]
    metals_df=metals_df.append({'Metal':metal,'Total material released': weight},ignore_index=True)
top=metals_df.nlargest(7,'Total material released').reset_index()
dfpie=pd.DataFrame({'Total':[top['Total material released'][row] for row in range(len(top))]},
                   index=[top['Metal'][row] for row in range(len(top))])

#total material for all but the top
others=metals_df[-metals_df.isin([top['Metal'][i]for i in range(len(top))])]
total=0
for release in range(len(others)):
        total+=others['Total material released'][release]

others_df=pd.DataFrame({'Total':total},index=['others']) 
dfpie=dfpie.append(others_df)
plot=dfpie.plot.pie(y='Total', figsize=(5, 5),autopct='%1.1f%%')
plot.legend(loc='center right',bbox_to_anchor=(2.5, 0.5))


# In[ ]:


Create method to view breakdown of chemicals within an industry sector.


# In[ ]:


def sector_breakdown(industry_sector,num_comparisons):
    metals=get_sector(industry_sector)['CHEMICAL'].value_counts().index
    df=get_sector(industry_sector)
    metals_df=pd.DataFrame()

    for metal in metals:
        metal_df=df[df['CHEMICAL'].isin([metal])].reset_index()
        weight=0
        for release in range(len(metal_df)):
            if metal_df['UNIT OF MEASURE'][release]=='Grams':
                weight+=(metal_df['TOTAL RELEASES'][release])*0.00220462
            else:
                weight+=metal_df['TOTAL RELEASES'][release]
        metals_df=metals_df.append({'Metal':metal,'Total material released': weight},ignore_index=True)
    top=metals_df.nlargest(num_comparisons,'Total material released').reset_index()
    dfpie=pd.DataFrame({'Total':[top['Total material released'][row] for row in range(len(top))]},
                       index=[top['Metal'][row] for row in range(len(top))])

    #total material for all but the top
    others=metals_df[-metals_df.isin([top['Metal'][i]for i in range(len(top))])]
    total=0
    for release in range(len(others)):
            total+=others['Total material released'][release]

    others_df=pd.DataFrame({'Total':total},index=['others']) 
    dfpie=dfpie.append(others_df)
    plot=dfpie.plot.pie(y='Total', figsize=(5, 5),autopct='%1.1f%%')
    plot.legend(loc='center right',bbox_to_anchor=(2.5, 0.5))
    return 


# Builds off the previous method by looking at carcinogens only.

# In[ ]:


def sector_breakdown_carcinogenic(industry_sector,num_comparisons):
    metals=get_sector(industry_sector)['CHEMICAL'].value_counts().index
    df=get_sector(industry_sector)
    metals_df=pd.DataFrame()

    for metal in metals:
        metal_df=df[df['CHEMICAL'].isin([metal]) & df['CARCINOGEN'].isin(['YES'])].reset_index()
        weight=0
        for release in range(len(metal_df)):
            if metal_df['UNIT OF MEASURE'][release]=='Grams':
                weight+=(metal_df['TOTAL RELEASES'][release])*0.00220462
            else:
                weight+=metal_df['TOTAL RELEASES'][release]
        metals_df=metals_df.append({'Metal':metal,'Total material released': weight},ignore_index=True)
    top=metals_df.nlargest(num_comparisons,'Total material released').reset_index()
    dfpie=pd.DataFrame({'Total':[top['Total material released'][row] for row in range(len(top))]},
                       index=[top['Metal'][row] for row in range(len(top))])

    #total material for all but the top
    others=metals_df[-metals_df.isin([top['Metal'][i]for i in range(len(top))])]
    total=0
    for release in range(len(others)):
            total+=others['Total material released'][release]

    others_df=pd.DataFrame({'Total':total},index=['others']) 
    dfpie=dfpie.append(others_df)
    plot=dfpie.plot.pie(y='Total', figsize=(5, 5),autopct='%1.1f%%')
    plot.legend(loc='center right',bbox_to_anchor=(2.5, 0.5))
    return 


# In[ ]:


sector_breakdown('Food',5)
sector_breakdown_carcinogenic('Food',5)

