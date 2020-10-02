#!/usr/bin/env python
# coding: utf-8

# **PROJECT - GDP ANALYSIS OF INDIAN STATES**

# ### PART 1
# 
# 1.  Remove the rows: '(% Growth over the previous year)' and 'GSDP - CURRENT PRICES (` in Crore)' for the year 2016-17.
# 2.  Calculate the average growth of states over the duration 2013-14, 2014-15 and 2015-16 by taking the mean of the row '(% Growth over previous year)'. Compare the calculated value and plot it for the states. Make appropriate transformations if necessary to plot the data. Report the average growth rates of the various states:
#      - Which states have been growing consistently fast, and which ones have been struggling?
#      - Curiosity exercise - what has been the average growth rate of your home state, and how does it compare to the national average over this duration?
# 3.  Plot the total GDP of the states for the year 2015-16:
#      - Identify the top-5 and the bottom-5 states based on total GDP

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing data from CSV file into pandas dataframe

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
 
df_rawgsdp = pd.read_csv('/kaggle/input/ab40c054-5031-4376-b52e-9813e776f65e.csv.csv')
df_rawgsdp


# In[ ]:


##Remove the rows: '(% Growth over the previous year)' and 'GSDP - CURRENT PRICES (` in Crore)' for the year 2016-17
df_rawgsdp = df_rawgsdp[df_rawgsdp.Duration != '2016-17']

#Setting index to item duration so that i can chose rows based on item description value
df_rawgsdp = df_rawgsdp.set_index('Items  Description')
df_rawgsdp


# In[ ]:


#Dividing the dataframe into two part for GSDP Value and %Growth Value
df_gsdpcurrent = df_rawgsdp.filter(like='GSDP', axis=0)
df_gsdpgrowth = df_rawgsdp.filter(like='Growth', axis=0)

#using transpose for unpivoting and to have states in column
df_gsdpcurrent = df_gsdpcurrent.set_index('Duration').T
df_gsdpgrowth = df_gsdpgrowth.set_index('Duration').T

df_gsdpcurrent.index.name = 'States'
df_gsdpgrowth.index.name = 'States'

df_gsdpcurrent = df_gsdpcurrent.add_prefix('GSDP_')
df_gsdpgrowth = df_gsdpgrowth.add_prefix('Percentage Growth ')

#checking data
del df_gsdpcurrent.columns.name
df_gsdpcurrent.head(10)


# In[ ]:


#checking data
del df_gsdpgrowth.columns.name
df_gsdpgrowth.head(10)


# ### Calculate the average growth of states over the duration 2013-14, 2014-15 and 2015-16 by taking the mean of the row '(% Growth over previous year)'.
# 
# -  Compare the calculated value and plot it for the states. Make appropriate transformations if necessary to plot the data. Report the average growth rates of the various states:
# 
# -  Which states have been growing consistently fast, and which ones have been struggling?

# In[ ]:


#dropping row for year 2012-13 because analysis only has to be done for 2013-14, 2014-15 and 2015-16
df_gsdpgrowth = df_gsdpgrowth.drop('Percentage Growth 2012-13', axis=1)

#dropping row for West Bengal Value since it contains no data
df_gsdpgrowth = df_gsdpgrowth.dropna(axis=0, thresh=1)
df_gsdpgrowth


# In[ ]:


#checking dtypes of columns

df_gsdpgrowth['Average Growth Percentage'] = df_gsdpgrowth.mean(axis=1)
df_gsdpgrowth=df_gsdpgrowth.sort_values(by='Average Growth Percentage', ascending = False)
df_gsdpgrowth=df_gsdpgrowth.round({'Average Growth Percentage': 2})
df_gsdpgrowth


# In[ ]:


df_gsdpgrowth_avg=df_gsdpgrowth.filter(like='Average', axis=1)
del df_gsdpgrowth_avg.columns.name
df_gsdpgrowth_avg


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
plot_gsdp_meangrowth = sns.barplot(x=df_gsdpgrowth['Average Growth Percentage'], y=df_gsdpgrowth.index, data=df_gsdpgrowth)
plt.xlabel("Average Growth Percentage")
plt.ylabel("States")
plt.title("Average Growth Rates of States over 2013 to 2016")
plt.show()


# In[ ]:


#top 5 states, consistetly growing
df_gsdpgrowth[['Average Growth Percentage']].head()


# In[ ]:


#bottom 5 states, struggling to grow
df_gsdpgrowth[['Average Growth Percentage']].tail()


# ### Plot the total GDP of the states for the year 2015-16:
# - Identify the top-5 and the bottom-5 states based on total GDP

# In[ ]:


#creating a new dataframe with relevant values
df_totalgdp15_16 = df_gsdpcurrent.filter(items=['GSDP_2015-16'], axis=1)

#sorting based on GDP values
df_totalgdp15_16 = df_totalgdp15_16.sort_values(by='GSDP_2015-16', ascending = False)

#dropping rows with null values and all india GDP value from dataframe
df_totalgdp15_16 = df_totalgdp15_16.dropna()
df_totalgdp15_16 = df_totalgdp15_16.drop('All_India GDP', axis=0)
del df_totalgdp15_16.columns.name
df_totalgdp15_16


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
plot_totalgsdp = sns.barplot(x=df_totalgdp15_16['GSDP_2015-16'], y=df_totalgdp15_16.index, data=df_totalgdp15_16)
plt.xlabel("Total GDP of States")
plt.ylabel("States")
plt.title("GSDP for all States in 2015-2016")
plt.show()


# In[ ]:


#top 5 states
df_totalgdp15_16.head()


# In[ ]:


#Bottom 5 states
df_totalgdp15_16.tail()


# ### PART 2
# #### Perform the analysis only for the duration : 2014-15. 
# 1. Filter out the Union Territories (Delhi, Chandigarh, Andaman and Nicobar Islands etc.) for further analysis since they are governed directly by the centre, not state governments.
# 2. Plot the GDP per capita for all the states.
#     - Identify the top-5 and the bottom-5 states based on GDP per capita.
#     - Find the ratio of highest per capita GDP to the lowest per capita GDP
# 

# In[ ]:


import pandas as pd
import os
#dirs=os.listdir('/kaggle/input')
dirs = os.listdir('../input/')
df_1=[ ]
for items in dirs:
    if items.find('GSVA')>0 and items.find('csv')>0:
        x="../input/"
        i=x+items
        df_temp2=pd.read_csv((i), encoding='ISO-8859-1')
        df_temp2=df_temp2.loc[::,['S.No.','Item','2014-15']]
        df_temp2['State']=items.split('-')[1]
        df_1.append(df_temp2)
mastergdp=pd.concat(df_1,axis=0, sort=False)
mastergdp.State = mastergdp.State.str.replace('_', ' ')
mastergdp.head(10)


# In[ ]:


#creating a new dataframe with relevant values
df_gdp_percapita=mastergdp.loc[32,['2014-15','State']]

df_gdp_percapita = df_gdp_percapita.set_index('State')
df_gdp_percapita.rename(columns = {'2014-15':'Per Capita GDP in 2014 15'}, inplace = True)

#sorting based on GDP values
df_gdp_percapita = df_gdp_percapita.sort_values(by='Per Capita GDP in 2014 15', ascending = False)
df_gdp_percapita
df_gdp_percapita_x=df_gdp_percapita.copy()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
plot_percapitagdp = sns.barplot(x=df_gdp_percapita['Per Capita GDP in 2014 15'], y=df_gdp_percapita.index, data=df_gdp_percapita)
plt.xlabel("Per Capital GDP of States")
plt.ylabel("States")
plt.title("Per Capita GDP for States in 2014-2015")
plt.show()


# In[ ]:


# Identifying the top 5 states
df_gdp_percapita.head()


# In[ ]:


# Identifying the bottom 5 states
df_gdp_percapita.tail()


# #### Find the ratio of highest per capita GDP to the lowest per capita GDP

# In[ ]:


Ratio_highest_lowest = round(max(df_gdp_percapita['Per Capita GDP in 2014 15'])/min(df_gdp_percapita['Per Capita GDP in 2014 15']),2)
print ("The Ratio of highest GDP per capita to lowest GDP per capita is",Ratio_highest_lowest)


# #### Plot the percentage contribution of primary, secondary and tertiary sectors as a percentage of total GDP for all the states.

# In[ ]:


#creating a new dataframe with relevant values
df_gdpcontribution=mastergdp.loc[mastergdp['Item'].isin(['Primary','Secondary','Tertiary','Gross State Domestic Product']), ['Item','2014-15','State']]
df_gdpcontribution.reset_index(drop=True)
df_gdpcontribution.head()


# In[ ]:


# Cleaning and preparing data for analyis
df_gdpcontribution.rename(columns = {'2014-15':'Total GSDP 2014-15'}, inplace = True)
df_gdpcontribution =df_gdpcontribution.pivot(index='State', columns='Item', values='Total GSDP 2014-15')
df_gdpcontribution = df_gdpcontribution.sort_values(by='Gross State Domestic Product', ascending = False)
columnsTitles = ['Primary','Secondary','Tertiary','Gross State Domestic Product']
df_gdpcontribution = df_gdpcontribution.reindex(columns=columnsTitles)
del df_gdpcontribution.columns.name
df_gdpcontribution.head()


# In[ ]:


df_gdpcontribution_1=df_gdpcontribution.iloc[:,0:3].apply(lambda s: s*100 / df_gdpcontribution.iloc[:, 3])
df_gdpcontribution_1=df_gdpcontribution_1.add_suffix('_Percentage_Contribution')
df_gdpcontribution_1.head()


# In[ ]:


colors = ["#808080", "#00FA9A","#20B2AA"]
df_gdpcontribution_1.loc[:,['Primary_Percentage_Contribution','Secondary_Percentage_Contribution','Tertiary_Percentage_Contribution']].plot.barh(stacked=True, color=colors, figsize=(15,12))
plt.show()


# ### Categorise the states into four categories based on GDP per capita 
# - (C1, C2, C3, C4 - C1 would have the highest per capita GDP, C4 the lowest). The quantile values are (0.20,0.5, 0.85, 1)
# 
# #### For each category C1, C2, C3, C4:
# - Find the top 3/4/5 sub-sectors (such as agriculture, forestry and fishing, crops, manufacturing etc.) which contribute to approx. 80% of the GSDP of each category
# - Plot the contribution of the sub-sectors as a percentage of the GSDP of each category.

# In[ ]:


#creating a new dataframe with relevant values
df_gdp_percapita_1=mastergdp.loc[mastergdp['Item'].isin(['Per Capita GSDP (Rs.)']), ['Item','2014-15','State']]
df_gdp_percapita_1=df_gdp_percapita_1.set_index('State')
df_gdp_percapita_1.head()


# In[ ]:


# Cleaning Data
df_gdp_percapita_1 = df_gdp_percapita_1.drop('Item', axis=1)
df_gdp_percapita_1.rename(columns = {'2014-15':'GDP Per Capita in 2014-2015'}, inplace = True)
df_gdp_percapita_1 = df_gdp_percapita_1.sort_values(by='GDP Per Capita in 2014-2015', ascending = False)
df_gdp_percapita_1


# #### Categorise the states into four categories based on GDP per capita
# - C1, C2, C3, C4 - C1 would have the highest per capita GDP, C4 the lowest. The quantile values are (0.20,0.5, 0.85, 1)
# 

# In[ ]:


#Dividing it into 4 quantiles based on q value
df_gdp_percapita_1['Quantile_rank']=pd.qcut(df_gdp_percapita_1['GDP Per Capita in 2014-2015'],q=[0,0.20,0.5,0.85,1], labels=['C4','C3','C2','C1'])
df_gdp_percapita_2=df_gdp_percapita_1.drop('GDP Per Capita in 2014-2015', axis=1)
df_gdp_percapita_2


# In[ ]:


#Merging Quantile Values with main dataset
df_merged = pd.merge(df_gdp_percapita_2, mastergdp, on='State')
df_merged.head()


# In[ ]:


#Removing Total Values
df_merged = df_merged[df_merged['S.No.'] != 'Total']

df_merged.loc[:, ['S.No.']] = df_merged.loc[:, ['S.No.']].astype(float)

#Removing sub-sub sectors
df_merged.set_index('S.No.', inplace=True)
df_merged_1=df_merged.filter(like='.0', axis=0)
df_merged_1.drop([12.0,13.0,14.0,16.0,17.0], axis=0, inplace=True)

df_merged_1=df_merged_1.reset_index(drop=True)
df_merged_1.rename(columns = {'2014-15':'GDP per Sector'}, inplace = True)
df_merged_1.head()


# In[ ]:


#Dividing merged dataframe into 4 different dataframes for C1, C2, C3, C4
df_merged_c1=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C1']
df_merged_c2=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C2']
df_merged_c3=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C3']
df_merged_c4=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C4']
df_merged_c1.head(10)


# In[ ]:


#Using groupby to aggregate all values belonging to the same sector

def agg_by_sector(df, sector):
    df=df.groupby(['Item'])
    df_1=pd.DataFrame(df['GDP per Sector'].sum().sort_values(ascending = True))
    df_1.rename(columns = {'GDP per Sector':'GDP per Sector for %s' %sector}, inplace = True)
    return (df_1)
    print(df_1)    


# In[ ]:


df_c1=agg_by_sector(df_merged_c1, 'C1')
df_c1


# In[ ]:


#calculating percentage contribution of each sector

def percentage_contribution(df,sector):
    df['Percentage Contribution for %s'%sector] = df.iloc[0:-1, :].apply(lambda s: s*100 / df.iloc[-1,0])
    df=df.drop('Gross State Domestic Product', axis=0)
    df = df.sort_values(by='Percentage Contribution for %s'%sector, ascending = False)
    return df
    print(df)


# In[ ]:


df_c1=percentage_contribution(df_c1,'C1')
df_c1


# In[ ]:


#calculating cumulative sum 
def cumulative_sum(df,sector):
    df['Cumulative Sum of GDP Contribution for %s'%sector] = df["GDP per Sector for %s"%sector].cumsum()
    df['Cumulative Percentage of GDP Contribution for %s'%sector] = df["Percentage Contribution for %s"%sector].cumsum()
    return df
    print (df)


# In[ ]:


df_c1=cumulative_sum(df_c1,'C1')
df_c1


# In[ ]:


#selecting categories with ~80% contribution to GDP
C1_categories = df_c1.loc[(df_c1['Cumulative Percentage of GDP Contribution for C1']  < 82)]
C1_categories.iloc[:, [-1]]


# In[ ]:


#plotting a pie chart
def plot_pie(df,sector):
    plot_x = df.plot.pie(y='Percentage Contribution for %s'%sector, figsize=(7, 7))
    plot_x.legend_ = None
    plt.show()
    
plot_pie(df_c1,'C1')


# #### Repeating Same Steps for Other sectors (C2,C3,C4)

# In[ ]:


df_c2=agg_by_sector(df_merged_c2, 'C2')
df_c2=percentage_contribution(df_c2,'C2')
df_c2=cumulative_sum(df_c2,'C2')
df_c2


# In[ ]:


#selecting categories with ~80% contribution to GDP
C2_categories = df_c2.loc[(df_c2['Cumulative Percentage of GDP Contribution for C2']  < 82)]
C2_categories.iloc[:, [-1]]


# In[ ]:


plot_pie(df_c2,'C2')


# In[ ]:


df_c3=agg_by_sector(df_merged_c3, 'C3')
df_c3=percentage_contribution(df_c3,'C3')
df_c3=cumulative_sum(df_c3,'C3')
df_c3


# In[ ]:


#selecting categories with ~80% contribution to GDP
C3_categories = df_c3.loc[(df_c3['Cumulative Percentage of GDP Contribution for C3']  < 82)]
C3_categories.iloc[:, [-1]]


# In[ ]:


plot_pie(df_c3,'C3')


# In[ ]:


df_c4=agg_by_sector(df_merged_c4, 'C4')
df_c4=percentage_contribution(df_c4,'C4')
df_c4=cumulative_sum(df_c4,'C4')
df_c4


# In[ ]:


#selecting categories with ~80% contribution to GDP
C4_categories = df_c4.loc[(df_c4['Cumulative Percentage of GDP Contribution for C4']  < 82)]
C4_categories.iloc[:, [-1]]


# In[ ]:


plot_pie(df_c4,'C4')


# In[ ]:


#merging contribution for all categories
df_c1.reset_index(drop=False, inplace=True)
df_c2.reset_index(drop=False, inplace=True)
df_c3.reset_index(drop=False, inplace=True)
df_c4.reset_index(drop=False, inplace=True)


# In[ ]:


df_c4


# In[ ]:


df_merged_all = pd.merge(df_c1, df_c2, on='Item')
df_merged_all = pd.merge(df_merged_all, df_c3, on='Item')
df_merged_all = pd.merge(df_merged_all, df_c4, on='Item')
df_merged_all.set_index('Item', inplace=True)
df_merged_all_1=df_merged_all.filter(like='Percentage Contribution', axis=1)
df_merged_all_1


# In[ ]:


colors = ["#808080", "#00FA9A","#20B2AA", "#20B2AB"]
df_merged_all_1.loc[:,['Percentage Contribution for C1','Percentage Contribution for C2','Percentage Contribution for C3','Percentage Contribution for C4']].plot.barh(stacked=True, color=colors, figsize=(15,12))
plt.show()


# ## GDP and Education
# 1. Analyse if there is any correlation of GDP per capita with dropout rates in education (primary, upper primary and secondary) for the year 2014-2015 for the states. Choose an appropriate plot to conduct this analysis. 
# 2. Write the key insights you observe from this data:[](http://)
#     - Form at least one reasonable hypothesis for the observations from the data

# In[ ]:


#importing csv to dataframe
df_education_raw = pd.read_csv('/kaggle/input/rs_session243_au570_1.1.csv')
df_education_raw.head()


# In[ ]:


df_gdp_percapita_x = df_gdp_percapita_x.sort_values(by='State', ascending = True)
df_gdp_percapita_x = df_gdp_percapita_x.reset_index(drop=False)
df_gdp_percapita_x


# In[ ]:


df_dropout_rates = df_education_raw.copy()
df_dropout_rates= df_dropout_rates.set_index('Level of Education - State', drop=True)
df_dropout_rates


# In[ ]:


#filtering column only for 2014-2015
df_dropout_rates=df_dropout_rates.filter(like='2014-2015', axis=1)

#Dropping 2nd column as per TA instruction
df_dropout_rates=df_dropout_rates.drop(['Primary - 2014-2015.1','Senior Secondary - 2014-2015'], axis=1)
df_dropout_rates=df_dropout_rates.drop('All India', axis=0)


UT = pd.Series(['A & N Islands','Chandigarh','Dadra & Nagar Haveli','Daman & Diu','Delhi','Jammu and Kashmir','Lakshadweep','Puducherry'])
df_dropout_rates.drop(UT, axis=0, inplace=True)

#data clean up and preparation for analysis
df_dropout_rates= df_dropout_rates.reset_index(drop=False)
df_dropout_rates.rename(columns = {'Level of Education - State':'State','Primary - 2014-2015':'Primary_DropOut_Rate_14_15','Upper Primary - 2014-2015':'Upper_Primary_DropOut_Rate_14_15','Secondary - 2014-2015':'Secondary_DropOut_Rate_14_15'}, inplace = True)
df_dropout_rates.State = df_dropout_rates.State.str.replace('Chhatisgarh', 'Chhattisgarh')
df_dropout_rates.State = df_dropout_rates.State.str.replace('Uttrakhand', 'Uttarakhand')
df_dropout_rates


# In[ ]:


df_merged_dropout = pd.merge(df_dropout_rates, df_gdp_percapita_x, on='State')
df_merged_dropout = df_merged_dropout .set_index('State', drop=True)
df_merged_dropout = df_merged_dropout.sort_values(by='Per Capita GDP in 2014 15', ascending = True)
df_merged_dropout


# In[ ]:


# plotting scatter maps
f = plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
plt.title('Primary Drop out Rates')
plt.scatter(df_merged_dropout['Primary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])

# subplot 2
plt.subplot(2, 2, 2)
plt.title('Upper Primary Drop out Rates')
plt.scatter(df_merged_dropout['Upper_Primary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])

# subplot 3
plt.subplot(2, 2, 3)
plt.title('Secondary Drop out Rates')
plt.scatter(df_merged_dropout['Secondary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])


plt.show()


# In[ ]:


#calculating correlation
df_merged_dropout_correlation = df_merged_dropout.corr()
round(df_merged_dropout_correlation, 3)


# In[ ]:


# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(df_merged_dropout_correlation, cmap="YlGnBu", annot=True)
plt.show()


# **Key insights and hypothesis observed from this data**
# * It is observed from both scatter plot and heat map that all three drop-out rates are negatively correlated to per capita GDP contribution. Although, anomalies are present in all 3 correlations.
# * A possible reason for this could be that states contributing less to GDP have low resources to contribute to GDP and so they do not have good educational institutions which results in high drop out rates.
# * The other possibility could be that the people in states with low GDP per capita may also have low per capita income. This would result in people dropping out of schools and colleges to get employed in jobs or start a business.
# * Vice Versa, states with low drop out rates have employable skilled people which leads to high income per capita and high per capita GDP contribution

# In[ ]:




