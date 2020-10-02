#!/usr/bin/env python
# coding: utf-8

# Hi all ,
# 
# I will use this kernel as an opportunity to explore various tricks to present data in better way(either by graph or by meaningful tables) . I will try my level best to share my domain knowledge on Healthcare industry to keep you interested thourghout this notebook
# 
# We have enough data here to analyze and find interesting patterns to understand data from year 2014,2015 and 2016.
# Lets start our journey with BenefitsCostSharing and I will handle rest of the dataset in further sections of this kernel .
# This kernel will remain WIP until  I manage to answer all the questions asked in the overview section.
# 
# **1. BenefitsCostSharing.csv**     
# *Lets start our analysis with above dataset  to answer How do plan rates and benefits vary across states?*
# 
# 
# **2.BusinessRules.csv**
# 
# **3.Crosswalk2015.csv**
# 
# **4.Crosswalk2016.csv**
# 
# **5.Network.csv**
# 
# **6.PlanAttributes.csv**
# 
# **7.Rate.csv**
# 
# **8.ServiceArea.csv**
# 
# **9.database.sqlite**
# 
# **10.hashes.txt**
# 
# **11.raw**
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Lets gather all the required tools to tame this Big dataset  

# In[ ]:


import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing BenefitsCostSharing dataset**

# In[ ]:


CostShare_df = pd.read_csv("../input//BenefitsCostSharing.csv")


# Lets take a look at how this dataset looks

# In[ ]:


CostShare_df.head(n=10)


# Lets find out which columns do not have null values and lets fill out empty records from our dataset 

# In[ ]:


# Fill empty and NaNs values with NaN
CostShare_df = CostShare_df.fillna(np.nan)

# Check for Null values
CostShare_df.isnull().sum()


# Lets check how many records we have in our dataset. 
# I have chosen BenefitName field here since it contains no null values and its name is self explnatory that it is related to benefit plan sold in USA healthcare

# In[ ]:


print ('Total records in file:%d' %CostShare_df.BenefitName.count())
print ('Unique benefits pesent in the file:%d' %CostShare_df.BenefitName.nunique())


# So we have total 861 type of different plans sold in USA in the year 2014 to 2016

# In[ ]:


### lets Summarize data
# Summary and statistics
CostShare_df.describe()


# Data produced above will not make sense at all for few features here.
# 
# But features like  Minimum Stay point out that on an average minimum stay of 47 days is covered through benefit plans. 
# 
# LimitQty does give out significant information but it is too early to make comment on it . since we still dont know how many plans are indvidual or family oriented and how premium dollar amount paid varies for these plans, so we may have to segragate the data to have better intuition on this feature.
# 
# Lets analyze all Features in one go using below enumerate function. 
# We can take help of these stattistics to proceed.

# In[ ]:


v_features = CostShare_df.ix[:,0:32].columns
for i, cn in enumerate(CostShare_df[v_features]):
    print(i,cn)
    print(CostShare_df[cn].describe())
    print("-"*40)


# Lets Analyze which benefit type topping the chart in a given business year 

# In[ ]:


CostShare_df[["BusinessYear","BenefitName"]].groupby('BusinessYear').describe()


# Lets Analyze benefit penetration statewise . Below code will help us to get the intermediate variables required to plot the data

# In[ ]:


CostShare_df[["StateCode","BenefitName"]].groupby('StateCode').count().sort_values("BenefitName")
Unique_State = CostShare_df.StateCode.unique()
benefitarray = []

for state in Unique_State:
    state_benefit =  len(CostShare_df[CostShare_df["StateCode"] == state])    
    benefitarray.append(state_benefit)   


# In[ ]:



f, ax = plt.subplots(figsize=(15, 15)) 
ax.set_yticklabels(Unique_State, rotation='horizontal', fontsize='large')
g = sns.barplot(y = Unique_State,x=benefitarray)
plt.show()

# Set number of ticks for x-axis
"""
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(Unique_State, rotation='vertical', fontsize='small')
plt.show()
"""


# Grapth produced above may become diffficult to read since it is not in sorted order of benefit plans sold statewise. Lets add few lines of code again to get desired graphical presentation to read benefitplan sold in descending order statewise.

# In[ ]:



df = pd.DataFrame(
    {'state': Unique_State,
     'Count' : benefitarray
     })

df = df.sort_values("Count", ascending=False).reset_index(drop=True)

f, ax = plt.subplots(figsize=(15, 15)) 
ax.set_yticklabels(df.state, rotation='horizontal', fontsize='large')
g = sns.barplot(y = df.state, x= df.Count)
plt.show()


# Lets use Choropleth plot to visualize above data on USA map. 
# Zoom in or Zoom out on the plot and put your cursor on the state to see the information about the state. 

# In[ ]:


data = dict(type = 'choropleth',
           locations = df['state'],
           locationmode = 'USA-states',
           colorscale = 'YIOrRed',
            text = df['state'],
            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),
           z = df['Count'],
           colorbar = {'title':'No of Benefit plans'})

layout = dict(title = 'Benefit plan spread across state',
         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 

choromap2 = go.Figure(data = [data],layout=layout)
iplot(choromap2)


# Now we are in a much better condition to draw conclusion that state WI is larger consumer of healthcare services. 
# We may have to consider the current population of each state and above stats(graph)to decide our strategy to cover more people accordingly.
# 
# Lets have a one more look at which benefit plan is consumed more by the population of the respective state.

# In[ ]:


CostShare_df[["StateCode","BenefitName"]].groupby('StateCode').describe()


# 
# Lets understanf what is Copay and Coinsurance here along with In network and Out of network since we will be dealing with these features.
# 
# Copay            :- Copayment is a payment defined in an insurance policy and paid by an insured person each time a medical service is accessed.
# 
# **Additional Information on Deductible to understand Coinsurance better**
#  
# Deductible :-   suppose you are in a calender year 2017 , and your policy says you must spend dollar 5000 first to start availing the benefits from you policy . Lets assume you spent dollar5000 by month of March. now Coinsurance plays a major role after that . (Note:- Coinsurance is expressed in percentage)
#  
# Coinsurance :-  After you have spent your deductible and again you are in need to avail medical service this time you visit doctor again , you pay your fixed copay amount but rest other expenses will be shared by your Payer (company from which you purchased your plan). suppose you have coninsurance of 20%  and your total cost is 100$ , so you will spend $20 and remaining 80% will i,e. dollar 80 will paid by payer. 
# 
# **One more additional concept to make you understand how long we should follow this Coinsurance**
#  
# Out of pocket limit :- We have this limit to make your Helathcare payer take 100% 
# resonsibility of of your medical services. Suppose you keep on paying coinsurance till some month for the calender year 2017 . Suppose by end month October you spent  dollar 10000 (your Deductible spent till march from above example + Copay paid till date from jan or start of calender year + Coinsurance you started after march i.e. after exhausting your deductible till march ) 
# 
# Out of poclet $10000   < (Deductible + Copay + Coinsurance )
# If this condition gets satisfied your Healthcare (Payer) pays 100% 
# of your medical expense after that for that calender year .
# 
# All these benefits resets again back to what they were in previous year for new calender year and you start paying your deductibles and coinsurance again :)  Unless you have renewal plans in effect.
# 
# In Network    :-  Avaialing medical services from the hospitals and  labs who are associated with your healthcare Payers. 
# 
# Our Network :- Avaialing medical services from the hospitals and  labs who are not associated with your healthcare Payers.
# 
# benefit plan costs are mostly higher for out of network services than benefit plan cost of in netwrok services.
# 

# Small help from google to know what is Tier based Copay and Coinsurance. read below lines on different tiers available.
# 
# Tier 1 always carries the lowest copay and typically applies only to generic drugs. Tier 2 is often for "preferred" brand-name drugs. Tier 3 is often for "non-preferred" brand names. Tier 4 is for usually for "specialty" drugs, meaning very expensive ones and those used to treat rare conditions.

# In[ ]:



#Coinsurance
print('Coinsurance details')
print(CostShare_df.CoinsInnTier1.unique())
print('*'*50)
print(CostShare_df.CoinsInnTier2.unique())
print('*'*50)
print(CostShare_df.CoinsOutofNet.unique())
print('_'*50)
print('_'*50)

"""Copay
print('Copay details')
print(CostShare_df.CopayInnTier1.unique())
print('*'*50)
print(CostShare_df.CopayInnTier2.unique())
print('*'*50)
print(CostShare_df.CopayOutofNet.unique())
print('_'*50)
print('_'*50)
"""


# The problem with above output is , we are not able to understand the spread of the data among the population 
# and we need to present this data either by gaussian distribution grapths or frequency table. We may need to modify this data a bit to get what we want.
# 
# Lets write some code to get rid of the texts written after Coinsurance percentage like 'Coinsurance after deductible'
# As I have already explained when Coinsurance is used , this extra text is not required now. Simillrly look for other text which can be removed so that we will left with numbers for better analysis.

# In[ ]:



CoinsInnTier1 = []
CoinsInnTier1_real = np.asarray(CostShare_df.CoinsInnTier1)
            
for i, cn in enumerate(CoinsInnTier1_real):
       if (str(cn) == 'nan' or str(cn) == '$0' or str(cn) == 'Not Applicable') :
             continue     
       else :
             if  cn.replace("%","").strip().split(' ')[0] != 'No' :   
                 CoinsInnTier1.append(cn.replace("%","").strip().split(' ')[0])
   


# Differet ways to get Top frequent Coinsurance 
# 1) using Value_counts method
# 2) write complex function

# In[ ]:


# 1) use Value_counts method
CoinsInnTier1
CoinsInnTier1 = pd.to_numeric(CoinsInnTier1, errors='coerce')
Codf = pd.DataFrame(
    {'Coinsurance1': CoinsInnTier1
     })
Codf['Coinsurance1'].value_counts().head(5)


# In[ ]:


# 2) write complex function

Coinsarray = []

Unique_Coinsurance = Codf.Coinsurance1.unique()
for Coinsurance in Unique_Coinsurance:
    Freq_Coinsurance =  len(Codf[Codf["Coinsurance1"] == Coinsurance])    
    Coinsarray.append(Freq_Coinsurance) 


# Below piece of code will show that Coinsurance with their frequency  We can use below codes for rest of Copays and Coinsurance.

# In[ ]:



Coins_df = pd.DataFrame(
    {'Coinsurance': Unique_Coinsurance,
     'Coinsfrequency' : Coinsarray
     })

Coins_df = Coins_df.sort_values("Coinsfrequency", ascending=False).reset_index(drop=True)

Coins_df


# Lets visualize the Coinsurance distribution using Distplot

# In[ ]:


sns.distplot(Codf['Coinsurance1'],kde=False,bins=15)


# This kernel will remain in WIP status until I answer all the questions asked in the overview section.
