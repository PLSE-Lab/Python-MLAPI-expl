#!/usr/bin/env python
# coding: utf-8

# We are looking at the 2012 Election Pooling data.This was a contest between Obama ans Romney.We know in the Hindsight that the election was won by Obama.Lets look at the data and see if there were signs of Obama victory in this pre poll survey.This Kernel is work in process.If you like my work please do vote.

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


# **Importing Python Modules**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import Series,DataFrame
import warnings
import seaborn as sns
warnings.filterwarnings('ignore') 


# **Obama:** was the first African American to become the Presedent of United States.He belongs to the Democratic Party.He took over the Presidency from George W Bush in the year 2008.Obama Steered US economy through the 2008 financial crisis.He will be remembered for his great Oratory Skills.His sense of humor and Human touch made him popular US president.

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/obama-yes-we-can/We_Can.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# **Importing the dataset**

# In[ ]:


df=pd.read_csv('../input/2012-election-obama-vs-romney/2012-general-election-romney-vs-obama.csv')
df.head()


# In[ ]:


df.info()


# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# **What are modes of Survey?**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Mode'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Mode of Survey')
ax[0].set_ylabel('Count')
sns.countplot('Mode',data=df,ax=ax[1],order=df['Mode'].value_counts().index)
ax[1].set_title('Count of Mode of Survey')
plt.show()


# There are 5 modes of conducting the surevey Live Phone,Automated Phone,Internet Mode, IVR/Live Phone and IVR/Online

# **What Types of Survey?**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Partisan'].value_counts().plot.pie(explode=[0.05,0.05,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Types of Survey')
ax[0].set_ylabel('Count')
sns.countplot('Partisan',data=df,ax=ax[1],order=df['Partisan'].value_counts().index)
ax[1].set_title('Survey Type')
plt.show()


# So most surveys are conducted by Non Partisan people.Very few of the survey are sponsored by political parties

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Affiliation'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Afflication of Survey')
ax[0].set_ylabel('Count')
sns.countplot('Affiliation',data=df,ax=ax[1],order=df['Affiliation'].value_counts().index)
ax[1].set_title('Count of Survey Affliction')
plt.show()


# Maximum Surveys have no afflications. 15.8% affliction with Democrats and 1.5% affliction with Repiblicans.

# **Population Consists of whom?**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Population'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Population Breakup')
ax[0].set_ylabel('Count')
sns.countplot('Population',data=df,ax=ax[1],order=df['Population'].value_counts().index)
ax[1].set_title('Population Count')
plt.show()


# Four types of people in the data are Likely Voters,Registered Voters,Adults & Likely Voters-Rebublican

# **Factor Plot of Afflication based on Population**

# In[ ]:


#sns.factorplot('Population',data=df,hue='Population')
#plt.ioff()


# **How Obama and Romney pitch against each other?**

# In[ ]:


df.mean()


# In[ ]:


avg=pd.DataFrame(df.mean())
avg


# In[ ]:


avg.drop('Number of Observations',axis=0,inplace=True)


# In[ ]:


std=pd.DataFrame(df.std())
std.drop('Number of Observations',axis=0,inplace=True)


# In[ ]:


avg.plot(yerr=std,kind='bar',legend=False)
plt.ioff()


# We have considered the Average of percentage people in favor of candidates.But we did consider the inpact of standard deviation.With standard deviation we can see that Obama and Romney overlap with eachother.

# In[ ]:


poll_avg=pd.concat([avg,std],axis=1)
poll_avg


# In[ ]:


poll_avg.columns=['Average','STD']
poll_avg


# **How Romney Fares Against Obama as time went one ?**

# In[ ]:


df.plot(x='End Date',y=['Obama','Romney','Undecided'],linestyle='',marker='o')
plt.xlabel('Time')
plt.ylabel('Popularity')
plt.ioff()


# Time is increasing from Right to Left.We can see that the percentage of voters in favor of Romney started increasing and the gap closed as the campaign continued.Undecided voter finally decide the outcome of the election.

# In[ ]:


from datetime import datetime


# In[ ]:


df['Difference']=(df.Obama-df.Romney)
df.head()


# In[ ]:


df.columns


# In[ ]:


df=df.groupby(['Start Date'],as_index=False).mean()
df.head()


# In[ ]:


df.plot('Start Date','Difference',figsize=(12,4),marker='o',linestyle='-',color='purple')
plt.ioff()


# If the Difference is positive it means that Obama is leading and if the Difference is Negative it means Romney is leading.In the initial Phase Obama had more lead over RomneyWe can sse as we move from Mar 2019 to Oct-2012 the gap between Obama and Romney reduced.

# In[ ]:


df.loc[df['Difference'].idxmin()]


# Using the above method we can find out the day on which the Difference was more in favor of Romney.By Checking news of the day we can find out what was the reason for sudden Popularity increase of Mit Romney.

# In[ ]:


row_in=0
xlimit=[]

for date in df['Start Date']:
    if date[0:7]=='2011-10':
        xlimit.append(row_in)
        row_in+=1
    else:
        row_in+=1
print(min(xlimit))
print(max(xlimit))      


# In[ ]:


df.plot('Start Date','Difference',figsize=(12,4),marker='o',linestyle='-',color='purple',xlim=(96,110))
plt.ioff()


# In[ ]:


df.plot('Start Date','Difference',figsize=(12,4),marker='o',linestyle='-',color='purple',xlim=(96,110))
# Oct 3rd
plt.axvline(x=96+4,linewidth=4,color='grey')
#Oct 11th
plt.axvline(x=96+8,linewidth=4,color='grey')
#Oct 22nd           
plt.axvline(x=96+12,linewidth=4,color='grey')
plt.ioff()


# **Lets Analyse Donar Data**

# In[ ]:


dd=pd.read_csv('../input/2012-election-obama-vs-romney/Election_Donor_Data.csv')
dd.head()


# In[ ]:


dd.info()


# **Summary of Data**

# In[ ]:


print('Rows     :',dd.shape[0])
print('Columns  :',dd.shape[1])
print('\nFeatures :\n     :',dd.columns.tolist())
print('\nMissing values    :',dd.isnull().values.sum())
print('\nUnique values :  \n',dd.nunique())


# In[ ]:


dd['contb_receipt_amt'].value_counts()


# Most common Donation is 100$

# In[ ]:


dd_mean=dd['contb_receipt_amt'].mean()
dd_std=dd['contb_receipt_amt'].std()
print('The average donation was %.2f with a std %.2f'%(dd_mean,dd_std))


# So the average donation is 298 $ but the standard deviation is very high.This means there are people who have made big contribution.

# In[ ]:


top_donor=dd['contb_receipt_amt'].copy()
top_donor.sort_values


# There are some negative value sowe have to remove them 

# In[ ]:


top_donor=top_donor[top_donor>0]
top_donor.sort_values(ascending=False).head()


# In[ ]:


com_don=top_donor[top_donor<2500]
com_don.hist(bins=100,range=[0, 2500])
plt.ioff()


# So most donations are less than 1000 $

# In[ ]:


com_don=top_donor[top_donor<2500]
com_don.hist(bins=100,range=[0, 500])
plt.ioff()


# 100 $ is the most contributed amount towards donation

# In[ ]:


dd.columns


# In[ ]:


dd.cand_nm.unique()


# **Dictionary of Afflication**

# In[ ]:


party_map={'Bachmann, Michelle':'Republican', 'Romney, Mitt':'Republican', 'Obama, Barack':'Democratic',
       "Roemer, Charles E. 'Buddy' III":'Republican', 'Pawlenty, Timothy':'Republican',
       'Johnson, Gary Earl':'Republican', 'Paul, Ron':'Republican', 'Santorum, Rick':'Republican',
       'Cain, Herman':'Republican', 'Gingrich, Newt':'Republican', 'McCotter, Thaddeus G':'Republican',
       'Huntsman, Jon':'Republican', 'Perry, Rick':'Republican'}


# In[ ]:


dd['Party']=dd.cand_nm.map(party_map)


# In[ ]:


dd.head()


# In[ ]:


dd=dd[dd.contb_receipt_amt>0]


# In[ ]:


dd.head()


# In[ ]:


dd.groupby('cand_nm')['contb_receipt_amt'].count()


# In[ ]:


dd.groupby('cand_nm')['contb_receipt_amt'].sum()


# In[ ]:


dd.groupby('cand_nm')['contb_receipt_amt'].mean()


# We can see that highest number of people contributed to Obama (589127 Nos)
# 
# Obama received the highest contribution $ 1.358774e+08
# 
# Perry Rich has a mean contribution of $ 1597.746007 which is very high compared to Obama who got  230.641996

# In[ ]:


cand_amount=dd.groupby('cand_nm')['contb_receipt_amt'].sum()

i=0

for don in cand_amount:
    print('The candidate %s raise %.0f dollars'%(cand_amount.index[i],don))
    print('\n')
    i+=1


# **Which Candidate got More Contribution?**

# In[ ]:


cand_amount.sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Name of Candidate')
plt.ylabel('Money is Dollars')
plt.ioff()


# **Which Party got more contribution?**

# In[ ]:


dd.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')
plt.xlabel('Name of Party')
plt.ylabel('Money is Dollars')
plt.ioff()


# We can see that even though Obama got highest contribution amoung candidates Republican party got more contribution that Democratic party.This could be because Republic Party has many candidates for the Presidential Elections 

# In[ ]:


occupation_dd=dd.pivot_table('contb_receipt_amt',index='contbr_occupation',columns='Party',aggfunc='sum')
occupation_dd.head()


# In[ ]:


occupation_dd.shape


# In[ ]:


occupation_dd=occupation_dd[occupation_dd.sum(1)>1000000]


# In[ ]:


occupation_dd.shape


# In[ ]:


occupation_dd.plot(kind='bar',figsize=(12,8))
plt.ioff()


# Retired peope,homemaker and Attorney make Highest contribution to Election Fund

# In[ ]:


occupation_dd.plot(kind='barh',figsize=(10,12),cmap='seismic')
plt.ioff()


# In[ ]:


occupation_dd.drop(["INFORMATION REQUESTED PER BEST EFFORTS","INFORMATION REQUESTED"],axis=0,inplace=True)


# In[ ]:


occupation_dd.loc['CEO']=occupation_dd.loc['CEO']+occupation_dd.loc['C.E.O.']
occupation_dd.drop('C.E.O.',inplace=True)


# In[ ]:


occupation_dd.plot(kind='barh',figsize=(10,12),cmap='seismic')
plt.ioff()


# Retired peope,homemaker and Attorney make Highest contribution to Election Fund
