#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame as df 
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt 
import warnings
warnings.simplefilter('ignore')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/Serial Killers Data.csv",encoding = "ISO-8859-1")


# In[ ]:


data=data.loc[(data['Century']=='1900')|(data['Century']=='2000')]
# I wanted to make sure that I was only looking at killers from the 20th and 21st century and cut the data as such
#list(data)

#I noted there was a higher number of data available for US based killers and decided to base my path of inquiry
# where there would be a shared cultural lexicon.

justUS= data.loc[data['country']=='US'] # selects just those noted as being in the US exclusively
TwoKills=justUS.loc[justUS.NumVics<=2] # creates a dataframe of 2 or less victims
threePlusKills=justUS.loc[justUS.NumVics>2] # creates a dataframe of 2 or more victims


# In[ ]:


# Pie charts of how partners are being used between the two groups
part1=TwoKills['Partner'].value_counts().tolist()
part2=threePlusKills['Partner'].value_counts().tolist()
labels='No Partner','Partner'

plt.figure(1)

plt.subplot(211)
plt.title('Two Kills or Less-Partner Ratio')
plt.pie(part1,labels=labels)

plt.subplot(212)
plt.title('Three or More Kills-Partner Ratio')
plt.pie(part2,labels=labels)


# In[ ]:


import seaborn as sns

sns.distplot(TwoKills['YearsBetweenFirstandLast'].dropna())
plt.title('Years Between First and Last Kill for 2 or Less Murders')
plt.ylabel('Proportion')


# In[ ]:


sns.distplot(threePlusKills['YearsBetweenFirstandLast'].dropna())
plt.title('Years between First and Last Kill for 3 or More Murders')
plt.ylabel('Proportion')

# note a 10% increase between 1st and last kills for the 0 to 1 year category


# In[ ]:


# A key thing to note here is that the total population of 3 or more kills data is much higher 
dat=threePlusKills['Race'].value_counts().tolist()
threePlusKills['Race'].value_counts()
labels='White','Black','Hispanic','Asian','Native American'

plt.pie(dat,labels=labels)
plt.title('Race of Killers for Three or More Kills Group')


# In[ ]:


# The ratio of White and Black offenders is more equal in the two or less bracket
dat2=TwoKills['Race'].value_counts().tolist()
labels='White','Black','Hispanic','Native American','Asian'
plt.pie(dat2,labels=labels)
plt.title('Race of Killers for Two or Less Kills Group')


# In[ ]:


print('Summary Statistics of Age of first Kill for Two Kills or Less Group')
print(TwoKills['Age1stKill'].dropna().describe())
print(TwoKills['Age1stKill'].isnull().sum())
print('_______________________')
print('Summary Statistics of Age of first Kill for more than 2 kills Group')
print(threePlusKills['Age1stKill'].dropna().describe())
print(threePlusKills['Age1stKill'].isnull().sum())

#median age of first kill appears to be comparative


# We want to confirm that the mean age of first kills is the same between groups. Below will be a visual
# inspection initially. I included the number of null values to indicate that relative to the sample size
# there are few null values.

# In[ ]:





# In[ ]:


sns.distplot(TwoKills['Age1stKill'].dropna())
plt.title('Age of First Kill for Two Kills or less group')


# In[ ]:


sns.distplot(threePlusKills['Age1stKill'].dropna())
plt.title('Age of First Kill for Three Kills or More group')


# The distributions are of a similar shape. So I performed a t-test utilizing the summary statistics
# with the assumption that the variances are not equal.

# In[ ]:


stats.ttest_ind_from_stats(26.507804,8.649313,961,26.996287,8.495126,1616,equal_var=False)


# Given the data, we would state that the populations appear to begin killing at approximately the same
# times.

# In[ ]:


# Breaking the groups down by motive. There's a better way to do this. I just wanted to keep things
#absolutely clear
twoKillsCode1=TwoKills.loc[(TwoKills['Code']>=1) & (TwoKills['Code']<2)]
twoKillsCode2=TwoKills.loc[(TwoKills['Code']>=2) & (TwoKills['Code']<3)]
twoKillsCode3=TwoKills.loc[(TwoKills['Code']>=3) & (TwoKills['Code']<4)]
twoKillsCode4=TwoKills.loc[(TwoKills['Code']>=4) & (TwoKills['Code']<5)]
twoKillsCode5=TwoKills.loc[(TwoKills['Code']>=5) & (TwoKills['Code']<6)]
twoKillsCode6=TwoKills.loc[(TwoKills['Code']>=6) & (TwoKills['Code']<7)]
twoKillsCode7=TwoKills.loc[(TwoKills['Code']>=7) & (TwoKills['Code']<8)]
twoKillsCode8=TwoKills.loc[(TwoKills['Code']>=8) & (TwoKills['Code']<9)]
twoKillsCode9=TwoKills.loc[(TwoKills['Code']>=9) & (TwoKills['Code']<10)]
twoKillsCode10=TwoKills.loc[(TwoKills['Code']>=10) & (TwoKills['Code']<11)]
twoKillsCode11=TwoKills.loc[(TwoKills['Code']>=11) & (TwoKills['Code']<12)]

threePlusKillsCode1=threePlusKills.loc[(threePlusKills['Code']>=1) & (threePlusKills['Code']<2)]
threePlusKillsCode2=threePlusKills.loc[(threePlusKills['Code']>=2) & (threePlusKills['Code']<3)]
threePlusKillsCode3=threePlusKills.loc[(threePlusKills['Code']>=3) & (threePlusKills['Code']<4)]
threePlusKillsCode4=threePlusKills.loc[(threePlusKills['Code']>=4) & (threePlusKills['Code']<5)]
threePlusKillsCode5=threePlusKills.loc[(threePlusKills['Code']>=5) & (threePlusKills['Code']<6)]
threePlusKillsCode6=threePlusKills.loc[(threePlusKills['Code']>=6) & (threePlusKills['Code']<7)]
threePlusKillsCode7=threePlusKills.loc[(threePlusKills['Code']>=7) & (threePlusKills['Code']<8)]
threePlusKillsCode8=threePlusKills.loc[(threePlusKills['Code']>=8) & (threePlusKills['Code']<9)]
threePlusKillsCode9=threePlusKills.loc[(threePlusKills['Code']>=9) & (threePlusKills['Code']<10)]
threePlusKillsCode10=threePlusKills.loc[(threePlusKills['Code']>=10) & (threePlusKills['Code']<11)]
threePlusKillsCode11=threePlusKills.loc[(threePlusKills['Code']>=11) & (threePlusKills['Code']<12)]


# In[ ]:


for i in range(1,12):
    
    threeKills=print(eval("threePlusKillsCode"+ str(i))['Code'].value_counts().sum())
N=11
values=(453,11,648,224,12,13,27,117,10,0,92)
ind=np.arange(N)
width=0.35
label=('Financial Gain','Attention','Enjoyment','Anger','Mental Illness','Cult','Avoid Arrest', 'Gang Activity',
      'Convenience','Wild West Outlaw','Multiple Motivations')

plt.bar(ind+width/2,values,color='r',align='center')
plt.xticks(ind,label,rotation='vertical')
plt.title('Motivations for 3 or More Kills')
plt.xlabel('Motivation Type')
plt.ylabel('Frequency')
print(sum(values))


# In[ ]:


for i in range(1,12):
    print(eval("twoKillsCode"+ str(i))['Code'].value_counts().sum())

N=11
values=(272,3,236,185,4,8,13,46,3,0,135)
ind=np.arange(N)
width=0.35
label=('Financial Gain','Attention','Enjoyment','Anger','Mental Illness','Cult','Avoid Arrest', 'Gang Activity',
      'Convenience','Wild West Outlaw','Multiple Motivations')

plt.bar(ind,values,color='g',align='center')
plt.xticks(ind+width/2,label,rotation='vertical')
plt.title('Motivations for 2 or Less Kills')
plt.xlabel('Motivation Type')
plt.ylabel('Frequency')
print(sum(values))


# I'll list the top 3 categories by Motive here
# Financial gain motivations are approximately the same between the two groups: approximately 30%. 
# 
# A difference is noted in the Enjoyment category: 2 kills or less-approximately 26% and 3 or more accounts for approximately 40% of motivations
# 
# Anger= 2 kills or less-20%, 3 kills or more-14%

# In[ ]:


#bifurcate population by education level. Arbitrarily choosing high school graduation
HS3= threePlusKills.loc[threePlusKills['Educ']>=12]
notHS3 =threePlusKills.loc[threePlusKills['Educ']<12]


# In[ ]:


HS3['NumVics'].hist(bins=15)
plt.title("Distribution of Kills for 3+ kills group  that have 12 or more years of education")
plt.xlabel('Number of kills')
plt.ylabel('Frequency')


# In[ ]:


notHS3['NumVics'].hist(bins=15,color='g')
plt.title("Distribution of Kills for 3 or more kills group that have less than 12 years of education")
plt.xlabel('Number of kills')
plt.ylabel('Frequency')


# I chose to stick with the 3 or more kills group as there is a higher number of data points for this. There appears to be no difference between 12 years of education or more and less than 12 years of education in terms of distribution of people killed.

# In[ ]:





# In[ ]:


HS3=HS3.convert_objects(convert_numeric=True)
notHS3=notHS3.convert_objects(convert_numeric=True)
HS3= HS3.loc[HS3['DecadeStarted']>1950]# breaking it down into more recent timeframes
notHS3= notHS3.loc[notHS3['DecadeStarted']>1950]

diffHS3=HS3['AgeLastKill']-HS3['Age1stKill']


# In[ ]:


diffHS3.hist(bins=15)
plt.title('Length of time between first and Last Kill for 3+ group, 12+ years of education')
plt.xlabel('Number of years')
plt.ylabel('Frequency')


# In[ ]:


diffNotHS3=notHS3['AgeLastKill']-notHS3['Age1stKill']
diffNotHS3.hist(bins=15)
plt.title('Length of time between first and Last Kill for 3+ group, Less than 12 years of education')
plt.xlabel('Number of years')
plt.ylabel('Frequency')


# In[ ]:


HS3['MethodDescription'].value_counts()/HS3['MethodDescription'].count() *100


# In[ ]:


notHS3['MethodDescription'].value_counts()/notHS3['MethodDescription'].count() *100


# The only difference of note between methods used between 12+ years of education and  less than 12 years of education groupings is a higher rate of poisonings for the higher educated group (6.73% vs 1.92%)

# In[ ]:


HS3['TypeofKiller'].value_counts()/HS3['TypeofKiller'].count() * 100


# In[ ]:


notHS3['TypeofKiller'].value_counts()/notHS3['TypeofKiller'].count() * 100


# From the data, it would appear those with less education are more likely to be involved in a team when committing 3 or more murders, but there's a ton of caveats to this statement

# In[ ]:





# I had a thought to breakdown the data by comparing overall number of events vs. number of purely looking at number of kills.  Currently, you have 67 categories of types of serial killers. While this might have some use for post-classification, I might suggest simplifying the categories as your data stream is potentially too stratified

# In[ ]:


# Just a quick breakdown of the categorical data into the two events or two kills data vs everything else
# will refine later
TwoKorE='Serial-Two'
teamTwo='Serial-Team-Two'
double='Doublemurder'
lst=df(data['TypeofKiller'],dtype='str')
#lst1 =[lambda y: str(y) for y in lst ]
x=len(TwoKorE)

twoSingle=[]
twoTeam=[]
justSerial=[]
for value in lst['TypeofKiller'].dropna():
    if (value[0:x]==TwoKorE) | (value[0:len(double)]==double) :
        twoSingle.append(1) #two kills or events or double murders
    elif value[0:len(teamTwo)]== teamTwo :
        twoTeam.append(2) # two kills or events but with a team
    else:
        justSerial.append(3) # everything else
                
            
            
        #if value[val]==TwoKorE:
                #twoMurdNote.append(1) #looking for two kills and two events


# In[ ]:


values=(1217,174,2529) #1217,174,2529
N=3
values=(1217,174,2529)
ind=np.arange(N)
width=0.35
label=('Two Kills or Events-Alone','Two Kills or Events-Team', 'Greater than 2 Kills or Spree')

plt.bar(ind,values,color='r',align='center')
plt.xticks(ind+width/2,label,rotation='vertical')
plt.title('Two Kills or Events breakdown')
#plt.xlabel('Category')
plt.ylabel('Frequency')


# In[ ]:


#breaking down the data to perform intragroup comparisons for anyone playing at home
TwoKorE='Serial-Two'
teamTwo='Serial-Team-Two'
double='Doublemurder'
lst=df(data['TypeofKiller'],dtype='str')
#lst1 =[lambda y: str(y) for y in lst ]
x=len(TwoKorE)
data['TypeofKiller']=data['TypeofKiller'].apply(lambda x: str(x))
twoSingle=df(index=data.index,columns=data.columns)
twoTeam=df(index=data.index,columns=data.columns)
justSerial=df(index=data.index,columns=data.columns)
for index,row in data.iterrows():
    if (row['TypeofKiller'][0:x]==TwoKorE)|( row['TypeofKiller'][0:len(double)]==double):
        twoSingle.loc[index]=row
    elif row['TypeofKiller'][0:len(teamTwo)]==teamTwo:
        twoTeam.loc[index]=row
    else:
        justSerial.loc[index]=row


# In[ ]:


#removing the NA values in the type of killers column
twoSingle=twoSingle.loc[twoSingle['TypeofKiller'].isnull()==False]
twoTeam=twoTeam.loc[twoTeam['TypeofKiller'].isnull()==False]
justSerial=justSerial.loc[justSerial['TypeofKiller'].isnull()==False]


# In[ ]:


# Since we have more than 1200 data points per variable in this set
for column in twoSingle:
    if (twoSingle[column].isnull().sum()/1217 * 100) > 50:
        del twoSingle[column]

twoSingle.info()


# In[ ]:


for column in twoTeam:
    if (twoTeam[column].isnull().sum()/174 *100)>40:
        del twoTeam[column]


# In[ ]:


for column in justSerial:
    if (justSerial[column].isnull().sum()/len(justSerial) *100)>50:
        del justSerial[column]


# Removed categories that had many NA values for now. This reduced the number of categories for comparisons to 95 categories.

# In[ ]:


twoSingle['Age1stKill'].mean()


# In[ ]:


sns.distplot(twoSingle['Age1stKill'].dropna())


# In[ ]:


sns.distplot(twoTeam['Age1stKill'].dropna())


# In[ ]:


sns.distplot(justSerial['Age1stKill'].dropna())


# In[ ]:


twoTeam['WhiteMale20s'].dropna().value_counts()


# In[ ]:


twoSingle['WhiteMale20s'].dropna().value_counts()


# In[ ]:


plt.scatter(twoSingle['Race'],twoSingle['AgeLastKill'])


# In[ ]:


highNum=twoSingle.loc[twoSingle['NumVics']>4]


# In[ ]:


highNum['Code'].value_counts()
 


# In[ ]:


highNumCode1=highNum.loc[(highNum['Code']>=1) & (highNum['Code']<2)]
highNumCode2=highNum.loc[(highNum['Code']>=2) & (highNum['Code']<3)]
highNumCode3=highNum.loc[(highNum['Code']>=3) & (highNum['Code']<4)]
highNumCode4=highNum.loc[(highNum['Code']>=4) & (highNum['Code']<5)]
highNumCode5=highNum.loc[(highNum['Code']>=5) & (highNum['Code']<6)]
highNumCode6=highNum.loc[(highNum['Code']>=6) & (highNum['Code']<7)]
highNumCode7=highNum.loc[(highNum['Code']>=7) & (highNum['Code']<8)]
highNumCode8=highNum.loc[(highNum['Code']>=8) & (highNum['Code']<9)]
highNumCode9=highNum.loc[(highNum['Code']>=9) & (highNum['Code']<10)]
highNumCode10=highNum.loc[(highNum['Code']>=10) & (highNum['Code']<11)]
highNumCode11=highNum.loc[(highNum['Code']>=11) & (highNum['Code']<12)]

for i in range(1,12):
    print(eval("highNumCode"+ str(i))['Code'].value_counts().sum())

N=11
values=(5,0,2,19,2,0,1,0,0,0,4)
ind=np.arange(N)
width=0.35
label=('Financial Gain','Attention','Enjoyment','Anger','Mental Illness','Cult','Avoid Arrest', 'Gang Activity',
      'Convenience','Wild West Outlaw','Multiple Motivations')

plt.bar(ind,values,color='r',align='center')
plt.xticks(ind+width/2,label,rotation='vertical')
plt.title('Motivations for 2 events and Greater than 4 Kills')
plt.xlabel('Motivation Type')
plt.ylabel('Frequency')


# In[ ]:


lowNum=twoSingle.loc[twoSingle['NumVics']<=4]


# In[ ]:


lowNumCode1=lowNum.loc[(lowNum['Code']>=1) & (lowNum['Code']<2)]
lowNumCode2=lowNum.loc[(lowNum['Code']>=2) & (lowNum['Code']<3)]
lowNumCode3=lowNum.loc[(lowNum['Code']>=3) & (lowNum['Code']<4)]
lowNumCode4=lowNum.loc[(lowNum['Code']>=4) & (lowNum['Code']<5)]
lowNumCode5=lowNum.loc[(lowNum['Code']>=5) & (lowNum['Code']<6)]
lowNumCode6=lowNum.loc[(lowNum['Code']>=6) & (lowNum['Code']<7)]
lowNumCode7=lowNum.loc[(lowNum['Code']>=7) & (lowNum['Code']<8)]
lowNumCode8=lowNum.loc[(lowNum['Code']>=8) & (lowNum['Code']<9)]
lowNumCode9=lowNum.loc[(lowNum['Code']>=9) & (lowNum['Code']<10)]
lowNumCode10=lowNum.loc[(lowNum['Code']>=10) & (lowNum['Code']<11)]
lowNumCode11=lowNum.loc[(lowNum['Code']>=11) & (lowNum['Code']<12)]

for i in range(1,12):
    print(eval("lowNumCode"+ str(i))['Code'].value_counts().sum())

N=11
values=(259,3,330,251,8,0,16,0,10,0,183)
ind=np.arange(N)
width=0.35
label=('Financial Gain','Attention','Enjoyment','Anger','Mental Illness','Cult','Avoid Arrest', 'Gang Activity',
      'Convenience','Wild West Outlaw','Multiple Motivations')

plt.bar(ind,values,color='g',align='center')
plt.xticks(ind+width/2,label,rotation='vertical')
plt.title('Motivations for 2 events/murders and 4 or less Kills')
plt.xlabel('Motivation Type')
plt.ylabel('Frequency')


# After looking at the motivations for low number of kills(4 or less) vs. high number of kills(4 or more), High number of kills appear to be related to events vs. just murders. So a good question you'd want to ask in the future would be is this classification a good fit to be included in the data? For example, it would appear there was a high number of kills pertaining to a stabbing and a fire. I'm not an expert, but it seems feasible that this might not be reasonable criteria for inclusion in serial murders.
# 
# Take away points:
# -High number of kills appear occur in the events category, appear to be more based in anger than anything else
# 
# -Low number of kills appear to have a enjoyment as a significant motivating factor.
# 
# - To help resolve some of the discrepancies, perhaps choose a different categorization system for Type of Killer. 

# In[ ]:





# In[ ]:




