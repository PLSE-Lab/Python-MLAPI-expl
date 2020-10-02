#!/usr/bin/env python
# coding: utf-8

# # As we all know that things become more understandable and clear when we are able to visualise them. So, keeping this in mind i have created this Kernel which consists of 80% Data Visualisation. 
# 
# ## Let's Begin....

# In[ ]:


# Import Important Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action="ignore")


# In[ ]:


# Load Data

df = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')
df.head()


# In[ ]:


# Drop the unrequired columns
df.drop(columns=['SYMBOL','ASSETS','LIABILITIES','GENERAL\nVOTES','POSTAL\nVOTES','OVER TOTAL ELECTORS \nIN CONSTITUENCY','TOTAL ELECTORS'],inplace=True)

# Renaming of COLUMNS
df.rename(columns={'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY':'VOTE PERCENTAGE','TOTAL\nVOTES':'TOTAL VOTES','CRIMINAL\nCASES':'CRIMINAL CASES'},inplace=True)


# In[ ]:


# Checking for NULL values in DataFrame

df.isnull().sum()


# ## The NULL values consists of NOTA(None of the Above) records. We don't require them as of now. So, we will create a Copy of original DataFrame excluding all the NOTA records.

# In[ ]:


# Creating copy of DataFrame excluding NOTA records

df_exclude_NOTA = df.copy()
df_exclude_NOTA.dropna(inplace=True)


# In[ ]:


df_exclude_NOTA.shape


# In[ ]:


df_exclude_NOTA.head()


# In[ ]:


df_exclude_NOTA.EDUCATION.unique()


# In[ ]:


df_exclude_NOTA['EDUCATION'][df_exclude_NOTA['EDUCATION']=='Post Graduate\n'] = 'Post Graduate'


# # How Qualified our Candidates were?? Let's find out...

# In[ ]:


# Education Qualification of Candidates for LOK SABHA - 2019

ax = df_exclude_NOTA.EDUCATION.value_counts().plot.bar(
figsize=(12,4),
color = 'green',
fontsize =14    
)

ax.set_title('Education Qualification of Candidates for LOK SABHA - 2019',fontsize=18)
ax.set_ylabel('Number of candidates',fontsize=16)

sns.despine()


# ## List of Illiterate Candidates in Lok Sabha Election - 2019 :

# In[ ]:


df[df['EDUCATION']=='Illiterate']


# ## **List of Illiterate Candidates WINNING Lok Sabha Election - 2019 : **

# In[ ]:


df[(df['EDUCATION']=='Illiterate')&(df['WINNER']==1)]


# # Which Party Contested how many Seats ? Let's Find Out...

# In[ ]:


# Number of Seats Contested by PARTIES (TOP 20)

ax=df_exclude_NOTA.PARTY.value_counts().head(20).plot.bar(
figsize=(18,5),
color = '#2A89A1',
fontsize=12)

ax.set_title('Number of Seats Contested by PARTIES (TOP 20)',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(left=True,bottom=True)


# ## Let's find out SEAT CONVERSION RATE of Parties(Win Percentage)...

# In[ ]:


def win_percent_convertor(party):
    total_contested_seats = df[df['PARTY']==party].shape[0]
    total_seats_won = df[(df['PARTY']==party)&(df['WINNER']==1)].shape[0]
    win_percent = (total_seats_won/total_contested_seats)*100
    return win_percent


# In[ ]:


# Creating a SERIES containing information of Win Percentage PARTYWISE

party_win_percent = {}

for party in df['PARTY'].unique():
    party_win_percent[party] = win_percent_convertor(party)
    
party_win_percent_series = pd.Series(party_win_percent)  

party_win_percent_series


# In[ ]:


# Let's find out Seat Conversion Rate PARTYWISE

ax=party_win_percent_series.sort_values(ascending=False).head(36).plot.bar(
figsize=(17,5),
color='lawngreen'    
)

ax.set_title('Seat Conversion Rate',fontsize=20)
ax.set_xlabel('Political Parties',fontsize=14)
ax.set_ylabel('Win Percentage',fontsize=14)

sns.despine(bottom=True,left=True)


# # Which PARTY WON how many seats?? Let's find out...

# In[ ]:


# Number of Seats WON by Parties (TOP 20)

ax=df_exclude_NOTA['PARTY'][(df_exclude_NOTA['WINNER']==1)].value_counts().head(20).plot.bar(
figsize=(16,4),
color='#EC7063'
)

ax.set_title('Number of Seats WON by Parties (TOP 20)',fontsize=20)
ax.set_ylabel('Seats Won',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)
sns.despine(bottom=True,left=True)


# ## Let's compare PARTYWISE number of Seats WON and LOST by them...

# In[ ]:


# Top 20 Parties on the basis of Seats Contesting 

top_20_parties = pd.Series(df_exclude_NOTA['PARTY'].value_counts().head(21))
top_20_parties = top_20_parties.index.drop(['IND'])

top_20_parties


# In[ ]:


# Creating DataFrame which consists of Top 20 Parties on the basis of Seats Contested

df_partiwise_seats_comparison = pd.DataFrame(columns=df_exclude_NOTA.columns)

for count,party in enumerate(df['PARTY']):
    if party in top_20_parties:
        df_partiwise_seats_comparison = df_partiwise_seats_comparison.append(df.loc[count],ignore_index=True)


# In[ ]:


# Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)

plt.figure(figsize=(17,6))
ax = sns.countplot(x='PARTY',hue='WINNER',data=df_partiwise_seats_comparison,palette='Set1')
ax.set_title('Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)',fontsize=20)
ax.legend(['Seats Lost','Seats Won'],loc='upper right',frameon=False),
ax.set_xlabel('Political Parties',fontsize=16)
ax.set_ylabel('Number of Seats',fontsize=16)

sns.despine(bottom=True,left=True)


# # Do Political Parties give tickets to Candidates with CRIMINAL BACKGROUND?? Let's find out...
# 
# ## ** BY THE END OF THIS SECTION, YOU WILL BE SHOCKED TO SEE THE RESULTS

# In[ ]:


# Number of Tickets given by PARTIES to People with CRIMINAL BACKGROUND

ax=df_exclude_NOTA['PARTY'][df_exclude_NOTA['CRIMINAL CASES']!='0'].value_counts().head(20).plot.bar(
figsize=(18,6),
color='red'    
)

ax.set_title('Number of Tickets given by PARTIES to People with CRIMINAL BACKGROUND',fontsize=20)
ax.set_ylabel('Number of Tickets',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


def criminal_or_not(value):
    if value !='0':
        criminal_value = 1
    else:
        criminal_value = 0
    return criminal_value


# In[ ]:


# Creating 1 column in DataFrame named 'CRIMINAL BACKGROUND'

df_exclude_NOTA['CRIMINAL BACKGROUND'] = df_exclude_NOTA['CRIMINAL CASES'].apply(criminal_or_not)
df_exclude_NOTA.head()


# In[ ]:


# Creating a Series consisting of Names of Top 20 Political Parties having most number of CRIMINAL CANDIDATES

top_20_crim_cand_parties = df_exclude_NOTA['PARTY'][df_exclude_NOTA['CRIMINAL CASES']!='0'].sort_index().value_counts().head(20)
top_20_crim_cand_parties = top_20_crim_cand_parties.index

top_20_crim_cand_parties


# In[ ]:


# Creating DataFrame consisting of Top 20 Political Parties having most number of CRIMINAL CANDIDATES

df_top_20_criminal_parties = df_exclude_NOTA.copy()

for party,index in zip(df_top_20_criminal_parties['PARTY'],df_top_20_criminal_parties['PARTY'].index):
    if party not in top_20_crim_cand_parties:
        df_top_20_criminal_parties.drop(index=index, inplace=True)


# ## Let's check which PARTY gave how many tickets to candidates with CRIMINAL BACKGROUND

# In[ ]:


df_exclude_NOTA.head()


# In[ ]:


# Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)

plt.figure(figsize=(16,5))
ax = sns.countplot(data=df_top_20_criminal_parties,x='PARTY',hue='CRIMINAL BACKGROUND')

ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],loc='upper right',frameon=False)
ax.set_title('Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Creating DataFrame consisting of only Candidates having CRIMINAL BACKGROUND

df_criminal = pd.DataFrame(columns=df_exclude_NOTA.columns)
df_criminal = df_exclude_NOTA.copy()
df_criminal = df_criminal[df_criminal['CRIMINAL BACKGROUND']==1]

for party,index in zip(df_criminal['PARTY'],df_criminal['PARTY'].index):
    if party not in top_20_crim_cand_parties:
        df_criminal.drop(index=index, inplace=True)


# In[ ]:


df_criminal.shape


# ## Let's see how many Criminal Background Candidates WON and LOST Partywise :

# In[ ]:


# Comparison of Seats Won and Lost by CRIMINAL CANDIDATES of Parties (TOP 20 PARTIES)

plt.figure(figsize=(17,5))
ax = sns.countplot(x='PARTY',hue='WINNER',data=df_criminal,palette='husl')
ax.set_title('Comparison of Seats Won and Lost by CRIMINAL CANDIDATES of Parties (TOP 20 PARTIES)',fontsize=20)
ax.legend(['Seats Lost','Seats Won'],loc='upper right',frameon=False),
ax.set_xlabel('CRIMINAL CANDIDATES of Political Parties',fontsize=16)
ax.set_ylabel('Number of Seats',fontsize=16)

sns.despine(bottom=True,left=True)


# # STATEWISE analysis of CRIMINAL BACKGROUND Candidates :

# ## 1. BIHAR

# In[ ]:


# Total Number of Candidates in BIHAR

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')],x='CRIMINAL BACKGROUND',palette='dark')

ax.set_title('Total Number of Candidates in BIHAR',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)

sns.despine(bottom=True,left=True)


# ### In Bihar, Candidates with CRIMINAL BACKGROUND were more in the Election as compared to CLEAN IMAGE Candidates

# In[ ]:


# Total Number of Candidates in BIHAR

plt.figure(figsize=(16,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')

ax.set_title('Total Number of Candidates in BIHAR',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)
ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)

sns.despine(bottom=True,left=True)


# # What a SHOCKING RESULT !!!!
# 
# ## In BIHAR, large number of Candidates with CRIMINAL BACKGROUND WON the election as compared to CLEAN IMAGE Candidates

# # 2. Uttar Pradesh

# In[ ]:


# Total Number of Candidates in Uttar Pradesh

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')],x='CRIMINAL BACKGROUND',palette='dark')

ax.set_title('Total Number of Candidates in Uttar Pradesh',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Total Number of Candidates in Uttar Pradesh

plt.figure(figsize=(16,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')

ax.set_title('Total Number of Candidates in Uttar Pradesh',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)
ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)

sns.despine(bottom=True,left=True)


# # What a SHAME !!!
# 
# ## Like Bihar, in UP also number of Candidates with CRIMINAL BACKGROUND WINNING Election was more then number of Candidates with CLEAN IMAGE.

# ## 3. WEST BENGAL

# In[ ]:


# Total Number of Candidates in West Bengal

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')],x='CRIMINAL BACKGROUND',palette='dark')

ax.set_title('Total Number of Candidates in West Bengal',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Total Number of Candidates in West Bengal

plt.figure(figsize=(16,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')

ax.set_title('Total Number of Candidates in West Bengal',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)
ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)

sns.despine(bottom=True,left=True)


# # And again the same picture. REALLY SAD !!!
# 
# ## In West Bengal also, Candidates with CRIMINAL BACKGROUND WINNING the Election were more than Candidates with CLEAN IMAGE
# 

# ## 4. PUNJAB

# In[ ]:


# Total Number of Candidates in Punjab

plt.figure(figsize=(16,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Punjab')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')

ax.set_title('Total Number of Candidates in Punjab',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)
ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)

sns.despine(bottom=True,left=True)


# # FINALLY !!!
# 
# ## Finally, we found one State in which Candidates with CLEAN IMAGE WINNING Election are much more as compared to Candidates with CRIMINAL BACKGROUND

# ### In the similar way, you can check the stats for whichever State you want to see.

# # How AGED and YOUNG our Candidates were ? Let's find out...

# In[ ]:


# Political Parties having CANDIDATES ABOVE 70

ax = df_exclude_NOTA['PARTY'][df_exclude_NOTA['AGE']>70].value_counts().plot.bar(
figsize=(17,6),   
color='#EB984E'
)

ax.set_title('Political Parties having CANDIDATES ABOVE 70',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Political Parties having CANDIDATES BELOW 35

ax = df_exclude_NOTA['PARTY'][df_exclude_NOTA['AGE']<35].value_counts().plot.bar(
figsize=(17,6),   
color='#1ABC9C'
)

ax.set_title('Political Parties having CANDIDATES BELOW 35',fontsize=20)
ax.set_ylabel('Number of Candidates',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# ## **5 Most Aged People Contesting ELECTION :**

# In[ ]:


df_exclude_NOTA.sort_values(by='AGE',ascending=False).head()


# ## **5 Most Aged People Contesting ELECTION and WINNING their seat :**

# In[ ]:


df_exclude_NOTA[df_exclude_NOTA['WINNER']==1].sort_values(by='AGE',ascending=False).head()


# ## **5 Most YOUNG People Contesting ELECTION and WINNING their seat**

# In[ ]:


df_exclude_NOTA[df_exclude_NOTA['WINNER']==1].sort_values(by='AGE').head()


# # How many MALES and FEMALES Contested Elections and WON it ?? Let's find out...

# In[ ]:


# Number of MALES and FEMALES contesting Election

ax = sns.countplot(data=df_exclude_NOTA,x='GENDER',palette='cubehelix')

ax.set_title('Number of MALES and FEMALES contesting Election',fontsize=20)
sns.despine(bottom=True,left=True)


# In[ ]:


# Comparison of Seats WON and LOST by Candidates

ax = sns.countplot(data=df_exclude_NOTA,x='GENDER',hue='WINNER',palette='hls')

ax.legend(['seats lost','seats won'],frameon=False)
ax.set_title('Comparison of Seats WON and LOST by Candidates',fontsize=20)
sns.despine(bottom=True,left=True)


# In[ ]:


# Elected FEMALE PARLIAMENTARIANS - State Wise

ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['GENDER']=='FEMALE')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(
figsize=(16,5),
color='#AF7AC5'
)

ax.set_title('Elected FEMALE PARLIAMENTARIANS - State Wise',fontsize=20)
ax.set_ylabel('Number of Elected Parliamentarians',fontsize=16)
ax.set_xlabel('States',fontsize=16)

sns.despine(bottom=True,left=True)


# # Let's see how 2 National Parties - BJP and CONGRESS performed STATEWISE :

# In[ ]:


# Seats won by BJP - Statewise

ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['PARTY']=='BJP')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(
figsize=(16,5),
color ='#FBA21C'
)

ax.set_title('Seats won by BJP - Statewise',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('States',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Seats won by Congress - Statewise

ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['PARTY']=='INC')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(
figsize=(16,5),
color ='#29AA2C'
)

ax.set_title('Seats won by Congress - Statewise',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('States',fontsize=16)

sns.despine(bottom=True,left=True)


# # STATEWISE Total Seats Analysis :

# In[ ]:


# Seats in Uttar Pradesh

plt.figure(figsize=(8,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')&(df_exclude_NOTA['WINNER']==1)],x='PARTY')

ax.set_title('Seats in Uttar Pradesh',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Seats in West Bengal

plt.figure(figsize=(8,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')&(df_exclude_NOTA['WINNER']==1)],x='PARTY')

ax.set_title('Seats in West Bengal',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Seats in Maharashtra

plt.figure(figsize=(8,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Maharashtra')&(df_exclude_NOTA['WINNER']==1)].sort_values(by='WINNER'),x='PARTY')

ax.set_title('Seats in Maharashtra',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# In[ ]:


# Seats in Bihar

plt.figure(figsize=(8,5))
ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')&(df_exclude_NOTA['WINNER']==1)].sort_values(by='WINNER'),x='PARTY')

ax.set_title('Seats in Bihar',fontsize=20)
ax.set_ylabel('Number of Seats',fontsize=16)
ax.set_xlabel('Political Parties',fontsize=16)

sns.despine(bottom=True,left=True)


# ## Thanks for watching this Kernel !!
# 
# ### Your Suggestions are always welcomed :)
