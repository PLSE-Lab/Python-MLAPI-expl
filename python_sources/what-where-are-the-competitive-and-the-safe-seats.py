#!/usr/bin/env python
# coding: utf-8

# # Seat Analysis: how they voted, who is holding and who is targeting?
# ### Regional Information
# Let's open the dataset and take a look at how the constituencies are spread across the United Kingdom.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
get_ipython().run_line_magic('matplotlib', 'inline')

filename = '../input/uk_election_data.csv'

df = pd.read_csv(filename, sep = ",", header = 0)

print(df.groupby('Region')['Region'].count().sort_values(ascending=False))


# After that, we take a broader look at the results from the 2015 elections. First, let's see how the parties performed in each region of the United Kingdom.

# In[ ]:


tab = pd.pivot_table(df[['WinningParty','Region']],
            index='WinningParty',columns='Region', aggfunc = len, fill_value = 0)
tab['Total'] = tab.apply(sum, axis = 1)
tab.sort_values(['Total'], ascending = False, inplace=True)

print(tab)


# Then, let's see how many percent of the votes the winning party received at each constituency. To achieve that, we draw an histrogram and get a description of the `WinningPct` column

# In[ ]:


df['WinningPct'].hist(bins=30)
plt.xlim(0,100)
plt.ylim(0,70)
plt.xlabel('Votes received by winning party (%)')
plt.ylabel('Number of constituencies')
plt.show()
print(df['WinningPct'].describe())


# 

# In[ ]:


df['MajorityPct'].hist(bins=30)
plt.xlim(0,100)
plt.ylim(0,70)
plt.xlabel('Majority of winning party (%)')
plt.ylabel('Number of constituencies')
plt.show()
print(df['MajorityPct'].describe())


# To see how easy/difficult was the election at the constituency level, we take a look at `MajorityPct` column and see that more than half of the constituencies were won in landslides (margin of victory greater than 20%).

# In[ ]:


df["SeatStatus"] = ['battleground' if x < 5.
                    else 'leaning' if x < 10. else 'likely safe' if x < 20.
                    else 'safe' for x in df["MajorityPct"]]

tab = pd.pivot_table(df[['WinningParty','SeatStatus']],
            index='WinningParty',columns='SeatStatus', aggfunc = len, fill_value = 0)
tab['Total'] = tab.apply(sum, axis = 1)
tab.sort_values(['Total'], ascending = False, inplace=True)

print(tab)
print()
print(df.groupby('SeatStatus')['SeatStatus'].count().sort_values(ascending=False))


# ---
# ### Competitiveness status for each seat
# Let us find out how many seats were competitive in 2015. We define four possible status for each constituency.
# 
# + battleground seat: if the margin of victory was smaller than 5%
# + leaning seat: margin of victory between 5%-10%.
# + likely safe seat: margin of victory between 10%-20%.
# + safe seat: margin of victory above 20%.
# 
# The distribution of seat status for each party is given below

# In[ ]:


tab = pd.pivot_table(df[['Region','SeatStatus']],
            index='Region',columns='SeatStatus', aggfunc = len, fill_value = 0)
tab['Total'] = tab.apply(sum, axis = 1)
tab.sort_values(['Total'], ascending = False, inplace=True)

print(tab)


# We can see that the races in 2017 not be competitive (probably) for almost 60% of the seats.
# 
# ---
# ### Brexit status for each seat
# Repeating the procedure for the 2016 referendum result. This timesix possible status for each constituency.
# 
# + Strong Leave: Leave had more than 60% of the vote.
# + Moderate Leave: Leave had 55-60% of the vote.
# + Weak Leave: Leave had 50-55% of the vote.
# + Weak Remain: Leave had 45-50% of the vote.
# + Moderate Remain: Leave had 40-45% of the vote.
# + Strong Remain: Leave had less than 40% of the vote.

# In[ ]:


df['LeavePct'].hist(bins=30)
plt.xlim(0,100)
plt.ylim(0,70)
plt.xlabel('Leave votes in 2016 referendum (%)')
plt.ylabel('Number of constituencies')
plt.show()

df['BrexitStatus'] = ['strong remain' if x < -20.
                    else 'moderate remain' if x < -10.
                    else 'weak remain' if x < 0.
                    else 'weak leave' if x < 10.
                    else 'moderate leave' if x < 20.
                    else 'strong leave' for x in df["LeaveMajority"]]
print(df.groupby('BrexitStatus')['BrexitStatus'].count().sort_values(ascending=False))


# Now the `BrexitStatus` for each party. It is interesting to see that Labour has a large number of seats to defend on both strong remain and strong leave constituencies.

# In[ ]:


tab = pd.pivot_table(df[['WinningParty','BrexitStatus']],
            index='WinningParty',columns='BrexitStatus', aggfunc = len, fill_value = 0)
tab = tab.iloc[:,[2,0,4,5,1,3]]
tab['Total'] = tab.apply(sum, axis = 1)
tab.sort_values(['Total'], ascending = False, inplace=True)
print(tab)


# Taking a look at how each what is the `BrexitStatus` across the regions.

# In[ ]:


tab = pd.pivot_table(df[['Region','BrexitStatus']],
            index='Region',columns='BrexitStatus', aggfunc = len, fill_value = 0)
tab = tab.iloc[:,[2,0,4,5,1,3]]
tab['Total'] = tab.apply(sum, axis = 1)
tab.sort_values(['Total'], ascending = False, inplace=True)
print(tab)


# ---
# ## Finding the vulnerable and target seats for each party
# 
# Defining some auxiliary code.

# In[ ]:


parties = {
        'Con': 'Conservative',
        'Lab': 'Labour',
        'LD': 'Liberal Democrat',
        'Grn': 'Green',
        'PC': 'Plaid Cymru',
        'SF': 'Sinn Fein',
        'UUP': 'Ulster Unionist',
        'DUP': 'Democratic Unionist',
        'Ind': 'Independent',
        'UKIP': 'UKIP',
        'SNP': 'SNP',
        'Spk': 'Speaker',
        'Others': 'Others',
        'Alliance': 'Alliance'
}

def vulnerable_seats(party, num=150):

    if (num > 650) or (num < 1): 
        raise ValueError('Number of seats should be between 1 and 650')
        
    row_index = df['WinningParty'] == parties[party]
    vulnerable_seats = df.loc[row_index,['Constituency', 'Region', 'SeatStatus',
        'MajorityPct','SecondPlace']]
    vulnerable_seats.rename(columns={'SecondPlace': 'TargetedBy'}, inplace=True)
    vulnerable_seats.rename(columns={'MajorityPct': 'MajPct'}, inplace=True)
    vulnerable_seats['TargetedBy'] = vulnerable_seats['TargetedBy'].replace('Liberal Democrat','Lib Dem')

    sorted_vulnerable_seats = vulnerable_seats[['Constituency', 'Region',
        'MajPct','TargetedBy']].sort_values('MajPct')\
        .reset_index(drop=True)[:num]
    sorted_vulnerable_seats.index += 1
    return sorted_vulnerable_seats

def table(df):
    return tabulate(df,headers = df.columns)
    #return tabulate(df,headers = df.columns, tablefmt='psql')


def target_seats(party, num=150, swing=50):
    if (num > 650) or (num < 1): # allow only realistic no-show rates
        raise ValueError('Number of seats should be between 1 and 650')
    
    row_index = df['WinningParty'] != parties[party]
    target_seats = df.loc[row_index, :]
    target_seats.rename(columns={'WinningParty': 'CurrentParty'}, inplace=True)

    target_seats = target_seats.drop('Majority', 1)
    target_seats = target_seats.drop('MajorityPct', 1)
    target_seats['Majority'] = target_seats.loc[:,'WinningVotes'] - target_seats.loc[:,party]
    target_seats['MajorityPct'] = 100*target_seats.loc[:,"Majority"]/target_seats.loc[:,"ValidVotes"]
    target_seats['MajorityPct'] = target_seats.loc[:,'MajorityPct'].map('{:0,.2f}'.format).astype(float)
    target_seats['FromSecondPlace'] = 0.5*target_seats.loc[:,'MajorityPct']
    target_seats['ToSecondPlace'] = 100*(target_seats.loc[:,'SecondPlaceVotes'] - target_seats.loc[:,party])/ target_seats.loc[:,'ValidVotes']
    target_seats['Swing'] = target_seats.loc[:,['FromSecondPlace','ToSecondPlace']].max(axis=1)
    target_seats['Swing'] = target_seats.loc[:,'Swing'].map('{:0,.2f}'.format).astype(float)

    target_seats = target_seats.loc[:,['Constituency', 'Region',
        'Swing','CurrentParty']]
    target_seats['CurrentParty'] = target_seats['CurrentParty'].replace('Liberal Democrat','Lib Dem')

    row_index = target_seats['Swing'] <= swing
    target_seats = target_seats.loc[row_index,:]
    sorted_target_seats = target_seats.sort_values(['Swing'])        .reset_index(drop=True)[:num]
    sorted_target_seats.index += 1
    return sorted_target_seats


# ### Conservative vulnerable seats
# 
# Seats sorted by the majority (in percent) that the party is defending at each constituency.

# In[ ]:


party = 'Con'
print(table(vulnerable_seats(party, num=50)))


# ### Conservative target seats
# Seats sorted by ```Swing```, where *swing* is defined as the percentage of vote a party needs to take from the party that is currently holding the seat.

# In[ ]:


print(table(target_seats(party, swing = 5)))


# ### Labour vulnerable seats

# In[ ]:


party = 'Lab'
print(table(vulnerable_seats(party, num=50)))


# ### Labour target seats

# In[ ]:


print(table(target_seats(party, swing = 5)))


# ### Liberal Democrat vulnerable seats

# In[ ]:


party = 'LD'
print(table(vulnerable_seats(party)))


# ### Liberal Democrat target seats

# In[ ]:


print(table(target_seats(party, swing = 5)))


# ### Scottish National Party vulnerable seats

# In[ ]:


party = 'SNP'
print(table(vulnerable_seats(party, num=50)))


# ### Scottish National Party target seats

# In[ ]:


print(table(target_seats(party, swing = 5)))


# ### UKIP vulnerable seats

# In[ ]:


party = 'UKIP'
print(table(vulnerable_seats(party)))


# ### UKIP target seats

# In[ ]:


print(table(target_seats(party, swing = 5)))

