#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ** In this kernel I would like to find the "True Winners"" for 2018 general elections. I consider "True Winners" as those candidates who secure more than 50% of the total valid votes casted in the constitutency.  ** 
# 
# ** Reason behind this is that if more than 50% of the votes have been casted in favor of one candidate only then he can claim to be the real winner otherwise majority of the voters prefered other candidates over this candidate **

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


## 2018 Elections
Winner18 = pd.DataFrame([])
NA2018 = pd.read_csv("../input/NA-Results2018 Ver 2.csv", encoding = "ISO-8859-1")
Constituencies = NA2018.Constituency_Title.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
#    NA = NA2018.loc[NA2018['Constituency_Title']==Const, :]
    NA = NA2018.loc[NA2018['Constituency_Title']==Const, ['Constituency_Title','Part', 'Votes','Total_Votes']]
    NA = NA.sort_values(by=['Votes'], ascending = False)
    NA = NA.iloc[0:1,:] # Only Winner
    NA['Percent_Votes'] = NA.Votes/NA.Total_Votes
    Winner18 = pd.concat([Winner18,NA])
Winner18 = Winner18.loc[Winner18['Constituency_Title'] != 'NA-42',:] # Removing NA-42 as there is some problem in results
Winner18 = Winner18.loc[Winner18['Constituency_Title'] != 'NA-39',:] # Removing NA-39 as there is some problem in results
Winner18 = Winner18.set_index('Constituency_Title')


# In[ ]:


Winner = (Winner18.groupby(['Part'])['Part'].count())
Winner = Winner.sort_values(ascending=False)
plt.figure(figsize=(15,10))
plt.title('Elections 2018')
Winner[0:5].plot(kind='bar',fontsize=15) # Display only Top 5
plt.rc('axes',labelsize=0)
plt.rc('axes',titlesize=20) 
plt.grid(True)
plt.show()


# In[ ]:


print("Number of seats won by each party")
Winner


# ** So in terms of total number of seats won PTI leads the way by 114 seats, followed by PML-N 64, and PPPP 43 **
# 
# ** But let us go bit deep and find out the True Winners **

# In[ ]:


True_Win = Winner18.loc[Winner18['Percent_Votes']>=0.5,:]
#True_Win = True_Win.loc[True_Win['Constituency_Title'] != 'NA-42',:] # Removing NA-42 as there is some problem in results
print("Number of True Winners : ", True_Win.shape[0])
True_Win.groupby(['Part'])['Part'].count()


# # Analysis
# 
# ** So an interesting observation is that only 51 constituencies *(approx. 19%)*  have seen the winning candidate receiving more than 50% of the valid votes casted ** 
# 
# ** Another interesting observation is that PPPP has won 21 of these constituency which is most, and second is PML(N) winning 15 , whereas PTI is third on this list by winning 13 such seats **

# In[ ]:



PTI = Winner18.loc[Winner18['Part']=='Pakistan Tehreek-e-Insaf',:]
PMLN = Winner18.loc[Winner18['Part']=='Pakistan Muslim League (N)',:]
PPPP = Winner18.loc[Winner18['Part']=='Pakistan Peoples Party Parliamentarians',:]

print("Mean Percentage vote Overall : %.2f "%(Winner18.Percent_Votes.mean()*100))
print("Mean Percentage vote for PTI : %.2f "%(PTI.Percent_Votes.mean()*100))
print("Mean Percentage vote for PML(N) : %.2f"%(PMLN.Percent_Votes.mean()*100))
print("Mean Percentage vote for PPPP : %.2f"%(PPPP.Percent_Votes.mean()*100))


# ** We can also see that for PTI winners the Percentange Vote per constituency is less than the overall average **

# In[ ]:


## 2018 Elections
Runnerup18 = pd.DataFrame([])
NA2018 = pd.read_csv("../input/NA-Results2018 Ver 2.csv", encoding = "ISO-8859-1")
Constituencies = NA2018.Constituency_Title.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
#    NA = NA2018.loc[NA2018['Constituency_Title']==Const, :]
    NA = NA2018.loc[NA2018['Constituency_Title']==Const, ['Constituency_Title','Part', 'Votes']]
    NA = NA.sort_values(by=['Votes'], ascending = False)
    NA = NA.iloc[1:2,:] # Only Runnerup
#    NA['Percent_Votes'] = NA.Votes/NA.Total_Votes
    Runnerup18 = pd.concat([Runnerup18,NA])
Runnerup18 = Runnerup18.loc[Runnerup18['Constituency_Title'] != 'NA-42',:] # Removing NA-42 as there is some problem in results
Runnerup18 = Runnerup18.loc[Runnerup18['Constituency_Title'] != 'NA-39',:] # Removing NA-39 as there is some problem in results
Runnerup18 = Runnerup18.set_index('Constituency_Title')


# In[ ]:


Runnerup18 = Runnerup18.rename(columns = {'Part':'RunnerUp', 'Votes':'RunnerUp_Votes'})
Winner18 = Winner18.rename(columns = {'Part':'Winner', 'Votes':'Winner_Votes'})
Win_Run = pd.concat([Winner18,Runnerup18],axis=1)
Win_Run = Win_Run.loc[:,['Winner','RunnerUp','Winner_Votes','RunnerUp_Votes','Total_Votes']]
Win_Run['Win%'] = Win_Run['Winner_Votes']/Win_Run['Total_Votes']
Win_Run['Run%'] = Win_Run['RunnerUp_Votes']/Win_Run['Total_Votes']
Win_Run['Diff%'] = (Win_Run['Win%'] - Win_Run['Run%'])
Sorted_Win_Run = Win_Run.sort_values(by=['Diff%'], ascending = True)


# In[ ]:


Sorted_Win_Run_Less10 = Sorted_Win_Run.loc[Sorted_Win_Run['Diff%']<=0.1,:]
Sorted_Win_Run_Less10


# ** From above table we can see that for 136 constituencies (approx. 50%) , the margin of victory was less than 10% ** 
# 
# ** Lets see who was Winner in majority of these constituencies**

# In[ ]:


Winner_Less10 = Sorted_Win_Run_Less10.groupby(['Winner'])['Winner'].count()
print("Party-wise winners with less than 10% margin")
Winner_Less10


# In[ ]:


PTI_Less10 = Winner_Less10['Pakistan Tehreek-e-Insaf']/Winner['Pakistan Tehreek-e-Insaf']
PMLN_Less10 = Winner_Less10['Pakistan Muslim League (N)']/Winner['Pakistan Muslim League (N)']
PPPP_Less10 = Winner_Less10['Pakistan Peoples Party Parliamentarians']/Winner['Pakistan Peoples Party Parliamentarians']

print("PTI Wins with less than 10%% margin = %.2f"%PTI_Less10)
print("PMLN Wins with less than 10%% margin = %.2f"%PMLN_Less10)
print("PPPP Wins with less than 10%% margin = %.2f"%PPPP_Less10)


# ** Again we see that about 52% of the seats won by PTI, had margin of victory less than 10% **
# 
# ** So according to above analysis the  rankings of the True Winners for 2018 elections should be as follows ** 

# # 1. Pakistan Peoples Party Parliamentarians (PPPP)
# 
# # 2. Pakistan Muslim League-N (PML-N)
# 
# # 3. Pakistan Tehreek-e-Insaf (PTI)
