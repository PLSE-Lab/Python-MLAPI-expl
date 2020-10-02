#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# In my previous kernal["Finding out Safe Seats"](https://www.kaggle.com/ammarmalik/finding-out-safe-seats), I tried to find out the seats which had been won by same party over the last 3 elections.  But we found out that as far as parties are concerned only *Pakistan Peoples Party Parlimentarians (PPPP)* and *Muttahida Qaumi Movement Pakistan (MQM)* were able to retain the seats for previous 3 elections, all those belonged to province of *Sindh*. But we all know that certain seats in other provinces are also considered strong holds for particular political party. In 2002 Elections, the two biggest parties *Pakistan Muslim League-N (PML-N)* and *PPPP* were contesting the elections without their leaders i.e. *Nawaz Sharif* and *Benazir Bhutto* who were in exile at that time. Before 2008 Elections, both the leaders returned to lead their respective parties, (although *Benazir Bhutto* was shot just before the elections). So, it would be interesting to see if we ignore 2002 elections and consider only 2008 and 2013 elections to look for the seats that were retained by same party, and hence are **On a Hat-trick** to retain it for the third time as well.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## 2008 Elections ##
NA2008 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
Winner08 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA08 = NA2008.loc[NA2008['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes', 'Seat']]
    if NA08.empty == True:
        print("2008 Missing:",Const) # missing constituency
    if NA08.empty == False:
        MAX = (NA08.loc[NA08['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes'], MAX['Seat']], ).T
        temp.index = [i]
        Winner08 = pd.concat([Winner08,temp])
Winner08 = Winner08.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes', 4:'Seat'}) # Winners of 2008 Elections


## 2013 Elections ##
NA2013 = pd.read_csv("../input/National Assembly 2013 - Updated.csv", encoding = "ISO-8859-1")
Winner13 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA13 = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes', 'Seat']]
    Votes = NA13['Votes'].astype('int64')
    NA13['Votes'] = Votes
    if NA13.empty == True:
        print("2013 Missing:",Const) # missing constituency
    if NA13.empty == False:
        MAX = (NA13.loc[NA13['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes'], MAX['Seat']]).T
        temp.index = [i]
        Winner13 = pd.concat([Winner13,temp])
Winner13 = Winner13.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes', 4:'Seat'})


# In[ ]:


#Winner08


# In[ ]:


Total_Seats = 272
Winners = pd.DataFrame([])
Con = pd.DataFrame([])

for i in range(Total_Seats):
    Const = "NA-%d"%(i+1)
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        tempCon = (Winner13.loc[Winner13['Constituency']==Const,['Constituency']])
        tempCon = tempCon.values.ravel()
        tempSeat = (Winner13.loc[Winner13['Constituency']==Const,['Seat']])
        tempSeat = tempSeat.values.ravel()
        temp13 = (Winner13.loc[Winner13['Constituency']==Const,['Party']])
        temp13 = temp13.values.ravel()
        temp08 = (Winner08.loc[Winner08['Constituency']==Const,['Party']])
        temp08 = temp08.values.ravel()
        temp = pd.DataFrame([tempCon, tempSeat,temp08,temp13])
        temp.columns = [i]
#        temp = temp.rename(columns = {0:'Winner'})
        Winners = pd.concat([Winners,temp], axis = 1)
        Con = pd.concat([Con,pd.DataFrame([Const])])


# In[ ]:


Final = Winners.T
Final = Final.rename(columns = {0: 'Constituency', 1: 'Seat', 2:'2008', 3:'2013'})
Final['2008'] = Final['2008'].replace(['MUTTHIDA\xa0MAJLIS-E-AMAL\xa0PAKISTAN'], 'Muttahidda Majlis-e-Amal Pakistan')
Final['2008'] = Final['2008'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League(QA)')
Final['2013'] = Final['2013'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League(QA)')


# **List of Winners**
# 
# So, finally we can observe the winners from every constituency for 2008 and 2013 elections  in the following table.

# In[ ]:


Final


# In[ ]:


Total_Seats = 272
Safe = pd.DataFrame([])
for i in range(Total_Seats):
    Const = "NA-%d"%(i+1)
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        tempCon = (Final.loc[Final['Constituency']==Const,['Constituency']])
        tempCon = tempCon.values[0][0]
        tempSeat = (Final.loc[Final['Constituency']==Const,['Seat']])
        tempSeat = tempSeat.values[0][0]
        Party = (Final.loc[Final['Constituency']==Const,['2008']])
        Party = Party.values[0][0]
        Num = len(np.unique(Final.loc[Final['Constituency'] == Const]))-2
#        if (Num == 2):
#            Num = 0
#            Num = 100
        temp = pd.DataFrame([tempCon, tempSeat, Party, Num]).T
        temp.index = [i]
        Safe = pd.concat([Safe,temp])

Safe_Const = Safe[Safe[3]==1]
Safe_Const


# Now we can observe that out of 265 constitutencies analyzed *(accounting for the 8 constituencies for which the results were not available in the dataset)* 116 were retained by same parties or won by *Independent* candidates in last 2 elections. Which means that in the upcoming elections approximately **~44%** seats are **ON-A-HAT-TRICK** (*i.e. have the channce to complete Three in a Row*) 

# Lets find out which parties are **on a hattrick** in upcoming elections:

# In[ ]:


np.unique(Safe_Const[2])


# So, now unlike results of previous kernel, in addition to *PPPP*, *MQM* and *Independent* candidates, this time we can see *Pakistan Muslim League (N) (PML-N)*,  *Pakistan Muslim League (F) (PML-F)*, *Awami National Party (ANP)*, and *National Peoples Party (NPP)* are also in the list.

# In[ ]:


#Safe
PPPP_Safe = (Safe_Const[2]=='Pakistan Peoples Party Parliamentarians').sum()
MQMP_Safe = (Safe_Const[2]=='Muttahida Qaumi Movement Pakistan').sum()
Ind_Safe =  (Safe_Const[2]=='Independent').sum()
PMLN_Safe = (Safe_Const[2]=='Pakistan Muslim League (N)').sum()
PMLF_Safe = (Safe_Const[2]=='Pakistan Muslim League (F)').sum()
ANP_Safe =  (Safe_Const[2]=='Awami National Party').sum()
NNP_Safe =  (Safe_Const[2]=='National Peoples Party').sum()

x = np.arange(len(np.unique(Safe_Const[2])))
value = [PPPP_Safe, MQMP_Safe, Ind_Safe, PMLN_Safe, PMLF_Safe, ANP_Safe, NNP_Safe]

plt.figure(figsize=(14,8))
plt.grid()
pp, mq, nd, pmn, pmf, anp, nnp = plt.bar(x,value)
plt.xticks(x,('PPPP', 'MQM-P', 'Ind', 'PML-N', 'PML-F', 'ANP', 'NNP'))
plt.ylabel('Seats')
pp.set_facecolor('r')
mq.set_facecolor('g')
nd.set_facecolor('b')
pmn.set_facecolor('y')
pmf.set_facecolor('k')
anp.set_facecolor('m')

plt.show()


# **Rating the Safe Seats**
# 
# Now we know about seats that were retained in previous 2 elections. But it is well known, that for some of these seats in 2013 Elections, the results were very close. For example, from **NA-61**, *Sardar Mumtaz Khan* of *PML-N* got **114282** votes (approx. 43%), however, the runnerup from this constituency was *Ch. Pervaiz Ellahi* of *PML-Q* who got **99373** votes (approx. 38%). Hence, the difference was of approximately 5%, which indicates the result was very close. So, now we try to rate the safer seats found in terms of percentage difference.

# In[ ]:


Safe_Rating = pd.DataFrame([])
for i in range(len(Safe_Const)): #len(Safe_Const)
    Const = Safe_Const.iloc[i,0]
#    print(Const)
    NA = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes','TotalVotes', 'Seat']]
    Votes = NA['Votes'].astype('int64')
    NA['Votes'] = Votes
    NA = NA.sort_values(by=['Votes'], ascending = False)
    NA = NA.iloc[0:2,:]
    if Const == 'NA-46':  ### Total Votes for NA-46 missing in original data
        NA['TotalVotes'] = 16857 
    for j in range(len(NA)):
        temp = (NA.iloc[j,:])
        V = temp['Votes']*100/temp['TotalVotes']
        NA.iloc[j,2] = V
    
    Win = NA.iloc[0,1]
#    print(Win)
    Diff = NA.iloc[0,2] - NA.iloc[1,2]
    temp = pd.DataFrame([Const,Diff,Win]).T
    temp.index = [i]
    Safe_Rating = pd.concat([Safe_Rating,temp])

Safe_Rating = Safe_Rating.sort_values(by=[1], ascending = True)
Safe_Rating = Safe_Rating.reset_index()
Safe_Rating = Safe_Rating.drop(['index'], axis=1)
Safe_Rating = Safe_Rating.rename(columns = {0:'Constituency', 1:'Diff(%)', 2:'Winner'})
Safe_Rating = Safe_Rating.set_index('Constituency')
Safe_Rating['Diff(%)'] = Safe_Rating['Diff(%)'].astype('float16')


# In[ ]:


#Safe_Rating


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10,20))
sns.heatmap(Safe_Rating.loc[:,['Diff(%)']], annot=True, cmap='viridis')
plt.show()


# From the above plot we can see that for **NA-242**, the percentage difference between winning candidate and runnerup candidate was whooping 77%, hence this seat can be labelled as **strong hold** for the winning party (*MQM-P*) in this case. Whereas, for **NA-192** the difference was only 0.65%. Therefore, despite *PPPP* winning this seat consecutively for the last two elections, this seat can not be labelled as *strong hold* of *PPPP*. Lets plot the histogram.

# In[ ]:


plt.figure(figsize=(14,8))
sns.distplot(Safe_Rating['Diff(%)'],kde=False, rug=True);


# So, we can see that around 30 constituencies, which were retained by political parties in previous 2 elections, had the difference of approx. less than 10%, so there is high probablity they might not be lucky enough to win for the consecutive third time. Lets see the Top 10 constituencies with the highest difference and Top 10 with the lowest difference.

# In[ ]:


Top_High_Diff = Safe_Rating.tail(10)
Top_High_Diff


# In[ ]:


Top_Low_Diff = Safe_Rating.head(10)
Top_Low_Diff


# **Please upvote** and feel free to comment and discuss regarding this kernel. 
