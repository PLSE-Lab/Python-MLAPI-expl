#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# ### Loading Data Files 

# In[48]:


df_electrol_data14 = pd.read_csv("../input/LS2014Electors.csv")
df_electrol_data09 = pd.read_csv("../input/LS2009Electors.csv")
df_candidate_data14 = pd.read_csv("../input/LS2014Candidate.csv")
df_candidate_data09 = pd.read_csv("../input/LS2009Candidate.csv")
df_candidate_data09.head()
df_candidate_data14.head()


# ### Total Electors Count in 2009

# In[49]:


total_electors09=df_electrol_data09["Total_Electors"].sum()
print ("There are a total of ",+total_electors09 ,"electors in India")


# ### Total Electors Count in 2014

# In[50]:


total_electors=df_electrol_data14["Total_Electors"].sum()
print ("There are a total of ",+total_electors ,"electors in India")


# ### Total Voters Count in 2009  

# In[51]:


total_voters09=df_electrol_data09["Total voters"].sum()
print("There are a total of ",+total_voters09 ,"voters in India")


# ### Total Voters Count in 2014

# In[52]:


total_voters=df_electrol_data14["Total voters"].sum()
print("There are a total of ",+total_voters ,"voters in India")


# ###  Total Turnout in 2009

# In[53]:


total_turnout09 = round(total_voters09/total_electors09*100,2)
print("Total Turnout in 2009 is ",+total_turnout09,"%")


# ### Total Turnout in 2014 

# In[54]:


total_turnout = round(total_voters/total_electors*100,2)
print("Total Turnout in 2014 is ",+total_turnout,"%")


# ###  2009 Candidates Gender Distribution 

# In[55]:


candidate_sex = df_candidate_data09["Candidate Sex"].value_counts()
candidate_sex


# There were 7475 Male & 552 Female Candidates in 2009.

# ### Candidates Gender Distribution in 2009 - INC vs BJP

# In[56]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(df_candidate_data09[(df_candidate_data09["Party Abbreviation"]=='INC')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)

fig = plt.gcf() 
fig.suptitle("Candidates Gender Distribution in 2009 - INC vs BJP", fontsize=14) 
ax = fig.gca() 
label = ax.annotate("INC", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(df_candidate_data09[(df_candidate_data09["Party Abbreviation"]=='BJP')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)
fig = plt.gcf() 
ax = fig.gca() 
label = ax.annotate("BJP", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()
plt.show();


# In 2009 , BJP fielded more women candidates (10.2%) than Congress (9.8%).

# ## 2014 Candidates Gender Distribution

# In[57]:


candidate_sex = df_candidate_data14["Candidate Sex"].value_counts()
candidate_sex


# So there where a total of 7578 Male,668 Female and 6 Third Gender Candidates in 2014.

# ### Candidates Gender Distribution in 2014 - INC vs BJP

# In[58]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='INC')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)

fig = plt.gcf() 
fig.suptitle("Candidates Gender Distribution in 2014 - INC vs BJP", fontsize=14) 
ax = fig.gca() 
label = ax.annotate("INC", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='BJP')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)
fig = plt.gcf() 
ax = fig.gca() 
label = ax.annotate("BJP", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()
plt.show();


# In 2014 , Congress(12.9%) fielded more women candidates than BJP (8.9%)

# ### Party Wise Winning Women Candidates in 2009

# In[59]:


df_womenwinners09 = df_candidate_data09[(df_candidate_data09['Position']==1)&(df_candidate_data09["Candidate Sex"]=="F")]
 
ax = df_womenwinners09["Party Abbreviation"].value_counts().plot(kind="pie",radius=2,autopct='%1.1f%%', startangle=90)
x = df_womenwinners09["Party Abbreviation"].value_counts()
x


# In 2009, Congress had 23 women MP's followed by BJP with 13 , BSP & AITC with 4 each.

# ### Party Wise winning Women Candidates in 2014 

# In[60]:


df_womenwinners14 = df_candidate_data14[(df_candidate_data14['Position']==1)&(df_candidate_data14["Candidate Sex"]=="F")]
ax1 = df_womenwinners14["Party Abbreviation"].value_counts().plot(kind="pie",radius=2,autopct='%1.1f%%', startangle=90)
x = df_womenwinners14["Party Abbreviation"].value_counts()
x


# In 2014, BJP had 30 women MP's followed by AITC with 11 , INC & ADMK with 4 each.

# ## Age Distribution of the Winners 

# In[61]:


Age09=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)]['Candidate Age'].tolist()
Age14=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)]['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([Age09, Age14], bins, label=['2009', '2014'])

plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Distribution of Age of the winners')
plt.show()


# There are more MP's in the Age group 50-80 .

#  **Analysing Alliances (2009)**

# In[79]:


df_candidate_data09["Alliance"] = df_candidate_data09["Party Abbreviation"]

df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["INC","AITC","DMK","NCP","NC","JMM","MUL","VCK","KEC(M)","AIMIM"],value="UPA")
df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["BJP","JD(U)","SHS","RLD","SAD","TRS","AGP","INLD"],value="NDA")
df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["CPM","CPI","RSP","AIFB","BSP","BJD","ADMK","TDP","JD(S)","MDMK","HJS","PMK"],value="Third Front")
df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["SP","RJD","LJP"],value="Fourth Front")
df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["AUDF","JKM(P)","NPF","BOPF","SWP","BKA","SDF","IND","JKN","HJCBL","BVA","JVN","JVM"],value="Others")                                                


# **Analysing Alliances (2014)**

# In[80]:


df_candidate_data14["Alliance"] = df_candidate_data14["Party Abbreviation"]

df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['INC','NCP', 'RJD', 'DMK', 'IUML', 'JMM','JD(s)','KC(M)','RLD','RSP','CMP(J)','KC(J)','PPI','MD'],value="UPA")
df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['BJP','SHS', 'LJP', 'SAD', 'RLSP', 'AD','PMK','NPP','AINRC','NPF','RPI(A)','BPF','JD(U)','SDF','NDPP','MNF','RIDALOS','KMDK','IJK','PNK','JSP','GJM','MGP','GFP','GVP','AJSU','IPFT','MPP','KPP','JKPC','KC(T)','BDJS','AGP','JSS','PPA','UDP','HSPDP','PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF'],value="NDA")

df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['YSRCP',"AITC",'AAAP',"BJD","ADMK",'IND', 'AIUDF', 'BLSP', 'JKPDP',"CPM","TRS","TDP","SP", 'JD(S)', 'INLD', 'CPI', 'AIMIM', 'KEC(M)','SWP', 'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 'BVA', 'HJCBL', 'JVM','MDMK'],value="Others")                            


# ## Alliance wise Distribution of Age of the winners 

# In[81]:


Age09UPA=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)&(df_candidate_data09.Alliance=="UPA")]['Candidate Age'].tolist()
Age09NDA=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)&(df_candidate_data09.Alliance=="NDA")]['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([Age09UPA, Age09NDA], bins, label=['UPA', 'NDA'])
plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Alliance wise Distribution of Age of the winners in 2009')
plt.show()




Age14UPA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="UPA")]['Candidate Age'].tolist()
Age14NDA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="NDA")]['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([Age14UPA, Age14NDA], bins, label=['UPA', 'NDA'])
plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Alliance wise Distribution of Age of the winners in 2014')
plt.show()


# In 2009, UPA had more MP's in the age group 60-70 , whereas in 2014 NDA had more MP's in the age group 70-80 .

# ## 2009 vs 2014  Party Wise Seat Winners
# 

# In[66]:


df_winners09 = df_candidate_data09[df_candidate_data09['Position']==1]
DF09 = df_winners09['Party Abbreviation'].value_counts().head(10)
DF09


# In 2009 General Election, INC emerged as the single largest party with 206 seats followed by BJP with 116 seats.

# In[67]:


df_winners09 = df_candidate_data09[df_candidate_data09['Position']==1]
DF09 = df_winners09['Party Abbreviation'].value_counts().head().to_dict()
S09 = sum(df_winners09['Party Abbreviation'].value_counts().tolist())
DF09['Other Regional Parties'] = S09 - sum(df_winners09['Party Abbreviation'].value_counts().head().tolist())
fig = plt.figure()

ax09 = fig.add_axes([0, 0,.5,.5], aspect=1)
colors = ["#264CE4","#FF5106","#E426A4","#44A122","#F2EC3A","#C96F58"]
ax09.pie(DF09.values(),labels=DF09.keys(),autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2,colors = colors)
ax09.set_title("2009",loc="center",fontdict={'fontsize':20},position=(0.5,1.55))
plt.show()


# In 2009, Congress won 38% of total seats.

# ## 2014 Party wise seat winners

# In[68]:


df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]
DF14 = df_winners14['Party Abbreviation'].value_counts().head(10)
DF14


# In 2014 General Election, BJP emerged as the single largest party with 282 seats followed by INC with 44 seats.

# In[69]:


df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]
DF14 = df_winners14['Party Abbreviation'].value_counts().head().to_dict()
S14 = sum(df_winners14['Party Abbreviation'].value_counts().tolist())
DF14['Other Regional Parties'] = S14 - sum(df_winners14['Party Abbreviation'].value_counts().head().tolist())
fig = plt.figure()

ax14 = fig.add_axes([0, 0,.5,.5], aspect=1)
colors = ["#FF5106","#264CE4","#E426A4","#44A122","#F2EC3A","#C96F58"]
ax14.pie(DF14.values(),labels=DF14.keys(),autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2,colors=colors)
ax14.set_title("2014",loc="center",fontdict={'fontsize':20},position=(0.5,1.55))
plt.show()


# In 2014, BJP won the majority of total seats(52%).

# ## 2009 Party wise Vote Share 

# In[71]:


votespartywise09 = df_candidate_data09.groupby('Party Abbreviation')['Total Votes Polled'].sum()
x09 = votespartywise09.sort_values(ascending=False)[:10].plot(kind="bar")
x09.set_xlabel('Party Abbrevations')
x09.set_ylabel('Votes in Million(100)')
votespartywise09.sort_values(ascending=False)[:10]


# In 2009, Congress got total of  119,111,019  votes. 

# ## 2014 Party wise Vote Share 

# In[72]:


votespartywise14 = df_candidate_data14.groupby('Party Abbreviation')['Total Votes Polled'].sum()
x14 = votespartywise14.sort_values(ascending=False)[:10].plot(kind="bar")
x14.set_xlabel('Party Abbrevations')
x14.set_ylabel('Votes in Million(100)')
votespartywise14.sort_values(ascending=False)[:10]


# In 2014, BJP got total of  171,660,230  votes. 

# ## State Wise Poll Percentage (2009) 

# In[73]:


pollper = df_electrol_data09.groupby("STATE").mean()
LS09 = pollper[['POLL PERCENTAGE']].round(1).sort_values('POLL PERCENTAGE',ascending=False)
ax1 =LS09['POLL PERCENTAGE'].plot(kind='bar',figsize=(20, 15))
for p in ax1.patches:
    ax1.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+2),fontsize=12)


# In 2009,Nagaland casted 90% of its total votes.

# ## State Wise Poll Percentage(2014) 

# In[74]:


pollper14 = df_electrol_data14.groupby("STATE").mean()
LS14 = pollper14[['POLL PERCENTAGE']].round(1).sort_values('POLL PERCENTAGE',ascending=False)
ax14 =LS14['POLL PERCENTAGE'].plot(kind='bar',figsize=(20, 15))
for p in ax14.patches:
    ax14.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+2),fontsize=12)


# In 2014, Eventhough there is a slight reduction in the voting percentage, Nagaland remains the top position in voting.

# In[75]:


df_electrol_data14["STATE"] = df_electrol_data14["STATE"].replace(to_replace = ["Odisha"],value="Orissa")
df_electrol_data14["STATE"] = df_electrol_data14["STATE"].replace(to_replace = ["Chhattisgarh"],value="Chattisgarh")


# ## State wise Poll Percentage (2009 vs 2014) 

# In[76]:


pollper = df_electrol_data09.groupby("STATE").mean()
LS09 = pollper[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y09=[2009 for i in range(35)]
S09=list(LS09['POLL PERCENTAGE'].keys())
P09=list(LS09['POLL PERCENTAGE'].values())

pollper14 = df_electrol_data14.groupby("STATE").mean()
LS14 = pollper14[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y14=[2014 for i in range(35)]
S14=list(LS14['POLL PERCENTAGE'].keys())
P14=list(LS14['POLL PERCENTAGE'].values())


Data = {'YEAR':Y09+Y14,'STATE':S09+S14,'Poll_Percentage':P09+P14}
DF = pd.DataFrame(data=Data)
ax = plt.subplots(figsize=(6, 20))
sns.barplot(x=DF.Poll_Percentage,y=DF.STATE,hue=DF.YEAR)


# Compared to 2009,All the Indian states casted above 50% votes in 2014 general election.

# ## Seats won by the Alliances  (2009)

# In[82]:


SeatsWin = df_candidate_data09[(df_candidate_data09.Position==1)].groupby(['Alliance'])['Position'].sum()
SeatsWin
SeatsWin.plot(kind ="pie",autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2)


# UPA won 47.5 % of total seats in 2009 General Election.

# ## Seats won by Alliances  (2014)

# In[83]:


SeatsWin14 = df_candidate_data14[(df_candidate_data14.Position==1)].groupby(['Alliance'])['Position'].sum()

SeatsWin14.plot(kind ="pie",autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2)


# In 2014, NDA won 58.6% of total seats.

# ## Statewise seats won per Alliance (2009)

# In[84]:


s09 = df_candidate_data09[df_candidate_data09["Position"]==1].groupby("State name")["Alliance"].value_counts()
alliance09 = df_candidate_data09[df_candidate_data09["Position"]==1]["Alliance"].unique()


# In[85]:


l = []
for v in ["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',
       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Orissa', 'Punjab',
       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',
       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',
       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',
       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry']:
         win_party09 = s09[v][alliance09]
         l.append(win_party09.values)
            
df = pd.DataFrame(l,index=["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',
       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',
       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',
       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',
       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry'],columns=alliance09)
s=df.plot(kind="bar",stacked=True,figsize=(18,9),fontsize=15)
s.set_title("State wise seats won per Alliance (2009)",color='g',fontsize=30)
s.set_xlabel("Staes",color='b',fontsize=20)
s.set_ylabel("No. of seats",color='b',fontsize=20)


# In 2009, UPA won seats in almost all the Indian States and Union Tertories.

# ## State wise seats won per Alliance (2014) 

# In[86]:


s14 = df_candidate_data14[df_candidate_data14["Position"]==1].groupby("State name")["Alliance"].value_counts()
alliance14 = df_candidate_data14[df_candidate_data14["Position"]==1]["Alliance"].unique()


# In[87]:


l = []
for v in ["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',
       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',
       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',
       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',
       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry']:
         win_party = s14[v][alliance14]
         l.append(win_party.values)
            
df = pd.DataFrame(l,index=["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',
       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',
       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',
       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',
       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry'],columns=alliance14)
s=df.plot(kind="bar",stacked=True,figsize=(18,9),fontsize=15)
s.set_title("Statewise seats won per Alliances (2014)",color='g',fontsize=30)
s.set_xlabel("States",color='b',fontsize=20)
s.set_ylabel("Seats",color='b',fontsize=20)


# In 2014, Except few NorthEastern states & Kerala, NDA won seats in all other Indian states. 

# In[ ]:




