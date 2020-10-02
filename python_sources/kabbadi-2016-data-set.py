
# coding: utf-8

# In[174]:


import pandas as pd


df= pd.read_csv('../input/kabbadi_data.csv', delimiter=",")

df


# In[8]:




# 1. games won by teams and its winning rate

# 2.does toss result affecting match result

# 3. best defending team & best raiding team

# 4. closet match in the tournament


# In[77]:


#1 games won by teams
# for this we need to slice the dataframe
from bokeh.io import output_notebook,show
from bokeh.layouts import column,row
from bokeh.plotting import figure

#from bokeh.plotting import plot
import matplotlib.pyplot as plt
win = df.loc[:,['team','oppTeam','matchResult']]

winarr=[]
losearr=[]


def winteams(win):
    
    for i in range(len(win)):
        if (win.iloc[i,2]==0):
            winarr.append(win.iloc[i,1])
            losearr.append(win.iloc[i,0])
        
        if (win.iloc[i,2]==1):
            winarr.append(win.iloc[i,0])
            losearr.append(win.iloc[i,1])
    
winteams(win)
wining= pd.DataFrame(winarr,columns=['winteams'])
#wincount
losing = pd.DataFrame(losearr,columns=['lostteams'])
#winarr
#get_ipython().magic('matplotlib inline')
wint= pd.Series(winarr)
wint=wint.unique()
wint1=[]
wint
wint2=wint.tolist()
for i in range(len(wint)):
    count=0
    for j in range(len(winarr)):
        
        if(wint[i]==winarr[j]):
            count=count+1
    wint1.append(count)
    
p= figure(x_range=wint2,plot_height=400,plot_width=300)
p.vbar(x=wint2, top=wint1, width=0.1)




#lint= pd.Series(losearr)
#lint=lint.unique()
lint1=[]
#lint
lint2=wint.tolist()
for i in range(len(wint)):
    count=0
    for j in range(len(losearr)):
        
        if(wint[i]==losearr[j]):
            count=count+1
    lint1.append(count)
    
p1= figure(x_range=lint2,plot_height=400,plot_width=400)
p1.vbar(x=lint2, top=lint1, width=0.1)


show(row(p,p1))





# In[91]:




# if u see india has won most of the matches and lost only 2 matches 
# hence we can say india is best team in the tournament



# find total numbers of matches played by the teams

wint
totalmatches=[]
for i in range(0,(len(lint1)-1)):
    totalmatches.append(wint1[i]+lint1[i])
    
p1= figure(x_range=wint2,plot_height=400,plot_width=400)
p1.vbar(x=lint2, top=totalmatches, width=0.1)
show(p1)


# In[92]:


# we can clearly see india and iran are the only one who played 14 matches . Hence they are the finalists

# we can also observe korea and thailand are the teams who fought for 3Rd place


# In[96]:


#2. HOW TOSS EFFECTING MATCH


toss = df.loc[:,['team', 'oppTeam','tossResult','matchResult']]

tossresult=0

for i in range(len(toss)):
    if(toss.iloc[i,2]==toss.iloc[i,3]):
        tossresult=tossresult+1

tosspercentage = (tossresult/len(toss))*100
tosspercentage


# In[97]:


# 60% of the teams who won the toss have won the matches in this league'

# hence we can predict that if we win a toss we will have more points of winning the match


# In[178]:


#4 . closest match in this tournament
toss1 = df.loc[:,['totalPntsDiff']]

toss2= toss1.as_matrix()

small= toss2[0]
count=0
for i in range(len(toss2)):
    if(toss2[i]<=small):
        small=toss2[i]
        count=i

#small[0]
count

teamslist=[]

teamslist.append(df.loc[count,['team','oppTeam']])

teamslist[0]


# In[ ]:




