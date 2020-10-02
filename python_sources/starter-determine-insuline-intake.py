#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This Dataset and Kernel is not only my attempt to explore and eventually solve insuline regulation.   
# It also serves as a starting kernel for others and better real data for people interested in this topic.   
# The kernel contains some usefull data reshapes, and basic plotting.  
# We dont have that much data yet, but i will update the data every week or month.
# 
# In this kernel i will first focus on reshaping the data, and getting the weekday morning data.      
# Mornings do not require a lot of energy (carbonhydrates / sugar), the body is 'fairly' stable and relaxed then..    
# However under certain conditions a diabetics can in their sleep make sugar in the liver  (happens to my friend).   
# So my friend has frequently mornings with higher glucose readings. The human metabolism is a complex.   
# 
# Later on I will hope to use also use the mid-day readings and beyond, taking into acount total insuline intake during that day.    
# (as overal this would stick to some average over multiple days).  
# 
# Side  note 1, because i know this person quite good i have noticed that glucose level impact activity level.   
# And because activity level impacts glucose, i strongly believe the final math will not be linear.  
# 
# ## Spoiler allert
# The main value of my posting turned out to be the custom graph's and data filterings I made.
# A graph showing lines combining previous level - dose - outcome level, and with a history based upon transparancy.   
# I'm showing multiple dimensional data in a clear view (i have a background in art  and design).
# The graph's are a novel way to to find the glucose needs, and from it  people with unstable diabetic reactions, might be better able to determine a proper dose.
# 
# Realizing the effect of the graphics, I made handmade tables of the most common safe dosis and it worked out well. With an aim to keep the diabetic at around 3.5 - 6 (later the hospital wanted it a bit higher  inbetween 6 - 10). It worked remarkable well though.
# Eventually data collection was stopped, as he got an omnipod and the way the device works made him less randomly react to it (using a different insulin as well). Its still sometimes that we look upon it, and so i believe these graph's have great value.
# 
# I have trained a neural net on it, that worked quite well, though paper tables dont require batteries, and thus are more safe.
# The idea here is to make better educated guesses (as a neural net would) but leave it up to the user, as he can see his historic data and the effect of his doses, be free to use this yupiter notebook inside your own google drive where you can keep your own privat dataset.

# In[ ]:





# ## Exploratory Analysis
# At first i will load the data and import required libraries, you can expand the code to see it.
# The bloodGlucose level is inside M---- while the inslune dose in in ----i where  (--- = time ).  M for measurement i for insulin  *(  i  small not capital so you wont mistake it for another character.)*

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import sys
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(os.listdir('../input'))

nRowsRead = None # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/diabetic.csv',sep=',',decimal='.', nrows = nRowsRead,header=0,error_bad_lines=False)
#df1.dataframeName = 'diabetic.csv'
#nRow, nCol = df1.shape
#print(f'There are {nRow} rows and {nCol} columns')

#start with a date (as for simple logging with time stamp)
#print pd.to_datetime(pd.Timestamp.toda, format='%Y-%m-%d', errors='raise', infer_datetime_format=False, exact=True)

#print (pd.to_datetime(pd.Timestamp.toda, format='%Y-%m-%d', errors='raise', infer_datetime_format=False, exact=True))
#range = pd.date_range('2018-01-21', '2015-12-31',freq=24hour) # freq='15min')
#df = pd.DataFrame(index = range)


# repair malformated data updates :
df1.rename( columns={"weekday":"day"},inplace ="true") # to better fit screen
df1.rename (columns = {"0700i":"i0700","0930i":"i0930","1300i":"i1300","1500i":"i1500","1800i":"i1800","2300i":"i2300"},inplace = "true") 
df1.rename (columns={"1i300":"i1300"},inplace = "true")
#print(pd.get_option("display.max_columns"))
                     
df1.head(25)


# ## The dataframe : dfi,  sequental verticle list of insulin
# To investigate this problem some might prefer a long sequental list.  

# In[ ]:


dfi =df1[['day','i0700','i0930','i1300','i1500','i1800','i2300']].copy()
dfi.fillna(0,inplace=True) # while i dont want to fill unknown measurements we can replace NAN in insuline dose safely. (allows for math)
b = dfi.set_index('day').stack(dropna=False)

dfi=b.reset_index(drop=False, level=0).reset_index()
dfi.columns= ['InsulinTime', 'InsulinDay', 'Insulin']
dfi.head(7)


# ## The dataframe : DfiDayTotal, total insulin per day
# A simple table of Insuline totals per day, to check for high / low demand or other statistics.

# In[ ]:


dfiDayTotal = df1[['day','i0700','i0930','i1300','i1500','i1800','i2300']].copy()
dfiDayTotal.fillna(0,inplace=True)
dfiDayTotal['InsulinTotal'] = dfiDayTotal['i0700']+dfiDayTotal['i0930']+dfiDayTotal['i1300']+dfiDayTotal['i1500']+dfiDayTotal['i1800']+dfiDayTotal['i2300']
dfiDayTotal.drop(['i0700','i0930','i1300','i1500','i1800','i2300'], 1, inplace=True)
dfiDayTotal['DayIndex']=np.arange(len(dfiDayTotal))
dfiDayTotal.head(7)


# ## DF2 sequental insuline 
# Showing 11 entries here but this table is quit large since all samples follow each per row now
# Also this table contains column insulin total for that day

# In[ ]:


df2 = df1.loc[:, 'day':'M2300']
a = df2.set_index('day','Glucose').stack(dropna=False)

df2 = a.reset_index(drop=False, level=0).reset_index()
df2.columns= ['time', 'day', 'GlucoseLevel']

df2['ID'] = np.arange(len(df2))
df2 =  pd.concat([df2, dfi], axis=1, sort=False)
df2.drop('InsulinDay', 1, inplace=True)
df2.drop('InsulinTime', 1, inplace=True)
df2.columns= ['time', 'day', 'GlucoseLevel','ID','Insulin']
df2 = df2[['ID', 'time','day','GlucoseLevel','Insulin']]

df2['DayIndex']= np.ceil((df2['ID']+1)/7).astype(int)-1 # to merge it with Daytotal.
#temp = dfiDayTotal.filter(['DayIndex','InsulinTotal'],axis=1)# old.filter(['A','B','D'], axis=1)
#df2 = pd.merge(df2, temp, left_on='DayIndex', right_on='DayIndex')
df2 = pd.merge(df2, dfiDayTotal.filter(['DayIndex','InsulinTotal'],axis=1), left_on='DayIndex', right_on='DayIndex') #joined without using temp
df2.head(11)


# ## Deleting weekends from all dataframes
# Just to improve data, as its known that the weekends are not that regular as work activity.   
# If one skips this step then weekends are included (todo would be nice to have some checkbox / question here ea include weekends ? (j/n)).

# In[ ]:


df2 = df2[df2.day != 6] # deleting saturdays
df2 = df2[df2.day != 7] # deleting sundays

df1 = df1[df1.day != 6] # deleting saturdays
df1 = df1[df1.day != 7] # deleting sundays

dfi = dfi[dfi.InsulinDay != 6] # deleting saturdays
dfi = dfi[dfi.InsulinDay != 7] # deleting sundays

dfiDayTotal = dfiDayTotal[dfiDayTotal.day != 6] # deleting saturdays
dfiDayTotal = dfiDayTotal[dfiDayTotal.day != 7] # deleting sundays
print('Weekend data deleted')


# ## Create a simple graph of it.
# I'm no expert in this, but here we have a simple graph of the data.  
# I took a scatter its less distracting then lines, used the Index as sequence  
# Vagely it seams to show two bands upper and lower ( like if it doesnt like to be at 6 )... (hmm but its too early still).  
# To me though it shows cleary how chaotic the response is, alsmost as if a thermostate went crazzy (thinking back about fussz logic   
# For the first graphics each day is colored in a different color ea red green red green.. (otherwise i got lost here..).   
# Amazingly week 2, where is used a table advice to keep him at 4, isnt even using the most insulin over a week...   
# I think i hit something important here, but 4 isnt our goal...

# In[ ]:


cc=['g','g','g','g','g','g','g','r','r','r','r','r','r','r']# days swap collor, its simple i'm not a pro in this.

ax = df2.plot.scatter(x='ID', y='InsulinTotal', color='DarkBlue', label='Insulin day total',alpha=0.1);
bx = df2.plot.scatter(x='ID', y='Insulin', color='LightBlue', label='Insulin dose',alpha=0.2,ax=ax,s=65);
df2.plot(kind='scatter',x='ID',y='GlucoseLevel',label = 'GlucoseLevel',c=cc,figsize=(20,8),ax=bx)#,c='day'
df2.plot(kind='scatter' ,x='ID', y='Insulin',color='black',figsize=(20,4),label = 'Insulin per dose')


#from pandas.plotting import scatter_matrix 
#scatter_matrix(df2, alpha=0.2,figsize=(20,20))

dfiDayTotal.plot(kind='scatter',x='DayIndex',y='InsulinTotal',label = 'Total insulin per day',figsize=(20,4))#,c='day')


# ## Back to the non linear sequence, lets sort it on glucose level off only workday mornings.
# Great title, but put plain simple, what if one doesnt want to use the previous single column.  
# **df1** constains the original data as from the csv file   
# But from that i can create **dfMorning**, with only a few columns in it, and research their possible relations.   
# Because weekends are verry chaotic i also simply skip that data, lets solve that later.   

# In[ ]:


dfMorning = df1.filter(['day','M0700','M0930','i0700'])
#dfMorning = dfMorning[dfMorning.weekday != 6] # deleting saturdays
#dfMorning = dfMorning[dfMorning.weekday != 7] # deleting sundays


dfMorningSorted = dfMorning.sort_values('M0700')
##dfMorning.head(50)
dfMorningSorted['InsulineEffect']=(dfMorningSorted['M0700']-dfMorningSorted['M0930'])/dfMorningSorted['i0700']
dfMorningSorted.head(12)


# ## Is the reaction to insuline usually equal ?  
# Or is there some relation depending to bloodsuger level (on the bottem).   
# Notice sometimes insuline doenst seam to help at all, he calls them bad days 
# 
# **Notice we dont have that much samples**   
# So this might be a bit biased view for the moment.   
# With more data in the future this might get interesting.  

# In[ ]:


dfm = dfMorningSorted [dfMorningSorted.InsulineEffect>0.0] # for the moment deleting strange negative values
dfNegative = dfMorningSorted [dfMorningSorted.InsulineEffect<0.0] 

# plot a scatter and connect the dots
ax = dfm.plot(kind='scatter', x='M0700', y='InsulineEffect',color='blue')
dfm.plot(kind='line', x='M0700', y='InsulineEffect',color='green',ax=ax, label = 'Effective insuline')

bx = dfNegative.plot(kind='scatter', x='M0700', y='InsulineEffect',color='blue',ax=ax)
dfNegative.plot(kind='line', x='M0700', y='InsulineEffect',color='red',ax=ax,linestyle = ':', label = 'On bad days' )
ax.set(ylabel="Insuline effect per unit", xlabel="blood glucose level")


# ## Visually comparing insulin effect of dose in the morning.
# We humans are better in visuals then we are in numbers, having good graphics is important.  
# So far we've seen the diabetic reaction to insulin is a has a noisy / chaotic pattern.   
# This makes it extra difficult to analyze, we dont have a simple calculation where the result of A + B = C   
# Its more like A + B + random = C + random.   
# Random has side effects it can mean no effect of dropping in glucose, or dropping to much.
# 
# 
# **I need create a new kind of graphics to visualize this, to make it clear :**
# * What starting glucose level is and how much insuline was used to get to a certain value
# * The dangerous zone values say below 3.5 or a 11.5 make it red
# * The good values in between 6 - 10 use a non warning color make it bleu (raising) or geen (lowering)
# * Do we went in between dangerous and good zones make it orange we dont want to be there either.
# * By adjusting transparancy let old samples be vaguely visible and new samples be strongly colored.
# 
# 
# ### Quick conclusion  below graph
# Adding a dose of 4.0 usually lowers, only 2 blue lines + 1 orange raise, 11 go down.   
# Adding a dose of 3.3 stays near stable 2 times, one time raises strong (from 6 to 8)   
# Adding dose of around 3.7 lowers slightly 2 times.   
# When on 9 a dose of 4 can stay 9 or drop to 3, i think both are exceptional reactions, some how from 9 a lot of wild lines apear...   

# In[ ]:


plt.style.use('ggplot') 

plt.rcParams["figure.figsize"] = (13,12)
plt.ylim([2, 15])
plt.xlim([2, 6.5])
plt.title("Glucose from 7:00 morning to 9:30 by insulin adjustment", fontsize=20)
plt.ylabel('Glucose level', fontsize=20)
plt.xlabel('Insulin dose', fontsize=20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
#plt.xticks(np.arange(min(0), max(6)+1, 1.0))

plt.yticks(np.arange(0, 15, step=1))
plt.xticks(np.arange(0, 8.5, step=.5))


for index, row in df1.head(len(df1)).iterrows():
    mhw=0.15
    mlw=1.5
    x =0
    a = 1 /len(df1)*index*0.7+0.3
    y= row['M0700']
    dx = row['i0700']
    dy = row['M0930']-row['M0700']
    if ((row['M0930']> 5) and (row['M0930']<10)):
        if (dy>0):
            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='blue')
        else:
            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='green')
    else:
        if ((row['M0930']> 3.5) and (row['M0930']<11.5)):
            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='orange')
        else:
            plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='red')
    #plt.text(dx, dy, dx, fontsize=9)
plt.show()


# # Visually compare insulin effect after lunch time (13:00)
# The previous morning graph would continou with the graph below.  
# Hwever we cannot extend the graph since its based upon dose.  
# One might also notice a bit higher doses here, but keep in mind that he's taking a meal and has to take a bit more.  
# The graph goes often towards 4 as result of a earlier (but wrong) goals, we need to focus on between 6 and 10 here.   

# In[ ]:


#Needs a FIX because an extra doses could have been given at 15:00 (kink in a line..)
plt.style.use('ggplot') 
plt.rcParams["figure.figsize"] = (20,10)
plt.ylim([2, 15])
plt.xlim([2, 6.5])
plt.title("Glucose from 13:00 morning to 15:00 by insulin adjustment", fontsize=20)
plt.ylabel('Glucose level', fontsize=20)
plt.xlabel('Insulin dose', fontsize=20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)

plt.yticks(np.arange(0, 15, step=1))
plt.xticks(np.arange(0, 12.5, step=.5))

linestyle='-'
mhw=0.15
mlw=1.5
for index, row in df1.head(len(df1)).iterrows():
    x =0
    y= row['M1300']
  
    dx = row['i1300']
    a = 1 /len(df1)*index*0.7+0.3
    if (not np.isnan(row['i1500'])):
        #dx=dx+row['i1500']
        #linestyle=':'
        temp=0
    
    dy = row['M1500']-row['M1300'] 

    if(x+y+dx+dy>0):   #PLACE FOR AFIX 
        if ((row['M1500']> 5) and (row['M1500']<10)):
            if (dy>0):
                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='blue',linestyle=linestyle)
            else:
                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='green',linestyle=linestyle)
        else:
            if ((row['M1500']> 3.5) and (row['M1500']<11.5)):
                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='orange',linestyle=linestyle)
            else:
                plt.arrow(x,y,dx,dy,head_width=mhw,shape='right',lw=mlw,alpha=.9*a,color='red',linestyle=linestyle)
plt.show()


# ## The evening
# Data is less stable after work hours (evening meals differ more)   
# this graph is not yet ready therefor (data requires more processing).   
# **NOT READY MATH NEEDS ADJUSTMENTS**

# In[ ]:


# todo make a graph for the evening but there is some incosistancy of later measurements 20:00 andd 23:00 and sometimes none.
# not often insuline is used at 1500 how to deal with that..
plt.style.use('ggplot') 

plt.rcParams["figure.figsize"] = (10,10)
plt.ylim([2, 15])
plt.xlim([2, 6.5])
plt.title("Glucose from 13:00 morning to 18:00 by insulin adjustment", fontsize=20)
plt.ylabel('Glucose level', fontsize=20)
plt.xlabel('Insulin dose', fontsize=20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
#plt.xticks(np.arange(min(0), max(6)+1, 1.0))
plt.yticks(np.arange(0, 15, step=1))
plt.xticks(np.arange(0, 6.5, step=.5))

for index, row in df1.head(len(df1)).iterrows():
    end = 'M1800'
    start = 'M1300'
    x =0
    y= row['M1300']
    dx = row['i1500']
    dy = row[end]-row['M1300']
    a = 1 /len(df1)*index*0.7+0.3
    if ((row[end]> 5) and (row[end]<10)):
        if (dy>0):
            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='blue')
        else:
            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='green')
    else:
        if ((row[end]> 3.5) and (row[end]<11.5)):
            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='orange')
        else:
            plt.arrow(x,y,dx,dy,head_width=.1,shape='right',alpha=.9*a,color='red')

plt.show()


# # Finding bad days
# Lets define bad days differently not by insulin effectiveness.   
# But by how many extra doses where given that day.   
# Maybe a pattern can be found.like mondays are worse (weekend recovery), kinda doubt that.  
# Or perhaps near the end of the week (tired of work) things get worse.  
# Reminding diabetis is a disease that distorts the bodies energy process.  
# 
# not ready not ideal maybe different  

# In[ ]:


# to be written ...


# ## Work in progress 
# This is a work in progress I require more data for sure (as we can clearly see from above we have strange results see 9.0 mornings.) Those numbers are real though, and it oultines the problems he has to live width on a daily basis.  From above i think i should pre select values maybe on average response or ignore weird responses. (no worries as if his doses is wrong he will check no 9:30 and takes aditional insulin if required, again by a best guess. Side notes in above you see a lot of 4.0 doses thats because he usely did do 4 in the morning. I want to have a base of measurements as well, but one might determine how unlikely some responses are, and how common other responses are. (ea last on 14.7 to 11.7 doses 5 seams still to low dose, but drop is roughly as in the one above it (still to high as well), but seams a reasonable response.  (kinda wished i could plot colors in above table).

# In[ ]:




