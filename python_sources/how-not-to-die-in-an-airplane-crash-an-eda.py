#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


#Read the data
airplane =pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv', parse_dates=['Date'], index_col='Date' )
airplane.head(2)


# In[ ]:


# Add another column in airplane dataframe for survivors 
airplane['Survivor']=airplane['Aboard']-airplane['Fatalities']
airplane.head(2)


# In[ ]:


# plot using pandas plot
survior=airplane.resample('YS').agg({'Survivor':'sum',
                                    'Ground':'sum',
                                    'Fatalities':'sum',
                                    'Aboard':'sum'})
sx=survior.plot(kind='line',figsize=(15,5))


# We see that from early 40s till early 70s the fatalities were on rise. The rate started reducing by the late 90s. We can assume that by this time the technology was lot better than it was ever before which might have helped reduce no. of crashes and hence no. of fatalities.
#  
# From the above chart, we can see that no. of people Aboard, Fatalities and Survivor lines follow appromaximately the same trend. Ground fatalities have been consistently lowest through out but there is an abrupt spike. Let's find out more about it.

# ### Find details about when maximum ground fatalities happened

# In[ ]:


#Find % of Total ground fatalities and % of total 
ag=airplane.resample('YS').agg({'Ground':['sum',lambda x: x.sum()/airplane['Ground'].sum()]})

#Label the columns
ag.columns = ag.columns.map('_'.join).str.replace('<lambda>','% of Total')

#Find the year when maximun ground fatalities happened
ag.sort_values(by=['Ground_% of Total'],ascending=False).head()


# 2001 has the maximum ground fatalities, 67% of the total ground fatalities happened in just 1 year.
# WTC attack hapenned in year 2001 and this explains the reason of sudden spike.

# ### Which was the worst year to fly ?
# 

# In[ ]:


aa=airplane.resample('YS').agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'], ascending =False)
aa.rename(columns={'Fatalities':'Total Fatalities per year'}).head(10)


# 70s and 80s are among the worst years for airline industry. 
# 1972 is the worst so far, causing maximum no. of fatalities.
# 

# ### Who has turned out to be worst operator?

# In[ ]:


# Find sum and % of Total fatalities by Operator
gp=airplane.groupby('Operator').agg({'Fatalities':['sum',lambda x: x.sum()/airplane['Fatalities'].sum()]})

#Flatten the multiindex 
gp.columns=gp.columns.map('_'.join)

#Rename Columns
gp.rename(columns={'Fatalities_sum':'Total Fatalities',
                   'Fatalities_<lambda>':'% of Total Fatalities'},inplace=True)

#Sort by Total Fatalities
gp.sort_values(by='Total Fatalities', ascending=False).head()


# Aeroflot is by far the most dangerous operator. People have 68% chances of dying if they flew with Aeroflot than any other operator.

# ### Did things become better over time for Aeroflot?

# In[ ]:


filt=airplane['Operator']=='Aeroflot'
airplane2=airplane[filt]
airplane2.resample('YS').agg({'Fatalities':'sum'}).plot(kind='line',figsize=(15,5));


# Early 70s was exceptionally bad period for Aeroflot but things started getting better by later half of 70s.
# 
# Fatalities reduced after 1977-1978 but did things really get better or Aeroflot just decided to fly less and thus less no. of fatalites? There is not enough data to find it out so I'll leave this analysis here.

# ### Multiple operators with word 'Aeroflot' in the name
# There are multiple operators with word 'Aeroflot' in them. Let's find out more about them to see if all such Operators need to be combined together or not.

# In[ ]:


gp.reset_index(inplace=True)


# In[ ]:


#List of all operators with word aeroflot 
filt=gp['Operator'].str.contains('Aeroflot')
aeroflot=gp[filt]
aeroflot.head()


# In[ ]:


#aeroflot['Operator'].unique()

#no. of uniques operators with word 'Aeroflot' in it
aeroflot['Operator'].nunique()


# There are 7 unique Operators under the name of Aerofloat.

# In[ ]:


aeroflot.groupby('Operator').agg({'Total Fatalities':sum}).sort_values(by='Total Fatalities', ascending=True).plot.barh();


# There is huge difference between Aeroflot and all the other operators combined together. We can combine top 2 together but it will not make any difference in the analysis whether we choose to combine rest of the operator or not.

# ### Lets dig more on why Aerofloat is performing so bad. Is it using a particular type of aircraft that is causing crashes?

# In[ ]:


filt=airplane['Operator']=='Aeroflot'
aerotype=airplane.loc[filt,['Operator','Type','Fatalities']]
aerotype.head()


# In[ ]:


aerotype.groupby('Type').agg({'Fatalities':'sum'}).sort_values('Fatalities',ascending=False).head(10)


# ### There is no one Type stands out as the reason. Can it be the location of crash? Let's find out.

# In[ ]:


gp=airplane.groupby('Location').agg({'Fatalities':['sum',lambda x: x.sum()/airplane['Fatalities'].sum()]})

#Flatten the multiindex 
gp.columns=gp.columns.map('_'.join)

#Rename Columns
gp.rename(columns={'Fatalities_sum':'Total Fatalities',
                   'Fatalities_<lambda>':'% of Total Fatalities'},inplace=True)

#Sort by Total Fatalities
gp.sort_values(by='Total Fatalities', ascending=False).head()


# Russia is at 3rd and 4th place but Tenerife, Canary Islands and Japan are the top 2 location of crash. 
# It is interesting to notice how small islands are the leading location of crashes. 
# 
# It would have been interesting to check how my total flight flew over these islands and what percentange of flights ended up crashing. There is not enough data to complete this analysis but we can look into the summary field to find out if there was any particular reason of crashes. I'll create another kernel to do text analysis on the summary filed but for now I'll leave it at this point.

# ### Does Registartion No. has anything to do with the crashes?

# In[ ]:


tg=pd.Grouper(freq='YS')
airplane.groupby([tg,'Registration']).agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'],ascending=False).head(10)


#  Registration no. doesn't looks like a culprit here

# ### Can it be Type No. ?

# In[ ]:


#tg=pd.Grouper(freq='YS')
airplane.groupby(['Type']).agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'],ascending=False).head(10).plot.bar()


# Douglas DC-3 is atleast 5 times more probable to crash than any other type of aircraft
# 

# ### What are the operators that use Douglas DC-3?

# In[ ]:


filt=airplane['Type']=='Douglas DC-3'
doug=airplane.loc[filt,['Operator','Fatalities']].sort_values('Fatalities', ascending=False)
doug.head(10)


# Aerofloat is not among top 10 users of Douglas DC-3. So we can conclude that it's not the combination of Aerofloat and Douglas DC-3 that has been fatal.

# ### Let's find out if there is any pointer in Route information

# In[ ]:


airplane['Route'].isna().count()


# In[ ]:


airplane['Route'].notna().count()


# Route field is only 50% filled. There is not enough information available to impute the values as well. We can't rely much on this field but let's see if there is anything that stands out in 50% of the data where the route is filled in.

# In[ ]:


#No. of plane crashes per route
route=airplane.groupby('Route').agg({'Fatalities':['sum',lambda x:x.sum() / airplane['Fatalities'].sum()]})
route.head()


# In[ ]:


route.columns=route.columns.map(''.join)
route.head()


# In[ ]:


route.reset_index(inplace=True)
route.head()


# In[ ]:


route.rename(columns={'Fatalitiessum':'Total Fatalities',
                      'Fatalities<lambda>':'% of Total Fatalities'}, inplace=True)
route.head()


# In[ ]:


#Sort by desc
route.sort_values(by='Total Fatalities', ascending=False).head()


# Top 2 routes are same as we found with Location. Canary Islands and Japan have the highest no. of fatalities. But it is interesting to notice that 'Training' is the 3rd highest Route for crash.
# 
# Lets see a sample of records for 'Training'

# In[ ]:


route=airplane.loc[airplane['Route']=='Training',['Route','Aboard','Fatalities']]
route.head()


# In[ ]:


# No. of Trainig flights per year
route.resample('YS').size().plot(kind='line', figsize=(10,5));


# In[ ]:


route.resample('YS').agg({'Fatalities':['sum'],
                         'Aboard':'sum'}).plot(kind='line', figsize=(10,5));


# There isn't any trend evident from the chart above.

# ### Conclusion:
# 1. Don't fly with Aeroflot, there is 68% chance of you dying.
# 2. Don't fly in a Douglas DC-3, you are 5 times more probale to die.
# 3. And finally, don't take any flight that flies over canary Islands or Japan.
