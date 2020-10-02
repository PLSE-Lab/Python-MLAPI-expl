#!/usr/bin/env python
# coding: utf-8

# # Crash Course Pandas 1
# Im creating these series for people that have beginner/intermediate knowledge about pandas library.I m going to share my working notes(after tidying) to pass the info,nothing more. Certain codeblocks could get repetitive. Some info looks contradictory even though i have tried to remove most of them. Sorry for it beforehand. Also goodluck on your learning journey. This is my first kernel as well. Wish me luck. Contact me for any question, i will help you out if i can.
# 
# Contents:
# * [Dataset](#1)
# * [Missing Value Handling](#2)
# * [Men or Women Speed More Often](#3)
# * [Gender Effects on Who Gets Searched During Stop](#4)
# * [Missing Values on Search Type Column](#5)
# * [During Search, How Often Drivers Get Frisked](#6)
# * [Which Year Had Least Number of Stops](#7)
# * [How Does Drug Activity Change by Time of Day](#8)
# * [Do Most Stops Occur at Night](#9)
# * [Find the Bad Data in Stop Duration Column](#10)

# <a id="1"></a> 
# ## Dataset

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ri = pd.read_csv("../input/police.csv")
ted = pd.read_csv("../input/ted.csv")


# In[ ]:


ri.head()


# In[ ]:


ri.dtypes


# In[ ]:


ri.shape


# In[ ]:


ri.isnull().sum()


# <a id="2"></a> 
# ## Missing Value Handling

# In[ ]:


#Never use inplace=True on the first time,check the output first then use inplace=True.Unless you are really sure about f()
ri.drop("county_name",axis=1,inplace=True)
#ri.drop("county_name",axis=columns,inplace=True)


# In[ ]:


ri.head()


# In[ ]:


ri.columns


# In[ ]:


#alternative value
ri.dropna(axis=1,how="all")
#ri.dropna(axis="columns",how="all")


# <a id="3"></a> 
# ## Men or Women Speed More Often

# In[ ]:


ri.head()


# In[ ]:


#answer is "male"
ri[ri.violation=="Speeding"].driver_gender.value_counts()


# In[ ]:


#if u want percentage base result-normalize
ri[ri.violation=="Speeding"].driver_gender.value_counts(normalize=True)


# In[ ]:


ri[ri.driver_gender=="M"].violation_raw.value_counts(normalize=True)


# In[ ]:


ri[ri.driver_gender=="F"].violation_raw.value_counts(normalize=True)


# In[ ]:


#if we want see the result with 1 line of code in same df i would use groupby.
ri.groupby("driver_gender").violation_raw.value_counts(normalize=True)
#for every gender(for each gender) check violations and count them


# In[ ]:


#this is series.which is vectors for R
type(ri.groupby("driver_gender").violation_raw.value_counts(normalize=True))


# In[ ]:


#more in depth approach.
ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).loc[:,"Speeding"]


# In[ ]:


#this data has multi index so unstack ll make it 1 index.lets try it
ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).index


# In[ ]:


#now your data becomes dataframe.
ri.groupby("driver_gender").violation_raw.value_counts(normalize=True).unstack()


# <a id="4"></a> 
# ## Gender Effects on Who Gets Searched During Stop

# In[ ]:


ri.head()


# In[ ]:


ri.search_conducted.value_counts(normalize=True)


# In[ ]:


#~%3 percent of people get searched.
ri.search_conducted.mean()
#91741*0.0348372047=3196 get searched.


# In[ ]:


ri.shape


# In[ ]:


#value counts doesnt count nan values,however this col has no null values.
ri.search_conducted.value_counts()


# In[ ]:


ri.groupby("driver_gender").search_conducted.mean()
#62895*0.043326=2725(M), 23511*0.020033=471(F) 2725+471=3196


# In[ ]:


ri.driver_gender.value_counts()


# In[ ]:


#which violation committed by which gender by what percentages (multiple groupby statements incoming)
ri.groupby(["violation","driver_gender"]).search_conducted.mean()
#now we can understand which gender commits which violation by what percentage.Some violations more prone to searched.
#not for every violation you get searched. pulled over by cop != getting searched.


# <a id="5"></a> 
# ## Missing Values on Search Type Column

# In[ ]:


ri.isnull().sum()


# In[ ]:


#why search_type is missing 88k times?
#bcs there is no search on in that pulled over cases.


# In[ ]:


ri.search_conducted.value_counts()


# In[ ]:


#as we mentioned earlier value_counts doesnt count nan values.Total is 3196.
ri.search_type.value_counts()


# In[ ]:


#why is this an empty series? not just an usual output (series with 88545 as a result)
ri[ri.search_conducted==False].search_type.value_counts()
#by default nan/na is dropped.so you python cant count them.WHEN s_c is False s_t is nan as u could guess.


# In[ ]:


#if u dont want to drop nan values
ri[ri.search_conducted==False].search_type.value_counts(dropna=False)


# In[ ]:


#if u want to see whole picture.
ri.search_type.value_counts(dropna=False)


# <a id="6"></a> 
# ## During Search, How Often Drivers Get Frisked

# In[ ]:


#there are python built in string methods like upper() and there are pandas string methods which are much more broader.


# In[ ]:


#when search type has "protective frisk" in it.There are multiple search_type cases with protective frisk in it.
ri.search_type.str.contains("Protective Frisk")


# In[ ]:


#this string method (contains) can be used with Series. bcs search_type is a series.
ri["frisk"]=ri.search_type.str.contains("Protective Frisk")


# In[ ]:


ri.frisk.value_counts(dropna=False)


# In[ ]:


#mean() doesnt count nan values. 274/(274+2922)= ~0.086
ri.frisk.mean()


# <a id="7"></a> 
# ## Which Year Had Least Number of Stops

# In[ ]:


ri.head()


# In[ ]:


#lets take only year from our stop_date col.
ri.stop_date.str.slice(0,4)
#u get the years.


# In[ ]:


#lets do value_counts() to see the picture.
ri.stop_date.str.slice(0,4).value_counts()


# In[ ]:


#alternative method-1
#combining 2 string cols first.
combined=ri.stop_date.str.cat(ri.stop_time,sep=" ")
combined


# In[ ]:


ri["stop_datetime"]=pd.to_datetime(combined)
ri.dtypes


# In[ ]:


ri.stop_datetime.dt.year


# In[ ]:


#Alternative method-2
ri = pd.read_csv("../input/police.csv")
ri["year2"]=pd.to_datetime(stop_date)
ri.year2.dt.year
#is not working,to use date_time method your format must have date and time part,just date part isnt enough.Important to note


# In[ ]:


ri = pd.read_csv("../input/police.csv")
combined=ri.stop_date.str.cat(ri.stop_time,sep=" ")
ri["stop_datetime"]=pd.to_datetime(combined)
ri.stop_datetime.dt.year.value_counts()
#auto descending order by default.


# In[ ]:


ri.stop_datetime.dt.year.value_counts().sort_values()
#auto ascending by default.


# In[ ]:


#result is series therefore u can use index[] attributes with it.
ri.stop_datetime.dt.year.value_counts().sort_values().index[0]


# <a id="8"></a> 
# ## How Does Drug Activity Change by Time of Day

# In[ ]:


#lets find drug related stops.
ri.head()


# In[ ]:


ri.drugs_related_stop.mean()


# In[ ]:


#for each hour what is the drug activity?
#ri.groupby("hour").drugs_related_stop.mean()
#this could work if we have a hour col.so lets create one or find a way to use hour from the cols.


# In[ ]:


ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()


# In[ ]:


#lets plot it.Auto plot is lineplot.
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[ ]:


#harder to understand however its a different approach.
ri.groupby(ri.stop_datetime.dt.time).drugs_related_stop.mean().plot()


# In[ ]:


#other exploratory data codes.
#ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.value_counts().plot()
#ri.groupby(ri.stop_datetime.dt.time).drugs_related_stop.value_counts().plot()


# <a id="9"></a> 
# ## Do Most Stops Occur at Night

# In[ ]:


ri.head()


# In[ ]:


#this is a series and series have 2 sorting methods. sort_index() and sort_values() they are ascending by default.
ri.stop_datetime.dt.hour.value_counts()


# In[ ]:


ri.stop_datetime.dt.hour.value_counts().sort_values()


# In[ ]:


#problem is that indexes are not in order so plotting becomes very problematic.we need to use sort_index()
ri.stop_datetime.dt.hour.value_counts().sort_values().plot()


# In[ ]:


#now its ok
ri.stop_datetime.dt.hour.value_counts().sort_index().plot()


# In[ ]:


#different approach that doesnt have a plotting.
ri[(ri.stop_datetime.dt.hour>4)&(ri.stop_datetime.dt.hour<22)].shape


# In[ ]:


#if we consider night as btw 22-04 then ~23k of stops occurred on night 68k of stops occurred on day.
ri.shape


# In[ ]:


#another alternative.
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count()


# In[ ]:


ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()


# <a id="10"></a> 
# ## Find the Bad Data in Stop Duration Column

# In[ ]:


#what counts as bad data.
ri.head()


# In[ ]:


#what is the meaning of "fix it".And how can we fix it?
ri.stop_duration.value_counts(dropna=False)
#we can set 1 and 2 in this result as missing (nan). bcs stop_time is an str col using str.replace could be helpful...


# In[ ]:


ri.dtypes


# In[ ]:


#here is one way to solve it.
ri[(ri.stop_duration==1)|(ri.stop_duration==2)].stop_duration="NaN"
#But there are couple problems
#stop_duration col is string / "NaN" is string not a null. /


# In[ ]:


#this approach ll cause an SettingWithCopyWarning and couldnt handle the situation
ri[(ri.stop_duration=="1")|(ri.stop_duration=="2")].stop_duration="NaN"
ri.stop_duration.value_counts()


# In[ ]:


#moving slowly but surely.
ri.loc[(ri.stop_duration=="1")|(ri.stop_duration=="2"),:]


# In[ ]:


#string NaN isnt same as null nan. we need to import numpy library and use its nan attribute to handle this.
#but i intentionally cause a problem first then try to solve it.
ri.loc[(ri.stop_duration=="1")|(ri.stop_duration=="2"),"stop_duration"]="NaN"


# In[ ]:


#as u can see there are 2 NaN.one of them is string(latest one)
ri.stop_duration.value_counts(dropna=False)


# In[ ]:


import numpy as np
ri.loc[ri.stop_duration=="NaN","stop_duration"]=np.nan


# In[ ]:


#thats it.
ri.stop_duration.value_counts(dropna=False)


# In[ ]:




