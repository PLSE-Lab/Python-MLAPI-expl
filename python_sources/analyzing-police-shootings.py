#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Functions

# In[ ]:


def simplify_armed(x):
    '''Simplify the armed column so that it only has three values'''
    if (pd.isnull(x) or x == "undetermined"):
        return "undetermined"
    elif x!="unarmed":
        return "armed"
    else:
        return "unarmed"

def impute_race(x):
    '''Impute the race column so that nan values return O'''
    if pd.isnull(x):
        return "O"
    else:
        return x
        
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(),2)
        axes.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def grouped_bars(fig, axes, labels,bar_data,xcoors,n_sub_bars,w,sub_labels,xlb,ylb,title):
    '''Create a grouped bar graph'''
    rects = []
    width = w / n_sub_bars
    n = len(bar_data)
    for i in range(n):
        rects.append(axes.bar(xcoors - 0.25 + (i * (w/n_sub_bars)), bar_data[i], width, label=sub_labels[i]))
    
    axes.set_xlabel(xlb)
    axes.set_ylabel(ylb)
    axes.set_title(title)
    axes.set_xticks(xcoors)
    axes.set_xticklabels(labels)
    axes.legend()
    for i in range(n):
        autolabel(rects[i])
    fig.set_size_inches(18,8)
    
def get_city_race_values(df,cities,race):
    values = np.zeros(len(cities), dtype=int)
    i=0
    for city in cities:
        tmp = df.query("city_state == @city and race == @race")["id"]
        if tmp.tolist() == []:
            values[i] = 0
        else:
            values[i] = int(tmp.tolist()[0])
        i = i+1
    return values


# # Exploratory Data Analysis

# Let's begin by importing the data and looking at the dimensions of the data frame, as well as the column names, types, and percentage of missing data.

# In[ ]:


ps_df = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")
pd.Series(data=list(ps_df.shape),index=['Rows','Columns'])


# In[ ]:


pd.DataFrame({"Data_Type": ps_df.dtypes, "Percent_Null": ps_df.isnull().mean() * 100})


# As we can see, there is about 10% of the race data missing. Since "O" label is used to indicate when race was unknown, let's go ahead and impute nan values with the "O" label since the difference in interpretation is negligible.

# In[ ]:


ps_df["race"] = list(map(impute_race,ps_df["race"])) #impute the race column


# In[ ]:


ps_df.head()


# While this is a relatively small data set, the data is largely complete&mdash;due partly to imputation of the race column. If we need to later on, we can handle missing data, but for now we can proceed with our analysis without worrying about handling missing items.
# 
# Let's start off with some basic visualizations.

# ### Number of Deaths by Date

# In[ ]:


deaths_by_date = ps_df.groupby("date").agg("count").filter(["id"]).rename(columns={"id":"count"})
plt.figure(figsize=(18,8))
plt.plot(deaths_by_date) #time-series by date


# Immediately we should ask ourselves: is this time series stationary? If the series is not stationary this may suggest that the incidents of police shootings are changing with time; happening on average more often or less often, occuring with variance etc. 
# 
# From the graph, it does not appear to have any trend, but variability is hard to check. Indeed the graph looks very much like noise. However, it is easy and worthwhile to perform a more formal test for stationarity. Let's use the Dickey Fuller statistical test to investigate stationarity of this series.

# In[ ]:


import statsmodels
from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(deaths_by_date, autolag='AIC')

pd.Series({"adf":adf_test[0],"p-value":adf_test[1],"usedlag":adf_test[2],"crit_1_pct":adf_test[4]["1%"],"crit_5_pct":adf_test[4]["5%"],"crit_10_pct":adf_test[4]["10%"]})


# So, by p-value = $ 0 $ and observing the graph, we feel comfortable suggesting that police shootings don't seem to be trending either up or down, or changing in variability. 

# ### Deaths by Age

# In[ ]:


plt.figure(figsize=(18,8))
sns.distplot(ps_df["age"])


# In[ ]:


ps_df["age"].describe()


# The distribution appears unimodal, skewed right. The average age appears to be about 37, with a devastating minimum of 6 years of age.

# ### Deaths by Gender

# In[ ]:


bar_data = ps_df.groupby("gender").agg("count").rename(columns={"id":"Deaths By Gender Count"})
plt.figure(figsize=(18,8))
sns.barplot(bar_data.index,bar_data["Deaths By Gender Count"])


# Clearly, the data are showing that the persons are predominantly male.

# ### Deaths by Race

# In[ ]:


bar_data = ps_df.groupby("race").agg("count").rename(columns={"id":"Deaths By Race Count"})
plt.figure(figsize=(18,8))
sns.barplot(bar_data.index,bar_data["Deaths By Race Count"])


# For more information on the race abbreviations (A,B,etc), see [this page](http://www.dcf.ks.gov/services/PPS/Documents/PPM_Forms/Section_5000_Forms/PPS5460_Instr.pdf).
# 
# 

# In[ ]:


ps_df["armed_simple"] = list(map(simplify_armed,ps_df["armed"])) #add a new column with simplified values from armed column.

#Add plot, same as above, broken down by armed_simple
bar_data = ps_df.groupby(["race","armed_simple"]).agg("count")["id"].reset_index().rename(columns={"id":"count"}) #Get the data for the plot
tots = ps_df.groupby(["race"]).agg("count")["id"].reset_index().rename(columns={"id":"count"})
labels = bar_data["race"].unique() 
labels = labels[pd.isnull(labels) == False] #Create x-tick labels
armed_pct = (bar_data.query("armed_simple == 'armed'")["count"].values / tots["count"]).values #Data for armed_count rectangles
unarmed_pct = (bar_data.query("armed_simple == 'unarmed'")["count"].values / tots["count"]).values #Data for unarmed_count rectangles
undetermined_pct = (bar_data.query("armed_simple == 'undetermined'")["count"].values / tots["count"]).values #Data for undetermined_count rectangles
xcoor = np.arange(1,len(bar_data["race"].unique()) - pd.isnull(bar_data["race"].unique()).sum()+1) #positions of x-tick marks
num_sub_bars = len(bar_data["armed_simple"].unique()) - pd.isnull(bar_data["armed_simple"].unique()).sum() #number of rectangles per category
width = 0.75/num_sub_bars #width of a rectangle
fig, axes = plt.subplots() #Create the plot
grouped_bars(fig,axes,labels,[armed_pct,unarmed_pct,undetermined_pct],xcoor,num_sub_bars,0.75,["armed","unarmed","undetermined"],"Race","Deaths Percent Armed and Race","Deaths by Race and Armed Simple")


# The data show a variety of arms (gun, knife, toy weapon, etc.). As a result, we simplified these values to simply "armed", "not armed", or "undetermined" to take a closer look at shooting incidents with respect to this category.
# 
# Of note:
# - the vast majority of shootings seem to be against armed persons.
#  - It might be worthwhile to do a more thorough engineering of the armed column. Some of the items, such as "toy weapon" and "pepper spray" might be better placed in another category such as non-lethal.
# - African Americans show the highest rate of police shootings while unarmed at 9%.

# ### Top 10 Cities by Deaths and Race

# In[ ]:


ps_df["city_state"] = ps_df["city"] + "," + ps_df["state"]
bar_data = ps_df.groupby("city_state", as_index=False).agg("count").rename(columns={"id":"Deaths By City"}).sort_values(by="Deaths By City",ascending=False).head(10)
city_pops = [39,16,23,6.4,15,27,5.6,8.9,6.4,7.1]
fig, axes = plt.subplots()
fig.set_size_inches(18,8)
axes.plot(city_pops,color="red",label="population in hundreds of thousands")
sns.barplot(bar_data["city_state"],bar_data["Deaths By City"], ax=axes)
fig.legend()


# The graph above shows the top 10 cities by number of police shootings by city. The red line plot shows the population size in hundreds of thousands. It appears that the police shootings trend down as the population trends downwards. However, both Phoenix and Las Vegas seem to rank farily high relative to their population size.
# 
# Data for population sizes was obtained at the respective pages [here](https://datausa.io/).

# In[ ]:


#add a plot here that is the same as above except it splits by race.
highest_cities = bar_data["city_state"]
bar_data = ps_df.groupby(["city_state","race"], as_index=False).agg("count").filter(["city_state","race","id"]).query("city_state in @highest_cities")
bar_data["orig_order"] = [7,7,7,6,6,6,8,8,8,10,10,10,10,10,10,3,3,3,3,3,4,4,4,4,4,1,1,1,1,1,9,9,9,9,9,2,2,2,2,2,5,5,5,5]
bar_data = bar_data.sort_values(by="orig_order",ascending=True)
labels = bar_data["city_state"].unique().tolist()
A_data = get_city_race_values(bar_data,labels,"A")
B_data = get_city_race_values(bar_data,labels,"B")
H_data = get_city_race_values(bar_data,labels,"H")
N_data = get_city_race_values(bar_data,labels,"N")
O_data = get_city_race_values(bar_data,labels,"O")
W_data = get_city_race_values(bar_data,labels,"W")
xcoors = np.arange(1,len(labels)+1)
num_sub_bars = len(bar_data["race"].unique())
sub_bar_labels = ["A","B","H","N","O","W"]
xlb = "City and Race"
ylb = "Death Count"
title = "Deaths by Top 10 Cities and Race"

fig, axes = plt.subplots()
grouped_bars(fig,axes,labels,[A_data,B_data,H_data,N_data,O_data,W_data],xcoors,num_sub_bars,0.75,sub_bar_labels,xlb,ylb,title)


# I encourage the reader to compare these statistics with more information found at [data usa](https://datausa.io/) for any city or cities of your choosing. It might be interesting to compare demographics and number of shootings by race to see if the data are corresponding.
# 
# For example Los Angeles has large populations of both white (non-hispanic) and white (hispanic) which may account for the higher numbers associated with those demographics. On the other hand, the african american popluation is small in comparison which fails to account for the troubling numbers of shootings associated with african americans (2nd highest). Also the difference between hispanic and white is largely out of proportion with the difference in hispanic and white populations.

# ### Deaths with Fleeing

# In[ ]:


bar_data = ps_df.groupby("flee").agg("count")
plt.figure(figsize=(18,8))
sns.barplot(bar_data.index,bar_data["id"])


# In[ ]:


#add a plot here that is the same as above except it splits by manner of death
bar_data = ps_df.groupby(["flee","manner_of_death","armed_simple"],as_index=False).agg("count").filter(["flee","manner_of_death","armed_simple","id"]).rename(columns={"id":"count"})
labels = bar_data["flee"].unique()
shot_armed_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'armed'")["count"].values
shot_unarmed_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'unarmed'")["count"].values
shot_und_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'undetermined'")["count"].values
shot_and_tasered_armed_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'armed'")["count"].values
shot_and_tasered_unarmed_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'unarmed'")["count"].values
shot_and_tasered_ind_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'undetermined'")["count"].values
xcoors = np.arange(1,len(labels)+1)
n_sub_bars = len(bar_data["manner_of_death"].unique())
sub_labels = ["shot armed","shot unarmed","shot undetermined","shot tasered armed","shot tasered unarmed","shot tasered undetermined"]
xlb = "Flee and Manner of Death"
ylb = "Deaths Count"
title = "Deaths by Flee and Manner of Death"
fig, axes = plt.subplots()
grouped_bars(fig,axes,labels,[shot_armed_data, shot_unarmed_data,shot_und_data,shot_and_tasered_armed_data, shot_and_tasered_unarmed_data,shot_and_tasered_ind_data],xcoors,n_sub_bars,0.25,sub_labels,xlb,ylb,title)


# This graph may show the most troubling information in this report. With the assumption that tasering occurs before shooting&mdash;we hope there are no instances of tasering someone who has already been subdued by gunshot&mdash;this shows that there seems to be a **systematic recourse to use of deadly force before attempting less-lethal methods against unarmed persons**. The data are showing that, in cases when persons are unarmed, shooting is far more common that tasering and shooting&mdash;again, assuming that tasering indicates a less-lethal method was the initial attempt to subdue the subject.
# 
# The data show it may be necessary to take a closer look, on a case-by-case basis, for those instances where a person was unarmed but shot, without first using the taser, to determine if there is something that could be implemented at a systematic level to avoid unnecessary use of force.

# ### Deaths by Race and Manner

# In[ ]:


#as the title says. Just one plot.
bar_data = ps_df.groupby(["race","manner_of_death"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["race","manner_of_death","count"])
race_counts = ps_df.groupby("race").agg("count")["id"].values
labels = bar_data["race"].unique()
shot_data = ((bar_data.query("manner_of_death == 'shot'")["count"] / race_counts)*100).values
tasered_data = ((bar_data.query("manner_of_death == 'shot and Tasered'")["count"] / race_counts)*100).values
xcoors = np.arange(1,len(labels)+1)
n_sub_bars = len(bar_data["manner_of_death"].unique())
sub_labels = ["shot","shot and tasered"]
xlb = "Race and Manner of Death"
ylb = "Deaths Percent by Race"
title = "Deaths by Race and Manner"
fig, axes = plt.subplots()
grouped_bars(fig,axes,labels,[shot_data,tasered_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)


# ### Deaths with Signs of Mental Illness

# In[ ]:


bar_data = ps_df.groupby("signs_of_mental_illness").agg("count").rename(columns={"id":"count"})
fig = plt.figure(figsize=(18,8))
sns.barplot(["False","True"],bar_data["count"])
mi_factor = bar_data["count"][0] / bar_data["count"][1]


# The data here are showing that police shootings where mental-illness is a factor are actually quite frequent. This strongly suggests training in identifying and reacting to incidents where mental illness is a factor should be required for any responders.

# In[ ]:


#add a plot here, same as above, splits on flee.
bar_data = ps_df.groupby(["signs_of_mental_illness","flee"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","flee","count"])
labels = ["No Mental Illness","Has Signs of Mental Illness"]
car_data = (bar_data.query("flee == 'Car'")["count"] * [1,mi_factor]).values
foot_data = (bar_data.query("flee == 'Foot'")["count"] * [1,mi_factor]).values
not_flee_data = (bar_data.query("flee == 'Not fleeing'")["count"] * [1,mi_factor]).values
other_data = (bar_data.query("flee == 'Other'")["count"] * [1,mi_factor]).values
xcoors = np.arange(1,len(labels)+1)
n_sub_bars = len(bar_data["flee"].unique())
sub_labels = bar_data["flee"].unique().tolist()
xlb = "Mental Illness and Fleeing"
ylb = "Adj Deaths Count"
title = "Adjusted Deaths by Mental Illness and Fleeing"
fig, axes = plt.subplots()
grouped_bars(fig,axes,labels,[car_data,foot_data,not_flee_data,other_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)


# In[ ]:


#add a plot here, same as above, splits on flee, only unarmed.
tmp = ps_df.query("armed == 'unarmed'").groupby(["signs_of_mental_illness"], as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","count"])
mi_factor = tmp["count"][0]/tmp["count"][1]
bar_data = ps_df.query("armed_simple == 'unarmed'").groupby(["signs_of_mental_illness","flee"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","flee","count"])
labels = ["No Mental Illness","Has Signs of Mental Illness"]
car_data = (bar_data.query("flee == 'Car'")["count"] * [1,mi_factor]).values
foot_data = (bar_data.query("flee == 'Foot'")["count"] * [1,mi_factor]).values
not_flee_data = (bar_data.query("flee == 'Not fleeing'")["count"] * [1,mi_factor]).values
other_data = (bar_data.query("flee == 'Other'")["count"] * [1,mi_factor]).values
xcoors = np.arange(1,len(labels)+1)
n_sub_bars = len(bar_data["flee"].unique())
sub_labels = bar_data["flee"].unique().tolist()
xlb = "Mental Illness and Fleeing Unarmed"
ylb = "Adj Deaths Count"
title = "Adjusted Deaths by Mental Illness and Fleeing Unarmed"
fig, axes = plt.subplots()
grouped_bars(fig,axes,labels,[car_data,foot_data,not_flee_data,other_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)


# The data here and above were adjusted to scale those incidents where mental illness was a factor to account for difference in population. Scaling the data shows that current response methodologies seem to punish mental illness when the persons are not fleeing, even when they are unarmed. This increases the urgency of the need, mentioned above, to integrate mental health services into police response methodologies.
# 
# It is in the opinion of this analyst that any use of deadly force be carefully scrutinized and that justice swiftly answer to any misuse of force.
