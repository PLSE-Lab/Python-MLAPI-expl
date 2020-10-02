#!/usr/bin/env python
# coding: utf-8

# # Thank you 
# 
# for taking the time to view my notebook.  I hope you like it!  Please comment for any feedback you have.
# 
# I tried to make this notebook accessible to as much people as possible, which might mean that many comments will carry information you already know.  Also, if something didn't make sense to you (or I'm mistaken in some respect) please let me know!

# ---
# 
# <a id="toc"></a>
# # Table of Contents
# 
# <div hidden>I took this formating from sban's kernel https://www.kaggle.com/shivamb/homecreditrisk-extensive-eda-baseline-model</div>
# 
# 1.) [Imports](#imports)    
# 2.) [Load Data](#load)    
# 3.) [Introduction](#intro)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.1.) [Background](#background)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.2.) [Questionnaire](#questionnaire)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.3.) [Dataset](#dataset)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.4.) [Tracts](#tracts)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.5.) [Counties](#counties)    
# &nbsp;&nbsp;&nbsp;&nbsp; 3.6.) [States](#states)    
# 4.) [General Focus](#general)    
# &nbsp;&nbsp;&nbsp;&nbsp; 4.1.) [Missing values - County](#nan_county)    
# &nbsp;&nbsp;&nbsp;&nbsp; 4.2.) [Missing values - Tract](#nan_tract)    
# &nbsp;&nbsp;&nbsp;&nbsp; 4.3.) [Nation as a Whole](#whole_nation)    
# &nbsp;&nbsp;&nbsp;&nbsp; 4.4.) [Correlation Maps](#gen_corr)    
# 5.) [State Focus](#state)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.1.) [Population](#st_pop)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.2.) [Transportation](#st_trans)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.3.) [Commute time](#st_commute)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.4.) [Unemployment](#st_unemploy)   
# &nbsp;&nbsp;&nbsp;&nbsp; 5.5.) [Income by County](#count_income)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.6.) [Poverty by County](#count_pov)    
# &nbsp;&nbsp;&nbsp;&nbsp; 5.7.) [Work type by County](#count_work)   
# &nbsp;&nbsp;&nbsp;&nbsp; 5.8.) [Racial Population and Representation](#st_race)    
# 6.) [Numeric Focus](#numeric)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.1.) [Income](#nu_income)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.2.) [Commute times](#nu_commute)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.3.) [Unemployment v. Poverty](#unemply_pov)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.4.) [Poverty v. Income](#pov_income)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.5.) [Poverty v. Carpool](#pov_carpool)    
# &nbsp;&nbsp;&nbsp;&nbsp; 6.6.) [MeanCommute v. Transit](#commute_trans)    
# 7.) [Focus on Fun!](#fun)    
# &nbsp;&nbsp;&nbsp;&nbsp; 7.1.) [Gender imbalance](#fun_gender)    
# &nbsp;&nbsp;&nbsp;&nbsp; 7.2.) [Selected Counties](#fun_selected)    
# 8.) [Appendix](#appendix)    
# &nbsp;&nbsp;&nbsp;&nbsp; 8.1.) [A: Gender imbalance](#app_a)    
# 9.) [Resources](#resources)    
# 10.) [ACS Criticism](#criticism)    

# ---
# <a id="imports"></a>
# # [^](#toc) <u>Imports</u>

# In[21]:


### Data handling imports
import pandas as pd
import numpy as np

### Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Advanced plotting... Plotly
from plotly import tools
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# Statistics imports
import scipy, scipy.stats

# df.head() displays all the columns without truncating
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


# ### Styling helpers
# 
# I found out about the color class from this [Stack Overflow question](https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python) (thanks [Boubakr](https://stackoverflow.com/users/1770999/boubakr)!)

# In[22]:


# A short hand way to plot most bar graphs
def pretty_bar(data, ax, xlabel=None, ylabel=None, title=None, int_text=False):
    
    # Plots the data
    fig = sns.barplot(data.values, data.index, ax=ax)
    
    # Places text for each value in data
    for i, v in enumerate(data.values):
        
        # Decides whether the text should be rounded or left as floats
        if int_text:
            ax.text(0, i, int(v), color='k', fontsize=14)
        else:
            ax.text(0, i, round(v, 3), color='k', fontsize=14)
     
    ### Labels plot
    ylabel != None and fig.set(ylabel=ylabel)
    xlabel != None and fig.set(xlabel=xlabel)
    title != None and fig.set(title=title)

    
### Used to style Python print statements
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# ---
# <a id="load"></a>
# # [^](#toc) <u>Load data</u>

# In[23]:


county = pd.read_csv("../input/acs2015_county_data.csv")
tract = pd.read_csv("../input/acs2015_census_tract_data.csv")


# ### Remove all rows in tract with zero population
# 
# A population is essential for Census data.  Having looked at the tract data already, I noticed there are a sizeable number of rows completely missing a population.
# 
# Let's delete these rows because they carry no information

# In[4]:


before_N = len(tract)
tract = tract.drop(tract[tract.TotalPop == 0].index)
after_N = len(tract)

print("Number of rows removed with zero population: {}{}{}".format(color.BOLD, before_N - after_N, color.END))
del before_N, after_N


# ---
# <a id="intro"></a>
# # [^](#toc) <u>Introduction</u>

# <a id="background"></a>
# ### [^](#toc) Background
# 
# Every 10 years, the US governemt conducts a survey of the entire nation to understand the current distribution of the population.  Every citizen in the States receives a questionaire (see [questionaire below](#questionnaire)).  The idea of a Census has been since the county's founding with the first Census taken in 1790 under Secretary of State, Thomas Jefferson.
# 
# Around 1960, there began to be a greater demand for more data at regular intervals.  And after 45 years of discussion, planning, and allocation of funds, the US government expanded the Census Bureau to administer the American Community Survey (ACS).  However, there are a number of important differences in how the ACS and the 10-year Census are conducted.  The 10-year Census is required by everyone in the nation at the same time however the ACS is a rolling sample and sends out surveys to 295,000 addresses monthly (or 3.5 million per year). 
# 
# The purpose is the Census was originally to help update the Electoral College adjust to a moving population.  However, the role has since expand, with knowledge of populations shifts and distributations helping the US government allocate $400 billion in funds each year
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/a/a4/1940_Census_-_Fairbanks%2C_Alaska.jpg" style="width:350px"/>
# 
# "This 1940 Census publicity photo shows a census worker in Fairbanks, Alaska. The dog musher remains out of earshot to maintain confidentiality." - A nice Wikipedia caption
# 
# Original photo source: https://www.flickr.com/photos/fdrlibrary/6872102290/

# <a id="questionnaire"></a>
# ### [^](#toc) Questionnaire
# 
# The actual American Community Survey is a very long, meaning I can't comfortably fit it here.  Although, if you'd like to see it, you can find a [sample form](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2018/quest18.pdf) on the US Census website.
# 
# To get a rough idea of the questions they ask, below is the 2010 Census form ([pdf](https://www.census.gov/history/pdf/2010questionnaire.pdf))
# 
# <img src="https://www.e-education.psu.edu/natureofgeoinfo/sites/www.e-education.psu.edu.natureofgeoinfo/files/image/ch4_census2010.png" style="width:600px"/>

# <a id="dataset"></a>
# ### [^](#toc) Dataset
# 
# The data comes from Kaggle user [Muon Neutrino](https://www.kaggle.com/muonneutrino) who took this data from the DP03 and DP05 tables of the 2015 American Community Survey 5-year estimates.  I recommend just using [his data](https://www.kaggle.com/muonneutrino/us-census-demographic-data) as I found the American Fact Finder website slow and a bit hard to navigate.
# 
# The two tables have essentially the same information.  The data is collected in tracts which are subsections of a county while the county data is an accumulation of all the tract data.

# In[5]:


print("Shape of county", county.shape)
print("Shape of tract", tract.shape)
print("Columns", county.columns)
county.head()


# <a id="tracts"></a>
# ### [^](#toc) Tracts
# 
# The ACS is completed in Tracts.  One or many tracts make up a County however, one tract may not span multiple Counties.
# 
# To get a sense of scale, they can range from 3 to 53,812 people but are generally around 4,000 people. I realize that's a huge range, but hopefully the plots below will help!

# In[25]:


max_tract = tract.iloc[np.argmax(tract.TotalPop)][["CensusTract", "State", "County"]]
min_tract = tract.iloc[np.argmin(tract.TotalPop)][["CensusTract", "State", "County"]]

print("The most populated Tract is: {}{}, {}{}".format(color.BOLD, max_tract.County, max_tract.State, color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(tract.TotalPop), color.END))
print("The least populated Tract is: {}{}, {}{} ".format(color.BOLD, min_tract.County, min_tract.State, color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(tract.TotalPop), color.END))
print("The median number of people sampled in a Tract is: {}{}{}".format(color.BOLD, int(tract.TotalPop.median()), color.END))

### Plotting the different distributions
fig, axarr = plt.subplots(2, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Distribution of Tract populations", fontsize=18)

sns.distplot(tract.TotalPop, ax=axarr[0][0]).set(title="KDE Plot")
sns.violinplot(tract.TotalPop, ax=axarr[0][1]).set(title="Violin Plot")
sns.boxplot(tract.TotalPop, ax=axarr[1][0]).set(title="Box Plot")
sorted_data = tract.TotalPop.sort_values().reset_index().drop("index", axis=1)
axarr[1][1].plot(sorted_data, ".")
axarr[1][1].set_title("Tract Populations")
axarr[1][1].set_xlabel("Tract index (after sorting)")
axarr[1][1].set_ylabel("Population")
del sorted_data, min_tract, max_tract


# <a id="counties"></a>
# ### [^](#toc) <u>Counties</u>
# 
# A County is a political subdivison of a State.  There are varying levels of influence each county can exert.  Alaska for instance has very low political power in its counties (technically they call them boroughs there) and operates mostly on the state level, however large city-counties like Los Angeles and New York are very strong on the county level.
# 
# For the purposes of this notebook it may help to think of Counties as just a collection of people that belong to a state.

# In[22]:


county_pop = county.groupby(["State", "County"]).TotalPop.sum()
print("The most populated County is: {}{}{}".format(color.BOLD, ", ".join(np.argmax(county_pop)[::-1]), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(county_pop), color.END))
print("The least populated County is: {}{}{}".format(color.BOLD, ", ".join(np.argmin(county_pop)[::-1]), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(county_pop), color.END))
print("The median number of people living in a County is: {}{}{}".format(color.BOLD, int(county_pop.median()), color.END))


### Plotting the different distributions
fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("County overview", fontsize=18)

counties = sorted(county.groupby("State").County.agg(len))
x = np.linspace(1, len(counties), len(counties))
counties = pd.DataFrame({"x":x, "Counties": counties})
(
    sns.regplot(x="x", y="Counties", data=counties, fit_reg=False, ax=axarr[0])
       .set(xlabel="State index (after sorting)", ylabel="Number of counties", title="Number of counties (in each state)")
)

sns.violinplot(county.TotalPop, ax=axarr[1]).set(title="County populations")
del county_pop, counties, x


# <a id="states"></a>
# ### [^](#toc) <u>States</u>
# 
# Okay, we all know what a State it, but let's see some visualizations anyways

# In[24]:


state_pop = county.groupby("State").TotalPop.sum()

print("The most populated State is: {}{}{}".format(color.BOLD, np.argmax(state_pop), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(state_pop), color.END))
print("The least populated State is: {}{}{}".format(color.BOLD, np.argmin(state_pop), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(state_pop), color.END))
print("The median number of people living in a State is: {}{}{}".format(color.BOLD, int(state_pop.median()), color.END))

### Plotting the different distributions
fig, axarr = plt.subplots(2, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Distribution of State populations", fontsize=18)

sns.distplot(state_pop, ax=axarr[0][0]).set(title="KDE Plot")
sns.violinplot(state_pop, ax=axarr[0][1]).set(title="Violin Plot")
sns.boxplot(state_pop, ax=axarr[1][0]).set(title="Box Plot")

axarr[1][1].plot(state_pop.sort_values().reset_index().drop("State", axis=1), ".")
axarr[1][1].set_title("State Populations")
axarr[1][1].set_xlabel("State index (after sorting)")
axarr[1][1].set_ylabel("Population")
del state_pop


# <a id="general"></a>
# # [^](#toc) <u>General</u>
# 
# Before we really start looking at the data, we need to see first if any preprocessing needs to be done.  This can take form in multiple ways such as incorrect data, removing outliers, formating errors, adding or removing columns.
# 
# I'd say universally, you want to look at the data to see if it's missing any values and drop or fill in those rows as necessary.  Let's look at both our dataframes for missing values

# <a id="nan_county"></a>
# ### [^](#toc) <u>Missing values - County</u>
# 
# There doesn't seem to be anything serverely wrong with these missing values.  Both rows are only considering around 100 people and are complete for the most part.  Let's move on without filling in or dropping any rows.

# In[10]:


missing_cols = [col for col in county.columns if any(county[col].isnull())]
print(county[missing_cols].isnull().sum())

# Look at rows with missing values
county[county.isnull().any(axis=1)]


# <a id="nan_tract"></a>
# ### [^](#toc) <u>Missing values - Tract</u>
# 
# Tract data appears to have a lot more missing values than the County data, so let's look into this a bit more.

# In[11]:


missing_cols = [col for col in tract.columns if any(tract[col].isnull())]
print(tract[missing_cols].isnull().sum())

# Look at rows with missing values
tract[tract.isnull().any(axis=1)].head()


# ### A deeper look at missing values in Tract data
# 
# It appears most of the very small tract samples have missing values.  This is definitely not conclusive, but it does offer a suggestion.  It's possible that survey responders only put the number of people in their household and leave everything else blank.  If that is the case, Census workers may create a new tract for people that refuse to answer questions.  There are many people like this in the United States as [this video](https://www.youtube.com/watch?v=bYwdOxOBwgM) shows.
# 
# Should more measures be taken with this data?  I'd argue no, the county dataset contains almost exactly the same information.  I'll use the county dataset mostly and look at the tract dataset as needed

# In[12]:


tract.sort_values("TotalPop").head(20)


# <a id="whole_nation"></a>
# ### [^](#toc) <u>Nation as a Whole</u>
# 
# I include both Tract and County data for the dual purpose of validating that they are equal

# In[13]:


pd.DataFrame({
    "Population": [tract.TotalPop.sum(), county.TotalPop.sum()],
    "Women": [tract.Women.sum(), county.Women.sum()],
    "Men": [tract.Men.sum(), county.Men.sum()],
    "Citizens": [tract.Citizen.sum(), county.Citizen.sum()],
    "States": [len(tract.State.unique()), len(county.State.unique())],
    "Counties": [len(tract.groupby(["State", "County"])), len(county.groupby(["State", "County"]))],
    "Employed": [tract.Employed.sum(), county.Employed.sum()],
}, index=["Tract data", "County data"])


# <a id="gen_corr"></a>
# ### [^](#toc) <u>Correlation Maps</u>
# 
# This looks at the relationship of every variable to every other variable.  There's a lot of information that can be gathered from these plots.
# 
# <strong>NOTE: Since there are so many columns, I split the map into three.</strong>

# In[14]:


fig, axarr = plt.subplots(3, 1, figsize=(16, 42))
data = county.drop("CensusId", axis=1).corr()

sns.heatmap(data.head(12).transpose(), annot=True, cmap="coolwarm", ax=axarr[0])
sns.heatmap(data.iloc[12:21].transpose(), annot=True, cmap="coolwarm", ax=axarr[1])
sns.heatmap(data.tail(13).transpose(), annot=True, cmap="coolwarm", ax=axarr[2])
del data


# ---
# 
# <a id="state"></a>
# # [^](#toc) <u>State</u>
# 
# ### Can two states have the same county name?
# 
# The answer is a resounding, patriotic yes

# In[15]:


dup_counties = (county
 .groupby("County")
 .apply(len)
 .sort_values(ascending=False)
)
dup_counties.where(dup_counties > 1).dropna()


# <a id="st_pop"></a>
# ### [^](#toc) <u>Population</u>
# 
# The results seem to line up nicely with Wikipedia's page ([link](https://en.wikipedia.org/wiki/County_statistics_of_the_United_States#Nationwide_population_extremes)), this is to be expected since the Wikipedia is using the 2016 estimate provided by the US Census bureau.
# 
# To others that may not know...
# - Cooks county in Illnois is home to Chicago
# - Harris, Texas contains Houston
# - Maricopa, Arizona contains Phoenix
# 
# #### Fun facts!
# 
# - If the largest county, Los Angeles, were to become it's own State, it be the 8th most populous State.
# - Kalawao County, Hawaii has no elected government and until 1969 it was used to quarantined people with leprosy

# In[16]:


##### County Plots

fig, axarr = plt.subplots(1, 2, figsize=(16,6))
fig.subplots_adjust(wspace=0.3)
fig.suptitle("Population extremes in Counties", fontsize=18)

county_pop = county.groupby(["State", "County"]).TotalPop.median().sort_values(ascending=False)

pretty_bar(county_pop.head(10), axarr[0], title="Most populated Counties")
pretty_bar(county_pop.tail(10), axarr[1], title="Least populated Counties")
plt.show()

##### State Plots

fig, axarr = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Total population in all 52 states", fontsize=18)

state_pops = county.groupby("State")["TotalPop"].sum().sort_values(ascending=False)

pretty_bar(state_pops.head(13), axarr[0][0], title="Largest population")
pretty_bar(state_pops.iloc[13:26], axarr[0][1], title="2nd Largest population", ylabel="")
pretty_bar(state_pops.iloc[26:39], axarr[1][0], title="2nd Smallest population")
pretty_bar(state_pops.tail(13), axarr[1][1], title="Smallest population", ylabel="")
del county_pop, state_pops


# <a id="st_trans"></a>
# ### [^](#toc) <u>Transportation</u>
# 
# #### By County
# 
# Some things to note:
# 
# - People don't drive in New York City, but they do take transit
# - Clay, Georgia has the highest rate of carpooling
# - Alaska likes to walk and use "OtherTransp" which I like to think as sea planes.  We will soon see that Alaska has the lowest mean commute time as well

# In[24]:


transportations = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']

datas = []
for tran in transportations:
    datas.append(county.groupby(["State", "County"])[tran].median().sort_values(ascending=False).head(10))

traces = []

for data in datas:
    traces.append(go.Box(
                            x=data.index,
                            y=data.values, 
                            showlegend=False
                        ))
buttons = []

for i, tran in enumerate(transportations):
    visibility = [i==j for j in range(len(transportations))]
    button = dict(
                 label =  tran,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': 'Top counties for {}'.format(tran)}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

layout = dict(title='Counties with most popular transportation methods', 
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=traces, layout=layout)

iplot(fig, filename='dropdown')


# #### By State
# 
# Some things to note:
# - Georgia is close to California in terms of carpooling

# In[25]:


transportations = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']

datas = []
for tran in transportations:
    county["trans"] = county.TotalPop * county[tran]
    data = county.groupby("State")["trans"].sum() / county.groupby("State")["TotalPop"].sum()
    datas.append(data.sort_values(ascending=False))

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('1st Quartile', '2nd Quartile',
                                                          '3rd Quartile', '4th Quartile'))

for i in range(4):
    for data in datas:
        start_i = 13 * i
        end_i   = start_i + 13
        
        trace = go.Bar(
                        x=data.iloc[start_i: end_i].index,
                        y=data.iloc[start_i: end_i].values, 
                        showlegend=False
                    )
        
        row_num = 1 + (i // 2)
        col_num = 1 + (i % 2)
        fig.append_trace(trace, row_num, col_num)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(transportations):
    visibility = [i==j for j in range(len(transportations))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

fig['layout']['title'] = 'Transportation across the states'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

# Remove created column
county = county.drop("trans", axis=1)
del transportations

iplot(fig, filename='dropdown')


# <a id="st_commute"></a>
# ### [^](#toc) <u>Commute time</u>
# 
# Wow!  Look how short Alaska's commute tends to be!

# In[26]:


fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(wspace=0.8)

commute = county.groupby(["State", "County"])["MeanCommute"].median().sort_values(ascending=False)

pretty_bar(commute.head(20), axarr[0], title="Greatest commute times")
pretty_bar(commute.tail(20), axarr[1], title="Lowest commute times")
del commute


# <a id="st_unemploy"></a>
# ### [^](#toc) <u>Unemployment</u>
# 
# I never knew Puerto Rico is suffering so much.  We'll see later that Puerto Rico has the lowest Income and the highest rate of poverty too.
# 
# On the bright side, North Datoka has low unemployment because of fracking ([article link](http://fortune.com/north-dakota-fracking/)).
# 
# To refresh your understanding of Poe's law, see this [link](https://en.wikipedia.org/wiki/Poe%27s_law).

# In[27]:


##### County Plots

fig, axarr = plt.subplots(1, 2, figsize=(18,8))
fig.subplots_adjust(hspace=0.8)
fig.suptitle("Unemployment extremes in Counties", fontsize=18)

unemployment = county.groupby(["State", "County"])["Unemployment"].median().sort_values(ascending=False)

pretty_bar(unemployment.head(12), axarr[0], title="Highest Unemployment")
pretty_bar(unemployment.tail(12), axarr[1], title="Lowest Unemployment")
plt.show()

##### State Plots

fig, axarr = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Unemployment percentage in all 52 states", fontsize=18)

county["Tot_Unemployment"] = county.Unemployment * county.TotalPop
unemployment = county.groupby("State").Tot_Unemployment.sum() / county.groupby("State").TotalPop.sum()
unemployment = unemployment.sort_values(ascending=False)

pretty_bar(unemployment.head(13), axarr[0][0], title="1st Quartile")
pretty_bar(unemployment.iloc[13:26], axarr[0][1], title="2nd Quartile", ylabel="")
pretty_bar(unemployment.iloc[26:39], axarr[1][0], title="3rd Quartile")
pretty_bar(unemployment.tail(13), axarr[1][1], title="4th Quartile", ylabel="")

# Remove created column
county = county.drop("Tot_Unemployment", axis=1)
del unemployment


# <a id="count_income"></a>
# 
# ### [^](#toc) <u>Income in Counties</u>
# 
# As we saw above, Puerto Rico has the highest rate of Unemployment, so it is not too much of a surprise to see it's counties be some of the poorest in the nation.

# In[28]:


fig, axarr = plt.subplots(2, 2, figsize=(14,12))
fig.subplots_adjust(wspace=0.5)

county_income_per_cap = county.groupby(["State", "County"])["IncomePerCap"].median().sort_values(ascending=False)
county_income = county.groupby(["State", "County"])["Income"].median().sort_values(ascending=False)

pretty_bar(county_income_per_cap.head(10), axarr[0][0], title="Richest IncomePerCap Counties")
pretty_bar(county_income_per_cap.tail(10), axarr[0][1], title="Poorest IncomePerCap Counties", ylabel="")

pretty_bar(county_income.head(10), axarr[1][0], title="Richest Income Counties")
pretty_bar(county_income.tail(10), axarr[1][1], title="Poorest Income Counties", ylabel="")
del county_income, county_income_per_cap


# <a id="count_pov"></a>
# 
# ### [^](#toc) <u>Poverty by County</u>
# 
# Once again Puerto Ricans seem to be suffering, yet this issue only receieved attention after Hurricane Maria and has since been forgotten.
# 
# To those that don't know Puerto Rico is a bit different than the rest of the country.  It is not a State, it is a territory.  This means that Puerto Ricans are citizens of the United States, but do not receive any votes for the US Congress.  There has been favorable discussion towards statehood, however it doesn't look like this will happen soon.

# In[29]:


fig, axarr = plt.subplots(2, 2, figsize=(14,12))
fig.subplots_adjust(wspace=0.5)

poverty = county.groupby(["State", "County"])["Poverty"].median().sort_values(ascending=False)
child_poverty = county.groupby(["State", "County"])["ChildPoverty"].median().sort_values(ascending=False)

pretty_bar(poverty.head(10), axarr[0][0], title="Highest in Poverty")
pretty_bar(poverty.tail(10), axarr[0][1], title="Lowest in Poverty", ylabel="")

pretty_bar(child_poverty.head(10), axarr[1][0], title="Highest in Child Poverty")
pretty_bar(child_poverty.tail(10), axarr[1][1], title="Lowest in Child Poverty", ylabel="")
del poverty, child_poverty


# <a id="count_work"></a>
# 
# ### [^](#toc) <u>Work type by County</u>
# 
# There are several variables in the ACS estimate that can be grouped together.  I decided to group together the variables 'Professional', 'Service', 'Office', 'Construction', 'Production' all as careers.  Then I decided to put the variables 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork' all as sectors.
# 
# Some things I noticed
# 
#  - The highest fraction of people in a 'Professional' career live in Fall Church City, Virginia.  This is the same county with the highest income.
#  
#  - 'PublicWork' highest scorers include Kalawao - Hawaii, the smallest county with 85 people, and Lassen - California, a county with 5 national parks.
#  
#  - Counties with the lowest portion of Construction jobs include New York City, San Francisco, and Washington D.C.  This is a good sanity check as they are already developed and any level of developement jobs are diluted by their large population.

# In[30]:


sectors = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork']

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Highest', 'Lowest',))

datas = []
for sector in sectors:
    data = county.groupby(["State", "County"])[sector].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={sector: "Values"})
    datas.append(data)
    
for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)

for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(sectors):
    visibility = [i==j for j in range(len(sectors))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Sectors'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus

iplot(fig, filename='dropdown')


# In[31]:


careers = ['Professional', 'Service', 'Office', 'Construction', 'Production']

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Highest', 'Lowest',))

datas = []
for career in careers:
    data = county.groupby(["State", "County"])[career].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={career: "Values"})
    datas.append(data)
    
for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)

for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(careers):
    visibility = [i==j for j in range(len(careers))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Careers'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus

iplot(fig, filename='dropdown')


# <a id="st_race"></a>
# ### [^](#toc) <u>Racial Population and Representation</u>

# In[33]:


################ Setup ################

#### Create new column: total population for each race

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

for race in races:
    county[race + "_pop"] = (county[race] * county.TotalPop) / 100


# #### By County
# 
# Cook, Illinois (the county with Chicago) has the highest Black Population and the 4th highest Hispanic Population.  Also LA seems to top the charts in terms of diversity

# In[34]:


races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Highest Population',     'Lowest Population',
                                                          'Highest Representation', 'Lowest Representation'))

###################################### Population ######################################

datas = []
for race in races:
    data = county.groupby(["State", "County"])[race + "_pop"].sum().map(int).sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={race + "_pop": "Values"})
    datas.append(data)
    

for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)
    
for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)
    
###################################### Representation ######################################

datas = []
for race in races:
    data = county.groupby(["State", "County"])[race].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={race: "Values"})
    datas.append(data)

for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 2, 1)
    
for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 2, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(races):
    visibility = [i==j for j in range(len(races))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Racial Population in Counties'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

iplot(fig, filename='dropdown')


# #### By State

# In[35]:


races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

datas = []
for race in races:
    data = county.groupby("State")[race + "_pop"].sum().sort_values(ascending=False)
    datas.append(data)

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('1st Quartile', '2nd Quartile',
                                                          '3rd Quartile', '4th Quartile'))

for i in range(4):
    for data in datas:
        start_i = 13 * i
        end_i   = start_i + 13
        
        trace = go.Bar(
                        x=data.iloc[start_i: end_i].index,
                        y=data.iloc[start_i: end_i].values, 
                        showlegend=False
                    )
        
        row_num = 1 + (i // 2)
        col_num = 1 + (i % 2)
        fig.append_trace(trace, row_num, col_num)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(races):
    visibility = [i==j for j in range(len(races))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

fig['layout']['title'] = 'Racial Population in all 52 States'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

iplot(fig, filename='dropdown')


# In[36]:


#### Remove created variables

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
county = county.drop([race + "_pop" for race in races], axis=1)

del races, datas, fig, buttons


# ---
# <a id="numeric"></a>
# # [^](#toc) Numeric Focus
# 
# 
# ### Quick look
# 
# Just by looking at 6 columns we can see a few things:
# 
# - Growing unemployment leads to poverty
# - Poverty decreases with higher income
# - Poverty encourages people to carpool
# - Commuties that take Transit have longer commutes
# 
# However, we don't have any hard numbers yet, we'll draw conclusions after a closer look

# In[ ]:


numeric_cols = ['Poverty', 'Transit', 'IncomePerCap', 'MeanCommute', 'Unemployment', "Carpool"]

sns.pairplot(county[numeric_cols].sample(1000))
del numeric_cols


# <a id="unemply_pov"></a>
# ### [^](#toc) Unemployment v. Poverty
# 
# There clearly is a very strong relationship here.  Our intuition is further bolstered by p = 0 indicating low probability for the null hypothesis (the two variables are connected in some way).  In addition, the Pearson correlation coeffient is close to 1 meaning the variables have a positive relationship (larger values of Unemployment are more likely with higher values of Poverty).
# 
# This is just what we'd expect from a relationship between Poverty and Unemployment

# In[43]:


sns.jointplot(x='Unemployment', y='Poverty', data=county, kind="reg")
_ = sns.jointplot(x='Unemployment', y='Poverty', data=county, kind='kde')


# <a id="pov_income"></a>
# ### [^](#toc) Poverty v. Income
# 
# Another very strong relationship and its not too surprising.  Once again p = 0 so the relationship is very likely and Pearson's R is close to -1 showing a negative relationship (low poverty tends to correlate with high income).

# In[44]:


sns.jointplot(x='Poverty', y='Income', data=county, kind="reg")
_ = sns.jointplot(x='Poverty', y='Income', data=county, kind="kde")


# <a id="pov_carpool"></a>
# ### [^](#toc) Poverty v. Carpool
# 
# There seems to be no evidencefor a relationship between these two variables.  Pearson's R is very near zero in this case making a relationship unlikely.  The plots look like a skewered ball which isn't a good sign.
# 
# However, more data would be nice (especially for high levels of Carpooling).  The bottom graph does suggest that there might be a slight relationship between higher Poverty leading to higher Carpooling.

# In[45]:


sns.jointplot(x='Poverty', y='Carpool', data=county, kind="reg")
_ = sns.jointplot(x='Poverty', y='Carpool', data=county, kind="kde")


# <a id="commute_trans"></a>
# ### [^](#toc) MeanCommute v. Transit
# 
# 
# This plot is really helpful, from the pairplot I suspected a bigger relationship than what I see here.
# 
# The lower plot (the KDE one) seems to strongly show that there is no relationship between mean commute and the amount of people that transit.  However, the top graph clearly shows that all points with high Transit values also have high Commute times.  We'll look at this relationship [again](#nu_comute) and I'll conclude that there is a correlation.

# In[46]:


sns.jointplot(x='MeanCommute', y='Transit', data=county, kind="reg")
_ = sns.jointplot(x='MeanCommute', y='Transit', data=county, kind="kde")


# <a id="nu_income"></a>
# ### [^](#toc) <u>Income</u>
# 
# #### Setup

# In[47]:


high = county[county.Income > 80000]
mid  = county[(county.Income < 80000) & (county.Income > 32000)]
low  = county[county.Income < 32000]

print("Number of low income counties: {}{}{}".format(color.BOLD, len(low), color.END),
      "  Number of middle income counties: {}{}{}".format(color.BOLD, len(mid), color.END),
      "  Number of high income counties: {}{}{}".format(color.BOLD, len(high), color.END))


# #### Plots
# 
# The first two plots are simply to get an idea of how income is distributed.  The other two plots are a bit more exciting.
# 
# It's interesting to see that hardly anyone describes their career as 'FamilyWork'.  Notice also in the "Career Distribution" plot how the fraction of people in 'professional' careers grows with higher income while the number of people in 'Service' and 'Construction' shrink

# In[48]:


#########################   Income Distribution Plots   #########################

fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

income = county.groupby(["State", "County"])["Income"].median().sort_values().values
axarr[0].plot(income)
axarr[0].set(title="Sorted Incomes", xlabel="County index (after sorting)", ylabel="Income")

(
        county
            .groupby(["State", "County"])["Income"]
            .median()
            .sort_values()
            .plot(kind="kde", ax=axarr[1])
            .set(title="KDE plot of income", xlabel="Income")
)
plt.show()

#########################   Career Type Plots   #########################

works = [ 'Professional', 'Service', 'Office', 'Construction','Production']

pd.DataFrame({
    "Small income (< $32,000)":  low[works].sum(axis=0) / low[works].sum(axis=0).sum(),
    "Mid income":  mid[works].sum(axis=0) / mid[works].sum(axis=0).sum(),
    "High income (> $80,000)": high[works].sum(axis=0) / high[works].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of workers", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Career distribution", fontsize=18)
plt.show()

#########################   Career Sector Plots   #########################

works = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork']

pd.DataFrame({
    "Small income (< $32,000)":  low[works].sum(axis=0) / low[works].sum(axis=0).sum(),
    "Mid income":  mid[works].sum(axis=0) / mid[works].sum(axis=0).sum(),
    "High income (> $80,000)": high[works].sum(axis=0) / high[works].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of workers", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Sector distribution", fontsize=18)
del high, mid, low, income, works


# <a id="nu_comute"></a>
# ### [^](#toc) <u>Commute times</u>
# 
# #### Setup

# In[49]:


high = county[county.MeanCommute > 32]
mid = county[(county.MeanCommute < 32) & (county.MeanCommute > 15)]
low  = county[county.MeanCommute < 15]
print("Number of short commutes: {}{}{}".format(color.BOLD, len(low), color.END),
      "  Number of average commutes: {}{}{}".format(color.BOLD, len(mid), color.END),
      "  Number of long commutes: {}{}{}".format(color.BOLD, len(high), color.END))


# #### Plots
# 
# Similar story as the Income, the first two plots are just to get a sense of distribution.
# 
# It appears that most people who walk to work have short commutes.  In addition, the fraction of people that take Transit grows for longer Commute times.  This is a very good result as it helps solidify a relationship proposed by '[MeanCommute v. Transit](#commute_trans)' above.

# In[50]:


#########################   Commute Distribution Plots   #########################

fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

commute_times = county.groupby(["State", "County"])["MeanCommute"].median().sort_values().values
axarr[0].plot(commute_times)
axarr[0].set(title="Sorted Commute times", xlabel="County index (after sorting)", ylabel="Commute time (min)")

_ = (
        county
            .groupby(["State", "County"])["MeanCommute"]
            .median()
            .sort_values()
            .plot(kind="kde", ax=axarr[1])
            .set(title="KDE plot of commute times", xlabel="Commute time (min)", xlim=(0,60))
)
plt.show()

#########################   Commute Transportation Plots   #########################

trans = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', "WorkAtHome"]

pd.DataFrame({
    "Short commutes (< 15min)":  low[trans].sum(axis=0) / low[trans].sum(axis=0).sum(),
    "Medium commutes":  mid[trans].sum(axis=0) / mid[trans].sum(axis=0).sum(),
    "Long commutes (> 32min)": high[trans].sum(axis=0) / high[trans].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of commuters", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Commute time", fontsize=18)
del high, mid, low, commute_times, trans


# <a id="fun"></a>
# # [^](#toc) <u>Focus on Fun!</u>
# 
# ### Longest and Shortest County names
# 
# Not much too say about this first script, hopefully this as fun for you too!

# In[51]:


longest_county_name_on_census_dataset_index = np.argmax(county.County.map(len))
s_i = np.argmin(county.County.map(len))

county[(county.index == longest_county_name_on_census_dataset_index) | (county.index == s_i)]


# ### Largest 'Income' and 'IncomePerCap' errors
# 
# I wonder if this indicates high income inequality

# In[52]:


max_income_err  = county[county.IncomeErr == max(county.IncomeErr)]
max_income_place = (max_income_err.County + ", " + max_income_err.State).sum()

max_per_cap_err = county[county.IncomePerCapErr == max(county.IncomePerCapErr)]
max_per_cap_place = (max_per_cap_err.County + ", " + max_per_cap_err.State).sum()

print("The County with the biggest income error is: {}{}{}".format(color.BOLD, max_income_place, color.END),
      "with an error of:", color.BOLD, "$" + str(max_income_err.IncomeErr.median()), color.END)
print("The County with the biggest income per cap error is: {}{}{}".format(color.BOLD, max_per_cap_place, color.END),
      "with an error of:", color.BOLD, "$" + str(max_per_cap_err.IncomeErr.median()), color.END)
del max_income_err, max_income_place, max_per_cap_err, max_per_cap_place


# <a id="fun_gender"></a>
# ### [^](#toc) <u>Gender imbalance</u>
# 
# #### General comments
# 
# The demographics I see on Wikipedia do not match what I see here.  Norton, Virginia says there are 0.818 men to every women ([link](https://en.wikipedia.org/wiki/Norton,_Virginia)) not 0.682 men; Forest, Pennslyvannia says there are 1.112 men to every women ([link](https://en.wikipedia.org/wiki/Forest_County,_Pennsylvania)) not 2.73 men; Sussex, Virginia says there are 1.351 men to every women ([link](https://en.wikipedia.org/wiki/Sussex_County,_Virginia)) not 2.177.
# 
# This can be the result of two things:
# 
# (1) All of these counties are on the lower side of population (around 5,000 people), so small flucuations in gender populations can account for some of these results.  In addition the ACS only samples 1-in-480 households a month.  The low sampling rate in addition to the small populations could result in high errors.
# 
# (2) However it could also be the result of single gender prisons.  This would explain both why the demographic discrepancy on Wikipedia and why the population would be so low.
# 
# #### High Men to Women
# 
# Forest - Pennslyvania ([link](https://en.wikipedia.org/wiki/State_Correctional_Institution_%E2%80%93_Forest)), Bent - Colorado ([link](http://bentcounty.org/2010/11/bent-county-correctional-facility-cca/)), and Sussex - Virginia ([prison 1](https://en.wikipedia.org/wiki/Sussex_I_State_Prison)) ([prison 2](https://en.wikipedia.org/wiki/Sussex_II_State_Prison)) all have prisons within the county so maybe I'm onto something.  Also, I have been able to confirm that one of Sussex's prisons and Forest's prison are all male.
# 
# #### Low Men to Women
# 
# Pulaski, Georgia contains an all women prison ([link](http://www.dcor.state.ga.us/GDC/FacilityMap/html/S_50000214.html)).
# 
# In addition to women prisons, I expect any population on the older side will skew toward more women since women tend to live longer than men.
# 
# #### Fun facts!
# 
# - Chillicothe, county seat of Livingston - Missouri, was the first place to introduce commericial bread slicing.  And has since led the standard in "best thing"
# - The Lorton, Virginia Wikipedia page has a section for [famous prisoners](https://en.wikipedia.org/wiki/Lorton_Reformatory)

# In[53]:


county["Men to women"] = county.Men / county.Women
men_to_women = county.groupby(["County", "State"])["Men to women"].median().sort_values(ascending=False)

fig, axarr = plt.subplots(1, 2, figsize=(18,8))
fig.subplots_adjust(wspace=0.3)

pretty_bar(men_to_women.head(10), axarr[0], title="Men to Women")
pretty_bar(men_to_women.tail(10), axarr[1], title="Men to Women")
del men_to_women


# <a id="fun_selected"></a>
# ## [^](#toc) <u>Selected Counties</u>
# 
# Finally to end this data exploration I included a chance to compare places you've lived!  I've been in the Bay Area most of my life so I choose some counties that I was interested to compare.

# In[54]:


################  Configure me!!  ################

state = "California"

##################################################

print("{}{}NOTE{}{}: This is just to help you explore different counties{}"
      .format(color.UNDERLINE, color.BOLD, color.END, color.UNDERLINE, color.END))

county[county.State == state].County.unique()


# #### Setup
# 
# Change the configure me here to look at any county you desire!

# In[55]:


################  Configure me!!  ################

counties = [("Santa Clara", "California"),   ("San Diego", "California"),
            ("Monterey", "California"),      ("Alameda", "California"),
            ("San Francisco", "California"), ("Contra Costa", "California"),
            ("Los Angeles", "California"),   ("Fresno", "California")]

##################################################


# In[56]:


commute, income, income_percap, men, women = ([],[],[],[],[])
hispanic, white, black, native, asian, pacific = ([],[],[],[],[],[])

def total_race(df, race):
    total_pop = df[race] * df.TotalPop
    frac_pop = (total_pop / 100).sum()
    return int(frac_pop)
    
for c, s in counties:
    curr_county = county[(county.County == c) & (county.State == s)]

    commute.append(curr_county.MeanCommute.median())
    men.append(   int(curr_county.Men.median())   )
    women.append( int(curr_county.Women.median()) )
    
    ### NOTE: These demographics are
    hispanic.append( total_race(curr_county, "Hispanic") )
    white.append(    total_race(curr_county, "White")    )
    black.append(    total_race(curr_county, "Black")    )
    native.append(   total_race(curr_county, "Native")   )
    asian.append(    total_race(curr_county, "Asian")    )
    pacific.append(  total_race(curr_county, "Pacific")  )
    income.append(curr_county.Income.median())
    income_percap.append(curr_county.IncomePerCap.median())

counties = pd.DataFrame({
                "Women": women, "Men": men, "Mean Commute": commute,
                "Hispanic": hispanic, "White": white, "Black": black,
                "Native": native, "Asian": asian, "Pacific": pacific,
                "IncomePerCap": income_percap, "Income": income
            }, index=counties)

counties["Men to women"] = counties.Men / counties.Women
del commute, income, income_percap, men, women, hispanic, white, black, native, asian, pacific
counties.head()


# #### Plots
# 
# A few things to note
# 
# - Contra Costa has cheaper housing than Alameda, SF, so it makes sense that it's commute is long (live in Contra Costra, commute to SF or Alameda)
# - I was surprised to learn LA has a shorter commute than SF
# - I included the gender ratios because at these population levels, they all should be centered at 1 (see [Appendix A](#app_a) for more analysis)
# - California appears to have a fair amount of racial diversity
# - SF population is a lot smaller than I expected
# - Income and IncomePerCap isn't surprising especially regarding SF, Santa Clara, and Fresno

# In[57]:


plt.figure(figsize=(16, 12))

### Nuanced way of creating subplots
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))

plt.suptitle(", ".join([c for c,s in counties.index]), fontsize=18)

pretty_bar(counties["Mean Commute"], ax1, title="Mean Commute")
pretty_bar(counties["Men to women"], ax2, title="Men to women")

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
counties[races].plot(kind="bar", title="Population make up", stacked=True, ax=ax3, rot=0)

pretty_bar(counties["IncomePerCap"], ax4, title="Income per capita")
pretty_bar(counties["Income"], ax5, title="Income")
del races


# ---
# 
# <a id="appendix"></a>
# # [^](#toc) <u>Appendix</u>

# <a id="app_a"></a>
# ### [^](#toc) 
# 
# <img src="http://patrickstetz.com/img/census_app_a.png" style="width:700px" />

# #### Try for yourself!

# In[58]:


################  Configure me!!  ################

selected_county = ("Monterey", "California")

##################################################


# In[59]:


# Gets the selected county from the data
selected_county = county[(county.State == selected_county[1]) & (county.County == selected_county[0])]

### Gets the total population and the number of men
n = selected_county.TotalPop.sum()
men = selected_county.Men.median()
women = selected_county.Women.median()

# Calculates the number of standard deivations
distance = abs((n / 2) - men)
sigma = distance / np.sqrt(n / 4)

# Get the probability distribution for a population this size
x = np.linspace(.4*n, .6*n, n+1)
pmf = scipy.stats.binom.pmf(x, n, 0.5)

### Plots the probability distribution and the actual value
plt.figure(figsize=(12, 6))
plt.plot(x, pmf, label="Gender PMF")
plt.axvline(men, color="red", label="Male Actual")
plt.axvline(women, color="k", label="Female Actual")

### Limits the plot to the only interesting sectiton
llim, rlim = n/2 - 1.2*distance, n/2 + 1.2*distance
plt.xlim(llim, rlim)

# Labels the plot
plt.title("{} - Ratio is {} $\sigma$ away".format(selected_county.County.iloc[0], round(sigma, 3)), fontsize=14)
plt.xlabel("Number of people")
plt.ylabel("Probability")
_ = plt.legend(frameon=True, bbox_to_anchor=(1.1, 1.05)).get_frame().set_edgecolor('black')


# # Thank you for making it so far!
# 
# This is my first kernel on Kaggle so please let, me know anything I did horribly wrong (or something you liked).

# <a id="resources"></a>
# # [^](#toc) <u>Resources</u>
# 
# - The American Community Survey website ([link](https://www.census.gov/programs-surveys/acs/))
# - The American Community Survey Information Guide ([link](https://www.census.gov/programs-surveys/acs/about/information-guide.html))
# - Here is the dataset for this project ([link](https://www.kaggle.com/muonneutrino/us-census-demographic-data)).  Thank you again to [MuonNeutrino](https://www.kaggle.com/muonneutrino) and [Kaggle](https://www.kaggle.com/)
# - PBS Newshour has a nice explanation of the Census along with concerns for the 2020 Census ([link](https://www.youtube.com/watch?v=1Y6PI3EtA54))

# <a id="criticism"></a>
# # [^](#toc) <u>ACS Criticism</u>
# 
# - To my fellow Americans: take a guess who would be first! ([link](https://www.lewrockwell.com/2004/07/ron-paul/its-none-of-your-business/))
# 
# - I'm worried how often people refuse to answer questions and how that affects data quality ([see example](https://www.youtube.com/watch?v=bYwdOxOBwgM)).
# 
# - The estimated budget for 2019 is $3.8 billion ([link to census budget](https://www2.census.gov/about/budget/2019-Budget-Infographic-Bureau-Summary.pdf)).
# 
# - Take a look at the [sample questionnaire](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2018/quest18.pdf) from 2018 and decide if the questons are too invasive.

# In[ ]:




