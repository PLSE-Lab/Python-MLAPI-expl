#!/usr/bin/env python
# coding: utf-8

# ![](https://www.elnacional.com/wp-content/uploads/2020/02/Kobe.jpg)
# 

# In[ ]:


import csv
import sklearn
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        f_format = filename.split(".")[1]
        fpath = os.path.join(dirname, filename)  
        if "csv" in f_format:
            df_allstar = pd.read_csv(fpath, sep=",")
      


# > ##INTRODUCTION 
# 
# Given my growing passion to the basketball game and the tragic passing of Kobe Bryant I saw a nice oportunity to get my hands into this dataset that I saw a few weeks ago containing some of the data collected for the AllStar games played between 2000 and 2016, both editions included. Although at a first glimpse there seems to be no abundance in data, I decided to give answer to the various questions that arised while having a first look at the data. Along these lines, the following sections or cells, show a chronological breakdown of questions that came up throughout the process and their corresponding attempts to provide a straight solution in pandas environment with one liners(if that expression exists, :). 
# 
# For that reason I tried to move from the old way programming habits with python towards progressively leveraging the possibilities that brings utilizing pandas methods and given that I have been lately involved in programming with MySQL I look forward to transfer knowledge from pandas to mysql and the other way around.
# 
# 
# As goals do stand out the further familiarization with built-in methods of pandas and the playing around with some visualization, placing a special emphasis on dynamic color mapping for graphs. With all, I modestly tried to openly raise questions having in mind that the more opened and spontaneous the question is, the more variability there could be in the tackling of the problem in a programmatic way. It would be nice for the sake of expanding the exploratory skills if anyone out there throws in some question. I would be up and ready to provide a programmatic way to get around the problem.

# In[ ]:


df_allstar.head()


# In[ ]:


df_allstar.info()


# > ## Data Preparation

# 
# A few changes, conversions and transformations follow up, namely : 
# 
# ***Data Cleaning, Preparation***  -> "HT" and "NBA Draft Status" where data is stored as "str"(object). Providing that those values are likely to be part of a computation where              aggregation is required we turned them into floats. Notice how we make use of regex patterns to simplify the transformation process with                  .str.extract(). The returned series object is stored in three newly created columns ["NBA Draft Year", "NBA Draft Round", "NBA Draft Pick"].
# 
# 

# In[ ]:


#### Transforming Data one column at a time and expanding dimension space. 
#### regex works here nice to extract exactly the set of characters we wish to match. 
df_allstar["HT"] = df_allstar["HT"].str.replace("-", ".")
df_allstar["NBA Draft Year"] = df_allstar["NBA Draft Status"].str.extract(r"([0-9]{4})")
df_allstar["NBA Draft Round"] = df_allstar["NBA Draft Status"].str.extract(r"(?<=Rnd )(\d)(?=\sPick)").fillna(0)
df_allstar["NBA Draft Pick"] = df_allstar["NBA Draft Status"].str.extract(r"((?<=Pick )\d)").fillna(0)
df_allstar["NBA Drafted"] = df_allstar["NBA Draft Status"].apply(lambda x : 0  if "NBA" in x else 1)
df_allstar.drop(columns=["NBA Draft Status"])


# ***Data Cleaning, Preparation***  -> we convert those resulting "str" values into "int" or "float".

# In[ ]:


df_allstar["HT"] = df_allstar["HT"].astype(float)
df_allstar["NBA Draft Year"] = df_allstar["NBA Draft Year"].astype(int)
df_allstar["NBA Draft Round"] = df_allstar["NBA Draft Round"].astype(int)
df_allstar["NBA Draft Pick"] = df_allstar["NBA Draft Pick"].astype(int)


# 
# 
# ***Data Cleaning, Preparation***  -> Last but not least, we drop the original column from which we generated the three columns : [NBA Draft Year, NBA Round, NBA Pick] and check the results. 
# 

# In[ ]:


df_allstar = df_allstar.drop(columns="NBA Draft Status")


# In[ ]:


df_allstar.head()


# *** FROM QUESTIONS TO PANDAS ***
# 
# A bunch of questions streaming from a pure basketball lover offers plenty of bias, :), so don't be surprised if you come across questions that seem a bit off the scope of what imagined when dealing with this type of data. 

# 1 - **It's mamba who holds the record of allstar participations in those 16 years?**. I would assume he is in the top end
#     per number of participations but let's break it down player by player. 
#         
#     
#    Notice how .groupby() function in pandas allows to group data on an index basis. When given a list of columns, pandas finds unique instances of unique    values for this columns and crunches down the dataframe on which the operation is performed. 
# 
# 

# In[ ]:


df_participations = df_allstar.groupby(["Player"]).count().sort_values(by="Year", ascending=False)["Year"]
### Plotting in a pandas fashion.
df_participations.reset_index().rename(columns={"Year":"Nr.Participations"})


# ** 2. Average Weights Per Team and Year**

# In[ ]:


## We compute average weights per team and per year. We round the number and sort values in a descending order. Only the first 20
## instances(by weight) are printed.
dfAvgWeights = df_allstar.groupby(["Year", "Team"])["WT"].mean().round(2).astype(int).sort_values(ascending=False).reset_index()
dfAvgWeights


# 3 -** How much does it take to players since they get drafted until they make it to the AllStar.**
# 
#  

# In[ ]:



#  x["Year"].min()- x["NBA Draft Year"]** to get years from Draft to 1st AllStar
RoadToAllStar = df_allstar.groupby(["Player"]).apply(lambda x: x["Year"].min()- x["NBA Draft Year"].min())
df_final = pd.merge(df_allstar, RoadToAllStar.reset_index(), on="Player", how="inner")


#   * **Note 3 : In our temporary table we get a new column [0] gets appended along the horizontal axis. 

# In[ ]:


df_final.head()


# In[ ]:


df_allstar["DraftToAllStar"] = df_final.dropna()[0]
df_allstar = df_allstar.dropna()
df_allstar["DraftToAllStar"] = df_allstar["DraftToAllStar"].astype(int)


# In[ ]:


df_allstar.head()


# > ## A look at Guards in the Allstar

# ![](https://vignette.wikia.nocookie.net/mundodelbasket/images/8/8e/Allen_Iverson-_Top_10_Career_Plays/revision/latest/scale-to-width-down/340?cb=20170324121352&path-prefix=es)

# * Allen Iverson had a huge fan based support over the years. Guards get many of the accolades and so I want to deepen the analysis on players holding this position.

# 
# 
# ** 4.1. Allen Iverson ranks up there in the list but who is hand in hand?**.
# 

# In[ ]:


position = "G"  # filtering condition/clause
df_allstar.where(df_allstar["Pos"] == position).groupby(["Player"]).count().sort_values(by="Year" ,ascending=False)["Year"].reset_index()


# ** 4.2. Selection Type broken down for each player **

# In[ ]:


columns = df_allstar.columns.values
clause = (df_allstar["Pos"] == position) 
filtered_df = df_allstar.where(clause).dropna()
GroupingColumns = ["Player", "Selection Type"]
filtered_df.groupby(["Player", "Selection Type"]).count()["Year"].reset_index()


#   * 4.2 Now we reformulate the question, we want to see whose players and how many times were they selected under selection types.

# In[ ]:


GroupColumns = ["Selection Type", "Player"]
grouped_df = filtered_df.groupby(GroupColumns).count()["Year"].reset_index().rename(columns={"Year": "TotalNrYears"})
grouped_df


#  5.** What positions coaches tend to vote more in an Allstar?.**
# 

# In[ ]:


coach_selection = df_allstar.where(df_allstar["Selection Type"].str.contains("Coaches")).dropna()
coach_position_selection = coach_selection.groupby(["Pos"]).count().sort_values(by="Year", ascending=False)["Year"].reset_index().rename(columns={"Year":"Total_Selections"})
coach_position_selection


# Having focused primarily on MySQL Language over the last year in my career I tend to look at analysis on tabular data more and more as I do when I work with MySQL. The same way it would in this language pandas allows to concatenate a series of instructions, and compute in one go the desired instruction.
# 
# Needless to say that when this "query" gets very large its not recommendable to put all that syntax in one long line, instead breaking down the instructions into a few lines would be the way to go, for reusability and readibility. 
# 

#  6.** What other nationalities are present in the AllStar?.**
#         

# In[ ]:


df_allstar["Nationality"].value_counts().reset_index().rename(columns={"index": "Country"}).style.set_properties(**{"text-align": "left"})


# > ## Data Visualization #####
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# In[ ]:


df_allstar.head()


# > ## Drafted / Non-Drafted
# 

# In[ ]:


plt.clf()
data = df_allstar["NBA Drafted"].apply(lambda x : "Not Drafted" if x == 0 else "Drafted").value_counts().to_dict()

array = plt.pie(x=data.values(), labels=data.keys(), explode=(0.5, 0.2), autopct='%1.1f%%', shadow=True, startangle=180)
plt.title("Drafted vs Non-Drafted Players between 2000-2016")

wedges = array[0]
wedges[1].set_animated(True)


# > ## Draft Pick / Totals

# In[ ]:


plt.clf()

df = df_allstar["NBA Draft Pick"].value_counts().reset_index().rename(columns={"index": "Draft Pick", "NBA Draft Pick": "TotalNrPlayers"})
df = df[(df["Draft Pick"] != 0)]

scalars = df["Draft Pick"]
height = df["TotalNrPlayers"]

plt.bar(scalars, height=height, color="turquoise")

plt.title("Total Number of Players per Draft Pick")
plt.xlabel("Draft Pick", fontsize=15)
plt.ylabel("Total Number of Players", fontsize=15)
plt.xticks(list(range(scalars.min(), scalars.max()+1)), fontsize=10)
plt.yticks(fontsize=10)


# > ## Nationalities / Year

# In[ ]:


def subplot(years):

    df = df_allstar.groupby(["Year", "Nationality"])["Player"].count().reset_index().rename(columns={"Player":"TotalNrPlayers"})
    df = df[(df["Year"] >= min(years)) & (df["Year"] <= max(years))]
    
    fig = plt.figure(figsize=(75,30))
    plt.title("Players in the AllStar and their Nationalities({}-{})".format(min(years), max(years)), fontsize=80, pad=40)
    plt.ylabel("AllStar Edition", fontsize=80)
    plt.xlabel("Total Number of Players / Nationality", fontsize=80)
    #years = list(range(2000,2017))
    plt.xticks(years, fontsize=40)
    plt.yticks(list(range(0,30)), fontsize=40)

    ##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 
    import random 
    r = lambda: random.randint(0,255)

    #### Mapping Containers used afterwards as helper objects.
    color_map = { nation :'#%02X%02X%02X' % (r(),r(),r()) for nation in df["Nationality"].unique() }
    year_map = { year: i for i, year in enumerate(years) }


    j=0
    reference=np.zeros(len(years))

    for nation in df.sort_values(by="TotalNrPlayers",ascending=True).drop_duplicates(subset=["Nationality"])["Nationality"]:

        total_players = df[df["Nationality"]==nation][["Year", "TotalNrPlayers"]]
        KeepIndices = [ year_map[year] for year in total_players["Year"] ]

        stacked_values = [0 for i in range(0, len(years))]
        values = total_players["TotalNrPlayers"].values

        for i, index in enumerate(KeepIndices):
            stacked_values.insert(index, values[i])
            stacked_values.pop(index+1)

        if j>0:
            plt.bar(years, stacked_values, width=0.4, edgecolor='white', label=nation, bottom=reference, color=color_map[nation], align="center")
            reference += np.array(stacked_values) 
        else: 
            plt.bar(years, stacked_values, width=0.4, edgecolor='white', label=nation, color=color_map[nation], align="center")
            reference = np.array(stacked_values)


        j+=1

    plt.legend(fontsize=40, framealpha=0.85)

plt.tight_layout()
   


# In[ ]:



period = list(range(2000,2017))

for i in range(0,4):
    init = i*4
    end = init+4
    if init > 10:
        end+=1
        print(init,end)
    subplot(period[init:end])


# > ## Selection Type / Team

# Click Expand on the nex cell to see future some chunk of the logic to event handlers in matplotlib. 
# 

# ### Setting Plotting Styles via the Backend (Matplotlib)

# In[ ]:


plt.clf()

plt.rcdefaults()


SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# > ## Selection Type / Team
# 

# In[ ]:



df = df_allstar.groupby(["Team", "Selection Type"])["Year"].count().reset_index().rename(columns={"Year":"Total"})

TopTeams = df.groupby(["Team"]).sum()["Total"].sort_values(ascending=False).rename(columns={""}).reset_index()


fig = plt.figure(figsize=(25,15))
plt.title("Proportion of Selection Type per Team for the Period 2000-2016", fontsize=25, pad=50)
plt.ylabel("Totals", fontsize=25)
plt.xlabel("Teams", fontsize=25)
plt.yticks(list(range(0,50)))



##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 
import random 
r = lambda: random.randint(0,255)

#### Mapping Containers used afterwards as helper objects.
color_map = { selection_type :'#%02X%02X%02X' % (r(),r(),r()) for selection_type in df["Selection Type"].unique() }


teams = TopTeams["Team"].values
team_map = { team: i for i, team in enumerate(teams) }
reference = np.zeros(len(teams)+1)

selection_types = df["Selection Type"].unique()
df.groupby(["Team", "Selection Type"]).count()


j=0
for selection_type in df.sort_values(by="Total",ascending=True).drop_duplicates(subset="Selection Type")["Selection Type"]:
    
    team_totals = df[df["Selection Type"]==selection_type][["Team", "Total"]]
    KeepIndices = [ team_map[team] for team in team_totals["Team"] ]
    stacked_values = [0 for i in range(0, len(teams))]
    values = team_totals["Total"].values
    
    for i, index in enumerate(KeepIndices):
        stacked_values.insert(index, values[i])
        stacked_values.pop(index+1)

    if j>0:
        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=selection_type, bottom=reference, color=color_map[selection_type], align="center")
        reference += np.array(stacked_values) 
    else: 
        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=selection_type, color=color_map[selection_type], align="center")
        reference = np.array(stacked_values)
  

    j+=1
    
plt.xticks(teams, rotation=-75)
xlim = plt.xlim()
plt.legend()
    


# > ## Teams / Positions
# 
# 

# In[ ]:


df = df_allstar.groupby(["Team", "Pos"])["Year"].count().reset_index().rename(columns={"Year":"Total"})

TopTeams = df.groupby(["Team"]).sum()["Total"].sort_values(ascending=False).reset_index()


fig = plt.figure(figsize=(25,15))
plt.title("Proportion of Pos per Team for the Period 2000-2016", fontsize=25).set_position([.5, 1.05])
plt.ylabel("Totals", fontsize=25)
plt.xlabel("Teams", fontsize=25)
plt.yticks(list(range(0,50)))



##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 
import random 
r = lambda: random.randint(0,255)

#### Mapping Containers used afterwards as helper objects.
color_map = { pos_type :'#%02X%02X%02X' % (r(),r(),r()) for pos_type in df["Pos"].unique() }


teams = TopTeams["Team"].values
team_map = { team: i for i, team in enumerate(teams) }
reference = np.zeros(len(teams)+1)

pos_types = df["Pos"].unique()
df.groupby(["Team", "Pos"]).count()


j=0
for pos_type in df.sort_values(by="Total",ascending=True).drop_duplicates(subset="Pos")["Pos"]:
    
    team_totals = df[df["Pos"]==pos_type][["Team", "Total"]]
    KeepIndices = [ team_map[team] for team in team_totals["Team"] ]
    stacked_values = [0 for i in range(0, len(teams))]
    values = team_totals["Total"].values
    
    for i, index in enumerate(KeepIndices):
        stacked_values.insert(index, values[i])
        stacked_values.pop(index+1)

    if j>0:
        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=pos_type, bottom=reference, color=color_map[pos_type], align="center")
        reference += np.array(stacked_values) 
    else: 
        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=pos_type, color=color_map[pos_type], align="center")
        reference = np.array(stacked_values)
  

    j+=1
    
plt.xticks(teams, rotation=-75)
xlim = plt.xlim()
plt.legend()
plt.tight_layout()


# > ## Players and Teams in the AllStar

# 4.1 - We plot players against their total number of participations in the allstar. 
# 
#     Note : palettes from seaborn are utilized to enhance the visualization by grading to darker tones as the number of participations increase. 
#            df1 receives the computed dataframe result of grouping/aggregating/renaming/sorting. 
#            Additionally, we can customize the number of samples, that is, the number of players by selection one for each 3 players.

# In[ ]:


title1 = "Number of Participations per Player(2000-2016)"
fig1, ax1 = plt.subplots(figsize=(10,15))


###### Extracting label and scalar data and plotting 
df1 = df_allstar.groupby(["Player"])["Year"].count().reset_index().rename(columns={"Year": "Total(Years)"}).sort_values(by="Total(Years)", ascending=False)
 
n_subsampling = 3
blue_palette = sns.cubehelix_palette(n_colors=len(df1["Total(Years)"][::3]), start=0.2, rot=.7, reverse=True)

ax1.tick_params(pad=30)
plt.title(title1)
ax1.barh(df1["Player"][::3], df1["Total(Years)"][::3], height=0.7, color=blue_palette)

widths = []
patches = list(ax1.patches)
patches.reverse()
for p in patches:

    width = p.get_width()
    if width not in widths:
        height = p.get_y() + 0.2*p.get_height()
        ax1.annotate("{}".format(width), xy=(width, height), xytext=(width+0.1, height))
        widths.append(width)

plt.xticks([])


# 4.2 In a similar token we aim at visualizing the distribution between Participations "Total(Years)" and "Team" which is self-explanatory.

# In[ ]:


fig2, ax2 = plt.subplots(figsize=(10,15))


df2 = df_allstar.groupby(["Team"])["Year"].count().reset_index().rename(columns={"Year": "Total(Years)"}).sort_values(by="Total(Years)", ascending=False)
x2, y2 = df2["Total(Years)"], df2["Team"]

purple_palette = sns.cubehelix_palette(n_colors=len(y2), start=2.0, rot=.1, reverse=True)

ax2.tick_params(pad=30)
xticks = list(range(0,x2.max()))
plt.title("Total Number of Participations per Team (2000-2016)")
ax2.barh(y2, x2, height=0.7, color=purple_palette, edgecolor="white")



for p in ax2.patches:

    width = p.get_width()
    height = p.get_y() + 0.4*p.get_height()

    ax2.annotate("{}".format(width), xy=(width, height), xytext=(width+0.1, height))
    

plt.xticks([])
plt.gcf().set_facecolor('white')


# 
# 

# ## Takeaways
#   * Breaking columns into additional columns will not always provide more insights, but its worth a try. The cleaner data the better it gets later on. Its a must to spend a good deal of time on it.
#   
#   * Visualization using combination of back-ends is in the to-do list for future projects (ggplot,seaborn,plotly, matplotlib, etc). 
#   
#   * To perform operations in a single line, that is, concatenating calls to different functions might boost up times to analyze data faster and effectively, but this comes at the expense of a poorer reusability and readibility. Despite all of this, this type of programming might suit some cases where we want to analyze certain patterns. 
#   
#   * Correlation between values depends on the nature of variables themselves. Are these discrete, continuous or categorical?. If we count with categorical values and the number of levels that is the number of unique categories go over 2, then any correlation measure breaks down given that the whole nature of correlation is to measure lineal relationship between two variables. 
#   
#   * Use of lambda and map functions to support pandas based operations. 
#   
#   
#     
