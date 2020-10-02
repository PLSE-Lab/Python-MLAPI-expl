#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This analysis explores the world of Starcraft 2, an online massive multiplayer PC based game produced by Blizzard Entertainment.  The specific purpose of this notebook is to create models that predict League Index  (model target) based off certain player characteristics (potential model features).   It opens with a data dive for general exploration, points out some high level observations and concerns, and works up to creating and evaluating various predictive models.
# 
# **Additional Comments**  
#  - Thanks to *Simon Fraser University - Summit* for providing this data set  
#  - This is a work in progress primarily used to help share Python knowledge with a small group, so please excuse the pedagogical tone and multiple published updates as I continue to work in clearer explanations.  
#  - Please feel free to comment (even on snooty word choices like pedagogical), point out any goofs, and / or ideas!

# ## Data Exploration and Analysis

# In[ ]:


# Import packages
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # Deprecated
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Import dataset and see what type of data we're working with
df_raw = pd.read_csv('../input/starcraft.csv',sep=',')
df_raw.info()


# **Lots of floats and integers across 3.4k rows of data.  2 quick takeaways here:**
# 
#  - Required memory to analyze this data should be a non-issue
#  - The entirely numeric based data types should make for an easy look into the data without too much cleaning, but let's take a look at the some of the data and the spread within each attribute

# In[ ]:


# First 5 rows of data
df_raw.head()


# ####A previous look at this data showed some invalid values in the raw data, which were attributed to NaN values.  Here we drop those NaNs and see how many rows were affected.

# In[ ]:


df = df_raw.dropna()
df = df[df['LeagueIndex']!=8]

# Number of rows with incomplete data and / or with LeagueIndex value of 8
print("Number of incomplete data rows dropped: " + str(len(df_raw) - len(df)))
print("Remaining records = " + str(len(df)))


# ####Losing 57 rows on account of incomplete data when we started with 3.4k rows works for me.  Now that the data set is "clean" lets take look at the dependent variable that our models will predict, League Index.

# ----------

# ### League Index Player Distribution (Dependent Variable)
# ####How are players distributed across the 7 SC2 leagues?  A cursory look below.

# In[ ]:


# Select Style
plt.style.use('fivethirtyeight')

# Create Figure
fig, ax = plt.subplots(1,2, figsize = (14,8))
fig.suptitle('Starcraft 2 League Player Distribution', fontweight='bold', fontsize = 22)

# Specify histogram attributes
bins = np.arange(0, 9, 1)
weights = np.ones_like(df['LeagueIndex']) / len(df['LeagueIndex'])

# Count Histogram
p1 = plt.subplot(1,2,1)
p1.hist(df['LeagueIndex'], bins=bins, align='left') # Pure Count
plt.xlabel('League Index', fontweight='bold')
plt.title('Count')

# Percentage Histogram
p2 = plt.subplot(1,2,2)
p2.hist(df['LeagueIndex'], bins=bins, weights = weights, align='left') # % of Total (weights)
plt.xlabel('League Index', fontweight='bold')
plt.title('Percentage', )
yvals = plt.subplot(1,2,2).get_yticks()
plt.subplot(1,2,2).set_yticklabels(['{:3.1f}%'.format(y*100) for y in yvals])

plt.show()


# ####**Histogram Comments**
# The League Descriptions overview detailed here, 'http://wiki.teamliquid.net/starcraft2/Battle.net_Leagues', says:  
#  
# *"Players are placed in a league after having completed 5 placement matches. After that, a player may get moved to another league, depending on performance. Though the time and frequency of these movements are kept explicitly hidden. Regardless of a player's performance, however, placement matches currently do not place players in the highest league, Grandmaster. With even a perfect placement record, a player must work their way through the initial placement division(s) before being placed in Grandmaster."*
# 
# 
# If we couple that quoted context with the histograms directly above, it's clear why only about 1% of all League players are in the Grandmaster League (League Index 7).  These are the best of the best SC2 players, where even having a perfect undefeated record in a players first 5 placement matches doesn't guarantee Grandmaster designation.  
# 
# From a data perspective, the lack of a sample size (roughly 34 rows) might make it hard for our models to accurately predict Grandmaster Level players.  That said, I also wonder if there are any attributes that clearly differentiate Master (League Index 6) from Grandmaster League players.

# ----------

# ### Player Attributes (Independent Variables)
# #### Here we jump into the majority of the data, taking a helicopter view of some player attributes while digging deeper into others.  Leveraging the powerful Python describe method is how we jump in.  This also serves as a good double check that the remaining data points in the dataframe df have removed all non-helpful data points.

# In[ ]:


df.describe()


# ####**Immediate reactions:**
#  - I might consider removing **GameID** for later analyses as that data provides little to no context here.  The only value I feel it could bring is in a time series analysis.  The smaller the GameID, the earlier on the game was played.  Could be interesting to see player attributes changed over time across leagues.  
#  - I'm surprised at the 16 min and 44 max **Age** aren't more extreme.  Surely some 12 year old is playing competitive SC2 back Mom and Dad's back.  And with prize money in the millions, it's surprising to see there isn't some some 55 year old Grandmaster League player.  Grandfathermaster?
#  - A player logged 1 million **TotalHours**?  Seems a tad too much.  1 million hours / 24 hours in a day / 365 days in a year = some SC2 player gaming for 114.6 years.  That contradicts the 44 max player age.  This math is bonkers.  Or maybe I'm bonkers.  Need to look into this and potentially remove that 1 million TotalHours data point.
#  - I wish I had an average of 15.9 **HoursPerWeek** week to play Starcraft 2!  And DAMNIT I knew I should have used more hot keys when I played.
#  - I don't get the **WorkersMade, UniqueUnitsMade** metrics.  How do you make an average of 0.001031 workers?  I'm missing something here.

# **Stronger attribute metric context is sorely needed.**  
# 
# A refresh on those descriptions, taken from the Kaggle Starcraft II Replay Analysis page is below.
# 
# This dataset contains 21 variables:  
# - **GameID:** Unique ID for each game  
# - **LeagueIndex:** 1-8 for Bronze, Silver, Gold, Diamond, Master, GrandMaster, Professional leagues  
# - **Age:** Age of each player  
# - **HoursPerWeek:** Hours spent playing per week  
# - **TotalHours:** Total hours spent playing  
# - **APM:** Action per minute  
# - **SelectByHotkeys:** Number of unit selections made using hotkeys per timestamp  
# - **AssignToHotkeys:** Number of units assigned to hotkeys per timestamp  
# - **UniqueHotkeys:** Number of unique hotkeys used per timestamp  
# - **MinimapAttacks:** Number of attack actions on minimal per timestamp  
# - **MinimapRightClicks:** Number of right-clicks on minimal per timestamp  
# - **NumberOfPACs:** Number of PACs per timestamp  
# - **GapBetweenPACs:** Mean duration between PACs (milliseconds)  
# - **ActionLatency:** Mean latency from the onset of PACs to their first action (milliseconds)  
# - **ActionsInPAC:** Mean number of actions within each PAC  
# - **TotalMapExplored:** Number of 24x24 game coordinate grids viewed by player per timestamp  
# - **WorkersMade:** Number of SCVs, drones, probes trained per timestamp  
# - **UniqueUnitsMade:** Unique units made per timestamp  
# - **ComplexUnitsMade:** Number of ghosts, investors, and high templars trained per timestamp  
# - **ComplexAbilityUsed:** Abilities requiring specific targeting instructions used per timestamp  
# - **MaxTimeStamp:** Time stamp of game's last recorded event  
# 

# Tackling the **Age** attribute first.
# 
# My understanding is that each row records a particular players game stats for a specific SC2 League game, represented by GameID.  The **Age** description above confirms this too, but I'll trust this a bit more if the distinct age values are all whole numbers (suggesting it's not some average of players within the game).

# In[ ]:


pd.unique(df['Age'])


# Awesome, each row representing a single player and his gameplay metrics seems to check out.  
# 
# As far as the 0.001031 **WorkersMade** unit, it's clear this metric is per time stamp and captures a very specific moment in time.  My guess is it's down to the second.  So making only 0.001031 workers per second seems to make sense.  Same applies for a lot of the other **TimeStamp** based metrics.  
# 
# Time to investigate that 1 million **TotalHours** data point.

# In[ ]:


# Looking at that 1 million TotalHours row
df[df['TotalHours']==1000000]


# So an 18 year old kid has already played 1 million hours, which we previously calculated to be 114.6 years.  Riiiiiight.  Before throwing this data point out it's probably good to check that no other players have TotalHours (converted to years) that exceed their Age.

# In[ ]:


df_temp = df[['Age', 'TotalHours']].copy(deep=True)
df_temp['TotalHoursYears'] = df_temp['TotalHours'] / 24 / 365
df_temp['Age_Less_GP_Years'] = df_temp['Age'] - df_temp['TotalHoursYears']
df_temp.head()


# Time to see if any other data points show players with more SC2 playing years than they've been alive.

# In[ ]:


df_temp[df_temp['Age_Less_GP_Years']<0]


# Okay, great I'm not crazy and that 1 million data point needs to be removed.

# In[ ]:


# Removing that 1 million TotalHours row
df = df[df['TotalHours']!=1000000]
print('Remaining records in df= ' + str(len(df)))

# Deleting df_temp
del(df_temp)


# Time to start moving towards visualizations for more insight.  A quick setup of the columns we want to focus on is followed by some plots below.

# In[ ]:


# Create list of player attributes (potential features) for data analysis
lst_pf = list(df.columns)
lst_pf = lst_pf[2:20]
num_of_pf = len(lst_pf)
lst_counter = list(np.arange(1,num_of_pf + 1,1))
print(lst_pf)
print(lst_counter)


# In[ ]:


# Create and apply dictionary of League Index Names for future plot labels
dct_league_names = {1:'Bronze', 2:'Silver', 3:'Gold', 4:'Platinum', 5:'Diamond', 
                    6:'Master', 7:'Grandmaster'}
df['LeagueName'] = df['LeagueIndex'].map(dct_league_names)

# Check mapping was applied correctly
df[['LeagueIndex', 'LeagueName']].head(10)

pvt_num_lpt = df.pivot_table(index='LeagueName', values=['LeagueIndex'], aggfunc='count', 
                             margins=False)


# In[ ]:


# Setup graph formatting
plt.style.use('seaborn')
sns.set_palette("dark")
sns.set_style("whitegrid")

# Boxplots of the potential features
fig, axes = plt.subplots(num_of_pf, 1, sharex = True, figsize = (14,30))
fig.suptitle('Attribute Percentile Distributions', fontsize=22, 
             fontweight='bold')
fig.subplots_adjust(top=0.95)

for pf, c in zip(lst_pf, lst_counter):
    p = plt.subplot(num_of_pf,1,c) # (rows, columns, graph number)
    sns.boxplot(x = 'LeagueIndex', y = pf, data=df, showfliers=False) 
    # outliers excluded from plots given visual density, but data points have not been removed
    if c < num_of_pf: # remove xtick labels and xaxis title for all plots, excluding the last
        labels = [item.get_text() for item in p.get_xticklabels()]
        empty_string_labels = [''] * len(labels)
        p.set_xticklabels(empty_string_labels)
        p.set(xlabel='')
    if c== 1:
        p.set_title('Box and Whisker Plots\n', fontsize=16)
     
plt.show()


# **Observations**
# 
# Hope you have a giant and / or portrait oriented monitor to see all 18 plots!  If not a strong index finger to scroll down should do.  In either case, here are a few observations:
# 
#  1. **Age** of players across leagues is essentially the same.  Comments speaking to the wider (taller) interquartile range (IQR = the box) for the bronze league and higher median for the Grandmaster league could be made but I believe the HoursPerWeek attribute would be stronger "experience" attribute to think through.  
# 
#  2. **HoursPerWeek, TotalHours, APM, SelectByHotKeys, AssignToHotkeys, UniqueHotkeys** all suggest the same volume = player strength theme.   Every attribute (with the exception of the **GapBetweenPACs** and **ActionLatency**, where lower metrics translate to stronger game play) trends up and to the right across Leagues.
# 
#  3. It's worth noting that the **HoursPerWeek** attribute has the greatest IQR and median.  If a player is in the most competitive Grandmaster league, it makes sense that some of those players put in a lot of time to reach that level vs others that are "naturally" talented at the game (speaking to the wide range).  Regardless, Grandmaster level players do play more from a median perspective.  
# 
#  4. **NumberofPACs**, where a PAC stands for a *Perception Action Cycle* and is defined by it's wiki as "when one changes screen location and performs 1+ actions before changing screen location again to repeat."  So the more a player increases his visibility and performs game actions the more likely he is to be a better player.  Makes sense.  
# 
#  5. **GapBetweenPACs** works in the opposite direction where the fewer seconds that pass between PACs means a player is more active with his mouse and increasing his **NumberofPACs**.  I suspect we may run into some ***multicollinearity*** issues from this later, something to watch out for.
# 
# Additional commentary speaking to individual attributes to be added later.

# ### Attribute Analysis - UniqueUnitsMade  
# 
# I've separated the **UniqueUnitsMade** attribute out from the above because it is more interesting to me than any other at this early stage of analysis.  The reason here is because UniqueUnitsMade is the only attribute that reverses the clear trend you see from Bronze to Master league for other attributes, where the median and IQR all continue in the same direction.  
# 
# With **UniqueUnitsMade**, Grandmaster league players appear to create less unique units per time stamp than any other level- even less than Bronze level players!  My guess is that the amount of micromanagement required to control multiple unique units with different abilities outweighs the benefits that having that diversity.  In other words, being able to direct 20 of the same units and abilities may be more effective than 4 different groups of 5 units.  
# 
# Another thought is that Grandmaster league players may "rush" their opponents at the start of a game more frequently than players in other leagues.  Spending more time early on trying to find and attack your enemy vs building up a vast fleet of units and attacking them later in the game seems to agree with having less **UniqueUnitsMade.** The **TotalMapExplored** attribute seems to support this too.
# 
# However, it's still very intriguing that Master Level players don't show even a slight decline vs Diamond Level players and the drastic drop off of **UniqueUnitsMade** from Master to Grandmaster.  The small sample size for Grandmaster Level players could also speak to the break in trend.  Enough thinking, time for some digging.

# In[ ]:


sns.set_palette("dark")
sns.set_style("whitegrid")

# Use Facetgrid to visualize distributions of UniqueUnitsMade across LeagueIndexes
g = sns.FacetGrid(df, col='LeagueIndex', col_wrap=3, margin_titles=True)
g.map(plt.hist, 'UniqueUnitsMade')
g.fig.suptitle('Unique Units Made', fontweight='bold', fontsize=16)
plt.subplots_adjust(top=0.90)


# We already knew from the first set of histograms at the start of this notebook that the number of Grandmaster players is quite small in comparison to other leagues.  Don't think this plot tells us very much.  The range of the y-axis is a bit too wide for Grandmaster League data, requiring new plot.

# In[ ]:


sns.set_palette("dark")
sns.set_style("whitegrid")

plt.hist(df['UniqueUnitsMade'][df['LeagueIndex']==7])
plt.title('Grandmaster League (Index = 7)')
plt.ylabel('Count')
plt.xlabel('Unique Units Made')
plt.show()


# **Eh, the value of this plot doesn't really tell me much either.**  At best I can say that this distribution very ***loosely*** resembles the general shape seen in other leagues.  The small sample size makes it difficult to say much else here.  I'm going to side step this for now and hope to have a shower thought about this later.  Epiphany to come!

# ## Attribute Relationships 
# We've looked at the player attributes on a standalone basis in the previous section.  It's now time to see how these characteristics relate to one another through a correlation matrix.  We don't really need GameID here, so we start off by removing this attribute.  We then build the correlation matrix and can easily see how each of the potential features (independent variables) correlate to our target (dependent variable) in **LeagueIndex**, cleanly positioned at the origin of the matrix.

# In[ ]:


# Copy all df columns to new df, excluding GameID for easy matrix reading
df_for_r = df.copy(deep=True)
del df_for_r['GameID']

# Set figure style
plt.style.use('fivethirtyeight')

# Create figure
fig, axes = plt.subplots(nrows=1, ncols = 1, figsize = (14,10))
fig.suptitle('Attribute Relationships', fontsize=22, fontweight='bold')
# fig.subplots_adjust(top=0.95)

# Generate a mask to hide the upper triangle for a cleaner heatmap.  Less visual noise the better.
mask = np.zeros_like(df_for_r.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Create correlation matrix heatma[]
r_matrix = df_for_r.corr().round(decimals=1)
sns.heatmap(r_matrix, mask=mask, square=True, cmap='YlGnBu', linewidths=.5, annot=True, fmt='g', 
            annot_kws={'size':10})
axes.set_title('     Correlation Matrix\n')
plt.show()


# I really love correlation matrices because of the clear, concise, and clean message they're often accompanied by.  After doing some prep work in the code to minimize noise, we can see how each of the attributes within the data (outliers included), are positively or negatively correlated with **LeagueIndex.**  
# 
# Positive values closest to 1 mean a strong correlation, where the variables sharing that correlation metric increase or decrease together.  The reverse holds true for negative values, where if one variable increases, the other variable decreases.
# 
# Here, it looks like the darkest blue boxes in (APM, SelectByHotKeys, AssignToHotKeys, NumberOfPACs) and the lightest yellow boxes (GapBetween PACs, ActionLatency) have the strongest associations with **LeagueIndex.**    So players with greater **APM** (Actions Per Minute) tend to be associated with a higher **LeagueIndex.**
# 
# However, any stats 101 tutorial will say that correlation doesn't necessarily imply causation - and the same holds true here.  But, at the same time, I do believe that APM contributes to how likely a player is to win a match, ultimately translating to how high of a **LeagueIndex** an SC2 player finds himself in.

# In[ ]:


lst_best_features = ['TotalHours', 'APM','SelectByHotkeys', 'AssignToHotkeys', 'NumberOfPACs',
                     'GapBetweenPACs', 'ActionLatency']


# ## Model Creation (to be continued)

# ![estimator](http://scikit-learn.org/stable/_static/ml_map.png)

# In[ ]:


df.head()


# In[ ]:


# SelectKBest Work


# In[ ]:


x_features = df.copy(deep=True)
x_features = x_features[lst_best_features]
y_target = df['LeagueIndex']
x_features.head()


# In[ ]:


lin_model = svm.LinearSVC()
lin_model = lin_model.fit(x_features, y_target)
print('SVM Score:', lin_model.score(x_features, y_target))
predicted = lin_model.predict(x_features)
print(sorted(pd.unique(predicted)))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
# Naive Bayes Model
bayes_model = GaussianNB()

bayes_model.fit(x_features, y_target)
bayes_model.score(x_features, y_target)


# In[ ]:


log_model = LogisticRegression()
log_model = log_model.fit(x_features, y_target)
log_model.score(x_features, y_target)
print('Log Score:', log_model.score(x_features, y_target)) 
predicted = log_model.predict(x_features) 
print(sorted(pd.unique(predicted)))


# In[ ]:


test_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = test_sizes[0])
log_model = LogisticRegression()
log_model = log_model.fit(x_train, y_train)
log_model.score(x_train, y_train)
print('Log Score:', log_model.score(x_train, y_train)) 
predicted = log_model.predict(x_train) 
print(sorted(pd.unique(predicted)))


# In[ ]:


# KNN model
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model = knn_model.fit(x_features, y_target)
knn_model.score(x_features, y_target)
print('KNN Score:', knn_model.score(x_features, y_target))
predicted = knn_model.predict(x_features)
print(sorted(pd.unique(predicted)))


# In[ ]:


test_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = test_sizes[5])
                                                    
# KNN Model by Training Size
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model = knn_model.fit(x_train, y_train)
knn_model.score(x_train, y_train)  


# In[ ]:


x_train.shape


# In[ ]:


# Decision tree model
tree_model = tree.DecisionTreeClassifier(splitter='best')
tree_model = tree_model.fit(x_features, y_target)
tree_model.score(x_features, y_target)
print('Tree Score:', tree_model.score(x_features, y_target))
predicted = tree_model.predict(x_features)
print(sorted(pd.unique(predicted)))


# In[ ]:


sorted(pd.unique(df['LeagueIndex']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




