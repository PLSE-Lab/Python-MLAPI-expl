#!/usr/bin/env python
# coding: utf-8

# # Visualising NBA players of the week 
# This dataset contains information about which player won the [player of the week award](https://basketball.realgm.com/nba/awards/by-type/Player-Of-The-Week/30) in the NBA each week; it additionally contains information about the players themselves. This could allow us to find relationships between positions, physical characteristics, teams and other variables with the NBA player of the week award. 

# In[ ]:


# Imports 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


# Firstly we are going to have to read in the data and take a look at the information contained in the *.csv* file that has been provided.  
# 
# __Note:__ I'm parsing the date of the award into the primary index here so that it makes it easier to filter if we need to later on. 

# In[ ]:


data = pd.read_csv('../input/NBA_player_of_the_week.csv', index_col = 'Date', parse_dates = True)
data.head()


# We can already note some interesting variables that we be good to investigate and visualise here: position, height and weight. 

# ## Exploring some of the columns
# Let's dive deeper into this data and see if we need to perform any cleaning of the data before we start to visualise the stats. 

# In[ ]:


unique_heights = data.Height.unique()
unique_heights.sort()
unique_heights


# Heights appear to be stored as feet and inches which is going to make it difficult to plot; I will write a function that we can apply to the heights to tranform the metric to cm. 

# In[ ]:


unique_weights = data.Weight.unique()
unique_weights.sort()
unique_weights


# In[ ]:


print(data.shape)
data.Conference.value_counts()


# We can see that we are missing values for the `Conference` column of the data. It is unlikely that I'm going to explore this data alongside the award, but I will have a quick look at the values to see if we can deduce anything related to time and the conference system. 

# In[ ]:


data.Conference.isnull().astype(int).plot()
plt.title('Missing value for conference by award date')
plt.show()


# In[ ]:


max(data[data.Conference.isnull()].index)


# It looks as though the latest award that is missing the value for `Conference` is 15th April 2001, every award that has been recorded since that time has a Conference value of either 'East' or 'West'. 

# ## Cleaning heights
# Let's write a function that can change the height from being in feet and inches to cms so that we can visualise them on a continuous numeric scale slightly easier.  
# 
# A quick [google search](https://www.google.com/search?q=inches+to+cm&rlz=1C5CHFA_enGB740GB740&oq=inches+to+&aqs=chrome.0.0j69i57j0l4.1880j1j7&sourceid=chrome&ie=UTF-8) will tell you that one inch equals 2.54cm. Our function will therefore take two steps; translate the feet and inches to just inches, and then convert that into cms using the above ratio. 

# In[ ]:


def clean_heights(height):
    '''
    Converts the height column to a common numeric format and unit (cm)
    '''
    total_inches = 0
    
    # There were initially a mixture of values so this just a try catch 
    if height.find('cm') != -1:
        return int(height.replace('cm', ''))
    else:
        feet = int(height.split('-')[0])
        inches = int(height.split('-')[1])
        total_inches += feet * 12
        total_inches += inches

    return total_inches * 2.54


# In[ ]:


data.Height = data.Height.apply(clean_heights)


# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (12, 4))
sns.set_style("whitegrid")

plt.sca(axes[0])
sns.distplot(data.Height, color = '#27ae60')
plt.title('Distribution of player heights (cm)')

plt.sca(axes[1])
sns.distplot(data.Weight, color = '#e67e22')
plt.title('Distribution of player weights (lb)')
plt.show()


# We can combine these into a single plot using `jointplot` as below. I have kept the colours of the histograms for consistency. 

# In[ ]:


g = sns.jointplot(x=data.Height, y=data.Weight, kind = 'reg', marginal_kws={'color': '#27ae60'})
plt.setp(g.ax_marg_y.patches, color='#e67e22')
plt.setp(g.ax_marg_y.lines, color = '#e67e22')
plt.show()


# In[ ]:


figure = plt.figure(figsize = (10,10))
sns.scatterplot(x = data.Height, y = data.Weight, hue = data.Position)
plt.show()


# A useful type of plot that we can easily achieve here is the facet grid. This creates a plot for each value of a categorical variable. We can use `FacetGrid` and `.map` to create a scatter plot of Height vs. Weight for each position type. As the y and x axis are shared then this makes it slightly simpler to see what is going on when compared to the above plot which is difficult to interpret with so many different position values and points. 

# In[ ]:


g = sns.FacetGrid(data, col = 'Position', col_wrap=4, hue='Position')
g.map(sns.scatterplot, 'Height', 'Weight')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Heights and Weights of players by position')
plt.show()


# ## Repeat winners 
# We can see here that by looking at the number of wins, the majority of players win the award only a handful of times, with the top players going on to become outliers and win the award more than 10 times throughout their career. 

# In[ ]:


data.Player.value_counts().describe()


# Let's visualise this, and find the player that has won the 'Player of the week' award the most throughout the history of the information that we have. 

# In[ ]:


# Set some plotting parameters
plt.rcParams['figure.figsize']=(10, 6)
plt.style.use('fivethirtyeight')

sns.distplot(data.Player.value_counts(), rug = True, color = 'y')
plt.xlabel('Number of awards for a single player')
plt.title('Distribution of "Player of the week" awards')
plt.axvline(data.Player.value_counts().max(), color = 'purple')
plt.annotate(data.Player.value_counts().idxmax(), xy = (61, 0.4), bbox=dict(boxstyle="round", fc="none", ec="purple"), xytext = (-125, 0), textcoords='offset points')
plt.show()


# In[ ]:


data.Player.value_counts().head(10).plot.bar(fontsize = 10)
plt.title('Players with the most "Player of the week" awards')
plt.xticks(rotation = 45)
plt.show()


# In[ ]:


plt.boxplot(data.Age, vert = False)
plt.title("Distribution of Age of 'Player of the week' winners")
plt.xlabel("Age of award winner")
plt.yticks([])
plt.show()


# In[ ]:


sns.distplot(data.Age)
plt.title("Distribution of Age of 'Player of the week' winners")
plt.xlabel("Age of award winner")
plt.axvline(data.Age.mean(), color = 'red')
plt.show()


# In[ ]:


lebron = data[data['Player'] == 'LeBron James'].sort_index()
lebron.groupby('Season').size().plot.bar()
plt.xticks(rotation = 60)
plt.title('LeBron James awards by NBA Season')
plt.show()


# In[ ]:


data['Seasons in league'].value_counts().sort_index().plot.bar()
plt.xticks(rotation = 'horizontal')
plt.xlabel('Seasons in the league of winner')
plt.title('League experience vs. awards given')
plt.show()


# There were 23 players that won a "Player of the week" award when they were in the rookie season. There appears to be a peak with the years in the league, that is very similar in shape to the age. This suggests that there is a physical peak that comes and coincides with experience to mean that you are performing at your best during those 'peak' years. 

# --- 
# I will keep adding to and improving this kernel!  
# 
# If you have any suggestions, or have enjoyed/learned from this kernel then I would greatly appreciate an upvote or a comment! 
