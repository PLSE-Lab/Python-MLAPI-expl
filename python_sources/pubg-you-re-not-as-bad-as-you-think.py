#!/usr/bin/env python
# coding: utf-8

# If you've played PUBG, you've probably played your fair share of bad games and gotten frustrated. With this data, lets see if you really are bad, or maybe just average. Also, find out if you're a cut above the rest.

# #### Data Prep
# Let's get started by loading up some of the data, and prepping it.

# In[1]:


# environment preparations
import numpy as np
import pandas as pd

kills_paths = []
kills_paths += ["../input/aggregate/agg_match_stats_0.csv"]
kills_paths += ["../input/aggregate/agg_match_stats_1.csv"]
kills_paths += ["../input/aggregate/agg_match_stats_2.csv"]
# kills_paths += ["../input/aggregate/agg_match_stats_3.csv"]
# kills_paths += ["../input/aggregate/agg_match_stats_4.csv"]

# these are the columns we care about, leaving out the data we won't use
col_filter = [
#                 'match_id',
                'party_size', # 1, 2, 4
#                 'match_mode', # fpp, tpp - theyre all tpp
#                 'player_name',
                'player_kills',
                'team_placement',
                'player_dmg',
                  ]


# combine all the data files into one array
kills = None
for kill_file_path in kills_paths:
    new_kills = pd.read_csv(kill_file_path, usecols=col_filter)
    kills = pd.concat([kills, new_kills])
    
    
# Filtering the data

# solo
kills_solo=kills[kills['party_size']==1]
# kills_duo=kills[kills['party_size']==2]
kills_squad=kills[kills['party_size']==4]

kills_solo.drop(columns=['party_size'])
# kills_duo.drop(columns=['party_size'])
# kills_squad.drop(columns=['party_size'])

# Take a sample
# sample_size = 7500
# kills_solo = kills_solo.sample(sample_size)
# kills_duo = kills_duo.sample(sample_size)
# kills_squad = kills_squad.sample(sample_size)

#save some memory
del kills

print(len(kills_solo))
kills_solo.head()


# Lets take a look at our data, see what we're dealing with.

# In[2]:


rank_and_kills = kills_solo[['team_placement', 'player_kills']]
plot = rank_and_kills.plot.scatter(x='team_placement', y='player_kills', color='green')
plot.set_xlabel("rank")
plot.set_ylabel("kills")
plot.grid(color='black', axis=['x', 'y'], linestyle='solid')


# # Quartiles
# Let's look at some chunks of the player base. I'll start with the lowest ranking quarter and work my way up. (to clarify, that means 100th to 76th place as the lowest quarter and so on)

# In[3]:


# format is ("name", <=, >)
groups = [("Q1", 100,75), ("Q2", 75, 50), ("Q3", 50,25),("Q4", 25,0),("T10", 10,0),("T7", 7,0), ("T5", 5,0), ("T3", 3,0), ("Winner", 1,0)]
print("    Kills\t\t   Damage\n    average\tmedian\t   average\tmedian")
for (name, lte, gt) in groups:
    mean_kills = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_kills'].mean()
    median_kills = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_kills'].median()
    mean_dmg = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_dmg'].mean()
    median_dmg = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_dmg'].median()
    print(name+": "+str(mean_kills)[0:5]+"  \t"+str(median_kills)+"\t   "+str(mean_dmg)[0:5]+"  \t"+str(median_dmg))


# #### Top 5

# In[4]:


my_hist = kills_solo[kills_solo['team_placement'] <= 5]['player_kills'].hist(bins=25, range=[0,25], color='green')
my_hist.set_title("Top 5")
my_hist.set_xlabel("num kills")
my_hist.set_ylabel("count")


# Take a look at this hitogram. Look how many "top 5" players get that high without even getting a single kill! Being a pacifist can get your quite far.

# # How good do you have to be?
# Narrowing in on the top 25 - what does it take?

# In[11]:


max_rank = 25
avg_kills = [0]+list(range(1,max_rank+1))
stddev_kills = [0]+list(range(1,max_rank+1))

for rank in avg_kills:
    k = kills_solo[kills_solo['team_placement'] == rank]['player_kills']
    avg_kills[rank] = k.mean()
    stddev_kills[rank] = k.std()

d = {"average kills": avg_kills}
df = pd.DataFrame(data=d)
p = df.plot(yerr=stddev_kills, color='green')
p.set_ylabel("kill count")
p.set_xlabel("rank")
# ignore the error message about a NaN
# it's because there's no data for players ranked as 0th place


# The above plot is average kills, with the vertical bars being 1 standard deviation. Clearly higher rank and higher kill count are correlated, but as you can see with the standard deviation, it is not rare to rank highly and have a low kill count.

# In[6]:


kill_count = list(range(0,10))
print("kills\tavg rank")
for k in kill_count:
    print(str(k)+"\t"+str(kills_solo[kills_solo['player_kills'] == k]['team_placement'].mean())[0:5])
print(">"+str(k)+"\t"+str(kills_solo[kills_solo['player_kills'] > kill_count[-1]]['team_placement'].mean())[0:5])


# If you're getting more kills, you're a lot more likely to also be ranking higher. This isn't surprising.

# # Chicken Dinner Club
# #### Kills

# In[7]:


winners = kills_solo[kills_solo['team_placement'] == 1]
first_place_kills = winners['player_kills']
my_hist = first_place_kills.plot.hist(bins=25, range=(0,25), color='gold')
my_hist.set_xlabel("kills")
my_hist.grid(color='black', axis='x', linestyle='solid')


# Seems like every now and then you can win with 0 or 1 kills, but the bulk of 1st placers are getting between 3 and 6 kills.

# #### Damage per Kill

# In[8]:


dmg_per_kill = winners['player_dmg'] / winners['player_kills']
dmg_per_kill = dmg_per_kill.replace(np.inf, -4)
plot = dmg_per_kill.plot.hist(bins=102, range=(-4,200), color='gold')
plot.grid(color='black', axis='x', linestyle='solid')
plot.set_xlabel("damage dealt / number of kills")


# It's roughly around 100 damage points per kill, which is expected as each person has 100 points of health. Though the data does seem to skew a little bit toward doing more than 100 damage per kill. This could be that they fight people who heal mid-battle or that they take shots at people who end up getting away.
# 
# Looking into it more, you'll notice that to the left of the peak is somewhat higher than to the right. I believe this is because players often lose health to the zone or a fall and only have about 90 health when in combat. However, more of the distribution is about 100 dmg/kill. This tells me that the winners 
# 
# Also if you've noticed that there's some data below zero, that's where I put the people who did damage and got no kills, because (damage / zero kills) is a divide-by-zero error.

# #### Total Damage

# In[9]:


plot = winners['player_dmg'].plot.hist(bins=50, range=(0,3000), color='gold')
plot.grid(color='black', axis='x', linestyle='solid')
plot.set_xlabel("total damage dealt")

