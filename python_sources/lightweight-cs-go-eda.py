#!/usr/bin/env python
# coding: utf-8

# # CS:GO Analysis
# 
# *Note: The dataset referenced here seems to have had data removed.  The 'private' dataset I'm using is just the old CSVs that are now missing.*
# 
# *NOTE: This is a conversion of my [GitHub notebook](https://github.com/KasumiL5x/csgo-eda).  It should hopefully work the same!*
# 
# In this notebook, I'm going to be taking [this](https://www.kaggle.com/skihikingkevin/csgo-matchmaking-damage) great dataset and exploring some aspects of it.  For the uninitiated, Counter Strike: Global Offensive (CS:GO) is an online first-person shooter developed and published by Valve Software.  It is a strikingly popular competitive shooter with professional leagues.  This dataset contains a list of around 1400 competitive matchmaking games with at least 16 rounds played per game.  The data also includes official top-down map images and various coordinates, which makes for potentially interesting plots.  The majority of the data represents a single engagement between two characters.  That is, whenever a player was hurt (even by fall damage, for instance), a record was logged.
# 
# I should also clarify that I am quite a casual CS:GO player, so there won't be too much inside knowledge!
# 
# Additionally, this will be quite a casual stroll through the data; I won't dig super deep.
# 
# Instead of exploring the entire dataset variable-by-variable, I have some questions I would like to answer.  To save exploring for decades to come, I will likely restrict my analysis to a map that I am familiar with.  Even then, some cross-map analysis will likely creep in, too.
# 
# One thing to keep in mind about this dataset is that is records **attacks**, not deaths, which means unless we track damage per player cumulatively per round, we cannot know if they died.  There was a discussion [here](https://www.kaggle.com/skihikingkevin/csgo-matchmaking-damage/discussion/65054) which suggested issues with the data in circumstances where bots were overtaken by players that had died, resulting in >100 damage to their health.  For simplicity, I'm going to stick with looking at the attacks rather than deaths.
# 
# Firstly, the obvious questions:
# 
# * Which team wins most often?
# * ~~When Counter Terrorists win, is defusing the bomb or saving hostages more common?~~ (Cannot do this as hostage data isn't present.)
# * What are the most popular weapons ~~and what are their stats (attacks, damage, etc.)~~? (Reliable stats aren't available.)
# * Which part of the body is hit most often? What about per weapon type?
# * ~~Does being on a particular side affect the win as both Terrorists or Counter Terrorists?~~ (Maybe)
# 
# And then some more interesting questions like:
# 
# * Where do attacks happen most often?  Are there any hotspots?
# * ~~Where does fall damage happen most often?~~ (Coordinates are borked, `World` type is unclear)
# * Animate attacks over time, just for fun!

# ## Common imports

# In[ ]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# ## Preprocessing
# The format that the data comes in is a little strange.  There is an entry for each damage point taken, for each round, for each file, which means there is some redundant information.  It requires a lot of grouping and fiddling around to get any per-round or per-game information out of it without duplicates.
# 
# Before we continue, I'm going to do a little manipulation to create some nicer tables for this kind of data.

# In[ ]:


# Load the original data (index and drop as there's a dummy index I don't want to keep).
master_demos = pd.read_csv('../input/csgofixed/mm_master_demos.csv', index_col=0).reset_index(drop=True)


# This table is for stats per game.  The killer here is that there is not a single row for each round per game, but many, many rows.  This means there's a lot of duplicate information.  If the grouping is applied and the first row taken (i.e., all but one of the duplicate rows for a given group combination are dropped), then all repeated data throughout a given round for a given game can be extracted.

# In[ ]:


# Group by `file` and `round` and take the first entry to avoid duplicates.  This gives us a row for each pair.
master_by_round = master_demos.groupby(['file', 'round'], as_index=False).first()

# Group again by `file`, which lets us then extract repeated information for all rounds for a given file.
master_by_file = master_by_round.groupby('file')

# Extract the first occurrence of `map` per file.
map_by_file = master_by_file['map'].first().reset_index()

# Extract and tally `winner_side` per file.  CT/T values may not always be present, so fill those with 0.
side_per_file = master_by_file['winner_side'].value_counts().reset_index(name='amount')
side_per_file = side_per_file.pivot_table(index='file', columns='winner_side', values='amount', fill_value=0).reset_index()
side_per_file.columns = ['file', 'ct_wins', 't_wins']

# Merge the data together per file.
per_game_df = map_by_file.merge(side_per_file, on='file')

# Cleanup.
del master_by_round
del master_by_file
del map_by_file
del side_per_file

per_game_df.head()


# ## Which side wins most often?
# Let's begin by answering which team wins most often.
# 
# As we can see here, terrorists come out on top over all games, but not by much.

# In[ ]:


df = pd.DataFrame(per_game_df[['ct_wins', 't_wins']].sum().values, columns=['amount'], index=['Counter Terrorists', 'Terrorists'])
sns.barplot(data=df, x=df.index, y='amount')
sns.despine()
plt.xlabel('Winning Side')
plt.ylabel('Total Wins')
plt.title('Total wins per side')
plt.show()

del df


# Let's now look at the stats per map.  Unfortunately the wins per map differ so greatly (popular maps, it seems) that a combined bar chart is almost unreadable.

# In[ ]:


wins_per_map = per_game_df.rename({'ct_wins': 'Counter Terrorist', 't_wins': 'Terrorist'}, axis=1).groupby('map').sum().reset_index()
wins_per_map = wins_per_map.melt(id_vars='map').rename({'variable': 'side', 'value': 'amount'}, axis=1)
_ = plt.figure(figsize=(10, 10))
g = sns.barplot(data=wins_per_map, x='map', y='amount', hue='side')
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.xlabel('Map')
plt.ylabel('Wins')
plt.title('Wins per side per map')
plt.show()

del wins_per_map
del g


# Let's instead plot many subplots per map.
# 
# Keep in mind that all axes here are unique - we cannot compare horizontally.  A quick eyeball shows that terrorists lose more often on `de_dust` but win more often on the revised `de_dust2`.  That said, the former has significantly less samples.  Overall, maps with plenty of samples seem to be relatively well balanced.  Some have terrorists just win by count, but others have counter terrorists do the same.  With more data and motives, we could of course dig into this in more detail per map.

# In[ ]:


def wins_per_map(df, map_name, ax):
    # Filter out the map we want.
    map_df = df.loc[df.map == map_name]
    # Sum again but based on this new df.
    wins_df = pd.DataFrame(map_df[['ct_wins', 't_wins']].sum().values, columns=['amount'], index=['Counter Terrorists', 'Terrorists'])
    # Plot!
    sns.barplot(data=wins_df, x=wins_df.index, y='amount', ax=ax)
    sns.despine()
    ax.set(title=map_name, ylabel='Total Wins')
    return wins_df

fig, axes = plt.subplots(7, 3, figsize=(15, 15), sharex=False)
fig.subplots_adjust(hspace=0.5)
for ax, curr_map in zip(axes.flatten(), sorted(per_game_df.map.unique())):
    wins_per_map(per_game_df, curr_map, ax)
fig.tight_layout()


# ## Which guns are most popular?
# Let's next check out which weapon is most popular overall.  Perhaps we can look at the per-map details, too.  I was considering pulling in the stats of the weapons to see if any trends could be found between the most popular weapons, but I don't have the data from the time these games were played and it is constantly tweaked.
# 
# Since we're dealing with total weapon usage, we can just sum up all occurrences of each weapon.  We do see in the list below pretty much what I expected - AK47 and M4A4/M4A1 topping the charts.  This is not surprising because the are the goto assault rifles for the Terrorists and Counter Terrorists respectively.  We already know that terrorists win slightly more often.  If we combine both M4 variations, the AK47 still reigns supreme with 78,504 extra entries.  It's unlikely that Terrorists shoot *this much* more than Counter Terrorists.  More likely is that Counter Terrorists pickup and keep Terrorist weapons.
# 
# Finally, let's pay tribute to those 10 poor souls that were killed by a Decoy.

# In[ ]:


top_5_weaps = master_demos.wp.value_counts().head(5)
sns.barplot(x=top_5_weaps.index, y=top_5_weaps.values)
sns.despine()
plt.xlabel('Weapon')
plt.ylabel('Total occurrences')
plt.title('Five most popular weapons overall')
plt.show()

del top_5_weaps


# In[ ]:


worst_5_weaps = master_demos.wp.value_counts().tail(5)
sns.barplot(x=worst_5_weaps.index, y=worst_5_weaps.values)
sns.despine()
plt.xlabel('Weapon')
plt.ylabel('Total occurrences')
plt.title('Five least popular weapons overall')
plt.show()

del worst_5_weaps


# Let's now explore the same information per-map.  I won't be plotting them for now in Python, but I can imagine some nice infographics could be made based on this information.
# 
# To no surprise, the AK47 and M4** top most of the charts.  That said, de_aztec interestingly has the MP7 as the most favored weapon, although its samples are quite low.  de_shipped has the Nova as the favored weapon, but given the close quarters, this shotgun makes sense.

# In[ ]:


# Group by `map` and `wp` then see how many there are.  This gives us an entry for each map/wp combination and the total number of occurrences.
weap_by_map = master_demos.groupby(['map', 'wp'], as_index=False).size().reset_index().rename({0: 'amount'}, axis=1)

for curr_map in weap_by_map['map'].unique():
    print(f'{curr_map} top 5 weapons:')
    curr_map_weap = weap_by_map.loc[weap_by_map['map'] == curr_map]
    for curr_weap in curr_map_weap.sort_values('amount', ascending=False)[:5].iterrows():
        print(f'\t{curr_weap[1][1]} ({curr_weap[1][2]})')
    print()
    
del weap_by_map


# ## Which body parts are hit most often?
# Sticking with our weapons theme, which parts of the body (i.e., the hit boxes) are hit the most often?  How does this differ per weapon type?  I'm not interested in a statistical test here, but a high-level observation.
# 
# Again, let's start with overall stats.  Given that the chest and stomach supposedly the largest of the hit boxes and the fact that it's safer to shoot than trying for a headshot, it makes sense that these values are dominant.  The upper body in particular makes sense as you want to be aiming around the head, but adjusting for the fact that you may miss!  Aiming more so for the chest is a safer way to play.

# In[ ]:


g = sns.countplot(master_demos['hitbox'], order=master_demos['hitbox'].value_counts().index)
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.xlabel('Hitbox location')
plt.ylabel('Amount')
plt.title('Most common hitboxes')
plt.show()

del g


# We can investigate this for weapons of sniper class only.  Given the required accuracy of a sniper, I'd assume that the chest will be the most common target.

# In[ ]:


sniper_box = master_demos.loc[master_demos['wp_type'] == 'Sniper', 'hitbox']
g = sns.countplot(sniper_box, order=sniper_box.value_counts().index)
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.xlabel('Hitbox location')
plt.ylabel('Amount')
plt.title('Most common hitboxes - Sniper')
plt.show()

del sniper_box
del g


# What about for pistols?  Gotta get that ONE DEAG!  According to the numbers, apparently not.  That said, the head here is proportionally larger than expected, so it's definitely attempted more!

# In[ ]:


pistol_box = master_demos.loc[master_demos['wp_type'] == 'Pistol', 'hitbox']
g = sns.countplot(pistol_box, order=pistol_box.value_counts().index)
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.xlabel('Hitbox location')
plt.ylabel('Amount')
plt.title('Most common hitboxes - Pistol')
plt.show()

del pistol_box
del g


# ## Where do attacks happen most often? Are there any hotspots?
# I'm now going to plot positions of attacks as an overlay on the maps.  To save space and time, I'm going to focus on a map I at least know vaguely well, `de_dust2`.  The idea here is to find clusters of attack points.  This could help us identify choke points and whatnot in a more complete analysis.  We could also do some things like cluster detection based on the distribution of points, but I want to keep this lightweight.

# ### Preprocessing
# The coordinates we have need translating into the coordinate frame of the specific map they are for.  Let's preprocess everything now, per map, to save time.
# 
# Unfortunately, the data provided only has 7 maps worth of data, but we have more maps.  [This repository](https://github.com/akiver/CSGO-Demos-Manager/tree/376cc90eb49425050b351bc933940480f6d48075/Core/Models/Maps) has more maps and the coordinates match.  However, only two of their maps can be used.  We are still missing c`cs_agency`, `de_dust`, `de_austria`, `cs_assault`, `de_thrill`, `de_blackgold`, `cs_insertion`, `cs_office`, `de_shipped`, `de_canals`, and `de_aztec`!

# In[ ]:


import math

def convert_x_to_map(start_x, size_x, res_x, x):
    x += (start_x * -1.0) if (start_x < 0) else start_x
    x = math.floor((x / size_x) * res_x)
    return x

def convert_y_to_map(start_y, size_y, res_y, y):
    y += (start_y * -1.0) if (start_y < 0) else start_y
    y = math.floor((y / size_y) * res_y)
    y = (y - res_y) * -1.0
    return y

map_data = pd.read_csv('../input/csgofixed/map_data.csv', index_col=0)
map_data.loc['de_overpass'] = {'StartX': -4820, 'StartY': -3591, 'EndX': 503, 'EndY': 1740, 'ResX': 1024, 'ResY': 1024}
map_data.loc['de_nuke'] = {'StartX': -3082, 'StartY': -4464, 'EndX': 3516, 'EndY': 2180, 'ResX': 1024, 'ResY': 1024}
map_data


# In[ ]:


# Create the columns that will contain the converted coordinates.
master_demos['AttackPosX'] = np.nan
master_demos['AttackPosY'] = np.nan
master_demos['VictimPosX'] = np.nan # I think this is for victim position?
master_demos['VictimPosY'] = np.nan


for map_name in master_demos['map'].unique():
    if map_name not in map_data.index:
        print(f'Data not found for map: {map_name}')
        continue
    # Pull metadata for the map in question.
    data = map_data.loc[map_name]
    start_x = data['StartX']
    start_y = data['StartY']
    end_x = data['EndX']
    end_y = data['EndY']
    size_x = end_x - start_x
    size_y = end_y - start_y
    res_x = data['ResX']
    res_y = data['ResY']
    
    # Apply the conversion functions to the appropriate columns and store them in the dummy columns created earlier.
    print(f'Converting coordinates for {map_name}', end='')
    master_demos.loc[master_demos['map'] == map_name, 'AttackPosX'] =  master_demos.loc[master_demos['map'] == map_name, 'att_pos_x'].apply(lambda x: convert_x_to_map(start_x, size_x, res_x, x))
    master_demos.loc[master_demos['map'] == map_name, 'AttackPosY'] =  master_demos.loc[master_demos['map'] == map_name, 'att_pos_y'].apply(lambda y: convert_y_to_map(start_y, size_y, res_y, y))
    master_demos.loc[master_demos['map'] == map_name, 'VictimPosX'] =  master_demos.loc[master_demos['map'] == map_name, 'vic_pos_x'].apply(lambda x: convert_x_to_map(start_x, size_x, res_x, x))
    master_demos.loc[master_demos['map'] == map_name, 'VictimPosY'] =  master_demos.loc[master_demos['map'] == map_name, 'vic_pos_y'].apply(lambda y: convert_y_to_map(start_y, size_y, res_y, y))
    print('...done!')

# Cleanup.
del map_data


# ### Hotspots overall
# Let's begin with a high-level plot of overall positions of attack and victim positions.
# 
# For reference, [here](https://steamcommunity.com/sharedfiles/filedetails/?id=157442340) is an image of the de_dust2 callouts.
# 
# <img src="../input/csgofixed/de_dust2_callouts.png" width=50% />

# Below, both attackers and victims are plotted on separate graphs.  We can see from the darker regions on the maps that there are obvious choke points, which is how the map is designed.  Discussion applies to both maps as they are almost identical in hotspots.
# 
# Starting on the left, the tunnels (upper) are a clear designed choke point which provide access to bomb site B from T spawn.  This is one of the main access points and has very little in the way of free movement; it's a risky move going through the tunnel, hence the vast number of encounters.  Similarly, the lower tunnels has a choke point on some stairs which provide very little in the way of visibility and require a risky exposure to get a proper line of sight.  Within bomb site B, the common hiding spots (back plat, big box, window) are all also dense.  Once the bomb is planted, it is common for people to hold out here in defense, or vice versa waiting for the bomb to be brought.  Similarly, the doors to B are quite dense.  This is the second entrance to the bomb site but it also has obscured vision due to the angled doors which makes it risky to peek.  The story is similar to the tunnels - once you become offensive there, you are exposed, and it is one of the main points of access.
# 
# Moving onto the center of the map, the mid doors and its surrounding area are also dense.  The doors hold a similar purpose to those just described - they are a primary yet blind method of crossing the map.  Interestingly, the part above the mid doors is relatively full, too.  If we move south from here to the Terrorist spawn, we can also see a slither of dense blue.  This is a common sniper position to directly hit the doors and beyond, which likely explains the spread of encounters above the mid doors.  We can split by weapon later to confirm this.
# 
# Moving to the right, the long doors hold the same situation as the previous two doors - they are a main port of entry to both sides and provide a restricted view which means exposing yourself to proceed.  Because of this, moving up is risky and encounters are frequent.  Slightly to the right just above the pit is a dense corner.  This, in my experience, is either for Terrorists pushing to A to plant the bomb, or Counter Terrorists pushing to A to defuse the bomb after being mislead to other areas of the map.  The corner is dense because, like the doors, it provides little in the way of cover and requires exposing your player fully to proceed.  In comparison, people north of this position at the ramp do have limited but more cover.  Finally, we can look at the density around the a plat (quad), which has density surrounding crates which people use as cover just like the back plat, big box, and window in bomb site B.
# 
# I could go on, but let's save some words!

# In[ ]:


map_name = 'de_dust2'

dust_data = master_demos.loc[master_demos['map'] == map_name]

# Plot attack positions.
plt.figure(figsize=(20, 20))
plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
plt.scatter(dust_data['AttackPosX'], dust_data['AttackPosY'], alpha=0.005, c='blue')
plt.title(f'Attacker positions for {map_name}', fontsize=20)
plt.show()

# Plot victim positions.
plt.figure(figsize=(20, 20))
plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
plt.scatter(dust_data['VictimPosX'], dust_data['VictimPosY'], alpha=0.005, c='red')
plt.title(f'Victim positions for {map_name}', fontsize=20)
plt.show()


# As mentioned, I believe that the density around the Terrorist spawn and just above mid doors is because of long-distance sniping.  Let's filter out the weapons to plot only sniper shots and see if the densities still remain.
# 
# As you can see below, the Terrorist spawn and mid section are respectively filled with attack positions and victim positions respectively.  This shows that there is a clear line of sight that players who spawn (or happen to be around) the Terrorist spawn abuse.  I've seen matches where everybody storms the door to get wiped out by a set of sniper rounds from the Terrorist spawn.
# 
# While we're here, it's worth having a look at the other choke points.  Those that stand out are the window for bomb site B, where attackers perch themselves to defend the door entrance (which again likely contributes to the density in the mid section).  Similarly, the corner near the pit has a lot of attack positions.  Based on the density and parallelism of the red victim blob around the long doors, it would be safe to say that this corner is used to snipe victims coming through the doorway.  In comparison to our corner earlier which was obviously used to hide behind to avoid the long straight, this corner is used in reverse, exposing oneself to the long A stretch knowing enemies are likely at the doorway instead.

# In[ ]:


dust_sniper_data = dust_data.loc[dust_data['wp_type'] == 'Sniper']

plt.figure(figsize=(20, 20))
plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
plt.scatter(dust_sniper_data['AttackPosX'], dust_sniper_data['AttackPosY'], alpha=0.01, c='blue')
plt.scatter(dust_sniper_data['VictimPosX'], dust_sniper_data['VictimPosY'], alpha=0.01, c='red')
plt.title(f'Attacker and Victim positions for {map_name} with snipers', fontsize=20)
plt.show()


# We can always of course confirm our suspicions by masking out particular coordinates for either attack or victim positions and plotting both.  This way, for example, we could get all samples around the window at bomb site B an confirm where the hits are landing.  We could also further split this by Terrorists vs. Counter Terrorists to see if each team take advantage of different spots.  I'll leave this as an exercise to the reader.
# 
# Below I've split the data where attackers are in the Terrorist spawn (roughly).  We can clearly see that indeed players shoot from the Terrorist spawn through the middle lane where victims meet their demise.
# 
# We could of course delve deeper into this per weapon, per map, per team, per round type, and so on, if we wanted to do better analysis of common gameplay.  This kind of analysis is used in game design to check areas of maps that need improvement during development and refinement.

# In[ ]:


filtered_dust_sniper_data = dust_sniper_data.loc[
    (
        (dust_sniper_data.AttackPosX > 400) &
        (dust_sniper_data.AttackPosX < 500) &
        (dust_sniper_data.AttackPosY > 900)
    )
]

plt.figure(figsize=(20, 20))
plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
plt.scatter(filtered_dust_sniper_data['AttackPosX'], filtered_dust_sniper_data['AttackPosY'], alpha=0.01, c='blue')
plt.scatter(filtered_dust_sniper_data['VictimPosX'], filtered_dust_sniper_data['VictimPosY'], alpha=0.01, c='red')
plt.title(f'{map_name} Terrorist spawn sniper shots', fontsize=20)
plt.show()


# I won't be using the following image for analysis, but it demonstrates how the positions can be combined to draw rays from the attacking position to victim position.

# In[ ]:


from matplotlib.collections import LineCollection

rays_origin = list(zip(filtered_dust_sniper_data.AttackPosX, filtered_dust_sniper_data.AttackPosY))
rays_dest = list(zip(filtered_dust_sniper_data.VictimPosX, filtered_dust_sniper_data.VictimPosY))
lines_formatted = list(map(lambda x, y: [x, y], rays_origin, rays_dest))
lc = LineCollection(lines_formatted, linestyles='dashed', colors=[(1, 0, 0, 0.05)])

fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
ax.add_collection(lc)
plt.title(f'{map_name} Terrorist spawn sniper shots', fontsize=20)
plt.show()

del rays_origin
del rays_dest
del lines_formatted
del lc
del fig
del ax


# In[ ]:


# A bit of cleanup before continuing.
del filtered_dust_sniper_data
del dust_sniper_data


# ## Animations!
# Just for fun, I'm going to take a single demo file and animate the attack points over time.  With some proper refinement, this could be a valuable game design debugging tool, too, as we could see how and where encounters appear throughout many matches visually.  If we had access to more player data, we could also plot their positions.
# 
# The documentation I managed to find for an older version of this dataset says that `tick` is the game state at which the event took place, where in competitive play, 64 states are recorded per second.  Alternatively, `seconds` are a converted format in seconds since the match started.  I'll probably go with seconds.
# 
# The idea here is to take all encounters over a given time period and plot them as time progresses.  Ideally I'd like to plot this per second or so, but looking at the data, the seconds and ticks are a bit oddly formatted.  Instead, I'm just going to use a generic sliding window across the data.

# **WARNING**
# 
# This will take a **very** long time.  If you want to test it yourself, maybe subset the first 100 entries or so using `[:100]` at the end of `round_data = ...`.

# In[ ]:


# If true, will run the animation process. READ THE WARNING FIRST.
do_animation = False

# If true, will save the animation out.
save_animation = False


# In[ ]:


from IPython.display import HTML
from matplotlib import animation, rc

if do_animation:
    # Filter out a particular round.
    round_data = dust_data.loc[dust_data['round'] == 1]

    # How many encounters to show at the same time?
    window_size = 100
    # How many sliding windows will we have?
    window_count = len(round_data) - window_size

    # Create and configure the figure.
    fig, ax = plt.subplots(1, figsize=(10, 10))
    title = ax.text(0.5, 0.95, 'ayyy lmao', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha='center')
    image = ax.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))
    attack_scatter = ax.scatter([], [], alpha=1.0, c='blue')
    victim_scatter = ax.scatter([], [], alpha=1.0, c='red', marker='x')

    # Animation.
    def update(i):
        attack_subset = list(zip(round_data['AttackPosX'][i:i+window_size], round_data['AttackPosY'][i:i+window_size]))
        attack_scatter.set_offsets(attack_subset)

        victim_subset = list(zip(round_data['VictimPosX'][i:i+window_size], round_data['VictimPosY'][i:i+window_size]))
        victim_scatter.set_offsets(victim_subset)

        title.set_text(f'{map_name} Attacks & Victims {i} -> {i+window_size}')

        return (title,)

    anim = animation.FuncAnimation(fig, update, interval=20, frames=window_count, blit=True)


# In[ ]:


# Export out the video into a HTML video.
if do_animation:
    HTML(anim.to_html5_video())


# In[ ]:


if do_animation and save_animation:
    writer_ffmpeg = animation.writers['ffmpeg']
    writer = writer_ffmpeg(fps=15, bitrate=1800)
    anim.save('anim.mp4', writer=writer)


# In[ ]:




