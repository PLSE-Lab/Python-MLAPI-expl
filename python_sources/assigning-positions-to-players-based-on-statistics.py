#!/usr/bin/env python
# coding: utf-8

# Modern basketball blurs traditional player's positions. "Small ball" lineups, decreasing role of centers, increasing number of long distance shooters, etc. 
# In script below I'll use K-means clustering for assigning players to 5 groups based on chosen statistics like points, rebounds, asists, blocks, steals and 3-point shoot attempts. Results will be compared to real positions, average statistics of each group will be presented to visualise their skillset. Afterwards I'll try to choose "leaders" and "perfectly" assigned players (with stats nearest to cluster's centres).
# 
# Let's open the dataset.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

df = pd.read_csv('../input/nba201718.csv')

print(df.head(10))
print(df.tail(10))

df.info()


# Null values appear in shoot performance statistics as a result of no attempts. I'll put zeros there.

# In[ ]:


df.fillna(0, inplace=True)


# I'll check if any player appears more than once.

# In[ ]:


len(df['Player'].unique())


# 540 unique values among of total 664. Let's inspect it.

# In[ ]:


df[df.groupby('Player').transform(len)['Tm'] > 1].head(10)


# Playing for more than one team is represented by multiple rows. Fortunately first row with each player name contains total season stats. I'll use only it for players who played in multiple teams.

# In[ ]:


df = df.groupby('Player').first().reset_index()
df.info()


# Nice. Let's look into the numbers.

# In[ ]:


df.describe()


# Maximum values are unbelievable. I suppose they come from players who played extremely short. I'll delete stats of people with less than 240 minutes played. 

# In[ ]:


df.drop(df[df['MP'] < 240].index, inplace = True)
df = df.reset_index()
df.describe()


# Much better.
# 
# Choosing statistics for use.
# My choice: total rebounds, assists, steals, blocks, 3-point attempts, points

# In[ ]:


col_simple = ['TRB', 'AST', 'STL', 'BLK','PTS','3PA']
df_simple = df[col_simple]


# Clustering:

# In[ ]:


df_simple_normalized = pd.DataFrame(normalize(df_simple), columns = df_simple.columns)
df_simple_normalized.index = df_simple.index
kmeans = KMeans(n_clusters=5, random_state = 123)
group = kmeans.fit_predict(df_simple_normalized.values)
df_simple = df_simple.assign(Group = pd.Series(group, index = df_simple.index))
print(df_simple['Group'].value_counts())
df = df.assign(Group =  df_simple['Group'].values)
df_simple_normalized = df_simple_normalized.assign(Group =  df_simple['Group'].values)


# Let's look into each group profile (average statistics and distribution of official players positions).

# In[ ]:


df_grouped = df_simple.groupby('Group').mean()
df_norm_grouped = df_simple_normalized.groupby('Group').mean()
    
df_grouped.T


# In[ ]:


by_pos = df.groupby('Group')['Pos'].value_counts()
print(by_pos)


# Now I can say something about the results, even name the groups.
# 
# 0 - Sniper, long distance shooter, typical shooting guard.
# 
# 1 - Classic center, scores from short distance, rebounds and blocks shots (post player).
# 
# 2 - Versatile forward, plays close to basket and throws 3-pointers.
# 
# 3 - Playmaker, ball handler, responsible for getting his team into its offense. Capable to create shooting opportunities for colleagues and for himself, long distance shooter.
# 
# 4 - Frontcourt (plays close to the rim), can throw longshots, but is more focused on playing in the paint.

# Skillset visualisation:

# In[ ]:


# data transformation which equalizes maximum value of every skill,
# what makes mastery of them comparable

cluster_labels = ['sniper', 'classic center', 'versatile forward', 'playmaker', 'frontcourt']

df_draw = df_norm_grouped.copy()
maximums = []
for column in df_draw:
    maximums.append(df_draw[column].max())
top = max(maximums)
for i,column in enumerate(df_draw):
    df_draw[column] = df_draw[column] * top / maximums[i]

# plotting 5 radar charts
colors = ['red', 'blue', 'green', 'yellow', 'purple']
fig=plt.figure(figsize = (25,10))
for i in range(1, 6):
    stats = df_draw.loc[i-1]
    angles=np.linspace(0, 2*np.pi, len(col_simple), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    ax = fig.add_subplot(1, 5, i, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=3, color = colors[i-1])
    ax.fill(angles, stats, alpha=0.25, color = colors[i-1])
    ax.set_thetagrids(angles * 180/np.pi, col_simple)
    ax.set_yticklabels([])
    sub_title = 'Group ' + str(i-1) + ' - ' + cluster_labels[i-1]
    ax.set_title(sub_title)
    ax.title.set_fontsize(18)
    ax.set_rmin(0)
    ax.grid(True)
plt.show()


# Who is the best fitted representative of each group?

# In[ ]:


closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_simple_normalized[col_simple])
for i in range(5): 
    print("Group",i,"-",df.loc[closest[i],'Player'].split("\\")[0])


# Need better known names?
# 
# Let's show players who played the most often (y axis) and accuracy of matching finded role (x-axis, closer to the left means better matching). 

# In[ ]:


# calculating distance to nearest cluster's centre
dist = np.min(kmeans.transform(df_simple_normalized[col_simple].values), axis=1)
df = df.assign(dist = pd.Series(dist, index = df.index))

pos_color = {'PG':'red',
             'SG':'gold',
             'SF':'green',
             'PF': 'cyan',
             'C':'blue',
             'PG-SG': 'sandybrown',
             'SF-SG': 'olive'
        }

for g in range(5): 
    df_g = df.loc[df['Group']==g].sort_values(by=['MP'], ascending=False).head(25).reset_index()
    fig, ax = plt.subplots(figsize=(20,10)) 
    for p in df_g.Pos:
        ax.scatter(df_g.dist, df_g.MP, s = 150, c=df_g.Pos.map(pos_color))
    ax.legend(pos_color, labelspacing=2)
    leg = ax.get_legend()
    for nmr, c in enumerate(pos_color):
        leg.legendHandles[nmr].set_color(pos_color[c])
    title = "Group " + str(g) + ' - ' + cluster_labels[g]
    ax.set_title(title)
    ax.title.set_fontsize(24)
    for i, txt in enumerate(df_g.Player):
        ax.annotate(txt.split("\\")[0], (df_g.dist[i], df_g.MP[i]), fontsize = 16)
plt.show()


# In group (sniper) we have Paul George, Bradley Beal, Klay Thompson. Sounds like perfect match. Also combo guards (playmakers who make many shot attempts) appear there: Kemba Walker, Donovan Mitchell, Goran Dragic.
# 
# Group 1 (classic center): Aron Baynes, Marcin Gortat, Rudy Gobert, Clint Capela, Dwight Howard, Andre Drummond. All of them play the same role on the court, which is in description of the group.
# 
# Group 2 (veratile forward): As expected variety of styles. Many tall players with good long distance shoot, like: Ersan Ilyasova, Lauri Markanen, Dirk Nowitzki.
# 
# Group 3 (playmaker): And point guards there, Ricky Rubio, Frank Ntilikina, Chris Paul. Interesting fact: LeBron James was classified here, what makes sense, as he is  ball-handler often.
# 
# Group 4 (frontcourt): Or in other words more versatile centers and poer forwards. So there are: Joel Embiid, Markieff Morris, Karl-Anthony Towns, Anthony Davis, Nikola Jokic and Giannis Antetokounmpo (I expected him in group 2 or maybe even in 1).
# 

# That's it. 
# 
