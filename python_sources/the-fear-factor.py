#!/usr/bin/env python
# coding: utf-8

# The distance to the closest defender during a shot varies dramatically from one shot to another. What can we learn from that? Is it because the player with the ball chooses to shoot from far away, in order to make sure that the shot will not be blocked? is it because the defending player want to have a buffer that would allow him to compensate for a speed disadvantage? either way, this distance represents some kind of FEAR. We will explore this fear factor in the following script using Kaggle's great new feature that allows extracting data from multiple data sets in one script. Namely, we would use the 2014-2015 NBA shots data set with a players data set from the same year which I have uploaded. 
# 
# In this script:
# 
#  1. How does the height and other factor affect the defender distance?
#  2. Who are the players who keep their defenders closest?
#  3. Did Stephen Curry rise draw the attention of coaches and changed the way he is guard (or - can we see this obvious effect in the data?)
# 
# and more.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from subprocess import check_output
plt.style.use('fivethirtyeight')
print(check_output(["ls", "../input"]).decode("utf8"))


# Read the data

# In[ ]:


players = pd.read_csv('../input/nba-players-stats-20142015/players_stats.csv')
shots = pd.read_csv('../input/nba-shot-logs/shot_logs.csv')


# In[ ]:


players['Name'] = players.Name.apply(lambda x: x.strip(',.').lower())
shots['Defender_Name'] = shots.CLOSEST_DEFENDER.apply(lambda x: x.split(',')[0])


# First we need to merge the data sets and add the player general stats to the shots data set:

# In[ ]:


def surname(full_name):
   split = full_name.split(' ')
   if len(split) == 2:
       return split[1]
   else:
       return ' '
  
players['Name'] = players.Name.apply(lambda x: x.strip(',.').lower())
players['Surname'] = players.Name.apply(lambda x: surname(x))
shots['Defender_Name'] = shots.CLOSEST_DEFENDER.apply(lambda x: x.split(',')[0].strip(',.').lower())

shots['Player_Height'] = shots.player_name.apply(lambda x: players.Height[players.Name == x].values[0] if len(players.Height[players.Name == x].values)>0 else 0) 
shots['Defender_Height'] = shots.Defender_Name.apply(lambda x: players.Height[players.Surname == x].values[0] if len(players.Height[players.Surname == x].values)>0 else 0) 
shots['Height_Diff'] = shots.Player_Height - shots.Defender_Height


# Since there are some missing values, let's get rid of those and move on with a slightly smaller data frame:

# In[ ]:


shots = shots[np.abs(shots.Height_Diff) < 50]
shots = shots[shots.Defender_Height>0]
shots = shots[shots.Player_Height > 0]


# Now we can see whether height, and more particularly the height difference, plays a role in the distance between the attacker and the defender:

# In[ ]:



plt.subplot(1,3,1)
lim = 23
dist_data = shots[shots.SHOT_DIST<=lim+1]
dist_data = dist_data[dist_data.SHOT_DIST>lim-1]
dist_data = dist_data[dist_data.CLOSE_DEF_DIST<8]

x = dist_data.Height_Diff + np.random.normal(0,1.25,len(dist_data))
y = dist_data.CLOSE_DEF_DIST*0.3048
plt.plot(x,y,'o',alpha = 0.3, markersize = 5)
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.xlabel('Height Difference [cm]')
plt.ylabel('Distance From Defender [m]')
plt.title('3 Point Shots')


plt.subplot(1,3,2)
lim = 15
dist_data = shots[shots.SHOT_DIST<=lim+1]
dist_data = dist_data[dist_data.SHOT_DIST>=lim-1]
dist_data = dist_data[dist_data.CLOSE_DEF_DIST<8]

x = dist_data.Height_Diff + np.random.normal(0,1.25,len(dist_data))
y = dist_data.CLOSE_DEF_DIST*0.3048
plt.plot(x,y,'o',alpha = 0.3, markersize = 5)
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.xlabel('Height Difference [cm]')
plt.ylabel('Distance From Defender [m]')
plt.title('Middle Range')

plt.subplot(1,3,3)
lim = 10
dist_data = shots[shots.SHOT_DIST<=lim+1]
dist_data = dist_data[dist_data.SHOT_DIST>=lim-1]
dist_data = dist_data[dist_data.CLOSE_DEF_DIST<8]

x = dist_data.Height_Diff + np.random.normal(0,1.25,len(dist_data))
y = dist_data.CLOSE_DEF_DIST*0.3048
plt.plot(x,y,'o',alpha = 0.3, markersize = 5)
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.xlabel('Height Difference [cm]')
plt.ylabel('Distance From Defender [m]')
plt.title('Close Shots')


# Consistently, the larger the difference is (that is - the attacker is taller than the defender), the small the distance is. 
# 
# We see this phenomenon even when we naively control for the shot distance (obviously the closer the shooter is, the closer his defender will be). 
# 
# But it is still unclear whether this comes from the attacker - he prefers to take a step back when facing a tall defender who poses a high block-risk, or the defender who is not as worried when a tall player has the ball - knowing that tall players have, on average, lower field goals percentage. 

# In[ ]:


x = dist_data.Defender_Height + np.random.normal(0,1.25,len(dist_data))
y = dist_data.CLOSE_DEF_DIST*0.3048
plt.plot(x,y,'o',alpha = 0.4)
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.xlabel('Defender Height [cm]')
plt.ylabel('Distance From Defender [m]')


# So as seen in the plot above, if we simply look at the defender distance vs his height (only for close distance shots), we see that the distance is larger when the defender is taller. since these are close shots, the possibility of a tall defender taking a few steps back to avoid penetration is less likely (because the attacker is already relatively close to the basket). This means that we can see that the taller the defender is, the more fearful the attacker is.
# 
# While this is not surprising, corroborating hypotheses using the data is always very satisfying (well, for me at least). 
# 
# Now let's bring in other factors such as the weight, age and position (which I would label from 1 to 5, 1 being PG and 5 being C, since this is a reasonable proxy for the average distance from the basket [where centers are usually the closest])

# In[ ]:


shots['Player_Weight'] = shots.player_name.apply(lambda x: players.Weight[players.Name == x].values[0] if len(players.Weight[players.Name == x].values)>0 else 0) 
shots['Defender_Weight'] = shots.Defender_Name.apply(lambda x: players.Weight[players.Surname == x].values[0] if len(players.Weight[players.Surname == x].values)>0 else 0) 

shots['Player_Age'] = shots.player_name.apply(lambda x: players.Age[players.Name == x].values[0] if len(players.Age[players.Name == x].values)>0 else 0) 
shots['Defender_Age'] = shots.Defender_Name.apply(lambda x: players.Age[players.Surname == x].values[0] if len(players.Age[players.Surname == x].values)>0 else 0) 


# In[ ]:


shots['Player_Pos'] = shots.player_name.apply(lambda x: players.Pos[players.Name == x].values[0] if len(players.Age[players.Name == x].values)>0 else 0) 
shots['Defender_Pos'] = shots.Defender_Name.apply(lambda x: players.Pos[players.Surname == x].values[0] if len(players.Age[players.Surname == x].values)>0 else 0) 


# ## Linear Regression
# 
# Let's try the old trick of training  a linear regression and then examine the feature coefficients to determine the way they influence the target variable in our case the defender distance. 

# In[ ]:


shots = shots[shots['Player_Weight']>0]
shots = shots[shots['Defender_Weight']>0]

shots = shots[shots['Player_Age'] >0]
shots = shots[shots['Defender_Age'] > 0]

def map_position(x):
    if x == 'PG':
        return 1
    if x == 'SG':
        return 2
    if x =='SF':
        return 3
    if x =='PF':
        return 4
    if x == 'C':
        return 5
    if x == 0:
        return 0

shots['Player_Pos'] = shots.Player_Pos.apply(lambda x: map_position(x))
shots['Defender_Pos']  = shots.Defender_Pos.apply(lambda x: map_position(x))
X = shots[['Player_Height','Player_Weight','Defender_Height','Defender_Weight','Player_Age','Defender_Age','Player_Pos','Defender_Pos','SHOT_DIST']]

X_std = StandardScaler().fit_transform(X)
y = shots.CLOSE_DEF_DIST*0.3048

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)

model.predict(X_test)

plt.barh(range(len(X.columns)),model.coef_)
plt.yticks(range(len(X.columns)),['Player_Height','Player_Weight','Defender_Height','Defender_Weight','Player_Age','Defender_Age','Player_Pos','Defender_Pos','SHOT_DIST'], fontsize = 10)
plt.title('Regression Coefficients')

print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))


# Before analyzing the coefficients, it is important to notice that this model explained only 0.27 of the variance in the results, so the feature importance claim are not very strong.
# 
# Nonetheless - we see that as expected the shit distance has the strongest weight. but if we put that aside:
# 
#  - we see that the closer the player's position is, the further the
#    defender is (defender would be closer to a point guard rather than a
#    center having all other conditions equal - this straightness the
#    fear-of-attacker theory, which means that defenders get closer that
#    higher the shooting percentage are.
#  - The attacker the defender, the further the defender is. this can be a proxy to the attacker's speed - meaning that again the fear is from the defender side (we can reach to the same conclusion using the attacker weight)
#  - We again see the influence of the defender height - although it is unclear whether this means that the attacker prefers to stay away, or the defender prefers to take a few steps back to allow for a better response to a quick penetration attempt. 
# 
# Let's look at the data from a more aggregate point of view - players yearly average:

# In[ ]:


players['Average_Def_Dist'] = players.Name.apply(lambda x: np.mean(shots.CLOSE_DEF_DIST[shots.player_name == x]))
players['Average_Dist_as_Def'] = players.Surname.apply(lambda x: np.mean(shots.CLOSE_DEF_DIST[shots.Defender_Name == x]))


# In[ ]:


result = players[~np.isnan(players.Average_Def_Dist) == True]
result = result.sort(['Average_Def_Dist'], ascending= 0)
plt.subplot(2,1,1)
plt.barh(range(10),result.tail(10).Average_Def_Dist*0.3048)
plt.yticks(range(10),result.tail(10).Name)
plt.xlim([0,2])
plt.title('Top ten')

plt.subplot(2,1,2)
plt.title('Botton ten')
plt.barh(range(10),result.head(10).Average_Def_Dist*0.3048)
plt.yticks(range(10),result.head(10).Name)
plt.xlabel('Average Defender Distance [m]')


# Since this plot doesn't control for shot distance, we see that tall players who usually play closer to the rim drag the defenders closer. but this is trivial.
# 
# Let's look at the defender point of view - who prefers to stay far away and who sticks to the attacker? 

# In[ ]:


result = players[~np.isnan(players.Average_Dist_as_Def) == True]
result = result.sort(['Average_Dist_as_Def'], ascending= 0)

plt.subplot(2,1,1)
plt.barh(range(10),result.tail(10).Average_Dist_as_Def*0.3048)
plt.yticks(range(10),result.tail(10).Name)
plt.xlim([0,6])
plt.title('Top Ten')

plt.subplot(2,1,2)
plt.title('Bottom Ten')
plt.barh(range(10),result.head(10).Average_Dist_as_Def*0.3048)
plt.yticks(range(10),result.head(10).Name)
plt.xlabel('Average Distance as Defenders [m]')


# Hmmm. not enough famous names (besides Kobe). 

# In[ ]:


players['Dist_for_3'] = players.Name.apply(lambda x: np.mean(shots.CLOSE_DEF_DIST[(shots.player_name == x) &(shots.SHOT_DIST > 24)]))


# Let's limit ourselves to 3-pointers to control for shot distance and hopefully get more interesting names. Let's also only look at players who shot at least 15 times from beyond the arc:

# In[ ]:


result = players[~np.isnan(players.Dist_for_3) == True]
result = result[result['3PA'] > 15]

result = result.sort(['Dist_for_3'], ascending= 0)

plt.subplot(2,1,1)
plt.barh(range(10),result.tail(10).Dist_for_3*0.3048)
plt.yticks(range(10),result.tail(10).Name)
plt.xlim([0,4])
plt.title('Top 10')

plt.subplot(2,1,2)
plt.title('Bottom 10')
plt.barh(range(10),result.head(10).Dist_for_3*0.3048)
plt.yticks(range(10),result.head(10).Name)
plt.xlim([0,4])
plt.xlabel('Average Defender Distance [m]')


# The top and bottom names are interesting:
#  1. Kobe - mind you that this is 2015 when he still played (although way beyond his prime).
#  2. Pau Gasol -defenders seem to be incredibly non-intimidated by his shots. But is it justified? 

# In[ ]:


plt.subplot(1,2,1)
plt.xlabel('3P%')
plt.ylabel('Average Defender Distance')
plt.plot(result['3P%'],result.Dist_for_3*0.3048,'o')
x = (result['3P%'])
y = result.Dist_for_3*0.3048
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')

plt.subplot(1,2,2)
plt.xlabel('3P made')
plt.ylabel('Average Defender Distance')
plt.plot(result['3PM'],result.Dist_for_3*0.3048,'o')
x = (result['3PM'])
y = result.Dist_for_3*0.3048
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')


# Defenders definitely take into account the number of 3's a player made. The 3P% are not as important. This does make sense since 3P% is not the perfect measure for how big of a threat a players poses if it doesn't take into account the number of threes he actually makes (and of course this figure is very noisy for players who took fewer shots).
# 
# Pau Gasol is a clear out-liar though. Clearly defenders don't take him as seriously as they should when he shots from down town. It probably has something to do with the general lack-of-grace of his style.

# In[ ]:


players[players.Name == 'pau gasol']


# In[ ]:


players['Average_shot_distance'] = players.Name.apply(lambda x: np.mean(shots.SHOT_DIST[shots.player_name == x]))


# In[ ]:


X = players[['Height','Average_shot_distance','PTS','3PM','Age','Average_Def_Dist']]

X = X[np.isnan(X.Height) == False]
X = X[np.isnan(X.Average_shot_distance) == False]

X_std = StandardScaler().fit_transform(X.drop(['Average_Def_Dist'], axis = 1))
y = X.Average_Def_Dist*0.3048

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)

model.predict(X_test)

plt.barh(range(len(X.columns)-1),model.coef_)
plt.yticks(range(len(X.columns)-1),['Height','Average_shot_distance','PTS','3PM','Age'], fontsize = 10)
plt.title('Regression Coefficients')

print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))


# Predicting the average shot distance is easier, given that there's lower variance in such configuration. 
# 
# We see that the defender gets closer the older the attacker is, and the more points and 3Ps he makes.
# 
# The taller and further from the basket the attacker is, on the other side, the further the defender stays from the shooter. 
# 
# So, it seems that the 3PM, general points and height coefficients  represent a situation where the defender chooses his distance from the attacker based on the attacker average shooting skills - the better they are , the close the defender would be.
# 
# The age can be a proxy to a different situation, where the defender would get closer to an older player knowing that he is not fast enough and is therefore less likely to pass him.

# ## The rise of Stephen Curry
# 
# While Curry's numbers were already quite amazing in 2014, it was by the end of 2015 when he won his first MVP award, and since then is arguably the best player in the world (a title he shares with at least 3 other players in today's interesting league). 
# 
# Did his great season and progress change the way defenders treated him, in terms distance at least?
# 
# let's look at the average defender distance during a 3-points shot and the aggregated 3 points percentage along the year. 

# In[ ]:


curry = shots[shots.player_name == 'stephen curry']


# In[ ]:


average_dist = []
aggregated_pct = []
non_aggregated_pct = []
j = 0
shots = 0
shots_made = 0
for game in np.unique(curry.GAME_ID):
    average_dist.append(np.mean(curry.CLOSE_DEF_DIST[(curry.GAME_ID == game) & (curry.SHOT_DIST > 23)]))
    shots += len(curry[(curry.GAME_ID == game) & (curry.SHOT_DIST > 23)])
    shots_made += len(curry[(curry.GAME_ID == game) & (curry.FGM == 1) & (curry.SHOT_DIST > 23)])
    aggregated_pct.append(shots_made/shots)
    non_aggregated_pct.append(len(curry[(curry.GAME_ID == game) & (curry.FGM == 1) & (curry.SHOT_DIST > 23)])/len(curry[(curry.GAME_ID == game) & (curry.SHOT_DIST > 23)]))
     
def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

avg_dist = runningMean(average_dist,4)*0.3048
plt.plot(avg_dist[4:len(avg_dist)-4],'o')
x = range(len(avg_dist[4:len(avg_dist)-4]))
y = avg_dist[4:len(avg_dist)-4]
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.ylabel('Average Defender Distance [m]')
plt.title('Defender Distance along the season')
plt.show()

pct = runningMean(aggregated_pct,4)
plt.plot(pct[4:len(avg_dist)-4],'o')
x = range(len(pct[4:len(pct)-4]))
y = pct[4:len(pct)-4]
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.title('3-Points % along the season')
plt.ylabel('Aggregated 3-ponts %')
plt.show()

plt.plot(average_dist,non_aggregated_pct,'o')
x = average_dist
y = non_aggregated_pct
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x, fit_fn(x),'r')
plt.ylabel('3 points % in the game')
plt.xlabel('Average defender distance in the game')
plt.show()
  


# While it is very obvious that curry's 3 points percentage improved over the year (though definitely not in a linear fashion as the fitted trend line might imply), it's more difficult to conclude whether the average defender got closer. The trend line does show that but the fit goodness maybe too low to be conclusive. 
# 
# Notice that the x axis in the first 2 graphs is the outcome of a running average smoothing and therefore doesn't have a meaningful name\unit. 
