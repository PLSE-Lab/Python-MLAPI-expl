#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import mean


# In[ ]:


plt.style.use('ggplot')
data = pd.read_csv('../input/chess/games.csv')
data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.head()


# In[ ]:


data.nunique()


# To begin with, lets look at the relationship between number of games won and piece color...

# In[ ]:


plt.figure(figsize=(15,10))
sb.countplot(x='winner', data=data)


# As shown, White has a higher total number of wins than Black does. This implies that playing as white pieces gives you a slight advatage. This is well known in the chess community, since white gets the first move, they get the initiative
# 
# Let's look more into this relationship...

# In[ ]:


plt.figure(figsize=(15,10))
sb.countplot(x='winner', hue='victory_status', data=data)


# In[ ]:


white = data.loc[data['winner']=='white']
black = data.loc[data['winner']=='black']
white['victory_status'].value_counts().plot.pie(autopct="%.1f%%", label="White Wins")


# In[ ]:


black['victory_status'].value_counts().plot.pie(autopct="%.1f%%", label="Black Wins")


# As shown by the plots and charts, black has a slighty bigger proportion of wins due to the opponent running out of time, while white has a slightly bigger proportion of wins by resigns and checkmates. It makes sense that when you are a slight disadvantage, playing to the clock can be helpful.
# 
# Now lets look at the most common openings played by white, and their corresponding number of wins...

# In[ ]:


WhiteTop10Openings = white['opening_name'].value_counts().nlargest(10)
plt.figure(figsize=(15,10))
df = white[white['opening_name'].isin(WhiteTop10Openings.index)]
sb.countplot(x='opening_name', data=df)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("# Games Won as White", fontsize=15)
plt.yticks(fontsize=16)
plt.title("White # Win vs Openings")


# As you can see, the most common opening with played by white that results in the highest number of wins is the "Scandinavian Defense". However this does not paint the whole picture, since it is one of the most common openings, it makes sense it would have a large number of wins.
# 
# Instead let's look at the win percentage of these openings...

# In[ ]:


plt.figure(figsize=(15,10))
openingSet = data[data['opening_name'].isin(WhiteTop10Openings.index)]
woS = df.groupby(['opening_name']).count()
oS = openingSet.groupby('opening_name').count()
perc = (woS/oS)*100
sb.barplot(x=perc.index , y=perc.id)

plt.xticks(rotation=90, fontsize=12)
plt.ylabel("% Games Won as White", fontsize=15)
plt.yticks(fontsize=16)
plt.title("White % Win vs Openings")


# This shows us that the best opening for white appears to be the "Philiodor Defense #3" with a win percentage around 65%. Meanwhile, Van't Kruijs Opening has the worst win % for white.
# 
# Lets do the same thing for Black...

# In[ ]:


BlackTop10Openings = black['opening_name'].value_counts().nlargest(10)
plt.figure(figsize=(15,10))
df = black[black['opening_name'].isin(BlackTop10Openings.index)]
sb.countplot(x='opening_name', data=df)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("# Games Won as Black", fontsize=15)
plt.yticks(fontsize=16)
plt.title("Black # Win vs Openings")


# In[ ]:


plt.figure(figsize=(15,10))
openingSet = data[data['opening_name'].isin(BlackTop10Openings.index)]
boS = df.groupby(['opening_name']).count()
oS = openingSet.groupby('opening_name').count()
perc = (boS/oS)*100
sb.barplot(x=perc.index , y=perc.id)

plt.xticks(rotation=90, fontsize=12)
plt.ylabel("% Games Won as Black", fontsize=15)
plt.yticks(fontsize=16)
plt.title("Black % Win vs Openings")


# Here we can see the Van't Kruijs Opening has the best win % for black, which makes sense since it has the worst win % for white.
# 
# Im interested in seeing how a difference in rating affects wins...

# In[ ]:


plt.figure(figsize=(20,10))
data['ratingDiff'] = data['white_rating'] - data['black_rating']
sb.catplot(x='winner', y='ratingDiff', kind='boxen', data=data)
plt.ylabel("Rating Difference (White-Black)")


# Here you can see that when white has a higher rating, the mean is skewed toward white winning, similar story for black. Meanwhile the mean for a draw game is a 0 difference in rating (probably coulda came up with this hypothesis without the chart...)
# 
# It'd would also be interesting to see how this differnce relates to the length of a games

# In[ ]:


plt.figure(figsize=(20,15))
sb.scatterplot(x='turns', y="ratingDiff", data=data)
plt.xticks(fontsize=16)
plt.xlabel("# of Turns", fontsize=15)
plt.ylabel("Rating Difference (White-Black)", fontsize=15)
plt.yticks(fontsize=16)


# Here it looks like the longer a game goes on the more equally matched the two players are
# 
# Lets use a regplot to see this relationship clearer...

# In[ ]:


plt.figure(figsize=(20,20))
data['AbsDiff'] = data['ratingDiff'].abs()
sb.regplot(x='turns', y='AbsDiff', x_estimator=mean, ci=False, data=data)
plt.xticks(fontsize=16)
plt.xlabel("# of Turns", fontsize=15)
plt.ylabel("|Rating Difference|", fontsize=15)
plt.yticks(fontsize=16)


# This confirms that the > number of turns, the smaller the differnce in player rating...
# 
# So suppose you do come across an opponent stronger than you, how can you increase you chances of winning?
# Lets look at the openings that resulted in the largest number of wins for Black when the faces an opponent with +250 rating on them.

# In[ ]:


plt.figure(figsize=(15,10))
df = data.loc[data['ratingDiff']>250]
blackWins = df.loc[df['winner']=='black']
mostWins = blackWins['opening_name'].value_counts().nlargest(10)
sb.barplot(x=mostWins.index, y=mostWins.values)
plt.xticks(rotation=90, fontsize=16)
plt.ylabel("Games Black Won", fontsize=15)
plt.yticks(fontsize=16)
plt.xlabel("Opening", fontsize=15)
plt.title("Games Black Won With -250 Rating on White")


# Here you can see "Owen Defense" resulted in the highest number of wins for black when facing a more skilled opponent.
# 
# How about white?

# In[ ]:


plt.figure(figsize=(15,10))
df = data.loc[data['ratingDiff']<-250]
whiteWins = df.loc[df['winner']=='white']
mostWins = whiteWins['opening_name'].value_counts().nlargest(10)
sb.barplot(x=mostWins.index, y=mostWins.values)
plt.xticks(rotation=90, fontsize=16)
plt.ylabel("Games White Won", fontsize=15)
plt.yticks(fontsize=16)
plt.xlabel("Opening", fontsize=15)
plt.title("Games White Won With -250 Rating on Black")


# This is interesting, Van't Krujis Opening has the highest number of wins for white against a skilled opponent, but the lowest % of wins for white overall among the most comon openings. This is because this particular opening is a "trap" opening, meaning it is used to lure black into traps. However, if black has a decent knowledge of the opening, it can be nuetralized and give the advantage over to them. 
# 
# I suspect that when weaker players come up against stronger opponents, they use this opening to in hopes the stronger opponent may not have knowledge of it resulting in an upset.
