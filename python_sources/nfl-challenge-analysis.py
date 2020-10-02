#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 


import os
print(os.listdir("../input"))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Combining a few tables to get a clean look at the details of plays that resulted in concussions

video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')


# In[ ]:


video_review.head()


# In[ ]:


position = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')


# In[ ]:


position.head()


# In[ ]:


concussion_play_data = pd.merge(video_review,position)


# In[ ]:


concussion_play_data.head()


# Let's start to explore. Are there activities or positions that are more likely to sustain concussions?

# In[ ]:


concussion_play_data.Role.value_counts()


# In[ ]:


concussion_play_data.groupby('Player_Activity_Derived').Role.value_counts()


# It seems like there might be something here, but we don't have a clear view of which team is more impacted. Let's map the positions to the punting or receiving team based on the position charts provided in the dataset.

# In[ ]:


team = {'GL':'Punting','PLT':'Punting','PLG':'Punting','PLS':'Punting','PRG':'Punting','PRT':'Punting','GR':'Punting','PLW':'Punting','PRW':'Punting','PC':'Punting','PPR':'Punting','P':'Punting','VRo':"Receiving",'VRi':"Receiving",'PDR1':"Receiving",'PDR2':"Receiving",'PDR3':"Receiving",'PDL3':"Receiving",'PDL2':"Receiving",'PDL1':"Receiving",'PLR':"Receiving",'PLM':"Receiving",'PLL':"Receiving",'PFB':"Receiving",'PR':"Receiving",'PDL4':"Receiving",'VLo':"Receiving",'VR':'Receiving'}

concussion_play_data['Team']=concussion_play_data.Role.replace(team)


# In[ ]:


concussion_play_data.Team.value_counts()


# Very interesting! Players from the punting team sustained concussions at a rate of nearly 3:1 over the receiving team!

# In[ ]:


concussion_play_data.groupby('Team')['Player_Activity_Derived'].value_counts()


# In[ ]:


sns.countplot(x='Player_Activity_Derived', data=concussion_play_data, hue='Team')


# Tackles are dangerous! 
# 
# Clearly, a good outcome of the proposed rule changes will be reducing the number of collisions on punt plays.
# 
# Are there any additional areas that we can focus on? The chaos from a turnover? Gunners launching themselves on a tackle?

# In[ ]:


concussion_play_data['Turnover_Related'].value_counts()


# No concussions were the result of a turnover

# In[ ]:


concussion_play_data.groupby('Team').Role.value_counts()


# In[ ]:


sns.countplot(x='Role', hue='Team', data = concussion_play_data)


# We would fully expect that Punt Returners (PR) would be at risk of concussion. It also stands to reason that players on the outside of the punting formation would be at risk as they are most often involved in tackling.
# 
# Are there types of impacts that are riskier?

# In[ ]:


concussion_play_data['Primary_Impact_Type'].value_counts()


# In[ ]:


sns.countplot(x='Primary_Impact_Type', data=concussion_play_data)


# Not really. Helmet-to-helmet hits have been a focus of the league and will continue to be so, but occurred at the same rate as helmet-to-body contact, which would be nearly impossible to prohibit.
# 
# How do these plays compare to the remaining punt plays that did not result in a concussion?

# In[ ]:


#Create a unique identifier for each game by combining GameKey and PlayID

concussion_play_data['GameKey_PlayID'] = concussion_play_data['GameKey'].map(str) + concussion_play_data['PlayID'].map(str)


# In[ ]:


concussion_play_data.head()


# In[ ]:


#Assign True value for all concussions in this data set
concussion_play_data['Concussion'] = 'True'


# In[ ]:


#Read in remaining play data (including concussion punts)

all_plays = pd.read_csv('../input/play-informationcsv/play_information.csv')


# In[ ]:


all_plays.info()


# In[ ]:


#A little offline manipulation for time's sake. 
#Here Territory = 0 for a punt from the punting team's side of the field; Territory = 1 for a punt from the receiving team's side of the field

all_plays.head()


# In[ ]:


#To merge tables, check that PlayID is unique

all_plays.PlayID.value_counts()


# In[ ]:


# PlayID is not unique! So we create a unique identifier to merge the tables by combining GameKey and PlayID and removing the lone duplicate
all_plays['GameKey_PlayID'] = all_plays['GameKey'].map(str) + all_plays['PlayID'].map(str)
all_plays['GameKey_PlayID'].value_counts()


# In[ ]:


#Visualize the only duplicate from our unique identifier

all_plays[all_plays['GameKey_PlayID']=='613849']


# In[ ]:


#Remove duplicate by adding a and b

all_plays.iloc[697, all_plays.columns.get_loc('GameKey_PlayID')] = '613849a'
all_plays.iloc[6124, all_plays.columns.get_loc('GameKey_PlayID')] = '613849b'


# Much of the interesting details about the plays are contained in the long string called 'PlayDescription', so let's play with that and encode some more variables before merging tables

# In[ ]:


#Assigns True to 'fair_catch' if the 'PlayDescription' contains the phrase 'fair catch'

all_plays['fair_catch'] = all_plays['PlayDescription'].str.contains('fair catch')


# In[ ]:


#Assigns True to 'Touchback' if the 'PlayDescription' contains the phrase 'Touchback'

all_plays['Touchback'] = all_plays['PlayDescription'].str.contains('Touchback')


# In[ ]:


#Assigns True to 'out_of_bounds' if the 'PlayDescription' contains the phrase 'out_of_bounds'

all_plays['out_of_bounds'] = all_plays['PlayDescription'].str.contains('out of bounds')


# In[ ]:


#Assigns True to 'out_of_bounds' if the 'PlayDescription' contains the phrase 'out_of_bounds'

all_plays['downed'] = all_plays['PlayDescription'].str.contains('downed')


# In[ ]:


#Manipulate the string for 'PlayDescription' to capture the distance of the punt

new = all_plays['PlayDescription'].str.split(pat='punts',n=1, expand=True)
all_plays['split1'] = new[0]
all_plays['split2'] = new[1]
all_plays['distance'] = all_plays['split2'].str.extract('(\d+)')
all_plays.drop(columns=["split1","split2"], inplace=True)


# In[ ]:


all_plays['distance']=all_plays['distance'].fillna(0)
all_plays['distance'] = all_plays['distance'].map(int)


# In[ ]:


#Great, let's merge tables to get the full universe of all punt plays from the 16-17 seasons with our new variables
full_data = pd.merge(all_plays, concussion_play_data, on='GameKey_PlayID', how='left')
full_data.info()


# We now have a universe of **6,681 punt plays** from the 16-17 seasons that resulted in **37 concussions**.
# 
# Let's visualize once again the 37 plays that resulted in a concussion.

# In[ ]:


full_data[full_data['Concussion'] == "True"]


# In[ ]:


# 'Concussion' will be our target, so we fill the concussion column with False for NA

full_data['Concussion']=full_data['Concussion'].fillna('False')


# In[ ]:


full_data.info()


# In[ ]:


#Because the 'YardLine' field reports numbers on the 0-50-0 scale of a football field, I did a little manipulation to be able to show where the play orinigated on one axis

full_data['starting_yard'] = full_data['YardLine'].str.extract('(\d+)')
full_data['starting_yard'] = full_data['starting_yard'].map(int)


# In[ ]:


def yardline (row):
    if row['Territory'] == 0:
        return row['starting_yard']
    if row['Territory'] == 1:
        return 100-row['starting_yard']

full_data['100yd'] = full_data.apply (lambda row: yardline (row), axis=1)
full_data.head()


# In[ ]:


full_data.drop(columns='starting_yard')


# In[ ]:


full_data.sort_values('100yd', ascending=False)


# In 2016, the Niners punted from the Bears 31? That seems...ill-advised. 
# 
# But cool, the funciton worked and now we can graph the punts and outcomes.

# In[ ]:


#It would also be helpful to encode a variable for whether or not the play resulted in a return, because it sure seems like returns are dangerous

def returned (row):
    if row['fair_catch'] == 1:
        return False
    if row['Touchback'] == 1:
        return False
    if row['out_of_bounds'] == 1:
        return False
    if row['downed'] == 1:
        return False
    if row['distance'] == 0:
        return False
    return True

full_data['returned'] = full_data.apply (lambda row: returned (row),axis=1)        


# In[ ]:


full_data.head()


# In[ ]:


full_data.info()


# Now that we have the full set of play data, we're going to use Machine Learning to create a decision tree to parse this large dataset and help us visualize which types of punts result in Concussions and how we can try to influence safer outcomes from punt plays.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# Which variables should we focus on? I'm not going to include variables that we couldn't (or nearly couldn't) influence. 
# 
# For instance, should we include the type of game? To what end? Are we going to ban punting in the postseason or tell teams not to practice it in the preseaon? That doesn't seem feasible or safe.
# 
# Score? Temperature? Playing surface? Quarter of the game? How could we possible use those to generate better outcomes without fundamentally altering the game?
# 
# So, instead, let's focus on aspects of gameplay that could reasonably be changed by the scope of this exercise, the rules of the game.

# In[ ]:


target_name = 'Concussion'
feats = ['Territory','fair_catch','Touchback','out_of_bounds','downed']
X = full_data[feats]
y = full_data[target_name]


# In[ ]:


import graphviz 
from sklearn.tree import export_graphviz

treeclf = DecisionTreeClassifier()

treeclf.fit(X, y)
dot_data = export_graphviz(treeclf, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph


# This provides some insights (*there were only two concussions on any plays that ended in a fair catch*), but the tree is overfit and difficult to read.
# 
# Let's put some constraints on the model to control for overfitting.

# In[ ]:


treeclf_light = DecisionTreeClassifier(min_samples_split=100, min_samples_leaf=100, max_depth = 7)

treeclf_light.fit(X, y)
dot_data = export_graphviz(treeclf_light, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph


# ### Much Better
# 
# #### Now we can clearly see, of the 36 concussions on punt plays in those two seasons (the 37th concussion was on a fake punt - - see below):
# 
#  - There were **0 concussions** on punts out of bounds
#  - There were **0 concussions** on punts that went for a touchback
#  - Only **2 concussions** were on plays involving a fair catch
#  - Only ** 3 concussions** were on plays where the punt was downed without a return
# 
# ### And shorter distance punts were far more dangerous than longer distances.
# 
# #### Punt distance between:
#  - 0 and 47 yards: 18 concussions
# 
#  - 47 and 57 yards: 10 concussions
# 
#  - 57 and 61 yards: 5 concussions
# 
#  - 61+ yards: 1 concussion

# In[ ]:


#What was the deal with that punt of 0 yards? It wasn't a punt at all. The concussion occurred on a fake.

full_data['PlayDescription'].loc[2749]


# In[ ]:


#Going from the more simplistic classification based on the side of the field to look more closely at where the play originated by yardline

target_name = 'Concussion'
feats = ['100yd','fair_catch','Touchback','out_of_bounds','downed']
X = full_data[feats]
y = full_data[target_name]

treeclf_light = DecisionTreeClassifier(min_samples_split=100, min_samples_leaf=100, max_depth = 4)

treeclf_light.fit(X, y)
dot_data = export_graphviz(treeclf_light, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph


# 26 of the 37 concussions originated inside of the 30 yard line because the incentives greatly favor teams taking a shot at returning the punt

# In[ ]:


#Adding a static variable for graphing purposes
full_data['punt'] = "Punt"


# With this variable, I can create all of the below graphs that show us the frequency of these outcomes by where the punt play starts on the field.
# 
# The x-axis is the yard line (where 60 is the opponent's 40, 70 is the opponent's 30, etc.)

# In[ ]:


g = sns.catplot(x='100yd', y='Concussion', data=full_data,orient='h', height= 7, aspect = 2)


# In[ ]:


#One bar
g = sns.catplot(x='100yd', y='punt', data=full_data, hue='returned',orient='h', height= 7, aspect = 2, palette = 'Greys', alpha=1)


# In[ ]:


#Two bars (just for different visualization types)

g = sns.catplot(x='100yd', y='returned', data=full_data,orient='h',height= 7, aspect = 2, palette='Blues')


# In[ ]:


g = sns.catplot(x='100yd', y='punt', data=full_data,orient='h', hue='fair_catch',height= 7, aspect = 2, palette='Blues', alpha=.8)


# In[ ]:


#Two bars (just for different visualization types)

g = sns.catplot(x='100yd', y='fair_catch', data=full_data,orient='h',height= 7, aspect = 2, palette='Blues')


# Finally, some breakdowns to aid with the potential impact of any rule changes

# In[ ]:


full_data['downed'].value_counts()


# In[ ]:


full_data['fair_catch'].value_counts()


# In[ ]:


full_data[full_data['100yd']<31].fair_catch.value_counts()


# In[ ]:


full_data[full_data['100yd']<31].downed.value_counts()


# In[ ]:


full_data[(full_data['100yd']<31) & (full_data['fair_catch']==True)]


# In[ ]:


full_data[full_data['100yd']<31].distance.mean()


# In[ ]:


full_data[full_data['100yd']<31].returned.value_counts()


# In[ ]:


full_data.returned.value_counts()


# In[ ]:


full_data.groupby('returned')['100yd'].mean()


# In[ ]:


full_data.groupby('returned')['100yd'].median()


# In[ ]:


full_data.groupby('fair_catch')['100yd'].mean()


# In[ ]:


full_data.groupby('fair_catch')['100yd'].median()


# In[ ]:


g = sns.catplot(x='100yd', y='punt', data=full_data, hue='out_of_bounds',orient='h', height= 7, aspect = 2,palette = 'Greens')


# In[ ]:


g = sns.catplot(x='100yd', y='out_of_bounds', data=full_data,orient='h',height= 7, aspect = 2, palette='Greens')


# In[ ]:


g = sns.catplot(x='100yd', y='punt', data=full_data, hue='downed',orient='h', height= 7, aspect = 2,palette = 'Reds')


# In[ ]:


g = sns.catplot(x='100yd', y='downed', data=full_data,orient='h',height= 7, aspect = 2, palette='Reds')


# In[ ]:


g = sns.catplot(x='100yd', y='punt', data=full_data, hue='Touchback',orient='h', height= 7, aspect = 2,palette = 'Purples')


# In[ ]:


g = sns.catplot(x='100yd', y='Touchback', data=full_data,orient='h',height= 7, aspect = 2, palette='Purples')


# # Conclusion
# 
# Several things are clear:
# 
#  - Returns are dangerous
#  - More punts are returned when the punt starts inside the 30
#  - Fair catches are a safer outcome on the play and fewer fair catches happen on punts inside the 30
#  - The rule change has to decrease the number of live returns and incentivize fair catches could be a good way to do that
#  
#  ## Proposed rule changes are included in the attached slide deck

# In[ ]:




