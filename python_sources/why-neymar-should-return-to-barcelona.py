#!/usr/bin/env python
# coding: utf-8

# # Why Neymar Should Return to Barcelona?
# 
# #### Since Neymar left FC Barcelona in 2017, there have been a lot of rumors about returning to Barcelona. Lionel Messi is 32 years old and is going through the process of renewal contract. So a lot of journalists say that a substitute will have to be found for Messi in the near future. In this notebook, I would like to say the reason why Neymar should return to Barcelona.
# <img src="https://pbs.twimg.com/media/DKV6FXkXUAYow0M.jpg" width="500px">
# 

# In[ ]:


import numpy as np
import pandas as pd
from math import pi

import matplotlib.pyplot as plt


# # Lionel Messi 'to sign new Barcelona contract and Neymar could still make summer transfer' claims ex-president Rousaud. - The Sun Football, 16 April 2020
# 
# ### .....
# ### Rousaud said that I think it's possible that Neymar could return in the summer. It does not seem to me an impossibility at all. 
# <img src="https://api.time.com/wp-content/uploads/2014/06/ap711283694022.jpg" width="400px">
# 
# 

# ## 1. Similarity of players
# #### The inner product is used to find a similarity between players. To be specific, 1 refers to the completely same as Messi, and 0 means that there is no similarity. The concept of the similarity is super simple but powerful to find suitable players for our purpose.

# In[ ]:


class Similarity():
    
    def __init__(self, name):
        self.player_name = name
    
    def PlayerList(self):    
        data = pd.read_csv('../input/fifa19/data.csv') 
        data = data[data['Overall'] > 82] # Lower Overall
        attributes = ['Name','Nationality','Club','Age','Position','Overall','Potential','Preferred Foot','Value']
        abilities = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys',
                 'Dribbling','Curve','FKAccuracy','LongPassing','BallControl',
                 'Acceleration','SprintSpeed','Agility','Reactions','Balance',
                 'ShotPower','Jumping','Stamina','Strength','LongShots',
                 'Aggression','Interceptions','Positioning','Vision','Penalties',
                 'Composure','Marking','StandingTackle','SlidingTackle']
   
        # Unit directional vector
        AbilitiesData = data[abilities]
        vec_length = np.sqrt(np.square(AbilitiesData).sum(axis=1))
        mat_abt = AbilitiesData.values.reshape(AbilitiesData.shape)
        
        for i in np.arange(AbilitiesData.shape[0]):
                mat_abt[i] = mat_abt[i,:]/vec_length[i]
                
        df_norm = pd.DataFrame(mat_abt, columns=abilities) 
    
        # Inner Product
        compared_player = df_norm[data['Name'] == self.player_name].iloc[0]
        
        data['Inner Product'] = df_norm.dot(compared_player)
        
        threshold_idp = 0.991 
        lower_potential = 85 # High potential
        substitutes = data[(data['Inner Product'] >= threshold_idp) & (data['Potential'] >= lower_potential)]
        
        if substitutes.shape[0] <= 1:
            print('There is no recommendation.')
            
        else:    
            substitutes = substitutes.sort_values(by=['Inner Product'], ascending=False)
            
            # Maximum of Player Recommendations = 3 players
            if substitutes.shape[0] > 4:
                substitutes = substitutes[0:4]
                
            substitutes = substitutes[attributes]
            substitutes.reset_index(drop=True)
            
            # Save the Scout list
            substitutes.to_csv('./scout_list.csv', index=False)
            
            standard_player = data[abilities][data.Name == self.player_name]
            
            for player_list in substitutes['Name'][1:]:
                
                add = data[abilities][data.Name == player_list]
                standard_player = standard_player.append([add])

            player_name = substitutes['Name'].values
            
            return standard_player, abilities, player_name
                   


# In[ ]:


def RadorChart(graph, abilities, player_name):
    len1 = graph.shape[0]
    len2 = graph.shape[1]
    temp = graph.values.reshape((len1, len2))
    
    tmp = pd.DataFrame(temp, columns = abilities)
    Attributes =list(tmp)
    AttNo = len(Attributes)
    
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    
    colors = ['green', 'blue', 'red', 'gold', 'orange', 'lightskyblue', 'black', 'pink']
    
    for i in range(len1):
        values = tmp.iloc[i].tolist() #
        values += values [:1]
    
        angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
        angles += angles [:1]
        
        plt.xticks(angles[:-1],Attributes)
        ax.plot(angles, values, color = colors[i])
        ax.fill(angles, values, colors[i], alpha=0.1)
        plt.figtext(0.8, 0.25-0.025*i, player_name[i], color = colors[i], fontsize=20)
    
    plt.savefig('RadarChart.png')
    plt.show()
    


# In[ ]:


Scouter = Similarity('L. Messi')
players, abilities, player_names = Scouter.PlayerList()


# ## 2. Radar Chart

# In[ ]:


abilities_view = ['PAS','SHO','SPE','PHY','DRI','DEF']


# In[ ]:


players['PAS'] = (players['ShortPassing']+players['LongPassing']+players['Crossing'])//3
players['SHO'] = (players['ShotPower']+players['LongShots']+players['Finishing'])//3
players['SPE'] = (players['SprintSpeed']+players['Acceleration']+players['Agility'])//3 
players['PHY'] = (players['Stamina']+players['Strength']+players['Balance']+players['Reactions'])//4
players['DRI'] = (players['Dribbling']+players['BallControl'])//2
players['DEF'] = (players['Marking']+players['StandingTackle']+players['Interceptions'])//3


# In[ ]:


RadorChart(players[abilities_view], abilities_view, player_names)


# <img src="https://www.fctables.com/uploads/infographics/player_versus/291274/271592/neymar-vs-lionel_messi.jpg" width="500px">
# 
# 
# #### According to the result, Dybala could also be a good replacement for Messi. However, Neymar is better than him in terms of speed. In other abilities, they are almost the same. As a big fan of Messi and Neymar, I hope I will watch their chemistry in the Camp Nou stadium. :)
# 
# ### Please hit the upvote and stay Healthy!

# In[ ]:




