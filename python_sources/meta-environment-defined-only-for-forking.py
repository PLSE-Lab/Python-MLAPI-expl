#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning from Scratch: Environment Fully Defined within a Python Notebook
# 
# ## Solving an Example Task of Throwing Paper into a Bin
# 
# 
# This notebook defines the environment and shows the optimal policy calculated from value iteration so that you may try and apply your own Reinforcement Learning algorithm to solve this.
# 
# The optimal policy can be imported from the data file and is fixed given the bin is at (0,0) and the probabilities are calculated as shown in the function below. 
# 
# The aim is to find the best action between throwing or moving to a better position in order to get paper into a bin. In this problem, we may throw from any position in the room but the probability of it is relative to the current distance from the bin and the direction in which the paper is thrown. Therefore the actions available are to throw the paper in any 360 degree direction or move to a new position to try and increase the probability that a throw made will go into the bin.
# 
# 

# In[ ]:


import time
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import clear_output


# The optimal policy data shows the best action for each state and some states may have more than one optimal action.
# 
# The values of -1000 in the move_x, move_y and throw_dir_2 columns are placeholder values that mean the other action was taken. In other words, -1000 in throw_dir_2 means this is a move action.
# 
# u and v can be used as vector components for quiver plots as shown below.

# In[ ]:


optimal_policy = pd.read_csv('../input/OptimalPolicy_angletol45.csv')
optimal_policy.head()


# In[ ]:


optimal_policy.dtypes


# In[ ]:


# Create Quiver plot showing current optimal policy in one cell
optimal_action_list = optimal_policy.copy()

x = optimal_action_list['state_x']
y = optimal_action_list['state_y']
u = optimal_action_list['u'].values
v = optimal_action_list['v'].values
plt.figure(figsize=(10, 10))
sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action', alpha = 0.3)
plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')
plt.title("Optimal Policy for Given Probabilities")
plt.show()


# ### Probability Function 
# 
# This function defines the probability of a sucessful throw from any given state and is calculated by the following: 
# 
#    - First, if the position is the same as the bin (i.e. the person is directly inside the bin already) then the probability is fixed to 100%. 
# 
#    - Next, we have to re-define the throwing direction in two cases to accomodate for the fact that 360 degrees is the same as 0 degrees. For example, if we are south-west of the bin and throw 350 degrees, this would be the same as -10 degrees and would then relate to a bearing from the person to the bin less than 90 correctly.
# 
#    - Then the euclidean distance is calculated followed by the max distance a person could be from the bin.
# 
#    - We then calculate the bearing from the person to the bin following the previous figure and calcualte the score bounded within a +/- 45 degree window. Throws that are closest to the true bearing score higher whilst those further away score less, anything more than 45 degrees (or less than -45 degrees) are negative and then set to a zero probability. 
# 
#    - Lastly, the overall probability is related to both the distance and direction given the current position. 

# In[ ]:


# Probability Function
def probability(bin_x, bin_y, state_x, state_y, throw_deg):


    #First throw exception rule if person is directly on top of bin:
    if((state_x==bin_x) & (state_y==bin_y)):
        probability = 1
    else:
        
        
        # To accomodate for going over the 0 degree line
        if((throw_deg>270) & (state_x<=bin_x) & (state_y<=bin_y)):
            throw_deg = throw_deg - 360
        elif((throw_deg<90) & (state_x>bin_x) & (state_y<bin_y)):
            throw_deg = 360 + throw_deg
        else:
            throw_deg = throw_deg
            
        # Calculate Euclidean distance
        distance = ((bin_x - state_x)**2 + (bin_y - state_y)**2)**0.5

        # max distance for bin will always be on of the 4 corner points:
        corner_x = [-10,-10,10,10]
        corner_y = [-10,10,-10,10]
        dist_table = pd.DataFrame()
        for corner in range(0,4):
            dist = pd.DataFrame({'distance':((bin_x - corner_x[corner])**2 + (bin_y - corner_y[corner])**2)**0.5}, index = [corner])
            dist_table = dist_table.append(dist)
        dist_table = dist_table.reset_index()
        dist_table = dist_table.sort_values('distance', ascending = False)
        max_dist = dist_table['distance'][0]
        
        distance_score = 1 - (distance/max_dist)


        # First if person is directly horizontal or vertical of bin:
        if((state_x==bin_x) & (state_y>bin_y)):
            direction = 180
        elif((state_x==bin_x) & (state_y<bin_y)):
             direction = 0
        
        elif((state_x>bin_x) & (state_y==bin_y)):
             direction = 270
        elif((state_x<bin_x) & (state_y==bin_y)):
             direction = 90
              
        # If person is north-east of bin:
        elif((state_x>bin_x) & (state_y>bin_y)):
            opp = abs(bin_x - state_x)
            adj = abs(bin_y - state_y)
            direction = 180 +  np.degrees(np.arctan(opp/adj))

        # If person is south-east of bin:
        elif((state_x>bin_x) & (state_y<bin_y)):
            opp = abs(bin_y - state_y)
            adj = abs(bin_x - state_x)
            direction = 270 +  np.degrees(np.arctan(opp/adj))

        # If person is south-west of bin:
        elif((state_x<bin_x) & (state_y<bin_y)):
            opp = abs(bin_x - state_x)
            adj = abs(bin_y - state_y)
            direction =  np.degrees(np.arctan(opp/adj))

        # If person is north-west of bin:
        elif((state_x<bin_x) & (state_y>bin_y)):
            opp = abs(bin_y - state_y)
            adj = abs(bin_x - state_x)
            direction = 90 +  np.degrees(np.arctan(opp/adj))

        direction_score = (45-abs(direction - throw_deg))/45
      
        probability = distance_score*direction_score
        if(probability>0):
            probability = probability
        else:
            probability = 0
        
    return(probability)
    
    
    


# ### Initialise State-Action Pairs
# Before applying the algorithm, we intialise each state-action value into a table. First we formthis for all throwing actions then all moving actions. 
# 
# We can throw in any direction and therefore there are 360 actions for each degree starting from north as 0 clockwise to 359 degrees. 
# 
# Although movement may seem simpler in that there are 8 possible actions (north, north east, east, etc) there are complications in that unlike being able to throw in any direction from any position, there are some movements that aren't possible. For example, if we are at the edge of the room, we cannot move beyong the boundary and this needs to be accounted for. Although this could be coded nicer, I have done this manually with the if/elif statements shown that skips the row if the position and movement is not possible. 
# 

# In[ ]:


#Define Q(s,a) table by all possible states and THROW actions initialised to 0
Q_table = pd.DataFrame()
for z in range(0,360):
    throw_direction = int(z)
    for i in range(0,21):
        state_x = int(-10 + i)
        for j in range(0,21):
            state_y = int(-10 + j)
            reward = 0
            Q = pd.DataFrame({'throw_dir':throw_direction,'move_dir':"none",'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])
            Q_table = Q_table.append(Q)
Q_table = Q_table.reset_index(drop=True)
print("Q table 1 initialised")

#Define Q(s,a) table by all possible states and MOVE actions initialised to 0

for x in range(0,21):
    state_x = int(-10 + x)
    for y in range(0,21):
        state_y = int(-10 + y)
        for m in range(0,8):
            move_dir = int(m)
            
            # skip impossible moves starting with 4 corners then edges
            if((state_x==10)&(state_y==10)&(move_dir==0)):
                continue
            elif((state_x==10)&(state_y==10)&(move_dir==2)):
                continue
                
            elif((state_x==10)&(state_y==-10)&(move_dir==2)):
                continue
            elif((state_x==10)&(state_y==-10)&(move_dir==4)):
                continue
                
            elif((state_x==-10)&(state_y==-10)&(move_dir==4)):
                continue
            elif((state_x==-10)&(state_y==-10)&(move_dir==6)):
                continue
                
            elif((state_x==-10)&(state_y==10)&(move_dir==6)):
                continue
            elif((state_x==-10)&(state_y==10)&(move_dir==0)):
                continue
                
            elif((state_x==10) & (move_dir == 1)):
                continue
            elif((state_x==10) & (move_dir == 2)):
                continue
            elif((state_x==10) & (move_dir == 3)):
                continue
                 
            elif((state_x==-10) & (move_dir == 5)):
                continue
            elif((state_x==-10) & (move_dir == 6)):
                continue
            elif((state_x==-10) & (move_dir == 7)):
                continue
                 
            elif((state_y==10) & (move_dir == 1)):
                continue
            elif((state_y==10) & (move_dir == 0)):
                continue
            elif((state_y==10) & (move_dir == 7)):
                continue
                 
            elif((state_y==-10) & (move_dir == 3)):
                continue
            elif((state_y==-10) & (move_dir == 4)):
                continue
            elif((state_y==-10) & (move_dir == 5)):
                continue
                 
            else:
                reward = 0
                Q = pd.DataFrame({'throw_dir':"none",'move_dir':move_dir,'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])
                Q_table = Q_table.append(Q)
Q_table = Q_table.reset_index(drop=True)
print("Q table 2 initialised")
Q_table.tail()


# In[ ]:


# Initialise V values for all state-action pairs
Q_table['V'] = 0


# In[ ]:


# Calculate Probability of each State-Action pair, 1 for movement else use probability function
bin_x = 0
bin_y = 0

prob_list = pd.DataFrame()
for n,action in enumerate(Q_table['throw_dir']):
    # Guarantee 100% probability if movement
    if(action == "none"):
        prob = 1
    # Calculate if thrown
    else:
        prob = probability(bin_x, bin_y, Q_table['state_x'][n], Q_table['state_y'][n], action)
    prob_list = prob_list.append(pd.DataFrame({'prob':prob}, index = [n] ))
prob_list = prob_list.reset_index(drop=True)
Q_table['prob'] = prob_list['prob']


# In[ ]:


Q_table.head()

