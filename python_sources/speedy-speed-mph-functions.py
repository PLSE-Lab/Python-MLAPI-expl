#!/usr/bin/env python
# coding: utf-8

# # Get the Speed of the Players Involved in Injury
# #### Given a punt play with an injury, the functions below will return a data frame with the max speed (in miles/hour) and avg speed for both the injured player and the primary partner.

# In[ ]:


#import packages
import numpy as np
import pandas as pd


#  The Net Gen Stats data frames have a distance column called 'dis'.  Each value in this column shows the yard per tenth of a second traveled.  We can convert this column to miles per hour for easier use.

# In[ ]:


def convert_to_mph(dis_vector, converter):
    mph_vector = dis_vector * converter
    return mph_vector


# The conversion from 1 yard/second to miles/hour is 1 * 2.04545.  Since each row is a tenth of a second, we need to  also multiply by 10.
# 
# The main fuction below reads in one of the csv's and filters for one of the specific punt plays were an injury occured.  It then converts the distance field to MPH and returns the max and mean speeds for the player injured [1st row] and primary partner [2nd row]

# In[ ]:


def get_speed(ng_data, playId, gameKey, player, partner):
    ng_data = pd.read_csv(ng_data)
    ng_data['mph'] = convert_to_mph(ng_data['dis'], 20.455)
    player_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                               & (ng_data.GSISID == player)].sort_values('Time')
    partner_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                              & (ng_data.GSISID == partner)].sort_values('Time')
    player_grouped = player_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    player_grouped['involvement'] = 'player_injured'
    partner_grouped = partner_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    partner_grouped['involvement'] = 'primary_partner'
    return pd.concat([player_grouped, partner_grouped], axis = 0)[['involvement',
                                                                   'max_mph',
                                                                   'avg_mph']].reset_index(drop=True)


# In[ ]:


#Run an example
get_speed('../input/NGS-2016-pre.csv', 3129, 5, 31057, 32482)


# This is just a small snippet of code that I've been using in my project (which is still very much a work in progress).  I hope to continue to contribute little nuggets like this as I work towards a conclusion.

# In[ ]:




