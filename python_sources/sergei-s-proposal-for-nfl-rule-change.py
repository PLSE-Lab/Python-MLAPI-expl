#!/usr/bin/env python
# coding: utf-8

# **Executive Summary**
# 
# We would like to give coaches an incentive to "go for it" on fourth down more often. We propose the following amendment to Section 33, Article 6:
# 
# *When the offensive team fails to reach the line to gain on fourth down, the ball is moved from the previous spot half the distance to the goal line of the defensive team.*
# 
# Our simulations, which you can run below, show that with this rule change, there would be between 1 and 12 fewer concussions per season and between 2 and 9 more points per game. Whenever safety and scoring improves, we believe that most NFL stakeholders are happier.
# 

# <img src="https://www.playsmartplaysafe.com/wp-content/uploads/2018/01/injury-data-slide-1.jpg" width="640"/>

# In[ ]:


import numpy as np
import pandas as pd
import random


# In[ ]:


DATA_DIR = '../input/NFL-Punt-Analytics-Competition/'
video_review_df = pd.read_csv(DATA_DIR + 'video_review.csv', dtype='unicode')
# Load a dataset of all plays between 2016 and 2017
DATA_DIR = '../input/nflscraprdataplay-by-play-data-20162017/'
pbp_df = pd.read_csv(DATA_DIR + 'pre_pbp_2016.csv', dtype='unicode')
pbp_df = pbp_df.append(pd.read_csv(DATA_DIR + 'reg_pbp_2016.csv', dtype='unicode'), ignore_index=True)
pbp_df = pbp_df.append(pd.read_csv(DATA_DIR + 'post_pbp_2016.csv', dtype='unicode'), ignore_index=True)
pbp_df = pbp_df.append(pd.read_csv(DATA_DIR + 'pre_pbp_2017.csv', dtype='unicode'), ignore_index=True)
pbp_df = pbp_df.append(pd.read_csv(DATA_DIR + 'reg_pbp_2017.csv', dtype='unicode'), ignore_index=True)
pbp_df = pbp_df.append(pd.read_csv(DATA_DIR + 'post_pbp_2017.csv', dtype='unicode'), ignore_index=True)

print('The following plays are distinguished in the play-by-play dataset: %s.'     % ', '.join([str(pt) for pt in pbp_df.play_type.unique()]))


# In[ ]:


# yardline_100 is the number of yards left till the end zone of the defensive team
numeric_cols = ['down', 'ep', 'epa', 'qtr', 'third_down_converted', 'yardline_100', 'yards_gained', 'ydstogo']
pbp_df[numeric_cols] = pbp_df[numeric_cols].apply(pd.to_numeric)


# In[ ]:


concussion_count = 217 + 235
kickoff_concussion_count = int(0.12*concussion_count)
punt_concussion_count = len(video_review_df)
punt_return_concussion_count = 22
#
game_count = len(pbp_df.game_id.unique())
pass_run_concussion_ceiling = concussion_count - punt_concussion_count - kickoff_concussion_count
#
pass_run_count = len(pbp_df[pbp_df.play_type == 'pass']) + len(pbp_df[pbp_df.play_type == 'run'])
punt_return_count = len(pbp_df[(pbp_df.play_type == 'punt')                        & (pbp_df.punt_in_endzone == '0')                        & (pbp_df.punt_out_of_bounds == '0')                        & (pbp_df.punt_downed == '0')                        & (pbp_df.punt_fair_catch == '0')                        ])
#
punt_return_concussion_risk = punt_return_concussion_count / punt_return_count
pass_run_concussion_risk = pass_run_concussion_ceiling / pass_run_count


# In[ ]:


print('According to the table above, there was a total of 217 + 235 = %d concussions during' 
      ' all NFL games between 2016 and 2017.'\
      % concussion_count)
print('After reviewing the provided videos, we count that in the same time period, there were'
      ' at least %d concussions on punt returns (out of %d concussions on all punts).'\
      % (punt_return_concussion_count, punt_concussion_count))
print('According to the challenge description, 12%% of concussions in the 2015-2017 seasons'
      ' occurred on kickoffs. Assuming that the risk of kickoff concussions had remained constant'
      ' before a new rule was implemented in 2018, we can esimate that there were %d concussions'
      ' on kickoffs during the 2016-2017 seasons.'\
     % kickoff_concussion_count)
print('So we estimate that at most %d concussions occurred on pass/run plays.'     % pass_run_concussion_ceiling)
print('----------------------')
print('Since there were %d pass/run plays and %d punt returns during the 2016-2017 seasons,'
      ' we calculate that players had more than %.1f times the risk of concussions on punt returns'
      ' compared to running or passing plays.'\
      % (pass_run_count, punt_return_count, punt_return_concussion_risk / pass_run_concussion_risk))
print('Therefore, our goal is to reduce the number of punt returns by providing coaches with incentives'
     ' to "go for it" on fourth down.')
print('----------------------')


# **PROPOSED RULE CHANGE : Amend Section 33, Article 6**
# 
# When the offensive team fails to reach the line to gain on fourth down, the ball is moved from the previous spot half the distance to the goal line of the defensive team.

# In[ ]:


punts_df = pbp_df[pbp_df.play_type == 'punt']
punt_gains_by_yardline_100 = {}
for idx in punts_df.index:
    yardline_100 = pbp_df.iloc[idx].yardline_100
    try:
        gain = int(pbp_df.iloc[idx+1].yardline_100 - (100 - yardline_100))
    except:
        # Ignore very few instance where we have NaNs
        continue
    if yardline_100 not in punt_gains_by_yardline_100:
        punt_gains_by_yardline_100[yardline_100] = []
    punt_gains_by_yardline_100[yardline_100].append(gain)
    
avg_punt_gain_by_yardline_100 = {}
for yardline_100, gains in punt_gains_by_yardline_100.items():
    avg_punt_gain_by_yardline_100[yardline_100] = int(sum(gains)/len(gains))
    
def avg_punt_gain(yardline_100):
    return avg_punt_gain_by_yardline_100[yardline_100]


# In[ ]:


# We believe that this rule change will increase the number of fourth down pass/run plays, therefore reducing the number of punt returns.
# Let's estimate the impact of the rule change by running a simulation. To simplify, we will only analyze punts in the first and third quarters 
# (when clock issues can be ignored).
# We will use Expected Points (EP) to evaluate the options available to the coach.
# A coach can be conservative, neutral, or aggressive in his decisions.

# For all statistical estimates, we will use the values from the play-by-play dataset
stats_df = pbp_df[(pbp_df.qtr.isin([1,3])) & (pbp_df.play_type.isin(['run', 'pass']))]
ep_df = stats_df[(stats_df.down == 1) & (stats_df.ydstogo == 10)]
ep_df = ep_df[['yardline_100', 'ep']]
# ToDo: Investigate why EP values are not all the same  under these circumstances
ep_df  = ep_df.groupby('yardline_100').mean()

def ep_1st_and_10(yardline_100):
        return ep_df.ep[yardline_100]
    
# Estimate that probability that a fourth down is converted by using third down conversion data, because we have a lot more third down data than fourth down data

third_down_df = stats_df[stats_df.down == 3]
third_down_df = third_down_df[['third_down_converted', 'ydstogo']]
# We use the mean as the estimate for the fourth down conversion probability
third_down_df  = third_down_df.groupby(['ydstogo']).mean()

def fourth_down_converted_prob(ydstogo):
    if ydstogo < 1:
        # Sanity check
        return 1
    try:
        return third_down_df.third_down_converted[ydstogo]
    except KeyError:
        # No such scenario in our dataset, so try to approximate
        return fourth_down_converted_prob(ydstogo - 1) - 0.1

    


# In[ ]:


# Now, the simulation

def coach_fourth_down_prob_estimate(ydstogo, coach_type='neutral'):
    actual_prob = fourth_down_converted_prob(ydstogo)
    if coach_type == 'conservative':
        return actual_prob - 0.1
    if coach_type == 'aggressive':
        return actual_prob + 0.1
    #if coach_type == 'neutral':
    return actual_prob

# Our new rule says to move the ball back half the distance towards opponent's end zone
def yardline_100_after_new_rule(yardline_100_before_new_rule):
    return 100 - int(yardline_100_before_new_rule/2);
    
def simulate(coach_type):
    punt_return_reductions = 0
    min_ep_diff = 0
    for idx, play in punts_df.iterrows():
        is_actual_punt_return = play.punt_in_endzone == '0'                        and play.punt_out_of_bounds == '0'                        and play.punt_downed == '0'                        and play.punt_fair_catch == '0'                    
        next_play = pbp_df.iloc[idx+1]
        actual_ep = next_play.ep
        if np.isnan(actual_ep):
            continue


        # Let's make a decision
        # Calculate my EP (= -1 * opponent's EP), if I choose to punt
        expected_opp_yardline_100_if_punt = 100 - (play.yardline_100 - avg_punt_gain(play.yardline_100))
        ep_if_punt = -1 * ep_1st_and_10(expected_opp_yardline_100_if_punt)

        # Instead of punting, should I go for it on fourth down?

        # Calculate my minimum EP, if fourth down converted
        min_my_yardline_100_if_converted = play.yardline_100 - play.ydstogo
        try:
            min_ep_if_converted = ep_1st_and_10(min_my_yardline_100_if_converted)
        except:
            # ToDo: fix a bug
            continue

        # Calculate my minimum EP, if fourth down failed
        # For the Kaggle challenge, we deem losing 2 yards to be the worst case
        # ToDo: in the future, cases where more than 2 yards are lost should be taken into consideration
        max_opp_yardline_100_if_failed = yardline_100_after_new_rule(play.yardline_100 + 2)
        min_ep_if_failed = -1 * ep_1st_and_10(max_opp_yardline_100_if_failed)

        # Now I can make the decision
        my_success_prob = coach_fourth_down_prob_estimate(play.ydstogo, coach_type)
        min_ep_if_going_for_it = my_success_prob * min_ep_if_converted             + (1 - my_success_prob) * min_ep_if_failed
        if min_ep_if_going_for_it < ep_if_punt:
            # Nah, let's punt - just as we were
            continue
        # Go for it!
        if is_actual_punt_return:
                punt_return_reductions += 1
        fourth_down_converted = random.random() < fourth_down_converted_prob(play.ydstogo)
        if fourth_down_converted:
            simulated_ep = min_ep_if_converted
        else:
            simulated_ep = abs(min_ep_if_failed)
        # Calculate how many EPs would be gained/lost compared to the actual outcome
        min_ep_diff += (simulated_ep - actual_ep)
    concussion_reductions = int((punt_return_concussion_risk - pass_run_concussion_risk) * punt_return_reductions)
    print('If coaches were %s, there may be %d fewer punt returns, and therefore %d fewer concussions.'          % (coach_type, punt_return_reductions, concussion_reductions))
    print('Also, we expect there would be %.1f more points scored per game.'          % (min_ep_diff/game_count))
    print()
    
    
simulate('neutral')
simulate('aggressive')
simulate('conservative')
# ToDo: survey real coaches to measure a more realistic impact of the rule change

