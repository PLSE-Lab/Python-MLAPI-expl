#!/usr/bin/env python
# coding: utf-8

# ## Get some context:
# - 2018 Rules Changes And Points of Emphasis: https://operations.nfl.com/the-rules/2018-rules-changes-and-points-of-emphasis/
# - NFL 2018 Health and Safety Report: https://annualreport.playsmartplaysafe.com/#data-injury-reduction-plan
# - NFL 2017 Injury Data: https://www.playsmartplaysafe.com/newsroom/reports/2017-injury-data/
# - ESPN coverage of NFL Call to Action: http://www.espn.com/nfl/story/_/id/24743994/really-changed-nfl-call-action-concussions
# - 2018 Kickoff Rule Changes: https://www.si.com/nfl/2018/09/07/nfl-kickoff-rule-changes-explained-onside-return-clarified
# - Chronology of Kickoff rule changes: http://www.footballzebras.com/2018/05/chronology-of-kickoff-rules-changes/
# - Relevant Concussion paper: https://journals.sagepub.com/doi/full/10.1177/0363546518804498

# ## Game Plan
# - Look at the film, be the film...
#     - Find key attributes that increase risk of concussion
#     - Note if any new rules ('Use of Helmet' Rule) exist for 2018 which possibly could have prevented that concussion
# - Compare those features to the control videos and routes of players from punts without concussions
# - The goal should be to propose rules that improve player safety, but also to maintaining the integrity of the game. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ## GAME DATA
# - <b>Game Data</b>: Game level data that specifies the type of season (pre, reg, post), week, and hosting city and team. Each game is uniquely identified across all seasons using GameKey.
# - <b>GameKey</b>: is unique
# - Two seasons: 2016, 2017
#     - 65 preseason games
#     - 256 regular season games
#     - 11 post season, 1 allstar
# - Preseason splits: [1:65], [334:398]
# - Regular split: [66:321], [399:654]
# - Postseason split: [322:332], [655:665]
# - Probowl split: 333, 666
# - You can explore this dataset on your own, but i didn't find it very helpful other than understanding the seasonal splits (above)

# In[ ]:


game_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
print(game_df.shape)
game_df.head(1)

# Cleanup Memory
del game_df


# ## PLAY INFORMATION
# - <b>Play Information</b>: Play level data that describes the type of play, possession team, score and a brief narrative of each play. Plays are uniquely identified using a its PlayID along with the corresponding GameKey. <b>PlayIDs ARE NOT UNIQUE.</b>
# - All plays are punts (just check counts of 'Play_Type')
# - This dataset has alot of information that can be useful in your analysis. 'PlayDescription' tells you the summary of the play which can be used to parse and classify each play. You know where the ball is being punted from ('YardLine') and who is punting ('Poss_Team') so you can combine this information with 'PlayDescription' to synthesize how far a player returns the ball as well as create your own 'reward' metrics for placing a value on the punt return. <b>If you're going to implement a rule change that effects how the ball is received, you should understand what plays in this dataset, you'll have affected (usually negated).</b>
#     - I'm sure many folk are considering the CFL rule for a "no yards penalty" or some variation of it. Essentially the punt receiver (PR) gets at least a 5 yard buffer against any opponent player to allow for receiving the punt. Penalties are doled out if a punt team player that is not the punter or was behind the punter upon the punt enters this restricted area. Note CFL does not have a fair catch rule. Anyway a rule where there is some restricted area/safety area for the PR would result in many plays in this play dataset being negated. It's important to understand whats being negated and understand what value to the game if any is being lost as a result.
#     - The funny CFL punt: https://www.sbnation.com/2017/9/9/16280946/cfl-punt-return-weird-plays-bc-lions-montreal-alouettes-2017

# In[ ]:


play_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
print(play_df.shape)
play_df.head(1)


# In[ ]:


# HOW MANY GAMES HAD NO PUNTS AND WHICH GAMES
stuff = []
# Collect all game id's in punt data
for element in play_df['GameKey']:
    stuff.append(element)
print('Number of games without a punt:', 666 - len(set(stuff)))

for element in [i for i in range(1, 667)]:
    if element not in set(stuff):
        print('Game', element, 'had no punts')


# - Game 1: 'Hall of Fame Game' was cancelled due to weather
# - Game 333: Probowl game
#     - If you search the game and find the box-score, there were 3 punts
# - Game 399: Was cancelled due to Hurricane Harvey :(
# - Game 666: Probowl game
#     - If you search the game and find the box-score, there were 8 punts

# - I'm gonna drop <b>alot of columns</b> because I don't find them very useful. I'll look specifically at <b>'PlayDescription'</b> to get a rough idea of how a play panned out, use that data to create one-hot encodings of types of plays (touchback, punt return, blocked kick, etc) and miscellaneous attributes of a play (fumble, muffed, etc). This will help to understand how many plays could be deemed 'interesting' (exciting, action after the catch, a blocked kick) and 'uninteresting' (out of bounds kicks, touchbacks, fair catches, etc.). This labeling is subjective and is used later to place value on the result of a punt both to a team and its fanbase.
# - Note: I'll sometimes interuse the words play and punt. Sorry if there is any confusion, I always mean the entirety of the punt play when I use those words.

# - Example Play Descriptions for one-hot-encodings:
#     - <b>Interesting outcomes</b>:
#         - <b>Returned Punt</b>: B.Nortman punts 40 yards to BUF 23, Center-C.Holba. B.Tate to BUF 34 for 11 yards (D.Payne).
#         - <b>Muffed catch</b>: S.Waters punts 36 yards to BLT 15, Center-J.Jansen. K.Clay MUFFS catch, RECOVERED by CAR-F.Whittaker at BLT 12. F.Whittaker to BLT 12 for no gain (K.Clay).
#         - <b>Blocked Punt</b>: B.Wing punt is BLOCKED by B.Carter, Center-Z.DeOssie, recovered by NYG-J.Currie at NYG 15. J.Currie to NYG 15 for no gain (J.Burris).
#         - <b>Fumbles</b>: M.Darr punts 42 yards to TEN 14, Center-J.Denney. K.Reed to TEN 21 for 7 yards (Dan.Thomas). FUMBLES (Dan.Thomas), RECOVERED by MIA-J.Denney at TEN 23. J.Denney to TEN 23 for no gain (K.Byard).
#         - <b>Touchdown</b>: J.Locke punts 61 yards to CIN 20, Center-K.McDermott. A.Erickson for 80 yards, TOUCHDOWN.
#         - <b>Fake Punt</b>: P.McAfee pass deep right to E.Swoope to PIT 8 for 35 yards (J.Gilbert).
#             - Passing: P.McAfee pass deep right to E.Swoope to PIT 8 for 35 yards (J.Gilbert).
#             - Running: C.Jones left end to PHI 43 for 30 yards (D.Sproles). Fake punt run around left end.
#                 - Lots of variations in descriptions for these bad boys
#     - Uninteresting outcomes:
#         - <b>Fair Catch</b>: J.Locke punts 47 yards to GB 10, Center-K.McDermott, fair catch by M.Hyde.
#         - <b>Downed Punt</b>: J.Locke punts 50 yards to GB 9, Center-K.McDermott, downed by MIN-J.Kearse.
#             - This is a play where the punting team controls the ball before any receiving team player after the ball has been punted
#         - <b>Touchbacks</b>: J.Hekker punts 50 yards to end zone, Center-J.Overbaugh, Touchback.
#         - <b>Out of Bounds Punt</b>: J.Schum punts 35 yards to MIN 34, Center-B.Goode, out of bounds.
#         - <b>Dead Ball</b>: B.Nortman punts 51 yards to BUF 34, Center-C.Holba. B.Tate, dead ball declared at BUF 34 for no gain.
#         - <b>No Play</b>: (:04) (Punt formation) PENALTY on ATL-M.Bosher, Delay of Game, 5 yards, enforced at ATL 49 - No Play.
#             - Some 'No Play' or '(Punt formation) Penalty' descriptions vary where a punt was executed and a penalty occurred that would negate the play, such that the punt is reattempted
#             - Such penalties include: False Start, Illegal Substitution, Delay of Game, Illegal Formation, Neutral Zone Infraction, Player Out of Bounds on Punt, Defensive 12 On-field, Ineligible Downfield Kick, Illegal Shift, Unnecessary Roughness, Roughing the Kicker, Defensive Offside, Ineligible Downfield Kick, Offensive Holding
# 
# - Note: a play may have more than one of the above classifications.

# In[ ]:


# Create condensed version of play data
keeper_columns = ['GameKey', 'PlayID', 'PlayDescription', 'Poss_Team', 'YardLine']
condensed_play_df = play_df[keeper_columns].copy()


# In[ ]:


# # Get an idea of how a play is described
# for element in (play_df['PlayDescription']):
#     print(element)


# In[ ]:


def find_that_play_word(keyword, df):
    """Help to find keywords"""
    df[keyword] = 0
    count = 0
    for i, description in enumerate(df['PlayDescription']):
        game_key = df.loc[i, 'GameKey']
        play_id = df.loc[i, 'PlayID']
        # Find keyword in lowercased string of play description
        if description.lower().find(keyword) != -1:
#             print('Keyword', keyword, 'found for (game, play):', '(' + str(game_key) + ',' + str(play_id) + ')')
#             print('Play description:', description)
#             print('---')
                
            # One-hot encode with keyword
            df.loc[i, keyword] = 1
            count += 1

    print('# of', keyword, 'occuring on a punt play:', count)


# - Choice of strings to parse for were determined based off just perusing the 'PlayDescription' until my eyes bled. There are probably cases where I'm still making poor assumptions, but I'll have to live with it.

# In[ ]:


# 'Uninteresting Outcomes'
find_that_play_word('fair catch', condensed_play_df)
find_that_play_word('touchback', condensed_play_df)
find_that_play_word('downed', condensed_play_df)
find_that_play_word(', out of bounds', condensed_play_df)
find_that_play_word('dead ball', condensed_play_df)
find_that_play_word('no play', condensed_play_df)
find_that_play_word('(punt formation) penalty on', condensed_play_df)


# In[ ]:


# Reduce play_df even further 
where_condition = (
    (condensed_play_df['fair catch'] == 1) |
    (condensed_play_df['touchback'] == 1) |
    (condensed_play_df['downed'] == 1) |
    (condensed_play_df[', out of bounds'] == 1) |
    (condensed_play_df['dead ball'] == 1) |
    (condensed_play_df['no play'] == 1) |
    (condensed_play_df['(punt formation) penalty on'] == 1))
interesting_plays_df = condensed_play_df[~where_condition].reset_index(drop=True)

print('There are now', len(interesting_plays_df), '"interesting plays" from', len(condensed_play_df), 'punt plays')
print('Proportion of interesting punts:', len(interesting_plays_df)/len(condensed_play_df))
interesting_plays_df.head(1)


# - So we can see that around 55.4% of punts result in a play that is 'uninteresting'. Maybe the punt isn't worth the time. :P
# - The touchback rate was 57.6% for kickoffs in 2016. Just for perspective of 'uninteresting outcomes'.
#     - Reference: http://www.espn.com/nfl/story/_/id/18393780/kickoff-returns-reduced-18-percentage-points-2016-season
# - Now that we have a condensed set of punt plays where something potentially interesting occurred, lets parse for the more interesting than interesting plays on punts (touchdowns, fumbles, blocks, etc.).
# - Note: If you're curious on this filtering, it does filter out some plays that involved concussions (GameKey, PlayID): (21, 2587), (234, 3278), (266, 2902), (280, 2918), (399, 3312), (607, 978)

# In[ ]:


# I only have this here for reference of what I've filtered by
uninteresting_keywords = ['fair catch', 'touchback.', 'downed', ', out of bounds', 'dead ball', 'no play',
                         '(punt formation) penalty on']
interesting_keywords = ['muffs', 'blocked by','touchdown.', 'fumble', 'ruling', 'fake punt',
                        'up the middle', 'pass', 'right end', 'left end', 'right guard',
                        'direct snap', 'touchdown nullified']


# In[ ]:


# 'Interesting outcomes'
find_that_play_word('muffs', interesting_plays_df)
find_that_play_word('blocked by', interesting_plays_df)
find_that_play_word('touchdown.', interesting_plays_df)
find_that_play_word('fumble', interesting_plays_df)
find_that_play_word('ruling', interesting_plays_df)
find_that_play_word('fake punt', interesting_plays_df)
find_that_play_word('safety', interesting_plays_df)
find_that_play_word('up the middle', interesting_plays_df)
find_that_play_word('pass', interesting_plays_df)
find_that_play_word('right end', interesting_plays_df)
find_that_play_word('left end', interesting_plays_df)
find_that_play_word('right guard', interesting_plays_df)
find_that_play_word('direct snap', interesting_plays_df)
find_that_play_word('touchdown nullified', interesting_plays_df)


# In[ ]:


# Create a dataset where plays are currently assumed to be actual punt returns 
where_condition = (
    (interesting_plays_df['muffs'] == 1) |
    (interesting_plays_df['blocked by'] == 1) |
    (interesting_plays_df['touchdown.'] == 1) |
    (interesting_plays_df['fumble'] == 1) |
    (interesting_plays_df['ruling'] == 1) |
    (interesting_plays_df['fake punt'] == 1) |
    (interesting_plays_df['safety'] == 1) |
    (interesting_plays_df['up the middle'] == 1) |
    (interesting_plays_df['pass'] == 1) |
    (interesting_plays_df['right end'] == 1) |
    (interesting_plays_df['left end'] == 1) |
    (interesting_plays_df['right guard'] == 1) |
    (interesting_plays_df['direct snap'] == 1) |
    (interesting_plays_df['touchdown nullified'] == 1))
remainder_df = interesting_plays_df[~where_condition].reset_index(drop=True)


# In[ ]:


# Drop unnecessary columns
keeper_columns = ['GameKey', 'PlayID', 'PlayDescription', 'Poss_Team', 'YardLine']
remainder_df = remainder_df[keeper_columns]
print(remainder_df.shape)
remainder_df.head()


# - So now we have a condensed set of punts that result in some return minus the above filtered 'interesting' plays.
# - We'll now look at this set of plays and extract some information from the 'PlayDescription'

# In[ ]:


# # Only use to check the playdescriptions
# for i, element in enumerate(remainder_df['PlayDescription']):
#     print(i)
#     print(element)


# In[ ]:


find_that_play_word('penalty on', remainder_df)


# - The following work-up/analysis does not adjust for penalties on the play. I know this isn't clean and big returns on punts do have a pretty good chance of a penalty was helping with the success of the return, but I'm only parsing the PlayDescription to get a rough idea of the return amounts and potential value of a return. I'll go back and parse more properly in future versions of the notebook.

# In[ ]:


'''
Need to parse through PlayDescription in order to get return distance of play and distance to touchdown
Patterns that return two distances for yardage on play are lateral plays
'''
import re

trouble_maker_index = 1402         # Don't want to make a regex for this one

# Regex for them patterns
punt_distance_pattern = re.compile(r'punts ((-?)\d+) yards? to(\s| \w+ )((-?)\d+)')
yards_gained_pattern = re.compile(r'for ((-?)\d+) yard')
no_yards_gained_pattern = re.compile(r'([A-Z]\w+) ((-?)\d+) (for no gain)')

remainder_df['punt distance'] = 0
remainder_df['side ball lands'] = ''
remainder_df['yardline received'] = 0
remainder_df['yardage on play'] = 0

for i, element in enumerate(remainder_df['PlayDescription']):
#     print(i)
    punt_distance = punt_distance_pattern.findall(element) # ('Punt distance', '', 'Side Ball Lands', 'Yardline Received')
    yards_gained = yards_gained_pattern.findall(element)   # ('Yardage on Play', '', )
    no_gain = no_yards_gained_pattern.findall(element)
    
#     print(punt_distance)
#     print(yards_gained)
#     print(no_gain)
    
    # A play that results in yards gained or lossed
    if yards_gained != []:
#         print('Punt Distance:', punt_distance[0][0])
        remainder_df.loc[i, 'punt distance'] = int(punt_distance[0][0])
        remainder_df.loc[i, 'side ball lands'] = punt_distance[0][2]
        remainder_df.loc[i, 'yardline received'] = int(punt_distance[0][3])
        
        # A normal return
        if len(yards_gained) == 1:
#             print('Yards Gained:', yards_gained[0][0])
            remainder_df.loc[i, 'yardage on play'] = int(yards_gained[0][0])
            
        # For laterals
        else:
#             print('Yards Gained:', str(yards_gained[1][0] + yards_gained[1][1]))
            remainder_df.loc[i, 'yardage on play'] = int(yards_gained[0][0]) + int(yards_gained[1][0])
            
    # A play that resulted in no gain in yards
    elif no_gain != []:
#         print('Punt Distance:', punt_distance[0][0])
        remainder_df.loc[i, 'punt distance'] = int(punt_distance[0][0])
        remainder_df.loc[i, 'side ball lands'] = punt_distance[0][2]
        remainder_df.loc[i, 'yardline received'] = int(punt_distance[0][3])
#         print('Return', no_gain[0][3])

#     print('---')


# In[ ]:


# Doing some hand processing of specific returns where the yardage gained on return was
# officially changed (I know not elegant, especially if dataframe indices change overtime)
culprits = [476, 891, 1062, 1064, 1096, 2193]
yard_changes = [14, 6, 0, 3, 0, 4]
for i, element in enumerate(culprits):
    remainder_df.loc[element, 'yardage on play'] = yard_changes[i]


# In[ ]:


remainder_df.head()


# In[ ]:


# # Create dataset for external usage
# remainder_df.to_csv('data/punt_returns.csv', index=False)


# - We'll calculate distance to a touchdown for each play to create a reward metric for each play
# - A more proper metric for the value of a punt return should also take into account the current score, time remaining, playoff implications, and return by the home team or not just to name a few factors. I don't do this just to have a more simplified model for reward. 
#     - Note that this model is just a proportion, which is also a bit flawed in the sense that if a kick is a very short kick in the punt teams territory, the chances of a reasonable return are very low although the value of the play both for the return team and fans of that team are high despite a 'reward' from my calculation will show up as being low. 

# In[ ]:


# Check if punt team is always punting from their side of the field
count = 0
for i in range(len(remainder_df)):
    team_name_len = len(remainder_df.loc[i, 'Poss_Team'])
    if remainder_df.loc[i, 'Poss_Team'] == remainder_df.loc[i, 'YardLine'][:team_name_len]:
        continue
    else:
        count += 1
print("Number of plays where punt team is punting in opponents territory:", count)
print("Proportion of plays that are in opponents territory:", count/remainder_df.shape[0])


# In[ ]:


def calculate_distance_to_td (data_sample):
    '''Calculate distance needed for touchdown for each play'''
    # Punts that land on the 50 yard line
    if data_sample['yardline received'] == 50:
        distance_to_touchdown = 50
    
    # Punting on punting team's side of field
    elif data_sample['Poss_Team'] == data_sample['YardLine'][:len(data_sample['Poss_Team'])]:
        # Ball remains on punt team's side of field
        if data_sample['side ball lands'] == data_sample['YardLine'][:len(data_sample['Poss_Team'])]:
            distance_to_touchdown = data_sample['yardline received']
        # Ball is punted to return team's side of field
        else:
            distance_to_touchdown = (50 - data_sample['yardline received']) + 50
            
    # Punting on opponents side of field
    else:
        distance_to_touchdown = (50 - data_sample['yardline received']) + 50
    return distance_to_touchdown


# In[ ]:


# Calculate the value of a punt return based solely on the proportion of yardage gained on the return
# Relative to how many yards are needed to score a touchdown from where the punt initially lands
remainder_df['reward'] = 0
for i in range(len(remainder_df)):
    yards_on_return = remainder_df.loc[i, 'yardage on play']
    distance_to_touchdown = calculate_distance_to_td(remainder_df.iloc[i, :])
    remainder_df.loc[i, 'reward'] = yards_on_return / distance_to_touchdown
#     print('Value of return:', yards_on_return / distance_to_touchdown)

remainder_df.head()


# #### <a id='return_yardage'>Plot of Return Yardage</a>
# [Jump to NGS Analysis](#ngs_analysis)

# In[ ]:


'''Plot of distribution of punt return distances'''
sns.set()
bins = [i for i in range(-15, 95, 1)]
plt.hist(remainder_df['yardage on play'], bins=bins)

plt.title('Distribution of punt return distances')
plt.xlabel('Yards')
plt.ylabel('count')
plt.show()

print(remainder_df['yardage on play'].describe())


# - Punt returns really tapper off after 20 yards. Really should consider what is the 'value' of a return when considering implementing a particular rule.

# In[ ]:


'''Plot of distribution of value of return'''
bins = [i * 0.01 for i in range(-15, 100, 1)]
plt.hist(remainder_df['reward'], bins=bins)

plt.title('Distribution of value of return')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.show()

print(remainder_df['reward'].describe())


# - So there's alot you can garner from the 'PlayDescription' and play_information dataset. This workup and analysis is to extract information that maybe useful for understanding the impact of a rule change. I'll follow up this notebook with additional versions with more analysis that incorporates the NGS dataset. Cheers!
