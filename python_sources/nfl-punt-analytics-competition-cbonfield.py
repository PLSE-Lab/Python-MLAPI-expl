#!/usr/bin/env python
# coding: utf-8

# # NFL Punt Analytics Competition
# (Last Modified: 2019/01/09) 
# 
# *Charlie Bonfield* <br/>
# 
# This notebook is organized into the following sections: 
# 1. Background 
# 2. Proposed Changes 
# 3. Exploratory Data Analysis
#     * NGS Data (Velocity/Acceleration)
#     * NGS Data (Orientation/Direction) 
#     * Game Conditions 
# 4. Summary
# 
# GitHub repo (additional code): https://github.com/cbonfield/nfl_punt_analysis

# ## Background
# 
# In recent years, there has been a more concerted effort to reduce the number of player concussions in the NFL. In light of the changes made to the helmet rule and on kickoffs prior to the 2018 season, it is clear that times are changin'! However, there have yet to be any changes made to the rules surrounding punt plays to reduce the risk of concussions. In the kernel below, I present a number of fixes that I feel would go a long way in mitigating the risk of concussions on punts, and with it, I perform some analysis to motivate the proposed changes. 
# 
# A few notes from the author about **preprocessing**:
# * The process of calculating player velocities/accelerations from the NGS data is not overly complicated, but it takes a while to run given the size of the datasets. Consequently, I did so offline so that I would not need to run all of those steps from within the kernel, and the scripts that I used to do all of the preprocessing are in my public GitHub repo for this project (here's my [script](https://github.com/cbonfield/nfl_punt_analysis/blob/master/code/preprocess_ngs_data.py)). 
# * Tying into the last point, I further trimmed the NGS data down to what I would actually need to run my analysis end-to-end from within the kernel. When coding, my personal preference is to use Atom+Hydrogen (here's a nice blog post about it: https://acarril.github.io/posts/atom-hydrogen) and it was simpler to do all of the work there before sticking it into a Kaggle kernel. I will make said datasets publicly available when this kernel is published.  

# ## **Proposed Changes**
# 
# I decided to put my set of proposed changes right at the top of my kernel for those interested in reading the changes that I would propose for punt returns without viewing all that I did for my analysis (as I'm sure the sheer volume of kernels will make this a daunting task!). Without further ado, I propose the following changes that should/could be implemented: 
# 
# ### Proposed Change 1: 
# *Limit the number of players allowed to move across the line of scrimmage prior to the punt.*
# * Since concussions that occur on punt plays occur after the ball is punted, there is more of a need to create conditions post punt that reduce player speed on all parts of the field. 
# * Decreasing the number of players crossing the line will (1) naturally move players downfield sooner and (2) prevent spaces from opening between the first and second wave of coverage that would enable players on either team to speed up and/or continue to move at high speeds.  
# * In light of the proposed formation change in PC #2, I would recommend the limit to be five players. This would allow the punting team to keep an additional blocker to protect the punter while also moving players outside to decrease the amount of unoccupied space on the return.   
# 
# ### Proposed Change 2: 
# *Require a player to move off each side of the line for both teams that must be positioned at the line of scrimmage and between the hashes and numbers.*
# * When outside players on the punting team (gunners) move downfield when the ball is snapped, almost all of the space off the line (around the outside of the hashes) is available for players to use to gain speed prior to impacting players on the opposing team. 
# * Forcing players to move off the line should, in theory, not make the rest of the field available for players to cut out wide and move downfield without resistance.  
# 
# ### Proposed Change 3
# *Add a ten yard "no blocking" zone from the line of scrimmage to enable players to move downfield.*
# * Preventing blocking close to the line of scrimmage will force players to move downfield quicker and consequently, reduce the risk of players moving without interactions with members of the opposing team for extended periods of time. This is especially important given how quickly players are able to accelerate to their top speed. 
# * This probably goes without saying, but this rule is only enforced immediately following the punt - once the play starts to develop on the return, players (should) have moved beyond this zone regardless.
# * This is in the same spirit as the fifteen yard "no blocking" zone imposed on kickoffs prior to the 2018 season. While the length of the playing field is shorter than on kickoffs due to the nature of punts, there is still much more available to players than on typical offensive/defensive series. 
# * While the purpose of this change is not to intentionally reduce the number of returned punts, it stands to reason that could, in fact, be part of the outcome. 
# 
# ### Solution Efficacy 
# * In the kernel below, I present player speed as the singlemost (and likely the only) factor that contributes to concussions on punt plays.  
#     * It is worth noting that all concussions involving two players contained either helmet-to-helmet or helmet-to-body impact. The changes to the helmet rule should do a lot to decrease the likelihood of those types of plays, but regardless, you have to be moving quickly to generate enough force to cause a concussion.  
#     * While I also considered at the effect of relative player-partner orientation/direction, the only knowledge that I gleaned from it is that players are more likely to be moving in the same direction prior to impact. In my opinion, this is expected due to the fact that concussions occur during the return. 
#     * I also include a plug about the use of helmet sensors for concussion research in the NFL - the instantaneous accelerations experienced by players are far lower than we would expect for concussions, and while this is undoubtedly a consequence the frequency of sampling for NGS data, the motion of the head/neck relative to the rest of the body is really what one needs to consider for concussions. (As an added bonus, this could even present a route to automatically identifying in-game concussion risk for players - I digress though). 
# * Towards the end of the kernel, I took a cursory look to see if there were gameflow specific changes that could be made to punts (i.e., when they would be allowed) that would decrease the risk of concussion, but in truth, I did not see anything statistically significant. In retrospect, while proposing something like limiting when teams can punt would be exciting, it may cross the line with respect to game integrity at the present time. 
# 
# ### Game Integrity 
# * Since my proposed changes relate to formation changes, I feel that the NFL would be able to implement them without much resistance from players, coaches, owners, and fans. Fundamentally, my proposed changes are all intended to reduce concussion risk by impeding player motion over the course of a punt play.   
# * While player safety is paramount, I believe my proposed changes also strike a balance between decreasing the risk of concussions and preventing players from ever actually being able to return the ball. Although it is too early to tell if the changes to the kickoff will have a lasting impact on the length of returns and/or the number of return touchdowns, I can say that as a causal football fan, I did not see any marked changes in the complexity associated with returning or defending kickoffs this season.   
# * In the process of devising my set of proposed changes, I could not think of any particular player(s) that would be at increased risk - there is still an extra player back to block the punter (and for the returner), there are less players allowed to come at the punter, and players will be less likely to go full throttle across the entire length of the field.  
# 

# In[ ]:


## IMPORTS 
import glob 
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
from scipy import interpolate
from sklearn.neighbors import KernelDensity

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
init_notebook_mode(connected=True)


# In[ ]:


## FUNCTIONS (utility/preprocessing)
DDIR = '../input/NFL-Punt-Analytics-Competition/'

def collect_outcomes(data):
    """
    Extract the punt outcome from the PlayDescription field.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info =  data['play_info']

    def _process_description(row):
        tmp_desc = row.PlayDescription
        outcome = ''

        if 'touchback' in tmp_desc.lower():
            outcome = 'touchback'
        elif 'fair catch' in tmp_desc.lower():
            outcome = 'fair catch'
        elif 'out of bounds' in tmp_desc.lower():
            outcome = 'out of bounds'
        elif 'muff' in tmp_desc.lower():
            outcome = 'muffed punt'
        elif 'downed' in tmp_desc.lower():
            outcome = 'downed'
        elif 'no play' in tmp_desc.lower():
            outcome = 'no play'
        elif 'blocked' in tmp_desc.lower():
            outcome = 'blocked punt'
        elif 'fumble' in tmp_desc.lower():
            outcome = 'fumble'
        elif 'pass' in tmp_desc.lower():
            outcome = 'pass'
        elif 'declined' in tmp_desc.lower():
            outcome = 'declined penalty'
        elif 'direct snap' in tmp_desc.lower():
            outcome = 'direct snap'
        elif 'safety' in tmp_desc.lower():
            outcome = 'safety'
        else:
            if 'punts' in tmp_desc.lower():
                outcome = 'return'
            else:
                outcome = 'SPECIAL'

        return outcome

    play_info.loc[:, 'Punt_Outcome'] = play_info.apply(_process_description, axis=1)

    def _identify_penalties(row):
        if 'penalty' in row.PlayDescription.lower():
            return 1
        else:
            return 0

    play_info.loc[:, 'Penalty_on_Punt'] = play_info.apply(_identify_penalties, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def expand_play_description(data):
    """
    Expand the PlayDescription field in a standardized fashion. This function
    extracts a number of relevant additional features from PlayDescription,
    including punt distance, post-punt field location, and a few other derived
    features.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info = data['play_info']

    def _split_punt_distance(row):
        try:
            return int(row.PlayDescription.split('punts ')[1].split('yard')[0])
        except IndexError:
            return np.nan

    def _split_field_position(row):
        try:
            return row.PlayDescription.split(',')[0].split('to ')[1]
        except IndexError:
            return ''

    def _post_punt_territory(row):
        if row.Poss_Team == row.Post_Punt_FieldSide:
            return 0
        else:
            return 1

    def _start_punt_field_position(row):
        try:
            field_position = int(row.YardLine.split(' ')[1])
        except:
            print(row.YardLine)

        if row.Poss_Team in row.YardLine:
            return field_position
        else:
            return 100 - field_position

    def _field_position_punt(row):
        if 'end zone' in row.Post_Punt_YardLine:
            return 0
        elif '50' in row.Post_Punt_YardLine:
            return 50
        else:
            try:
                yard_line = int(row.Post_Punt_YardLine.split(' ')[1])
                own_field = int(row.Post_Punt_Own_Territory)

                if not own_field:
                    return 100 - yard_line
                else:
                    return yard_line
            except:
                return -999

    play_info.loc[:, 'Punt_Distance'] = play_info.apply(_split_punt_distance, axis=1)
    play_info.loc[:, 'Post_Punt_YardLine'] = play_info.apply(_split_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_FieldSide'] = play_info.Post_Punt_YardLine.apply(lambda x: x.split(' ')[0])
    play_info.loc[:, 'Post_Punt_Own_Territory'] = play_info.apply(_post_punt_territory, axis=1)
    play_info.loc[:, 'Pre_Punt_RelativeYardLine'] = play_info.apply(_start_punt_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_RelativeYardLine'] = play_info.apply(_field_position_punt, axis=1)

    # Extract additional information from play info (home team, away team, score
    # differential, home/away punt identifier).
    play_info.loc[:, 'Home_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[0])
    play_info.loc[:, 'Away_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[1])
    play_info.loc[:, 'Home_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[0]).astype(int)
    play_info.loc[:, 'Away_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[1]).astype(int)

    def _home_away_punt_bool(row):
        if row.Home_Team == row.Poss_Team:
            return 1
        else:
            return 0

    play_info.loc[:, 'Home_Visit_Team_Punt'] = play_info.apply(_home_away_punt_bool, axis=1)

    def _get_score_differential(row):
        if not row.Home_Visit_Team_Punt:
            return int(row.Away_Points - row.Home_Points)
        else:
            return int(row.Home_Points - row.Away_Points)

    play_info.loc[:, 'Score_Differential'] = play_info.apply(_get_score_differential, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def load_data(raw_bool=False):
    """
    When called, this function will load in all of the data and do the relevant
    preprocessing (mainly just a series of merges to link the injury data with a
    few of the other data sources). The output of this function is a dictionary
    with key/value pairs that are labels/DataFrames, respectively.

    Parameters:
        raw_bool: bool (default False)
            Boolean indicating whether you wish to perform the necessary
            preprocessing steps (False) or not (True).
    """

    # Load data.
    game_data = pd.read_csv(f'{DDIR}game_data.csv')
    play_info = pd.read_csv(f'{DDIR}play_information.csv')
    play_role = pd.read_csv(f'{DDIR}play_player_role_data.csv')
    punt_data = pd.read_csv(f'{DDIR}player_punt_data.csv')

    video_injury = pd.read_csv(f'{DDIR}video_footage-injury.csv')
    video_review = pd.read_csv(f'{DDIR}video_review.csv')
    video_control = pd.read_csv(f'{DDIR}video_footage-control.csv')

    if raw_bool:
        pass
    else:
        # Rename columns to match format (between video_injury/video_control and
        # everything else).
        ren_dict = {
            'season': 'Season_Year',
            'Type': 'Season_Type',
            'Home_team': 'Home_Team',
            'gamekey': 'GameKey',
            'playid': 'PlayId'
        }

        video_injury.rename(index=str, columns=ren_dict, inplace=True)
        video_control.rename(index=str, columns=ren_dict, inplace=True)
        video_review.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)

        # Join video_review to video_injury.
        video_injury = video_injury.merge(video_review, how='outer',
                                          left_on=['Season_Year', 'GameKey', 'PlayId'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId'])

        # Process punt_data - it's possible to have multiple numbers for the same
        # player, so we'll drop number to get rid of duplicates.
        punt_data.drop('Number', axis=1, inplace=True)
        punt_data.drop_duplicates(inplace=True)

        # Add player primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='inner', on=['GSISID'])
        video_injury.rename(index=str, columns={'Position':'Player_Position'},
                            inplace=True)

        # Fix a few values in Primary_Partner_GSISID that will cause the next
        # merge to barf (one nan, one 'Unclear').
        video_injury.replace(to_replace={'Primary_Partner_GSISID':'Unclear'},
                             value=99999, inplace=True)
        video_injury.replace(to_replace={'Primary_Partner_GSISID':np.nan},
                             value=99999, inplace=True)
        video_injury.loc[:, 'Primary_Partner_GSISID'] = video_injury.Primary_Partner_GSISID.astype(int)

        # Add primary partner primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='left',
                                          left_on=['Primary_Partner_GSISID'],
                                          right_on=['GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Position':'Primary_Partner_Position'},
                            inplace=True)

        # Add punt specific play role for players to video_injury.
        play_role.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.rename(index=str, columns={'Role':'Player_Punt_Role'}, inplace=True)

        # Add punt specific play role for primary partners to video_injury.
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'Primary_Partner_GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Role':'Primary_Partner_Punt_Role'},
                            inplace=True)

    # Stick everything in a dictionary to return as output.
    out_dict = {
        'game_data': game_data,
        'play_info': play_info,
        'play_role': play_role,
        'punt_data': punt_data,
        'video_injury': video_injury,
        'video_control': video_control,
        'video_review': video_review
    }

    return out_dict

def parse_penalties(play_info_df):
    """
    Extract penalty types for plays on which we had penalties.

    Parameters:
        play_info_df: pd.DataFrame
            DataFrame containing play information.
    """

    pen_df = play_info_df.loc[play_info_df.Penalty_on_Punt == 1].reset_index(drop=True)

    def _extract_penalty_type(row):
        try:
            tmp_desc = row.PlayDescription.lower()
            pen_suff = tmp_desc.split('penalty on ')[1]
            drop_plr = pen_suff.split(', ')[1]

            penalty = drop_plr.split(',')[0]
        except:
            penalty = 'EXCEPTION'

        return penalty

    pen_df.loc[:, 'Penalty_Type'] = pen_df.apply(_extract_penalty_type, axis=1)

    return pen_df

def trim_player_partner_data(ngs_df):
    """
    Given a DataFrame with NGS data for player/partner on punt play, cut out
    the relevant NGS data.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
    """

    # Isolate player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER'].dropna().reset_index(drop=True)
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER'].dropna().reset_index(drop=True)

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = play_df.loc[play_df.Event == 'punt'].index[0]
        part_st = part_df.loc[part_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = play_df.loc[play_df.Event == 'ball_snap'].index[0]
            part_st = part_df.loc[part_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = play_df.index.min()
            part_st = part_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = play_df.loc[play_df.Event == 'tackle'].index[0] + 50
        part_ei = part_df.loc[part_df.Event == 'tackle'].index[0] + 50
        
        play_ps = play_df.loc[play_df.Event == 'play_submit'].index[0]
        part_ps = part_df.loc[part_df.Event == 'play_submit'].index[0]
        
        while play_ei > play_ps:
            play_ei -= 10

        while part_ei > part_ps:
            part_ei -= 10
    except IndexError:
        try:
            play_ei = play_df.loc[play_df.Event == 'penalty_flag'].index[0]
            part_ei = part_df.loc[part_df.Event == 'penalty_flag'].index[0]
        except IndexError:
            play_ei = play_df.index.max()
            part_ei = part_df.index.max()

    # Slice out the data that we actually need.
    play_df = play_df.iloc[play_st:play_ei]
    part_df = part_df.iloc[part_st:part_ei]

    return play_df, part_df


# ## Exploratory Analysis 
# 
# In the course of making some snappy visualizations for my slide deck, I tallied up a few statistics by hand that I figured would be worth including here. Most notably, I saw that:
# * **27** out of **37** concussed players were on the punting team. 
# * Of the ten concussed players on the receiving team, the punt returner was concussed **five** times.
# * Punt returners were involved in concussion events more than any other player on the field (**13** out of **37** times, **five** as a "concussee" and **eight** as a "concusser"). 
# 
# Without doing anything fancy, we can already start to get a sense for what sorts of things will give rise to increased risk of concussion - players that move the fastest and cover the most ground, for instance, are the most likely to be concussed. This aligned with my gut instinct, but it was interesting to see all the same. 

# To get a sense for what players were actually doing on field, however, I decided to make a plot of the player-partner trajectories superposed on a football field. The interactive plot below shows just that - the path of the player is in red, the path of the partner is in blue, and the colorbars indicate the speed of each respective player (in m/s). For ease of presentation, I indexed the punt plays (hence the number displayed by the slider), and I would suggest looking through it one frame at a time.

# In[ ]:


# Load data from plays with concussions.
WDIR = '../input/ngs-dataset-playerpartnerinjuries/'
ngs_data = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

# Add column for easy indexing.
merge_cols = ['Season_Year', 'GameKey', 'PlayID']
ind_df = ngs_data.drop_duplicates(merge_cols).reset_index(drop=True)
ind_df.loc[:, 'eventIndex'] = ind_df.index.values
play_indexes = ind_df.eventIndex.tolist()

ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
ngs_data = ngs_data.merge(ind_df, how='inner', left_on=merge_cols, right_on=merge_cols)

# Run to generate animated figure.
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

## CUSTOM
field_xaxis=dict(
        range=[0,120],
        linecolor='black',
        linewidth=2,
        mirror=True,
        showticklabels=False
)
field_yaxis=dict(
        range=[0,53.3],
        linecolor='black',
        linewidth=2,
        mirror=True,
        showticklabels=False
)
field_annotations=[
        dict(
            x=0,
            y=0.5,
            showarrow=False,
            text='HOME ENDZONE',
            textangle=270,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=24,
                color='white'
            )
        ),
        dict(
            x=1,
            y=0.5,
            showarrow=False,
            text='AWAY ENDZONE',
            textangle=90,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=24,
                color='white'
            )
        ),
        dict(
            x=float(17./120.),
            y=1,
            showarrow=False,
            text='10',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(27./120.),
            y=1,
            showarrow=False,
            text='20',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(37./120.),
            y=1,
            showarrow=False,
            text='30',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(50./120.),
            y=1,
            showarrow=False,
            text='40',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(60./120.),
            y=1,
            showarrow=False,
            text='50',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(70./120.),
            y=1,
            showarrow=False,
            text='40',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(80./120.),
            y=1,
            showarrow=False,
            text='30',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(93./120.),
            y=1,
            showarrow=False,
            text='20',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(103./120.),
            y=1,
            showarrow=False,
            text='10',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        )
]
field_shapes=[
        {
            'type': 'line',
            'x0': 10,
            'y0': 0,
            'x1': 10,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 2
            },
        },
        {
            'type': 'line',
            'x0': 110,
            'y0': 0,
            'x1': 110,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 2
            },
        },
        {
            'type': 'line',
            'x0': 20,
            'y0': 0,
            'x1': 20,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 30,
            'y0': 0,
            'x1': 30,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 40,
            'y0': 0,
            'x1': 40,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 50,
            'y0': 0,
            'x1': 50,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 60,
            'y0': 0,
            'x1': 60,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 70,
            'y0': 0,
            'x1': 70,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 80,
            'y0': 0,
            'x1': 80,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 90,
            'y0': 0,
            'x1': 90,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 100,
            'y0': 0,
            'x1': 100,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 15,
            'y0': 0,
            'x1': 15,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 25,
            'y0': 0,
            'x1': 25,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 35,
            'y0': 0,
            'x1': 35,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 45,
            'y0': 0,
            'x1': 45,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 55,
            'y0': 0,
            'x1': 55,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 65,
            'y0': 0,
            'x1': 65,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 75,
            'y0': 0,
            'x1': 75,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 85,
            'y0': 0,
            'x1': 85,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 95,
            'y0': 0,
            'x1': 95,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 105,
            'y0': 0,
            'x1': 105,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        }
]


# fill in most of layout
figure['layout']['autosize'] = True
figure['layout']['showlegend'] = False
figure['layout']['plot_bgcolor'] = '#008000'

figure['layout']['xaxis'] = field_xaxis
figure['layout']['yaxis'] = field_yaxis
figure['layout']['annotations'] = field_annotations
figure['layout']['shapes'] = field_shapes

figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': 0,
    'plotlycommand': 'animate',
    'values': play_indexes,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Play Index: ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# Make data (for single play).
plt_dicts = []
pidx = 0

sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

# Grab some stuff for labeling saved figure.
sy = sp_data.Season_Year.values[0]
gk = sp_data.GameKey.values[0]
pi = sp_data.PlayID.values[0]

plt_dict = {}
plt_dict['playIndex'] = pi
plt_dict['seasonYear'] = sy
plt_dict['gameKey'] = gk
plt_dict['playID'] = pi
plt_dicts.append(plt_dict)

# Get player/partner data (reduced).
rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

for i in range(2):
    if not i:
        ngs_dataset = rd_play_df.copy()
        plt_name = 'Player'
        color_scale = 'Reds'
        cb_loc = 1.0
    else:
        ngs_dataset = rd_part_df.copy()
        plt_name = 'Partner'
        color_scale = 'Blues'
        cb_loc = 1.1

    data_dict = {
        'x': list(ngs_dataset['x']),
        'y': list(ngs_dataset['y']),
        'mode': 'markers',
        'marker': {
            'color': list(ngs_dataset['s']),
            'colorbar': {'x':cb_loc},
            'colorscale':color_scale,
            'size':12
        },
        'name':plt_name
    }
    
    if i == 1:
        data_dict['marker']['reversescale'] = True
        
    figure['data'].append(data_dict)

# Add data for yardline trace.
data_dict = {
    'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
    'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'mode': 'text',
    'text': ['10','20','30','40','50','40','30','20','10'],
    'textposition': 'top center',
    'textfont': {
        'family': 'sans serif',
        'size': 20,
        'color': 'white'
    }
}
figure['data'].append(data_dict)

# Make frames.
for pidx in play_indexes:
    frame = {'data': [], 'name': pidx}
    try:
        sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

        # Grab some stuff for labeling saved figure.
        sy = sp_data.Season_Year.values[0]
        gk = sp_data.GameKey.values[0]
        pi = sp_data.PlayID.values[0]

        plt_dict = {}
        plt_dict['playIndex'] = pi
        plt_dict['seasonYear'] = sy
        plt_dict['gameKey'] = gk
        plt_dict['playID'] = pi
        plt_dicts.append(plt_dict)

        # Get player/partner data (reduced).
        rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

        for i in range(2):
            if not i:
                ngs_dataset = rd_play_df.copy()
                plt_name = 'Player'
                color_scale = 'Reds'
                cb_loc = 1.0
            else:
                ngs_dataset = rd_part_df.copy()
                plt_name = 'Partner'
                color_scale = 'Blues'
                cb_loc = 1.1

            data_dict = {
                'x': list(ngs_dataset['x']),
                'y': list(ngs_dataset['y']),
                'mode': 'markers',
                'marker': {
                    'color': list(ngs_dataset['s']),
                    'colorbar': {'x':cb_loc},
                    'colorscale':color_scale,
                    'size':12
                },
                'name':plt_name
            }
            
            if i == 1:
                data_dict['marker']['reversescale'] = True
                
            frame['data'].append(data_dict)

        # Add data for yardline trace.
        data_dict = {
            'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
            'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'mode': 'text',
            'text': ['10','20','30','40','50','40','30','20','10'],
            'textposition': 'top center',
            'textfont': {
                'family': 'sans serif',
                'size': 20,
                'color': 'white'
            }
        }
        frame['data'].append(data_dict)

        # Add frames.
        figure['frames'].append(frame)
        slider_step = {'args': [
            [pidx],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': pidx,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
    except TypeError:
        continue

figure['layout']['sliders'] = [sliders_dict]


# In[ ]:


iplot(figure)


# **Note**: *I did what I could to trim off irrelevant parts of the NGS data when making these visualizations - I sliced out data that occured prior to the punt and after either the tackle on the ball carrier was made or a penalty flag was thrown*. 
# 
# Clicking through the figures, we can see a few things just by eye:
# * It's common for players involved in concussions to cover large distances on the field, thus allowing them to reach top speed, prior to making contact with members of the other team (examples: play index 0, 13, 28, 36). 
# * For the most part, players look to move without much interaction with other players prior to making contact. While I would have liked to done a more thorough analysis to prove systematically that this is indeed the case, this is evident from the lack of abrupt changes to player velocities when looking through player trajectories (we would expect to see lighter patches of color in the case of significant interactions). 
# * There does not appear to be much rhyme or reason to how players move on the field, which is in line with what we would expect for punt plays - players move towards the location of the ball carrier and take whichever route would appear to minimize that distance.
# * Since each point is a time step (0.1 s), the fact that we see mostly dark trajectories indicates that players are able to accelerate quickly to their maximum (presumably sprint) speeds. Again, while this would've been nice to analyze further, it highlights the need for more prolonged player interaction over the course of the playing field to effectively inhibit that kind of rapid acceleration and sustained speed. 
# 
# However, let's dive a little deeper into the NGS data to see if we can find the most likely conditions that will lead to concussions.

# ### NGS Data: Velocity/Acceleration
# 
# For the first part of the analysis, I chose to use the NGS position/time data for players (and partners) involved in documented concussion events to try to determine if we could identify conditions leading to increased risk. Since concussions are tied to large impacts/accelerations, I started there - for each player on punt plays, I looked to the maximum acceleration experienced by each player (independent of player role):

# In[ ]:


## FUNCTIONS 
def ecdf(data):
    """
    Compute empirical cumulative distribution function for set of 1D data.

    Parameters:
        data: np.array
    Returns:
        x: np.array
            Sorted data.
        y: np.array
            ECDF(x).
    """

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y

def make_histogram(ngs_df, col_to_plot, title, add_lines=False):
    """
    Generate plotly histogram using maximum accelerations experienced by
    players.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
        col_to_plot: str
            Name of column to plot.
        title: str 
            Title to display over histogram. 
        add_lines: bool (default False)
            Boolean indicating whether we wish to superpose lines for values
            from concussed players.
    """

    trace = (
        go.Histogram(
            x=ngs_df[col_to_plot].values,
            histnorm='probability',
            nbinsx=100
        )
    )

    layout = go.Layout(
                title=title, 
                autosize=True,
                xaxis=dict(
                    range=[0,100],
                    title='Maximum Acceleration (m/s^2)'
                ),
                yaxis=dict(
                    title='Normalized Density'
                )
             )

    if add_lines:
        _ = 'placeholder'
    else:
        data = [trace]

    fig = go.Figure(data=data, layout=layout)

    return fig

def plot_ecdf(int_ecdf, inj_ecdf, title, unit, max_x):
    """
    Plot ECDF from entire set of play data and from the subset where a concussion
    occurred.

    Parameters:
        int_ecdf: np.array (values: [quantity, ecdf])
            Values from interpolated ECDF from entire dataset.
        inj_ecdf: np.array (values: [quantity, ecdf])
            ECDF/values for players in the injury set. 
        title: str 
            Title to display over figure. 
        unit: str 
            Unit to stick on horizontal axis. 
        max_x: int
            Max x-value (used for plot range).
    """

    int_trace = go.Scatter(
                    x = int_ecdf[:,0],
                    y = int_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(opacity=0.3)
                )

    inj_trace = go.Scatter(
                    x = inj_ecdf[:,0],
                    y = inj_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(color='red')
                )

    data = [int_trace, inj_trace]

    layout = go.Layout(
                title=title,
                autosize=True,
                showlegend=False,
                xaxis=dict(
                    range=[0,max_x],
                    title=f'{title} ({unit})'
                ),
                yaxis=dict(
                    title='CDF'
                )
             )

    fig = go.Figure(data=data, layout=layout)

    return fig


# In[ ]:


# Load data.
WDIR = '../input/ngs-data-summary-statistics/sumdynamics/sumdynamics/'
files = glob.glob(f'{WDIR}*.csv')
flist = []

for f in files:
    tmp_df = pd.read_csv(f)
    flist.append(tmp_df)

sum_df = pd.concat(flist, ignore_index=True)

# Drop unreasonable accelerations.
sum_df = sum_df.loc[sum_df.max_a <= 150.]

# Construct ECDF for players.
srt_spds, spd_ecdf = ecdf(sum_df.max_s.values)
int_s_ecdf = interpolate.interp1d(srt_spds, spd_ecdf)
srt_accs, acc_ecdf = ecdf(sum_df.max_a.values)
int_a_ecdf = interpolate.interp1d(srt_accs, acc_ecdf)

# Load in set of summary statistics for players involved in concussions.
WDIR = '../input/ngs-data-speedacceleration-summary-stats/'
inj_df = pd.read_csv(f'{WDIR}spd_acc_summary.csv')

ren_dict = {'season_year':'Season_Year', 'game_key':'GameKey',
            'play_id': 'PlayID'}
inj_df.rename(index=str, columns=ren_dict, inplace=True)

# Add column for player/partner action.
WDIR = '../input/NFL-Punt-Analytics-Competition/'
ppa_df = pd.read_csv(f'{WDIR}video_review.csv')
ppa_df = ppa_df.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Partner_Activity_Derived']]

mer_cols = ['Season_Year', 'GameKey', 'PlayID']
inj_df = inj_df.merge(ppa_df, how='inner', left_on=mer_cols, right_on=mer_cols)

def _identify_moving_pp(row):
    player_activity = row.Player_Activity_Derived

    if 'ing' in player_activity:
        return 1
    elif 'ed' in player_activity:
        return 0
    else:
        raise ValueError('Check derived activity!')

def _moving_play_s(row, dyn_opt):
    pp_activity = row.PP_Activity

    if dyn_opt == 's':
        return pp_activity * row.max_play_s + abs(pp_activity - 1) * row.max_part_s
    elif dyn_opt == 'a':
        return pp_activity * row.max_play_a + abs(pp_activity - 1) * row.max_part_a
    else:
        raise ValueError('Check derived activity!')

inj_df.loc[:, 'PP_Activity'] = inj_df.apply(_identify_moving_pp, axis=1)
inj_df.loc[:, 'max_move_s'] = inj_df.apply(_moving_play_s, args=('s'), axis=1)
inj_df.loc[:, 'max_move_a'] = inj_df.apply(_moving_play_s, args=('a'), axis=1)

def _get_s_cdf(x):
    return int_s_ecdf(x)

def _get_a_cdf(x):
    return int_a_ecdf(x)

inj_df.loc[:, 's_cumprob'] = inj_df.max_play_s.apply(_get_s_cdf)
inj_df.loc[:, 'a_cumprob'] = inj_df.max_play_a.apply(_get_a_cdf)
inj_df.loc[:, 's_part_cumprob'] = inj_df.max_part_s.apply(_get_s_cdf)
#inj_df.loc[:, 'a_part_cumprob'] = inj_df.max_part_a.apply(_get_a_cdf)


# In[ ]:


# Make figure (histogram), then plot.
plotcol = 'max_a'

figure = make_histogram(sum_df, plotcol, 'Maximum Acceleration')
iplot(figure, filename='acc-histogram')


# And so, we see that it's typical for players to experience accelerations on the order of *2g* (*g* = 9.81 m/s<sup>2</sup>; in physics parlance, *g* is the acceleration due to gravity and can be thought of as a standard unit for acceleration), and while rare, they can extend well above *10g*. However, for concussions, we would expect to see something on the order of about *100g*. Nevertheless, let's see if players who are concussed experience atypically large accelerations:

# In[ ]:


# Make figure (acceleration ECDF), then plot.
a_vals = np.linspace(min(srt_accs), max(srt_accs),200)
a_ecdf = np.array([int_a_ecdf(x) for x in a_vals])
a_ecdf = np.vstack([a_vals,a_ecdf]).T

inj_a_data = inj_df.loc[:, ['max_play_a', 'a_cumprob']].values

figure = plot_ecdf(a_ecdf, inj_a_data, 'Maximum Player Acceleration', 'm/s^2', 60)
iplot(figure, filename='acc-ecdf-play')


# On the plot shown above, the blue points correspond to data from all players on punts, while the red points are for data from concussed players. While there are some players who experience accelerations larger than a few *g*, these accelerations are nowhere near large enough to be commeasurate with a concussion and their spread within the empirical CDF suggests that there's not much there either. 
# 
# In my opinion, using accelerations directly from the NGS data to characterize concussions is not viable - the relative motion of the head and neck is really what matters, and since the sensors used to register all of the data are located in the shoulder pads (and the neck/head is free to move independent of the shoulders, of course), we cannot definitively look to accelerations alone. That fact notwithstanding, we also may be missing part of the picture due to the rate at which we are receiving NGS data (0.1 s time steps), as the typical time over which players exchange momentum is smaller than that. Instead, it may be more fruitful to look at maximum player speed, as players need to be moving quickly in order to impart enough energy/momentum to cause a concussion.

# In[ ]:


# Make figure (speed ECDF), then plot.
s_vals = np.linspace(min(srt_spds), max(srt_spds),200)
s_ecdf = np.array([int_s_ecdf(x) for x in s_vals])
s_ecdf = np.vstack([s_vals,s_ecdf]).T

inj_s_data = inj_df.loc[:, ['max_play_s', 's_cumprob']].values

figure = plot_ecdf(s_ecdf, inj_s_data, 'Maximum Player Speed', 'm/s', 10)
iplot(figure, filename='spd-ecdf')


# In[ ]:


# Make figure (speed ECDF), then plot.
s_vals = np.linspace(min(srt_spds), max(srt_spds),200)
s_ecdf = np.array([int_s_ecdf(x) for x in s_vals])
s_ecdf = np.vstack([s_vals,s_ecdf]).T

inj_s_data = inj_df.loc[:, ['max_part_s', 's_part_cumprob']].values

figure = plot_ecdf(s_ecdf, inj_s_data, 'Maximum Partner Speed', 'm/s', 10)
iplot(figure, filename='spd-ecdf')


# *For context, 8 m/s clocks in at right around 18 mph. The top speed recorded in the NFL during the 2018 season was 22.09 mph, registered on a run by Matt Breida of the 49ers (courtesy of NFL's Next Gen Stats website).*
# 
# While it's still not a smoking gun per se, we see a little more of a tendency towards higher maximum speeds. Thus, as we might expect, both the players that are concussed and the partners doing the concussing are moving quickly on the play prior to the concussion.

# ### NGS Data: Direction/Orientation
# 
# To complement the velocities and accelerations derived from the NGS data, I decided to dig into the angular (orientation/direction) data a little bit as well. As mentioned in the data description, orientation tells us which way the players are facing on the field at any given moment, while direction tells us the way in which the players are moving on the field. 
# 
# For each player-partner combination on plays with concussions, I took the difference of the player angle and partner angle for both orientation and direction - the thought here is that if we are able to identify conditions related to the orientation of players prior to the concussion, we can provide guidance into formation changes that will ameliorate the risk of concussions. 
# 
# Since it's difficult to visualize what actual playing conditions relative angles correspond with, here are a few examples (note that player/partner roles can be reversed in any of these scenarios):
# * 0 degrees:
#     * Direction: Player and partner are moving in precisely the **same** direction. A simple example to visualize is the player chasing the partner (or vice versa), while something more subtle could be the player backpedaling while the partner runs in the same direction towards the player. 
#     * Orientation: Player is perfectly aligned with and facing the back of the partner. 
# * 90 (or 270) degrees:
#     * Direction: Player and partner are moving **perpendicularly** to each other. As an example, think about a player moving in a straight line and being approached by the partner from a right angle. 
#     * Orientation: Player is facing forwards while the partner is facing one of the player's shoulders. 
# * 180 degrees: 
#     * Direction: Player and partner moving in **opposite** directions. This is probably the worst case scenario for a tackle, as the player and partner would be moving straight towards each other. 
#     * Orientation: Player and partner are facing each other. 
#   
# *One final note before visualizing the data - I took the time at which the player-partner distance was at a minimum as the most probable point of contact for each player-partner combination (thus leading to the concussion). There is probably a more rigorous way to identify when the contact occurred, but I figured this would be good enough to first order.*

# In[ ]:


## FUNCTIONS
def calculate_pp_distance(ngs_df):
    """
    Given the NGS data for the injury set, calculate player-partner distance.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data for player/partner on plays with a
            concussion.
    """

    # Split player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER']
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER']
    play_df = play_df.drop(['Identifier'], axis=1)
    part_df = part_df.drop(['Identifier'], axis=1)

    # Rename columns for ease of merge.
    ren_cols = ['GSISID', 'x', 'y', 'dis', 'o', 'dir', 'vx', 'vy', 's', 'ax',
                'ay', 'a', 't']
    play_cols = {x:f'play_{x}' for x in ren_cols}
    part_cols = {x:f'part_{x}' for x in ren_cols}
    play_df.rename(index=str, columns=play_cols, inplace=True)
    part_df.rename(index=str, columns=part_cols, inplace=True)

    # Perform merge.
    mer_cols = ['Season_Year', 'GameKey', 'PlayID', 'Event',
                'eventIndex', 'Time']
    pp_df = play_df.merge(part_df, how='inner', left_on=mer_cols, right_on=mer_cols)

    # Add extra columns that will assist with determining when tackle was made.
    pp_df.loc[:, 'diff_x'] = pp_df.part_x - pp_df.play_x
    pp_df.loc[:, 'diff_y'] = pp_df.part_y - pp_df.play_y
    pp_df.loc[:, 'pp_dis'] = np.sqrt(np.square(pp_df.diff_x.values) + np.square(pp_df.diff_y.values))

    return pp_df

def find_impact(pp_df):
    """
    Given a DataFrame containing player-parter NGS data (for a single play),
    identify the most likely time of impact and return all data from one second
    around that time of impact.

    Parameters:
        pp_df: pd.DataFrame
            NGS data for player/partner pair.
    """

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = pp_df.loc[pp_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = pp_df.loc[pp_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = pp_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = pp_df.loc[pp_df.Event == 'penalty_flag'].index[0]
    except IndexError:
        try:
            play_ei = pp_df.loc[pp_df.Event == 'tackle'].index[0] + 50

            play_ps = pp_df.loc[pp_df.Event == 'play_submit'].index[0]

            while play_ei > play_ps:
                play_ei -= 10

        except IndexError:
            play_ei = pp_df.index.max()

    # Slice out the data that we actually need.
    pp_df = pp_df.iloc[play_st:play_ei].reset_index(drop=True)

    # Find the DataFrame index corresponding to the minimum player-partner
    # distance.
    min_dis_index = int(pp_df.loc[pp_df.pp_dis == pp_df.pp_dis.min()].index[0])

    # Grab a few rows around that index.
    min_dis_df = pp_df.iloc[(min_dis_index-5):(min_dis_index+5)]
    min_dis_df.loc[min_dis_index,'impact'] = 1
    min_dis_df.fillna(0, inplace=True)
    min_dis_df.reset_index(drop=True, inplace=True)

    return min_dis_df

def make_radial_plot(plot_df, angle_opt, title):
    """
    Make a radial plot using a DataFrame built specifically for this purpose
    (i.e., expected column names are hard-coded).

    Parameters:
        plot_df: pd.DataFrame
            DataFrame containing data to be plotted.
        angle_opt: str
            Angle to plot ('o', 'dir', 'dir_tt').
        title: str 
            Title to display over figure. 
    """

    if angle_opt == 'o':
        plt_col = 'pp_o_diff'
        color = 'orange'
    elif angle_opt == 'dir':
        plt_col = 'pp_dir_diff'
        color = 'green'
    elif angle_opt == 'dir_tt':
        plt_col = 'pp_dir_diff'

        impact_types = plot_df.Primary_Impact_Type.tolist()
        imp_col_dict = {'Helmet-to-helmet':'red', 'Helmet-to-body':'blue'}
        color = [imp_col_dict[x] for x in impact_types]

        #pa_types = plot_df.Player_Activity_Derived.tolist()
        #pa_col_dict = {
        #    'Blocking': 'red', 'Blocked': 'orange', 'Tackling': 'green',
        #    'Tackled': 'blue'
        #}
        #color = [pa_col_dict[x] for x in pa_types]
    else:
        raise ValueError('Not a valid option!')

    angles = plot_df.loc[:,plt_col].values

    def _fix_domain(ang):
        if ang < 0:
            return 360. - abs(ang)
        else:
            return ang

    radii = plot_df.acc_rank.astype(float).tolist()
    fix_angles = [_fix_domain(x) for x in angles]

    data = [
        go.Scatterpolar(
            r = radii,
            theta = fix_angles,
            mode = 'markers',
            marker = dict(
                color = color
            )
        )
    ]

    layout = go.Layout(
        title=title,
        showlegend = False,
        font=dict(
            family='sans serif',
            size=24,
            color='black'
        ),
        polar = dict(
            radialaxis = dict(
                showticklabels = False,
                showline=False,
                ticklen=0
            )
        )
    )

    fig = go.Figure(data=data,layout=layout)

    return fig


# In[ ]:


# Load data.
WDIR = '../input/ngs-dataset-playerpartnerinjuries/'
inj_df = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

# Add column for easy indexing.
merge_cols = ['Season_Year', 'GameKey', 'PlayID']
ind_df = inj_df.drop_duplicates(merge_cols).reset_index(drop=True)
ind_df.loc[:, 'eventIndex'] = ind_df.index.values

ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
inj_df = inj_df.merge(ind_df, how='outer', left_on=merge_cols, right_on=merge_cols)

# Get player-partner processed DataFrame.
play_part_df = calculate_pp_distance(inj_df)

# Step through each play, identifying the most likely point of impact and
# grabbing a few rows around it.
impacts = []

for play_idx in range(len(inj_df.eventIndex.unique())):
    try:
        sp_df = play_part_df.loc[play_part_df.eventIndex == play_idx].reset_index(drop=True)

        # Find most probable time for impact between player/partner.
        impact_df = find_impact(sp_df)
        impacts.append(impact_df)
    except TypeError:
        continue

pp_impact_df = pd.concat(impacts, ignore_index=True)

# Add angle difference columns.
pp_impact_df.loc[:, 'pp_dir_diff'] = pp_impact_df.play_dir - pp_impact_df.part_dir
pp_impact_df.loc[:, 'pp_o_diff'] = pp_impact_df.play_o - pp_impact_df.part_o

# Isolate most likely time step for impact.
just_impact = pp_impact_df.loc[pp_impact_df.impact == 1]

# Bring in summary stats for speed/acceleration (to be used on plot).
WDIR = '../input/ngs-data-speedacceleration-summary-stats/'
spd_acc_ss = pd.read_csv(f'{WDIR}spd_acc_summary.csv')
spd_acc_ss.rename(index=str, columns={'season_year': 'Season_Year', 'game_key': 'GameKey', 'play_id': 'PlayID'}, inplace=True)

just_impact = just_impact.merge(spd_acc_ss, how='inner', left_on=merge_cols,
                                right_on=merge_cols)

# Bring in impact types/player activity.
WDIR = '../input/NFL-Punt-Analytics-Competition/'
impact_type = pd.read_csv(f'{WDIR}video_review.csv')
impact_type = impact_type.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Impact_Type']]

just_impact = just_impact.merge(impact_type, how='inner', left_on=merge_cols,
                                right_on=merge_cols)

# Pluck out the columns relevant for plotting.
#keep_columns = ['max_play_a', 'pp_dir_diff', 'pp_o_diff']
#plot_df = just_impact.loc[:, keep_columns].sort_values(by='max_play_a').reset_index(drop=True)
plot_df = just_impact.sort_values(by='max_play_a').reset_index(drop=True)
plot_df.loc[:, 'acc_rank'] = (plot_df.index.values+1)/(plot_df.index.max())


# In[ ]:


# Make orientation plot. 
plt_opt = 'o'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Orientation')
iplot(figure, filename='pp-orientation')


# When looking at relative orientation, we really don't see any sort of clear trend in the data - the relative orientations are scattered fairly uniformly, and there's not a clear relation between relative orientation and maximum player acceleration (which is captured in the radial direction - I suppressed the axis label for clarity). If we think about how players move on punt plays, this really comes as no surprise - once the punt has been caught by the punt returner, the objective becomes either blocking for the PR or blocking players on the return, regardless of where they started or which way they are facing.  

# In[ ]:


# Make direction plot. 
plt_opt = 'dir'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Direction')
iplot(figure, filename='pp-direction')


# With relative direction, we do see a bit more of a tendency towards angles between +/- 90 degrees, which is also to be expected. Since much of the action between players of opposing teams occurs after the ball is in the hands of the returner, players of both teams will likely be heading in the same direction (towards the ball) to either block or try to make a play on the PR. Unfortunately, the few exceptions that fall closer to 180 degrees are the more gruesome plays in the dataset (e.g., the Darren Sproles concussion) and consequently, those are the ones where the concussed player experiences the largest acceleration (because the momentum exchange occurs along the same axis while the players are moving faster in opposite directions along that axis). 

# In[ ]:


# Make direction plot that utilizes tackle type. 
plt_opt = 'dir_tt'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Direction')
iplot(figure, filename='pp-direction-tackletype')


# Out of curiosity, I also decided to consider relative orientation as a function of the type of contact (helmet-to-helmet: red, helmet-to-body: blue) that occurred between the players and partners on the plays. Although the helmet rule should go a long way at mitigating the risk of concussions from these types of impacts, it was interesting to see that players are often more likely to target the body of the opposing player (conciously or not) when they are moving primarily in antiparallel directions. However, without having ever played an organized game of tackle football in my life, this may just be how football players are taught to tackle early on in life.  

# On the whole, the angular information provided to us also did not provide a clear view of whether players were approaching tackles on punt plays from odd angles. With the exception of the clear, hard-to-watch plays (fast speeds, head-on collisions), players generally seem to be moving in a similar direction to their partners prior to impact and do not appear to orient themselves in any particular way before the tackle. 

# ### Game Conditions
# 
# With the NGS data really just serving to validate a couple of my gut assumptions about the nature of player dynamics on punt plays with concussions, I decided to do a little exploration into the game conditions as well to determine if I could glean any sort of actionable insights. Prior to looking into the data, I whittled down the list of possible game conditions to ones that seemed reasonable, and they were as follows:
# * Quarter: are concussions more likely to occur in some times of the game than others?
# * Score Difference: are concussions more likely to occur when a team is leading/losing by a certain score? 
# * Punting Team Field Position (prior to punt): does where the punt is being kicked from have any influence on concussions? 
# * Receiving Team Field Position (after catch): does the field position of the punt returner have any bearing on concussions? 
# 
# To see if there was anything to the game conditions listed above, I chose to perform an (ad hoc) Kolmogorov-Smirnov test to determine if the distributions of quantities associated with each game condition varied significantly between all punt plays with some kind of play downfield (return, muffed punt, and downed punt) and those with concussions. This is, of course, a far cry from any sort of proper causal inference, but I figured that given the size of our dataset, this would be a quick and dirty way to get a zeroth-order approximation of the truth. 

# In[ ]:


## FUNCTIONS
def perform_ks_test(pop_df, sam_df, col_of_interest):
    """
    Perform Kolmogorov-Smirnov test for population (all punts) and sample
    (punts with concussions) distribution of provided quantity (col_of_interest).

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity for which you'd like to conduct the KS test.
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    stat, p = stats.ks_2samp(pdata, sdata)

    return (stat, p)

def plot_distribution(pop_df, sam_df, col_of_interest, plot_hp, title):
    """
    Plot distribution of quantity (col_of_interest) for population/sample.

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity that you'd like to plot.
        plot_hp: tuple (ints/floats)
            Hyperparameter for plotting (bandwidth for KDE, number of bins for
            histogram). Index 0 contains population hyperparameter, Index 1
            contains sample hyperparameter. 
        title: str 
            Title to display over figure. 
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    # Uncomment for histograms.
    pop_trace = go.Histogram(
                    x=pdata,
                    name='Population',
                    opacity=0.75,
                    marker=dict(color='red'),
                    histnorm='probability',
                    nbinsx=plot_hp[0]
                )
    sam_trace = go.Histogram(
                    x=sdata,
                    name='Sample (concussions)',
                    opacity=0.75,
                    marker=dict(color='blue'),
                    histnorm='probability',
                    nbinsx=plot_hp[1]
                )

    """
    # Uncomment for KDEs.
    # Make KDE for each sample.
    pop_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(pdata[:, np.newaxis])
    pop_plot = np.linspace(np.min(pdata), np.max(pdata), 1000)[:, np.newaxis]
    plog_dens = pop_kde.score_samples(pop_plot)

    sam_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sdata[:, np.newaxis])
    sam_plot = np.linspace(np.min(sdata), np.max(sdata), 1000)[:, np.newaxis]
    slog_dens = sam_kde.score_samples(sam_plot)

    pop_trace = go.Scatter(
                    x=pop_plot[:, 0],
                    y=np.exp(plog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='red', width=2)
                )

    sam_trace = go.Scatter(
                    x=sam_plot[:, 0],
                    y=np.exp(slog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=2)
                )
    """

    # Make figure.
    fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
    fig.append_trace(pop_trace, 1, 1)
    fig.append_trace(sam_trace, 2, 1)
    fig['layout'].update(
        barmode='overlay', 
        title=title, 
        xaxis=dict(title='Receiving Team Points - Punting Team Points'),
        yaxis1=dict(title='Normalized Density'),
        yaxis2=dict(title='Normalized Density')
    )

    #data = [pop_trace, sam_trace]
    #layout = go.Layout(barmode='overlay')
    #fig = go.Figure(data=data, layout=layout)

    return fig


# In[ ]:


# Load data.
data_dict = load_data()
data_dict = collect_outcomes(data_dict)
data_dict = expand_play_description(data_dict)

# Focus on play information.
play_info = data_dict['play_info']
outcomes = ['return', 'downed', 'muffed punt']
play_info = play_info.loc[play_info.Punt_Outcome.isin(outcomes)].reset_index(drop=True)
play_info.loc[:, 'playIndex'] = play_info.index.values

# Split out the plays on which we had an identified concussion.
inj_df = data_dict['video_injury']
inj_df.loc[:, 'concussionPlay'] = 1
drop_cols = ['Home_Team', 'Visit_Team', 'Qtr', 'PlayDescription', 'Week']
inj_df.drop(drop_cols, axis=1, inplace=True)
inj_df.rename(index=str, columns={'PlayId':'PlayID'}, inplace=True)

# Join onto play_info.
mer_cols = ['Season_Year', 'Season_Type', 'GameKey', 'PlayID']

inj_play_info = play_info.merge(inj_df, how='inner', left_on=mer_cols,
                                right_on=mer_cols)

# Exclude plays from injury set from population set.
play_info = play_info.loc[~play_info.playIndex.isin(inj_play_info.playIndex.tolist())].reset_index(drop=True)


# In[ ]:


cols_of_interest = ['Quarter', 'Score_Differential', 'Pre_Punt_RelativeYardLine', 'Post_Punt_RelativeYardLine']
hp_dict = {'Quarter':(5,4), 'Score_Differential':(20,20), 'Pre_Punt_RelativeYardLine': (20,20),
           'Post_Punt_RelativeYardLine': (20,20)}

# Perform KS Test for quantities of interest. 
print('Kolmogorov-Smirnov Test Results:')

for col in cols_of_interest:
    # Drop some values.
    if col == 'Post_Punt_RelativeYardLine':
        pop_info = play_info.loc[play_info.Post_Punt_RelativeYardLine != -999]
    else:
        pop_info = play_info.copy()

    ks_stat, pval = perform_ks_test(pop_info, inj_play_info, col)
    print(f'quantity: {col}, test-statistic: {ks_stat}, p-value:{pval}')


# Woof. Depending on the strength of your stomach (i.e., level of statistical significance), we *might* be able to get away with saying that there's something more to score differential. However, it's clear that the other three (quarter, pre-punt yard line, post-punt yard line) aren't relevant here.
# 
# To see the nature of the distributions of score difference, let's plot them:

# In[ ]:


# Generate plot for score difference. 
col = 'Score_Differential'
figure = plot_distribution(pop_info, inj_play_info, col, hp_dict[col], 'Score Difference')
iplot(figure, filename='score-difference-dist')


# Qualms about the sample size aside, we can see that there is much more weight in the side that's less than zero (and none beyond the +14 margin), suggesting that teams/returners may be more willing to gamble on returning a punt if they need to dig their team out of a hole. However, if we look at the relative frequency of possible outcomes on all plays with "action" (returns, muffed punts, downed punts), we see:

# In[ ]:


# Make heatmap for score difference using population set. 
# Split score differential into bins.
def _bin_score_differential(row):
    row_sd = row.Score_Differential

    if row_sd <= -21:
        return '< -21'
    elif (row_sd > -21) & (row_sd <= -14):
        return '-20 to -14'
    elif (row_sd > -14) & (row_sd <= -7):
        return '-13 to -7'
    elif (row_sd > -7) & (row_sd <= -1):
        return '-6 to -1'
    elif row_sd == 0:
        return 'TIE'
    elif (row_sd > 0) & (row_sd < 7):
        return '+1 to +6'
    elif (row_sd >= 7) & (row_sd < 14):
        return '+7 to +13'
    elif (row_sd >= 14) & (row_sd < 21):
        return '+14 to +20'
    elif row_sd >= 21:
        return '> +21'
    else:
        return 'ERROR'

play_info.loc[:, 'binSD'] = play_info.apply(_bin_score_differential, axis=1)

# Tally up outcomes as a function of score differential.
reorder_indices = ['< -21', '-20 to -14', '-13 to -7', '-6 to -1', 'TIE',
                   '+1 to +6', '+7 to +13', '+14 to +20', '> +21']
binned_outcomes = play_info.pivot_table(index='binSD', columns='Punt_Outcome',
                                        aggfunc='size', fill_value=0)
test = binned_outcomes.copy()
binned_outcomes = binned_outcomes.div(binned_outcomes.sum(axis=1), axis=0)
binned_outcomes = binned_outcomes.reindex(reorder_indices)

# Plot!
trace = go.Heatmap(
            z=binned_outcomes.values.T,
            x=binned_outcomes.index.tolist(),
            y=binned_outcomes.columns.tolist())
data=[trace]

layout = go.Layout(
            title='Score Difference (Punt Outcomes)',
            xaxis=dict(title='Receiving Team Points - Punting Team Points')
        )

figure = go.Figure(data=data, layout=layout)

iplot(data, filename='sd-heatmap')


# Gliding over the panes in the heatmap above, we see that there's only about a 4% increase in return probability (at the cost of downed punts) moving from TIE games to the games with the largest deficits (-21+ <), with the major caveat being that those situations are far less probable than close games. That said, a quick hypothesis test confirms that this is not a statistically significant difference. 
# 
# In conclusion, although it would be exciting to entertain the idea of forcing teams with large leads to go for it on fourth down independent of their position on the field, there is not much of a case for it if we're looking to do it on the basis of reducing the likelihood of concussions. 

# ## Summary
# In summary, we have seen the following from our analysis: 
# * Players that move the fastest and/or are allowed to move about the field unimpeded are at the greatest risk of concussion. 
# * There is a greater risk of concussion for players on the punting team, indicating that the increased risk is rooted in the return. 
# * There were no significant trends in how players orient/angle themselves when defending punts that would lead to any actionable change in the way of how players are tackling on punts. 
# * Outside of score difference (only marginally), there is no clear game condition that points to increased risk of concussion on punt plays.
# 
# With all that in mind, the changes that I have proposed above are intended to increase the likelihood of smaller impacts at the cost of players being able to move freely about the field uninhibited during punt plays. While I would like to do more to estimate the actual *impact* of my proposed changes (for instance, I'd love to see/hear about anyone in the world who has done anything related to simulating player dynamics, as I feel that would be necessary to truly characterize the effect that formation changes would have on punts prior to going live with them), it is my hope that the analysis that I've performed is sufficient to motivate the suggested changes. 

# In[ ]:




