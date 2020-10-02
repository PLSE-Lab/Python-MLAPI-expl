#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# The purpose of this notebook is to use National Football League (NFL) punt analytic data to attempt to propose changes to existing rules that may result in a decreased likelihood of concussion.
# 
# This report only includes injury incidents where I was able to identify a blindside block out of the total 37 incidents viewed. 
# 
# What follows is my collection of blindside block game id, play id, description and images of incident data, assessment of the proposed change as actionable or not, and how it may or may not affect the integrity of the game and punt.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# To view HTML links in this window
from IPython.display import HTML
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_colwidth', -1)  # makes columns wider


# In[ ]:


# Load the datasets, comment on lines we will not be using. 
game_data = pd.read_csv('../input/game_data.csv')
# post_2016 = pd.read_csv('../input/NGS-2016-post.csv')
# pre_2016 = pd.read_csv('../input/NGS-2016-pre.csv')
# reg_2016_wk16 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
# reg_2016_wk1317 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv')
# reg_2016_wk712 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv')
# post_2017 = pd.read_csv('../input/NGS-2017-post.csv')
# pre_2017 = pd.read_csv('../input/NGS-2017-pre.csv')
# reg_2017_wk16 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv')
# reg_2017_wk1317 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
# reg_2017_wk712 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv')
# play_information = pd.read_csv('../input/play_information.csv')
# player_role = pd.read_csv('../input/play_player_role_data.csv')
# player_punt = pd.read_csv('../input/player_punt_data.csv')
video_footage_injury = pd.read_csv('../input/video_footage-injury.csv')
video_footage_control = pd.read_csv('../input/video_footage-control.csv')
video_review = pd.read_csv('../input/video_review.csv')


# In[ ]:


# set gamekey playid column to GameKey and PlayId (for merging)
video_footage_injury.columns = ['Season_Year', 'Season_Type', 'Week', 'Home_Team', 'Visit_Team', 'Qtr', 'Play_Description', 'GameKey', 'PlayID', 'Preview_Link']

# Set up dataframe to merge info with video review files
video_injury_links = video_footage_injury[['GameKey', 'PlayID', 'Play_Description', 'Preview_Link', 'Week', 'Home_Team', 'Visit_Team', 'Qtr']]

# Create the injury_info dataframe by merging video_review and video_injury_links dataframes
injury_info = pd.merge(video_review, video_injury_links, how='left', on=['GameKey','PlayID'])

# MERGE with game info, selected columns if we want to analyze external game factors against concussions
game_data_merge = game_data[['GameKey', 'Season_Type', 'Game_Site', 'StadiumType', 'Turf', 'GameWeather', 'Game_Day','Start_Time', 'Temperature', 'OutdoorWeather']]

injury_and_game_data = pd.merge(injury_info, game_data_merge, how='left', on='GameKey')


# ## PROPOSED CHANGE #1 - Stricter Guidelines on Blindside Block Rules:
# 
# The blindside block is perhaps one of the most exciting yet vicious blocks to witness. It is the ultimate in blocks as the blocker is able to exhibit their blocking skill and strategic placement on the field. This style of block is comparable to a guerrilla warfare tactic, hiding in cover and surprising the tackler with immense force giving the tackler little to no time to react or brace. 
# 
# While there are a myriad of changes that can be made to the blindside block, I have proposed changes below, which can be combined or implemented alone. It is evident that something needs to be done, whether it be limiting the block or eliminating it completely as this style of block is visible in a large proportion of the injury videos provided. 
# 
# These changes can be applied to the specific situation where the blocker is running towards their own endzone, in the opposite direction of the Punt Returner, or to blindside blocks as a whole. Note that this analysis only reviews blindside blocks as it related to the Punt Return:
# * - Any blindside blocks must have an acceleration zone of less than or equal to 5 yds and/or the blocker must not appear to be "lining the tackler up for malicious purposes from an excessive distance ."
# * - Blindside blocks can only be applied as body-to-body or in a push-style similar to a double handed stiff arm. No helmet contact allowed whatsoever on blindside blocks (combine with option for heavier consequences if rule is broken). 
# * - Blindside blocks cannot be applied when there are 4+ players in a 3 yd radius of the block. 
# * - Heavier consequences (e.g. ejection from game/player fines) for certain circumstances where the hit appears to be malicious in nature or intended to injure. 
# 
# ## History:
# 
# Source: http://www.nfl.com/news/story/09000d5d81c8823a/printable/leagues-official-player-safety-rules
# Rule on BlindSide Blocks:
# 3. "Blindside" Block. It is an illegal "blindside" block if the initial force of the contact by a blocker's helmet (including facemask), forearm, or shoulder is to the head or neck area of an opponent when the blocker is moving toward his own endline and approaches his opponent from behind or from the side.
# 
# This page was last updated in 2012. The reviewed data pertains to the 2016 and 2017 seasons. Of the 37 video links provided, 10 our of 37 (just over 25% or 1 / 4 concussions) are caused by a blocker moving toward their own endzone, opposite direction to their Punt Returner, and applying a block to an unsuspecting chase tackler. Some of the video review shows excessive force to the helmet either by the helmet or body of the blocker. This indicates that although this type of block is deemed illegal, it is still happening on a frequent enough basis to result in a significant number of concussions (approximately 10 out of 37). 
# 
# ## ANALYSIS
# 
# Procedure: 
# 
# Merge the play links with concussion data. 
# 
# Watch each injury link and take notes on each video attempting to identify the concussion causing incident. 
# 
# Search for common occurrences or common incidents among the plays. 
# 
# Record the time of the clip, the YD line and any other data that may be visible and pertinent (e.g. player number) that will allow subsequent reviewers to identify the same incident. 
#  
# Below this you will see the code used to isolate these plays into the blindside_blocks listing, as well as the manual notes and video clip of each identified blindside block play. 

# In[ ]:


# BUCKET 1 - Blindside Blocks by Returning Team, Blocker going towards own endzone
blindside_blocks = injury_and_game_data.iloc[[4, 7, 9, 10, 17, 20, 21, 23, 30, 33]]
blindside_blocks.head(15)  # To preview merged table uncomment at beginning of line. Links for each identified play and clips are below. 


# **GAME ID 54 - PLAY ID 1045 **
# 
# (0:10s into clip at 35YD line) - #50  CAR applies blindside block to #41 PIT. #50 CAR can be seen in top right of video, calling play to left side of field. A few steps later #50 CAR locks in on #41 PIT and applies the block while heading in the direction of own endzone. #41 PIT is pushed into other players and then falls to the ground and appears injured. There are 4+ players within a short radius of the block. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153238/Punt_Return_by_Damiere_Byrd-IX9zynRU-20181119_154215217_5000k.mp4"></video>')


# **GAME ID 149 - PLAY ID 3663 **
# 
# (0:10s into clip near sideline of 46YD line) -  #55 CAR applies a blindside block on a NO player (46 YD line). The block knocks down the NO player, as well as a #42 CAR player who was engaging a different Saints player at the time and is also impacted hard. Block was applied onto the leftside - almost the back of the chaser and the chaser was 4-5Yds and unlikely to make the tackle. There are 4+ players within a short radius of the ball. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153240/Punt_by_Thomas_Morstead-eZpDKgMR-20181119_154525222_5000k.mp4"></video>')


# **GAME ID 218 - PLAY ID 3468 **
# 
# (0:04s into clip, on 35 YD line prior to PR making catch) - #37 IND applies a blindside block on #37 TEN. Both players heading towards PR during hangtime, catch not made. #37 TEN falls to ground and rolls, appears to get back into play. There is a clear line-up motion to the hit and it is applied when there are. There are 4+ players within a short radius of the block.

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153243/Punt_by_Brett_Kern-p3udGBnb-20181119_15513915_5000k.mp4"></video>')


# **GAME ID 231 - PLAY ID 1976**
# 
# (0:12s into clip on 20YD line) - #42 BAL applies blindside block on #42 CIN. Audible helmet-to-helmet contact sounds. Helmet appears to be shook loose and #42 CIN leaves feet, hitting #81 CIN and then falling to ground where #42 BAL falls on top. Blocker is running opposite the PR and Chaser. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4"></video>')


# **GAME ID 289 - PLAY ID 2341**
# 
# (0:10s into clip on 30yd line) - CAR player attempts a blindside block on #54 WSH but misses and impacts #47 WSH in body-to-helmet impact. Sound of contact is audible. Block is heading towards own endzone. There are 4+ players within a short distance. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153247/Punt_by_Tress_Way-QsI21aYF-20181119_160141260_5000k.mp4"></video>')


# **GAME ID 364 - PLAY ID 2489**
# 
# (0:07s into clip an 37 YD line) - #45 WSH attempts blindside block on #58 GB who reacts and players appear to make helmet-to-helmet contact. Block is 3-4 Yds away from PR. Both players remain standing but helmet-to-helmet contact is visible and audible. No other players within short radius of play. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153252/44_yard_Punt_by_Justin_Vogel-n7U6IS6I-20181119_161556468_5000k.mp4"></video>')


# **GAME ID 364 PLAY ID 2764**
# 
# (0:10s into clip between 45-35 YD line) - #45 WSH applies blindside block on GB player. Falls to ground. There are 4+ players in short radius. #48 WSH applies blindside block on #49 GB. There are 4+ players in short radius. Both blocks made in direction of own endzone. Multiple players can be seen being knocked to the ground and into other players. This injury occurred in the same game as the previous incident, approximately 3 minutes later.
# 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153253/Justin_Vogel_2-uaXi4twT-20181119_161626398_5000k.mp4"></video>')


# **GAME ID 392 PLAY ID 1088**
# 
# (0:16s into clip on 38YD line) - #31 KC applies blindside block to #49 TEN. #49 TEN has small reaction but is hit by body-to-helmet impact. #49 TEN appears to buckle at the knees and fall on the field, indicating he may have been knocked unconscious. The block seems to have been a 5+ YD lead up. There was 1 other players in a short radius of the block, the PR. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153258/61_yard_Punt_by_Brett_Kern-g8sqyGTz-20181119_162413664_5000k.mp4"></video>')


# **GAME ID 553 PLAY ID 1683**
# 
# (0:11s into clip on 32YD line) - #42 KC applies blindside block to #51 NYG. #42 KC leaves feet to make block. Contact very audible. #42 KC from applying the block does a turn to face opposite direction. #51 NYG falls to knees in buckling motion which would indicate being knocked unconscious. #42 KC uses other blocks to shield vision and surprises #51 NYG with severe helmet-to-helmet impact. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4"></video>')


# **GAME ID 585 PLAY ID 783**
# 
# (0:13s into clip on 32YD line) - #55 TEN applies blindside block to #44 HOU. #44HOU is in direct chase of PR. #55 TEN locks onto #44 HOU and accelerates over a 4-5 YD distance and delivers a helmet-to-helmet impact to #44 HOU who is knocked out of bounds and to the ground. 
# Texans #44 is blindside blocked in a visible helmet-to-helmet (coming from right side) hit as he is in chase of PR. Vikings #55 clearly has #44 in line and before impact lowers helmet into #44 driving him out of bounds and into the ground with great force. The video clearly shows #55 lining-up #44 and delivering a helmet-to-helmet blow and it is evident that Vikings #55 visually locked in on Texans #44.  There are audible sounds of helmet-to-helmet contact. There is one other player (PR) in a short radius of the play. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4"></video>')


# Common observations:
# 
# * - multiple instances were noted where the players appeared visibly injured, or potentially knocked unconscious.
# * - multiple instances where the player is completely blindsided or has so little time to react and brace for the block.
# * - multiple instances where helmet contact is audible. 
# * - multiple instances where there were 4+ players around the block. 
# * - multiple instances where there is a clear targeting of the tackler.
# 
# **WHAT PLAY FEATURE DOES THE PROPOSED CHANGE AFFECT and DISCUSS PLAY FEATURE AND HOW IT RELATES TO CONCUSSIONS.**
# 
# This proposed change affects the blocking aspect of the Punt, specifically after the Punt Returner (PR) has caught the punt and is running upfield. When a tackler is in chase of the PR and their focus is locked onto that player they become  susceptible to blocks from the side or blocks where a player comes from out-of-view (e.g. from behind another player) to both surprise and block the tackler who has little to no time to react or brace for impact. In some cases an upwards explosion method is used to blow up the player and helmet-to-helmet or body-to-helmet impact is likely to occur. These types of blocks are normally followed by the audible sounds of very heavy impact.
# 
# Upon manual video review of the 37 injury links, I noted 10 plays where this type of own endline blindside block was applied during a concussion injury event.
# 
# **HOW DOES THE PROPOSED CHANGE REDUCE RISK OF CONCUSSIONS?**
# 
# The hope is that through heavier penalization/fines and a more specific guideline on the blindside block, players initiating a blindside block will be more likely to:
# - deter from helmet-to-helmet or body-to-helmet blindside hits, where the unsuspecting players helmet is contacted. 
# - deter from lining players up and taking excessive acceleration distances when a minimal amount of force is required to knock unsuspecting players out of the play. 
# - deter players from any potential intention to injure another player through malicious hits such as these. 
# 
# Based on manual video analysis above, 10/37 instances of concussion occurred when video showed some kind of blindside block being applied during the injury event.
# 
# Guidelines to reduce the velocity and limit this type of block to body-to-body or push style block could reduce concussions by upwards of 25% on the Punt Play. 
# 
# **WHAT ARE THE NEW RISKS TO PLAYER SAFETY/GAME DYNAMICS FROM THE PROPOSED CHANGE?**
# 
# - If the acceleration zone is minimized, the Punt Return may lose an element of excitement/danger/fast paced feel as players will be forced to limit their acceleration zone, limiting the force of the impact and as such limiting the "hitting aesthetic" of the Punt Return. To achieve similar style impacts, players will be forced to better time their hits or possibly do more to force the tackler into the way of other players, resulting in injuries to other extremities. 
# 
# - If the style of block is limited to a push style (like a two arm stiff arm), there may be more risk to other extremities or limbs (wrists, arms, legs) as extremities may be flailing and other players could fall on them. This risk is also present under the current situation but may be a higher degree (blockers may try to blindside players into other players with a push style blindside block). The individual blocking could be susceptible to more injuries to the arms if they are forced to use a push style block against a tackler heading the opposite direction. 
# 
# - If blindside blocks are limited to more open field areas (e.g. where there are less than 4 players in a short radius [say 3 yds], the punt return may lose an element of excitement as there will be less collateral blocks. It will be more difficult for the PR team to achieve larger returns since in some instances, collateral blocking can open up large lanes. The risk to player safety will be increased as there will be less instances of collateral damage, and less chance that an individual who is blindside blocked (and has little control over extremities) will be thrashed around or fallen on by other players. 
# 
# - If the blindside block is removed from the game completely, the game dynamics will shift in favour of the kicking team. It will be easier and more likely the PR will be tackled from behind or blindside tackled so the PR will have a much higher risk of injury. This is not ideal as the risk of injury for a Punt Returner is very high and removing this block would limit the protection available to the PR.
# 
# **WHAT ARE THE RISKS TO THE INTEGRITY OF THE PUNT and THE GAME OF FOOTBALL?**
# 
# The reality is that a very minimal amount of force is required to throw an unsuspecting tackler off balance when being blindside blocked. Sometimes a small shove or push is all that is required to take the tackler out of the play. It is plausible that there are very few, if any, circumstances where the application of such explosive force to an unsuspecting victim is warranted. 
# 
# More specific rules and guidelines and/or heavier penalties/fines will serve to uphold the integrity and sportsmanship of the Punt and the game of Football. A strong stance against this type of malicious hit will serve as a reminder of the NFL's continued support of the safety of the players and integrity of the game.
# 
# Elimination of the blindside block completely is not ideal as it opens ball carrier up to much larger risk of injury (less effective blocking of tacklers). It would serve the game much better if the allowable style of tackle being applied in a blindside scenario (push style, no direct body to helmet/helmet to helmet) and/or the velocity (yard distance with which a blocker can line-up a tackler) of the block vs. removing it from the game completely. 
# 
# **WHAT ARE SOME STEPS THAT COULD BE TAKEN TO IMPLEMENT (e.g. player training,  etc.)**
# 
# The below steps are optional, and at the discretion of the group implementing any rule changes. The purpose is to illustrate that this proposed change is actionable and steps can be taken to implement the change:
# - audit of special teams training during team practice to ensure new style of blindside block is being taught/practiced/applied.
# - heavier penalties and/or fines to players initiating illegal blindside blocks for what appear to be malicious hits (e.g. lining a player up 11 yds out and helmet to helmet blow when a push could be done may result in fine and expulsion/suspension). 
# - prepare specific video examples of what is allowed and what is not allowed so there is no confusion prior to implementation. Each team should sign or report that they have educated their players on this matter and if want more extensive application, each player should sign a waiver indicating their acceptance of this block as malicious and that if intent to injure is assessed, the player will be open to pay fines/expulsion/suspension from games. 
# 
# **FURTHER DATA ANALYSIS:**
# - Use the data to isolate the blocker in each instance and assess if there was a lining up of the player and the distance at which the direction seems to alter to block the chaser. 
# - Calculate the radius that these injuries are most likely to occur at (e.g. 7 yard lineups) and the min distance where injury is not likely. This should be the line-up distance allowed for acceleration into a block. 
# 
# **IS THE PROPOSED CHANGE ACTIONABLE BY THE NFL? (CONCLUSION)**
# 
# Based on the above analysis, this proposed change is actionable by the NFL. There are various combinations of the proposed options that could be performed to find the optimal result to both reduce concussions as well as uphold the integrity of the Punt Return and game.
# 
# There will be much data to analyze, especially on the x,y coordinates of each of these plays (calculations of velocity, measurement of line-up distances etc.) to identify the optimal line-up range (e.g. 5-10 yds?) but the evidence is clear that some change is needed on the blindside block. 
# 
# This style of block is clearly one of the highest injury causing blocks (almost 25% of concussions over 2016-2017 happened when an own endzone directed blindside block was applied). There needs to be some change in order to limit both the style of block (e.g. no helmet contact, push only etc.) and the velocity with which the block is applied (e.g. no line ups in excess of 5yds). 
# 
# I would not recommend removing this block completely as it will leave the Punt Returner, who is already very susceptible to injury on the Punt Return, to even more risk of serious injury by chasers who will freely tackle from the PR's blindside. 

# ## PROPOSED CHANGE #2  - ELIMINATION OF EXCESSIVE TACKLES TO DOWNED/BEING DOWNED PUNT RETURNERS
# 
# We have all seen it in a game and quite possibly experienced it if ever a punt returner. You are wrapped up and slowly falling to the ground, with a high likelihood of being downed when all of a sudden a tackling superhero from the opposing team decides to barrel into you, making sure you stay down with a vicious hit vs. a small tap by the hand which is all that is required to down a player. The player either strikes the PR directly or ends up causing damage to themselves or a teammate who has wrapped the PR and is falling down to the ground. 
# 
# The proposed change(s) are as follows:
# - eliminate the diving at the lower body style of tackle from the PR if the PR is deemed to be going down. Accompany this with heavy penalization/fines to the player, especially when leading with the helmet. 
# 
# ## History:
# 
# Source: http://www.nfl.com/news/story/09000d5d81c8823a/printable/leagues-official-player-safety-rules
# Rule on Tackling of PR:
# 2. Kicker/Punter During Kick or Return. During a kick or during a return, it is a foul if the initial force of the contact by a defender's helmet (including facemask), forearm, or shoulder is to the head or neck area of a kicker/punter.
# 
# This page was last updated in 2012. The reviewed data pertains to the 2016 and 2017 seasons. Of the 37 video links provided, 5 out of 37 (approximately 14%) are caused by a clearly excessive or unnecessary tackling by the defending team. 
# 
# ## ANALYSIS
# 
# Procedure: Merge the play links with concussion data. Watch each injury link and take notes on each video attempting to identify the concussion causing incident. Search for videos where excessive tackling (e.g. diving into a player already going down) is committed. 
# 
# The code below will lay out the 4 plays related to this bucket, as well as links to the video and further information about the specific game (e.g. stadium, temperature etc.)
# 
# Some observations during manual review:
# - In almost all instances, either the tackler injures another player or themselves by employing a lower body directed dive at the tackler.
# - In three of four cases, the tackle was applied when there was a high likelihood the PR would have been downed. 
# - In one case, the concussion occured on a fair catch that was not caught, where the PR appears to be scrambling to make up the play. 
#  
# Below this you will see the code used to isolate these plays into the excessive_tackle listing, as well as the manual notes and video clip of each identified play.

# In[ ]:


excessive_tackling = injury_and_game_data.iloc[[18, 22, 28, 29]]
# excessive_tackling.head(10)  # Uncomment at begininng of line to view table


# **GAME ID 296 PLAY ID 2667**
# 
# (0:08s into clip on 22YD Line) - #36 is hit by own player after #36 has wrapped up the punt returner. The friendly fire incident occurs as #37 drives into an already falling JAX player, striking his own teammate in a helmet-to-helmet blow. This was an unnecessary tackle to down an already falling PR.

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153249/Punt_by_Brett_Kern-KYTnoH51-20181119_161310312_5000k.mp4"></video>')


# **GAME ID 384 PLAY ID 183**
# 
# (0:10s into slip on 34YD line) - Two Jacksonville players are diving low on the Atlanta PR when #35 Jacksonville, who is tackling from behind nearly wrapping the feet is hit by his own player who is diving at the feet of the PR from the front side. The two individuals collided and #35 appears to have been injured. This was difficult to ascertain as excessive since the PR may have escaped #35 but the style of tackling (throwing the body at the feet) limits the ability to reduce collateral damage. The tackler from the front appears to throw the body at the knees of the PR, which could have serious injury to the PR as well. The distance and direction does not seem warrant this and a standard body to body tackle could have been applied. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153257/40_yard_Punt_by_Brad_Nortman-oSbtDlHu-20181119_162303930_5000k.mp4"></video>')


# **GAME ID 473 PLAY ID 2072**
# 
# (0:11s into clip 20YD line) - #49 Raiders dives at the feet of the PR, grabbing him with one hand. The force of the dive wraps rotates him to the other side where he impacts the lower body of #50 (appears to be the knee) in a helmet-to-body impact. #50 also appears injured and is slow to rise.  Diving tackle styles noted on injured player. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153273/King_62_yard_punt-BSOws7nQ-20181119_165306255_5000k.mp4"></video>')


# **GAME ID 506 - PLAY ID 1988**
# 
# (0:11s into clip on 30YD line) -  BAL #21 calls for a fair catch, when the ball bounces he picks it up and begins to run upfield. Around the 30YD line he begins to down himself prior to being tackled. The PR is surrounded by 3 MIA players directly with a 4th and 5th in the near vicinity, so there is little to no chance of further return happening. BAL #21 begings to downhimself when he is struck helmet to helmet by MIA #34 who elected to go for a vicious tackle vs. a less forceful technical downing. The PR was about to be wrapped up by another tackler when he was struck. A fumble happens and many more players jump ontop of the PR. 

# In[ ]:


HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153274/Haack_punts_41_yards-SRJMeOc3-20181119_165546590_5000k.mp4"></video>')


# **WHAT PLAY FEATURE DOES THE PROPOSED CHANGE AFFECT and DISCUSS PLAY FEATURE AND HOW IT RELATES TO CONCUSSIONS.**
# 
# This proposed change affects the tackling aspect of the Punt. The change limits the type of tackle that can be thrown when the PR is already being downed or has a low likelihood of further progress on the punt return. 
# 
# In the 4 videos reviewed, it was noted that injury was caused by excessive forms of tackling, most notably, a throwing of the body of the tackler into oncoming PR and play. It is common that the player behind the tackler, who can be wrapping the PR up is struck by friendly fire. This type of incident occurred in 4 out of 37 videos, or just over 10% of concussions. 
# 
# **HOW DOES THE PROPOSED CHANGE REDUCE RISK OF CONCUSSIONS?**
# 
# The hope is that through heavier penalization/fines players tackling the PR will be more likely to:
# - deter from helmet-to-helmet or body-to-helmet style diving tackles to ensure the PR is downed (when a simple touch could do the job)
# - deter from lining players up and throwing their body without control at the PR, reducing the risk of friendly fire and lowerbody injury to the PR
# 
# Based on manual video analysis above, 4/37 instances of concussion occurred when video showed some kind of excessive or diving tackling, when it is likely it may not have been necessary. Guidelines to reduce the velocity and limit this type of tackle could help to eliminate concussion frequency by 10%. 
# 
# **WHAT ARE THE NEW RISKS TO PLAYER SAFETY/GAME DYNAMICS FROM THE PROPOSED CHANGE?**
# 
# - The Punt Return is a fast paced play, with players running top speed in opposing directions. Reducing or limiting the style of tackle will make it more difficult for the kicking team to tackle and down the PR. Sometimes, downs are created by a player diving and sweeping the foot with one hand, tripping the PR up. This specific rule change applies only to the situation where a PR is already going down and a second, someitmes third tackler dives into the play to ensure the PR is downed. This is a largely unnecessary technique and it is difficult to warrant such a tackle. The risk of helmet to helmet or body to helmet injury increases as the PR being downed, or player wrapping them up is susceptile to being struck on the helmet. 
# 
# **WHAT ARE THE RISKS TO THE INTEGRITY OF THE PUNT and THE GAME OF FOOTBALL?**
# 
# - Elimination of a low style of tackle would favour the Return team, making it far more difficult to tackle a PR (can't dive to cover the last bit of distance required). Limiting this style of tackle to not allow application when a player is already being downed or in some cases, already downed will serve to limit the injury directly to the PR, tackler or to other players near the play (most notably the tackler who has wrapped the PR up). 
# 
# As with blindside blocks, this exvessive diving into an already downed or being downed PR is unnecessary and a down can be achieved with a far less impactful hit. 
# 
# **WHAT ARE SOME STEPS THAT COULD BE TAKEN TO IMPLEMENT (e.g. player training,  etc.)**
# 
# The below steps are optional, and at the discretion of the group implementing any rule changes. The purpose is to illustrate that this proposed change is actionable and steps can be taken to implement the change:
# - audit of special teams training during team practice to ensure new style of tackle is not being practiced and education to players not being delivered.
# - heavier penalties and/or fines to players initiating excessive tackle styles on already downed or players being downed, where likelihood of further progress of ball is unlikely. 
# - players represent (e.g. signing documents in players agreements) that they will not apply this tackle, for risk of expulsion/fines.
# 
# **FURTHER DATA ANALYSIS:**
# - Map specific plays where excessive tackling is visually noted. Check for more plays where PR is tackled by multiple individuals and assess if injury was present. 
# - Note other instances where injury occurred as a result of over zealous tackling. 
# 
# **IS THE PROPOSED CHANGE ACTIONABLE BY THE NFL? (CONCLUSION)**
# 
# Based on the above analysis, this proposed change is actionable by the NFL. Specific rules around excessive or forceful tacking on already downed/being downed PRs could help reduce concussions by up to 10%.

# In[ ]:


## REFERENCES:
"""
1. I did not know how to embed the videos into the kernel. I used the following code from the following kernel:
Source: https://www.kaggle.com/jmbishop/penalize-all-blindside-blocks

Code used:
from IPython.display import HTML
HTML('<video width="800" height="600" controls> <source src="[SPECIFIC GAME LINK]"></video>')
"""
