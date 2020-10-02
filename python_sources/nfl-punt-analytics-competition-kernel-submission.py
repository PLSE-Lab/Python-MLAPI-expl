#!/usr/bin/env python
# coding: utf-8

# **The Punt Play in Football**
# 
# In football, the punt is an **offensive** play where an offensive player kicks the ball down the field. The **defending** team punt returner can either call a fair catch, catch the ball and attempt to run it up field, or let it bounce and not touch it. If the punt returner calls fair catch, the **returning team** takes over possession of the ball where the ball is caught. Otherwise, the **returning team** takes over possession where either the returner is tackled, or the ball is first touched by the **kicking team**. The goal of punting is to gain a field position advantage over the other team, hence the returning team has an incentive to do the action that best gives them a chance to get the ball toward their goal, and the kicking team has an incentive to get to the returner/ball as fast as possible to prevent them from moving forward. Good punting is an essential part of a winning a football game. 
# 
# A normal punt play looks like the following:
# <img src="https://i.imgur.com/jmzZgvs.png" />

# **Common Punting Practices and Rules**
# 
# According to NFL rule 9-1-2: Only the end men or an eligible receiver aligned behind the line of scrimmage is permitted to advance one yard beyond the line of scrimmage before the ball is kicked. So in the image above, only GR, GL, PRW, PLW, and PPR would be eligble to run down the field before the ball is kicked. Hence, in order to maximize the chances of getting to the ball, the punting team usually has two players GR and GL immedietly run toward the PR at full speed to attempt to down the ball. PLW, PRW, and PPR are also allowed down the field, but ocasionally they may perform a chip block on one of the end men before running down field at full speed. PDR1, PDR2, PDR3, PDL3, PDL2, and PDL1 usually line up in the gaps created by the offensive linemen (PLT, PLG, PLS, PRG, and PRT) in order to try and rush through to block the kick. The offensive linemen block the defensive linemen from getting to the punter and then make their way downfield to tackle the returner/down the ball.  PLR usually drops back in order to block a player rushing downfield to down the ball. VL(i,o) and VR(i,o) are players designated to specifically block the GL and the GR. They perform blocks to prevent the gunner from reaching the PR at full speed. They usually run with the gunner and attempt to force them toward the sidelines so that they have a lesser angle toward the ball. 
# 
# According to NFL rule 9-1-3, a defensive team (returning team) player who is within one yard of the line of scrimmage must lineup outside the shoulders of the snapper. This is to protect the long snapper who has his head down completely in order to snap the 15 yards to the punter. This is the only restriction on the defense when it comes to formations, and the offense has no offensive restrictions other than NFL rule 7-5-1(a) where the offensive team must have 7 players on the line of scrimmage. The NFL doesn't place too many restrictions on formations and we believe that our rule change should maintain the integrity of this principle.  
# 
# On an aside, members of the special teams/punt unit are usually not players from normal offensive personnel. Rather, the "offensive linemen" on a punt team are usually comprised of players smaller than true offensive linemen (so that they can run downfield fast to down the ball) but large enough to lay blocks on defensive linemen. These players are usually linebackers and defensive backs (as we see in the video review, the numbers 30-59 are incredibly common on punt downs for the offense). We believe that this is a strategy used to 1) prevent injuries to offensive and defensive starters, and 2) so that the players on the punt coverage are more athletic. This is important because players on the field during punt downs are all usually very fast and very strong meaning collisions happen with huge momentum. 
# 
# It is important to note that punting is an offensive down, and **generally** happens on 4th down. Teams generally use the first 3 downs to try and make the yard to gain, and use the 4th down in order to kick the ball away and force worse field position for opposition. While this is the case, an incredibly important feature of football is misdirection, and teams may decide to perform a **fake** punt or even punt on early downs to catch the opposition off guard. 
# 
# '<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153248/Rush_by_Jon_Ryan-Csg9PS77-20181119_160437472_5000k.mp4" type="video/mp4"></video>'
# 
# These kind of fake punt plays are very successful sometimes because the opposition is not ready for it or the opposition did not expect it. It is a major part of football and its essential that we do not make significant rule changes that make punting a *non offensive play* so that the excitement and strategy of fake punts is not removed. Hence it is vital we do not make restrictive formation requirements or remove the punt completely. We will attempt to use the data provided in order to create a rule that reduces potential concussions on punting while preserving as much of the excitement, strategy, and appeal a punt play can potentially provide. 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# **Basic Experimental Data Analysis**

# In[ ]:


# Loading video_review.csv data
vr = pd.read_csv('../input/video_review.csv')
vr.head(10)


# **Histogram for type of impact in our 37 concussion causing plays.**
# 
# Helmet-to-helmet hits are illegal in NFL plays, yet we see that concussions still happen from these hits. 

# In[ ]:


primary_impact_type_hist = vr['Primary_Impact_Type'].value_counts().plot(kind='bar')


# **Most common type of activity of players at the time of receiving concussion.**
# 
# According to the data, tackling is the most common method of receiving a concussion.

# In[ ]:


player_activity_derived_hist = vr['Player_Activity_Derived'].value_counts().plot(kind='bar')


# **Most common type of activity of partner of players at the time of causing concussion.**
# 
# Interestingly, the most common activity of the person giving a concussion is being blocked and tackling.

# In[ ]:


primary_partner_activity_derived = vr['Primary_Partner_Activity_Derived'].value_counts().plot(kind='bar')


# **Retrieving Punt Play Formations**
# 
# The following two sections present exact formations of all concussion plays. We will refer back to these formations later on.

# In[ ]:


# Create play_player_role_data dataframe.
pprd = pd.read_csv('../input/play_player_role_data.csv')
# Create play_information dataframe.
play_inf = pd.read_csv('../input/play_information.csv')
# Create game_inf dataframe.
game_inf = pd.read_csv('../input/game_data.csv')
# Create video_review dataframe
vr = pd.read_csv('../input/video_review.csv')

# Extract roles column from play_player_role_data.csv and check if player is on punting team.
punting_pos = {"PLS","PLG", "PLT", "PLW","PRG","PRT","PRW","PC","PPR", "PPL","P","GL","GR"}
role_col = pprd['Role']
punting_team_col = [x in punting_pos for x in role_col.tolist()]

# Adding punting_team_col to pprd dataframe.
pprd['punting_team'] = punting_team_col

# Function that merges GameKey and PlayID given a row.
def merge(row):
    return str(row['GameKey']) + '_' + str(row['PlayID'])

# Combine GameKey and PlayID to make unique Key that organizes formations.
pprd_merged_col = pprd.apply(lambda row: merge(row), axis=1)
play_inf_merged_col = play_inf.apply(lambda row: merge(row), axis=1)
vr_merged_col = vr.apply(lambda row: merge(row), axis=1)

# Adds merged_col to pprd dataframe.
pprd['game_play_key'] = pprd_merged_col
play_inf['game_play_key'] = play_inf_merged_col
vr['game_play_key'] = vr_merged_col

# Groupby game_play_key and punting_team.
pprd_formations = pprd.groupby(['game_play_key','punting_team']).apply(lambda x: sorted(x['Role'].values.tolist()))
pprd_formations = pprd_formations.rename("form")
punting_team_boolean = [x[0] in punting_pos for x in pprd_formations.tolist()]
pprd_formations = pprd_formations.apply(lambda x: ', '.join(x)).to_frame()
pprd_formations['punting_team'] = punting_team_boolean
# Sorted formations
#print(pprd_formations.keys())
pprd_and_playinf = play_inf.merge(pprd_formations, on=['game_play_key'], how='outer')
pprd_and_playinf = pprd_and_playinf.drop(columns=['Season_Year', 'GameKey', 'PlayID'])

# Merge play_information.csv with play_player_role_data.csv by game_play_key.
merged = pprd_and_playinf.merge(vr, on=['game_play_key'], how='outer')
returnTeamForms = merged[merged['punting_team'] == False]['form']

concuss = merged[merged['Turnover_Related'].notnull()]
concussReturnTeamsForms = concuss[concuss['punting_team'] == False]['form']

count_formations_punting = concuss.loc[merged['punting_team'] == False, 'form']
count_formations_recv = concuss.loc[merged['punting_team'] == True, 'form']
# print(count_formations_punting)
a = count_formations_punting.value_counts()
b = count_formations_recv.value_counts()


# In[ ]:


print("Types of formations for punting team, counts")
a.head(7)


# In[ ]:


print("Types of formations for receiving team, counts")
b.head(7)


# In[ ]:



import time
start = time.time()
NGSDICT = {
    "pre-2016" :  pd.read_csv('../input/NGS-2016-pre.csv'),
    "pre-2017" :  pd.read_csv('../input/NGS-2017-pre.csv'),
    "post-2016" :  pd.read_csv('../input/NGS-2016-post.csv'),
    "post-2017" :  pd.read_csv('../input/NGS-2017-post.csv'),
    
    "reg-2016-wk1-6" :  pd.read_csv('../input/NGS-2016-reg-wk1-6.csv'),
    "reg-2016-wk7-12" :  pd.read_csv('../input/NGS-2016-reg-wk7-12.csv'),
    "reg-2016-wk13-17" :  pd.read_csv('../input/NGS-2016-reg-wk13-17.csv'),
    
    "reg-2017-wk1-6" :  pd.read_csv('../input/NGS-2017-reg-wk1-6.csv'),
    "reg-2017-wk7-12" :  pd.read_csv('../input/NGS-2017-reg-wk7-12.csv'),
    "reg-2017-wk13-17" :  pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
}
end = time.time()
print("Time to load:")
print(end- start)


def getNGSName(season,week,season_type):
    if season_type == 'Pre':
        return 'pre-' + str(season)
    elif season_type == 'Post':
        return 'post-' + str(season) 
    else:
        if(week <= 6):
            return 'reg-' + str(season)+ '-wk1-6'
        if(week >= 7 and week <=12):
            return 'reg-' + str(season)+ '-wk7-12'
        else:
            return 'reg-' + str(season)+ '-wk13-17'


# **Counting the number of injuries on punting team vs receiving team**
# 
# We can see from the chart above, that the punting teams recieved signficantly more injuries than the kicking team. This is interesting considering the most obvious candidate for a concussion would be the punt returner (PR) as the focus of the entire kicking team is to tackle him. 

# In[ ]:


#print(pprd.keys())
PUNTPOS = {"PLS","PLG", "PLT", "PLW","PRG","PRT","PRW","PC","PPR","P","GL","GR"}
RETURNPOS = {"PDR3","PDR2","PDR1","PDL1","PDL2","PDL3","PLM","PLL","PLR","PFB","PR","VRi","VRo","VLi","VLo","VR","VL"}

cc = vr['game_play_key'].tolist()
gsi = vr['GSISID'].tolist()
#print(cc)
#print(gsi)
# print(pprd['GSISID'] == gsi[0])
c = pd.DataFrame()
for x in range(len(cc)):
    curr = (pprd[(pprd['game_play_key'] == cc[x]) & (pprd['GSISID'] == gsi[x])])
    c = c.append(curr)
#print(c)
c['punting_team'] = c['punting_team'].replace(True,'Punting Team')
c['punting_team'] = c['punting_team'].replace(False, 'Recv Team')
count = c['punting_team'].value_counts().plot(title ="Concussions per Team",kind='bar')


# **Feature Generation & Feature Selection**
# 
# In order to determine the best rule change, it is first important to understand what exactly is causing concussions. We want to know what the features are of a play involving a concussion vs the a play without one. We will attempt to find these through feature generation and a mutual information based feature selection. For each feature, we find a mutual information score. The mutual information score is a measure of dependence between two variables. The score quantifies how much each feature tells us about whether a play will result in a concussion. From here we can analyze the plays themselves while focusing on our selected features to see how they contribute to a concussion. 
# 
# We must first generate the features of each play. Each play has associated game data containing game day information (venue, stadium type, turf type, visiting team, weather, temperature, etc) and associated play data (game clock, quarter, yardline, possession, score, etc). Now using the Next Gen Stats (NGS) data, we can generate some more features for each play. For each player, we calculate their average speed, average direction, and average orientation throughout the play. These values are also features that we will use. Using the NGS data we also found the hang time of the punt and used that as a feature. 
# 
# We want to see which features are most indicative of a concussion and we can do this by looking at which features provide the most mutual information. In the following code sample, we generate said features.

# In[ ]:


from multiprocessing import Queue, Process
from threading import Thread
featureVectors = Queue()
fvs = []
puntplays = play_inf.query("Play_Type == 'Punt'")
def makeFeatureVector(lo,hi,qq):
    #print(lo,hi)
    localcount = 0
    for rr in range(lo,hi):
        s = time.time()
        currow = puntplays.iloc[rr]
        nam = getNGSName(currow["Season_Year"],currow["Week"],currow["Season_Type"])
        
        #if(localcount > 0):
            #break
        ngsTotal = NGSDICT[nam]
        #print(currow["GameKey"],currow["PlayID"])
        thisplay = ngsTotal.query('Season_Year == %d and GameKey == %d and PlayID == %d' %(currow["Season_Year"], currow['GameKey'],currow['PlayID']))
        if(len(thisplay) > 0):
            puntinfo = thisplay.query("Event == 'punt'")
            prinfo = thisplay.query("Event == 'punt_received' or Event == 'fair_catch'")
            puntinfo = pd.to_datetime(puntinfo['Time'],format='%Y-%m-%d %H:%M:%S.%f')
            prinfo = pd.to_datetime(prinfo['Time'],format='%Y-%m-%d %H:%M:%S.%f')
            ht = 0
            if(len(puntinfo) > 0 and len(prinfo) > 0):
                kicked = min(puntinfo)
                gotten = min(prinfo)
                ht = (gotten-kicked).total_seconds()
            gamedata = game_inf.query("Season_Year == %d and GameKey == %d" % (currow["Season_Year"],currow["GameKey"]))
            smallp = pprd.query('Season_Year == %d and GameKey == %d and PlayID == %d' %(currow["Season_Year"], currow['GameKey'],currow['PlayID']))
            gamedata = gamedata.iloc[0]
            featureVector = {}
            featureVector['HangTime'] = ht
            featureVector['Week'] = gamedata["Week"]
            featureVector['Season_Type'] = gamedata["Season_Type"]
            featureVector['Game_Day'] = gamedata["Game_Day"]
            featureVector['Game_Site'] = gamedata["Game_Site"]
            featureVector['HomeTeamCode'] = gamedata["HomeTeamCode"]
            featureVector['VisitTeamCode'] = gamedata["VisitTeamCode"]
            featureVector['Turf'] = gamedata["Turf"]
            featureVector['GameWeather'] = gamedata["GameWeather"]
            featureVector['Temperature'] = gamedata["Temperature"]
            featureVector['OutdoorWeather'] = gamedata["OutdoorWeather"]
            featureVector['Stadium'] = gamedata["Stadium"]
            featureVector['StadiumType'] = gamedata["StadiumType"]
            featureVector['Start_Time'] = gamedata["Start_Time"]
            featureVector['Game_Clock'] = currow["Game_Clock"]
            featureVector['YardLine'] = currow["YardLine"]
            featureVector['Quarter'] = currow["Quarter"]
            featureVector['GameKey'] = currow["GameKey"]
            featureVector['PlayID'] = currow["PlayID"]
            featureVector['Season_Year'] = currow["Season_Year"]
            featureVector['Play_Type'] = currow["Play_Type"]
            featureVector['Poss_Team'] = currow["Poss_Team"]
            featureVector['HomeScore'] = currow["Score_Home_Visiting"].split(' - ')[0]
            featureVector['AwayScore'] = currow["Score_Home_Visiting"].split(' - ')[1]
            
            for x in PUNTPOS:
                featureVector[x + '_' + 'AVG SPEED'] = "NA"
                featureVector[x + '_' + 'AVG DIR'] = "NA"
                featureVector[x + '_' + 'AVG O'] = "NA"
            
            for x in RETURNPOS:
                featureVector[x + '_' + 'AVG SPEED'] = "NA"
                featureVector[x + '_' + 'AVG DIR'] = "NA"
                featureVector[x + '_' + 'AVG O'] = "NA"
            found = 0
            for ttt in set(thisplay['GSISID']):
                try:
                    k = smallp.query('GSISID == %d' %(ttt))
                except:
                    continue
                if(len(k) > 0):
                    localcount += 1
                    legalplayermovement = thisplay[thisplay['GSISID']== ttt]
                    #print(legalplayermovement['Event'].unique())
                    rl = k.iloc[0]["Role"]
                    split = len(legalplayermovement)//3
                    featureVector[rl + '_' + 'AVG DIR'] = legalplayermovement['dir'].mean()
                    featureVector[rl + '_' + 'AVG O'] = legalplayermovement['o'].mean()
                    wind = 0
                    '''
                    for x in range(0,len(legalplayermovement),split):
                        wind+= 1
                        tempdf = legalplayermovement[x:] if x+split >= len(legalplayermovement) else legalplayermovement[x:x+split]
                        featureVector[rl + '_' + 'AVG DIR_' + str(wind) +'-WINDOW' ] = legalplayermovement['dir'].mean()
                        featureVector[rl + '_' + 'AVG O' + str(wind) +'-WINDOW'] = legalplayermovement['o'].mean()
                        speeds = legalplayermovement["dis"].apply(lambda x: x * 20.5)
                        featureVector[rl + '_' + 'AVG SPEED'+ str(wind) +'-WINDOW'] = speeds.mean()
                    '''
                    speeds = legalplayermovement["dis"].apply(lambda x: x * 20.5)

                    featureVector[rl + '_' + 'AVG SPEED'] = speeds.mean() 
                    found += 1

            qq.put(featureVector)
        #print(thisplay)
        e = time.time()
        #print(e-s)
    print("done with the process")
    return

def empty_queue(q):
    while True:
        k = q.get()
        fvs.append(k)

def makeFeatures(threads):
    ss = time.time()
    ts = []
    numthreads = threads
    workperthread = int(len(puntplays)/numthreads)
    for rr in range(0,len(puntplays),workperthread):
        #if rr >= 100: break
        if(rr + workperthread >= len(puntplays)):
            curt = Process(target = makeFeatureVector, args = (rr,len(puntplays),featureVectors,))
        else:
            curt = Process(target = makeFeatureVector, args = (rr,rr+workperthread,featureVectors,))
        ts.append(curt)
    for t in ts:
        t.start()
    cc = 0
    monit = Thread(target=empty_queue, args=(featureVectors,))
    monit.start()
    for t in ts:
        print("waiting for %d" % cc)
        cc+= 1
        t.join()
        
    
    ee = time.time()
    print("Time Taken")
    print(ee - ss)
    print("Made this many feature vectors:")
    print(len(fvs))
    return 


# In[ ]:


# Make the features in parallel using 10 processes. 
makeFeatures(10)
print(len(fvs))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
import copy
import matplotlib.pyplot as plt

vf = pd.read_csv('../input/video_footage-injury.csv')
concussionset = set()
for x in range(len(vr)):
    cr = vr.iloc[x]
    idenstring = str(cr['Season_Year'])+ '_' + str(cr['GameKey']) + '_' + str(cr['PlayID'])
    concussionset.add(idenstring)

fvscopy = copy.deepcopy(fvs)
fvscopy2 = copy.deepcopy(fvs)
confvs = []
regularfvs = []

concussionFEATS= []
regularFEATS = []
for x in fvscopy2:
    fviden = str(x['Season_Year'])+ '_' + str(x['GameKey']) + '_' + str(x['PlayID'])
    if(fviden in concussionset):
        concussionFEATS.append(x)
    else:
        regularFEATS.append(x)
        
for x in fvscopy:
    fviden = str(x['Season_Year'])+ '_' + str(x['GameKey']) + '_' + str(x['PlayID'])
    jon_check1, jon_check2 = x['GameKey'],x['PlayID']
    del x['GameKey']
    del x['Season_Year']
    del x['PlayID']
    del x['Play_Type']
    #print(x["HangTime"])
    if(fviden in concussionset and not (jon_check1 == 274 and jon_check2 == 3609)):
        confvs.append(x)
    else:
        regularfvs.append(x)

consdf = pd.DataFrame(confvs)
regsdf = pd.DataFrame(regularfvs)
concussionFEATSdf = pd.DataFrame(concussionFEATS)
regularFEATSdf = pd.DataFrame(regularFEATS)
def subsetMean(c,n):
    comps = {}
    for k in c.keys():
        comps[k] = {}
        #print(k)
        ccworked = False
        ncworked = False
        try:
            if('O' in k or 'DIR' in k):
                comps[k]['Concussion'] = c[k].replace('NA',0).fillna(0).mean() - 180
                #cc.append(consdf[k].replace('NA',0).fillna(0).mean() - 180)
                ccworked = True
            else:
                comps[k]['Concussion'] = c[k].replace('NA',0).fillna(0).mean()
                #cc.append(consdf[k].replace('NA',0).fillna(0).mean())
                ccworked = True
            #print(k, consdf[k].mean(),'c')
        except: pass
        try:
            if('O' in k or 'DIR' in k):
                comps[k]['No Concussion'] = n[k].replace('NA',0).fillna(0).mean() -180
                #nc.append(regsdf[k].replace('NA',0).fillna(0).mean())
                ncworked = True
            else:
                comps[k]['No Concussion'] = n[k].replace('NA',0).fillna(0).mean()
                #nc.append(regsdf[k].replace('NA',0).fillna(0).mean())
                ncworked = True
        except: pass
        #print(k, regsdf[k].mean(),'nc')
    return comps

comps = subsetMean(consdf,regsdf)
compsdf = pd.DataFrame(comps)
def plot_comparisons(word,word2):
    subsetDIR = []
    subsetO = []
    subsetSPEED = []

    subsetDIRR = []
    subsetOR = []
    subsetSPEEDR = []

    for x in PUNTPOS:
        if(x == "PC" or x == "PPR"): continue
        subsetDIR.append(x+'_AVG' +' DIR')
        subsetO.append(x+'_AVG' +' O')
        subsetSPEED.append(x+'_AVG' +' SPEED')

    for x in RETURNPOS:
        if(x == "PC" or x == "PPR"): continue
        subsetDIRR.append(x+'_AVG' +' DIR')
        subsetOR.append(x+'_AVG' +' O')
        subsetSPEEDR.append(x+'_AVG' +' SPEED')
    smoldfDir = compsdf[subsetDIR]
    smoldfO = compsdf[subsetO]
    smoldfSPEED = compsdf[subsetSPEED]

    smoldfDirR = compsdf[subsetDIRR]
    smoldfOR = compsdf[subsetOR]
    smoldfSPEEDR = compsdf[subsetSPEEDR]
    if(word == "punt"):
        if(word2 == 'directions'):
            plt.figure()
            smoldfDir.T.plot.bar(title='Punt Team Directions (degrees)',figsize=(12,8))
        if(word2 == 'o'):
            plt.figure()
            smoldfO.T.plot.bar( title='Punt Team Orientations (degrees)',figsize=(12,8))
        if(word2 == 'speed'):
            plt.figure()
            smoldfSPEED.T.plot.bar( title='Punt Team Speeds (mph)',figsize=(12,8))
    else:
        if(word2 == 'directions'):
            plt.figure()
            smoldfDirR.T.plot.bar(title='Return Team Directions (degrees)',figsize=(12,8))
        if(word2 == 'o'):
            plt.figure()
            smoldfOR.T.plot.bar( title='Return Team Orientations (degrees)',figsize=(12,8))
        if(word2 == 'speed'):
            plt.figure()
            smoldfSPEEDR.T.plot.bar( title='Return Team Speeds (mph)',figsize=(12,8))
print(len(fvs))


# **Pre-Feature Selection Analysis**
# 
# Before selecting the most important features, we want to visualize our features better. We plot a bar chart that compares the average speed of positions in plays with and without concussions. 
# 
# We can see from "Punt Team Speeds" and "Receiving Team Speeds", speeds for different players are similar between concussion and non-concussion plays. 

# In[ ]:


plot_comparisons("punt",'speed')
plot_comparisons("recv",'speed')


# In the field diagram below, the horizontal line in the middle represents a degree of 0. If a player is standing above the line their angle relative to the 0 increase as they move further from the line. Conversly, if a player is standing below the line, the angle decreases negatively as they move further below the line. The players on the punting team would start the play with an orientation of 0 (facing the right endzone), and the players of the receiving team would start with an orientation of 180 (facing the left endzone).

# <img src="https://i.imgur.com/bNOEjqF.png" />

# The following charts compare the average direction of player motion between plays with and without concussions. It is interesting to see the punting team players' direction of motion varies depending whether or not the play resulted in a concussion, whereas this variation does not seem to exist for receving team players. In this case when we review concussion play video later on, we will pay close attention to the players' direction of motion and how that contributes to concussions. 

# In[ ]:


plot_comparisons("punt",'directions')
plot_comparisons("recv",'directions')


# We can see the differences in the average directions of motions of the punting team between the plays involving concussions and normal plays. It is interesting to note that the GL and the GR in regular plays have motions reflected across the 0 line, toward the PR who is in the middle of the field. Furthermore, we see that the VRs and the VLs should have a steeper angle toward the punt returner to block the gunners from getting to him. All of the players on the return team generally move towards the punt returner so they can make blocks. The punt returner generally moves toward the line of scrimmage. 

# <img src="https://i.imgur.com/K0RLft4.png" />

# Now looking at the orientations of the players. This is significant because often times, players are not oriented in the direction they are moving. This could lead to dangerous plays such as blindside blocks.
# 
# We see that player orientations in "Return Team Orientations" are not significantly different between plays with and without concussions. However, there is a significant difference in the punt team players' average orientation. In a normal play, the average orientations of GL and GR should be reflections across the horizontal. Referring to "Punt Team Orientations", we see that this is indeed the case. However, in plays with concussions the GL and GR have the same degree of orientation. Furthermore, the average orientation of three other positions (PRW, PLT, PLS) are also inverted. Because of this inconsistency in orientation between plays with and without concussions, we suggest there is some relationship between the orientation of a player and concussions. 

# In[ ]:


plot_comparisons("punt",'o')
plot_comparisons("recv",'o')


# **Feature Selection**
# 
# Now we would like to see what features of a play lead to a concussion. We will be using Sci-Kit Learn's SelectKBest method to perform a feature selection based on mutual information score. We will also be using the SMOTE algorithm in order to interpolate more concussion examples. 
# 
# We use SMOTE in order to generate simulated concussion samples by randomly sampling instances of concussion plays and interpolating new examples from this random set. We randomly generate a training set consisting (on average) one-fourth concussions and three-fourths regular plays. SMOTE then interpolates the remaining concussion plays. After 10 iterations of mutual information based feature selection using Sci-Kit Learn's SelectKBest, we rank the most important features in determining a concussion.

# In[ ]:


def randomSet(concs, notconcus):
    import random
    testset = []
    results = []
    cs,ncs = 0,0
    nex = 0
    for x in range(500):
        rando = random.randint(0,500)
        if(rando % 4 == 1):
            ind = random.randint(0,len(concs)-1)
            testset.append(concs[nex % len(concs)-1])
            nex += 1
            results.append(1)
            cs += 1
        else:
            ind = random.randint(0,len(notconcus)-1)
            testset.append(notconcus[ind])
            results.append(0)
            ncs += 1
    
    print("set contains %d concussions and %d regular plays" % (cs,ncs))
    return testset,results

def doSelect():
    ts,rs = randomSet(confvs,regularfvs)
    setdf = pd.DataFrame(ts)
    setdf = setdf.fillna(0)
    for s in setdf:
        setdf[s]=setdf[s].astype('str')
    #print(setdf)
    sm = SMOTE(random_state=2)
    fixed = setdf.apply(LabelEncoder().fit_transform)
    X_train_res, y_train_res = sm.fit_sample(fixed, rs)
    print(len(X_train_res))
    feats = SelectKBest(mutual_info_classif,k=50).fit(X_train_res,y_train_res)
    new_features = [] # The list of your K best features
    feature_names = list(setdf.columns.values)
    mask = feats.get_support()#indices=True)
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    cols = feats.get_support(indices=True)
    mutualInfoVal = dict(sorted(zip(fixed.columns.values,feats.scores_),key=lambda x: x[1]))
    ordered = reversed(list(mutualInfoVal))
    #print(new_features)
    for m in ordered:
        try:
            selectResults[m] += mutualInfoVal[m]
        except:
            selectResults[m] = mutualInfoVal[m]
            


# In[ ]:


selectResults = {}
for x in range(10):
    doSelect()
print('done selections')

featuresCount = 0
#print(selectResults)
for sR in reversed(sorted(selectResults,key=selectResults.get)):
    if(featuresCount > 50):break
    print(sR, selectResults[sR]/10)
    featuresCount += 1


# **Most Important Features Found**
# 
# We find that average orientation of punting team players and average direction of punting team players are some of the most important features in determining a concussion. We see that the yard line is also important in determining concussions. This makes sense because if a team is backed up into their own side of the field, the punter tries to kick the ball further, leading to more hang time, and a higher possibility of a return. Furthermore, a punter running/having an orientation that isn't downfield is a solid indication there is a return on the play because the punter is now moving to make a tackle on the returner. 
# 
# To reiterate, the most important features in determing a concussion are the average directions and orientations of the punting team. From the previous chart "Concussions per Team", we see that there are significantly more injuries to the punting team, we want to make a rule that makes it safer for the players on that side of the ball. 
# 
# In order to see why the orientations and directions are so different for concussion plays and regular plays, we should first see how starting formations are different. We will define the formations based on the number of outside defenders (jammers). The following formation has 3 jammers.
# 
# <img src="https://i.imgur.com/8nnckKA.png"/>

# In[ ]:


count = 0
jammerIDS = {}
def countJammerFormations(df,s):
    jammercounts = {}
    for x in range(len(df)):
        forms = df.iloc[x].split(', ')
        j = 0
        for y in forms:
            if y.startswith("V"):
                j += 1
        if(j == 5):
            j = 4
            #print(df.iloc[x].split(', '),j)
        try: jammercounts[j] +=1
        except: jammercounts[j] = 1
    for n in sorted(jammercounts):
        print("The formation with %d jammers appears %d times %s" % (n,jammercounts[n],s))
    #count += 1


# In[ ]:


countJammerFormations(returnTeamForms,'overall')
print()
countJammerFormations(concussReturnTeamsForms,'in the concussion set')


# We can see that concussions happened on formations with 2 jammers, 3 jammers, and 4 jammers, which are also the most common formations overall. Now we want to see what kind players are getting concussions in each of these formation types. 

# In[ ]:


def getFormationInjury(df):
    formationsInjury = {}
    for x in range(len(df)):
        rr = df.iloc[x]
        #print(rr.keys())
        forms = rr["form"].split(', ')
        j=0
        for y in forms:
            if y.startswith("V"):
                j += 1

        primary = rr['GSISID']
        g,p = rr['game_play_key'].split("_")
        prole = pprd.query("GameKey == %d and PlayID == %d and GSISID == %d" %(int(g),int(p),int(primary))).iloc[0]['Role']

        try:formationsInjury[j].append(prole)
        except:
            formationsInjury[j] = []
            formationsInjury[j].append(prole)
        
        #print(prole)
    
    for x in sorted(formationsInjury):
        injuredOffense = 0
        print("In the formations with %d jammers, the following positions received concussions:" % x)
        for y in formationsInjury[x]:
            if(y in PUNTPOS):
                injuredOffense += 1
                
            
        print("\t ", formationsInjury[x])
        print('\t Percent of injuries to the punting team: %f' % (injuredOffense*100/len(formationsInjury[x]) ))
    

    


# In[ ]:


getFormationInjury(concuss[concuss['punting_team'] == False])


# The interesting observation to point out here is that in 4 jammer formations, mostly players on the offensive line (PRW, PRG, PRT, PLS, PLT, PLG, PLW) were injured. Only two defensive players were injured on this play. Lets look at some video footage to understand what is happening in these formations. 
# <video width="800" height="600" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4" type="video/mp4"></video>
# 
# In this play, we see there are two sets of two jammers on either side of the field. As the ball is snapped and the play goes on, both jammers block the gunners and they are effectively taken out of the play; neither of them really get close to the ball. This means the kicking team now has numerous unblocked players running down field at the time of the kick (all linemen, the punter didn't move). The returner is the legendery Devin Hester, and the Bengals (kicking team) are clearly respecting his return ability, but they do not force or fill properly, and Hester makes 3 people miss and turns up field. Now there are 3 Bengals behind Devin Hester, chasing him upfield (their orientations are upfield and direction of motion is mostly upfield). As Hester turns the edge, number 42 of the return team runs downfield (orientation and direction of motion are mostly opposite that of the trailing Bengals, and toward his own goal) blocks one Bengal, ramming him into the trailing player. This is a legal block according to the NFL, as the player was not hit in the head or neck area. The concussion happens at this point in time. 
# 
# We see that this formation took two fast, athletic players out of the play and resulted in a really good punt returner gaining substantial yards up field and the subsequent blocks toward their own goal caused a concussion. A similar kind of play happens in the next video. 
# <video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153238/Punt_Return_by_Damiere_Byrd-IX9zynRU-20181119_154215217_5000k.mp4" type="video/mp4"></video>
# 
# In this play, the return team again has 4 jammers. As the play happens, the gunners try to avoid getting jammed by taking a hard inside release. This means they try to run past the jammers by cutting sharply toward the middle of the field. This vacates space on the outsides of the field. The jammers end up blocking the gunners, who are now unable to get close to the returner by the time he catches the ball. The returner then runs into the space vacated by the gunners and then turns up field. At this point, the returner has managed to get into open space so the kicking team players (Steelers) turn back towards their own goal to try and chase the returner down. However, the defensive linemen that initially tried to block the punt are now running towards their own goal. One of them lays a block toward his own goal. The concussion happens here. 
# 
# We can see the implications of formation on the return. The punt returner is much more likely to make a play if there are no gunners running at him as he has much more space to work with. The following play has the same 4 jammer formation, but in this case, the jammers don't succeed on the left side. 
# <video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153258/61_yard_Punt_by_Brett_Kern-g8sqyGTz-20181119_162413664_5000k.mp4" type="video/mp4"></video>
# 
# In this play, the right gunner makes his way downfield, albeit with a hard outside release, so he is running down the sideline. The gunner runs past the returner and effectively takes himself out of the play, so the returner now turns across the field and makes his way up the field. The punting long snapper (PLS) number 49, had gone past the returners current yardline so he turns back up field and across the field to chase after the returner. Again his orientation is toward his own goal and the direction of motion is similar to that of the returners. At this point he is blocked by a defensive linemen who is running downfield to make the block. The contact happens in the legal strike zone, and hence is not illegal, but still gives the opposition player a concussion. The direction of motion and orientation of the impacted player is opposite that of the player who blocked him, but both are oriented toward their own goal lines. 
# 
# The above three plays show offensive line injuries, but interestingly, we see there are two defensive linemen injuries (in the 4 jammer formation, referring back to the previous code block injuries based on jammer formation). The following video is a play where the defensive end (number 33) gets injured. This is, again, a 4 jammer formation. 
# <video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153272/Haack_42_yard_punt-iP6aZSRU-20181119_165050694_5000k.mp4" type="video/mp4"></video>
# 
# In this play, the gunners are limited in their way down the field, and the returner makes quite a few people miss. As the kicking team players turn to chase the returner, number 33 comes running downfield and tries to make a block toward his own goal on the opposing number 78 (Dolphins). In trying to make this block, 33 himself gets a concussion. 
# 
# <video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4" type="video/mp4"></video>
# 
# Similar play, same formation type, the defensive lineman on the Chiefs (second from the left on the defensive line) attempts to block the punt. He is not succesful and then proceeds to turn around in order to make a block for the punt returner. Tyreek Hill (a dangerous punt returner) sees the area vacated by the jammed gunners and attempts to run up that sideline. The defensive line player then comes in (direction of motion and orientation are opposite that of Hill and the players behind Hill at this point) and makes helmet-to-helmet contact with Giants number 51. The defensive end gave himself a concussion with that attempted block toward his own goal. Even though this was an illegal block, this video reveals the mindset of players on the returning team. They are more incentivized to make more dangerous blocks which would suggest why most concussions belong to the kicking team. 
# 
# We can see a pattern beginning to emerge. Kicking team players attempting to make a tackle are in dangerous positions because players on the returning team are coming back towards them at high speeds to make blocks. Additionally, these blocks are dangerous for both parties as they happen at high speeds.
# 
# Let's see generally what happens in this kind of play, the following shows how the play develops in stages:
# 
# The play starts out as a normal play with 2 sets of 2 jammers (a).
# <img src="https://i.imgur.com/7s2B4jm.png">
# 
# 
# 

# As the play starts, the general motion of the players is shown (b).
# 
# <img src="https://i.imgur.com/cgSWDaX.png" />

# As some of the players on the defensive line drop toward their own endline to make blocks, others try to block the punt. The offensive linemen block the defensive linemen and then proceed to make their way downfield. The gunners are stopped by the jammers and the returner can make his way toward one of those lanes vacated by the blocked gunner (c).
# 
# <img src="https://i.imgur.com/gZQLX1E.png" />

# As the blocks are made, the punt returner finds a lane near one of the sidelines and runs toward it, his motion and orientation are opposite that of many of the tacklers and some of his own teammates (PDR1, PDR2, PDL3, PDL1). After making some of the players on the blue team miss, the PR runs into the gap, and the blue team players now turn back upfield. toward their own goal, in order to make the tackle (d).
# 
# <img src="https://i.imgur.com/hRgl8va.png"/>

# Now we can see that PRT, PRG, PLS, and even PLW are in danger of getting blocked toward their own goal. Their orientation is turning up field so that they can meet the path of PR, but PDL1, PDL3, PDR2, PDR1 can now make a block on any of these players while running at high speed toward their own goal (on the right of the image below) (e).
# 
# <img src="https://i.imgur.com/pgM2GM9.png"  />

# The area highlighted between the red lines in the image below is a danger zone for the kicking team players. This is because as they turn to chase the returner, they are likely to be blindsided by blockers on the returning team. This danger area needs to be the place where punting team members are protected so that they do not receive concussions at the rate that they do on these kinds of plays. 
# 
# <img src="https://i.imgur.com/04e6y1l.png"  />

# Now we see why the 4 jammer formation is dangerous for the punting team. The area in the middle of the field is generally occupied and the punt returner usually has a lane up the sideline for a return. Now the two jammer and the three jammer formation also have more punting team concussions than returning. From our feature selection we see that the orientation and direction of the players on the offensive team are still really important in determing if there is a concussion or not, hence we will again look at the problem from that perspective. The first question that we have to answer is, since there are potentially two extra non-jammers in the two jammer and three jammer formation, what positions do these extra men take? We answer this by looking at the number of defensive linemen and linebackers for the 2 jammer and 3 jammer formations. 

# In[ ]:


count = 0
jammerIDSBackers = {}
def countBackerFormations(df,s):
    jammercounts = {}
    for x in range(len(df)):
        forms = df.iloc[x].split(', ')
        j = 0
        backer = 0
        dl = 0
        for y in forms:
            if(y == "PR" or y == "PFB"): continue
            if y.startswith("V"):
                j += 1
                continue
            if("DL" in y or "DR" in y):
                dl += 1
            else:
                backer += 1
        if(j == 5):
            j = 4
            #print(df.iloc[x].split(', '),j)
        if(j == 4): continue
        
        try: jammercounts[(backer,dl)] +=1
        except: jammercounts[(backer,dl)] = 1
    #for n in sorted(jammercounts):
        #print("The formation with %d linebackers and %d linemen appears %d times %s" % (n[0],n[1],jammercounts[n],s))
    #print()
    return jammercounts
            


# In[ ]:


a = countBackerFormations(returnTeamForms,'overall')
b = countBackerFormations(concussReturnTeamsForms,'in the concussion set')
for x in b:
     print("The formation with %d linebackers and %d linemen has concussion percentage %f" % (x[0],x[1],(b[x]*100/a[x])))


# Interestingly, we see that the formation with 1 linebacker and 5 linemen has a concussion percentage of 5% as opposed to the less than 1 percent for the other formations that had concussion. Since we know there are at most 3 jammers and a punt returner, the remaining position MUST be a PFB, or a punt full back. This formation was only used 60 times throughout the year, but there were 3 concussions on this formation. In order to understand why, we can look at 2 of the plays a fullback was used. 
# '<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153233/Kadeem_Carey_punt_return-Vwgfn5k9-20181119_152809972_5000k.mp4" type="video/mp4"></video>'
# We see in this play that there is a left jammer and two right jammers. The full back drops facing the away from the punt returner as the ball is kicked in the air. Then, when the ball is caught, the full back changes his direction of motion toward the closest covering player, in this case is the gunner on the side with only one jammer. He then blocks the player, and the returner is able to run toward the side with two jammers for a decent gain. We see that in this play, the full back acted as a kind of free-roam blocker, but the only player downfield fast enough for him to block is the gunner. By drawing the gunner in, and having a personal blocker for the punt returner, this gave the returner an interesting angle in his return and allowed him to run upfield. He is eventually hit by a helmet-to-helmet collision (illegal) which caused a concussion. 
# '<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153236/Punt_by_Brad_Wing-SMRxqgb2-20181119_153645589_5000k.mp4" type="video/mp4"></video>'\
# We see on this play that the full back acts as a linebacker and then drops straight back, oriented away from the punt returner and toward the oncoming tacklers. As he drops, he does not make a block until the punt returner catches the ball, after which the PFB attempts to make a block toward his own goal. Blocking towards ones own goal again causes a concussion for a player on the punting team. 
# 
# We see that this formation causes problems because players are drawn into the danger zone where they can potentially be blindsided by a block towards the blocker's own goal. 
# 
# Furthermore, if we look at the role of the linebackers in a punting formation we can understand where there is an orientation and direction issue. In a 3 jammer formation, the linebackers have freedom to make blocks as they wish. 
# 
# <img src="https://i.imgur.com/LMOtcis.png" />

# Referring to the image above, the PLR can drop to block the PLW or PPL (if there is one) and the PLL can drop to either block the gunner (GR) toward his own goal or block the PRW. The linebackers have the freedom to block the people with the greatest threat of getting downfield. Sometimes the linebackers can even rush the punter to try and get a blocked punt, but usually they drop and try to set up blocks for the punt returners.
# 
# The more linebackers there are, the more blocking the PR has set up for him to follow, but fewer players rush the punter. Usually this is an attempt to set up a return down the middle as the players are being forced outside in the example above. We can see that the collisions that happen during the blocking are intense and happen at high speeds and head on. If the PLW beats the PLR outside, and attempts to come back upfield, he is in the danger zone. Essentially, players turning up field to chase the returner are in the danger zone. 
# 
# The starting formations of the play imply a lot about the behavior of the return team players. We can use this information in order to define a rule change in accordance to the features we found through the video review and feature selection. 

# **Proposed Rule Change**
# 
# We are proposing 2 rule changes.
# 
# *Any player blocking toward their own end zone is assessed an unnecessary roughness penalty of 15 yards.*
# 
# We believe the NFL should make blocking towards your own end zone illegal during kicking plays in order to decrease the number of concussions. The main reasons are as follows:
# 
# 1. There are 16 plays with players blocking toward their own goal resulting in concussions. (16/37) concussions could be prevented with this rule change. 
# 2. Blocking toward your own goal is more likely to be a head-on or blindsided collisions. This maximizes the force of impact thus increasing the chances of a concussion and eliminates the danger area. Through our data analysis we have found that orientation and direction are two important features associated with concussions. Our proposed rule tackles (pun-intended) these two features directly. The rule will decrease the number of concussion causing punt plays.
# 3. This rule change is simple to integrate and easy to learn. Referees need to be aware of any players blocking towards their own goal.
# 4. This rule preserves the integrity of the sport. We believe that it would not decentivize punt returns, which we believe is an integral part of the sport. We will address how our proposed changes could introduce new risks to player safety in the next paragraph. Punting is a major part of football and many teams can gain a huge advantage through their punt return teams and punt coverage teams. Changing the rules too drastically would remove the importance this play type. 
# 
# Since players are not allowed to block toward their own end zone, they may be more inclined to play more off coverage, giving the PR a chance to make a return. Longer returns could lead to greater possilbilities of concussions. As the defensive linemen would not be able to come back down the field and make a block, teams may chose to use the defensive linemen more aggresively in rushing the punter, potentially leading to dangers for the offensive line and the punter. 
# 
# *Any player intiating helmet-to-helmet contact during a punt play is assessed an unnecessary roughness penalty of 15 yards and disqualified from the contest.*
# 
# Additionally, we think that intentional helmet-to-helmet contact should result in immediate ejection. The main reasons are as follows:
# 
# 1. There are 16 plays with helmet-to-helmet contact resulting in concussions. (16/37) concussions could be prevented with this rule change. 
# 2. In a punt play, since players are moving downfield at high speeds in many directions and orientations, a helmet-to-helmet contact would be especially dangerous. While there is a rule in place that makes helmet-to-helmet contact illegal, we see that it still occurs. By making the punishment for this dangerous play even harsher, we strongly incentivize players to not commit this infraction. 
# 3. This rule change is simple to integrate. Referees need to be aware of any players initiating head-to-head contact.
# 4. This rule change preserves the integrity of the sport. The NFL has added rules in place attempting to make this contact illegal for many years, the harsher punishment would make sure this happens. 
# 
# It is possible that players who have learned over the years to hit in one way may have to change their tackling technique in order to make sure they do not get disqualified by this rule. This can potentially lead to players injuring other parts of their body. This could be dangerous for both parties involved in the tackles. 

# **Rebuttal to Other Rule Changes**
# 
# In addition to our proposed rule changes we also considered other rules:
# 
# *Adding restrictions on starting formations*
# 
# We decided that adding restrictions on the starting formations would take away from the integrity of the sport. Punting is an offensive play, which means fake punt passes and fake punt throws are valid plays; we did not want to restrict the creativity of the teams on these plays. Furthermore, from our data analysis we see that formations are not the major feature causing concussions, rather it is the actions taken by the players in those formations. Making those actions illegal is safer than trying to restrict the formation.
# 
# *Incentivising fair catches*
# 
# We decided that incentivizing fair catches strongly impacts the integrity of the game. Adding some number of yards to the end of the fair catch would give the receiving team an advantage. It would allow the returning team to rush the punter without worrying about blocking for the returner. The kicking team is at a disadvantage regardless of whether the returner fair catches.
# 
# *Moving the Touchback from the 20 to the 15*
# 
# This would incentivize the kicking team to kick through the end zone on every punt, because they would guarantee bad field position for the other team. This changes the integrity of the game by significantly decreasing the chances of a return. We think that returns are one of the most essential and exciting aspects of football. 

# In[ ]:




