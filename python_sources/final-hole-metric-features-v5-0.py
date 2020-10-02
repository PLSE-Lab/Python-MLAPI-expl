#!/usr/bin/env python
# coding: utf-8

# This is the final version of a series of notebooks that I had once posted but have now made private. 
# These features exist to provide context to what is happening to the area around the offensive line, which is obviously important.
# 
# 
# There are four holes between the five offensive, I also added two holes, one for each side. Each play has been standardized such that
# each offensive team faces vertically up and as each offensive team gets closer to the endzone in which they are trying to score,
# the yardline decreases. So, say if the Bengals have possession on the one yard line on the farther side of the field, the yardline is '99'.
# If they have possession on the one yard line on the side of the field in which they are trying to score, the yardline is '1'.
# 
# The features include:
# * olnum: number of defenders in each hole, which extends vertically from the y value of the ball carrier to the yardline + 5 yards.
# * olnumeng: number of defenders engaged (meaning that the ol has hands on them, in each hole), which is defined by if an ol is within 2 yards.
# * numdefbf: number of defenders in the backfield (for each hole), which is defined if a defender is positioned between two ol but has crossed the linear line that connects the two ol.
# * bchole: this returns what hole the bc is currently line up behind noted as a 1 or 0 for each hole.
# * boxsafety: returns the number of defensive players that are lined up between the ol but are ten yards off the line of scrimmage.
# * leftright: returns 1 for first value if ballcarrier is rushing left or 1 for second value if the ball carrier is rushing right.
# * holesize: the euclidean distance between each offensive lineman.
# * holeslope: the slope of each hole.
# * push: using the yardline as y = 0, this returns the relative distance of the defender in each hole that has moved furthest into the backfield.
# * bcdir: the ball carrier's angle.
# 
# If you use these features, please cite or if you think there are others that I should provide, ask. Thanks.

# These are the import statements and options used in this notebook.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics as stats
import math
import os

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 100)


# cleanabv() fixes some of the issues found in the training set. I fixed a player error where Bradley Sowell was listed as a TE. I moved him to tackle even though he played both in the NFL.

# In[ ]:


def cleanabv(train):
    #   Clean Abbreviations
    train['ToLeft'] = train.PlayDirection == "left"
    train['IsBallCarrier'] = train.NflId == train.NflIdRusher
    train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"
    train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"
    train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"
    train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    train['Dir_std'] = np.mod(90 - train.Dir, 360)
    train.loc[train.DisplayName == "Bradley Sowell", "Position"] = "T"
    return train


# show_play() shows the players on the field and show_box() shows the players on the field with the box placed upon each hole. The ballcarrier has a black dot.

# In[ ]:


def show_play(play_id, train):
    df = train[train.PlayId == play_id]
    fig, ax = create_football_field()
    ax.scatter(df.Y, df.X, cmap='rainbow', c=~(df.Team == 'home'), s=100)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.Y, rusher_row.X, color='black')
    plt.title('Play # %i' %play_id, fontsize=20)
    plt.legend()
    plt.show()


# create_football_field() creates the football field for visual purposes. I took this from CPMP's notebook.

# In[ ]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(6.33*2, 12*2)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """

    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    # YardLines that span across field, correct

    plt.plot([0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white')

    if fifty_is_los:
        plt.plot([0, 53.3], [60, 60],  color='gold')
        plt.text(50, 62, '<- Player Yardline at Snap', color='gold')

    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 53.3, 10,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((0, 110), 53.3, 10,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.ylim(0, 120)
    plt.xlim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(5, x, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(53.3 - 5, x, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([0.4, 0.7], [x, x], color='white')
        ax.plot([53.0, 52.5], [x, x],  color='white')
        ax.plot([22.91, 23.57], [x, x],  color='white')
        ax.plot([29.73, 30.39], [x, x],  color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax


# Loading train.csv and cleaning it. We return a list of uniqueplays and uniqueteams which will be further used for indexing purposes.

# In[ ]:


train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
train = cleanabv(train)
uniqueplays = np.unique(train.loc[:,"PlayId"])
uniqueteams = np.unique(train.loc[:,"PossessionTeam"])

#train['Dir_rad'] = train.Dir * math.pi/180.0


# In[ ]:


def getoline(play):
    oline = play.loc[(play.loc[:,"Position"]=="T") | (play.loc[:,"Position"]=="G") | (play.loc[:,"Position"]=="C") | (play.loc[:,"Position"]=="OT") | (play.loc[:,"Position"]=="OG"),:]
    #,:]

    oline = oline.sort_values(by = 'Y')
    return oline


# In[ ]:


def getolinesix(play):
    oline = play.loc[(play.loc[:,"Position"]=="T") | (play.loc[:,"Position"]=="G") | (play.loc[:,"Position"]=="C") | (play.loc[:,"Position"]=="OT") | (play.loc[:,"Position"]=="OG"),:]
    oline = oline.sort_values(by = 'Y')
    oline = oline.reset_index()
    firstoline = oline.loc[0,"level_0"]
    lastoline = oline.loc[4,"level_0"]
    onetwo = 0
    if firstoline>10:
        offense = play[11:22]
        onetwo = 2
    else:
        offense = play[0:11]
        onetwo = 1
    offense = offense.sort_values(by = 'Y')
    offense = offense.reset_index()
    ox1 = offense.loc[offense.loc[:,"level_0"]==firstoline,:]
    ox1 = ox1.index
    ox5 = offense.loc[offense.loc[:,"level_0"]==lastoline,:]
    ox5 = ox5.index
    
    if len(ox1) == 0:
        ox1 = pd.Series([1])
    
    if len(ox5) == 0:
        ox5 = pd.Series([10])
    
    lh = 0
    li = -1
    rh = 0
    ri = -1
    
    for i in range(ox1[0]-1,-1,-1):
        tempx = offense.loc[i,"Y"]
        tempy = offense.loc[i,"X"]
        oxx = offense.loc[ox1,"Y"]
        oxy = offense.loc[ox1,"X"]
        tempd = distance(tempx,oxx,tempy,oxy)
        if tempd <= 4:
            if offense.loc[i,"Position"]=="TE":
                lh = 1
                #li = offense.loc[i,"index"]
                li = i
                break
        else:
            lh = 0
            li = -1
    
    
    for i in range(ox5[0]+1,11,1):
        tempx = offense.loc[i,"Y"]
        tempy = offense.loc[i,"X"]
        oxx = offense.loc[ox5,"Y"]
        oxy = offense.loc[ox5,"X"]
        tempd = distance(tempx,oxx,tempy,oxy)
        if tempd <= 4:
            if offense.loc[i,"Position"]=="TE":
                rh = 1
                #print(offense)
                #ri = offense.loc[i,"index"]
                #print(i)
                ri = i
                break
        else:
            rh = 0
            ri = -1
        
    
    play.reset_index()
    play.sort_values(by = 'Y')

    return oline,ri,rh,li,lh,offense


# In[ ]:


def encodepos(play,uniqueteams):
    possession = np.zeros(22)
    for i in range(0,22):
        for j in range(0,32):
            if play.loc[i,"PossessionTeam"]==uniqueteams[j]:
                possession[i]=j
    
    return possession
        


# In[ ]:


def encodefpos(play,uniqueteams):
    fposition = np.zeros(22)
    for i in range(0,22):
        for j in range(0,32):
            if play.loc[i,"FieldPosition"]==uniqueteams[j]:
                fposition[i]=j
    
    return fposition


# In[ ]:


def changedirection(play,pos,fpos):
    #ha = home/away
    ha = play.loc[play.loc[:,"NflId"]==play.loc[0,"NflIdRusher"],"Team"]
    bc = np.where(play.loc[:,"NflId"]==play.loc[0,"NflIdRusher"])[0]
    bcd = play.loc[bc, "Dir"]
    bcy = play.loc[bc,"X"]
    bcy = bcy.reset_index()
    bcd = bcd.reset_index()
    if bc > 10:
        meandy = stats.mean(play.loc[11:,'X'])
    else:
        meandy = stats.mean(play.loc[0:10,'X'])
    
    postemp = pos[0]
    fpostemp = fpos[0]
    
    if bcy.loc[0,"X"]>meandy:
        play.loc[:,"X"] = 100 - play.loc[:,"X"] + 20
        play.loc[bc, "Dir"] = 360 - bcd.loc[0, "Dir"]
    else:
        play.loc[:,"X"] = play.loc[:,"X"]
        play.loc[:,"Y"] = 53.3 - play.loc[:,"Y"]
        play.loc[bc, "Dir"] = 180 - bcd.loc[0, "Dir"]
    
    if postemp != fpostemp:
        play.loc[:,"YardLine"] = 100 - play.loc[0,"YardLine"]
    
   
    ydline =  play.loc[0,"YardLine"]
    leftright = np.zeros(2)
    bcdir = play.loc[bc,"Dir"].values[0]
    if (((bcdir < 180) and (bcdir >= 90)) or ((bcdir >= 270) & (bcdir < 360))):
        leftright[0] = 1
    else:
        leftright[1] = 1
    
    return play,ha,ydline,bc,leftright,bcdir
    


# In[ ]:


def distance(x1,x2,y1,y2):
    dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist


# In[ ]:


def linear(x1,x2,y1,y2):
    slope = (y1-y2)/(x1-x2)
    temp = slope*x1
    b = y1-temp
    return slope,b
    


# In[ ]:


def hmengagedextra(play,ha,ydline,bc):
    oline, ri, rh, li, lh, offense = getolinesix(play)
    # print(play)
    play = play.reset_index()

    if rh == 1:
        rol = offense.loc[[ri]]
        oline = pd.concat([oline, rol])

    else:
        temp = oline.loc[[4]]
        temp.loc[4, "Y"] = oline.loc[4, "Y"] + 2
        oline = pd.concat([oline, temp])

    if lh == 1:
        lol = offense.loc[[li]]
        oline = pd.concat([lol, oline])

    else:
        temp = oline.loc[[0]]
        temp.loc[0, "Y"] = temp.loc[0, "Y"] - 2
        oline = pd.concat([temp, oline])

    olx = oline.loc[:, "Y"]
    olx = olx.reset_index()
    oly = oline.loc[:, "X"]
    oly = oly.reset_index()

    bcy = play.loc[bc, "X"]
    bcx = play.loc[bc, "Y"]
    bcy = bcy.reset_index()
    bcx = bcx.reset_index()
    if ha.all() == "away":
        defense = play.loc[11:21, :]
    else:
        defense = play.loc[0:10, :]

    defense = defense.drop(columns="level_0")
    defense = defense.reset_index()

    olnum = np.zeros(6)
    olnumeng = np.zeros(6)
    numdefbf = np.zeros(6)
    holesize = np.zeros(6)
    holeslope = np.zeros(6)
    bchole = np.zeros(8)
    push = np.zeros(6)
    boxsafety = np.zeros(1)

    olxt = np.zeros((2, 6))

    temp = olx.loc[0:5, "Y"]
    temp = temp.to_numpy()
    olxt[0, 0:6] = temp
    temp = olx.loc[1:6, "Y"]
    temp = temp.to_numpy()
    olxt[1, 0:6] = temp

    olyt = np.zeros((2, 6))

    temp = oly.loc[0:5, "X"]
    temp = temp.to_numpy()
    olyt[0, 0:6] = temp
    temp = oly.loc[1:6, "X"]
    temp = temp.to_numpy()
    olyt[1, 0:6] = temp

    dx = defense.loc[:, "Y"]
    dx = dx.to_numpy()
    dy = defense.loc[:, "X"]
    dy = dy.to_numpy()
    iby = np.where((dy < ydline + 15) & (dy > bcy.loc[0, "X"]), 1, 0)
    ibxt = np.where((dx > olxt[0, 0]) & (dx <= olxt[1, 5]), 1, 0)
    safeties = np.where((dy < ydline + 35) & (dy > ydline + 15), 1, 0)
    safetemp = np.sum([ibxt, safeties], axis=0)
    safetemp = np.where(safetemp == 2, 1, 0)
    boxsafety[0] = boxsafety[0] + np.sum(safetemp)

    for i in range(0, 6):
        if i == 0:
            if bcx.loc[0, "Y"] < olxt[0, i]:
                bchole[0] = bchole[0]+1
        if i == 5:
            if bcx.loc[0, "Y"] > olxt[1, i]:
                bchole[7] = bchole[7]+1

        if (bcx.loc[0, "Y"] > olxt[0, i]) & (bcx.loc[0, "Y"] <= olxt[1, i]):
            bchole[i+1] = bchole[i+1]+1

        ibx = np.where((dx > olxt[0, i]) & (dx <= olxt[1, i]), 1, 0)
            
        dist = np.vectorize(distance)
        dist1 = dist(olxt[0, i], dx, olyt[0, i], dy)
        dist2 = dist(olxt[1, i], dx, olyt[1, i], dy)

        slope, b = linear(olxt[0, i], olxt[1, i], olyt[0, i], olyt[1, i])
        tempy = np.multiply(dx, slope)
        tempy += b
        bfline = np.where((dy <= tempy), 1, 0)
        engcomp = np.where((dist1 <= 1.5) | (dist2 <= 1.5), 1, 0)
        
        # Push
        if sum(ibx)==0:
            minp = ydline+10
        else:
            minp = min(dy[ibx==1])
        push[i] = minp - (ydline+10)
        
        # Olnum
        olnum[i] = olnum[i] + np.sum(np.where(np.sum([ibx, iby], axis=0) == 2, 1, 0))

        # olnumeng
        olnumeng[i] = olnumeng[i] + np.sum(np.where(np.sum([ibx, iby, engcomp], axis=0) == 3, 1, 0))

        # Backfield
        numdefbf[i] = numdefbf[i] + np.sum(np.where(np.sum([bfline, ibx, iby], axis=0) == 3, 1, 0))
        
        # Holesize
        holesize[i] = distance(olxt[0,i],olxt[1,i],olyt[0,i],olyt[1,i])
        
        # Holeslope
        slope,b = linear(olxt[0,i],olxt[1,i],olyt[0,i],olyt[1,i])
        holeslope[i] = slope
    return olnum, olnumeng, numdefbf, boxsafety, bchole, oline, holesize, holeslope, push


# In[ ]:


def get_dx_dy(radian_angle, dist):
    dx = dist * math.cos(radian_angle)
    dy = dist * math.sin(radian_angle)
    return dx, dy


# In[ ]:


def show_box(play_id, train, oline, ydline):
    df = train[train.PlayId == play_id]
    fig, ax = create_football_field()
    ax.scatter(df.Y, df.X, cmap='rainbow', c=~(df.Team == 'home'), s=100)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.Y, rusher_row.X, color='black')
    rush = np.full((7,1),rusher_row.X)
    ydl = np.full((7,1),ydline+15)
    ol1 = np.full((7,1),oline.Y[0])
    for i in range(0,7):
        ol = np.full((2,1),oline.Y[i])
        temp = np.full((2,1),rusher_row.X)
        temp[0,0] = ydline+15
        ax.plot(ol,temp,color='black')
        if i != 6:
            ax.plot([oline.Y[i],oline.Y[i+1]],[oline.X[i],oline.X[i+1]],color='black')
    ax.plot([oline.Y[0],oline.Y[0]],[ydline+15,ydline+35],color='black')
    ax.plot([oline.Y[6],oline.Y[6]],[ydline+15,ydline+35],color='black')
    ax.plot([oline.Y[0],oline.Y[6]],[ydline+35,ydline+35],color='black')
    
    x = rusher_row["X"].values[0]
    y = rusher_row["Y"].values[0]
    
    rusher_dir = rusher_row["Dir_rad"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)
    ax.plot([y,dy+y],[x,dx+x],color='black')
    #ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')
    
    ax.plot(oline.Y,rush,color='black')
    ax.plot(oline.Y,ydl,color='black')
    plt.title('Play # %i' %play_id, fontsize=20)
    plt.legend()
    plt.show()


# In[ ]:


play = train.loc[train.loc[:,"PlayId"]==uniqueplays[500],:]
play = play.reset_index()
playid = play.PlayId[0]
pos = encodepos(play,uniqueteams)
fpos = encodefpos(play,uniqueteams)
play,ha,ydline,bc,leftright,bcdir = changedirection(play,pos,fpos)
play['Dir_rad'] = np.mod(90 - play.Dir, 360) * math.pi/180.0
olnum,olnumeng,numdefbf,boxsafety,bchole,oline,holesize,holeslope,push = hmengagedextra(play,ha,ydline,bc)
#oline = oline.reset_index()
print("--------------------")
print(olnum)
print(olnumeng)
print(numdefbf)
print(bchole)
print(boxsafety)
print(leftright)
print(holesize)
print(holeslope)
print(push)
print(bcdir)
oline = oline.drop(columns="level_0")
oline = oline.reset_index()
show_box(playid,play,oline,ydline)
#play.head()


# In[ ]:


playsize = uniqueplays.shape[0]

olnums = np.zeros((playsize,6))
olnumseng = np.zeros((playsize,6))
numdefsbf = np.zeros((playsize,6))
boxsafeties = np.zeros((playsize,1))
bcshole = np.zeros((playsize,8))
leftsrights = np.zeros((playsize,2))
bcdirs = np.zeros((playsize,1))
holes_size = np.zeros((playsize,6))
holes_slope = np.zeros((playsize,6))
pushes = np.zeros((playsize,6))
ydlines = np.zeros((playsize,1))


for i in range(0,uniqueplays.shape[0]):
    play = train.loc[train.loc[:,"PlayId"]==uniqueplays[i],:]
    play = play.reset_index()
    playid = play.PlayId[0]
    pos = encodepos(play,uniqueteams)
    fpos = encodefpos(play,uniqueteams)
    play,ha,ydline,bc,leftright,bcdir = changedirection(play,pos,fpos)
    olnum,olnumeng,numdefbf,boxsafety,bchole,oline,holesize,holeslope,push = hmengagedextra(play,ha,ydline,bc)
    
    olnums[i,:] = olnum
    olnumseng[i,:] = olnumeng
    numdefsbf[i,:] = numdefbf
    boxsafeties[i,:] = boxsafety
    bcshole[i,:] = bchole
    leftsrights[i,:] = leftright
    bcdirs[i,:] = bcdir
    holes_size[i,:] = holesize
    holes_slope[i,:] = holeslope
    pushes[i,:] = push
    ydlines[i,:] = ydline+10

train_holefeatures = np.concatenate((olnums,olnumseng,numdefsbf,boxsafeties,bcshole,leftsrights,bcdirs,holes_size,holes_slope,pushes,ydlines),axis=1)
train_holefeatures = pd.DataFrame(train_holefeatures)
train_holefeatures.to_csv('train_holefeatures.csv')

