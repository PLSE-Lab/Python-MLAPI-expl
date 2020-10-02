#!/usr/bin/env python
# coding: utf-8

# 
# # Predicting the winner of UFC fights 
# <span style="color:red">Rough copy, I am currently working on this offline and will upload when I have finnished it.</span>
# 
# The goal of this project is to create a machine learing model that predicts the winner of ufc fights. UFC fights are incredibly hard to predict, fighters only tend to fight eachother if they are seen ass being about as good as each other, fighters rarely if ever fight someone they are confident they can beat unlike in boxing. This makes predicting the out come of UFC fights very difficult to predict. The UFC places the favoured fighter or the champion of the weight division in the red corner, they are right about the victor only 57% of time. Below is a description of the data that came with the data, which cam be found at https://www.kaggle.com/rajeevw/ufcdata. Unfortunately a lot of the data has been recorded/scraped incorrectly, the first one thousand fights all have the red corner winning. Most people who have used this data set on kaggle did not notice this as result there models could achieve an accuracy of 67% by just guessing Red wins every time.
# 
# Below is some information that cam with the data.
# 
# ### Context
# This is a list of every UFC fight in the history of the organisation. Every row contains information about both fighters, fight details and the winner. The data was scraped from ufcstats website. After fightmetric ceased to exist, this came into picture. I saw that there was a lot of information on the website about every fight and every event and there were no existing ways of capturing all this. I used beautifulsoup to scrape the data and pandas to process it. It was a long and arduous process, please forgive any mistakes. I have provided the raw files in case anybody wants to process it differently. This is my first time creating a dataset, any suggestions and corrections are welcome! In case anyone wants to check out the work, I have all uploaded all the code files, including the scraping module here
# 
# Have fun!
# 
# ### Content
# Each row is a compilation of both fighter stats. Fighters are represented by 'red' and 'blue' (for red and blue corner). So for instance, red fighter has the complied average stats of all the fights except the current one. The stats include damage done by the red fighter on the opponent and the damage done by the opponent on the fighter (represented by 'opp' in the columns) in all the fights this particular red fighter has had, except this one as it has not occured yet (in the data). Same information exists for blue fighter. The target variable is 'Winner' which is the only column that tells you what happened. Here are some column definitions:
# 
# ### Column definitions:
# 
# |Column name  |Description |
# |-----|-------------|
# |R_ and B_ |prefix signifies red and blue corner fighter stats respectively|
# |_opp_ | containing columns is the average of damage done by the opponent on the fighter|
# |KD |is number of knockdowns|
# |SIG_STR |is no. of significant strikes 'landed of attempted'|
# |SIG_STR_pct |is significant strikes percentage|
# |TOTAL_STR |is total strikes 'landed of attempted'|
# |TD |is no. of takedowns|
# |TD_pct |is takedown percentages|
# |SUB_ATT |is no. of submission attempts|
# |PASS |is no. times the guard was passed?|
# |REV |*Probably reversels*|
# |HEAD |is no. of significant strinks to the head 'landed of attempted'|
# |BODY |is no. of significant strikes to the body 'landed of attempted'|
# |CLINCH |is no. of significant strikes in the clinch 'landed of attempted'|
# |GROUND |is no. of significant strikes on the ground 'landed of attempted'|
# |win_by |is method of win|
# |last_round |is last round of the fight (ex. if it was a KO in 1st, then this will be 1)|
# |last_round_time |is when the fight ended in the last round|
# |Format |is the format of the fight (3 rounds, 5 rounds etc.)|
# |Referee |is the name of the Ref|
# |date |is the date of the fight|
# |location |is the location in which the event took place|
# |Fight_type |is which weight class and whether it's a title bout or not|
# |Winner |is the winner of the fight|
# |Stance |is the stance of the fighter (orthodox, southpaw, etc.)|
# |Height_cms |is the height in centimeter|
# |Reach_cms |is the reach of the fighter (arm span) in centimeter|
# |Weight_lbs |is the weight of the fighter in pounds (lbs)|
# |age |is the age of the fighter|
# |title_bout |Boolean value of whether it is title fight or not|
# |weight_class |is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)|
# |no_of_rounds |is the number of rounds the fight was scheduled for|
# |current_lose_streak |is the count of current concurrent losses of the fighter|
# |current_win_streak |is the count of current concurrent wins of the fighter|
# |draw |is the number of draws in the fighter's ufc career|
# |wins |is the number of wins in the fighter's ufc career|
# |losses |is the number of losses in the fighter's ufc career|
# |total_rounds_fought |is the average of total rounds fought by the fighter|
# |total_time_fought(seconds) |is the count of total time spent fighting in seconds|
# |total_title_bouts |is the total number of title bouts taken part in by the fighter|
# |win_by_Decision_Majority |is the number of wins by majority judges decision in the fighter's ufc career|
# |win_by_Decision_Split |is the number of wins by split judges decision in the fighter's ufc career|
# |win_by_Decision_Unanimous |is the number of wins by unanimous judges decision in the fighter's ufc career|
# |win_by_KO/TKO |is the number of wins by knockout in the fighter's ufc career|
# |win_by_Submission |is the number of wins by submission in the fighter's ufc career|
# |win_by_TKO_Doctor_Stoppage |is the number of wins by doctor stoppage in the fighter's ufc career|
# |avg         | average over number of rounds in the fight |
# 
# 

# In[ ]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import numpy as np
import os 
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns 
from matplotlib import colors
plt.figure(figsize=(10,10))

from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(os.listdir("/kaggle/"))
data=pd.read_csv("../input/ufcdata/data.csv")
data.head()


# The first thing I will show is that for around the first thousand or so fights (chronologically) all the victors have been recorded scraped as fighting in the Red corner. Other kaggle users who have used this data set failed to notice this. 

# In[ ]:


for index,fight in data[::-1].iterrows():
    if fight.Winner == "Blue":
        break
print("The first",len(data)-index,"fights have been recorded as red wins")        
DataToDisplay=data[::-1]
display(DataToDisplay.head(10))
print("........................................................................")
display(DataToDisplay[int(len(data)-index-10):int(len(data)-index)])


# In[ ]:


data.describe()


# In[ ]:


PrepropData=pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")
PrepropData.head()


# In[ ]:


RawFighterDetails=pd.read_csv("/kaggle/input/ufcdata/raw_fighter_details.csv")
RawFighterDetails.head()


# In[ ]:


print("The number of non nan Reach entries is",len(list(RawFighterDetails[RawFighterDetails["Reach"]!= float("nan")])))
print("The number of non nan Stance entries is",len(list(RawFighterDetails[RawFighterDetails["Stance"]!= float("nan")])))
print("The number of non nan DOB entries is",len(list(RawFighterDetails[RawFighterDetails["DOB"]!= float("nan")])))
print("The total number of entries is ",len(RawFighterDetails["fighter_name"]))


# Reach,Stance and DOB will not be used from this data

# In[ ]:


RawTotalFightData=pd.read_csv("/kaggle/input/ufcdata/raw_total_fight_data.csv",delimiter =";")
RawTotalFightData.head()


# In[ ]:


data.dropna()
data.head()


# ## Feature Engineering and Exploratory Data Analysis
# Lets join the raw data processed data together first so we decide what features to use and construct. 

# In[ ]:


Data=data.join(RawTotalFightData,lsuffix="",rsuffix="_raw").copy()
for key in Data.keys():
    if key[-4:] == "_raw":
        print(key)
        Data=Data.drop([str(key)],axis=1)
print("These are all duplicates or will give no indication of who is going to win the fight.")
        
Data["date_time"]=pd.to_datetime(Data["date"])
Data=Data.drop(["date"],axis=1)
Data=Data.sort_values(["date_time"],ascending=True)

#there is some error in the way to data is recorded the first fight winners are all red
Data=Data.reset_index(drop=True)
for index,fight in Data.iterrows():
    if fight.Winner == "Blue":
        StartPoint=index
        break
Data=Data[StartPoint:]
Data=Data.reset_index(drop=True)
display(Data)
AllData=Data


# In[ ]:


display(AllData.head(0))


# We will use the data frame above to calculate the features for all the fights. We will calculate historical features about each fighter. Such as:
# 
# * **Probability of landing a significant strike**. This will be the number of significant strikes a fighter has landed over the number they have thrown.
# * **Significant strike rate.** The rate at which a fighter attempts to land significant strikes. Preferably per second but could be given per round if necessary.
# * **Probability of landing a strike**
# * **Strike Rate**
# * **Probability of landing a takedown.** A takedown is where one fighter wrestles the other fighter to the ground.
# * **Takedown attempt rate.**
# * **Takedown,Significant Strike and Strike defence probability**. The probability of defending the attack method based on historical fights.  
# * **Knock down rate**. A knock down is when one fighter knocks anther one down with a strike. 
# * **Win probability**
# * **total number of fights**
# * **Age**
# * **How many title bouts the fighter has been in**
# * **Network Centrality.** A network will be constructed using networkx and the an eigenvector centrality score score will be calculated for every fighter, we will start with katz centrality but may use some others later like PageRank. We will begin with creating a network for all the fighters and one for each weight class. For a fight we will consider each fighters score for the network of all fighters and their score in the weight class of the fights. 
# * **Finnish rate**. In MMA finishing an opponent is when one fighter knocks out or taps out an opponent. Clearly a fighter has knocked out every opponent they have faced in the first round they are good fighter. This feature will be number of finishes over total fight time.
# * **Current loss streak**
# * **Current win streak** 
# 
# 
# For training and testing it makes sense to split the fights chronologically rather than randomly. Training the algorithms on the oldest fights and testing them on more recent ones. 
# 
# Some other features we may include later are:
# * **Reach difference**  A fighters reach is the distance between their fingertips when they hold the arms apart and parallel to the ground. Reach can approximate the distance a fighter can punch at.  
# * **Height difference** Will give and indication who is punching with and who is punching against gravity. 
# 
# Lets work throw away any data we do not need to calculate the above features. 

# In[ ]:


CorneredThingsToKeep=["fighter","current_lose_streak","current_win_streak","wins",
                      "losses","draw","age","SIG_STR.","TOTAL_STR.","TD","total_title_bouts",
                     "win_by_KO/TKO","KD","total_time_fought(seconds)"]

ThingsToKeep=["Winner","win_by","last_round","last_round_time","date_time"]
for thing in CorneredThingsToKeep:
    ThingsToKeep.append("R_"+thing)
    ThingsToKeep.append("B_"+thing)
Data=AllData[ThingsToKeep]
display(Data.head(1))


# First lets adjust our columns to give total numbers of strikes,takedowns etc.

# In[ ]:


def ConvertToLandedAttempted(DF,Feature):
    for corner in ["R_","B_"]:
        try:
            DF[[corner+Feature+"_landed",corner+Feature+"_attempted"]] = DF[corner+Feature].str.split(" of ",expand=True)
            for label in [corner+Feature+"_landed",corner+Feature+"_attempted"]:
                DF[label]=pd.to_numeric(DF[label])
            DF=DF.drop([corner+Feature],axis=1)
        except:
            pass
    return DF

FeaturesToSplit=["SIG_STR.","TOTAL_STR.","TD"]
    
for feat in FeaturesToSplit:
    Data=ConvertToLandedAttempted(Data,feat)

for corner in ["R_","B_"]:
    try:
        Data[corner+"total_no._fights"]=pd.to_numeric(Data[corner+"draw"]+Data[corner+"wins"]+Data[corner+"losses"])
        Data=Data.drop([corner+"draw",corner+"losses"],axis=1)
    except:
        pass
    
    
Data.head().T


# In[ ]:


Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter","R_total_time_fought(seconds)","B_total_time_fought(seconds)","date_time"]]


# Total fight time isn't working so lets drop it.

# In[ ]:


Data=Data.drop([corner+"_total_time_fought(seconds)" for corner in ["R","B"]],axis=1)


# In[ ]:


Data.head(2)


# In[ ]:


# now lets create totals for all the revelevent factors

CorneredThingsToKeep=["fighter","current_lose_streak","current_win_streak","wins",
                      "losses","draw","age","SIG_STR.","TOTAL_STR.","TD","total_title_bouts",
                     "win_by_KO/TKO","KD","total_time_fought(seconds)"]

ThingsToKeep=["Winner","win_by","last_round","last_round_time","date_time"]

ThingstoTotal=["SIG_STR.","TOTAL_STR.","TD"]

for corner in ["R_","B_"]:
    for feature in ThingstoTotal:
        for Type in ["_attempted","_landed"]:
            stat=corner+feature+Type
            Data[stat+"_total"]=np.zeros(len(Data))
            
            
for index,fight in tqdm(Data.iterrows(),total=len(Data)):
    for corner in ["R_","B_"]:
        for feature in ThingstoTotal:
            for Type in ["_attempted","_landed"]:
                stat=corner+feature+Type
                fighter=str(fight[corner+"fighter"])
                
                #find the index of the previou fight involving the fighter 
                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]
                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index
                if len(PreviousFightsIndex)> 0:
                    PrevIndex=PreviousFightsIndex[-1]
                else:
                    continue 
                PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                PrevStatLabel=PrevCorner+feature+Type
                PrevTotal=Data[PrevStatLabel+"_total"][PrevIndex] 
                PrevStat=Data[PrevStatLabel][PrevIndex]
                Data[stat+"_total"][index]=PrevTotal+PrevStat
                
def TestIfWorking(Data,Features):
    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter",*Features]])
                


# In[ ]:


def TestIfWorking(Data,Features):
    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter"]+Features])
            
TestIfWorking(Data,[corner+"SIG_STR._attempted_total" for corner in ["R_","B_"]]+[corner+"SIG_STR._attempted" for corner in ["R_","B_"]])            


# Now we will crate total fight time so we can calculate the rates. 

# In[ ]:


Data["last_round_time"]=[int(minutes)*60 + int(seconds) for minutes,seconds in Data.last_round_time.str.split(":")]
Data.last_round=pd.to_numeric(Data.last_round)
Data["fight_time"]=(Data.last_round-1)*60*5+Data.last_round_time


#totalling the time
def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):
    if type(Stats)==str:
        Stats=[Stats]
    for stat in Stats:
        for corner in ["R_","B_"]:
            try:
                Data[corner+stat+Descriptor+"_total"]
            except:
                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))
        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):
            for corner in ["R_","B_"]:
                fighter=str(fight[corner+"fighter"])
                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]
                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index
                if len(PreviousFightsIndex)> 0:
                    PrevIndex=PreviousFightsIndex[-1]
                else:
                    continue
                if CornerDependence==True:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                elif CornerDependence==False:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner=""
                elif CornerDependece == "Reverse":
                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+stat][PrevIndex]
    return Data


            
Data=TotalStat("fight_time",Data,CornerDependence=False)



TestIfWorking(Data,[corner+"fight_time"+"_total" for corner in ["R_","B_"] ]+["fight_time"])


# In[ ]:


#totaling the defensive stats

def TotalDefenceStat(Stats,Data,CornerDependence="Reverse"):
    if type(Stats)==str:
        Stats=[Stats]
    for stat in Stats:
        for ending,Descriptor in [["_attempted","_faced"],["_landed","_defended"]]:
            StatEnd=stat+ending
            for corner in ["R_","B_"]:
                try:
                    Data[corner+stat+Descriptor+"_total"]
                except:
                    Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))
            for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):
                for corner in ["R_","B_"]:
                    fighter=str(fight[corner+"fighter"])
                    PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]
                    PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index
                    if len(PreviousFightsIndex)> 0:
                        PrevIndex=PreviousFightsIndex[-1]
                    else:
                        continue
                    if CornerDependence==True:
                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                        PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    elif CornerDependence==False:
                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                        PrevCorner=""
                    elif CornerDependence == "Reverse":
                        PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"
                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    if Descriptor == "_faced":
                        Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+StatEnd][PrevIndex]
                    else:
                        Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+(Data[PrevCorner+stat+"_attempted"][PrevIndex]-Data[PrevCorner+StatEnd][PrevIndex])
    return Data

Features=[stat for stat in ["SIG_STR.","TOTAL_STR.","TD"]]




    
Data=TotalDefenceStat(Features,Data)
TestIfWorking(Data,[key for key in Data.keys() if "SIG_STR." in key])


# In[ ]:


def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):
    if type(Stats)==str:
        Stats=[Stats]
    for stat in Stats:
        for corner in ["R_","B_"]:
            try:
                Data[corner+stat+Descriptor+"_total"]
            except:
                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))
        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):
            for corner in ["R_","B_"]:
                fighter=str(fight[corner+"fighter"])
                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]
                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index
                if len(PreviousFightsIndex)> 0:
                    PrevIndex=PreviousFightsIndex[-1]
                else:
                    continue
                if CornerDependence==True:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                elif CornerDependence==False:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner=""
                elif CornerDependece == "Reverse":
                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+stat][PrevIndex]
    return Data


Data=TotalStat("KD",Data)
TestIfWorking(Data,[key for key in Data.keys() if "KD" in key])


# In[ ]:


#calculting the finnish probability for each fighter 
# we will do this by totaling number of finnishes and then dividing by total fight time
display("The different win_by options ",Data.win_by.unique())

def IsFinnish(DF,corner="Undefined"):
    CornerPrefix="R_" if corner=="Red" else "B_"
    if DF["Winner"]==corner and DF["win_by"] in ['KO/TKO', 'Submission']:
        return 1
    return 0

for corner in ["Red","Blue"]:        
    Data[corner[0]+"_finish"]=Data.apply(IsFinnish,**{"corner":corner},axis=1)

Data=TotalStat("finish",Data)
TestIfWorking(Data,[key for key in Data.keys() if "finish" in key ])


# Now we can construct all our features.

# In[ ]:


Features=[]
for corner in ["R_","B_"]:
    for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:
        Data[corner+AttackForm+"_probability"]=Data[corner+AttackForm+"_landed_total"]/Data[corner+AttackForm+"_attempted_total"]
        Features.append(AttackForm+"_probability")
        Data[corner+AttackForm+"_defence_probability"]=Data[corner+AttackForm+"_defended_total"]/Data[corner+AttackForm+"_faced_total"]
        Features.append(AttackForm+"_defence_probability")
        Data[corner+AttackForm+"_rate"]=Data[corner+AttackForm+"_attempted_total"]/Data[corner+"fight_time"+"_total"]
        Features.append(AttackForm+"_rate")
        
    Data[corner+"KD_rate"]=Data[corner+"KD"+"_total"]/Data[corner+"fight_time"+"_total"]
    Features.append("KD_rate")
    
    Data[corner+"win_probability"]=Data[corner+"wins"]/Data[corner+"total_no._fights"]
    Features.append("win_probability")
    
    Data[corner+"finish_rate"]=Data[corner+"finish_total"]/Data[corner+"fight_time"+"_total"]
    Features.append("finish_rate")
    
    Features+=[corner+factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak"]]


# In[ ]:


display(Data[-10:])


# It will be best to fill the NaNs in with medians. We will also remove fights that where won by (win_by in the Data Frame)  disqualification (DQ), doctors stoppage (TKO - Doctor's Stoppage), and if the decision is overturned (Overturned). These fights can be considered anomalous. Then we will set (Winner) to 1 if red wins and 0 if blue wins. The can train some algorithms before calculating centrality scores for the fighters.  

# In[ ]:


Data=Data[~Data.win_by.isin(["DQ","TKO - Doctor's Stoppage","Overturned"])]

def BinaryWinner(DF):
    if DF["Winner"]=="Red":
        return 1
    elif DF["Winner"]=="Blue":
        return 0 
    else:
        return np.nan
Data["Winner"]=Data.apply(BinaryWinner,axis=1)
Data.head()


# In[ ]:


print(len(Data.dropna())*100.0/len(Data),"% of the fights don't have NaN in them")
print(Data.Winner.mean()*100,"% of the fights were won ")


# It makes sens to split randomly to reduce the bias. The UFC puts the favoured fighter in the Red corner and it seems they may have changed how they decide who is the favoured fighter. We appear to get better results when we replace NaNs with median as well. Most of the NaNs occur for a fighters first fight as they we have we don't have any data for them in previous fights. All models all less accurate or only marginally more accurate than guessing red for every fight. Now I will construct networks of all the fights and calculate the centralities of the networks. 

# In[ ]:


FightNetwork=nx.MultiDiGraph()
for index,fight in tqdm(Data.iterrows(),total=len(Data)):
    if fight.Winner == 1:
        FightNetwork.add_edge(fight.B_fighter,fight.R_fighter,key=fight.date_time)
    elif fight.Winner == 0:
        FightNetwork.add_edge(fight.R_fighter,fight.B_fighter,key=fight.date_time)
fig=plt.figure(figsize=(12,12),dpi=100)
nx.draw(FightNetwork,pos=nx.spring_layout(FightNetwork,k=0.4),arrowsize=0.1,font_size=1,width=0.02,node_size=0,with_labels=True)
fig.show()


# In[ ]:


Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)')


# In[ ]:


def PrintScores(Scores,number):
    for i in  sorted(Scores,key=Scores.get,reverse=True)[:number]:
        print(i,Scores[i])

PrintScores(Centralities,30)


# It takes a long time to calculate the centrality scores so instead of recalculating after every fight we will train for all fights on the same day. 

# One method to find the best parameters for Katz centrality is to use gradient descent to maximize the accuracy of a linear SVM trained only on the Katz centrality features. This may be attempted at a later date. A computaitionaly less intensive approach would be to maximise the difference between the mean centrality scores for red wins and blue wins. 

# In[ ]:


RedScores,BlueScores=[],[]
RedFighters,BlueFighters=[],[]
TotalCentralities=[]
Cententralities={}

FightNetwork=nx.MultiDiGraph()
SumCentralities=np.nan
Indexes=list(Data.index)
for (index,fight),NextIndex in tqdm(zip(Data.iterrows(),Indexes[1:]),total=len(Data)-1):
    try:
        RedScores.append(Cententralities[fight.R_fighter])
    except:
        RedScores+=[np.median(RedScores) if len(RedScores)>0 else np.nan]
    try:
        BlueScores.append(Cententralities[fight.B_fighter])
    except:
        BlueScores+=[np.median(BlueScores) if len(BlueScores)>0 else np.nan]
    
    if fight.Winner == 1:
        FightNetwork.add_edge(fight.B_fighter,fight.R_fighter,key=fight.date_time)
    elif fight.Winner == 0:
        FightNetwork.add_edge(fight.R_fighter,fight.B_fighter,key=fight.date_time)
    
    TotalCentralities.append(SumCentralities)    
    if fight.date_time!=Data.date_time[NextIndex]:
        Cententralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)
        SumCentralities=np.sum(list(Cententralities.values()))


try:
    RedScores.append(Cententralities[list(Data.R_fighter)[-1]])
except:
    RedScores+=[np.median(RedScores)]
try:
    BlueScores.append(Cententralities[list(Data.B_fighter)[-1]])
except:
    BlueScores+=[np.median(BlueScores)]

TotalCentralities.append(TotalCentralities[-1])
    
Data["R_centrality"]=RedScores
Data["B_centrality"]=BlueScores
Data["total_centrality"]=TotalCentralities
  
    


# In[ ]:


Data["total_centrality"].describe()


# In[ ]:


print("The first 10 non NaN maximum centrality scores are:")
count=0
for index,fight in Data.iterrows():
    if fight.R_centrality >0 or fight.B_centrality > 0:
        print(np.nanmax([fight.R_centrality,fight.B_centrality]))
        count+=1
        if count == 10:
            break
            
print("\nThe last 10 maximum centrality scores are:")
count=0
for index,fight in Data[::-1].iterrows():
    if fight.R_centrality >0 or fight.B_centrality > 0:
        print(np.nanmax([fight.R_centrality,fight.B_centrality]))
        count+=1
        if count == 10:
            break
            
print("\nThe maximum centrality is",np.nanmax(list(Data.R_centrality)+list(Data.B_centrality)))


# Clearly the centrality for early fights is large compared to later fights. To try and account for this I will consider the difference between centralities (Red-Blue) over the sum of them and another feature with the difference over the product.

# Lets see if the centralities have any validity for predicting the outcome fights. I will use the full fight network too see how well the centralities predict the result of historical fights. This is representitive how the scores could be used to in reality as we cannot use the future to predict the past.  

# In[ ]:


import random
Cententralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)

# R_SIG_STR._attempted is just a random thing to plot against 
CentralititesData=Data[["R_fighter","B_fighter","Winner","R_SIG_STR._attempted"]]

CentralititesData["R_centrality"]=[Cententralities[fighter] if fighter in Cententralities.keys() else np.median(list(Cententralities.values())) for fighter in list(CentralititesData["R_fighter"])]
CentralititesData["B_centrality"]=[Cententralities[fighter] if fighter in Cententralities.keys() else np.median(list(Cententralities.values())) for fighter in list(CentralititesData["B_fighter"])]
CentralititesData["centrality_difference"]=(CentralititesData["R_centrality"]-CentralititesData["B_centrality"])/(CentralititesData["R_centrality"]+CentralititesData["B_centrality"])

Winners=list(CentralititesData["Winner"])
cols=["b","r"]
winner=[0.0,1.0]
Colors=[cols[winner.index(x)] if x in [0.0,1.0] else random.choice(cols) for x in Winners]
plt.scatter(CentralititesData["centrality_difference"],CentralititesData["R_SIG_STR._attempted"],color=Colors,alpha=0.1)
plt.xlabel("Centrality Difference")
plt.ylabel("Red Fighter's Number of Attempted Strikes")

plt.show()


# The centralities separate the fights well if we use the most the up to date fight network now lets see if they area as predictive if we only use historical data. 

# In[ ]:


Data["centrality_difference/total"] = (Data["R_centrality"] - Data["B_centrality"])/Data["total_centrality"]
Data["centrality_difference/product"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"]*Data["B_centrality"])

Features=[]
for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:
    Features.append(AttackForm+"_probability")
    Features.append(AttackForm+"_defence_probability")
    Features.append(AttackForm+"_rate")
Features.append("KD_rate")
Features.append("win_probability")
Features.append("finish_rate")
Features+=[factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak"]]
Features=[corner+feat for corner in ["R_","B_"] for feat in Features]
Features+=["R_centrality","B_centrality","centrality_difference/total","centrality_difference/product"]
print(Features)


# In[ ]:


import seaborn as sns 
StuffToPlot=Data[Features+["Winner"]].fillna(Data[Features+["Winner"]].median())
PairPlot=sns.pairplot(Data[Features+["Winner"]],hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.03})


# Unfortunately the data does not appear to be very separable. We could reduce the dimensionality of the data by changing all the features to differences between the red corner and blue corner over the sum of both. The data still doesn't appear very separable lets try and make some more predictive features such as: 
# 
# **Reach Difference** The difference in reach between the two fighters against divided by the sum of the reaches. To be made later. 

# **Expected Landing Rate for each Attack Form** for a given attack form $a$ given by
# $$ a_p(1-a^{opp}_{dp})a_r .$$ Where $a_p$ is the previously discussed probability of fighter landing a strike based their historical fights. $a^{opp}_{dp}$ is the probability of the opposition fighter defending the attack form. Finally $a_r$ is the rate the fighter attempts this attack form. The theoretical motivation for this feature is it is equal rate the fighter lands the attack form multiplied by the probability of the opposition fighter not defending the attack form. This is not actually the expected landing rate for a given attack form but can be considered related to it. This will be calculated for the three attack forms strikes, significant strikes and takedowns. 
# 
# An alternative approach would be to create two features:
# $$ a_r\frac{a_p}{a_p + a^{opp}_{dp}} \,\,\,\,\,\, \& \,\,\,\,\,\, \frac{a^{opp}_{dp}}{a_p + a^{opp}_{dp}} .$$ Here the fractions (probabilities?) total one like they would in reality.

# In[ ]:


#ELR = estimated landing rate
ELRFeatures=["TD","SIG_STR.","TOTAL_STR."]

NewFeatures=[]
Corners=["R_","B_"]
for feature in ELRFeatures:
    for corner in Corners:
        OtherCorner=Corners[Corners.index(corner)-1]
        NewFeature=corner+feature+"_ELR"  
        Start=corner+feature #  prob of landing     prob of not defending                                rate of attempts
        Data[NewFeature]=Data[Start+"_probability"]*(1-Data[OtherCorner+feature+"_defence_probability"])*Data[Start+"_rate"]
        NewFeatures.append(NewFeature)
        
ELRFeatures=NewFeatures 


# Lets inspect for fighters with at least 4 fights.

# In[ ]:


import seaborn as sns 

DataToPlot=Data[Data["R_total_no._fights"] > 3]
DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]
DataToPlot=DataToPlot[NewFeatures+["Winner"]]
DataToPlot=DataToPlot.dropna()
print(len(DataToPlot))

StuffToPlot=DataToPlot
PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.01})


# Lets also look at the difference between the ELRs.

# In[ ]:


ELRDifferenceFeatures=[]
for feature in ["TD","SIG_STR.","TOTAL_STR."]: 
    NewFeature=feature+"_ELR_difference"
    Data[NewFeature]=(Data["R_"+feature+"_ELR"]-Data["B_"+feature+"_ELR"])/(Data["R_"+feature+"_ELR"]+Data["B_"+feature+"_ELR"])
    ELRDifferenceFeatures.append(NewFeature)
    


# In[ ]:


DataToPlot=Data[Data["R_total_no._fights"] > 3]
DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]
DataToPlot=DataToPlot[ELRDifferenceFeatures+["Winner"]]
DataToPlot=DataToPlot.dropna()

DataToPlot=DataToPlot[ELRDifferenceFeatures+["Winner"]]
print(len(DataToPlot))


PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})


# *TD_ELR_difference* looks very predictive, now lets look at for any number of fights. 

# In[ ]:


DataToPlot=Data[ELRDifferenceFeatures+["Winner"]]
DataToPlot=DataToPlot.dropna()
print(len(DataToPlot))

PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})


# The ELR of takedowns appears to very predictive if we only consider fights where both fighter have had more than 4 fights. Lets see if we can make a more predictive statistic. We can create a new statistic we will call *Weighted attack Probability*,
# $$  = \frac{1}{\textit{Red Total Attacks Attempted}} \sum_{fights} (\textit{Blue Attack Defense})(\textit{Red Attacks landed})$$.
# 
# We well weight the attack forms significant strikes, strikes and takedowns in this way then calculate *ELR*s and *ELR* differences using the new probabilities.

# In[ ]:


def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):
    if type(Stats)==str:
        Stats=[Stats]
    for stat in Stats:
        for corner in ["R_","B_"]:
            try:
                Data[corner+stat+Descriptor+"_total"]
            except:
                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))
        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):
            for corner in ["R_","B_"]:
                fighter=str(fight[corner+"fighter"])
                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]
                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index
                if len(PreviousFightsIndex)> 0:
                    PrevIndex=PreviousFightsIndex[-1]
                else:
                    continue
                if CornerDependence==True:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                elif CornerDependence==False:
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    PrevCorner=""
                elif CornerDependece == "Reverse":
                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"
                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"
                    
                PrevStat=Data[PrevCorner+stat][PrevIndex] if not np.isnan(Data[PrevCorner+stat][PrevIndex]) else 0
                PrevTotal=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex] if not np.isnan(Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]) else 0
            
                Data[corner+stat+Descriptor+"_total"][index]=PrevTotal+PrevStat
    return Data


AttackForms=["SIG_STR.","TOTAL_STR.","TD"]

for index,corner in enumerate(["R_","B_"]):
    OtherCorner=["R_","B_"][index-1]
    for attack in AttackForms:
        Data[corner+attack+"_weighted_landed"]=Data[OtherCorner+attack+"_defence_probability"]*Data[corner+attack+"_landed"]

WeightedLanded=[attack+"_weighted_landed" for attack in AttackForms]

Data=TotalStat(WeightedLanded,Data)

WeightedProbs=[]
for corner in ["R_","B_"]:
    for attack in AttackForms:
        Data[corner+attack+"_weighted_probability"]=Data[corner+attack+"_weighted_landed_total"]/(Data[corner+attack+"_attempted_total"]+0.0000000001)
        WeightedProbs.append(corner+attack+"_weighted_probability")

Data.replace({feature:0.0 for feature in WeightedProbs},np.nan)
    
def TestIfWorking(Data,Features):
    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter",*Features]])

    
TestIfWorking(Data,[key for key in list(Data.keys()) if "SIG_STR." in str(key)])


# In[ ]:


DataToPlot=Data
DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]
DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]
DataToPlot=DataToPlot[[corner + x + "_weighted_probability" for x in AttackForms for corner in ["R_","B_"]]+["Winner"]]

for feature in [corner + x + "_weighted_probability" for x in AttackForms for corner in ["R_","B_"]]:
    DataToPlot[DataToPlot[feature] != 0]
DataToPlot=DataToPlot.dropna()

print("plot for",len(DataToPlot),"fights")


PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})


# In[ ]:


for attack in AttackForms:
    Data[attack+"_weighted_probability_difference"]=(Data["R_"+attack+"_weighted_probability"]-Data["B_"+attack+"_weighted_probability"])/(Data["R_"+attack+"_weighted_probability"]+Data["B_"+attack+"_weighted_probability"])


# In[ ]:


DataToPlot=Data
DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]
DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]
DataToPlot=DataToPlot[[attack+"_weighted_probability_difference" for attack in AttackForms]+["Winner"]]

for feature in [attack+"_weighted_probability_difference" for attack in AttackForms]:
    DataToPlot=DataToPlot[DataToPlot[feature] != 0]

DataToPlot=DataToPlot.dropna()
print("plot for",len(DataToPlot),"fights")


PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})


# In[ ]:


Corners=["R_","B_"]
for feature in ["TD","SIG_STR.","TOTAL_STR."]:
    for corner in Corners:
        OtherCorner=Corners[Corners.index(corner)-1]
        Data[corner+feature+"_weighted_ELR"]=Data[corner+feature+"_weighted_probability"]*(1-Data[OtherCorner+feature+"_defence_probability"])*Data[corner+feature+"_rate"]

FeaturesToPlot=[]
for feature in ["TD","SIG_STR.","TOTAL_STR."]:
    Data[feature+"_weighted_ELR_difference"]=(Data["R_"+feature+"_weighted_ELR"]-Data["B_"+feature+"_weighted_ELR"])/(Data["R_"+feature+"_weighted_ELR"]+Data["B_"+feature+"_weighted_ELR"])
    FeaturesToPlot.append(feature+"_weighted_ELR_difference")

Data["centrality_difference/total"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"] + Data["B_centrality"])
Data["centrality_difference/product"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"]*Data["B_centrality"])


# In[ ]:


import seaborn as sns

DataToPlot=Data
DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]
DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]
DataToPlot=DataToPlot[FeaturesToPlot+["Winner"]]


DataToPlot=DataToPlot.dropna()
print("plot for",len(DataToPlot),"fights")

PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})


# These weighted takedown ELRs don't appear to be as predictive as the standard ELRs so for now we will just train the models on those. 

# ## Model Training and evaluation
# Lets only train models on fights where both fighters have had at least 4 fights in UFC so we have data against a range of oppenants. 

# In[ ]:


import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

Features=[]
for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:
    Features.append(AttackForm+"_probability")
    Features.append(AttackForm+"_defence_probability")
    Features.append(AttackForm+"_rate")
Features.append("KD_rate")
Features.append("win_probability")
Features.append("finish_rate")
Features+=[factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak","centrality"]]
Features=[corner+feat for corner in ["R_","B_"] for feat in Features]
for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:
    Features.append(AttackForm+"_ELR_difference")
Features.append("centrality_difference/total")
Features.append("centrality_difference/product")

DataToUse=Data
DataToUse=DataToUse[DataToUse["R_total_no._fights"] > 3 ]
DataToUse=DataToUse[DataToUse["B_total_no._fights"] > 3 ]
DataToUse=DataToUse[Features+["Winner"]]
DataToUse=DataToUse.dropna()
print("We have data for",len(DataToUse),"fights")
TransformData=StandardScaler()

Continue=True
while Continue:
    TrainX,TestX,TrainY,TestY=train_test_split(DataToUse[Features].copy(),DataToUse["Winner"].copy(),test_size=0.25)
    if 0.99<np.mean(TrainY)/np.mean(TestY)<1.01:
        Continue=False

print("Red wins",np.mean(TrainY)*100,"% of the training fights")
print("Red wins",np.mean(TestY)*100,"% of the test fights")
TransformData.fit_transform(TrainX)
TransformData.transform(TestX)


# In[ ]:


parameters = {'kernel':['rbf'], 'C':[0.2,0.3,0.4,0.5]}
print("For a rbf SVC")
Model=SVC(gamma="auto")
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


print("For a polynomial SVC of degree 2")
Model=SVC(gamma="auto")
parameters = {'kernel':["poly"],"degree":[2], 'C':[0.1,0.3,0.7,0.9,0.5,1]}
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


print("For a linear SVC")
Model=SVC(gamma="auto")
parameters = {'kernel':["linear"], 'C':[1,3,5,7,10]}
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


print("For a Random Forrest")
Model=RandomForestClassifier(n_jobs=-1)
parameters = {'n_estimators':[100,200,300,1000],"max_depth":[None,200,120,100,80,10],"bootstrap":[False]}
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


print("For KNN")
Model=KNeighborsClassifier()
parameters = {'n_neighbors':[4,5,7,8],"weights":["uniform","distance"]}
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


print("For Logistic Regression")
Model=LogisticRegression(max_iter=1000000000,tol=0.00001)
parameters = {'penalty':["l1","l2"],"C":[0.01,0.1,0.2,0.33,1.0]}
ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")
ModelTuner.fit(TrainX,TrainY)
print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")
print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")
print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())


# In[ ]:


import tensorflow as tf
#from tensorflow.keras import backend.tensorflow_backend.set_session 
from tensorflow import keras
           
TrainX = np.array(TrainX.astype("float64"))
TestX = np.array(TestX.astype("float64"))

TrainY = np.array(TrainY.astype("float64"))
TestY = np.array(TestY.astype("float64"))


# In[ ]:



Optimiser=keras.optimizers.SGD(learning_rate=0.00006,momentum=0.7)

Model=keras.Sequential()
Model.add(keras.layers.Dense(32,input_dim=int(np.shape(TrainX)[1]),activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(1,activation="sigmoid"))

Model.compile(loss="binary_crossentropy",optimizer=Optimiser,metrics=["accuracy"])
History=Model.fit(TrainX,TrainY,batch_size=60,epochs=4000,validation_data=(TestX,TestY),verbose=0)


# In[ ]:


def MovingAverage(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    MovAv=list(a[:n-1])+list(ret[n - 1:] / n)
    return MovAv

Xaxis=range(0,len(History.history["accuracy"]),10)
YAcc=MovingAverage(History.history["accuracy"][::10])
YValAcc=MovingAverage(History.history["val_accuracy"][::10])
plt.plot(Xaxis,YAcc,label="Train set")
plt.plot(Xaxis,YValAcc,label= "Test set")
plt.grid(linestyle="-.")

plt.legend()
plt.show()    

print("Red wins",np.mean(TrainY),"% and",np.mean(TestY),"% of the training and test fights respectivly")
print("The max accurcy on the test set is",np.max(History.history["val_accuracy"]))


# In[ ]:


Optimiser=keras.optimizers.SGD(learning_rate=0.00004,momentum=0.7)

Model=keras.Sequential()
Model.add(keras.layers.Dense(32,input_dim=np.shape(TrainX)[1],activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(32,activation='elu'))
Model.add(keras.layers.Dense(1,activation="sigmoid"))

Model.compile(loss="binary_crossentropy",optimizer=Optimiser,metrics=["accuracy"])


BigNetHistory=Model.fit(TrainX,TrainY,batch_size=60,epochs=4000,validation_data=(TestX,TestY),verbose=0)


# In[ ]:


Xaxis=range(0,len(BigNetHistory.history["accuracy"]),10)

YAcc=MovingAverage(BigNetHistory.history["accuracy"][::10])
YValAcc=MovingAverage(BigNetHistory.history["val_accuracy"][::10])

plt.plot(Xaxis,YAcc,label="Train Set")
plt.plot(Xaxis,YValAcc,label="Test Set")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(linestyle="-.")
plt.show()    

print("Red wins",np.mean(TrainY),"% and",np.mean(TestY),"% of the training and test fights respectivly")
print("The max accurcy on the test set is",np.max(BigNetHistory.history["val_accuracy"]))


# The graphs appear to cross around 64% so after that I would expect the test score to decrease due to overtraining. This appears to be around the best accuracy we can get for traditional supervised learning models, $\approx$64%. 
# 
# Further algorithms that could be attempted are gradient boosting methods. 
