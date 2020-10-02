#!/usr/bin/env python
# coding: utf-8

# # SOCCER ANALYSIS

# **In this notebook, we aim to use the datasets from https://www.kaggle.com/secareanualin/football-events/data to analyse on the patterns of events happening in soccer matches.**
# 
# The outline of the notebook would be:
# 
# -  **1. Importing libraries and datasets**
# -  **2. Analysing Goals Scored**
# -  **3. Analysing Substitutions**
# -  **4. Analysing Yellow/Red Cards**
# -  **5. Analysing Penalties**
# -  **6. Analysing game odds and results**

# # 1. Importing libraries and datasets

# In[ ]:


import zipfile
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


events = pd.read_csv("../input/events.csv")
game_info=pd.read_csv("../input/ginf.csv")


# In[ ]:


events.head()


# In[ ]:


events.info()


# **Understanding the representation of interger coded variables**
# 
# The following would show the events related to each integer coded variable in the dataset.

# In[ ]:


encoding = pd.read_table('../input/dictionary.txt', delim_whitespace=False, names=('num','events'))
event_type=encoding[1:13]
event_type_2=encoding[14:18]
side=encoding[19:21]
shot_place=encoding[22:35]
shot_outcome=encoding[36:40]
location=encoding[41:60]
bodypart=encoding[61:64]
assist_method=encoding[65:70]
situition=encoding[71:75]


# In[ ]:


event_type


# In[ ]:


event_type_2


# In[ ]:


side


# In[ ]:


shot_place


# In[ ]:


shot_outcome


# In[ ]:


location


# In[ ]:


bodypart


# In[ ]:


assist_method


# In[ ]:


situition
#Note that 2 refers to set-piece excluding direct free kicks


# # 2. Analysing Goals Scored

# In[ ]:


goals=events[events["is_goal"]==1]


# # 2.1 Analysing goals against Time

# In[ ]:


fig=plt.figure(figsize=(8,6))
plt.hist(goals.time,width=1,bins=100,color="green")   #100 so 1 bar per minute
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.title("Number of goals against Time during match")


# The plot above shows:
# -  Most goals are scored around the **Half-Time (45mins + extra time)** and around **Full-Time (90mins + extra time)**
# 

# # 2.2 Analysing Home/Away goals

# In[ ]:


fig=plt.figure(figsize=(8,6))
plt.hist(goals[goals["side"]==1]["time"],width=1,bins=100,color="cyan",label="home goals")   
plt.hist(goals[goals["side"]==2]["time"],width=1,bins=100,color="grey",label="away goals") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by home/away side) against Time during match")


# The plot above shows:
# -  For every minute, most of the goals scored are by the **home side**
# 
# This supports the general notion that the home side has a statistical advantage.

# # 2.3 Analysing HOW and WHERE goals are scored

# In[ ]:


plt.subplot(2,1,1)
labels=["Right Foot","Left Foot","Headers"]
sizes=[goals[goals["bodypart"]==1].shape[0],goals[goals["bodypart"]==2].shape[0],goals[goals["bodypart"]==3].shape[0]]
colors=["cyan","grey","pink"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of bodyparts for goals",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)


plt.subplot(2,1,2)
plt.hist(goals[goals["bodypart"]==1]["time"],width=1,bins=100,color="cyan",label="Right foot")   
plt.hist(goals[goals["bodypart"]==2]["time"],width=1,bins=100,color="grey",label="Left foot") 
plt.hist(goals[goals["bodypart"]==3]["time"],width=1,bins=100,color="pink",label="Headers") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by body parts) against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.tight_layout()


# The plots above shows:
# -  Most of the goals (slightly more than half) scored are by **Right Footed, followed by Left Footed and lastly, by Headers.**
# 
# Perhaps this might be because the majority of humans are right-footed and hence, most players are right-footed. It is also not surprising that most goals have been scored by foot not head, as after all, soccer is meant to be played by foot.

# In[ ]:


plt.subplot(2,1,1)
labels=["Open Play", "Set Piece(Excluding direct Free kick)","Corners","Direct Free Kick"]
sizes=[goals[goals["situation"]==1].shape[0],goals[goals["situation"]==2].shape[0],goals[goals["situation"]==3].shape[0],goals[goals["situation"]==4].shape[0]]
colors=["cyan","grey","blue","yellow"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of each situation for goals",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)

plt.subplot(2,1,2)
plt.hist(goals[goals["situation"]==1]["time"],width=1,bins=100,color="cyan",label="Open play")   
plt.hist(goals[goals["situation"]==2]["time"],width=1,bins=100,color="grey",label="Set Piece (excluding direct FreeKick)") 
plt.hist(goals[goals["situation"]==3]["time"],width=1,bins=100,color="blue",label="Corners") 
plt.hist(goals[goals["situation"]==4]["time"],width=1,bins=100,color="yellow",label="Direct Free Kick") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by situations) against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.tight_layout()


# The plots show:
# -  About 70.8% of the goals scored are from **open play** 

# We now check for which "locations" that did not result in a goal so we would not need to include them in our analysis.

# In[ ]:


for i in range(20):
    if sum(goals["location"]==i)==0:
        print(i)


# In[ ]:


diff_angle_goals=sum(goals["location"]==6)+sum(goals["location"]==7)+sum(goals["location"]==8)
long_range_goals=sum(goals["location"]==16)+sum(goals["location"]==17)+sum(goals["location"]==18)
box_goals=sum(goals["location"]==3)+sum(goals["location"]==9)+sum(goals["location"]==11)+sum(goals["location"]==15)
close_range_goals=sum(goals["location"]==10)+sum(goals["location"]==12)+sum(goals["location"]==13)
penalties=sum(goals["location"]==14)
not_recorded=sum(goals["location"]==19)

labels=["Long Range goals","Difficult angle goals","Goals from around the box","Close range goals","Penalties","Not recorded"]
sizes=[long_range_goals,diff_angle_goals,box_goals,close_range_goals,penalties,not_recorded]
colors=["gray","yellow","aqua","coral","red","violet"]
plt.pie(sizes,colors=colors,autopct='%1.1f%%',startangle=60,pctdistance=0.8,radius=3)
plt.axis('equal')
plt.title("Percentage of each location for goals",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
plt.legend(labels)
fig=plt.gcf() 
fig.set_size_inches(12,8)
plt.show()


# The pie chart shows:
# -  Most of the goals scored are attempted from **around the box** (64.5%) and **close range** (19.4%)

# # 2.4 Analysing Assists for goals

# In[ ]:


labels=["None","Pass","Cross","Headed Pass","Through Ball"]
sizes=[sum(goals["assist_method"]==0),sum(goals["assist_method"]==1),sum(goals["assist_method"]==2),sum(goals["assist_method"]==3),sum(goals["assist_method"]==4)]
colors=["palegreen","yellow","aqua","coral","plum"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of each assist type for goals",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
fig=plt.gcf()  #gcf --> current figure
fig.set_size_inches(8,6)
plt.show()


# The pie chart shows:
# -  About 35.4% of the goals have been assisted by **direct passing**
# -  32.2% of the goals have **"No" assist** because they might be from penalties or direct free kicks

# # 3. Analysing Substitutions

# In[ ]:


substitution=events[events["event_type"]==7]


# In[ ]:


fig=plt.figure(figsize=(8,6))
plt.hist(substitution.time,width=1,bins=100,color="green")   
plt.xlabel("Minutes")
plt.ylabel("Number of substitutions")
plt.title("Number of substitutions against Time during match")


# The plot shows:
# -  Most substitutions happen in the **Second Half (45-90mins)**
# -  Bulk of substitutions around  **Half-Time (45mins + extra time)** and around **Full-Time (90mins + extra time)**
#   -  Understandable as players get more tired in the second half, managers has to change tactics and players as the game progresses
#  

# In[ ]:


plt.subplot(2,1,1)
labels=["Home","Away"]
sizes=[substitution[substitution["side"]==1].shape[0],substitution[substitution["side"]==2].shape[0]]
colors=["green","aqua"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.title("Percentage of substitutions by Home/Away",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)

plt.subplot(2,1,2)
plt.hist(substitution[substitution["side"]==1].time,width=1,bins=100,color="green",label="home")
plt.hist(substitution[substitution["side"]==2].time,width=1,bins=100,color="aqua",label="away")
plt.xlabel("Minutes")
plt.ylabel("Number of substitutions")
plt.legend()
plt.title("Number of substitutions (home/away) against Time during match",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")


# The plots show:
# -  Number of substitutions made are almost the same for both home and away (Obvious because both sides would have 3 substitutions per game)
# -  Around the 80-90mins, slightly more substitutions were made by the home side rather than the away side

# # 4. Analysing Yellow cards/Red cards

# **4.1 Yellow Cards**

# In[ ]:


yel_card=events[events["event_type"]==4]
yel_card.shape


# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(yel_card.time,width=8,bins=9,color="yellow")   
plt.xlabel("Minutes")
plt.ylabel("Number of yellow cards")
plt.title("Number of yellow cards against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")


# In[ ]:


yel_home=yel_card[yel_card["side"]==1].shape[0]
yel_away=yel_card[yel_card["side"]==2].shape[0]

yel_combi=pd.DataFrame({"home":yel_home,"away":yel_away},index=["Yellow cards"])
yel_combi.plot(kind="bar")
plt.title("Number of yellow cards (Home/Away)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
yel_combi


# We observe that:
# -  Most of the yellow cards are issued in the second half, particularly around the 80mins
#   -  As player accumulates fouls, or as temper rises in the match, players are more prone to making rash tackles, arguing, fighting etc which results in a booking for them
# -  More yellow cards has been issued to the Away Team compared to the Home Team 
#  

# **4.2 Red cards**

# In[ ]:


sec_yellow=events[events["event_type"]==5]
red=events[events["event_type"]==6]
reds=[sec_yellow,red]
red_cards=pd.concat(reds)
red_cards.event_type.unique()


# In[ ]:


fig=plt.figure(figsize=(8,6))
plt.hist(red_cards.time,width=8,bins=9,color="red")   
plt.xlabel("Minutes")
plt.ylabel("Number of red cards")
plt.title("Number of red cards against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")


# In[ ]:


red_home=red_cards[red_cards["side"]==1].shape[0]
red_away=red_cards[red_cards["side"]==2].shape[0]

red_combi=pd.DataFrame({"home":red_home,"away":red_away},index=["Red cards"])
red_combi.plot(kind="bar")
plt.title("Number of Red cards (Home/Away)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
red_combi


# We observe:
# -  Clear increasing pattern where the probability of observing a red card is higher as the minute increases 
# -  Sharp increase at about 80mins (similar to yellow cards)
#   -  Since this plot includes second yellow cards as a red card (as it is still a sending off for the player), this makes sense as the players will get more emotional and the atmosphere would be more tense towards the end of the game.
# - Similar to yellow cards, more red cards are issued to away teams rather than home teams
#   -  Do referee really favor the Home Team or is it just a coincidence?

# # 5. Analysing Penalties

# In[ ]:


penalties=events[events["location"]==14]
penalties.head()


# # 5.1 Placement of penalties

# We will analyse where players aim their penalty kicks and the outcome. We first take note of which "shot_place" that did not coincide with a penalty event and would not include it in our analysis. 

# In[ ]:


for i in range(14):
    if sum(penalties["shot_place"]==i)==0:
        print(i)


# In[ ]:


top_left=sum(penalties["shot_place"]==12)
bot_left=sum(penalties["shot_place"]==3)
top_right=sum(penalties["shot_place"]==13)
bot_right=sum(penalties["shot_place"]==4)
centre=sum(penalties["shot_place"]==5)+sum(penalties["shot_place"]==11)
missed=sum(penalties["shot_place"]==1)+sum(penalties["shot_place"]==6)+sum(penalties["shot_place"]==7)+sum(penalties["shot_place"]==8)+sum(penalties["shot_place"]==9)+sum(penalties["shot_place"]==10)

labels_pen=["top left","bottom left","centre","top right","bottom right","missed"]
num_pen=[top_left,bot_left,centre,top_right,bot_right,missed]
colors_pen=["aqua","royalblue","yellow","violet","m","red"]
plt.pie(num_pen,labels=labels_pen,colors=colors_pen,autopct='%1.1f%%',startangle=60,explode=(0,0,0,0,0,0.2))
plt.axis('equal')
plt.title("Percentage of each placement of penalties",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
fig=plt.gcf()  
fig.set_size_inches(8,6)
plt.show()


# We can see that:
# -  Most penalties has been dispatched to the bottom right/left
# -  About 6.3% of the penalties had been missed 

# # 5.2 Success rate of penalties

# In[ ]:


scored_pen=penalties[penalties["is_goal"]==1]
pen_rightfoot=scored_pen[scored_pen["bodypart"]==1].shape[0]
pen_leftfoot=scored_pen[scored_pen["bodypart"]==2].shape[0]
penalty_combi=pd.DataFrame({"right foot":pen_rightfoot,"left foot":pen_leftfoot},index=["Scored"])
penalty_combi.plot(kind="bar")
plt.title("Penalties scored (Right/Left foot)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
penalty_combi


# In[ ]:


missed_pen=penalties[penalties["is_goal"]==0]
missed_right=missed_pen[missed_pen["bodypart"]==1].shape[0]
missed_left=missed_pen[missed_pen["bodypart"]==2].shape[0]
missed_combi=pd.DataFrame({"right foot":missed_right,"left foot":missed_left},index=["Missed"])
missed_combi.plot(kind="bar")
plt.title("Penalties Missed (Right/Left foot)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
missed_combi


# In[ ]:


total_missed=[missed_right, missed_left]
total_scored=[pen_rightfoot, pen_leftfoot]
plt.bar(penalties["bodypart"].unique(),total_missed,color="black",label="Missed")
plt.bar(penalties["bodypart"].unique(),total_scored,color="red",bottom=total_missed,label="Scored")
plt.xticks(penalties["bodypart"].unique(),["Right foot","Left foot"])
plt.ylabel("Frequency")
plt.legend()
plt.title("Penalties scored/missed among right/left foot",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.show()
total_pen=penalty_combi.append(missed_combi)
total_pen


# In[ ]:


conversion_left_foot=total_pen["left foot"]["Scored"]/(total_pen["left foot"]["Scored"]+total_pen["left foot"]["Missed"])
conversion_right_foot=total_pen["right foot"]["Scored"]/(total_pen["right foot"]["Scored"]+total_pen["right foot"]["Missed"])
print("Left foot conversion rate is: " + str("%.2f" % conversion_left_foot))
print("Right foot conversion rate is: " + str("%.2f" % conversion_right_foot))


# We observe that:
# -  Most penalties are shot using the right foot
# -  However, there is a slightly higher conversion rate for a left-footed penalty (79% compared to 76%)

# # 5.3 Analysing individual players' penalties

# Let's create a function that will show the penalty stats of the specific player. 

# In[ ]:


def pen_stats(player):
    player_pen=penalties[penalties["player"]==player]
    right_attempt=player_pen[player_pen["bodypart"]==1]
    right_attempt_scored=right_attempt[right_attempt["is_goal"]==1].shape[0]
    right_attempt_missed=right_attempt[right_attempt["is_goal"]==0].shape[0]
    left_attempt=player_pen[player_pen["bodypart"]==2]
    left_attempt_scored=left_attempt[left_attempt["is_goal"]==1].shape[0]
    left_attempt_missed=left_attempt[left_attempt["is_goal"]==0].shape[0]
    scored=pd.DataFrame({"right foot":right_attempt_scored,"left foot":left_attempt_scored},index=["Scored"])
    missed=pd.DataFrame({"right foot":right_attempt_missed,"left foot":left_attempt_missed},index=["Missed"])
    combi=scored.append(missed)
    return combi
    


# In[ ]:


penalties[penalties["shot_place"].isnull()].is_goal.unique()


# This shows that the missing values in "shot_placed" all amounts to a goal.

# In[ ]:


pen_stats("eden hazard")


# The following function will give a more detailed stats of the placement of the player's past penalties.

# In[ ]:


def pen_full_stats(player):
    player_pen=penalties[penalties["player"]==player]
    scored_pen=player_pen[player_pen["is_goal"]==1]
    missed_pen=player_pen[player_pen["is_goal"]==0]
    
    top_left_rightfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==1].shape[0]
    top_left_leftfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==2].shape[0]
    bot_left_rightfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==1].shape[0]
    bot_left_leftfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==2].shape[0]
    top_right_rightfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==1].shape[0]
    top_right_leftfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==2].shape[0]
    bot_right_rightfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==1].shape[0]
    bot_right_leftfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==2].shape[0]
    centre_rightfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==1].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==1].shape[0]
    centre_leftfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==2].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==2].shape[0]
    scored_without_recorded_loc_rightfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==1].shape[0]
    scored_without_recorded_loc_leftfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==2].shape[0]
    missed_rightfoot=missed_pen[missed_pen["bodypart"]==1].shape[0]
    missed_leftfoot=missed_pen[missed_pen["bodypart"]==2].shape[0]
    
    right_foot=pd.DataFrame({"Top Left Corner":top_left_rightfoot,"Bottom Left Corner":bot_left_rightfoot,"Top Right Corner":top_right_rightfoot,"Bottom Right Corner":bot_right_rightfoot,"Centre":centre_rightfoot,"Unrecorded placement":scored_without_recorded_loc_rightfoot,"Missed":missed_rightfoot},index=["Right Foot attempt"])
    left_foot=pd.DataFrame({"Top Left Corner":top_left_leftfoot,"Bottom Left Corner":bot_left_leftfoot,"Top Right Corner":top_right_leftfoot,"Bottom Right Corner":bot_right_leftfoot,"Centre":centre_leftfoot,"Unrecorded placement":scored_without_recorded_loc_leftfoot,"Missed":missed_leftfoot},index=["Left Foot attempt"])
    
    fullstats=right_foot.append(left_foot)
    fullstats=fullstats[["Top Right Corner","Bottom Right Corner","Top Left Corner","Bottom Left Corner","Centre","Unrecorded placement","Missed"]]
    return fullstats


# In[ ]:


pen_full_stats("eden hazard")


# Perhaps this is also what goalkeepers use to analyse when preparing for penalties against different players.

# # 6. Analysing Game odds and Results

# In[ ]:


game_info.head()


# In[ ]:


game_info.info()


# # 6.1 Odds

# Odds are interpreted as the amount you will recieve back for every \$1 you bet on that result. For example, if the odds for a home win is 4, you will recieve \$4 for every \$1 you bet on a home win. Thus, from the perspective of the bookmaker, they would set a lower odd for the result they predict. 
# The following function would return the bookmakers' predicted result of a match based on the highest odds for each result for a particular match.

# In[ ]:


def odds_pred_result(odd_h,odd_d,odd_a):
    if odd_h<odd_d and odd_h<odd_a:
        return("Home Win")
    elif odd_d<odd_h and odd_d<odd_a:
        return("Draw")
    elif odd_a<odd_d and odd_a<odd_h:
        return("Away Win")


# The next function would return the actual result of the match.

# In[ ]:


def actual_result(fthg,ftag):
    if fthg>ftag:
        return("Home Win")
    elif fthg==ftag:
        return("Draw")
    elif fthg<ftag:
        return("Away Win")
    
def actual_result_encode(fthg,ftag):
    if fthg>ftag:
        return (1)
    elif fthg==ftag:
        return (2)
    elif fthg<ftag:
        return (3)


# We now compare the chances of us predicting the result of the game correctly just by looking at the odds.

# In[ ]:


def check_pred(data):
    correct=0
    wrong=0
    for i in range(1,data.shape[0]+1):
        odd_h=data[i-1:i]["odd_h"].item()
        odd_d=data[i-1:i]["odd_d"].item()
        odd_a=data[i-1:i]["odd_a"].item()
        fthg=data[i-1:i]["fthg"].item()
        ftag=data[i-1:i]["ftag"].item()
        oddsresult=odds_pred_result(odd_h,odd_d,odd_a)
        actresult=actual_result(fthg,ftag)
        if oddsresult==actresult:
            correct+=1
        else:
            wrong+=1
    return(str("%.2f"%(correct/(correct+wrong)))+str("% correct"))


# In[ ]:


check_pred(game_info)


# This shows that if we just guess the result based on looking at the odds, we would only be right about half the time. Clearly, we want to have a better indicator that would give us a better chance than this if we want to bet on the match.

# # 6.2 Predicting results

# We will try to use the odds and the difference among the odds to predict the result of a match.

# In[ ]:


x_var=game_info.iloc[:,9:14]
x_var.head()


# In[ ]:


x_var=game_info.iloc[:,9:14]
result=[]
for i in range(1,game_info.shape[0]+1):
    result.append(actual_result_encode(game_info[i-1:i]["fthg"].item(),game_info[i-1:i]["ftag"].item()))
y=pd.DataFrame(result)
x_var["diff_h_d"]=abs(x_var["odd_h"]-x_var["odd_d"])
x_var["diff_d_a"]=abs(x_var["odd_d"]-x_var["odd_a"])
x_var["diff_h_a"]=abs(x_var["odd_h"]-x_var["odd_a"])
x_var=x_var.drop(["fthg","ftag"],axis=1)
x_var.tail()


# **Spliting the data into training and test set, and applying cross validation**

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x_var,y,test_size=0.2,random_state=0)


# In[ ]:


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# We create a function to access the accuracy of our models.

# In[ ]:


def acc_score(model):
    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))


# In[ ]:


from sklearn.metrics import accuracy_score
def norm_score(model):
    return (accuracy_score(y_train,model.predict(x_train)))


# We then create another function to print for us the confusion matrix for each model.

# In[ ]:


from sklearn.metrics import confusion_matrix
def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_train,model_used.predict(x_train))
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Home Win","Predicted Draw","Predicted Away Win"]
    cm.index=["Actual Home Win","Actual Draw","Actual Away Win"]
    return cm


# **6.2.1 Logistic Regression**

# In[ ]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
print("Logistic Regression Accuracy (without cross validation): "+str(norm_score(log_reg)))
print("Logistic Regression Cross Validation Accuracy: "+str(acc_score(log_reg)))
confusion_matrix_model(log_reg)


# **6.2.2 Decision Tree Regression**

# In[ ]:


Dec_tree=DecisionTreeClassifier()
Dec_tree.fit(x_train,y_train)
print("Decision Tree Regression Accuracy (without cross validation): "+str(norm_score(Dec_tree)))
print("Decision Tree Regression Cross Validation Accuracy: "+str(acc_score(Dec_tree)))
confusion_matrix_model(Dec_tree)


# **6.2.3 K-Nearest Neighbour Regression**

# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=20)
KNN.fit(x_train,y_train)
print("KNN Regression Accuracy (without cross validation): "+str(norm_score(KNN)))
print("KNN Regression Cross Validation Accuracy: "+str(acc_score(KNN)))
confusion_matrix_model(KNN)


# **6.2.4 Support Vector Machine**

# In[ ]:


svm_reg=svm.SVC()
svm_reg.fit(x_train,y_train)
print("SVM Regression Accuracy (without cross validation): "+str(norm_score(svm_reg)))
print("SVM Regression Cross Validation Accuracy: "+str(acc_score(svm_reg)))
confusion_matrix_model(KNN)


# **6.2.5 Multinomial Naive Bayes**

# In[ ]:


naive_bayes=MultinomialNB()
naive_bayes.fit(x_train,y_train)
print("Multinomial Naive Bayes Regression Accuracy (without cross validation): "+str(norm_score(naive_bayes)))
print("Multinomial Naive Bayes Regression Cross Validation Accuracy: "+str(acc_score(naive_bayes)))
confusion_matrix_model(naive_bayes)


# Sub beautify()
#     
#     'create new sheet
#     Sheets.Add.Name = "Analysis"
#     
#     'copy specific rows from Sheet1 to analysis worksheet
#     Sheets("Sheet1").Columns(1).Copy Destination:=Sheets("Analysis").Range("A1")
#     
#     'combine entity into one column
#     Sheets("Sheet1").Select
#     Dim lRow As Long, i As Long
#     Dim rng As Range
#     Set rng = Range("B" & Rows.Count).End(xlUp)
#     lRow = rng.Row
#     For i = 2 To lRow
#         ActiveWorkbook.Sheets("Analysis").Cells(i, 2) = Cells(i, 2) & Cells(i, 3)
#     Next i
#     Sheets("Analysis").Cells(1, 2) = "Entity"
#     
#     'Combine the different issues and choices
#     'Assume issues has only 5 choices for A and B (columns(D-M))
#     Dim x As Long
#     Dim j As Long
#     
#     For i = 2 To lRow
#         x = 3
#         For j = 4 To 13  'cols D-M
#             If IsEmpty(Cells(i, j).Value) = False Then
#                 ActiveWorkbook.Sheets("Analysis").Cells(i, x) = Mid(Cells(1, j), 7, 1) & Mid(Cells(i, j), 1, 1)
#                 x = x + 3
#             End If
#         Next j
#     Next i
#     
#     'Add the dates and Remarks
#     'Assume dates and remarks columns are from N to AG
#     For i = 2 To lRow
#         x = 4
#         For j = 14 To 34 Step 2 'cols N to AG, step 2 coz only need to check dates
#             If IsEmpty(Cells(i, j).Value) = False Then
#                 ActiveWorkbook.Sheets("Analysis").Cells(i, x) = Cells(i, j).Value
#                 ActiveWorkbook.Sheets("Analysis").Cells(i, x + 1) = Cells(i, j + 1)
#                 x = x + 3
#             End If
#         Next j
#     Next i
#     
#     'Add column names
#     For i = 3 To 33 Step 3
#         If Application.WorksheetFunction.CountA(Worksheets("Analysis").Columns(i)) > 0 Then
#             ActiveWorkbook.Sheets("Analysis").Cells(1, i) = "Chosen"
#             ActiveWorkbook.Sheets("Analysis").Cells(1, i + 1) = "Date"
#             ActiveWorkbook.Sheets("Analysis").Cells(1, i + 2) = "Remarks"
#         End If
#     Next i
#     
# 
#     Sheets.Add.Name = "Neat"
#     
#     'column names
#     ActiveWorkbook.Sheets("Neat").Cells(1, 1) = "Name"
#     ActiveWorkbook.Sheets("Neat").Cells(1, 2) = "Entity"
#     ActiveWorkbook.Sheets("Neat").Cells(1, 3) = "Chosen"
#     ActiveWorkbook.Sheets("Neat").Cells(1, 4) = "Date"
#     ActiveWorkbook.Sheets("Neat").Cells(1, 5) = "Remarks"
#     
#     'format remarks/dates/choice
#     Sheets("Analysis").Select
#     Set rng = Range("A" & Rows.Count).End(xlUp)
#     lRow = rng.Row
#     a = 2
#     
#     For i = 2 To lRow
#         b = 3
#         j = 3
#         
#         Do While IsEmpty(Sheets("Analysis").Cells(i, j)) = False
#             ActiveWorkbook.Sheets("Neat").Cells(a, b) = Sheets("Analysis").Cells(i, j).Value
#             ActiveWorkbook.Sheets("Neat").Cells(a, b + 1) = Sheets("Analysis").Cells(i, j + 1).Value
#             ActiveWorkbook.Sheets("Neat").Cells(a, b + 2) = Sheets("Analysis").Cells(i, j + 2).Value
#             ActiveWorkbook.Sheets("Neat").Cells(a, 1) = Sheets("Analysis").Cells(i, 1)
#             ActiveWorkbook.Sheets("Neat").Cells(a, 2) = Left(ActiveWorkbook.Sheets("Neat").Cells(a, 3), 1)
#             a = a + 1
#             j = j + 3
#         Loop
#     Next i
# End Sub

# Although the decision tree regression result without cross-validation is 97%, it is not accurate as cross-validation would yield a more accurate accuracy result. 
# 
# We can see that these models yield about the same (or even worse) accuracy than by just guessing based on the odds (53% accuracy). So, obviously we need better predictors other than the odds. 
# 
# The next notebook would try to predict the result using different predictors.
