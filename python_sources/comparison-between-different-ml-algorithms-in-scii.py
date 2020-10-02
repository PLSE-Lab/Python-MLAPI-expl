#!/usr/bin/env python
# coding: utf-8

# This project is still in development. New additions will be added sequentially.

# > ## Introduction
# 

# -A Generic Machine Learning Research On Star Craft II.
# -At the end of the research, expected outcome will be a conclusion of different approaches in league prediction.

# ## Required Packages
# 

# In[ ]:


import numpy as np
import pandas as pd
##################################################
###################Classsifiers###################
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
###################################################
################Processing and EDA#################
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
##################################################
##############Disable Warnings####################
import warnings  
warnings.filterwarnings('ignore')
##################################################
import os


# **Brief information about columns for making easy to work with**
# 

# GameID: Unique ID for each game
# 
# LeagueIndex: 1-8 for Bronze, Silver, Gold, Diamond, Master, GrandMaster, Professional leagues
# 
# Age: Age of each player
# 
# HoursPerWeek: Hours spent playing per week
# 
# TotalHours: Total hours spent playing
# 
# APM: Action per minute
# 
# SelectByHotkeys: Number of unit selections made using hotkeys per timestamp
# 
# AssignToHotkeys: Number of units assigned to hotkeys per timestamp
# 
# UniqueHotkeys: Number of unique hotkeys used per timestamp
# 
# MinimapAttacks: Number of attack actions on minimal per timestamp
# 
# MinimapRightClicks: Number of right-clicks on minimal per timestamp
# 
# NumberOfPACs: Number of PACs per timestamp
# 
# GapBetweenPACs: Mean duration between PACs (milliseconds)
# 
# ActionLatency: Mean latency from the onset of PACs to their first action (milliseconds)
# 
# ActionsInPAC: Mean number of actions within each PAC
# 
# TotalMapExplored: Number of 24x24 game coordinate grids viewed by player per timestamp
# 
# WorkersMade: Number of SCVs, drones, probes trained per timestamp
# 
# UniqueUnitsMade: Unique units made per timestamp
# 
# ComplexUnitsMade: Number of ghosts, investors, and high templars trained per timestamp
# 
# ComplexAbilityUsed: Abilities requiring specific targeting instructions used per timestamp
# 
# MaxTimeStamp: Time stamp of game's last recorded event

# In[ ]:


#Total playing time is not chosen due to a player can give a long break to 
#his/her playing career, then they can start playing again
relatedColumnsList=["LeagueIndex","Age","HoursPerWeek","APM","SelectByHotkeys",
                    "AssignToHotkeys","UniqueHotkeys",
                    "MinimapAttacks","MinimapRightClicks","NumberOfPACs",
                    "GapBetweenPACs","ActionLatency",
                    "ActionsInPAC","TotalMapExplored","WorkersMade","UniqueUnitsMade",
                    "ComplexUnitsMade","ComplexAbilityUsed","MaxTimeStamp"]
df=pd.read_csv('../input/starcraft.csv')
df = df[relatedColumnsList]
df.dropna()
df = df[df['LeagueIndex']!=8]
df.head(n=6)


# In[ ]:


leagueOne=df[df['LeagueIndex']==1]['Age']
leagueTwo=df[df['LeagueIndex']==2]['Age']
leagueThree=df[df['LeagueIndex']==3]['Age']
leagueFour=df[df['LeagueIndex']==4]['Age']
leagueFive=df[df['LeagueIndex']==5]['Age']
leagueSix=df[df['LeagueIndex']==6]['Age']
leagueSeven=df[df['LeagueIndex']==7]['Age']


# In[ ]:


print("Percentages of leagues in total data:\n")
df['LeagueIndex'].value_counts(normalize=True)*100


# In[ ]:


plt.style.use('ggplot')
fig, ax = plt.subplots(1, figsize = (14,8))
fig.suptitle('League-Player Percentage', fontweight='bold', fontsize = 22,ha='center')
bins = np.arange(0, 9, 1)
weights = np.ones_like(df['LeagueIndex']) / len(df['LeagueIndex'])
p2 = plt.subplot(1,2,2)
p2.hist(df['LeagueIndex'], bins=bins, weights = weights, align='left')
plt.xlabel('League Index', fontweight='bold')
plt.title('Percentage',loc='left')
yvals = plt.subplot(1,2,2).get_yticks()
plt.subplot(1,2,2).set_yticklabels(['{:3.1f}%'.format(y*100) for y in yvals])
plt.show()


# In[ ]:


leagues=[leagueOne,leagueTwo,leagueThree,leagueFour,
         leagueFive,leagueSix,leagueSeven]
newLabels=["Bronze", "Silver", "Gold", "Platinum",
           "Diamond", "Master", "Grandmaster"]


# In[ ]:


fig=plt.figure(figsize=(25,15))
plt.title("Player League Number - Age Distribution")
for i in range(len(leagues)):
    leagues[i].hist(alpha=0.9,bins=60,label=newLabels[i])
    plt.legend(loc="best")


# In[ ]:


leagueOne=df[df['LeagueIndex']==1]['APM']
leagueTwo=df[df['LeagueIndex']==2]['APM']
leagueThree=df[df['LeagueIndex']==3]['APM']
leagueFour=df[df['LeagueIndex']==4]['APM']
leagueFive=df[df['LeagueIndex']==5]['APM']
leagueSix=df[df['LeagueIndex']==6]['APM']
leagueSeven=df[df['LeagueIndex']==7]['APM']


# In[ ]:


leagues=[leagueOne,leagueTwo,leagueThree,leagueFour,
         leagueFive,leagueSix,leagueSeven]


# In[ ]:


fig=plt.figure(figsize=(25,15))
plt.title("Player League Number - Action Per Minute Distribution")
for i in range(len(leagues)):
    leagues[i].hist(alpha=0.8,bins=80,label=newLabels[i])
    plt.legend(loc="best")


# In[ ]:


plt.style.use(['seaborn-dark'])
fig, axes = plt.subplots(nrows=1, ncols = 1, figsize = (15,8))
fig.suptitle('Attribute Relationships', fontsize=25, fontweight='bold')
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
r_matrix = df.corr().round(decimals=1)
sns.heatmap(r_matrix, mask=mask, annot=True, fmt='g',
            annot_kws={'size':10},linewidths=.3,cmap='coolwarm')
plt.show()


# In[ ]:


willBeFocusedColumns = ['APM','SelectByHotkeys', 'AssignToHotkeys',
                        'NumberOfPACs','GapBetweenPACs', 'ActionLatency']


# In[ ]:


ySelected = df['LeagueIndex']
xSelected = df[willBeFocusedColumns]


# In[ ]:


xTrain,xTest,yTrain,yTest=train_test_split(xSelected,ySelected,test_size=0.33)


# In[ ]:


dtc= DecisionTreeClassifier()
dtc.fit(xTrain,yTrain)
yPrediction=dtc.predict(xTest)
print("Decision Tree Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Decision Tree: ",dtc.score(xTest,yTest),"\n")


# In[ ]:


#import graphviz 
#from sklearn import tree
#dot_data = tree.export_graphviz(dtc, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("df") 
#graph


# In[ ]:


gnb=GaussianNB()
gnb.fit(xTrain,yTrain)
yPrediction=gnb.predict(xTest)
print("Naive Bayes Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Naive Bayes: ",gnb.score(xTest,yTest),"\n")


# In[ ]:


svmTrial = svm.LinearSVC()
svmTrial = svmTrial.fit(xTrain,yTrain)
yPrediction = svmTrial.predict(xTest)
print("Support Vector Machine Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Support Vector Machine: ",svmTrial.score(xTest,yTest),"\n")


# In[ ]:


rfc=RandomForestClassifier()
rfc.fit(xTrain,yTrain)
yPrediction=rfc.predict(xTest)
print("Random Forest Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Random Forest: ",rfc.score(xTest,yTest),"\n")


# ###########################################################################
# ######################## ONE VERSUS ALL CLASSIFICATION ####################
# ###########################################################################

# In[ ]:


def CalculateOneVsAll(targetRank):
    oneVsAllDataFrame = df.copy(deep=True)
    leagueIndexes=[1,2,3,4,5,6,7]
    if targetRank in leagueIndexes:
        leagueIndexes[targetRank-1]=0
    for i in range(len(oneVsAllDataFrame.index)):  
        if(oneVsAllDataFrame['LeagueIndex'][i]!=targetRank):
            for k in range(len(leagueIndexes)):
                oneVsAllDataFrame['LeagueIndex'].replace(leagueIndexes[k],0,inplace=True)
    
    OVAWillBeFocusedColumns = ['APM','SelectByHotkeys', 'AssignToHotkeys',
                            'NumberOfPACs','GapBetweenPACs', 'ActionLatency']
    yOVASelected = oneVsAllDataFrame['LeagueIndex']
    xOVASelected = oneVsAllDataFrame[OVAWillBeFocusedColumns]        
            
    xOVATrain,xOVATest,yOVATrain,yOVATest=train_test_split(xOVASelected,
                                                           yOVASelected,
                                                           test_size=0.33)  

    ######################DecisionTreeClassifier One vs All####################
    dtcOVA=DecisionTreeClassifier()
    dtcOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=dtcOVA.predict(xOVATest)
    print("One Versus All Decision Tree Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Decision Tree: ",dtcOVA.score(xOVATest,yOVATest),"\n")
    f1ScoreCalculation(yOVATest,yOVAPrediction)
    precisionScoreCalculation(yOVATest,yOVAPrediction)
    recallScoreCalculation(yOVATest,yOVAPrediction)
    ######################Naive Bayes Classifier One vs All####################
    gnbOVA=GaussianNB()
    gnbOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=gnbOVA.predict(xOVATest)
    print("One Versus All Naive Bayes Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Naive Bayes Classifier: ",gnbOVA.score(xOVATest,yOVATest),"\n")
    f1ScoreCalculation(yOVATest,yOVAPrediction)
    precisionScoreCalculation(yOVATest,yOVAPrediction)
    recallScoreCalculation(yOVATest,yOVAPrediction)
    ######################Support Vector Machine One vs All####################
    svmOVA = svm.LinearSVC()
    svmOVA = svmOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction = svmOVA.predict(xOVATest)
    print("One Versus All Support Vector Machine Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Support Vector Machine: ",svmOVA.score(xOVATest,yOVATest),
          "\n")
    f1ScoreCalculation(yOVATest,yOVAPrediction)
    precisionScoreCalculation(yOVATest,yOVAPrediction)
    recallScoreCalculation(yOVATest,yOVAPrediction)
    #########################RandomForestClassifier One vs All#################
    rfcOVA=RandomForestClassifier()
    rfcOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=rfcOVA.predict(xOVATest)
    print("One Versus All Random Forest Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of One Versus All Random Forest: ",rfcOVA.score(xOVATest,yOVATest),
          "\n")
    f1ScoreCalculation(yOVATest,yOVAPrediction)
    precisionScoreCalculation(yOVATest,yOVAPrediction)
    recallScoreCalculation(yOVATest,yOVAPrediction)


# In[ ]:


def f1ScoreCalculation(test,prediction):
    print("Macro F1: ",f1_score(test,prediction,average='macro'))
    print("Micro F1: ",f1_score(test,prediction,average='micro'))
    print("Weighted F1: ",f1_score(test,prediction,average='weighted'))
    print(f1_score(test,prediction,average=None),"\n")


# #### The F-measure can be interpreted as a weighted harmonic mean of the precision and recall. A measure reaches its best value at 1 and its worst score at 0.

# In[ ]:


def precisionScoreCalculation(test,prediction):
    print("Macro Precision: ",precision_score(test,prediction,average='macro'))
    print("Micro Precision: ",precision_score(test,prediction,average='micro'))
    print("Weighted Precision: ",precision_score(test,prediction,average='weighted'))
    print(precision_score(test,prediction,average=None),"\n")


# #### Precision is defined as the number of true positives over the number of true positives plus the number of false positives.

# In[ ]:


def recallScoreCalculation(test,prediction):
    print("Macro Recall: ",recall_score(test,prediction,average='macro'))
    print("Micro Recall: ",recall_score(test,prediction,average='micro'))
    print("Weighted Recall: ",recall_score(test,prediction,average='weighted'))
    print(recall_score(test,prediction,average=None),"\n")


# #### Recall is defined as the number of true positives over the number of true positives plus the number of false negatives.

# In[ ]:


notifier=["-----ONE VERSUS ALL FOR BRONZE LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR SILVER LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR GOLD LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR PLATINIUM LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR DIAMOND LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR MASTER LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR GRAND MASTER LEAGUE PREDICTIONS-----\n"]
endNotifier=["++++ End of Bronze league predictions++++\n",
          "++++ End of Silver league predictions++++\n",
          "++++ End of Gold league predictions++++\n",
          "++++ End of Platinium league predictions++++\n",
          "++++ End of Diamond league predictions++++\n",
          "++++ End of Master league predictions++++\n",
          "++++ End of Grand Master league predictions++++\n"]
j=1        
while j<=7:
    print(notifier[j-1])
    CalculateOneVsAll(j)
    print(endNotifier[j-1])
    j+=1

