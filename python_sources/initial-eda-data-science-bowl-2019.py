#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time


# In[ ]:


import matplotlib.pyplot as plt
import os


# In[ ]:


os.getcwd()


# In[ ]:


os.listdir()


# # EDA

# ## train.csv

# In[ ]:


TrainData=pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')


# In[ ]:


TrainData.shape


# In[ ]:


TrainData.keys()


# In[ ]:


TrainData.dtypes


# In[ ]:


TrainData.head(5)


# In[ ]:


TrainData.describe(include='all')


# * Q: what does event_count represent?
# 
#    let us look at how many unique values are there in this column

# In[ ]:


TrainData['event_count'].unique()


# In[ ]:


len(TrainData['event_count'].unique())


# In[ ]:


min(TrainData['event_count'].unique())


# In[ ]:


max(TrainData['event_count'].unique())


# There are 3368 unique values for event_count. These seem to be all the integers from 1 to 3368.
# 
# Let us look at the distribution various unique value under the column event_count

# In[ ]:


# frequency of each unique value under event_count (sorted)
TrainData.event_count.value_counts().sort_index()


# In[ ]:


# the event_count values which only occur once
event_count_freq=TrainData.event_count.value_counts().sort_index()
event_count_freq.loc[event_count_freq==1]


# In[ ]:


# bar-chart of frequencies of the first the first 100 unique values under event_count
# given that there are 3368 unique values, it is impossible to draw a single bar-chart with 
# all the unique values
plt.figure(figsize=(20,20))
TrainData['event_count'].value_counts()[0:100].plot(kind='bar')
plt.xlabel('event_count')
plt.ylabel('value_counts')
plt.title('distribution of values under the column event_count')
plt.show()


# In[ ]:


# all the 3368 unique values under event_count can't be put in the same bar-chart,
# therefore, it is best to use collect them into bins and look at these bins.
# We therefore draw a histogram with 300 bins for 'event_count' 
event_count_hist=TrainData.hist(column='event_count', figsize=(20,20), bins=300)


# The frequency of occurence for various unique values under event_count seems to rapidly decay with most frequently occuring values lying largely between 1 and 500.  '1' occurs extremely frequently while larger values such as '3183' onwards seem to only occur once. 

# * what does the column 'type' represent?
# 
#  To begin with, we note that 'type' takes 4 unique values: 'Clip', 'Activity', 'Game', 'Assessment'

# In[ ]:


TrainData.type.unique()


# This suggest it is a categorical label. The meaning of various types has been explained in [this](https://www.kaggle.com/c/data-science-bowl-2019/discussion/115034#latest-664872) kaggle post.

# In[ ]:


# frequency of the distinct 'type' values
plt.figure()
TrainData.type.value_counts().plot(kind='bar')
plt.xlabel('type')
plt.ylabel('number of instances')
plt.title('frequency of the distinct values under the column type ')
plt.show()


# Note that though the data has far less number of entries with 'type'='Clip', as suggested in the aforementioned Kaggle post, this is an artifact of how the data was collected and does NOT imply that clips are less popular!

#  Let us look at the entries which have 'type' = 'Clip'

# In[ ]:


TrainDataTpClip=TrainData.loc[TrainData['type']=='Clip']


# As suggested in the above mentioned Kaggle post, 'Clips' are introductory videos to the different possible environments in the PBS KIDS Measure Up! app. Different videos can be further identified by their title. So let's check out the various possible titles for these videos. This information seems to be provided under the column 'title'.  

# In[ ]:


TrainDataTpClip.title.unique()


# In[ ]:


# number of distinct titles for 'Clip'
len(TrainDataTpClip.title.unique())


# We therefore see that there are 20 different 'clips'

# In[ ]:


# confirm if there are any missing titles for 'Clip' 
TrainDataTpClip.title.isnull().values.any()


# Let us now look at the entries that have 'type'='Activity' 

# In[ ]:


TrainDataTpAct=TrainData.loc[TrainData['type']=='Activity']


# Once again, let us look at the various different 'titles' that 'Activity' can have. 

# In[ ]:


TrainDataTpAct.title.unique()


# In[ ]:


# number of different Activity-titles 
len(TrainDataTpAct.title.unique())


# In[ ]:


# are there missing entries under Activity-titles?
TrainDataTpAct.title.isnull().values.any()


# Thus we see that there are 8 different activities possible. As explained in the kaggle [post](https://www.kaggle.com/c/data-science-bowl-2019/discussion/115034#latest-664872), activity has no predefined goal and is meant to allow the kids to get acquianted with the environment.   

# let us now checkout the entries with 'type'='Game'

# In[ ]:


TrainDataTpGame=TrainData.loc[TrainData.type=='Game']


# In[ ]:


# different titles for 'type'='Game'
TrainDataTpGame.title.unique()


# In[ ]:


# number of different titles for 'type' = 'Game'
len(TrainDataTpGame.title.unique())


# In[ ]:


# entries with 'type'=Assesment
TrainDataTpAssess=TrainData.loc[TrainData.type=='Assessment']


# In[ ]:


# unique titles for 'Assessment'
TrainDataTpAssess.title.unique()


# In[ ]:


# no. of unique titles for 'Assessment'
len(TrainDataTpAssess.title.unique())


# We see that there are 5 different assessments!

# * what does the column 'world' represent?

# The information under the 'world' column seems to be categorical. Let's look at the different possible values that can be assigned to it. 

# In[ ]:


TrainData.world.unique()


# By looking at [this](https://measureup.pbskids.org/) web-version of the app, we see that there are three different environments that the app provides. These are: 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES'. This matches exactly with the values that can be assigned to 'world'. 
# 
# However, 'world' also has 'None' as one of it's values. This happens if and only if the title is 'Welcome to Lost Lagoon!'.

# In[ ]:


TrainData.loc[TrainData.world=='NONE'].title.unique()


# In[ ]:


TrainData.loc[TrainData.title=='Welcome to Lost Lagoon!'].world.unique()


# Since 'Welcome to Lost Lagoon!' is the introductory video in the app, it does not correspond to any of the environment. This explains why it corresponding entry under the 'world' column is 'NONE'.

# In[ ]:


# total number of assessment sessions in train.csv
num_assess_tr=len(TrainData.loc[TrainData.type=='Assessment'].game_session.unique())
print('train.csv contains {} assessment sessions'.format(num_assess_tr))


# In[ ]:


# total number of assessments that were completed 
# i.e. assessment where a solution was submitted resulting in an event with code 4100 or 4110
completed_assessments=len(TrainData.loc[(TrainData.type=='Assessment') 
                                        & ((TrainData.event_code==4100) | (TrainData.event_code==4110) )
                                       ].game_session.unique())
print('train.csv contains {} completed assessments'.format(completed_assessments))


# ## train_labels.csv

# In[ ]:


TrainLabels=pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')


# In[ ]:


TrainLabels.keys()


# In[ ]:


TrainLabels.dtypes


# In[ ]:


TrainLabels.head()


# In[ ]:


TrainLabels.describe(include='all')


# let us look at the different unique values for the column 'title' in this dataset.

# In[ ]:


TrainLabels.title.unique()


# From looking at the different unique values under the column 'title' in this dataset, we find that it always an assessment. Thus we conclude that this dataset contains assesment results. Having established this, it is straightforward to realize that the columns 'num_correct' and 'num_incorrect' represent the number of correct and incorrect answers leading to the assesment solution. 

# We conjecture that the column 'accuracy' simply represents the ratio of correct answers to the total number of attempts. In order to check that this is indeed the case, let's contruct another column with exactly this information and it's entries match with those under 'accuracy'

# In[ ]:


TrainLabels['ratio_of_correct']=TrainLabels['num_correct']/(
    TrainLabels['num_correct']+TrainLabels['num_incorrect'])


# To avoid false mismatch due to difference in precision, we will assume that ratio_of_correct is same as accuracy if the difference between them is less than 10**(-16)

# In[ ]:


(TrainLabels['ratio_of_correct']-TrainLabels['accuracy']<(10**(-16))).values.all()


# We thus see that 'accuracy' is indeed given by the ratio of correct answers to the total number of attempts.

# In[ ]:


# Delete ratio_of_correct as it is same as accuracy
TrainLabels.drop(['ratio_of_correct'], axis=1, inplace=True)


# In[ ]:


TrainLabels.keys()


# As the name suggest 'accuracy_group' is the accuracy_group assigned as a result of the assessment. This is our Target Variable.

# In[ ]:


# checking if num_correct was ever greater than 1
(TrainLabels.num_correct>1).values.any()


# In[ ]:


TrainLabels.num_correct.unique()


# As expected, num_correct is only takes the values 1 or 0

# In[ ]:


len(TrainData.installation_id.unique())


# In[ ]:


len(TrainLabels.installation_id.unique())


# Note that it is mentioned on the Data description page of the competition that the "training set contains many installation_ids which never took assessments". Thus it is no suprize that train.csv contains data about 17000 unique installation_ids but train_labels.csv only contains the assesment of a mere 3614 installation_ids. 

# Also, note that that it appears that the same installation_id could have taken the same assessment multiple times i.e. same assessment with multiple game sessions. For e.g. let's look at installation_id=0006a69f and assessment titled 'Mushroom Sorter (Assessment)'

# In[ ]:


TrainLabels.loc[(TrainLabels.installation_id=='0006a69f') 
                & (TrainLabels.title=='Mushroom Sorter (Assessment)')]


# We see that there are 3 game sessions associated to the installation_id '0006a69f' where they took the same assessment.  This actually gives us and idea: We can check how the scores of a particular installation_id improve as they take the same assessment again and again. For this we will need the time_stamp of each game session. 

# ## specs.csv

# In[ ]:


Specs=pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')


# In[ ]:


Specs.keys()


# In[ ]:


Specs.head()


# In[ ]:


Specs.describe()


# In[ ]:


Specs.iloc[0].args


# In[ ]:


Specs.iloc[0].info


# It seems like, Specs contains the description of various event_id's.

# # Further EDA

# To understand the meaning of various columns further let's merge TrainData, Specs and TrainLabels. 
# 
# Since Specs seems to contain the description of different event_ids, we will merge this with TrainData on event_id. 
# 
# Similarly, since TrainLabels contains the assessment information for each installation_id, we will merge this with TrainData on installation_id and the title of the corresponding assessment. Since, as we saw earlier in the EDA for train_labels.csv, the same installation_id at times takes the same assessment in multiple game sessions, we will also use 'game_session' for merging TrainData and TrainLabels. 

# ### Merging TrainData with Specs and TrainLabels

# In[ ]:


TrainData.shape


# In[ ]:


TrainData=TrainData.merge(Specs, how='left', on='event_id')


# In[ ]:


TrainData.shape


# In[ ]:


TrainData=TrainData.merge(TrainLabels, how='left', 
                          on=['installation_id', 'title', 'game_session'])


# In[ ]:


TrainData.shape


# In[ ]:


TrainData.keys()


# __In the following I will try to understand what does event_code represent__

# Let us look at all possible unique values of event_code

# In[ ]:


TrainData.event_code.unique()


# In[ ]:


len(TrainData.event_code.unique())


# We see that there are merely 42 unique event_codes.

# It is mentioned in the Data [section](https://www.kaggle.com/c/data-science-bowl-2019/data) of the competition that event_code 2000 always represents the start of a new game session. Let's understand this a little further.

# In[ ]:


TrDatEvtCd2000=TrainData.loc[TrainData.event_code==2000]


# To begin with, let's check that every game_session indeed has one and only one event code with event_code =2000 (since there will be exactly one event marking the start of the game). 

# If indeed this is true, then the number of entries with event_code 2000 should be equal to the number of unique game_sessions in the whole data

# In[ ]:


TrDatEvtCd2000.shape


# In[ ]:


len(TrainData.game_session.unique())


# In[ ]:


# checking that the number of entries with event_code 2000 is same as the number of unique game sessions
len(TrainData.game_session.unique())-TrDatEvtCd2000.shape[0]


# If event 2000 indeed represents the start of each game_session, then the list of unique sessions in TrDatEvtCd2000 should match with the list of unique sessions in TrainData

# In[ ]:


# checking that the list of unique sessions in TrDatEvtCd2000 matches exactly with that of unique sessions in TrainData
(TrDatEvtCd2000.game_session.unique()==TrainData.game_session.unique()).all()


# In[ ]:


TrDatEvtCd2000.head()


# By looking at the entries under the columns 'type' and 'title' we notice that each individual Clip, Activity, Game or Assessment counts as an individual game_session. 

# Now, let's look at at one particular person's (i.e. installation_id) one game session and try to understand what happened during that session. 
# 
# For no particular reason, let's choose the installation_id to be '0006a69f' and the game_session to be '901acc108f55a5a1'. 

# In[ ]:


TrDtIdSess=TrainData.loc[(TrainData.installation_id=='0006a69f')
              & (TrainData.game_session=='901acc108f55a5a1')]


# In[ ]:


# Obtaining the title of the game_session above
# Recall, that we had previously established that each game_session is associated 
# with an individual clip, activity, game or assessment. Thus it should have a unique title which
# can be obtained from the title of the very first entry
TrDtIdSess.title.iloc[0]


# In[ ]:


# just to make sure that indeed there is one and only one title associated 
# with the game_session above, let's check how many unique values are there under
# the column title. If we are correct, 
# then there should be only one i.e. 'Mushroom Sorter (Assessment)'
TrDtIdSess.title.unique()


# At this point we found it most useful to actually take the 'Mushroom Sorter (Assessment)' through [this](https://measureup.pbskids.org/) web-browser version of the app. We then compared what was happening in the assessment to the various events, that were recorded in the above game_session

# - Following is a brief summary of what happens in the 'Mushroom Sorter (Assessment)': 
# 
# We are instructed to "pull 3 mushrooms out of the ground and arrange them in increasing order of their height". -> Then we are told that "to pick a mushroom, pull it out of the ground with your finger" -> We pick out the first mushroom, at which point the computer exclaims "That's one!" -> We pick the second mushroom, and the computer exclaims "two!" -> For the third mushroom the computer exclaims "and three!" -> We are told "now order these mushrooms by height" -> We put the mushrooms in order and place them on the respective stumps. -> computer "Ok! When you want to check your answers,  tap here" and highlights the relevant button -> upon tapping on that button, we are given an aprropriate feedback: if wrong order then "hmm, that doesn't seem to be the right order", if the order is correct then "that's right! this one is the littlest mushroom and this one is the tallest (highlights the smallest and the tallest mushrooms respectively)".

# Now let us look at the data provided in the above game_session. In particular we find it is best to look at the event_data along with event_info for each event_code. We will also look at the event_count 

# In[ ]:


TrDtIdSess[['event_count', 'event_code', 'event_data', 'info']]


# The first thing we notice is that each event seems to be assigned a different event_count and infact the event_count seems to correspond to the order in which the events occurr. This clarifies the note in the Data section of the competition where they say that event_count is the "Incremental counter of events within a game session (offset at 1)". 

# Next if we consider the entries under 'info' and 'event_data' for the event_count=3, we find the following:

# In[ ]:


# info
TrDtIdSess.loc[TrDtIdSess.event_count==3]['info']


# The info tells us that this was a "system-initiated instruction event"

# In[ ]:


# event_data 
TrDtIdSess.loc[TrDtIdSess.event_count==3]['event_data']


# The event_data gives us a "description" of the instruction i.e. ""Pull three mushrooms out of th..." which exactly the instruction we got at the begining of the 'Mushroom Sorter (Assessment)'

# Similarly, from the info and event_data entries we see that event with event_count=4 marks the end of the the above system initiated instruction. 

# By looking at the event codes of the above two cases and also for those with event_counts=5, 7 respectively, we find that event_code 3010 is associated with the start of a system-initiated instruction and event_code 3110 is associated with the end of that system-initiated instruction. (each system-initiated instruction has an event marking its begining and another even marking its end)

# To further understand the entries under event_data and info, it also helps to look at the event with event_count=6.

# In[ ]:


# info
TrDtIdSess.loc[TrDtIdSess.event_count==6]['info']


# It looks like this event is associated with the player picking up one of the mushrooms. 

# In[ ]:


# info
TrDtIdSess.loc[TrDtIdSess.event_count==6]['event_data']


# This seems to contain the height and the (x,y) coordinates of the mushroom that was picked. 

# Note that it is mentioned that assesment attempts are assigned an event_code=4100 (except for Bird Measurer, which uses event_code 4110). Let us look at the corresponding event for the case at hand.  

# In[ ]:


# info 
TrDtIdSess.loc[TrDtIdSess.event_code==4100]['info']


# In[ ]:


# data 
TrDtIdSess.loc[TrDtIdSess.event_code==4100]['event_data']


# From the event_data for event with code 4100 we see that the player was successful. This is indicated by the phrase "correct":true in the event_data

# There are events with an event_count>100. This is a little curious. Let's look at some of these sessions. 

# In[ ]:


# game_session corresponding to events with event_count>100
EvCTLargeSess=TrainData.loc[TrainData.event_count>100]
GamesLargeEvCT=EvCTLargeSess.game_session.unique()

# number of game_sessions containing event_count>100
print("There are {} sessions where the event_count exceeds 100".format(len(GamesLargeEvCT)))


# Let's look at some of these sessions. 

# In[ ]:


session=GamesLargeEvCT[0]


# In[ ]:


TrainData.loc[TrainData.game_session==session]


# Turns out this was an 'activity'. It explains why it can have such a large number of events, since the player can continue activities indefinitely. 
# 
# Let us also look at the type of all the other game_session with event_count>100

# In[ ]:


# It seems the following code is a very slow implementation 
# I therefore abandoned this approach 
# a better implementation is given in the next cell
# There I use groupby to get the job done
# Therefore do NOT uncomment the code lines in this cell

#session_type=[]
#for session in GamesLargeEvCT:
#    typ=TrainData.loc[TrainData.game_session==session].type.iloc[0]
#    session_type.append([session, typ])

#LargeSessions=pd.DataFrame(session_type, columns=['game_session', 'type']) 


# In[ ]:


plt.plot()
EvCTLargeSess.groupby(['type']).game_session.unique().apply(lambda x: len(x)).plot(kind='bar')
plt.xlabel('type')
plt.ylabel('number of unique game_sessions')
plt.title('type of game_sessions that had an event_count>100')
plt.show()


# Since 'Activity' and 'Games' can be played indefinitely, it is possible that for some of the corresponding game sessions the event_count is high. There are also a small number of 'Assessments' for which the event count is high. Let's look at some of these. 

# In[ ]:


session=EvCTLargeSess.loc[EvCTLargeSess.type=='Assessment'].game_session.iloc[0]
TrainData.loc[TrainData.game_session==session]


# In the game_session (title= 'Mushroom Sorter (Assessment)') displayed in the previous cell, we see that the player seems to take some time to understand what he/she needs to do and fumbles around a little bit leading to a larger number of events. This explains why some assessments might end up having an unusually large event_count. 
# 
# It is perhaps of some interest to note that the player first gave an incorrect answer and upon getting a system feedback tried to attempt again but then exited the session in the middle of their attempt or else this is an incomplete record of events in this game_session. 

# There is a single game_session with more than 3500 events. Let us quickly look what happened here. 

# In[ ]:


session=TrainData.loc[TrainData.event_count>3200].game_session.unique()[0]
TrainData.loc[TrainData.game_session==session]


# Turns out, this was an 'activity' and the player continued to play this for a long time. The large number of events is therefore hardly suprising. 

# # EDA of test.csv

# In[ ]:


Test=pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# In[ ]:


Test.head()


# In[ ]:


Test.keys()


# In[ ]:


# let's plot the number of game_session of each type in Test

plt.figure()
Test.groupby(['type']).game_session.unique().apply(lambda x: len(x)).plot(kind='bar')
plt.xlabel('type')
plt.ylabel('number of game_sessions')
plt.title('number of game_session of each type in test.csv')
plt.show()


# In[ ]:


# let us look at the different kinds of events captured in the test data
# We can do this by looking at the kind of different event_codes
Test.event_code.unique()


# In[ ]:


len(Test.event_code.unique())


# Note that from our inspection of specs.csv, we had concluded that their of 42 distinct events that can occurr. Indeed, the test data seems to have all of them. 

# In[ ]:


# eye-balling the entries having type 'Activity'
Test.loc[Test.type=="Activity"].head(25)


# In[ ]:


# eye-balling the entries having type 'Game'
Test.loc[Test.type=="Game"].head(25)


# In[ ]:


# eye-balling the entries having type 'Assessment'
Test.loc[Test.type=="Assessment"].head(25)


#  let's look at all the events in the session with id = 8b38fc0d2fd315dc associated with the 'Cart Balancer (Assessment)' of the player with installation_id = 00abaee7

# In[ ]:


Test.loc[Test.game_session == '8b38fc0d2fd315dc']


# Turns out that the test data contains the full sequence of events and also the results of this assessment (result can be obtained from event with event_count 22 )

# let's look at all the assessments done by the player with installation_id = '00abaee7'

# In[ ]:


# In particular we wish to see the number of events in each assessment attempted by the chosen player
# One way to do this is to group their corresponding data according to their session_ids 
# The number of counts can then be obtained by using the function size()
# The reason this works is because each event has been recorded in a seperate row, so counting 
# the number of rows for each session (i.e. asking for their size) gives the number of events

# In order to obtain various statistics for each group in a pandas.GroupBy object 
# follow this stackoverflow discussion:
# https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby


Test.loc[(Test.installation_id =='00abaee7')
         & (Test.type=='Assessment')].groupby(['game_session']).size()


# Let us look at a few other players and checkout their assessments

# In[ ]:


player=Test.installation_id.unique()[5]
print('The chosen player has installation id: {}'.format(player))


# In[ ]:



Test.loc[(Test.installation_id ==player)
         & (Test.type=='Assessment')].groupby(['game_session']).size()


# In[ ]:


player=Test.installation_id.unique()[20]
print('The chosen player has installation id: {}'.format(player))


# In[ ]:



Test.loc[(Test.installation_id ==player)
         & (Test.type=='Assessment')].groupby(['game_session']).size()


# In[ ]:


player=Test.installation_id.unique()[35]
print('The chosen player has installation id: {}'.format(player))


# In[ ]:



Test.loc[(Test.installation_id ==player)
         & (Test.type=='Assessment')].groupby(['game_session']).size()


# As is also mentioned in the Data section of the competition, it looks like test.csv contains data about all the game_session of any particular player, except one randamly chosen assessment whose history is redacted after the start event. The aim of the competition then is predict the 'accuracy_group' for this assessment for each player in test.csv

# Let us compute the number of assessments taken by each player in test.csv and plot a histogram showing how many players took how many assessments.

# In[ ]:


num_assessments=Test.loc[Test.type=='Assessment'].groupby(['installation_id']).game_session.unique().apply(lambda x: len(x))


# In[ ]:


# minimum number of assessments taken by any player
print("The minimum number of assessments taken by any player in test data is:{}".
      format(num_assessments.min()))


# In[ ]:


# maximum number of assessments taken by any player
print("The maximum number of assessments taken by any player in test data is:{}".
      format(num_assessments.max()))


# In[ ]:


plt.figure()
num_assessments.hist(bins=56)
plt.xlabel('number of assessments')
plt.ylabel('number of players')
plt.title('Chart to show how many assessments were taken by how many players')
plt.show()


# In[ ]:


num_assessments.value_counts()


# Looks like a large number of players only took 1 or 2 assessments. 
