#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
gc.collect()

import pandas as pd
import numpy as np
 # Manually Annotated dataset from the above mentioned paper
df0 = pd.read_csv('../input/dataset-for-french/profilesmanualannotation.csv') 


# In[ ]:


import collections
df1 = df0[['UserId', 'party']] #Trimming down the first dataset
fr = pd.read_csv('../input/annotatedfriends/manualannotationFriends.csv', names=['id', 'friend']) #Dataset of Friends
fr.drop_duplicates(inplace = True)
Features = pd.read_csv('../input/features/possibleFeatures.csv' , names=['friend'])
#mlp_features = pd.read_csv('../input/mlp-importantfeatures/temp20 (1).csv')
#mlp_features = mlp_features['friend']
#Features


# In[ ]:


#Seperating all the parties
fnFriends = pd.merge(df0[df0['party'] == 'fn'], fr, how = 'inner', left_on='UserId' , right_on = 'id')
fiFriends = pd.merge(df0[df0['party'] == 'fi'], fr, how = 'inner', left_on='UserId' , right_on = 'id')
lrFriends = pd.merge(df0[df0['party'] == 'lr'], fr, how = 'inner', left_on='UserId' , right_on = 'id')
emFriends = pd.merge(df0[df0['party'] == 'em'], fr, how = 'inner', left_on='UserId' , right_on = 'id')
psFriends = pd.merge(df0[df0['party'] == 'ps'], fr, how = 'inner', left_on='UserId' , right_on = 'id')


# In[ ]:


fnFriendCount = fnFriends.groupby(['friend']).count()
fiFriendCount = fiFriends.groupby(['friend']).count()
lrFriendCount = lrFriends.groupby(['friend']).count()
emFriendCount = emFriends.groupby(['friend']).count()
psFriendCount = psFriends.groupby(['friend']).count()


# Merging datasets of mlp-features and friends.

# In[ ]:


#listDf = []
#for i in range(6,7):
#    listDf = []
#    listDf = (fnFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
#    fiFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
#    lrFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
#    emFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
#    psFriendCount.nlargest(i, 'UserId').index.values.tolist())
#    joe = pd.DataFrame({'friend' :listDf})
#    jpott = pd.merge(joe, fr, how = 'inner' , left_on = 'friend', right_on = 'friend' )
#    DicList = []
#    for group, frame in jpott.groupby('id'):
        
#        ak = frame['friend'].tolist()
#        dictOf = dict.fromkeys(ak , 1)
#        DicList.append(dictOf)
#    print(DicList[0])
#    from sklearn.feature_extraction import DictVectorizer
#    dictvectorizer = DictVectorizer(sparse = True)
#    features = dictvectorizer.fit_transform(DicList)
    #print(features)
    #print(features.todense().shape)
#    dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 
#                               index = jpott['id'].unique())
#    dataFrame.index.names = ['UserId']
#    print(dataFrame.head(1))
#    print(jpott['id'].unique())
    #print(jpott['id'] == 1278286086)
    


# In[ ]:


print(fr[fr['id']==1278286086])


# In[ ]:


#print(listDf)


# In[ ]:


joinedDF1 = pd.read_csv('../input/top6features/profile_manual_top6features (1).csv')
joinedDF1.shape


# In[ ]:


print(joinedDF1.friend.unique())


# In[ ]:


joinedDF1.friend.unique()


# In[ ]:


from scipy import sparse
accuracyScore = []
coreFeatures = []
listDf = []
for i in range(27, 28):
    listDf = []
    listDf = (fnFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
    fiFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
    lrFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
    emFriendCount.nlargest(i, 'UserId').index.values.tolist() + 
    psFriendCount.nlargest(i, 'UserId').index.values.tolist())
    joinDF = pd.DataFrame({'friend' :listDf})
    joinedDF = pd.merge(joinDF, fr, how = 'inner' , left_on = 'friend', right_on = 'friend' )
    
    #print(joinedDF.head(1))
    #merged = joinedDF1.merge(joinedDF, indicator=True, how='outer')
    #print(merged[merged['_merge'] == 'right_only'])
    #print(joinedDF.shape)
    DicList = []
    indexTobe = []
    for group, frame in joinedDF.groupby('id'):
        ak = frame['friend'].tolist()
        indexTobe.append(group)
        #break
        dictOf = dict.fromkeys(ak , 1)
        DicList.append(dictOf)
    #print(DicList[0])
    from sklearn.feature_extraction import DictVectorizer
    dictvectorizer = DictVectorizer(sparse = True)
    features = dictvectorizer.fit_transform(DicList)
    features.todense().shape
    dataFrame = pd.DataFrame.sparse.from_spmatrix(features, columns = dictvectorizer.get_feature_names(), 
                               index = indexTobe)
    #print(joinedDF['id'].unique())
    dataFrame.index.names = ['UserId']
    #print(dataFrame.head())
    mergedWithParties = pd.merge(dataFrame , df0, left_on = 'UserId', right_on = 'UserId', how= 'inner')
    mergedWithParties.drop(columns=['mediaConnection', 'gender', 'profileType'], inplace = True)
    mergedWithParties.fillna(0, inplace = True)
    #print('Before')
    #print(mergedWithParties.sample(random_state = 2))
    parties = {'fi': 1,'ps': 2,'em': 3,'lr': 4,'fn': 5,'fi/ps': 6,'fi/em': 7, 'fi/lr': 8,'fi/fn': 9, 'ps/em': 10,
    'ps/lr': 11, 'ps/fn': 12, 'em/lr': 13,'em/fn': 14, 'lr/fn': 15}
    #print(df1['party'])

    mergedWithParties['party'] = mergedWithParties['party'].map(parties)
    #print('After')
    #print(mergedWithParties.sample(random_state = 2))
    sanityCheck = pd.concat([mergedWithParties['UserId'],  mergedWithParties['party']], axis = 1)
    sanityCheck2 =  pd.concat([df0['UserId'] ,df0['party']], axis = 1)
    pd.set_option('display.max_columns', None)
    #print(pd.concat([sanityCheck, sanityCheck2], axis = 1))
    #sanity = sanityCheck.merge(sanityCheck2, indicator=True, how='outer')
    #print(merged[merged['_merge'] == 'right_only'])
    #print(merged.shape)
    
    mergedWithParties2 = mergedWithParties[(mergedWithParties['party']==1.0) | (mergedWithParties['party']==2.0)|
                                      (mergedWithParties['party']==3.0)| (mergedWithParties['party']==4.0)
                                      | (mergedWithParties['party']==5.0)]
    
    #print(mergedWithParties2[mergedWithParties2['party'] == 1.0])
    #print(mergedWithParties2[mergedWithParties2['party'] == 2.0])
    #print(mergedWithParties2[mergedWithParties2['party'] == 3.0].shape)
    #print(mergedWithParties2[mergedWithParties2['party'] == 4.0].shape)
    #print(mergedWithParties2[mergedWithParties2['party'] == 5.0].shape)
    
    from sklearn.ensemble import  RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(mergedWithParties2, test_size=0.2, shuffle=True)
    featureSelector = RandomForestClassifier(n_estimators=50)
    #print(train.iloc[:, :-1])
    featureSelector.fit(train.iloc[:, :-1], train['party'])
    accScore = featureSelector.score(test.iloc[:, :-1],pd.Series(test['party']))
    
    accuracyScore.append(accScore)
    coreFeatures.append(dictvectorizer.get_feature_names())
    print(accScore)
    print(coreFeatures)
    


# #adding jlm manually 

# In[ ]:


coreFeatures[0].append(int(80820758))


# In[ ]:


def getaff(af):
    pos = listDf.index(af)
    if (pos < len(listDf)/5):
        return 'fn'
    if (len(listDf)/5 < pos < 2*len(listDf)/5):
        return 'fi'
    if (2*len(listDf)/5 < pos < 3*len(listDf)/5):
        return 'lr'
    if (3*len(listDf)/5 < pos < 4*len(listDf)/5):
        return 'em'
    if (4*len(listDf)/5 < pos < 5*len(listDf)/5):
        return 'ps'
    
    


# In[ ]:


fn = []
fi = []
lr = []
ps = []
em = []
for feature in zip(dataFrame.columns, featureSelector.feature_importances_):
    if getaff(feature[0]) == 'fn':
        fn.append(feature)
    if getaff(feature[0]) == 'fi':
        fi.append(feature)
    if getaff(feature[0]) == 'ps':
        ps.append(feature)
    if getaff(feature[0]) == 'em':
        em.append(feature)
    if getaff(feature[0]) == 'lr':
        lr.append(feature)
    #if (feature[1] > 0.001 and feature[1] < 0.1):
        #print(str(feature) + ' ' + str(getaff(feature[0])))
print(sorted(fn, key=lambda x: x[1] , reverse=True))
print(sorted(fi, key=lambda x: x[1] , reverse=True))
print(sorted(ps, key=lambda x: x[1] , reverse=True))
print(sorted(em, key=lambda x: x[1] , reverse=True))
print(sorted(lr, key=lambda x: x[1] , reverse=True))


# In[ ]:


ListOfFriends = []
for i in dataFrame.columns:
    ListOfFriends.append(i)
#ListOfFriends.append(80820758)
ListOfFriends


# **Here Starts the code to find if polarization comes from the media or politicians**

# In[ ]:


importantFriends = pd.DataFrame(ListOfFriends, columns = ['Friend'])
polarizationData = pd.merge(fr, pd.DataFrame(importantFriends), left_on='friend', right_on='Friend', how='inner')
polarizationData


# Now I need to create mechanism that will that will determine if its media or politicians who create politicial polarization 

# In[ ]:


manualDetails = pd.read_csv('../input/asdasdasda/manuallyDone2.csv')
manualDetails


# In[ ]:


manualDetails = pd.read_csv('../input/partyprofessionparty/DifferentiatingFeatures3.csv', names = ['Id', 'ScreenName', 'Importance', 'WhichParty', 'AccountDescription'], encoding = "ISO-8859-1")
manualDetails.drop(manualDetails.index[0], inplace = True)

#manualDetails.loc[-1] = [80820758, 'jlmelenchon', 0.0, 'Fi', 'Party Leader']  # adding a row
#manualDetails.index = manualDetails.index + 1  # shifting index
#manualDetails.sort_index(inplace = True)  # sorting by index


# In[ ]:


ResultsList = [['myPartyLeader', 0], ['otherPartyLeader', 0], ['myAssociatedParty', 0]
              , ['otherAssociatedParty', 0], ['myMedia', 0], ['otherMedia', 0],
              ['myOfficialPartyAccount', 0], ['otherOfficialPartyAccount', 0], ['myUnofficialPartyAccount', 0]
              , ['otherUnofficialPartyAccount', 0], ['myForeignLearder', 0],
               ['otherForeignLearder', 0], ['myAlly', 0], ['otherAlly', 0]
              ]
FinalDataFrame = pd.DataFrame(ResultsList, columns = ['Description', 'count'])


# In[ ]:


Followers = polarizationData.id.drop_duplicates()
Followers = Followers.tolist()
importantFriends['score'] = 0
counter = 0
tempInt = 0
for i in Followers:
    tempData = polarizationData[polarizationData['id'] == i]
    tempData.reset_index(inplace = True)
    counter = counter + 1
    if counter == 100:
       break
    for j in tempData['friend']:
        tempInt = importantFriends[importantFriends['Friend'] == j]['score']
        addition = tempData.index[tempData['friend'] == j].tolist()
        additionValue = addition[0] + 1
        print('additionValue ' + str(additionValue))
        
        #Here starts the new code
        myParty = df0[df0['UserId'] == i]['party'].tolist()
        myParty = str(myParty[0])
        myParty = myParty.strip()
        myFriendsParty = manualDetails[manualDetails['Id'] == str(j)]['WhichParty'].tolist()
        myFriendsOccupation = manualDetails[manualDetails['Id'] == str(j)]['AccountDescription'].tolist()
        try:
            myFriendsParty = str(myFriendsParty[0])
            myFriendsParty = myFriendsParty.strip()
        except:
            myFriendsParty = ''
        try:
            myFriendsOccupation = str(myFriendsOccupation[0])
            myFriendsOccupation = myFriendsOccupation.strip()
        except:
            myFriendsOccupation = ''
        print(' ')
        print('My Start')
        print(myParty)
        print(myFriendsParty)
        print(myFriendsOccupation)
        if (len(myParty) > 0 and len(myFriendsParty) > 0 and len(myFriendsOccupation) > 0) and (myFriendsOccupation != None):
            #print(myParty)
            #print(myFriendsParty)
            if (myParty.lower() == myFriendsParty.lower()):
                print('My Own party')
                if (myFriendsOccupation == 'Media'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myMedia']['count']
                    #print(int(store) + additionValue)
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myMedia'),'count'] = store + additionValue
                if (myFriendsOccupation == 'Party Leader'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myPartyLeader']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myPartyLeader'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Associated with Party'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myAssociatedParty']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myAssociatedParty'), 'count'] = store + additionValue
                if(myFriendsOccupation == 'Official Party Account') or (myFriendsOccupation =='Party Official Account'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myOfficialPartyAccount']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myOfficialPartyAccount'),'count'] = store + additionValue
                if (myFriendsOccupation == 'unofficial Party account'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myUnofficialPartyAccount']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myUnofficialPartyAccount'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Foreign Leader'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myForeignLearder']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myForeignLearder'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Ally'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'myAlly']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'myAlly'),'count'] = store + additionValue
            else:
                print('Rival Party')
                if (myFriendsOccupation == 'Media'):
                    print('2')
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherMedia']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherMedia'),'count'] = store + additionValue
                if (myFriendsOccupation == 'Party Leader'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherPartyLeader']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherPartyLeader'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Associated with Party'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherAssociatedParty']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherAssociatedParty'), 'count'] = store + additionValue
                if(myFriendsOccupation == 'Official Party Account') or (myFriendsOccupation =='Party Official Account'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherOfficialPartyAccount']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherOfficialPartyAccount'),'count']  = store + additionValue
                if (myFriendsOccupation == 'unofficial Party account'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherUnofficialPartyAccount']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherUnofficialPartyAccount'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Foreign Leader'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherForeignLearder']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherForeignLearder'), 'count'] = store + additionValue
                if (myFriendsOccupation == 'Ally'):
                    store = FinalDataFrame[FinalDataFrame['Description'] == 'otherAlly']['count']
                    FinalDataFrame.loc[(FinalDataFrame['Description'] == 'otherAlly'),'count'] = store + additionValue
        print('My End')
        print(' ')       
        importantFriends.loc[(importantFriends['Friend'] == j), 'score'] = tempInt + additionValue
print(importantFriends[importantFriends['score'] != 0])
    #With Each follower, take the friends he has and give each of them a score based on 
    #their position in his hierachy (Check noted plan for furhter details)


# In[ ]:


print(FinalDataFrame)


# Attempt two, here I will seperate the parties

# In[ ]:


ResultsList = [['fnPartyLeader', 0, 0, 0, 0, 0], ['fiPartyLeader', 0, 0, 0, 0, 0], ['emPartyLeader', 0, 0, 0, 0, 0], 
               ['psPartyLeader', 0, 0, 0, 0, 0],['lrPartyLeader', 0, 0, 0, 0, 0], ['fnAssociatedwithParty', 0, 0, 0, 0, 0],
               ['fiAssociatedwithParty', 0, 0, 0, 0, 0], ['emAssociatedwithParty', 0, 0, 0, 0, 0], ['lrAssociatedwithParty', 0, 0, 0, 0, 0], 
               ['psAssociatedwithParty', 0, 0, 0, 0, 0], ['rightMedia', 0, 0, 0, 0, 0], ['centrerightMedia', 0, 0, 0, 0, 0],['centreMedia', 0, 0, 0, 0, 0],
               ['leftMedia', 0, 0, 0, 0, 0], ['centreleftMedia', 0, 0, 0, 0, 0],['fnPartyOfficialAccount', 0, 0, 0, 0, 0], ['fiPartyOfficialAccount', 0, 0, 0, 0, 0],
               ['psPartyOfficialAccount', 0, 0, 0, 0, 0], ['lrPartyOfficialAccount', 0, 0, 0, 0, 0],['emPartyOfficialAccount', 0, 0, 0, 0, 0],
               ['rightAlly', 0, 0, 0, 0, 0], ['centrerightAlly', 0, 0, 0, 0, 0],['centreAlly', 0, 0, 0, 0, 0],
               ['leftAlly', 0, 0, 0, 0, 0], ['centreleftAlly', 0, 0, 0, 0, 0]
              ]
FinalDataFrame = pd.DataFrame(ResultsList, columns = ['Description', 'fn', 'ps', 'em', 'lr', 'fi'])
occuranceFrequency = FinalDataFrame.copy()


# In[ ]:


#This code here is trying to make sense out of sequences of friendships


# In[ ]:



Followers = polarizationData.id.drop_duplicates()
Followers = Followers.tolist()
fnSeq = []
psSeq = []
emSeq = []
lrSeq = []
fiSeq = []
for i in Followers:
    concernedRow = df1[df1['UserId'] == i]
    myPartyIs = concernedRow['party'].tolist()
    myPartyIs = myPartyIs[0]
    allFriends = fr[fr['id'] == i]
    allFriends.reset_index(inplace = True, drop= True)
    onlyImpFriends = pd.merge(allFriends, pd.DataFrame(importantFriends), left_on='friend', right_on='Friend', how='inner', left_index = False)
    friendsList = onlyImpFriends['friend'].tolist()
    if (myPartyIs == 'fn'):
        fnSeq.append(friendsList) 
    if myPartyIs == 'ps':
        psSeq.append(friendsList) 
    if myPartyIs == 'lr':
        lrSeq.append(friendsList) 
    if myPartyIs == 'fi':
        fiSeq.append(friendsList) 
    if myPartyIs == 'em':
        emSeq.append(friendsList) 
    
    


# In[ ]:


get_ipython().system('pip install prefixspan')
from prefixspan import PrefixSpan
sequenceResults = PrefixSpan(fnSeq)
print(sequenceResults.frequent(20))


# In[ ]:


Followers = polarizationData.id.drop_duplicates()
Followers = Followers.tolist()
importantFriends['score'] = 0
#print(pd.DataFrame(importantFriends))
counter = 0
tempInt = 0
for i in Followers:
    tempData2 = fr[fr['id'] == i]
    tempData2.reset_index(inplace = True, drop= True)
    #print("tempData2")
    #print(tempData2)
    tempData3 = pd.merge(tempData2, pd.DataFrame(importantFriends), left_on='friend', right_on='Friend', how='inner', left_index = False)
    #print(tempData3.columns)
    #print("tempData3")
    #print(tempData3)
    #tempData = polarizationData[polarizationData['id'] == i]
    #tempData.reset_index(inplace = True)
    #print(tempData)
    counter = counter + 1
    if counter == 3:
       break
    for j in tempData3['friend']:
        tempInt = importantFriends[importantFriends['Friend'] == j]['score']
        addition = tempData3.index[tempData3['friend'] == j].tolist()
        additionValue = addition[0] + 1
        #print('additionValue ' + str(additionValue))
        
        #Here starts the new code
        myParty = df0[df0['UserId'] == i]['party'].tolist()
        myParty = str(myParty[0])
        myParty = myParty.strip()
        myFriendsParty = manualDetails[manualDetails['id'] == j]['belongsTo'].tolist()
        myFriendsOccupation = manualDetails[manualDetails['id'] == j]['Description'].tolist()
        try:
            myFriendsParty = str(myFriendsParty[0])
            myFriendsParty = myFriendsParty.strip()
        except:
            myFriendsParty = ''
        try:
            myFriendsOccupation = str(myFriendsOccupation[0])
            myFriendsOccupation = myFriendsOccupation.strip()
        except:
            myFriendsOccupation = ''
        #print(' ')
        #print('My Start')
        #print(myParty)
        #print(myFriendsParty)
        #print(myFriendsOccupation)
        rowName = myFriendsParty.lower() + myFriendsOccupation.replace(" ", "")
        #print(rowName)
        if (myParty != "" and myFriendsParty != "" and myFriendsOccupation != ""):
            store = FinalDataFrame[FinalDataFrame['Description'] == rowName][myParty]
            FinalDataFrame.loc[(FinalDataFrame['Description'] == rowName), myParty] = store + additionValue
            if (rowName == 'fnAssociatedwithParty'):
                print('fnAssociatedwithParty')
            time = occuranceFrequency[occuranceFrequency['Description'] == rowName][myParty]
            occuranceFrequency.loc[(occuranceFrequency['Description'] == rowName), myParty] = time + 1
        


# In[ ]:


occuranceFrequency.set_index('Description', inplace = True)
FinalDataFrame.set_index('Description', inplace = True)
print(FinalDataFrame/occuranceFrequency)


# In[ ]:


print(FinalDataFrame)


# Following code calculates the polarization score for each profiless

# In[ ]:


polarizationScoreEachProfile = pd.DataFrame({'id': polarizationData.id.drop_duplicates()})
polarizationScoreEachProfile.reset_index(inplace = True, drop = True)
polarizationScoreEachProfile['fn'] = 0
polarizationScoreEachProfile['ps'] = 0
polarizationScoreEachProfile['fi'] = 0
polarizationScoreEachProfile['lr'] = 0
polarizationScoreEachProfile['em'] = 0
polarizationScoreEachProfile['myParty'] = ''
polarizationScoreEachProfile['polarizationScore'] = 0
polarizationScoreEachProfile


# In[ ]:


counter = 0
Followers = polarizationData.id.drop_duplicates()
Followers = Followers.tolist()
for i in Followers:
    tempData2 = fr[fr['id'] == i]
    tempData3 = pd.merge(tempData2, pd.DataFrame(importantFriends), left_on='friend', right_on='Friend', how='inner', left_index = False)
    myParty = df0[df0['UserId'] == i]['party'].tolist()
    myParty = str(myParty[0])
    myParty = myParty.strip()
    polarizationScoreEachProfile.loc[(polarizationScoreEachProfile['id'] == i), 'myParty'] = myParty
    counter = counter + 1
    #if counter > 20:
    #    break
    for j in tempData3['friend']:
        
        myFriendsParty = manualDetails[manualDetails['id'] == j]['belongsTo'].tolist()
        myFriendsOccupation = manualDetails[manualDetails['id'] == j]['Description'].tolist()
        try:
            myFriendsParty = str(myFriendsParty[0])
            myFriendsParty = myFriendsParty.strip()
        except:
            myFriendsParty = ''
        try:
            myFriendsOccupation = str(myFriendsOccupation[0])
            myFriendsOccupation = myFriendsOccupation.strip()
        except:
            myFriendsOccupation = ''
        if myFriendsOccupation != 'Media' and myFriendsOccupation != 'Ally' and myFriendsParty != '':
            try:
                store = polarizationScoreEachProfile[polarizationScoreEachProfile['id'] == i][myFriendsParty]
                polarizationScoreEachProfile.loc[(polarizationScoreEachProfile['id'] == i), myFriendsParty] = store + 1
            except:
                print(myFriendsParty)
                print(myFriendsOccupation)
    #myPartyScore = polarizationScoreEachProfile[polarizationScoreEachProfile['id'] == i][myParty]
    #sumOfAll = 
    #polarizationScoreEachProfile.loc[(polarizationScoreEachProfile['id'] == i), 'polarizationScore'] = store + 1
print(polarizationScoreEachProfile)


# polarization Score Calculation 

# In[ ]:





# Following is a code to convert dataframe into csv and download it 

# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(polarizationScoreEachProfile)


# In[ ]:


polarizationData[polarizationData['Friend'] == 4884663951]


# **Here code for checking who creates polarization ends**

# In[ ]:


import matplotlib.pyplot as plt
plt.plot([0.6520629519353467, 0.7801532876159741, 0.8216432865731463, 0.8383162863886703, 0.8545668365346922, 0.8692427790788446, 0.8846601941747573, 0.8783888458559257, 0.8894433781190019, 0.8967766692248657, 0.8825554705432288, 0.8828691339183518, 0.8796190476190476, 0.8987823439878234, 0.897299353366299, 0.8950570342205323, 0.8854442344045369, 0.8945578231292517, 0.8947763998496806, 0.8813368381524597, 0.896021021021021, 0.887012012012012, 0.8942235558889723, 0.8909295352323838, 0.8871814092953523, 0.8865168539325843, 0.8865593410707601, 0.8877245508982036, 0.8986915887850467, 0.7769347496206374, 0.7836771844660194, 0.7685072815533981, 0.7668990603213095, 0.7693240375871476, 0.7750833585935132, 0.7771047849788008, 0.7817135937026946, 0.7651331719128329, 0.7723700120918985, 0.7772741009368389, 0.7739498337866425, 0.7613293051359517, 0.775226586102719, 0.7731117824773414, 0.7712990936555891, 0.7714371980676329, 0.7795893719806763, 0.7702988228191971, 0.7663748868095381])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.show()


# In[ ]:


print(len(featureSelector.feature_importances_))
plt.scatter(range(1, len(featureSelector.feature_importances_) + 1), featureSelector.feature_importances_)
plt.ylabel('importance')
plt.xlabel('Feature Number')
plt.show()


# In[ ]:


for i, j in zip(dataFrame.columns, featureSelector.feature_importances_):
    print (i)
    print(j)


# In[ ]:


coreFeatures[10]


# In[ ]:


#print(dictvectorizer.get_feature_names())
#print(dictvectoriz.get_feature_names())
import pandas as pd
DiList = []
indexTobe2 = []
validationData = pd.read_csv('../input/validationdata/mlp_validationdata.csv')
for group, frame in validationData.groupby('Id'):
        ak = frame['Friend'].tolist()
        indexTobe2.append(group)
        #break
        ditOf = dict.fromkeys(ak , 1)
        DiList.append(ditOf)
    #print(DicList[0])
from sklearn.feature_extraction import DictVectorizer
dictvectoriz = DictVectorizer(sparse = True)
fts = dictvectoriz.fit_transform(DiList)
fts.todense().shape
validationCheck = pd.SparseDataFrame(fts, columns = dictvectoriz.get_feature_names())
                               #index = indexTobe2)
missingcolu = list(set(test.columns.drop('party')) - set(dictvectoriz.get_feature_names()))
print(missingcolu )
validationCheck = validationCheck.reindex(columns=[*validationCheck.columns.tolist()] + missingcolu )
validationCheck.fillna(0.0, inplace = True )
print(validationCheck)
pred = featureSelector.predict(validationCheck)
counter = 0
for i in pred:
    if i == 1:
        counter = counter + 1
print(counter)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(accuracyScore)


# In[ ]:


joinedDF = pd.read_csv('../input/top6features/profile_manual_top6features (1).csv')
#joinedDF = pd.merge(Features, fr, left_on = 'friend', right_on = 'friend', how= 'inner')
#joinedDF.shape


# In[ ]:


#mergedWithParties = pd.merge(joinedDF , df0, left_on = 'id', right_on = 'UserId', how= 'inner')
#mergedWithParties[mergedWithParties['party'] == 'fi'].nunique()


# In[ ]:


DicList = []
for group, frame in joinedDF.groupby('id'):
    ak = frame['friend'].tolist()
    dictOf = dict.fromkeys(ak , 1)
    DicList.append(dictOf)

from sklearn.feature_extraction import DictVectorizer
dictvectorizer = DictVectorizer(sparse = True)
features = dictvectorizer.fit_transform(DicList)
features.todense().shape


# In[ ]:


dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 
                               index = joinedDF['id'].unique())


# In[ ]:


dataFrame.index.names = ['UserId']
dataFrame.head()


# In[ ]:


mergedWithParties = pd.merge(dataFrame , df0, left_on = 'UserId', right_on = 'UserId', how= 'inner')


# In[ ]:


mergedWithParties.head()


# In[ ]:


mergedWithParties.drop(columns=['mediaConnection', 'gender', 'profileType'], inplace = True)


# In[ ]:


mergedWithParties.fillna(0, inplace = True)


# In[ ]:


mergedWithParties.head()


# In[ ]:


parties = {'fi': 1,'ps': 2,'em': 3,'lr': 4,'fn': 5,'fi/ps': 6,'fi/em': 7, 'fi/lr': 8,'fi/fn': 9, 'ps/em': 10,
'ps/lr': 11, 'ps/fn': 12, 'em/lr': 13,'em/fn': 14, 'lr/fn': 15}
#print(df1['party'])

mergedWithParties['party'] = mergedWithParties['party'].map(parties)
mergedWithParties.head()


# In[ ]:


mergedWithParties2 = mergedWithParties[(mergedWithParties['party']==1.0) | (mergedWithParties['party']==2.0)|
                                      (mergedWithParties['party']==3.0)| (mergedWithParties['party']==4.0)
                                      | (mergedWithParties['party']==5.0)]
mergedWithParties2.head()


# In[ ]:


from sklearn.ensemble import  RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

train, test = train_test_split(mergedWithParties2, test_size=0.2, shuffle=True)
featureSelector = RandomForestClassifier(n_estimators=50)
featureSelector.fit(train.iloc[:, :-1], train['party'])


# In[ ]:


featureSelector.score(test.iloc[:, :-1],pd.Series(test['party']))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors = 800)
clf2.fit(train.iloc[:, :-1], train['party'])
clf2.score(test.iloc[:, :-1],pd.Series(test['party']))
print(test.shape)


# In[ ]:


Ypred = pd.Series(clf2.predict(test.iloc[:, :-1]))


# In[ ]:


allLabels = mergedWithParties2['party'].unique()
print(allLabels)


# In[ ]:


pred = list(pd.Series(featureSelector.predict(test.iloc[:, :-1])))
print(len(pred))
testY = list(pd.Series(test['party']))
labels = list(allLabels)
from collections import Counter
print(Counter(pred))
print(Counter(testY))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(pred, testY))
print(labels)


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


# In[ ]:


import pandas as pd
mlp_validationdata = pd.read_csv("../input/mlp_validationdata.csv")


# In[ ]:


import pandas as pd
data2 = pd.read_csv("../input/data2.csv")


# In[ ]:


import pandas as pd
DifferentiatingFeatures = pd.read_csv("../input/DifferentiatingFeatures.csv")


# In[ ]:


import pandas as pd
DifferentiatingFeatures3 = pd.read_csv("../input/DifferentiatingFeatures3.csv")


# In[ ]:


import pandas as pd
manuallyDone (version 1)_xlsb = pd.read_csv("../input/manuallyDone (version 1).xlsb.csv")


# In[ ]:


import pandas as pd
manuallyDone2 = pd.read_csv("../input/manuallyDone2.csv")


# In[ ]:


import pandas as pd
manuallyDone2 = pd.read_csv("../input/manuallyDone2.csv")

