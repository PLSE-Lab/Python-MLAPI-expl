#!/usr/bin/env python
# coding: utf-8

# # Predicting Survivors Aboard the Titanic

# ### Overview

# This is for an ongoing Kaggle competition where contestants are tasked with predicting the survival of passengers based the provided information. Let's begin.

# ### Loading in the Data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
dfTrain = pd.read_csv('../input/train.csv')
dfTrain.head()


# ### Names

# Create Names DF, leave Sex for comparison.

# In[ ]:


namesDF = dfTrain.drop(['Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId'],1)


# Extracting titles from Names

# In[ ]:


title = []
for index, row in namesDF.iterrows():
    t = row['Name'].split(' ')
    title.append(t[1])
print(Counter(title))
title = pd.Series(title)
namesDF['Name'] = title.values
namesDF.head()


# Create Dummies for Names, Sex

# In[ ]:


namesDF = pd.get_dummies(namesDF, prefix=['Name','Sex'],columns=['Name','Sex'])
namesDF.head()


# Drop titles with low counts

# In[ ]:


namesDF.drop(['Name_Col.','Name_Planke,','Name_Billiard,','Name_Impe,','Name_Major.',
              'Name_Gordon,','Name_Mlle.','Name_Carlo,','Name_Ms.','Name_Messemaeker,','Name_Mme.',
              'Name_Capt.','Name_Jonkheer.','Name_the','Name_Don.','Name_der',
              'Name_Mulder,','Name_Pelsmaeker,','Name_Shawah,','Name_Melkebeke,','Name_Velde,',
              'Name_Walle,','Name_Steen,','Name_Cruyssen,','Name_y'],1,inplace=True)


# Generate a correlation matrix

# Rename Colums

# In[ ]:


namesDF.columns = ['Survived','Dr','Master','Ms','Mr','Mrs','Rev','Female','Male']


# Generate a correlation matrix

# In[ ]:


namesCor = namesDF.corr()
mask = np.zeros_like(namesCor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(9, 7))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(namesCor, mask=mask, cmap=cmap, vmax=.3,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.title("Names Correlation")
plt.show()


# Drop Dr as it seems to have no correlation with survival:

# In[ ]:


namesDF = namesDF.drop(['Dr'],1)


# Add to the final DF

# In[ ]:


finalDF = namesDF
finalDF.head()


# ### Age

# In[ ]:


ageSexDF = dfTrain.drop(['Pclass','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId'],1)


# First, some summary stats

# In[ ]:


ageSexDF.head()


# In[ ]:


ageSexDF.mean()


# Seperate by Sex

# In[ ]:


ageWDF = ageSexDF[ageSexDF.Sex != 'male']
ageMDF = ageSexDF[ageSexDF.Sex != 'female']
print(ageWDF.mean())
print(ageMDF.mean())


# Run logistic regression on Survived vs Age by Sex

# In[ ]:


f,ax = plt.subplots(figsize=(8, 6))
logWAge = sns.regplot(x='Age',y='Survived',data=ageWDF,logistic=True,ax=ax,color="#F08080")
logMAge = sns.regplot(x='Age',y='Survived',data=ageMDF,logistic=True,ax=ax,color="#6495ED")
plt.title("LogOdds: Survival vs Age by Gender \n Women: Orange | Men: Blue")
plt.show(logWAge)
plt.show(logMAge)


# Age seems to be a benefit for women and a detriment for men.
# We can account for this discreprency by creating an interaction term.

# In[ ]:


ageSexDF = pd.get_dummies(ageSexDF,prefix=['Sex'],columns=['Sex'])
femAge = ageSexDF.Sex_female * ageSexDF.Age
malAge = ageSexDF.Sex_male * ageSexDF.Age
ageSexDF['FemAge'] = femAge
ageSexDF['MalAge'] = malAge
ageSexDF.drop(['Sex_female','Sex_male','Age'],1,inplace=True)
ageSexDF.head()


# And add it to the final DF

# In[ ]:


finalDF = pd.concat([finalDF.drop(['Female','Male'],1), ageSexDF.drop(['Survived'],1)], axis=1)
finalDF.head()


# ### Family

# Add siblings/spouse and parents/children together

# In[ ]:


totFam = dfTrain.SibSp + dfTrain.Parch
#Just Need an empty DF
famDF = dfTrain.drop(['Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId','Name','Sex'],1)
famDF['Family'] = totFam
famDF.head(10)


# Binarize Family

# In[ ]:


famDF.loc[famDF['Family'] > 0, 'Family'] = 1
famDF.head(10)


# Compare family vs no family

# In[ ]:


withFamDF = famDF[famDF.Family == 1]
noFamDF = famDF[famDF.Family ==0]
barFam = sns.factorplot(x='Family',y='Survived',data=famDF,kind='bar',palette = ["#F08080","#6495ED"])
plt.title("Survival vs Family Onboard")
plt.show(barFam)
print("With Family: ",withFamDF.mean()[0], " No Family: ",noFamDF.mean()[0])


# Twenty percent higher chance of survival if family on board.
# Good enough for me, let's add to the final DF.

# In[ ]:


finalDF = pd.concat([finalDF,famDF.drop(['Survived'],1)],axis=1)
finalDF.head()


# ### Wealth

# Here I'm going to analyze Class and Fare at the same time

# In[ ]:


wealthDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Cabin','Embarked','PassengerId','Name','Sex'],1)
wealthDF.head()


# Create some visuals
# (Noticed an outlier for Fare so I removed it from analyses for now)

# In[ ]:


barClass = sns.factorplot('Pclass','Survived',data=wealthDF,kind='bar',palette = ["#F08080","#6495ED","#A5FEE3"])
plt.title("Survival vs Class")
plt.show(barClass)

g,ax2 = plt.subplots(figsize=(8, 6))
logFare = sns.regplot(x='Fare',y='Survived',data=wealthDF[wealthDF.Fare < 500],logistic=True,ax=ax2,color="#F08080")
plt.title("LogOdds: Survival vs Fare")
plt.show(logFare)


# Interaction between wealth and Sex

# In[ ]:


wealthDF['Sex'] = dfTrain['Sex']
barClassSex = sns.factorplot(x = 'Sex', y='Survived', col='Pclass',data=wealthDF
                              ,kind='bar',palette = ["#6495ED","#F08080"])

wealthDFMen = wealthDF[wealthDF.Fare < 500][wealthDF.Sex == 'female']
wealthDFWomen = wealthDF[wealthDF.Fare < 500][wealthDF.Sex == 'male']
h,ax3 = plt.subplots(figsize=(9, 7))
fig5 = sns.regplot(x='Fare',y='Survived',data=wealthDFWomen,logistic=True,ax=ax3,color="#F08080")
fig6 = sns.regplot(x='Fare',y='Survived',data=wealthDFMen,logistic=True,ax=ax3,color="#6495ED")
plt.title("LogOdds: Survival vs Fare by Sex \n Women: Orange | Men: Blue")
plt.show(fig5)
plt.show(fig6)


# Looks like wealth wealth is more important for men than it is for
# women in terms of survival.<br>
# However, I went ahead and ran the interaction terms and the did not
# improve the accuracy by any.<br>
# Furthermore, the second class seems to be very good predictor of survival.<br>
# We can make a correlation matrix to check

# In[ ]:


wealthDFDum = pd.get_dummies(wealthDF,prefix=["Pclass"],columns=['Pclass'])
wealthDFDumCor = wealthDFDum.corr()
print(wealthDFDumCor)


# Class 2 has a pretty insignificant correlation to survival so I won't keep it in the final DF

# In[ ]:


finalDF = pd.concat([finalDF,wealthDFDum.drop(['Survived','Pclass_2','Sex'],1)],axis=1)
finalDF.head()


# ### Cabin

# Extract cabin group from Cabin

# In[ ]:


cabin = []
for index, row in dfTrain.iterrows():
    try:
        c = row['Cabin']
        cabin.append(c[0])
    except Exception:
        cabin.append(c)

print(Counter(cabin))
cabin = pd.Series(cabin)
cabinDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Fare','Embarked','PassengerId','Name','Sex','Pclass'],1)
cabinDF['Cabin'] = cabin.values
print(cabinDF.head(10))


# Create a barplot for cabin groups

# In[ ]:


barCabin = sns.factorplot(x = 'Cabin', y = 'Survived', kind='bar',data = cabinDF)
plt.title("Survival vs Cabin")
plt.show(barCabin)


# Looks like there might be some meaningful info here.<br>
# Let's drop G and T though due to size.

# In[ ]:


cabinDF = pd.get_dummies(cabinDF, prefix=['Cabin'], columns=['Cabin'])
cabinDF.drop(['Cabin_G','Cabin_T'],1,inplace=True)
finalDF = pd.concat([finalDF,cabinDF,],axis=1)
finalDF.head()


# ### Embarked

# Embarked is another categorical feature with 3 labels like Class so I'll
# start off with another bar plot

# In[ ]:


embarkedDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Fare','Cabin','PassengerId','Name','Sex','Pclass'],1)
embarkedDF.head()
barEmbarked = sns.factorplot('Embarked','Survived',data=embarkedDF,kind='bar',
                               palette = ["#F08080","#6495ED","#A5FEE3"])
plt.title("Survival vs Embarked")
plt.show(barEmbarked)


# Get some correlations

# In[ ]:


embarkedDFDum = pd.get_dummies(embarkedDF,prefix=['Embarked'],columns=['Embarked'])
print(embarkedDFDum.corr())


# Q has very low correlation with survival, might as well drop it.

# In[ ]:


finalDF = pd.concat([finalDF,embarkedDFDum.drop(['Embarked_Q','Survived'],1)],axis=1)
finalDF.head()


# ### Training a Classifier

# Prepare for training

# In[ ]:


finalDF = finalDF[finalDF.Fare < 500]
features = finalDF.drop(['Survived'],1)
labels = finalDF['Survived']


# For now, I'll drop NaN's (these are coming from unknown ages).
# This is only a hotfix. When you go to submit predictions, you'll
# have to include all data. I would just fill NaN's with the average 
# age that corrsponds to survival.

# In[ ]:


finalDF = finalDF.dropna()


# I played around with different algorithms and parameters and found
# random forest to be the best by a considerable margin.

# Testing against own data

# In[ ]:


finalDF = finalDF[finalDF.Fare < 500]
features = finalDF.drop(['Survived'],1)
labels = finalDF['Survived']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
accuracy = []
for i in range(100):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(features,labels)
    accuracy.append(clf.score(features,labels))
print(sum(accuracy)/len(accuracy))


# 99 percent accuracy is cool.... but we tested against our own data.
# Shuffle the data to test against never-before-seen data

# In[ ]:


from sklearn import cross_validation
accuracy2 = []
for i in range(100):
    featureTrain, featureTest, labelTrain, labelTest = cross_validation.train_test_split(
        finalDF.drop(['Survived'], 1),
        finalDF['Survived'], test_size=0.20)


    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(featureTrain,labelTrain)
    accuracy2.append(clf.score(featureTest,labelTest))

print(sum(accuracy2)/len(accuracy2))


# ...and that's the final accuracy, for now. Can always go back and try out 
# different interactions and models, but I'll hold here for now.

# In[ ]:




