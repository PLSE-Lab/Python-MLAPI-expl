#!/usr/bin/env python
# coding: utf-8

# Import Pandas Library to import dataset & analyse dataset 

# In[ ]:


import pandas as pd


# Import training & test data set

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Analyse by fetching information for dataframe

# In[ ]:


train_df.info()


# View sample records

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# Describe object columns of a dataframe

# In[ ]:


train_df.describe(include="O")


# Describe test data set to see how it differ from training dataset

# In[ ]:


test_df.describe()


# In[ ]:


test_df.describe(include="O")


# # Feature Engineering 
# 
#     1. Cabin  -- Analyse Cabin feature

# In[ ]:


train_df['Cabin_Start'] = train_df['Cabin'].str[0]


# In[ ]:


train_df[['Survived','Cabin_Start']].groupby(['Cabin_Start']).mean()


# In[ ]:


pd.crosstab(train_df['Survived'],train_df['Cabin_Start'])


# In[ ]:


train_df[['Pclass','Cabin_Start']].groupby(['Cabin_Start']).mean()


# In[ ]:


pd.crosstab(train_df['Pclass'],train_df['Cabin_Start'])


# In[ ]:


train_df[['Fare','Cabin_Start']].groupby(['Cabin_Start']).mean()


# In[ ]:


combine = [train_df, test_df]


# In[ ]:


for dataset in combine:
    dataset['Cabin'].fillna('0', inplace=True)
    dataset.loc[ dataset['Cabin'].str[0] == 'A', 'Cabin'] = 1
    dataset.loc[ dataset['Cabin'].str[0] == 'B', 'Cabin'] = 2
    dataset.loc[ dataset['Cabin'].str[0] == 'C', 'Cabin'] = 3
    dataset.loc[ dataset['Cabin'].str[0] == 'D', 'Cabin'] = 4
    dataset.loc[ dataset['Cabin'].str[0] == 'E', 'Cabin'] = 5
    dataset.loc[ dataset['Cabin'].str[0] == 'F', 'Cabin'] = 6
    dataset.loc[ dataset['Cabin'].str[0] == 'G', 'Cabin'] = 7
    dataset.loc[ dataset['Cabin'].str[0] == 'T', 'Cabin'] = 8
    dataset['Cabin'] = dataset['Cabin'].astype(int)


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include='O')


# In[ ]:


train_df = train_df.drop(["Cabin_Start"],axis=1)


#     2. PassengerId  -- Drop PassengerID columns as it is just a counter

# In[ ]:


train_df = train_df.drop(["PassengerId"],axis=1)


# 
#     3. Ticket 
#     
#     Get Ticket length & check if length or alpha numeric ticket has any impact on survival chances

# In[ ]:


train_df["Ticket_Length"] = train_df["Ticket"].str.len()


# In[ ]:


train_df.head()


# In[ ]:


train_df["Ticket_Contains_Alpha"] = train_df["Ticket"].str.contains('^[a-zA-Z]')


# In[ ]:


train_df.head()


# In[ ]:


train_df[["Survived","Ticket_Length"]].groupby(['Ticket_Length'],as_index=False).mean()


# In[ ]:


pd.crosstab(train_df['Survived'],train_df['Ticket_Length'])


# In[ ]:


train_df[["Survived","Ticket_Contains_Alpha"]].groupby(['Ticket_Contains_Alpha']).mean()


# In[ ]:


pd.crosstab(train_df['Survived'],train_df['Ticket_Contains_Alpha'])


# In[ ]:


train_df[["Survived","Ticket_Contains_Alpha","Ticket_Length"]].groupby(["Ticket_Contains_Alpha","Ticket_Length"]).mean()


#     Ticket columns doesn't seem to impact much so drop it 

# In[ ]:


train_df = train_df.drop(["Ticket","Ticket_Length","Ticket_Contains_Alpha"],axis=1)
test_df = test_df.drop(["Ticket"],axis=1)


# In[ ]:


train_df.head()


#     4. Pclass
#     
#     See how Pclass effect survival chances 

# In[ ]:


train_df[["Survived","Pclass"]].groupby(["Pclass"],as_index=False).mean()


# Only 3 unique values & with Pclass 1 with Surival chance of 63 % 

#     5. Name
#     
#     See how Name effect survival chances
#     
#     First extract title

# In[ ]:


train_df["Name_Title"] = train_df["Name"].str.extract('([A-Za-z]+\.)',expand=False)


# In[ ]:


train_df.head()


# In[ ]:


train_df["Name_Title"].unique()


# In[ ]:


train_df[["Survived","Name_Title"]].groupby(["Name_Title"],as_index=False).mean()


# Cross tabulation of Title field with Survived Columns

# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Name_Title"])


# Replace all rare gone cases with new type as gone 

# In[ ]:


train_df["Name_Title"] = train_df["Name_Title"].replace(['Capt.','Don.','Jonkheer.','Rev.'],'Gone.')


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Name_Title"])


# Replace all rare survived cases with new type as Left

# In[ ]:


train_df["Name_Title"] = train_df["Name_Title"].replace(['Countess.','Lady.','Mlle.','Mme.','Ms.','Sir.'],'Left.')


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Name_Title"])


# Replace all rare cases with new type as Half

# In[ ]:


train_df["Name_Title"] = train_df["Name_Title"].replace(['Col.','Dr.','Major.'],'Half.')


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Name_Title"])


# Do the same cleaning process for Test dataset

# In[ ]:


test_df["Name_Title"] = test_df["Name"].str.extract('([A-Za-z]+\.)',expand=False)
test_df["Name_Title"].unique()


# In[ ]:


test_df["Name_Title"] = test_df["Name_Title"].replace(['Capt.','Don.','Jonkheer.','Rev.'],'Gone.')
test_df["Name_Title"] = test_df["Name_Title"].replace(['Col.','Dr.','Major.'],'Half.')


# In[ ]:


test_df["Name_Title"].unique()


# As dataset have some additional title use sex code to classify them

# In[ ]:


pd.crosstab(test_df["Name_Title"],test_df["Sex"])


# In[ ]:


test_df["Name_Title"] = test_df["Name_Title"].replace(['Dona.','Ms.'],'Mrs.')


# In[ ]:


pd.crosstab(test_df["Sex"],test_df["Name_Title"])


# Do mapping of Name Title with numerical field

# In[ ]:


title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Gone.": 5, "Half.": 6, "Left.": 7}
train_df['Title_Encoded'] = train_df['Name_Title'].map(title_mapping)
train_df['Title_Encoded'] = train_df['Title_Encoded'].fillna(0)


# In[ ]:


train_df.head()


# In[ ]:


test_df['Title_Encoded'] = test_df['Name_Title'].map(title_mapping)
test_df['Title_Encoded'] = test_df['Title_Encoded'].fillna(0)


# Drop existing Title columns

# In[ ]:


train_df = train_df.drop(["Name","Name_Title"],axis=1)
test_df = test_df.drop(["Name","Name_Title"],axis=1)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include='O')


#     6. Sex

# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Sex"])


# In[ ]:


train_df[["Survived","Sex"]].groupby(["Sex"],as_index=False).mean()


# Female have a higher probability of survival with 74%

# In[ ]:


train_df["Sex"] = train_df["Sex"].map({'female':1,'male':0}).astype(int)


# In[ ]:


test_df["Sex"] = test_df["Sex"].map({'female':1,'male':0}).astype(int)


# In[ ]:


train_df.head(10)


#     7. Age 
#        
#     See how it impact survival  

# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Age"])


# Age has missing values, compare it with other columns to see which can be used to generate missing values

# In[ ]:


train_df[["Survived","Age","Pclass"]].groupby(["Pclass"],as_index=False).mean()


# In[ ]:


train_df[["Survived","Age","Pclass","Sex"]].groupby(["Pclass","Sex"],as_index=False).mean()


# In[ ]:


train_df[["Survived","Age","Pclass","Sex","Title_Encoded"]].groupby(["Pclass","Sex","Title_Encoded"],as_index=False).mean()


# In[ ]:


train_df[["Survived","Age","Sex","Title_Encoded"]].groupby(["Sex","Title_Encoded"],as_index=False).mean()


# In[ ]:


train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"],as_index=False).mean()


# Title Encoded makes more sense to use as it classify the person with master, mr which signify age level

# In[ ]:


train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"]).mean()


# In[ ]:


age_mapping = train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"]).mean().to_dict()


# In[ ]:


age_mapping


# In[ ]:


age_mapping["Age"]


# In[ ]:


train_df["Age"] = train_df["Age"].fillna(train_df["Title_Encoded"].map(age_mapping["Age"]))


# In[ ]:


train_df.describe()


# In[ ]:


test_df["Age"] = test_df["Age"].fillna(test_df["Title_Encoded"].map(age_mapping["Age"]))


# In[ ]:


test_df.describe()


# In[ ]:


train_df.head()


# Convert Age level vales in categorical numerical values

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["AgeBand"])


# In[ ]:


combine = [train_df, test_df]


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 
    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Drop AgeBand

# In[ ]:


train_df= train_df.drop(["AgeBand"],axis=1)


# In[ ]:


train_df.head()


#     8. SibSp
#     
#     See how SibSp effect Survival

# In[ ]:


train_df[["Survived","SibSp"]].groupby(["SibSp"],as_index=False).mean()


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["SibSp"])


#     9. Parch
#     
#     See how Parch effect Survival

# In[ ]:


train_df[["Survived","Parch"]].groupby(["Parch"],as_index=False).mean()


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Parch"])


#     10. Fare
#     
#     See how Fare influence Survival

# In[ ]:


train_df["FareBand"] = pd.qcut(train_df["Fare"],10)
train_df[["Survived","FareBand"]].groupby(["FareBand"],as_index=False).mean()


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["FareBand"])


# Test dataset have missing vale for Fare, complete it before converting it in categorical values

# In[ ]:


test_df["Fare"].fillna(test_df["Fare"].dropna().median(),inplace=True)


# In[ ]:


test_df.describe()


# In[ ]:


combine = [train_df, test_df]


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.55, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.55) & (dataset['Fare'] <= 7.854), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 8.05), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 10.5), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 14.454), 'Fare']   = 4
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 21.679), 'Fare']   = 5
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 27), 'Fare']   = 6
    dataset.loc[(dataset['Fare'] > 27) & (dataset['Fare'] <= 39.688), 'Fare']   = 7
    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 77.958), 'Fare']   = 8
    dataset.loc[ dataset['Fare'] > 77.958, 'Fare'] = 9
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df = train_df.drop(["FareBand"],axis=1)


# In[ ]:


train_df.head()


#     11. Embarked
#     
#     See how Embarked vales effet survival chances

# In[ ]:


train_df[["Survived","Embarked"]].groupby(["Embarked"],as_index=False).mean()


# In[ ]:


pd.crosstab(train_df["Survived"],train_df["Embarked"])


# In[ ]:


train_df.describe(include="O")


# In[ ]:


test_df.describe(include="O")


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].dropna().mode()[0])


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].map({"C":0,"Q":1,"S":2}).astype(int)
test_df["Embarked"] = test_df["Embarked"].map({"C":0,"Q":1,"S":2}).astype(int)


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


X_train = train_df.drop(["Survived"], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId"], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# Feature engineering completed now move to Model training

# # Model Training
# Import All required library

# In[ ]:


#from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import model_selection,metrics
from sklearn.metrics import confusion_matrix
import xgboost
from xgboost import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV


# In[ ]:


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X_train, Y_train,
                                                                    test_size=0.3,stratify=Y_train,random_state=0)


# In[ ]:


train_x.head()


# In[ ]:


xgboost_model = xgboost.XGBClassifier(objective='binary:logistic',learning_rate=0.1)


# In[ ]:


eval_set = [(train_x,train_y),(valid_x,valid_y)]


# In[ ]:


xgboost_model.fit(train_x,train_y,eval_metric=['error','logloss','auc'],eval_set=eval_set,verbose=True)


# In[ ]:


xgboost_model.score(train_x,train_y)


# In[ ]:


pred_y = xgboost_model.predict(valid_x)
metrics.accuracy_score(valid_y,pred_y)


# In[ ]:


pred_test = xgboost_model.predict(X_test)


# In[ ]:


submission = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived":pred_test})


# In[ ]:


submission.to_csv('submission2.csv', index=False)


# In[ ]:


len(submission[submission.Survived ==1 ])


# In[ ]:


plot_importance(xgboost_model)
plt.show()


# In[ ]:


results = confusion_matrix(valid_y, pred_y) 
print(results)


# In[ ]:




