#!/usr/bin/env python
# coding: utf-8

# # A) Data wrangling and training phase
#    # Importing libraries and reading the file

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


train_df=pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train_df['Target']=train_df['Survived'].astype(int)
train_df['Survived']=train_df['Survived'].map({1:'Yes',0:'No'})
train_df.head()


# # Checking for NaN values in our dataframe

# In[ ]:


train_df.info()


# In[ ]:


train_df['PassengerId'].isnull().any()


# In[ ]:


train_df['Pclass'].isnull().any()


# In[ ]:


train_df['Name'].isnull().any()


# In[ ]:


train_df['Sex'].isnull().any()


# In[ ]:


train_df['Age'].isnull().any()


# As we can see, there are ages that are missing. Let us see how many.

# In[ ]:


train_df['Age'].isnull().value_counts()


# Hence, we have 177 entries with NaN values.

# In[ ]:


train_df['SibSp'].isnull().any()


# The above column shows number of siblings or spouses.

# In[ ]:


train_df['Parch'].isnull().any()


# Shows number of parents or children aboard

# In[ ]:


train_df['Cabin'].isnull().value_counts()


# We have 687 cases with no cabin info

# In[ ]:


train_df['Embarked'].isnull().value_counts()


# # Data Visualisation

# Let us visualise the data of each field so that we can understand how to deal with the NaN values.

# ## 1. Age

# In[ ]:


sns.kdeplot(train_df['Age'],shade=True)


# The above distribution looks bimodal in nature containing two peaks. Most passengers were between 20-40

# In[ ]:


sns.boxplot('Age',data=train_df)


# In[ ]:


train_df['Age'].describe()


# In[ ]:


train_df['Age'].median()


# Instead of using a mean age of 29.7 (or 30) , we should select median age to replace the NaN values.

# In[ ]:


train_df['Age'].fillna(train_df['Age'].median(),inplace=True)


# In[ ]:


train_df['Age'].isna().any()


# Hence, all missing values of Age has been taken care of. Let us now plot a histogram to check which age groups are maximum in the Titanic passengers.

# In[ ]:


bins=np.arange(0,90,10) - 0.5
plt.figure(figsize=(10,8))
plt.hist(train_df['Age'],bins=bins)
plt.xticks(rotation=90)


# As we can see, ages between 20-30 is the highest followed by 30-40.

# Let us make another age group for children. Ages lower than 16 will be considered as a child. This way, we can segregate the children from the list of passengers.

# In[ ]:


def child(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex


# In[ ]:


train_df['Person']=train_df[['Age','Sex']].apply(child,axis=1)


# In[ ]:


train_df['Person'].value_counts()


# Let us check the age distribution of all adults and children.

# In[ ]:


fig1=sns.FacetGrid(train_df,hue='Sex',aspect=2,height=6)
fig1.map(sns.kdeplot,'Age',shade=True)
fig1.add_legend()


# For adult males and females, the age distribution is extremely similar. Let us now compare it with the children.

# In[ ]:


fig2=sns.FacetGrid(train_df,hue='Person',aspect=2,height=8)
fig2.map(sns.kdeplot,'Age',shade=True)
fig2.add_legend()


# Let us check if gender or being a child plays a role in survival from the ship.

# In[ ]:


sns.catplot('Person',data=train_df,kind='count',hue='Survived',aspect=2,height=6)


# From above data, we can say that maximum number of female adults survived the accident. Most male passengers died. For children, just above 50 percent survived while the rest could not.
# 
# The reason could be that most female passengers alongiwith their children were evacuated first from the ship. Males were probably evacuated in the end and hence, could not make it.
# 
# Let us now see how age played at important role in survival.

# In[ ]:


sns.lmplot('Age','Target',data=train_df,aspect=2,height=6)


# As we can see, with increase in age, probability of survival decreases. This could be due to the fact that older people found it more difficult to act quickly during the evacuation time as compared to the young and middle aged passengers.
# 

# ## 2. Embarked

# In[ ]:


train_df['Embarked'].unique()


# * S stands for Southampton.
# * C stands for Charlton.
# * Q stands for Queenstown.
# 

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


train_df['Embarked'].isnull().value_counts()


# As we can see, there are two passengers whose embarking station isn't mantioned. As max passengers embarked at Southampton, we can make a brute assumption that these 2 passengers may have embarked at Southhampton. Let us replace NaN with S.

# In[ ]:


train_df['Embarked'].replace(np.nan,'S',inplace=True)


# In[ ]:


train_df['Embarked'].isnull().any()


# Let us visualise the data of the embarking stations.

# In[ ]:


sns.catplot('Embarked',data=train_df,kind='count',aspect=2,height=6)


# From the above data, it can be seen that maximum passengers embarked in Southampton followed by Charlton and Queenstown. Let us now check the number of survivors from each of the embarking stations.

# In[ ]:


ax=sns.catplot('Embarked',data=train_df,kind='count',hue='Survived',aspect=2,height=6)
ax.set_xticklabels(['Southampton','Charlton','Queenstown'])


# ## 3. Cabin

# This particular feature is extremely problematic due to very high number of missing data. It'll be very difficult to fill null values with any cabin data. It is impossible to try to model this particular data. Hence, it will be wise to simply delete this feature. Inputting a feature with high missing values may induce high bias in our model. Never the less, let us check the correlation with surivival for the data we have.

# In[ ]:


deck=train_df['Cabin']


# In[ ]:


deck.isna().value_counts()


# In[ ]:


deck=deck.dropna()


# In[ ]:


levels=[]
for level in deck:
    levels.append(level[0])
    


# In[ ]:


cabin_df=pd.DataFrame(levels,columns=['Cabin level'])


# In[ ]:


cabin_df.head()


# In[ ]:


cabin_df['Cabin level'].value_counts().sort_values(ascending=False)


# In[ ]:


sns.catplot('Cabin level',data=cabin_df,kind='count',aspect=2,height=6,palette='summer_d')


# From the available data, we have maximum passengers from level C followed by B.

# In[ ]:


train_temp=train_df.copy()


# In[ ]:


train_temp.isna().any()


# In[ ]:


train_temp=train_temp.dropna(axis=0)


# In[ ]:


train_temp.reset_index(inplace=True,drop=True)


# In[ ]:


train_temp['Level']=cabin_df['Cabin level']


# In[ ]:


sns.catplot('Level',kind='count',hue='Person',aspect=2,height=6,data=train_temp)


# In[ ]:


sns.catplot('Level',kind='count',hue='Survived',aspect=2,height=6,data=train_temp)


# In[ ]:


sns.factorplot('Survived',
               data=train_temp,
               kind='count',col='Level',
               col_wrap=4,height=4,aspect=1)


# Hence, from the above, we can understand that the level location did have a significant impact on whether the person survived. Higher levels such as level A,B, C and D had good number of survivors. The lower decks had much fewer survivors. However, we can't add it to the model since maximum entries have missing entries. 

# ## 4. Passenger class

# Let us visualise the data for passenger class.

# In[ ]:


train_df['Pclass'].value_counts()


# In[ ]:


sns.catplot('Pclass',data=train_df, kind='count')


# The above data shows that 3rd class passengers were maximum followed by 1st and 2nd class passengers. Let us check if Pclass has any correlation with survival.

# In[ ]:


sns.catplot('Pclass',data=train_df,kind='count',hue='Survived',aspect=2,height=6,palette='winter')


# It could be said that 1st class passengers could have got a preferential treatment during evacuation process as a result of which, more 1st class passengers survived than ones that didn't. In 2nd class passengers, the survivors and deaths were nearly the same. 
# 
# As expected, the 3rd class passengers must have been evacuated in the end and hence, couldn't be saved. It could also be due to the fact that the population of 3rd class passengers were highest. Hence, evacuating the passengers couldn't be completed due to the paucity of time.

# To get a better understanding of the survival trend, let us make a lm plot to see the relation between age and survival with a hue of passenger class.

# In[ ]:


sns.lmplot('Age','Target',data=train_df,aspect=2,height=6,hue='Pclass')


# As it can be observed, the probability of survival from the same age group on class 1 was higher than class 3. Hence, it can be said that Passenger Class definitely has a good correlation to survival and will be an important feature of our dataset.

# Let us check how the survival of children, male and female adults vary.

# In[ ]:


sns.catplot('Pclass','Target',data=train_df,hue='Person',kind='point',aspect=2,height=6)


# Here, we can see that for children, max survivors were from 2nd class. For males, 3rd class survivors were higher than 2nd class. This is interesting to note as it was unexpected.

# ## 5. Siblings and spouses

# Let us check if having any siblings and spouses aboard had any relation to survival.

# In[ ]:


train_df['SibSp'].value_counts()


# In[ ]:


sns.catplot('SibSp',data=train_df,kind='count')


# As it can be seen, maximum number of passengers were actually alone. About 209 passengers had either a spouse or a sibling along with them.
# 
# Let us now check if this feature realates to survival.
# 

# In[ ]:


sns.catplot('SibSp',data=train_df,kind='count',hue='Target',aspect=2,height=6)


# As is expected, many of the single passengers survived. This could be due to the fact that weren't reqiured to wait for any of the family members to deboard the ship and could be far easily be vacated. Fair number of people with one family member survived since it is realtively easier to find one family member instead of multiple family members.
# 
# Hence, it can be said that as family members increases, survival reduced.
# Let us check this hypothesis through a lmplot.

# In[ ]:


sns.lmplot('SibSp','Target',data=train_df)


# The above plot proves our hypothesis that survival chances reduce as number of family members increase.

# ## 6. Sex

# Let us check if gender has any role to play in survival.

# In[ ]:


train_df['Sex'].value_counts()


# In[ ]:


sns.catplot('Sex',data=train_df,kind='count',hue='Survived')


# Most of the female passengers survived the accident as compared to males. This indicates that female passengers were evacuated first and hence, they survived. Let us check this with another plot.

# In[ ]:


sns.lmplot('Age','Target',data=train_df,hue='Sex',aspect=2,height=6)


# The above plot clearly indicates that with age, female passengers actually had a higher probability of survival.
# 
# This indicates that aged women were evacuated first and then the middle age and young ones.
# 
# For males, it was the opposite. Younger males had higher chance of survival.

# ## 7. Parents/ Children

# Let us check if having any parents or children played any important role in survival of the passengers.

# In[ ]:


train_df['Parch'].unique()


# In[ ]:


train_df['Parch'].value_counts()


# In[ ]:


sns.catplot('Parch',data=train_df,kind='count')


# As we can see, maximum passengers did not have any children aboard. Let us see if this has any relation with survival.

# In[ ]:


sns.catplot('Parch',data=train_df,kind='count',hue='Survived')


# As expected, many passengers who did not have to worry about any children survived. There are about 50% survivors amongst parents with 1 and 2 children aboard.

# During the data wrangling process, we can combine the Parch and SibSp columns as with relatives and without relatives to simplify the dataframe and prevent much data leakage. This process is shown below.

# In[ ]:


train_df['Total relatives']=train_df['Parch']+train_df['SibSp']


# In[ ]:


train_df.head()


# Let us check how total relative numbers change the probability of survival.

# In[ ]:


ax=sns.catplot('Total relatives','Target',kind='point',data=train_df,aspect=2,height=6)
ax.set_ylabels('Survival probability')


# As we can see from the abpve graph, if there are 0-3 relatives, chances of survival is high. However, anything more than that leads to a drop in survival rate. Hence, we can consider the number of relatives to be an important factor.

# ## 8. Checking correlation of the various features

# Let us check the correlation of the various features with each other using a heatmap.

# In[ ]:


correlations=train_df.corr()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(correlations,annot=True,cmap='summer')


# ## 9. Data Wrangling

# We have visualised all the features in the previous sections. Now, we shall make changes to the dataframe to keep only the relevant features. We may modify or drop features such that it shall give us a good model prediction.

# In[ ]:


train_mod=train_df.copy()


# In[ ]:


train_mod.columns.isna()


# In[ ]:


train_mod.head()


# We can drop columns such as *PassengerID,Survived, Name,Sex,SibSp,Parch,Cabin and Ticket*

# In[ ]:


train_mod.drop(['PassengerId','Survived','Sex','Name','Ticket','Cabin','Parch','SibSp'],axis=1,inplace=True)


# In[ ]:


train_mod.head()


# Let us encode the person column as :
# 
# 1: Male
# 
# 2: Female
# 
# 3: Child

# In[ ]:


train_mod['Person']=train_mod['Person'].map({'male':1,'female':2,'child':3})
train_mod.head()


# The only categorical data we have in the above data is the Embarked column. Let us one-hot encode the Embarked column.

# In[ ]:


temp=pd.get_dummies(train_mod['Embarked'])


# In[ ]:


train_mod=train_mod.merge(temp,on=train_mod.index)
train_mod.head()


# In[ ]:


train_mod.drop(['key_0','Embarked'],axis=1,inplace=True)


# In[ ]:


train_mod.head()


# To streamline the effect of age, we group the ages as follows:
# * Ages<=16 : 0
# * Ages <=32 & >16 : 1
# * Ages <=48 & >32 : 2
# * Ages <=64 & >48 : 3
# * Ages >64 : 4
# 
# 
# 
# 

# In[ ]:


train_mod.loc[train_mod['Age']<=16,'Age band']=0
train_mod.loc[(train_mod['Age']>16) & (train_mod['Age']<33),'Age band']=1
train_mod.loc[(train_mod['Age']>32) & (train_mod['Age']<49),'Age band']=2
train_mod.loc[(train_mod['Age']>48) & (train_mod['Age']<65),'Age band']=3
train_mod.loc[train_mod['Age']>64,'Age band']=4
train_mod.head()


# In[ ]:


train_mod.drop('Age',axis=1,inplace=True)


# Let us take a look at the fare paid by the customers.

# In[ ]:


plt.figure(figsize=(10,8))
plt.boxplot(train_df['Fare'])
plt.ylabel('Fare value')


# From the above boxplot, it is seen that the median fare is around 14 pounds while the high prices could go as high as 500 plus pounds. Hence, we need to divide these fares into fare bands. This will help take care of the non linear distribution of the fare and the presence of so many outliers.

# In[ ]:


plt.figure(figsize=(10,7))
sns.kdeplot(train_mod['Fare'],shade=True)


# The division of fares maybe done as follows:
# 
# * 0-50: 1 (General class)
# * 50-100: 2 (Economy class)
# * 100-200: 3 (Semi-premium)
# * 200+ : 4 (Premium)
# 

# In[ ]:


train_mod.loc[(train_mod['Fare']<51),'Fare band']=1
train_mod.loc[(train_mod['Fare']>50)&(train_mod['Fare']<101),'Fare band']=2
train_mod.loc[(train_mod['Fare']>100)&(train_mod['Fare']<201),'Fare band']=3
train_mod.loc[(train_mod['Fare']>200),'Fare band']=4
train_mod.head()


# In[ ]:


train_mod.drop('Fare',axis=1,inplace=True)


# In[ ]:


train_mod.head()


# In[ ]:


target_df=pd.DataFrame(columns=['Target'])
target_df['Target']=train_mod['Target']
target_df.head()


# In[ ]:


target_df['Target'].value_counts()


# In[ ]:


train_mod.drop('Target',axis=1,inplace=True)


# In[ ]:


train_mod.head()


# The above data is now preprocessed and can be used for Machine Learning.

# # Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_mod,target_df,test_size=0.2,shuffle=True,random_state=365)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)


# In[ ]:


y_knn_pred=knn.predict(X_test)


# In[ ]:


print('Score with KNN on test dataset:{}'.format(np.round(knn.score(X_test,y_test) *100,2)))
print('Score with KNN on train dataset:{}'.format(np.round(knn.score(X_train,y_train) *100,2)))


# In[ ]:


from sklearn.metrics import confusion_matrix
cnf_knn=confusion_matrix(y_knn_pred,y_test)


# In[ ]:


sns.heatmap(cnf_knn,annot=True,cmap='winter')


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


reg_log=LogisticRegression()


# In[ ]:


reg_log.fit(X_train,y_train)


# In[ ]:


y_log_pred=reg_log.predict(X_test)


# In[ ]:


print('Score with Logistic regression on test dataset:{}'.format(np.round(reg_log.score(X_test,y_test) *100,2)))
print('Score with Logistic regression on train dataset:{}'.format(np.round(reg_log.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_reg=confusion_matrix(y_test,y_log_pred)
sns.heatmap(cnf_reg,annot=True,cmap='gnuplot')


# In[ ]:


y_lr=reg_log.fit(X_train,y_train).decision_function(X_test)


# In[ ]:


from sklearn.metrics import roc_curve,auc,precision_recall_curve

fpr,tpr,_=roc_curve(y_test,y_lr)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],linestyle='--')
auc_reg=auc(fpr,tpr).round(2)
plt.title('ROC curve with AUC={}'.format(auc_reg))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# The above figure shows there is good amount of area under the curve.

# In[ ]:


precision,recall,threshold=precision_recall_curve(y_test,y_lr)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with Logistic Regression')
plt.xlabel('Precision')
plt.ylabel('Recall')


# Hence, an optimum precision vs recall will be about 0.83 and 0.65 respectively

# ###  SVC

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC(gamma=1e-07,C=1e9)
svc.fit(X_train,y_train)


# In[ ]:


y_svc_pred=svc.predict(X_test)


# In[ ]:


print('Score with SVC on test dataset:{}'.format(np.round(svc.score(X_test,y_test) *100,2)))
print('Score with SVC on train dataset:{}'.format(np.round(svc.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_reg=confusion_matrix(y_test,y_svc_pred)
sns.heatmap(cnf_reg,annot=True,cmap='summer',fmt='g')


# In[ ]:


y_svc=svc.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_svc)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],linestyle='--')
auc_reg=auc(fpr,tpr).round(2)
plt.title('ROC curve with AUC={}'.format(auc_reg))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# In[ ]:


precision,recall,threshold=precision_recall_curve(y_test,y_svc)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with SVC')
plt.xlabel('Precision')
plt.ylabel('Recall')


# Both the area under curve score and precision-recall curve are very close to logistic regression curves. Hence, their performances are very identical.

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


rfc=RandomForestClassifier()
param_grid={'n_estimators':[5,7,9,10], 'max_depth':[5,7,9,10]}
grid_search=GridSearchCV(rfc,param_grid,scoring='roc_auc')


# In[ ]:


grid_result=grid_search.fit(X_train,y_train)


# In[ ]:


grid_result.best_params_


# In[ ]:


grid_result.best_score_


# In[ ]:


y_rfc_pred=grid_result.predict(X_test)


# In[ ]:


print('Score with RFC on test dataset:{}'.format(np.round(grid_result.score(X_test,y_test) *100,2)))
print('Score with RFC on train dataset:{}'.format(np.round(grid_result.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_rfc=confusion_matrix(y_test,y_rfc_pred)
sns.heatmap(cnf_rfc,annot=True,fmt='g')


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)


# In[ ]:


y_dtc_pred=dtc.predict(X_test)


# In[ ]:


print('Score with DTC on test dataset:{}'.format(np.round(dtc.score(X_test,y_test) *100,2)))
print('Score with DTC on train dataset:{}'.format(np.round(dtc.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_dtc=confusion_matrix(y_dtc_pred,y_test)
sns.heatmap(cnf_dtc,annot=True,cmap='gist_ncar')


# ## Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


sgd=SGDClassifier(max_iter=10)
sgd.fit(X_train,y_train)


# In[ ]:


y_sgd_pred=sgd.predict(X_test)


# In[ ]:


print('Score with SGD on test dataset:{}'.format(np.round(sgd.score(X_test,y_test) *100,2)))
print('Score with SGD on train dataset:{}'.format(np.round(sgd.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_sgd=confusion_matrix(y_sgd_pred,y_test)
sns.heatmap(cnf_sgd,annot=True)


# In[ ]:


y_sgd=sgd.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_sgd)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],linestyle='--')
auc_reg=auc(fpr,tpr).round(2)
plt.title('ROC curve with AUC={}'.format(auc_reg))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# In[ ]:


precision,recall,threshold=precision_recall_curve(y_test,y_sgd)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with SGD')
plt.xlabel('Precision')
plt.ylabel('Recall')


# As we can see from the above curve, the optimum precision of this model is quite low (0.45) while it's recall is high. The other models fared better comparatively.

# ## Naive Bayes 

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


nb=GaussianNB()
nb.fit(X_train,y_train)


# In[ ]:


y_nb_pred=nb.predict(X_test)


# In[ ]:


print('Score with Naive Bayes on test dataset:{}'.format(np.round(nb.score(X_test,y_test) *100,2)))
print('Score with Naive Bayes on train dataset:{}'.format(np.round(nb.score(X_train,y_train) *100,2)))


# In[ ]:


cnf_nb=confusion_matrix(y_test,y_nb_pred)
sns.heatmap(cnf_nb,annot=True,cmap='viridis')


# ## XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb.score(X_train,y_train)


# In[ ]:


y_pred_xgb=xgb.predict(X_test)
xgb.score(X_test,y_test)


# In[ ]:


conf_mat_xgb=confusion_matrix(y_test,y_pred_xgb)
sns.heatmap(conf_mat_xgb,annot=True,fmt='g')


# ## LightGBM

# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgb=LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgb.fit(X_train,y_train)
lgb.score(X_train,y_train)


# In[ ]:


y_pred_lgb=lgb.predict(X_test)
lgb.score(X_test,y_test)


# In[ ]:


conf_mat_lgb=confusion_matrix(y_pred_lgb,y_test)
sns.heatmap(conf_mat_lgb,annot=True,fmt='g',cmap='gnuplot')


# # Comparison of training scores of the models

# In[ ]:


score_df=pd.DataFrame(columns=['Model name','Train score'])


# In[ ]:


models=['KNN','SVC','Naive Bayes','Decision Tree','Logistic Regression','Random Forest','SGD']
score_df['Model name']=models


# In[ ]:


scores=[np.round(knn.score(X_train,y_train) *100,2),
        np.round(svc.score(X_train,y_train) *100,2),
        np.round(nb.score(X_train,y_train) *100,2),
        np.round(dtc.score(X_train,y_train) *100,2),
        np.round(reg_log.score(X_train,y_train) *100,2),
        np.round(grid_result.score(X_train,y_train) *100,2),
        np.round(sgd.score(X_train,y_train) *100,2)]
score_df['Train score']=scores


# In[ ]:


score_df


# As we can see, Random forest classifier with max_depth=5 and n_estimators=10 gives the best train scores followed by Decision tree. Hence, tree based models have perfored better.

# Let us now perform a K-cross fold validation to prevent any overfitting issues.

# # K-Fold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


rfc_opt=RandomForestClassifier(max_depth=5,n_estimators=10)


# Let us perform a 5 fold cross validation.

# In[ ]:


score_cv=cross_val_score(rfc_opt,X_train,y_train,cv=5,scoring='accuracy')


# In[ ]:


cv_df=pd.DataFrame(columns=['Cross validated score'])
cv_scores=np.round(score_cv*100,2)


# In[ ]:


cv_df['Cross validated score']=cv_scores
cv_df.index=cv_df.index + 1


# In[ ]:


cv_df


# The above scores are far more realistic than the high result of approximately 90% shown.
# 
# Let us analyse the mean and standard deviation of all the cross validated scores.

# In[ ]:


print('Cross validated mean score: {}'.format(cv_scores.mean()))
print('Cross validated score standard deviation: {}'.format(np.round(cv_scores.std(),2)))


# Hence, the standard deviation is at an appreciated low value. This means our cross validation scores are nearly similar for each fold. 

# ## We can finalise that we shall be using random forest classifier on our final test dataset.

# # B) Testing phase

# In[ ]:


test_df=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_df.head()


# ## We shall perform the same data wrangling and preprocessing which we have performed on the training dataset for the model to predict accurately.

# In[ ]:


test_df.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


test_df['Total relatives']=test_df['SibSp']+test_df['Parch']
test_df.drop(['SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


embarks=pd.get_dummies(test_df['Embarked'])


# In[ ]:


test_df=test_df.merge(embarks,on=test_df.index)


# In[ ]:


test_df.drop(['key_0','Embarked'],axis=1,inplace=True)
test_df.head()


# In[ ]:


test_df['Person']=test_df[['Age','Sex']].apply(child,axis=1)
test_df.head()


# In[ ]:


test_df['Person']=test_df['Person'].map({'male':1,'female':2,'child':3})
test_df.head()


# In[ ]:


test_df['Age'].median()


# In[ ]:


test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())


# In[ ]:


test_df.loc[test_df['Age']<=16,'Age band']=0
test_df.loc[(test_df['Age']>16) & (test_df['Age']<33),'Age band']=1
test_df.loc[(test_df['Age']>32) & (test_df['Age']<49),'Age band']=2
test_df.loc[(test_df['Age']>48) & (test_df['Age']<65),'Age band']=3
test_df.loc[test_df['Age']>64,'Age band']=4
test_df.head()


# In[ ]:


test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())


# In[ ]:


test_df.isna().any()


# In[ ]:


test_df.loc[(test_df['Fare']<51),'Fare band']=1
test_df.loc[(test_df['Fare']>50)&(test_df['Fare']<101),'Fare band']=2
test_df.loc[(test_df['Fare']>100)&(test_df['Fare']<201),'Fare band']=3
test_df.loc[(test_df['Fare']>200),'Fare band']=4
test_df.head()


# In[ ]:


test_df.drop(['Sex','Age','Fare'],axis=1,inplace=True)
test_df.head()


# As we can see, the test data is now preprocessed and can be used for machine learning.

# ## Applying machine learning to the test dataframe

# In[ ]:


train_mod.head()


# In[ ]:


test_df[train_mod.columns].head()


# In[ ]:


rfc_opt.fit(X_train,y_train)


# In[ ]:


y_final_predictions=rfc_opt.predict(test_df[train_mod.columns])


# In[ ]:


final_predictions_df=pd.DataFrame(columns=['PassengerId','Survived'])
final_predictions_df['PassengerId']=test_df['PassengerId']


# In[ ]:


final_predictions_df['Survived']=y_final_predictions


# In[ ]:


final_predictions_df.isna().any()


# In[ ]:


final_predictions_df


# In[ ]:


y_xgb_preds=xgb.predict(test_df[train_mod.columns])


# In[ ]:


xgb_predictions_df=pd.DataFrame(columns=['PassengerId','Survived'])
xgb_predictions_df['PassengerId']=test_df['PassengerId']
xgb_predictions_df['Survived']=y_xgb_preds


# In[ ]:


xgb_predictions_df.head()


# In[ ]:


y_lgb_preds=lgb.predict(test_df[train_mod.columns])


# In[ ]:


lgb_predictions_df=pd.DataFrame(columns=['PassengerId','Survived'])
lgb_predictions_df['PassengerId']=test_df['PassengerId']
lgb_predictions_df['Survived']=y_lgb_preds


# In[ ]:


lgb_predictions_df.head()


# In[ ]:


lgb_predictions_df.to_csv('LGBM_predictions.csv',index=False)

