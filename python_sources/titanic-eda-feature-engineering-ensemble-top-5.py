#!/usr/bin/env python
# coding: utf-8

# ## Titanic : Survival prediction
# 
# RMS Titanic was a British passenger ship, which tragically sunk on 15th April 1912 after colliding with an iceberg during it's maiden journey from Southampton to New York. It was one of the deadliest peacetime marine disasters of modern history.
# 
# It was the largest passenger ship at it's time and was considered "unsinkable". There was a shortage of lifeboats  onboard, one of the main reasons which caused the deaths of 1502 out of 2224 passenger and crew. We are provided a dataset with the particulars of the passengers onboard along with the flag if they survived or not.
# 
# Our aim is to study this dataset and find out if it was more than chance that determined if a passenger survived or not. If there is a pattern we need to build a predictive model for the chance of survival of a passenger with a particular set of features(i.e. age, gender, economic status etc).
# 
# ### Edit- For some interesting hyperparameter tuning techniques, please check out the notebook in the link below:-
# https://www.kaggle.com/ankur123xyz/advanced-hyperparameter-tuning-techniques
# 
# 
# Let's move to the dataset and start by importing the requisite packages.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# We would import the train and test set into different datasets.

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# Getting a glimpse of the training dataset using the head function.

# In[ ]:


train.head()


# We move on to checking the number of rows and columns in the train and test set.
# <br>There would be an extra feature in the training set that would be the "Survived" flag.

# In[ ]:


print("Rows and columns in the training set are:-",train.shape[0],",",train.shape[1])
print("Rows and columns in the test set are:-",test.shape[0],",",test.shape[1])


# We will have a look at the datatypes for all the features.
# What we get is a mix of integers, floats and object data types. We will do some type conversions later.

# In[ ]:


train.info()


# Based on the sample data we saw earlier and the datatypes visibile here, we can go ahead and convert Pclass to categorical type.

# In[ ]:


train =train.astype({"Pclass":"category"})
test =test.astype({"Pclass":"category"})


# Now let us start with some visualtizations of the data to study it better.
# 
# 64.8% of the passengers were male whereas 35.2% were female.

# In[ ]:


plt.pie(train["Sex"].value_counts(),explode=[0,.1],startangle=90,labels=train["Sex"].value_counts().index,shadow=True,autopct='%1.1f%%')


# We should check for gender having any effect on the survival rate of the passengers.
# 
# Turns out the survival rate for females was much higher at 74% whereas for males it was only 19%. Possible reason could be preference given while boarding the lifeboats.

# In[ ]:


f, ax = plt.subplots(figsize=(8,10))
plot = sns.countplot(x="Sex",hue="Survived",data=train)
bars = ax.patches
half = int(len(ax.patches)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left,right in zip(left_bars,right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    height_t = height_l + height_r
    ax.text(left.get_x()+left.get_width()/2,height_l+5,'{0:.0%}'.format(height_l/height_t),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r+5,'{0:.0%}'.format(height_r/height_t),ha="center")
    ax.text(left.get_x()+left.get_width()/2,height_l/2,'{0}'.format(height_l),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r/2,'{0}'.format(height_r),ha="center")
ax.set_title("Survival by Gender")
ax.set_ylabel("Count of Passengers")
ax.set_xlabel("Gender")
ax.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])


# If we plot the survival rate according to age groups the higest survival rate is for children(0-18 Age group). We can use these age bin further for our analysis as part of feature transformation.

# In[ ]:


train["Age_bin"] = pd.cut(train.Age,[0,18,30,40,50,60,100])
f, ax = plt.subplots(figsize=(8,8))
plot = sns.countplot(x="Age_bin",hue="Survived",data=train)
bars = ax.patches
half = int(len(ax.patches)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left,right in zip(left_bars,right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    height_t = height_l + height_r
    ax.text(left.get_x()+left.get_width()/2,height_l+2,'{0:.0%}'.format(height_l/height_t),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r+2,'{0:.0%}'.format(height_r/height_t),ha="center")
    ax.text(left.get_x()+left.get_width()/2,height_l/2,'{0}'.format(height_l),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r/2,'{0}'.format(height_r),ha="center")
ax.set_title("Survival by Age")
ax.set_ylabel("Count of Passengers")
ax.set_xlabel("Age Group")
ax.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])


# The passengers in the 2nd and 3rd classes were less fortunate than the ones from 1st class. 63% of the 1st class passengers survived while only 47% and 24% of the passengers from 2nd and 3rd class survived respectively.

# In[ ]:


f, ax = plt.subplots(figsize=(8,8))
plot = sns.countplot(x="Pclass",hue="Survived",data=train)
bars = ax.patches
half = int(len(ax.patches)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left,right in zip(left_bars,right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    height_t = height_l + height_r
    ax.text(left.get_x()+left.get_width()/2,height_l+2,'{0:.0%}'.format(height_l/height_t),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r+2,'{0:.0%}'.format(height_r/height_t),ha="center")
    ax.text(left.get_x()+left.get_width()/2,height_l/2,'{0}'.format(height_l),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r/2,'{0}'.format(height_r),ha="center")
ax.set_title("Survival by Class")
ax.set_ylabel("Count of Passengers")
ax.set_xlabel("Class")
ax.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])


# If we look at the survival rates as per the port of embarkation, the highest rate is for those who embarked at Cherbourg. But if we dig deeper in to the class split according to port of embarkation, we see that ~50% of the passengers who boarded at Cherbourg were 1st class passengers, whereas for Queenstown and Southampton it was 2.6% and 19.7%. We have already seen earlier that 1st class passengers had a considerably higher chance of survival.

# In[ ]:


f, ax = plt.subplots(figsize=(8,8))
plot = sns.countplot(x="Embarked",hue="Survived",data=train)
bars = ax.patches
half = int(len(ax.patches)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left,right in zip(left_bars,right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    height_t = height_l + height_r
    ax.text(left.get_x()+left.get_width()/2,height_l+2,'{0:.0%}'.format(height_l/height_t),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r+2,'{0:.0%}'.format(height_r/height_t),ha="center")
    ax.text(left.get_x()+left.get_width()/2,height_l/2,'{0}'.format(height_l),ha="center")
    ax.text(right.get_x()+right.get_width()/2,height_r/2,'{0}'.format(height_r),ha="center")
ax.set_title("Survival by Port of Embarkation")
ax.set_ylabel("Count of Passengers")
ax.set_xlabel("Port")
ax.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])


# In[ ]:


pd.crosstab(train.Pclass,train.Embarked).apply(lambda c: c/c.sum(), axis=0).style.format("{:.2%}")


# Since Cabin has ~78% missing values in both training and test set, we can go ahead and drop this feature. 
# 
# Before we start operations on the dataset we should combine the train and test datasets as we would be using the running the test data through the model trained on the training data.

# In[ ]:


y=train["Survived"].values
test_index = len(y)-1
data = pd.concat([train,test],sort=False).reset_index(drop=True)
data.info()


# The Age feature will have to be imputed wherever missing, since we could see some co-relation between survival rates and Age earlier.
# 
# We can extract the salutation from the name and use that to get an Age to impute basis a measure of central tendency.
# 
# We map the less frequent salutations to the frequent onces.

# In[ ]:


data["Title"]= data["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data.replace({'Title': mapping}, inplace=True)


# Family Size is a useful feature which can give us the size of the group travelling together.
# We create a dictionary to map the Age basis the mean for a particular Title and the Family Size.
# 
# We create the family size feature from the Sibling/Spouse and Parent/Child features.

# In[ ]:


data["Fam_size"] = data["SibSp"]+data["Parch"]+1


def mapping(title,tick_size):
    for (ind,val) in zip(age_impute,age_impute.values()):
        if((ind[0]==title) & (ind[1]==tick_size)):
            return val
        

age_impute=data.groupby(["Title","Fam_size"])["Age"].median().astype("float64").to_dict()


# We impute the Age category basis the dicionary that we have created.

# In[ ]:


data.loc[data["Age"].isnull(),"Age"]= data.loc[data["Age"].isnull(),["Title","Fam_size"]].apply(lambda x: mapping(x[0],x[1]),axis=1)


# We bin the Ages for the passengers for the whole dataset and then encode it since there is a ordinal relation between the Age groups.

# In[ ]:


data["Age_bin"] = pd.qcut(data["Age"],10,duplicates="drop")
label = LabelEncoder()
data['Age_bin'] = label.fit_transform(data['Age_bin'].astype(str))


# We can make use of the last names since survival rate of a family is interlinked. The logic for the below is based on the logic from below kernel.Thanks to S.Xu
# https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever
# 
# Also we have imputed Fare for the family as of now grouped by class.

# In[ ]:


data["Fare"].fillna(data[~data["Fare"].isnull()].groupby("Pclass")["Fare"].mean()[3],inplace=True)
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])


data['Family_Survival'] = 0.5

for grp, grp_data in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_data) != 1):
        for ind, row in grp_data.iterrows():
            vmax = grp_data.drop(ind)['Survived'].max()
            vmin = grp_data.drop(ind)['Survived'].min()
            passid = row['PassengerId']
            if (vmax == 1.0):
                data.loc[data['PassengerId'] == passid, 'Family_Survival'] = 1
            elif (vmin==0.0):
                data.loc[data['PassengerId'] == passid, 'Family_Survival'] = 0
                
for grp, grp_data in data.groupby('Ticket'):
    if (len(grp_data) != 1):
        for ind, row in grp_data.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                vmax = grp_data.drop(ind)['Survived'].max()
                vmin = grp_data.drop(ind)['Survived'].min()
                passid = row['PassengerId']
                if (vmax == 1.0):
                    data.loc[data['PassengerId'] == passid, 'Family_Survival'] = 1
                elif (vmin==0.0):
                    data.loc[data['PassengerId'] == passid, 'Family_Survival'] = 0                


# Since we have 2 passenger for whom the Embarked location is missing, we impute it using the mode strategy.
# For the passenger whose Fare is missing we impute it with the mean fare of the 3rd Class passengers since the passenger had a 3rd Class ticket.
# 
# Also we divide the fare by the family size since the Fare is for the entire ticket(which may have multiple passengers) and not only for that particular passenger.

# In[ ]:


data["Embarked"].fillna(data["Embarked"].mode()[0],inplace=True)
data["Fare"]=data["Fare"]/data["Fam_size"]
data["Fare"].fillna(data[~data["Fare"].isnull()].groupby("Pclass")["Fare"].mean()[3],inplace=True)


# We can bin the Fares also in to 13 equal bins. We then encode it to preserve the orinal nature of Fare.

# In[ ]:


data["Fare_bin"] = pd.qcut(data["Fare"],13)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data['Fare_bin'] = label.fit_transform(data['Fare_bin'].astype(str))


# Finally let's drop all the features that we wouldnt't be using for our modelling.

# In[ ]:


data.drop(["Name","SibSp","Parch","Ticket","Age","Fare","PassengerId","Last_Name","Survived","Title","Cabin"],inplace=True,axis=1)


# We need to check for missing values in our dataset. Seems we are all good to go as we don't have null values.

# In[ ]:


data.isnull().sum()/data.shape[0]*100


# Onehot encoding the categorical variables

# In[ ]:


data = pd.get_dummies(data,drop_first=True)


# We split the dataframe to get our original passenger data from the training set and the test set.

# In[ ]:


train_1 = data.iloc[:test_index+1,:]
test_1 = data.iloc[test_index+1:,:]


# We further split the training set in to a train and test set to validate our model.

# In[ ]:



X_train,X_test,y_train,y_test = train_test_split(train_1,y,test_size=0.2, random_state=42)


# Coming to the modeling part. We first scale the data using standard scaler.
# We use grid search with stratified kfold validation for 9 algorithms.
# We get the scores from the cross validation for all these models and run a prediction on the test data from our train_test_split.
# For stacking we get the accuracy based on fitting the train and test set.

# In[ ]:


sk_fold = StratifiedKFold(10,shuffle=True, random_state=42)
sc =StandardScaler()
X_train= sc.fit_transform(X_train)
X_train_1= sc.transform(train_1.values)
X_test= sc.transform(X_test)
X_submit= sc.transform(test_1.values)
log_reg = LogisticRegression()
ran_for  = RandomForestClassifier()
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier(n_estimators=100)
hist_grad_boost = HistGradientBoostingClassifier()
knn = KNeighborsClassifier()
tree= DecisionTreeClassifier()
svc = SVC()
xgb = XGBClassifier()
clf = [("Logistic Regression",log_reg,{"penalty":['l2'],"C":[100, 10, 1.0, 0.1, 0.01]}),       ("Support Vector",svc,{"kernel": ["rbf"],"gamma":[0.1, 1, 10, 100],"C":[0.1, 1, 10, 100, 1000]}),       ("Decision Tree", tree, {}),       ("Random Forest",ran_for,{"n_estimators":[100],"random_state":[42],"min_samples_leaf":[5,10,20,40,50],"bootstrap":[False]}),       ("Adapative Boost",ada_boost,{"n_estimators":[100],"learning_rate":[.6,.8,1]}),       ("Gradient Boost",grad_boost,{}),       ("Histogram GB",hist_grad_boost,{"loss":["binary_crossentropy"],"min_samples_leaf":[5,10,20,40,50],"l2_regularization":[0,.1,1]}),       ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3,4,5],"learning_rate":[.01,.1,.2],"subsample":[.8],"colsample_bytree":[1],"gamma":[0,1,5],"lambda":[.01,.1,1]}),      ("K Nearest",knn,{"n_neighbors":[3,5,8],"leaf_size":[25,30,35]})]
stack_list=[]
train_scores = pd.DataFrame(columns=["Name","Train Score","Test Score"])
i=0
for name,clf1,param_grid in clf:
    clf = GridSearchCV(clf1,param_grid=param_grid,scoring="accuracy",cv=sk_fold,return_train_score=True)
    clf.fit(X_train,y_train.reshape(-1,1))
    y_pred = clf.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #train_scores.loc[i]= [name,cross_val_score(clf,X_train,y_train,cv=sk_fold,scoring="accuracy").mean(),(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]
    train_scores.loc[i]= [name,clf.best_score_,(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]
    stack_list.append(clf.best_estimator_)
    i=i+1
    
est = [("dec_tree",stack_list[2]),("ran_for",stack_list[3]),("ada_boost",stack_list[4]),("grad_boost",stack_list[5]),("hist_grad_boost",stack_list[6]),("svc",stack_list[1]),("lr",stack_list[0]),("knn",stack_list[8])]
sc = StackingClassifier(estimators=est,final_estimator = stack_list[2],cv=sk_fold,passthrough=False)
sc.fit(X_train,y_train)
y_pred = sc.predict(X_test)
cm1 = confusion_matrix(y_test,y_pred)
y_pred_train = sc.predict(X_train)
cm2 = confusion_matrix(y_train,y_pred_train)
train_scores.append(pd.Series(["Stacking",(cm2[0,0]+cm2[1,1,])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]),(cm1[0,0]+cm1[1,1,])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])],index=train_scores.columns),ignore_index=True)


# We fit and predict on the best XGB model that we derived based on the scores.

# In[ ]:


stack_list[7].fit(X_train_1,y)
y_submit = stack_list[7].predict(X_submit)
submit = pd.DataFrame({
        "PassengerId": test.PassengerId,
        "Survived": y_submit
    })


# Exporting the data to submit.

# In[ ]:


submit.PassengerId = submit.PassengerId.astype(int)
submit.Survived = submit.Survived.astype(int)
submit.to_csv("titanic_submit.csv", index=False)
    


# Got a score of 0.80861 on submission. Cheers ! 

# In[ ]:




