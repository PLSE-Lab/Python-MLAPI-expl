#!/usr/bin/env python
# coding: utf-8

#  # 0. Intro

# ** My getting started competition on the Titanic desaster. It is heavily based on:**
# 
# - Udemy course "Data Science and ML Bootcamp", by Jose Portilla. 
# - The great [Yassine Ghouzam notebook](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)

# # 1. Load and import esencial tools
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


import os
print(os.listdir("../input"))

# Reading train data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



# # 2. Data preprocessing

# In[ ]:


train.head(5)


# In[ ]:


# Summarie and statistics
train.describe()


# ## 2.1 Check for missing data

# In[ ]:


sns.heatmap(pd.concat(objs=[train, test], axis=0).reset_index(drop=True).isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


# Missing data
pd.concat(objs=[train, test], axis=0).reset_index(drop=True).isnull().sum()


# # 3. Feature analysis/treatment

# ## 3.2 Outlier detection 
# Tukey method (Tukey JW., 1977) 
# See [Yassine for details](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# 

# In[ ]:


# Outlier detection 
from collections import Counter

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# In[ ]:


train.loc[Outliers_to_drop] # Show the outliers rows


# In[ ]:


## Join train and test data to apply to both the same data preprocessing
train_len = len(train)
df =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[ ]:


# Replace empty  with np.NaNs 
df = df.fillna(np.nan)

# Check for Null values
df.isnull().sum()


# In[ ]:


# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# ## 3.1 Drop obviously useless features

# In[ ]:



df.drop('PassengerId',axis=1,inplace=True) # Obviously PassengerId is also irrelevant.


# ## 3.2 Numerical features (Survived, SibSp, Parch, Age, Fare)

# ### SibSp (# of siblings / spouses aboard the Titanic) 

# In[ ]:



g1 = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6)
g1.set_ylabels("survival probability")


# This calls for a new categorical feature that divedes values 0-2 which have higher probs of surviving.

# ### Parch (# of parents / children aboard the Titanic)

# In[ ]:


g2 = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6)
g2.set_ylabels("survival probability")


# Single ones and large families have a more difficult time surviving! The Parch survival probability distribution has esencially the same shape as the one related to SibSp. For this reason, it make sense to mixed them in so a new feature that separates singles, small and large families as follows:

# In[ ]:


# Create a family size feature
df["Fsize"] = df["SibSp"] + df["Parch"] + 1


# In[ ]:


# Create new categorical feature describing family size
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)
df.head(2)


# ### Age

# In[ ]:


sns.FacetGrid(train, col='Survived').map(sns.distplot, "Age")


# ** We see that when it come to surviving, it is good to be a child and not being old. However, since both distributions are peaked around 25's age, age by itself does not tell you much. 
# 
# The question at this point is how to fill the missing age data. For this purpose let us compare age with other features**

# In[ ]:


sns.factorplot(y="Age",x="Sex",data=df,kind="box")
sns.factorplot(y="Age",x="Pclass", data=df,kind="box")
sns.factorplot(y="Age",x="SibSp", data=df,kind="box")
sns.factorplot(y="Age",x="Parch", data=df,kind="box")


# Age distribution is the same for both sexes. Meanwhile Pclass and SbSp decrease it and SbSp increase it. The next correlation plot show that how are these variables individually correlated!

# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
df["Sex"] = df["Sex"].map({"male": 0, "female":1})
sns.heatmap(df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)


# In absence of more info, we choose to missing Age data with the median age of similar rows according to Pclass, Parch and SibSp.

# In[ ]:


index_NaN_age = list(df["Age"][df["Age"].isnull()].index)
age_med = df["Age"].median()
for i in index_NaN_age :    
    age_pred = df["Age"][(
        (df['SibSp'] == df.iloc[i]["SibSp"]) & 
        (df['Parch'] == df.iloc[i]["Parch"]) & 
        (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        df['Age'].iloc[i] = age_pred
    else :
        df['Age'].iloc[i] = age_med


# Additionally, we create a child categorical feature, because it gives good survival changes

# In[ ]:


#df['Child'] = df['Age'].map(lambda s: 1 if  s <= 8 else 0)


# In[ ]:


# Old approach
#def impute_age(cols):
#    Age = cols[0]
#    Pclass = cols[1]
#    fclass=train[train['Pclass']==1]['Age'].mean()
#    sclass=train[train['Pclass']==2]['Age'].mean()
#    tclass=train[train['Pclass']==3]['Age'].mean()
#    
#    if pd.isnull(Age):
#
#        if Pclass == 1:
#            return fclass
#
#        elif Pclass == 2:
#            return sclass
#
#        else:
#            return tclass
#
#    else:
#        return Age
#train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#train.head(6)


# ### Fare

# In[ ]:


df["Fare"].isnull().sum(),test["Fare"].isnull().sum()


# In[ ]:


sns.factorplot(y="Fare",x="Sex",data=df,kind="box")
sns.factorplot(y="Fare",x="Pclass", data=df,kind="box")
sns.factorplot(y="Fare",x="SibSp", data=df,kind="box")
sns.factorplot(y="Fare",x="Parch", data=df,kind="box")


# Again sex is irrelevant, a reasonable strategy would to fill the single missing data with the median of similar passengers. Because this is rather a continues variable let us look at the distrubion:

# In[ ]:


df["Fare"] = df["Fare"].fillna(df["Fare"].median())

index_NaN_age = list(df["Fare"][df["Fare"].isnull()].index)
age_med = df["Fare"].median()
for i in index_NaN_age :    
    age_pred = df["Fare"][(
        (df['SibSp'] == df.iloc[i]["SibSp"]) & 
        (df['Parch'] == df.iloc[i]["Parch"]) & 
        (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        df['Fare'].iloc[i] = age_pred
    else :
        df['Fare'].iloc[i] = age_med


# In[ ]:


# Explore Fare distribution 
sns.distplot(df["Fare"], label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")


# The Fare distribution is very skewed. This can cause overweight high values in the model, even if it is scaled.
# In this case, it is better to transform it with the log function to reduce this skew.
# 

# In[ ]:


# Log transformation to reduce skewness of Fare distribution
df["Fare"] = df["Fare"].map(lambda i: np.log(i+1))


# In[ ]:


sns.distplot(df["Fare"], color="b", label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")


# Not quite gaussian but at least skewness was reduced. 

# 
# ## 3.2 Categorical values

# ### Sex
# Not much to do with this feature. But at least we can have a look at how it influenciates survival

# In[ ]:


sns.barplot(x="Sex",y="Survived",data=train).set_ylabel("Survival Probability")


# ### Pclass

# In[ ]:


g = sns.factorplot(x="Pclass",y="Survived",hue='Sex',data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# ### Embark

# In[ ]:


df["Embarked"].value_counts()


# In[ ]:


#Fill Embarked NaN  with the most frequent value 'S' 
df["Embarked"] = df["Embarked"].fillna("S")


# In[ ]:


g= sns.factorplot(x="Embarked",y="Survived",data=train,kind="bar", size = 6 )
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Apparently, survival prob is higher from C embarked. Interestingly this is not necessary associated with passenger class as the following plots show

# In[ ]:


g=sns.factorplot("Pclass", col="Embarked",  data=train, size=6, kind="count")
g.despine(left=True)
g = g.set_ylabels("Count")


# In[ ]:


df.head(2)


# In[ ]:


df = pd.get_dummies(df, prefix="Embarked", columns = ["Embarked"])

#for elem in df['Embarked'].unique()[:-1]:
#    df['Embarked_'+str(elem)] = (df['Embarked'] == elem)/1


# In[ ]:


df.head(2)


# ### Cabin

# In[ ]:


df[['Cabin','Ticket']].head()


# This shows that ticket data do not includes cabin data in an obvios way. 
# 
# Since must people usually did not have a cabin, the NaN calues probably mean this. Let us see is bein a cabin of having nan has any impact on survival prop

# In[ ]:


df["Cabin"][df["Cabin"].isnull()]='X'
df["Cabin"]=df["Cabin"].apply(lambda x: x[0])


# In[ ]:


sns.factorplot(x="Cabin",y='Survived',data=df,kind="bar", size = 6).set_ylabels("survival probability")


# In[ ]:


sns.countplot(x="Cabin",data=df)


# In[ ]:


df["Cabin"]=df["Cabin"].apply(lambda x: "CA" if x!='X' else x)


# In[ ]:


df = pd.get_dummies(df, columns = ["Cabin"], prefix="Cabin",drop_first=True)

#for elem in df['Cabin'].unique()[1]:
#    df['Cabin_'+str(elem)] = (df['Cabin'] == elem)/1
#df.drop(['Cabin'],axis=1,inplace=True)


# In[ ]:


df.head(2)


# ### Ticket
# 
# Several tickets have a non-digit prefix, it is of any importance?****

# In[ ]:


Ticket = []
for i in list(df.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
        #Ticket.append('Y')
    else:
        Ticket.append("X")
        
df["Ticket Prefix"] = Ticket
plt.figure(figsize=(12,6))
sns.factorplot(x="Ticket Prefix",y='Survived',data=df,kind="bar", size = 6).set_ylabels("survival probability")


# In[ ]:


Ticket = []
for i in list(df.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
        #Ticket.append('Y')
    else:
        Ticket.append("X")
        
df["Ticket Prefix"] = Ticket
plt.figure(figsize=(12,6))
sns.factorplot(x="Ticket Prefix",y='Survived',data=df,kind="bar", size = 6).set_ylabels("survival probability")


# Nop, aparently, the prefix does not make a significant difference. So once and for all we thrwo this variable.

# In[ ]:


df.drop(['Ticket','Ticket Prefix'],axis=1,inplace=True)


# # 4. Feture engineering

# ## 4.1 Name

# In[ ]:


#Create featyre with tittle and throw name
df['titles']=df['Name'].apply(lambda x: x.split(' ')[1])


# In[ ]:


df['titles'].value_counts()


# In[ ]:


# Unifyiing equivalent tittles
equivalent_titles={'der':'Mr.','Mlle.':'Miss.','Mme.':'Mrs.','Don.':'Mr.','Ms.':'Miss.','Dr.':'Master.'}

df['titles']=df['titles'].apply(lambda x: equivalent_titles[x] if x in equivalent_titles else x)

rare_dict=dict(df['titles'].value_counts()<=8)
rare_list=[x for x in rare_dict if rare_dict[x]==True]

df['titles']=df['titles'].apply(lambda x: 'rare' if x in rare_list else x)


# In[ ]:


g = sns.factorplot(x="titles",y="Survived",data=df,kind="bar")
#g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# Ha! Always knew that is better to rare than to be a Mr.

# In[ ]:


for elem in df['titles'].unique():
    df[str(elem)] = (df['titles'] == elem)/1


# In[ ]:


df.head()


# In[ ]:



def feat_moder(cols):
  
   Age = cols[0]
   Sex = cols[1]
   Parch= cols[2]
   title=cols[3]
   
   if Age>=16 and  Parch>0 and Sex==1 and title!='Miss.':
       return 1
   else:
       return 0

#df['Mother'] = df[['Age','Sex', 'Parch','titles']].apply(feat_moder,axis=1)


# In[ ]:


df.drop(['Name','titles'],axis=1,inplace=True)


# In[ ]:


df.head(2)


# ## 4.2 Create categorical values for Pclass

# In[ ]:


#df["Pclass"] = df["Pclass"].astype("category")
#df = pd.get_dummies(df, columns = ["Pclass"],prefix="Pc")


# In[ ]:


df.head(2)


# In the next section we shall use a linear model, for it, we will trow th Parch and SibSp variables.

# In[ ]:


#df.drop(['Parch','SibSp'],axis=1,inplace=True)


# ## 4.3 Scaling data

# Finally, we again split the train and test sets.

# In[ ]:


train = df[:train_len]
test = df[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[ ]:


train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)


# # 5. Cross validation
# 
# Some popular models:
# 
# * Logistic regression
# * Support vector machines
# * Decision Tree
# * Random Forest
# * Gradient Boosting
# * KNN
# * Ridge
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# In[ ]:


# Create rules to create the cross validations datasets that are shuffled. 
#Cross validation also creates datasets but as far as I'm aware it does not have the shuffle option.
kfold = StratifiedKFold(n_splits=8,shuffle=True, random_state=42)


# In[ ]:



rs = 42
clrs = []

clrs.append(XGBClassifier())

clrs.append(make_pipeline(RobustScaler(),SVC(random_state=rs)))
clrs.append(DecisionTreeClassifier(random_state=rs))
clrs.append(RandomForestClassifier(random_state=rs))
clrs.append(GradientBoostingClassifier(random_state=rs))
clrs.append(make_pipeline(RobustScaler() ,KNeighborsClassifier()))
clrs.append(LogisticRegression(random_state = rs))
clrs.append(RidgeClassifier(random_state = rs))

cv_results = []
for clr in clrs :
    cv_results.append(cross_val_score(clr, X_train, y = Y_train, scoring = 'f1_weighted', cv = kfold, n_jobs=4))

    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())


# In[ ]:


cv_df = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algo":["XGBoost","SVC","DecisionTree",
"RandomForest","Gradient Boosting","KNN","Logistic Regression",'RidgeClassifier']})

g = sns.barplot("CrossValMeans","Algo",data = cv_df,orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
print(cv_df)


# # 6. Hyper-Parameter tunning
# 
# Each model has hyper-parameter, roughtly speaking these are parameter that control the complexity of the model to avoid overfitting. 
# 
# An appropiate 

# [Guidelines for tunning XGBoost ](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

# In[ ]:


# SVC Parameters tunning 
xgb = XGBClassifier()



## Search grid for optimal parameters
xgb_param_grid ={  
    "n_estimators": [90,100,110],
    'max_depth':[5,10],
    'min_child_weight':[5,10],
    "learning_rate": [0.05,0.1],
    'subsample':[0.8],
    'colsample_bytree':[0.8],
    "gamma": [0],
    'reg_alpha':[1e-5, 1e-2, 0.1],
    "scale_pos_weight": [1],
}


gsxgb=GridSearchCV(xgb,param_grid = xgb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0,refit=True)

gsxgb.fit(X_train,Y_train)

xgb_best = gsxgb.best_estimator_

# Best score
gsxgb.best_score_, gsxgb.best_params_


# In[ ]:


# SVC Parameters tunning 
SVClr = SVC()


## Search grid for optimal parameters
SVClr_param_grid = [{'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100],'probability':[True]}
                     ,{'C':[1,10],'kernel':['linear'],'probability':[True]}
                   ]

robustscaler = RobustScaler().fit(X_train)
robustscaled_X=robustscaler.transform(X_train)

gsSVClr=GridSearchCV(SVClr,param_grid = SVClr_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0,refit=True)



gsSVClr.fit(robustscaled_X,Y_train)

SVClr_best = gsSVClr.best_estimator_

# Best score
gsSVClr.best_score_, gsSVClr.best_params_


# In[ ]:


# DesicionTree Parameters tunning 
DTC = DecisionTreeClassifier()


## Search grid for optimal parameters
DTC_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 5],
              "min_samples_leaf": [2, 5, 7],
              "criterion": ["gini"],
               "random_state":[42]}


gsDTC = GridSearchCV(DTC,param_grid = DTC_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)

gsDTC.fit(X_train,Y_train)

DTC_best = gsDTC.best_estimator_

# Best score
gsDTC.best_score_, gsDTC.best_params_


# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [2, 3],
              "min_samples_split": [7, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}



rf_param_grid = { 
    'max_features':['auto'], 'oob_score':[True], 'random_state':[1],
    "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5], "min_samples_split" : [ 4, 10 ], "n_estimators": [ 100, 400, 700]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_, gsRFC.best_params_


# In[ ]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_,gsGBC.best_params_


# In[ ]:


# KNN Parameters tunning 
KNNclr = KNeighborsClassifier()


## Search grid for optimal parameters
KNNclr_param_grid = {'n_neighbors':np.arange(1,20)}
                   

robustscaler = RobustScaler().fit(X_train)
robustscaled_X=robustscaler.transform(X_train)
    
gsKNNclr=GridSearchCV(KNNclr,param_grid = KNNclr_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)


gsKNNclr.fit(robustscaled_X,Y_train)

KNNclr_best = gsSVClr.best_estimator_

# Best score
gsKNNclr.best_score_, gsKNNclr.best_params_


# In[ ]:


# We can actually check the elbow rule, as we increase complexity (k) here is a point in which the error stabilizes! 
# In fact, I think that the elbow scatter plot of complexity vs cv error, but need to double check
plt.scatter(gsKNNclr.param_grid['n_neighbors'], 1-gsKNNclr.cv_results_['mean_test_score'])


# In[ ]:


# Logistic regression Parameters tunning 
LRClr = LogisticRegression()


## Search grid for optimal parameters
LRClr_param_grid = {'penalty':['l2'],'C':[1,10,100],'random_state':[rs]}

gsLRClr=GridSearchCV(LRClr,param_grid = LRClr_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)



gsLRClr.fit(X_train,Y_train)

LRClr_best = gsLRClr.best_estimator_

# Best score
gsLRClr.best_score_, gsLRClr.best_params_


# In[ ]:


# Ridge classication Parameters tunning 
RClr = RidgeClassifier()


## Search grid for optimal parameters
RClr_param_grid = {'alpha':[10, 1,0.1,0.01,0.001],'random_state':[rs],'normalize':[True, False]}

gsRClr=GridSearchCV(RClr,param_grid = RClr_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)



gsRClr.fit(X_train,Y_train)

RClr_best = gsLRClr.best_estimator_

# Best score
gsRClr.best_score_, gsRClr.best_params_


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 0)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# # 7. Learning curves (fitting analysis)

# In[ ]:


def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




plot_learning_curve(gsDTC.best_estimator_,"DTrees learning curves",X_train,Y_train,cv=kfold)
plot_learning_curve(gsGBC.best_estimator_,"GBoosting ",X_train,Y_train,cv=kfold)
plot_learning_curve(gsRClr.best_estimator_,"Logistic learning curves",X_train,Y_train,cv=kfold)
plot_learning_curve(gsLRClr.best_estimator_,"Ridge learning curves",X_train,Y_train,cv=kfold)


# In[ ]:


plot_learning_curve(gsRFC.best_estimator_,"RForest learning curves",X_train,Y_train,cv=kfold)


# In[ ]:


plot_learning_curve(gsxgb.best_estimator_,"XGB learning curves",X_train,Y_train,cv=kfold)
plot_learning_curve(make_pipeline(RobustScaler(),gsKNNclr.best_estimator_),"KNN learning curves",X_train,Y_train,cv=kfold)
plot_learning_curve(make_pipeline(RobustScaler(),gsSVClr.best_estimator_),"SVC learning curves",X_train,Y_train,cv=kfold)

plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)


# # 8. Ensemble

# ## 8.1 Voting

# In[ ]:


test_Survived_EX = pd.Series(ExtC_best.predict(test), name="ET")
test_Survived_XGB = pd.Series(xgb_best.predict(test), name="XGB")
test_Survived_DT = pd.Series(DTC_best.predict(test), name="DT")
test_Survived_RF = pd.Series(RFC_best.predict(test), name="RF")
test_Survived_GB = pd.Series(GBC_best.predict(test), name="GB")
test_Survived_R = pd.Series(RClr_best.predict(test), name="R")
test_Survived_L = pd.Series(LRClr_best.predict(test), name="L")

test_Survived_KNN = pd.Series(KNNclr_best.predict(robustscaler.transform(test)), name="KNN")
test_Survived_SVC = pd.Series(SVClr_best.predict(robustscaler.transform(test)), name="SVC")





# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_EX,test_Survived_XGB,test_Survived_DT,test_Survived_RF,test_Survived_GB,test_Survived_R, test_Survived_L,
                              test_Survived_KNN,test_Survived_SVC],axis=1)


sns.heatmap(ensemble_results.corr(),annot=True)


# In[ ]:


votingC = VotingClassifier(estimators=[
     ('ET', ExtC_best)
     ,('XG', xgb_best)
    #, ('DT', DTC_best)
    , ('RF', RFC_best)
    , ('GB', GBC_best) 
    , ('R',RClr_best)
    #, ('L',LRClr_best)
    , ('SVC',make_pipeline(RobustScaler(), SVClr_best))
    #, ('KNN',make_pipeline(RobustScaler(), KNNclr_best))
                                      ], voting='soft', n_jobs=4)

#votingC = votingC.fit(X_train, Y_train)


# ### Cross Validation of voting system

# In[ ]:



cv_result=cross_val_score(votingC, X_train, y = Y_train, scoring = 'f1_weighted', cv = kfold, n_jobs=-1)

    
cv_result.mean(), cv_result.std()


# In[ ]:


votingC = votingC.fit(X_train, Y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived")

IDtest = pd.read_csv('../input/test.csv')["PassengerId"]
results_voting = pd.concat([IDtest,test_Survived],axis=1)

results_voting.to_csv("python_voting.csv",index=False)


# ## 8.2 Blending

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_blending_train, X_blending_cv, Y_blending_train, Y_blending_cv = X_train[:450], X_train[450:],  Y_train[:450], Y_train[450:]


# In[ ]:


df_val=X_blending_cv
df_test=test

models=[
        ExtC_best
        #, xgb_best
        #, DTC_best
        #, RFC_best
        , GBC_best
        , RClr_best
        #, LRClr_best
        , make_pipeline(RobustScaler(), SVClr_best)
        #, make_pipeline(RobustScaler(), KNNclr_best)
]

for i,model in enumerate(models):
    model.fit(X_blending_train, Y_blending_train)
    
    val_meta_features=model.predict(X_blending_cv)
    test_meta_features=model.predict(test)
   
    val_meta_features=pd.DataFrame(val_meta_features, index=range(450,450+len(val_meta_features)),columns=['mf_'+'i'])
    test_meta_features=pd.DataFrame(test_meta_features, index=range(891,891+len(test_meta_features)),columns=['mf_'+'i'])
    
    df_val= pd.concat([df_val,val_meta_features],axis=1)
    df_test= pd.concat([df_test,test_meta_features],axis=1)
    
    


# In[ ]:



kfold_blending = StratifiedKFold(n_splits=4,shuffle=True, random_state=42)

model = make_pipeline(RobustScaler(), SVClr_best)


cv_result=cross_val_score(model, df_val, y = Y_blending_cv, scoring = 'f1_weighted', cv = kfold_blending, n_jobs=-1)

    
cv_result.mean(), cv_result.std()


# **Blending does not seem to help significantly**

# In[ ]:


model.fit(df_val,Y_blending_cv)

test_Survived_blending = pd.Series(model.predict(df_test), name="Survived")

results_blending = pd.concat([IDtest,test_Survived_blending],axis=1)

results_blending.to_csv("python_blending.csv",index=False)


# In[ ]:


results_voting.head(8)


# In[ ]:


results_blending.head(8)


# In[ ]:


df.head()


# In[ ]:




