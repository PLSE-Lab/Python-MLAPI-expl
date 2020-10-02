#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will make machine that can predict someone's income, they have annual income more than 50.0000 dollars or not. The data that I take is from UCI Machine Learning which published on [Kaggle.com](https://www.kaggle.com/uciml/adult-census-income). The purposes of this project are knowing our customer's income in current area is important to build a business, so we can targeted the prospect market and estimate price of our product, knowing some behavior informations of the sample (occupation, how long they work per week, capital gain) from this data.
# 
# Before we start this project, i will describe some features in this data :
# 
# **Data Set Characteristics:**
# 
# * Number of rows: 32561
# 
# * Number of features: 15 (categorical, numeric).
#     
# * Daftar Feature:
#     - **age** : explaining the age of the sources.
#     - **workclass** : explaining class where they work is.
#     - **finalweight** : The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian noninstitutional population of the US.
#     - **education** : explaining the last education of the sources.
#     - **educationNumber** : labeling the last education in numerical.
#     - **maritalStatus** : explaining the marital status of the sources.
#     - **occupation** : explaining what job of the sources take.
#     - **relationship** : eplaining the relation status of the source.
#     - **race** : explaining what race is in this data.
#     - **sex** : There is 2 types in this feature 'Male' and 'Female'.
#     - **capitalGain** : explaining that the sources were involved in an investment or not and how much they get the gain of that.
#     - **capitalLoss** : explaining that the sources were involved in an investment or not and how much they get the loss of that.
#     - **hoursperweek** : explaining how long they worked per week.
#     - **nativeCountry** : explaining where they come from.
#     - **income** : this is a target that we want ti predict later.
# * Missing Attribute Values: 
#     1. workclass = 1836
#     2. occupation = 1843
#     3. nativeCountry = 583
# 
# 
# * Class Distribution: 24720 - '<=50K', 7841 - '>50K
# 
# The structure of this project is :
# 
#    1. Import Libraries and Data
#    2. Exploratory Data Analysis (EDA)
#        <br/>2.1. Fixing Columns
#        <br/>2.2. Describe Data
#        <br/>2.3. Analyze Target
#        <br/>2.4. Analyze the Features 
#        <br/>. . . . . .2.4.1 Analyze Categorical Feature
#        <br/>. . . . . . . . . . 2.4.1.1. Occupation
#        <br/>. . . . . . . . . . 2.4.1.2. Native Country
#        <br/>. . . . . . . . . . 2.4.1.3. Workclass
#        <br/>. . . . . . . . . . 2.4.1.4. Education
#         2.4.2 Analyze Numerical Feature
#         <br/>. . . . . . . . . . 2.4.2.1 Age
#         <br/>. . . . . . . . . . 2.4.2.2 Finalweight
#         <br/>. . . . . . . . . . 2.4.2.3 Capital Gain and Capital Loss
#         <br/>. . . . . . . . . . 2.4.2.4 Hours per Week
#    3. Dealing with Missing Data
#    4. Evaluate Model
#        <br/>. . . . . .4.1. Logistics Regression
#        <br/>. . . . . .4.2. Decision Tree
#        <br/>. . . . . .4.3. Random Forest
#        <br/>. . . . . .4.4. XGBoost
#        <br/>. . . . . .4.5. Gradient Boosting
#    5. Finalizing and Optimizing Model
#    6. Conclusion and Business Orientation
#        <br/>. . . . . .6.1. Comparing Class Income
#        <br/>. . . . . . . . . . 6.1.1. High-Income Target
#        <br/>. . . . . . . . . . 6.1.2. Low-Income Target
#    
# I hope you enjoyed this notebook and can get the benefits of it. Just remember, this is my first project so give feedback and we can learn together!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/adult-census-income/adult.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# ## 2. EDA

# #### 2.1 Fixing Columns

# In[ ]:


df.columns = ['age','workclass','finalweight',
              'education','educationNumber','maritalStatus',
              'occupation','relationship','race',
              'sex','capitalGain','capitalLoss',
              'hoursperweek','nativeCountry','income']
df.head()


# #### 2.2 Describe Data
# 
# I want to know the distribution of the data, so i describe it and make a table so i can see the information from dataset more clearly.

# In[ ]:


df.describe().T


# I create table to get more clearly information from datasets. It helps me to know type of data, how many null in this data, unique data and unique sample.

# In[ ]:


listItem = []
for col in df.columns :
    listItem.append([col, df[col].dtype, df[col].isna().sum(), round((df[col].isna().sum()/len(df[col]))*100,2),
                     df[col].nunique(), list(df[col].unique()[:3])])
dfDesc = pd.DataFrame( columns=['dataFeatures','dataType','null','nullPct','unique','uniqueSample'], data=listItem)
dfDesc


# #### 2.3 Analyze Target
# For the first, I change the target to 1 & 0. 1 for income >50K and 0 for income <=50K.

# In[ ]:


df['income'] = df['income'].apply(
    lambda x : 1 if x != '<=50K' else 0
)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x=df['income'],palette='Set1')


# Seen from above, we have an imbalance target that we will predict later, the income '<=50K' is  more than the income '>50K'.

# ### 2.4 Analyze the Features
# For the first i want to seperate between categorical feature from numerical feature.

# In[ ]:


categorical = [i for i in df.columns.drop(['income','nativeCountry']) if df[i].dtype == 'O']


# After that, i make some plots in each category and of some features to analyze the corralation between categorical features to income.

# In[ ]:


plt.figure(figsize=(10,60))
for i in range(len(categorical)) :
    plt.subplot(7,1,i+1)
    sns.countplot(x='income', hue=f'{categorical[i]}', data=df, palette='Set1')
    plt.xticks(rotation=90)
plt.show()


# Seen from above that workclass, occupation, education, relationship, maritalStatus has a correlation with the target, but for race and sex has a low correlation to target because the pattern between 0 and 1 almost same. Now i want to know the ratio from sex column.

# In[ ]:


# to see the ratio income per category in each feature
df.groupby(['sex'])['income'].mean() 


# I want to know per category when I sepearate it with the target.

# In[ ]:


plt.figure(figsize=(10,80))
for i in range(len(categorical)) :
    plt.subplot(7,1,i+1)
    sns.countplot(x=f'{categorical[i]}', hue='income', data=df)
    plt.xticks(rotation=90)
plt.show()


# #### 2.4.1 Analyze Categorical Feature

# #### 2.4.1.1 Occupation
# From the plot above the data shows that to earn income >50K, they don't have to be fixated on the high education they take, but it also depends on the type of work they do. I am interested in analyzing feature occupation, besides this feature also affects the level of income, there is also a unique category "?" in the feature.

# In[ ]:


df['occupation'].value_counts()


# In[ ]:


display(df[df['occupation']== '?'].head())
print(df[df['occupation']== '?'].shape)


# from the dataframe above, assuming I have every workclass that contains '?' then the occupation also follows '?'. To prove it, let's look at the dataframe below.

# In[ ]:


df[(df['occupation']=='?')&(df['workclass']!='?')]


# It turned out that my assumption was wrong, there was still an undefined occupation that was parallel to the Never-worked category in the workclass feature. It makes sense because when someone has never worked then he cannot fill the occupation feature.
# 
# Because there are two conditions in the '?' then, for the first condition then I will replace '?' be data that often appears in this feature (mode), but before that I will further analyze whether the mode in this feature can replace the '?' category or not. For this reason I want to compare patterns when viewed in terms of income as a target for occupations that are '?' with the mode of occupation.

# In[ ]:


sns.countplot(df[df['occupation']=='?']['occupation'],hue=df['income'])


# In[ ]:


sns.countplot(df[df['occupation']=='Prof-specialty']['occupation'],hue=df['income'])


# It turns out that the plot above explains that the income of a Prof-specialty is quite balanced between those <=50K and >50K, but for the category of occupation '?' it appears that for income >50K less than <=50K then I will not change occupation '?' become the mode of occupation itself. This is because it will change the pattern of the target itself. From the pattern in the dataframe above, there are two possibilities (assumptions) : 
# 
# 1. Someone did not fill the feature occupation, that is because they did not want to mention where they worked (it would affect the workclass feature which was also filled with '?')
# 2. Someone had never worked. 
# 
# For that, I will drop someone whose occupation and workclass is filled with '?' simultaneously because there are no other features that can explain what can replace the category '?' and if it is replaced it will also change its target (income), but I will let the occupation with the category '?' if the workclass is filled with 'Never-worked' and replaces '?' with 'None'. For starters I will replace '?' with 'None'.

# #### 2.4.1.2 Native Country

# In[ ]:


df['nativeCountry'].value_counts()


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot('nativeCountry',hue='income',data=df)
plt.xticks(rotation=90)
plt.show()


# #### 2.4.1.3 Workclass
# 
# Just like the occupation feature,the type of company where they work can also increase the income that will be earned, for that I want to analyze further, and also there is category '?' which is similar to in feauture occupation.

# In[ ]:


df['workclass'].value_counts()


# Because my assumptions when analyzing feature occupation are wrong, I also check the workclass feature. Is there a '?' in workclass when the occupation category is not '?'.

# In[ ]:


df[(df['workclass']=='?')&(df['occupation']!='?')]


# It turns out for the workclass feature, the result is empty. It is still possible that someone does not want to tell what and what his occupation is.

# In[ ]:


sns.countplot(df[df['workclass']=='Private']['workclass'],hue=df['income'])


# In[ ]:


sns.countplot(df[df['workclass']=='?']['workclass'],hue=df['income'])


# It turns out the plot above explains that the income of a Private and the income of '?' has a different pattern,I will not change occupation '?' become the mode of workclass itself. It's because it will change the pattern of the target itself. For that, I will drop someone whose occupation and workclass is filled with '?' simultaneously because there are no other features that can explain what can replace the category '?'.

# #### 2.4.1.4 Education

# In[ ]:


df[['education','educationNumber']].sort_values(by=['educationNumber']).head()


# In[ ]:


listEdu = list(df['education'].unique())


# In[ ]:


listItem = []
for i in listEdu:
    listItem.append([i,np.unique(df[df['education']==i]['educationNumber'])[0]])


# In[ ]:


dfEdu = pd.DataFrame(listItem,columns=['education','educationNumber']).sort_values(by=['educationNumber'])
dfEdu = dfEdu.reset_index()
dfEdu.drop('index',axis=1,inplace=True)
dfEdu


# Because educationNumber feature can explain the education feature, I will drop the education column. So that there aren't too many features (waste).

# In[ ]:


df = df.drop('education',axis=1)


# #### 2.4.2 Analyze Numerical Feature

# In[ ]:


numerical = [i for i in df.columns.drop(['income','nativeCountry']) if df[i].dtype != 'O']


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


sns.pairplot(pd.concat([df[numerical],df['income']],axis=1),hue="income")


# #### 2.4.2.1 Age

# In[ ]:


sns.distplot(df[df['income'] == 0]['age'], kde=True, color='darkred', bins=30)
sns.distplot(df[df['income'] == 1]['age'], kde=True, color='blue', bins=30)


# Based on plot above we know that there's still has age above 80 years. For that reason, I want to see if there are 90-year-old residents who are still working

# In[ ]:


sns.countplot(df[df['age']==90]['occupation'], palette = 'Set1')
plt.xticks(rotation=90)
plt.show()


# There are 43 peoples who are 90 years old who are still working, and most are still working in the Exec-managerial section, uniquely there is still 1 person who works in the security service and cleaners parts. After that I want to see how long someone is 90 years old work.

# In[ ]:


sns.distplot(df[df['age']==90]['hoursperweek'])
plt.xticks(rotation=90)
plt.show()


# It turns out there are also those who work above 70 hours per week, then I want to see what work makes the 70 years old population work for 80 hours per week.

# In[ ]:


df[(df['age']==90)&(df['hoursperweek']>70)]


# #### 2.4.2.2 Finalweight

# In[ ]:


sns.distplot(df[df['income'] == 0]['finalweight'], kde=True, color='darkred', bins=30)
sns.distplot(df[df['income'] == 1]['finalweight'], kde=True, color='blue', bins=30)


# #### 2.4.2.3 Capital Gain and Capital Loss

# In[ ]:


plt.figure(figsize=(8,10))
sns.distplot(df['capitalGain'],kde=False)
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(df[(df['capitalGain']>0)]['workclass'],hue=df['income'])
plt.xticks(rotation=90)
plt.show()


# Most of them who get capital gains has income> 50K per year.For who work in government, there is very little interest in them to invest.

# #### 2.4.2.4 Hours per Week

# In[ ]:


plt.figure(figsize=(8,10))
sns.distplot(df['hoursperweek'],kde=False)
plt.show()


# I see the anomaly of the plot formed above, because there is a sample that works 99 hours per week, for that I will analyze it further, I want to see what kind of work requires work hours that are more than 70 hours per week.

# In[ ]:


df[df['hoursperweek']>70]['occupation'].value_counts().plot(kind='bar',title='Occupation hours')


# Hereafter i want to see how much salary they get and what kind of job they take when they worked more than 80 hours per week.

# In[ ]:


sns.countplot(df[df['hoursperweek']>80]['occupation'],hue=df['income'])
plt.xticks(rotation=90)
plt.show()
print('Total',df[df['hoursperweek']>80]['occupation'].count())


# It turns out that for Exec-managerial there are around 27 people who work over 80 hours even though they are paid more than 50K or less than 50K. For a fisherman, it makes sense if they need more time when working because it adjusts the conditions to get the more fish.

# ## 3. Dealing With Missing Data
# 
# From EDA above I'll drop occupation '?' and workclass '?'

# In[ ]:


df['occupation'] = df[['occupation','workclass']].apply(lambda x : 'None' if x['occupation'] == '?' and x['workclass']=='Never-worked' else x['occupation'],axis=1)


# In[ ]:


listdrop = df[(df['occupation']=='?')&(df['workclass']=='?')].index
df.drop(listdrop,axis=0,inplace=True)


# In[ ]:


df[numerical].isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# ### Data with 13 Features

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


categorical = [i for i in df.columns.drop(['income']) if df[i].dtype == 'O']
print(categorical)


# In[ ]:


lewc = LabelEncoder()
lemar = LabelEncoder()
leoc = LabelEncoder()
lerl = LabelEncoder()
lerc = LabelEncoder()
lesx = LabelEncoder()
lenc = LabelEncoder()


# In[ ]:


lewc.fit(df['workclass'])
lemar.fit(df['maritalStatus'])
leoc.fit(df['occupation'])
lerl.fit(df['relationship'])
lerc.fit(df['race'])
lesx.fit(df['sex'])
lenc.fit(df['nativeCountry'])


# In[ ]:


# with open('lewc.pickle', 'wb') as f:
#     pickle.dump(lewc, f)
# with open('lemar.pickle', 'wb') as f:
#     pickle.dump(lemar, f)
# with open('leoc.pickle', 'wb') as f:
#     pickle.dump(leoc, f)
# with open('lerl.pickle', 'wb') as f:
#     pickle.dump(lerl, f)
# with open('lerc.pickle', 'wb') as f:
#     pickle.dump(lerc, f)
# with open('lesx.pickle', 'wb') as f:
#     pickle.dump(lesx, f)
# with open('lenc.pickle', 'wb') as f:
#     pickle.dump(lenc, f)


# In[ ]:


df['workclass'] = lewc.transform(df['workclass'])
df['maritalStatus'] = lemar.transform(df['maritalStatus'])
df['occupation'] = leoc.transform(df['occupation'])
df['relationship'] = lerl.transform(df['relationship'])
df['race'] = lerc.transform(df['race'])
df['sex'] = lesx.transform(df['sex'])
df['nativeCountry'] = lenc.transform(df['nativeCountry'])


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[numerical])


# In[ ]:


# with open('scaler.pickle', 'wb') as f:
#     pickle.dump(scaler, f)


# In[ ]:


dfScaled = pd.DataFrame(scaler.transform(df[numerical]),columns=df[numerical].columns)


# In[ ]:


dfLabeled = pd.DataFrame(df[categorical],columns=df[categorical].columns)
dfLabeled = dfLabeled.reset_index(drop=True)


# In[ ]:


dfScaled = pd.concat([dfLabeled,dfScaled],axis=1)


# In[ ]:


dfScaled.head()


# In[ ]:


data = dfScaled.drop(['nativeCountry'],axis=1)
target= df['income']


# ## 4. Evaluate Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import  XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, auc, log_loss, matthews_corrcoef,roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, learning_curve,KFold, train_test_split


# For the first i made kfold function, it helps me to get to compare the results of test and train.

# In[ ]:


def calc_train_error(X_train, y_train, model):
#     '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    logloss = log_loss(y_train,predictProba[:,1])
    report = classification_report(y_train, predictions)
    lossBuatan = (abs((y_train-predictProba[:,1]))).mean()
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'logloss': logloss,
        'lossBuatan': lossBuatan
    }
    
def calc_validation_error(X_test, y_test, model):
#     '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    logloss = log_loss(y_test,predictProba[:,1])
    report = classification_report(y_test, predictions)
    lossBuatan = (abs((y_test-predictProba[:,1]))).mean()
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'logloss': logloss,
        'lossBuatan':lossBuatan
    }
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
#     '''fits model and returns the in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error


# ### Train Test Split

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=.3,random_state=101)


# #### 4.1 Logistics Regression

# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[ ]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # instantiate model
    logreg = LogisticRegression(solver='lbfgs')

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, logreg)
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[ ]:


listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvalLR = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])
listIndex = list(dfEvalLR.index)
listIndex[-1] = 'Average'
dfEvalLR.index = listIndex
dfEvalLR


# In[ ]:


for item,rep in zip(range(1,6),train_errors) :
    print('Report Train ke ',item,':')
    print(rep['report'])


# In[ ]:


for item,rep in zip(range(1,6),validation_errors) :
    print('Report Test ke ',item,':')
    print(rep['report'])


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=logreg,
                                                       X=data,
                                                       y=target,
                                                       train_sizes=np.linspace(0.3, 0.8, 5),
                                                       cv=10,
                                                       scoring='accuracy')

print('\nTrain Scores : ')
print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print('\nTrain Mean : ')
print(train_mean)
print('\nTrain Size : ')
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)
print('\nTrain Std : ')
print(train_std)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print('\nTest Scores : ')
print(test_scores)
print('\nTest Mean : ')
print(test_mean)
print('\nTest Std : ')
print(test_std)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


predictProbaTrain=logreg.predict_proba(X_train)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


predictProbaTest=logreg.predict_proba(X_test)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# #### 4.2 Decision Tree

# In[ ]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # instantiate model
    dtree = DecisionTreeClassifier(max_depth=7,min_samples_leaf=25)

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, dtree)

    # append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[ ]:


listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss'],tr['lossBuatan'],val['lossBuatan']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvalDTC = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss','Train loss Buatan','Test Loss Buatan'])
listIndex = list(dfEvalDTC.index)
listIndex[-1] = 'Average'
dfEvalDTC.index = listIndex
dfEvalDTC


# In[ ]:


for item,rep in zip(range(1,6),train_errors) :
    print('Report Train ke ',item,':')
    print(rep['report'])


# In[ ]:


for item,rep in zip(range(1,6),validation_errors) :
    print('Report Test ke ',item,':')
    print(rep['report'])


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=dtree,
                                                       X=data,
                                                       y=target,
                                                       train_sizes=np.linspace(0.3, 0.9, 5),
                                                       cv=10,
                                                       scoring='accuracy')

print('\nTrain Scores : ')
print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print('\nTrain Mean : ')
print(train_mean)
print('\nTrain Size : ')
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)
print('\nTrain Std : ')
print(train_std)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print('\nTest Scores : ')
print(test_scores)
print('\nTest Mean : ')
print(test_mean)
print('\nTest Std : ')
print(test_std)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


predictProbaTrain=dtree.predict_proba(X_train)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


predictProbaTest=dtree.predict_proba(X_test)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


coef1 = pd.Series(dtree.feature_importances_,data.columns).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')


# #### 4.3 Random Forest

# In[ ]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]
    
    # instantiate model
    rfc = RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=101)

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, rfc)

    # append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[ ]:


listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvalRFC = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])
listIndex = list(dfEvalRFC.index)
listIndex[-1] = 'Average'
dfEvalRFC.index = listIndex
dfEvalRFC


# In[ ]:


for item,rep in zip(range(1,6),train_errors) :
    print('Report Train ke ',item,':')
    print(rep['report'])


# In[ ]:


for item,rep in zip(range(1,6),validation_errors) :
    print('Report Test ke ',item,':')
    print(rep['report'])


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=rfc,
                                                       X=data,
                                                       y=target,
                                                       train_sizes=np.linspace(0.3, 0.9, 5),
                                                       cv=10,
                                                       scoring='accuracy')

print('\nTrain Scores : ')
print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print('\nTrain Mean : ')
print(train_mean)
print('\nTrain Size : ')
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)
print('\nTrain Std : ')
print(train_std)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print('\nTest Scores : ')
print(test_scores)
print('\nTest Mean : ')
print(test_mean)
print('\nTest Std : ')
print(test_std)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


predictProbaTrain=rfc.predict_proba(X_train)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


predictProbaTest=rfc.predict_proba(X_test)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


coef1 = pd.Series(rfc.feature_importances_,data.columns).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')


# #### 4.4 XGBoost

# In[ ]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # instantiate model
    xgb = XGBClassifier(max_depth=10,min_child_weight=10, n_estimators=250, learning_rate=0.1)

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, xgb)

    # append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[ ]:


listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvalXGB = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])
listIndex = list(dfEvalXGB.index)
listIndex[-1] = 'Average'
dfEvalXGB.index = listIndex
dfEvalXGB


# In[ ]:


for item,rep in zip(range(1,6),train_errors) :
    print('Report Train ke ',item,':')
    print(rep['report'])


# In[ ]:


for item,rep in zip(range(1,6),validation_errors) :
    print('Report Test ke ',item,':')
    print(rep['report'])


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,
                                                       X=data,
                                                       y=target,
                                                       train_sizes=np.linspace(0.3, 0.9, 5),
                                                       cv=10,
                                                       scoring='accuracy')

print('\nTrain Scores : ')
print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print('\nTrain Mean : ')
print(train_mean)
print('\nTrain Size : ')
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)
print('\nTrain Std : ')
print(train_std)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print('\nTest Scores : ')
print(test_scores)
print('\nTest Mean : ')
print(test_mean)
print('\nTest Std : ')
print(test_std)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


predictProbaTrain=xgb.predict_proba(X_train)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


predictProbaTest=xgb.predict_proba(X_test)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


coef1 = pd.Series(xgb.feature_importances_,data.columns).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')


# #### 4.5 Gradient Boosting

# In[ ]:


train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # instantiate model
    gbc = GradientBoostingClassifier(max_depth=10, n_estimators=150, learning_rate=0.1)

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, gbc)

    # append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)


# In[ ]:


listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvalGBC = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])
listIndex = list(dfEvalGBC.index)
listIndex[-1] = 'Average'
dfEvalGBC.index = listIndex
dfEvalGBC


# In[ ]:


for item,rep in zip(range(1,6),train_errors) :
    print('Report Train ke ',item,':')
    print(rep['report'])


# In[ ]:


for item,rep in zip(range(1,6),validation_errors) :
    print('Report Test ke ',item,':')
    print(rep['report'])


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,
                                                       X=data,
                                                       y=target,
                                                       train_sizes=np.linspace(0.3, 0.9, 5),
                                                       cv=10,
                                                       scoring='accuracy')

print('\nTrain Scores : ')
print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print('\nTrain Mean : ')
print(train_mean)
print('\nTrain Size : ')
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)
print('\nTrain Std : ')
print(train_std)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print('\nTest Scores : ')
print(test_scores)
print('\nTest Mean : ')
print(test_mean)
print('\nTest Std : ')
print(test_std)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


predictProbaTrain=xgb.predict_proba(X_train)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


predictProbaTest=xgb.predict_proba(X_test)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


coef1 = pd.Series(xgb.feature_importances_,data.columns).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')


# In[ ]:


outside = ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy','Accuracy', 'Accuracy',
          'ROC_AUC', 'ROC_AUC', 'ROC_AUC', 'ROC_AUC', 'ROC_AUC','ROC_AUC', 'ROC_AUC',
          'F1','F1','F1','F1','F1', 'F1','F1',
          'LogLoss','LogLoss','LogLoss','LogLoss','LogLoss','LogLoss','LogLoss']
inside = [1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std']
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[ ]:


acc = []
roc = []
F1 = []
logloss = []

kol = {
    'acc' : 'Test Accuracy',
    'roc' : 'Test ROC AUC',
    'F1' : 'Test F1 Score',
    'logloss' : 'Test Log Loss'
}

for Elr,Edtc, Erfc, Exgb, Egbc in zip(dfEvalLR.iloc[:5].values,dfEvalDTC.iloc[:5].values,dfEvalRFC.iloc[:5].values, dfEvalXGB.iloc[:5].values, dfEvalGBC.iloc[:5].values):
    acc.append([Elr[1],Edtc[1],Erfc[1], Exgb[1], Egbc[1]])
    roc.append([Elr[3],Edtc[3],Erfc[3], Exgb[3], Egbc[3]])
    F1.append([Elr[5],Edtc[5],Erfc[5], Exgb[5], Egbc[5]])
    logloss.append([Elr[7],Edtc[7],Erfc[7], Exgb[7], Egbc[7]])

for i,j in zip([acc,roc,F1,logloss], ['acc','roc','F1','logloss']):
    i.append([dfEvalLR.iloc[:5][kol[j]].mean(),dfEvalDTC.iloc[:5][kol[j]].mean(),dfEvalRFC.iloc[:5][kol[j]].mean(), dfEvalXGB.iloc[:5][kol[j]].mean(), dfEvalGBC.iloc[:5][kol[j]].mean()])
    i.append([dfEvalLR.iloc[:5][kol[j]].std(),dfEvalDTC.iloc[:5][kol[j]].std(),dfEvalRFC.iloc[:5][kol[j]].std(), dfEvalXGB.iloc[:5][kol[j]].std(), dfEvalGBC.iloc[:5][kol[j]].std()])

dfEval = pd.concat([pd.DataFrame(acc),pd.DataFrame(roc),pd.DataFrame(F1),pd.DataFrame(logloss)], axis=0)
dfEval.columns = ['LR','DTC','RFC', 'XGB', 'GBC']
dfEval.index = hier_index
dfEval


# ## 5. Finalizing  & Optimizing Model
# 
# Consider the stability for the result from all models (Log loss,F1,ROC AUC and accuracy), I choose XGBoost for my model. This model does not indicate overfit. So, i want to optimize this model using SMOTE considering the target is imbalance.

# In[ ]:


data = dfScaled.drop(['nativeCountry'],axis=1)
target= df['income']


# In[ ]:


xtr,xts,ytr,yts = train_test_split(data,target,test_size=.3,random_state=101)


# In[ ]:


smot = SMOTE(random_state=101)
X_smot,ytr = smot.fit_sample(xtr,ytr)
X_smot = pd.DataFrame(X_smot, columns=xtr.columns)


# In[ ]:


xtr,xts,ytrn,yts = train_test_split(data,target,test_size=.3,random_state=101)


# In[ ]:


model = XGBClassifier(max_depth=10, n_estimators=250, learning_rate=0.1)


# In[ ]:


model.fit(X_smot,ytr)


# In[ ]:


predictTrain = model.predict(xtr)
predictTrain


# In[ ]:


print(classification_report(ytrn,predictTrain))


# In[ ]:


predictTest = model.predict(xts)
predictTest


# In[ ]:


print(classification_report(yts,predictTest))


# In[ ]:


predictProbaTrain=model.predict_proba(xtr)


# In[ ]:


pred = predictProbaTrain[:,1]
fpr, tpr, threshold = roc_curve(ytrn, pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


log_loss(ytrn,pred)


# In[ ]:


predictProbaTest=model.predict_proba(xts)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(yts, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


log_loss(yts,preds)


# In[ ]:


coef1 = pd.Series(model.feature_importances_,data.columns).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')


# In[ ]:


# with open('XGB.pickle', 'wb') as f:
#     pickle.dump(model, f)


# ## 6. Conclusion & Business Orientation

# Based on the Machine Learning Model that I created and implemented through the Marketing Mix Concept as shown above, the conclusion is:
# 
# 1. **Place** : My model is not very suitable for this concept, because I realize that to determine an area, the data that I use the majority to the United States so that, behavior for residents in that country with behavior in other countries will show different patterns and will tend to United States
# 2. **Product** : For the concept of product it will useful when using this machine learning, because from this machine learning we can determine which targets that will be introduced our products, if they are high income they will introduce premium products, as well as when they look at low income targets they will introduce products which suitable for people with low income
# 3. **Price** : The concept of price itself will also be affected after we can determine what products we will introduce in accordance with the targets that we already know
# 4. **Promotion** : Of course, promotion is also affected for products with high-income targets. We can think of promotional concepts that are appropriate to the class.

# #### 6.1 Comparing Class Income

# When we will predict the target apropriately so as not throw away many cost for promotion or produce a product, then we can play with the threshold from our model that i created.

# In[ ]:


predictProbaTest=model.predict_proba(xts)


# In[ ]:


preds = predictProbaTest[:,1]
fpr, tpr, threshold = roc_curve(yts, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characterisitc')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.show()


# In[ ]:


print('FPR:',fpr[-760:-750])
print('TPR:',tpr[-760:-750])
print('THRESHOLD:',threshold[-760:-750])


# In[ ]:


listProba = []
for x,y,z in zip(tpr,fpr,threshold):
    listProba.append([x,y,z])
dfProba = pd.DataFrame(listProba, columns=['TPR','FPR','Threshold'])
dfProba.head()


# #### 6.1.1 High Income Target
# 
# If we will target "high-income" so the result of prediction  for class (1) must be precise, then I will raise the recall in >50K class (1).

# In[ ]:


dfProba[dfProba['TPR']>0.17].head(20)


# In[ ]:


predictions = [1 if i > 0.16 else 0 for i in preds]


# In[ ]:


print(classification_report(yts,predictions))


# In[ ]:


sns.countplot(predictions)


# In[ ]:


sns.countplot(yts)


# #### 6.1.2 Low Income Target

# If we will target "low-income" so the result of prediction  for class (0) must be precise, then I will raise the precision in >50K class (1).

# In[ ]:


dfProba[dfProba['FPR']<0.014].tail(50)


# In[ ]:


predictions = [1 if i > 0.78 else 0 for i in preds]


# In[ ]:


print(classification_report(yts,predictions))


# In[ ]:


sns.countplot(predictions)


# In[ ]:


sns.countplot(yts)


# Of the two assumptions above, it's better if the company focuses on low-income targets. Although the model can predict income <=50K over >50K, it does not exclude the possibility of high-income customers buying products targeted at low income. But, if the company still wants to make premium products with high income, then we can choose the first option.

# # Thank You !
