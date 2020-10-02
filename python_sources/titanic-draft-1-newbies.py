#!/usr/bin/env python
# coding: utf-8

# Just finished an online data science course that took 100hours to complete, so I thought I challenge myself to my first proper Kaggle submission. The aim here is try to do simple analysis and perform machine learning on the data with reasonable accuracy (>70%). Theres much more to be learn, but for now I'm happy to settle with this. So lets start.

# > **"""The objective of this work is to use regression/machine learning process to predict whether or not the titanic passenger survive based on their personal data (ie: name, age, gender, socio-economic class, etc)."""**

# Before that, heres the Table of Content
# <br><br><br>
# 0. Importing data<br><br>
# 1. Data cleaning<br><br>
# 2. See stats feature (mainly countplot) & segregation of continous data<br><br>
# 3. Applying labelEncoder, scaling the data & splitting the training sets into train_train & train_test<br><br>
# 4. Applying the ML (KNN & XGB as starting point). Then make a bit of optimal investigation on data splitting<br><br>
# 5. Predict the test & prepare submission file

# > My comments were written in a mixed markdown section & with the hashtag '#' in the code section. Sorry I'm used to do this with Matlab, so trying to slowly get off the habit.  

# <br><br><br><br><br>
# **0. Importing data**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv');print(df.shape)
test = pd.read_csv('/kaggle/input/titanic/test.csv');print(test.shape)

# Lets combine train set( df) and test together so its easier to process everything in one go
# Just remember test set doesnt have 'Survived' column, so everything in NaN when we combine them
df = df.append(test, ignore_index=True)

# Lets just drop 'PassengerID' cause it won't be useful in this case
df.drop(columns='PassengerId',inplace=True)
df.info()

# Double checking how I should seperate train & test set for later
# temp = data.loc[:890,:]
# temp2 = data.loc[891:,:]


# <br><br><br><br><br><br>
# **1. Data Cleaning**<br><br>
# After importing & inspecting the train.csv, I've noticed that Cabin has more than half of null object. So I decided to remove it from the df. <br><br>Name and ticket can also be dropped. But wait, maybe good idea to categorise the name according to the persons title<br><br>The next part will be on cleaning the data & converting the object to category

# In[ ]:


df.Name.head(30)


# In[ ]:


# From what I see aove, we can extract the title by getting text that follows by a dot(.)
# Now lets get the title for each person

for name_string in df['Name']:
    df['Title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=True) # Use '([A-Za-z]+)\.' here to get string follow by dot

# Okay lets see how many value trns out
df['Title'].value_counts()


# In[ ]:


# Okay, after looking at several other kernels, I realised that Don, Sir, Col, Major, Jonkheer, Capt can be map as Mr. 
# Similarly, Mlle, Mme, Ms - Miss, and Lady, Countess, Dona as Mrs.

# Mapping by changing several similar titles to common ones (ie: Don to Mr)
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

# So replace the title that we want using the mapping dictionary
df.replace({'Title': mapping}, inplace=True)

# Get the new title count
df.Title.value_counts()


# In[ ]:


# Now lets start the imputation stuff. First we impute the age using the median of each title. 
# Get the median age in each categorical title, then impute it to the null value in age

# From above we get the following titles:
titles=['Mr','Miss','Mrs','Master','Rev','Dr']
# So use this to get the median age 
for title in titles:
    listOfMedianAgeForEachTitle = df.groupby('Title')['Age'].median() # Get the median for each title and put it in a list
    age_to_impute = listOfMedianAgeForEachTitle[titles.index(title)] # For the speciic title in the current loop, get the median age using the index
    print(title, age_to_impute) # Print the title and age for the current loop
    
    # So if the age of the current title is null, replace is to the age_to_impute
    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute
    
    # End Loop

# Check how which columns still null
df.isnull().sum()


# In[ ]:


# From what I observed in the data description in Kaggle, seems like ticket number varies for each person
# meaning that it might just be a non-meaningfull number. Lets doublecheck the number of values
df.Ticket.value_counts()


# In[ ]:


# Ticket got 929 distinct values in 1300++ entries, so won't be useful for the machine learning. Drop em'
# We already used Name and replace it to title, so lets drop em'
# Cabin got heaps of NaN. So lets just drop em'
df.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

df.info()


# In[ ]:


# So embarked & fare got null entries (Survived too but we know that test fle doesnt have em so lets ignore it now)
# Fare is numeric so can easily fill NaN as mean value
# Embarked is an object so lets check the value counts
df.Embarked.value_counts()


# In[ ]:


# Fill the NaN in Embarked column with the highest value count and Fare with the mean()
df.Embarked.fillna(df.Embarked.value_counts().argmax(),inplace=True)
df.Fare.fillna(df.Fare.mean(),inplace=True)
df.info()


# In[ ]:


# I learned that some category is easier if we convert them from category to Numeric. 
# So lets figure out the number of distinct values in each object columns

for col in df.columns:
    if df[col].dtype not in ['int64','float64']:
        totalDistinct = df[col].value_counts().count()
        print(col,totalDistinct)


# In[ ]:


# Okay so we have 3 in embarked, 2 in sex and 6 in title. maybe lets convert the embarked and sex first
# SEX - Convert female = 0, male = 1
df['Sex'] = df.Sex.apply(lambda x: 0 if x == "female" else 1)

#EMBARKED - Convert Q=0, C=1, S=2
df['Embarked'] = df.Embarked.apply(lambda x: 0 if x == "Q" else (1 if x=="C" else 2))

print(df.Sex.value_counts())
print(df.Embarked.value_counts())


# In[ ]:


# So wait, parch is the number of parents aboard & SibSp is the number of childern aboard.
# Maybe its better if we add both of them to a column 'NoOfFamilies'. Arguable here but I wanna take this route 
# to simplify things

df['NoOfFamilies']=df['SibSp']+df['Parch']

df.NoOfFamilies.value_counts()


# In[ ]:


# Not bad, 790 were single and the rest got families aboard. I could split them into category of 2 class, single & taken 
# but not too sure maybe theres somebody who came with 1 parent or something like that. 
# I'll maybe try and split them to 0-no family and 1-have family. Will come back to see if that improves the accuracy
haveFamilies = (df.NoOfFamilies>0) # For later. Not sure whether i'll use it or not

# Drop the Parch & SibSp columns cause I alread have the NoOfFamilies column
df.drop(columns=['Parch','SibSp'],inplace=True)

# Sweet, now i think we are readdy to move. Lets just double check that everything is good
df.info()


# In[ ]:


# Convert all object to category (in our case just the title actually)
for col in df.columns:
    if df[col].dtype not in ['int64','float64']:
        df[col]=df[col].astype('category')


# <br><br><br><br><br><br><br><br><br>
# **2. A bit of statistics on the data**<br><br>
# Okay, all the data cleaning is done. Let's have a look at the stats. Here I will look at:<br>1. The count of each variables with 'Survived' set as hue. This is to understand how different class/group affects the chance of survival<br>2. For continous data, I will segregate them into different groups particularly for Age & Fare. <br><br>
# So we'll take a closer look at each column starting from correlation plot, then Embarked, Sex, Age, Fare and NoOfFamilies. Other columns may not be necessary at this stage but will certainly be included later to see potential ways to improve accuracy

# In[ ]:


# Checking the correlation & put it in a heatmap plot
import seaborn as sns
import matplotlib.pyplot as plt

corrmat =df.corr()
corrInd = corrmat.index[abs(corrmat['Survived'])>0]
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),
            annot=True, cmap = 'Blues',
            vmin=0,vmax=1,square=True
           )


# What I observed here is that :<br>
# * 'Survived' is positively correlated with 'Fare'<br>
# * 'Survived' is negatively correlated with sex, pclass & age. (kinda logic here cause females (sex), younglings (age) were the priorities.)<br>
# * 'Survived' is negatively correlated with Embarked. Recall emabarked Q-0, C-1 & S-2. Seems passenger emarked from Q(queenstown) have better chances of survival. But also maybe because theres less passenger from Q. Lets look at it later <br>
# <br>Pclass 1 (1st class) seem to survive better here. Also makes sense, you pay more for better insurance I guess.
# <br><br> The rest had minor correlation, but maybe still useful. I think its best if we make a plot for each column against the survived

# In[ ]:


# Use describe just to inspect the values from each numeric columns. So I know whether column is boolean or not, whats the mean and std
df.describe()


# Embarked

# In[ ]:


# Plot Embarked countplot with Survived as the hue. Recall 0-Q, 1-C, 2-S
sns.countplot('Embarked',data=df,hue='Survived')
plt.show()

# What I can say here is that embarked may not be a huge factor for predicting the survival rate. You can see that
# the percentage of survival in each case is pretty much 50-50 except for 


# What I can say here is that embarked may not be a huge factor for predicting the survival rate. You can see that the percentage of survival in each case is pretty much 50-50 except for embarked from Southampton (2) but again, this might just be accidental because a lot of people died that day. I'm struggling to see the correlation whether you die or not depending on where you board the ship(Unless if the distance between the points of boarding location is really far). So I'm dropping this column in the dataset.
# <br>
# ![image.png](attachment:image.png)
# <br> 

# In[ ]:


# Dropping the embarked column
df.drop(columns='Embarked',inplace=True)


# Sex

# In[ ]:


# Plot the sex with survived as hue. Female-0, Male-1
sns.countplot('Sex',data=df,hue='Survived')
plt.show()

# The plot kinda makes sense as female were prioritised compared to male. So less female died & more male died


# Age

# In[ ]:


# Plot the age and survived as hue
sns.countplot('Age',data=df,hue='Survived')
plt.show()


# You can see that its a bit struggling to identify the age because the gap is huge. So what I'm thinking is converting this into categorical data where:<br>
# <= 5 is Baby<br> 
# 6-14 is Kids<br>
# 15-20 is Teenage<br>
# 21-34 is YoungAdult<br>
# 35-49 is Adult<br>
# 50 is Senior
# 

# In[ ]:


bins = [0, 6, 15, 21, 35, 50, 100]
names = ['Baby', 'Kids', 'Teenage', 'YoungAdult', 'Adult','Senior']

df['AgeRange'] = pd.cut(df['Age'], bins, labels=names)

print(df.AgeRange.value_counts())


# In[ ]:


# Plot the ageRange and survived as hue
sns.countplot('AgeRange',data=df,hue='Survived')
plt.show()

# Sweet and lets delete the age column cause its not necessary anymore
df.drop(columns='Age',inplace=True)


# Fare

# In[ ]:


# So I know fare will be the same as age. so I need to sepearte them into groups first. 
# But not sure whats the group to divide them into. According to this wiki page,
# $0-40 for 3rd class, $60 for 2nd class and $150 for 1st class. Check the link here:
# http://www.jamescamerononline.com/TitanicFAQ.htm 

bins = [-1, 50, 149, 1000] # (-1,51) (51,149) (150,1000)
names = [3, 2, 1]

df['FareRange'] = pd.cut(df['Fare'], bins, labels=names)

print(df.FareRange.value_counts())
print(df.Pclass.value_counts())

# So compare between both FareRange and Pclass, there were a bit difference here. Maybe some of them paid discounted price
# or early birds, so they got cheap price for a good Pclass.


# In[ ]:


# # This part was here because I previously set the lower limit in bins as 0 (see above) 
# # and got several NaN values in FareRange.
# # After looking at the Fare value of the ones that appear as NaN in the FareRange, I noticed all was 0.
# # So change the lower limit from 0 to -1 and it solves the issue. 

# Null_FareRange=df.FareRange[df.FareRange.isnull()==True]
# df.Fare.loc[Null_FareRange.index]


# In[ ]:


# Similarly, I think Fare is not important anymore as we have FareRange now. Lets drop Fare

df.drop(columns='Fare',inplace=True)


# In[ ]:


# countplot for FareRange with survived as hue
sns.countplot('FareRange',data=df,hue='Survived')
plt.show()

# Both FareRange 1 & 2 showed higher percentage of survival compaed to FareRange 3 (lowest range).
# If theres anything we learn here, its a good idea to pay for better class when you go for a cruise holiday.
# At least its evident here that they have better chance of survival!


# NoOfFamilies

# In[ ]:


sns.countplot('NoOfFamilies',data=df,hue='Survived')


# In[ ]:


# Looks to me that we can divide them into 3 groups:
# 0 - NoFamily (0)
# 1 - SmallFamily(1-3)
# 2 - BigFamily (4-10)

bins = [-1,0,3,1000] # (-1,0) (1,3) (4,1000)
names = [0,1,2]

df['FamilySize'] = pd.cut(df['NoOfFamilies'], bins, labels=names)

sns.countplot('FamilySize',data=df,hue='Survived')
plt.show()


# In[ ]:


# Okay we can drop NoOfFamilies column

df.drop(columns='NoOfFamilies',inplace=True)


# What we learn here:<br><br>1. Change all continous data into categorical type cause at the end of the day, we want to predict categorical outcome.
# <br>2. So Fare, NoOfFamilies and Age have been converted to different group class
# <br>3. Embarked seems pretty uncorrelated in my point of view. so dropping it.
# <br><br> Okay, now lets move on and get our hands dirty with ML! Wubba lubba dup dup!!!!

# <br><br><br><br><br><br><br><br><br>
# **3. Turn the category data to numeric (and scaling)**
# <br><br>Okay, done with the stats stuff, now lets get to the real meat! <br> 
# I'm going to introduce LabelEncoder to labelise the category<br><br>
# I dont think we need StandardScaler cause the min and max for each colum range between 0 and 5. But maybe I'll try use it later and see if it improves the accuracy

# In[ ]:


# Import necessary tools
from sklearn.preprocessing import LabelEncoder, StandardScaler

encoder = LabelEncoder()
sc = StandardScaler()

def labelNscale(df):
    # Find the columns of all category data and put it in list
    colsCateg = [col for col in df.columns if df[col].dtype not in ['int64','float64']]
    # For all category data, apply LabelEncoder on them
    for col in colsCateg:
        df[col]=encoder.fit_transform(df[col])
    
    # Now apply scaler on all data (numeric & category)
    df_ = sc.fit_transform(df)
    df = pd.DataFrame(data=df_,columns=df.columns)
    return df

df_R=labelNscale(df)
df_R.describe()


# **3.1 Splitting data into training and test set**
# <br><br>Now that I have labelise them, its time to split the data into training & test sets

# In[ ]:


# Recall we have train and test set? 
# Train has a shape of 891 rows & test has 418. 
# Now lets split them 
trainSet = df_R.loc[:890,:]
for_y = df.loc[:890,:]
testSet = df_R.loc[891:,:]

print(trainSet.shape)
print(testSet.shape)

# Doublecheck train info
trainSet.info()

# Drop the testSet survived column
testSet.drop(columns='Survived',inplace=True)
testSet.info()


# In[ ]:


# Sweet everythings good. Now split the trainSet into X & y
y = for_y['Survived']
X = trainSet.drop(columns = ['Survived']).copy()
print(X.shape)
y


# In[ ]:


# Splitting the train set into train_train and train_test.
# Im using StratifiedShuffleSplit cause I want the train_Train set to have 
# equal survival percentage values to the train_Test set. 
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[ ]:


# Double check percentage split of y_train and y_test
print((y_train.value_counts())/len(y_train))
print(y_test.value_counts()/len(y_test))


# <br><br><br><br><br><br><br><br><br>
# **4. Now apply the ML on them. Classifier not Regressor.**
# <br><br>Alright, now let's see how it goes when I apply several models to predict them.
# I used regressor at first and got really low accuracy. Only then realised I should use classifier instead<br><br> The outline here is:<br>
# a) Apply Knn & XGB, see whos more accurate<br>
# b) Compile the process of splitting data into function so we can loop over and see if changing the split size or nplits affect the accuracy by how much. Then choose the appropriate one.

# In[ ]:


# Use knn & XGB as first guess model

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier(max_depth=10,n_estimators =15,random_state=5)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
print('rfcScore: {:.4f}'.format(rfc.score(X_train,y_train)))
acc_rfc=accuracy_score(y_test,y_pred_rfc)
print('Score using rfc is: ',acc_rfc)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print('knnScore: {:.4f}'.format(knn.score(X_train,y_train)))
acc_knn=accuracy_score(y_test,y_pred)
print('Score using knn is: ',acc_knn)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)
print('xgbScore: {:.4f}'.format(xgb.score(X_train,y_train)))
acc_xgb=accuracy_score(y_test,y_pred_xgb)
print('Score using XGB is: ',acc_xgb)


# Now I am trying to find ways to improve the models. 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

lr = LogisticRegression()
dtc = DecisionTreeClassifier()

classifiers = [('K Nearest Neighbours', knn),('Random Forest Classifier',rfc),('XGB',xgb)]

for clf_name, clf in classifiers:
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    print('{:s} : {:.4f}'.format(clf_name,accuracy_score(y_test, y_pred)))
    
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train,y_train)
y_pred_vc = vc.predict(X_test)
print('VC : {:.4f}'.format(accuracy_score(y_test, y_pred)))


# Not bad with 84% of accuracy using the XGB. Definitely can get better accuracy if I put more effort on segregating the data. But happy with this right now. Lets try improve model?

# In[ ]:


# Define the process to split and predict as function with several input that we want to change
# This wil return the accuracy of teh XGB model

def findOptimalXGB(X,y,nsplit,testsize):

    sss = StratifiedShuffleSplit(n_splits=nsplit, test_size=testsize, random_state=1)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb=accuracy_score(y_test,y_pred_xgb)
    
    return acc_xgb, xgb


# In[ ]:


print('Optimal test size')
testSize = []
nsplits = []
accuracy = []

for i in np.arange(0.1,0.35,0.05):
    for j in range(2,17,2):
        acc,xgb = findOptimalXGB(X,y,j,i)
#         print('nsplits:{:.1f}, testSize:{:.2f}, acc:{:.4f}'.format(j,i,acc))
        testSize.append(i)
        nsplits.append(j)
        accuracy.append(acc)


# Now repeat for optimal settings so that its default for our test prediction next
# acc = findOptimalXGB(X,y,10,0.3)


# In[ ]:


# Plot graph

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot(testSize,accuracy,hue=nsplits,size = nsplits,sizes=(20, 200), palette=cmap)
plt.xlabel('Test Size (The legend shows nsplits size)')
plt.ylabel('Accuracy')
plt.show()


# Lets use the mid point - test size of 0.2, nsplits of 15

# In[ ]:


acc, optXGB = findOptimalXGB(X,y,10,0.25)

print(acc)
predictions = optXGB.predict(testSet)


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':range(3,11,1),'learning_rate':np.arange(0.01,0.2,0.04)}
for i in range(5,10,1):
    grid = GridSearchCV(xgb,param_grid=param_grid,cv=i)
    grid.fit(X_train,y_train)
    # print(grid.grid_scores_)
    print('cv:',i, grid.best_params_, 'Score:{:.4f}'.format(grid.best_score_))


# I'll choose cv6 here cause the score dropped in cv=7 and started to rise again in cv=8 & cv=9 which may be overfitting. 

# In[ ]:


# Set and get the opimal grid
grid = GridSearchCV(xgb,param_grid=param_grid,cv=6)
grid.fit(X_train,y_train)
optimalGrid = grid.best_estimator_


# In[ ]:


print(xgb.score(X_test,y_test))
print(optimalGrid.score(X_test,y_test))


# So using default option with n_splits of 10 and tset_size of 0.3 gave the best prediction of 86%. Lets stick with that. 

# In[ ]:


# Apply confusio matrix to see how good our fitting is
from sklearn.metrics import confusion_matrix, recall_score
y_pred = xgb.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print(confusion_matrix(y_test,y_pred).ravel())
print(recall_score(y_test,y_pred))
print(tp/(tp+fn))


# <br><br><br><br><br><br><br><br><br>
# **5. Predict the test and put into submission file **

# In[ ]:


predictions = xgb.predict(testSet)


# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
PassengerId = test['PassengerId']

submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
submission.to_csv(path_or_buf ="Titanic_Submission.csv", index=False)
print("Submission file is formed")


# <a href="Titanic_Submission.csv"> Download File </a>

# Certainly not the best model but I'm satisfied enough by how I understand the data and use manpulate it to predict unseen data. Wht I'll do next is to broaden my knowledge on making pipelines & perhaps use pd.get_dummies on some columns. Then I'll come back and see if it can improves the accuracy of the model. Cheers!

# In[ ]:




