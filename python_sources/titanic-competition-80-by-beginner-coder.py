#!/usr/bin/env python
# coding: utf-8

# # Data Importing

# In[ ]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib
import platform
import seaborn as sns
from sklearn import datasets 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

message="        Versions        "
print("*"*len(message))
print(message)
print("*"*len(message))
print("Scikit-learn version={}".format(sklearn.__version__))
print("Numpy version={}".format(np.__version__))
print("Pandas version={}".format(pd.__version__))
print("Matplotlib version={}".format(matplotlib.__version__))
print("Python version={}".format(platform.python_version()))

# shift-tab to show docstring: highlight and shift-tab: format
#?zip()
#%lsmagic
# Suppress Future Warnings
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# ## Hello W0rld
# 
# ### Welcome to my frist kernel. My name is Lester and I am an aspiring Data Scientist.
# 
# ### I am new to programming and my first competition here is Kaggle's Titanic Challenge. It took me awhile to achieve 80% accuracy. Hence, I would like to share my research and work to other beginners who would find this useful.
# 
# ### I would like to thank @MinsukHeo, @Jack Roberts, @Chris Deotte and SP Prof Leong and others who have inspired and helped me in this kaggle challenge.

# In[ ]:





# In[ ]:


titanictrain = pd.read_csv("../input/titanic/train.csv")
print(titanictrain.shape)
titanictrain[:5]


# In[ ]:


titanictest = pd.read_csv("../input/titanic/test.csv")
print(titanictest.shape)
titanictest[:5]


# # Exploratory Data Analysis

# In[ ]:


titanictrain.info()


# In[ ]:


titanictest.info()


# In[ ]:


titanictrain.isnull().sum()


# In[ ]:


titanictest.isnull().sum()


# In[ ]:


def bar_chart(feature):
    survived = titanictrain[titanictrain['Survived']==1][feature].value_counts()
    dead = titanictrain[titanictrain['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')
print('The Chart show Women more likely survivied than Men')


# In[ ]:


bar_chart('Pclass')
print('The Chart show 1st class more likely survivied than other classes and 3rd class more likely dead than other classes')


# In[ ]:


bar_chart('SibSp')
print('The Chart shows a person aboarded with more than 2 siblings or spouse (3 to 8) or a person aboarded  without siblings or spouse (0) is more likely dead')


# In[ ]:


bar_chart('Parch')
print('This chart shows a person aboarded with more than 3 parents or children (4 to 6) or a person aboarded alone (0) is more likely dead')


# In[ ]:


bar_chart('Embarked')
print('The Chart shows a person aboarded from Q or a person aboarded from S more likely dead')


# In[ ]:





# In[ ]:





# # Data Preparation

# ### Extracting Title from Name
# 
# #### Extracting the title from the Name column will allow us to group the passengers into specific types of Man and woman, particular, younger males and females. The most common titles are Mr, Miss, Mrs and Master. The other titles are very rare, therefore I will group them along other male and females.
# 
# #### Here, I have grouped the passengers into 4 groups, Man for older male passengers, Master for younger Male passengers, Mrs for older female and Ms younger female. I then Map these groups into the dataset. I will also use the average age of these groups to fill missing age data.

# In[ ]:


train_test_data = [titanictrain, titanictest] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False) #extract using regular expression


# In[ ]:


titanictrain['Title'].value_counts()


# In[ ]:


titanictest['Title'].value_counts()


# In[ ]:


title_mapping = {"Capt": 'Man',"Don": 'Man',"Major": 'Man',"Col": 'Man',"Rev": 'Man',"Dr": 'Man',"Sir": 'Man',"Mr": 'Man',"Jonkheer": 'Man', 
                 "Dona": 'Woman',"Countess": 'Woman',"Mme": 'Woman',"Mlle": 'Woman',"Ms": 'Woman',"Miss": 'Miss',"Lady": 'Woman',"Mrs": 'Mrs',
                 "Master": 'Boy' }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# 

# In[ ]:


titanictrain.head()


# ### Transforming Ticket and Surname Columns buy grouping them into Survival Probabilities
# 
# ##### In this section, I will transfrom the Ticket and Surname column into survival probabilities by finding the mean value of duplicates.
# 
# ##### After researching background of passengers, I have found out that families and friends are most likely to have the same Surname and Ticket number. During the disaster, families and friends are more likely to look out for each other and move together as a group, they either survive or die as a group.  Hence, if most of the familiy or friends within a specific duplicate group survives, it is safe to assume that a member of that group in the test data is also likely to survive.
# 
# ##### During the sinking of Titanic, woman and children are given priority to rescue boats. Therefore, if a woman or children in the test data has the same Surname or ticket of a specific duplicate group, that person most likely shares the same fate as the majority of the group.  For example, if in the training data, a woman, her 1st child and 2nd child survives but 3rd child dies. The mean survive rate for this particular group of duplicates is 0.75 (3/4 x 100%).  If there is a 4th child in the test data, discovered by matching Surnames and young age, that child is a assumed to survive as most people in that particular group of duplicate (family) survives, that child will be given a score of 0.75 (Score of 1 means survive, Score of 0 means dead). If a group of friends who have the same ticket number survives, a passenger in the test data who also have the same number is most likely to survive.
# 
# ##### This process is done by 1st counting the value of duplicates, sorting the datasets by duplicates, creating a new panda dataframe for duplicate groups and finding their mean chance of survival. For passengers who do not have duplicate surnames or ticket, they are given 0.5, 50% chance of dying or surviving. The surname and ticket column is subsequently mapped into the trainning data and test data.
# 

# In[ ]:


titanictrain['Ticket'].value_counts() #Check if there are common tickets


# In[ ]:





# In[ ]:


n_ticket = titanictrain.sort_values('Ticket') #sort dataset by tickets


# In[ ]:


df1 = pd.DataFrame(titanictrain.groupby('Ticket')['Survived'].count())
df1 = df1[df1['Survived']<2] #keep tickets that has only 1 count
df1.shape


# In[ ]:


df1 = df1.index.values.tolist() #to create a list of 1 count tickets


# In[ ]:


n_ticket = n_ticket[~n_ticket.Ticket.isin(df1)] #to remove 1 count tickets from n_ticket


# In[ ]:


tick_surv = n_ticket.groupby('Ticket')['Survived'].mean() #to find mean of survival of duplicate tickets


# In[ ]:


tick_surv


# In[ ]:





# In[ ]:


for dataset in train_test_data:
    dataset['Ticket'] = dataset['Ticket'].map(tick_surv)


# In[ ]:


for dataset in train_test_data:
    dataset['Ticket'] = dataset['Ticket'].replace(np.nan, 0.5)


# In[ ]:


titanictrain.head()


# In[ ]:


titanictest.head(400)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for dataset in train_test_data:
    dataset['Surname'] = dataset['Name'].str.extract('([A-Za-z]+)', expand=False) #extract using regular expression


# In[ ]:


titanictrain.head()


# In[ ]:


n_surname = titanictrain.sort_values('Surname')


# In[ ]:


df1 = pd.DataFrame(titanictrain.groupby('Surname')['Survived'].count())
df1 = df1[df1['Survived']<2]
df1.shape


# In[ ]:


df1 = df1.index.values.tolist()


# In[ ]:


n_surname = n_surname[~n_surname.Surname.isin(df1)]


# In[ ]:


surname_surv = n_surname.groupby('Surname')['Survived'].mean()


# In[ ]:


for dataset in train_test_data:
    dataset['Surname'] = dataset['Surname'].map(surname_surv)


# In[ ]:


for dataset in train_test_data:
    dataset['Surname'] = dataset['Surname'].replace(np.nan, 0.5)


# In[ ]:


titanictrain.head()


# In[ ]:





# In[ ]:


# delete unnecessary feature from dataset
titanictrain.drop('Name', axis=1, inplace=True)
titanictest.drop('Name', axis=1, inplace=True)


# In[ ]:





# #### Add missing age values
# 
# ##### In this secion, I used the mean age according to each title group and fill in missing data.

# In[ ]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Master, Others)
titanictrain["Age"].fillna(titanictrain.groupby("Title")["Age"].transform("mean"), inplace=True)
titanictest["Age"].fillna(titanictest.groupby("Title")["Age"].transform("mean"), inplace=True)


# In[ ]:


print('This chart shows that from ages 0 to 15 more likely to survive and ages 27 to 37 more likely to die')

facet = sns.FacetGrid(titanictrain, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanictrain['Age'].max()))
facet.add_legend()
plt.show()


# In[ ]:





# In[ ]:





# ##### Add missing Embarked Values

# In[ ]:


Pclass1 = titanictrain[titanictrain['Pclass']==1]['Embarked'].value_counts()
Pclass2 = titanictrain[titanictrain['Pclass']==2]['Embarked'].value_counts()
Pclass3 = titanictrain[titanictrain['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
print('This chart shows that regardless of class, more than 50% will embark from S. Hence, we will use S to replace missing values')


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:





# In[ ]:





# #### Add missing fare values
# 
# ##### In this section, I will use the median class fare according to each Pclass to fill missing data. Median is used as there are a wide range of prices within each class, I feel that median is more appropriate.

# In[ ]:


# fill missing Fare with median fare for each Pclass
titanictrain["Fare"].fillna(titanictrain.groupby("Pclass")["Fare"].transform("median"), inplace=True)
titanictest["Fare"].fillna(titanictest.groupby("Pclass")["Fare"].transform("median"), inplace=True)
titanictrain.head()


# In[ ]:


facet = sns.FacetGrid(titanictrain, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, titanictrain['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:





# In[ ]:





# ### Replace missing cabin values
# 
# ##### In this section, we will replace missing cabin values with no and yes for those with cabin. As there is a large amount of missing cabin data, I have group the dataset into 2 groups, those with and those without cabin. Background research has also showed that cabins are very expensive, it is likely that a sizeble group of passengers most likey 3rd class did not travel with cabins.

# In[ ]:





# In[ ]:





# In[ ]:


titanictrain['Cabin'].fillna('No', inplace=True)
titanictrain['Cabin'].replace(regex=r'^((?!No).)*$',value='Yes',inplace=True)
titanictrain.head(2)


# In[ ]:


titanictest['Cabin'].fillna('No', inplace=True)
titanictest['Cabin'].replace(regex=r'^((?!No).)*$',value='Yes',inplace=True)
titanictest.head(2)


# In[ ]:





# In[ ]:





# ### Combine Family Size and Create Alone Column
# 
# ##### SibSp and Parch is combine with each individual to find out the combine family size. An alone column is also created to determine if the passengers is travelling alone. A passenger travelling alone is more likely to die as women and children family groups are prioritised for survival.

# In[ ]:


titanictrain["FamilySize"] = titanictrain["SibSp"] + titanictrain["Parch"] +1
titanictest["FamilySize"] = titanictest["SibSp"] + titanictest["Parch"] +1


# In[ ]:


facet = sns.FacetGrid(titanictrain, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, titanictrain['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


titanictrain["Alone"] = titanictrain["FamilySize"]-1==0 
titanictest["Alone"] = titanictest["FamilySize"]-1==0 


# In[ ]:


titanictrain.head()


# In[ ]:


family_mapping = { True: 1, False: 0 }
for dataset in train_test_data:
    dataset['Alone'] = dataset['Alone'].map(family_mapping)


# In[ ]:





# #### Mapping PClass

# In[ ]:


class_mapping = { 1: 'First_Class', 2: 'Second_Class', 3: 'Third_Class' }
for dataset in train_test_data:
    dataset['Pclass'] = dataset['Pclass'].map(class_mapping)


# In[ ]:


titanictrain.head()


# In[ ]:


titanictest.head()


# In[ ]:





# #### Encode all the categorical variables and Find Correlation

# In[ ]:


titanictrain = pd.get_dummies(titanictrain)
titanictrain.head(5)


# In[ ]:


titanictest = pd.get_dummies(titanictest)
titanictest.head(5)


# In[ ]:


corr = titanictrain.corr()
corr.sort_values(["Survived"], ascending = False, inplace = True)
print(corr.Survived)


# In[ ]:





# In[ ]:





# ### Drop non-utilised feature
# 
# ##### In this section, I exported the datasets to Microsoft AzureML Studio for testing and analysis.
# 
# ##### Through extensive trial and error and logistical corelation, I have determined my best combination of features to achieve highest accuracy.
# 
# ##### I then drop the features that are not in use. This new dataset is later used to create predictive models in this notebook.

# In[ ]:


titanictrain.to_csv("azuretitanictrain.csv",index=False)
titanictest.to_csv("azuretitanictest.csv",index=False)


# In[ ]:


submission1 = pd.read_csv('azuretitanictrain.csv')
submission1.head()


# In[ ]:


features_drop = ['SibSp','Parch','Surname','FamilySize','Sex_female','Sex_male','Cabin_Yes','Cabin_No','Embarked_C','Embarked_Q','Embarked_S']
titanictrain = titanictrain.drop(features_drop, axis=1)
titanictest = titanictest.drop(features_drop, axis=1)
titanictrain = titanictrain.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = titanictrain.drop('Survived', axis=1)
target = titanictrain['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head(10)


# In[ ]:





# ##### The features for trainning and testing are: Age, Ticket, Fare, Alone, PClass and Title.
# ##### The rational for choosing these features are: 
# ##### Title and Age will seperate passengers into Man, Woman, Boys and girls
# ##### Ticket, Fare and PClass will show family/friends relationship
# ##### Alone will show which passengers are unlikely to receive help from others
# ##### The premise of this feature set is base on the assumptions that Women and Children have higher chances of survival and passengers survive in groups of families or friends.

# In[ ]:


train_data.info()


# In[ ]:


titanictest.head(10)


# In[ ]:





# In[ ]:


titanictest.info()


# In[ ]:





# # Train Model

# #### In this section, I will import the various models and prepare to split Data into train and test Sets using Cross fold validation

# In[ ]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:





# ## Train Model with different algorithms: 

# In[ ]:


print('Train using KNN')
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# kNN Score
round(np.mean(score)*100, 2)


# In[ ]:


print('Train using Decision Tree')
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# decision tree Score
round(np.mean(score)*100, 2)


# In[ ]:


print('Train using Random Forest')
clf = RandomForestClassifier(n_estimators=500,max_depth=6)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Random Forest Score
round(np.mean(score)*100, 2)


# In[ ]:


print('Train using Naive Bayes')
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# In[ ]:


print('Train using SVM')
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


clf = MLPClassifier(activation='logistic',
                    hidden_layer_sizes=(200, 80))
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


clf = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Score and Evaluate Model

# ## Score Model and Evaluate Model:
# 
# ##### I have decided to use RandomForestClassifier as it has the highest cross validation score. The classifer will have n_esimators of 500 and max depth of 6. My best result with AzureML Studio, using the Two-Class Decision Jungle Algo is 81.34%, my best score using RandomForestClassifier model here is 80.38%. As compared to the stupidbaseline, my model is at least 10% more accurate. I have also discovered that if we simply predict all female survive, it will output about 76% accuracy. Hence, my model is more accurate as compared to models with simpler feature designs.

# In[ ]:


train_data.head()


# In[ ]:


clf = RandomForestClassifier(n_estimators=500,max_depth=6)
clf.fit(train_data, target)

test_data = titanictest.drop("PassengerId", axis=1)
prediction = clf.predict(test_data)


# In[ ]:





# In[ ]:


submission = pd.DataFrame({
        "PassengerId": titanictest["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('titanicsubmission.csv', index=False)


# In[ ]:


submission = pd.read_csv('titanicsubmission.csv')
submission.head()


# In[ ]:





# ### References
# 
# ##### https://sites.google.com/site/hermaidenvoyage/titanic-luxury
# ##### https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818
# ##### https://www.kaggle.com/jack89roberts/titanic-using-ticket-groupings
# ##### https://www.dummies.com/education/history/suites-and-cabins-for-passengers-on-the-titanic/
# ##### https://qz.com/321827/women-and-children-first-is-a-maritime-disaster-myth-its-really-every-man-for-himself/
# ##### https://www.kaggle.com/minsukheo/titanic-solution-with-sklearn-classifiers
# ##### https://owlcation.com/humanities/Titanic-April-1912-3rd-class-passengers-survivors-died-1st-2nd-ship-maiden-voyage-iceberg-sinking-sank
# ##### https://autumnmccordckp.weebly.com/tickets-and-accomodations.html

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




