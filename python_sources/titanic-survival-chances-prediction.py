import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

data = pd.read_csv('../input/train.csv')

data['Age'].fillna(data['Age'].median(), inplace = True)


# plotting survival with sex 
 
survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
dead_sex     = data[data['Survived'] == 0]['Sex'].value_counts()
df1 = pd.DataFrame([survived_sex, dead_sex])
df1.index = ['Survived','Dead']
df1.plot(kind = 'bar', stacked = True, figsize = (10,8))
##  clearly shows that females survived more than males

# plotting survival with age

plt.figure(figsize = (20,8))
survived_age = data[data['Survived'] == 1]['Age']
dead_age     = data[data['Survived'] == 0]['Age']
plt.hist([survived_age, dead_age], bins = 50, stacked = True, rwidth = 0.8,
         color = ['g','r'], label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('No. of passengers')
plt.grid(True)
plt.legend()
## shows that people belonging to age group 27-29 died the most

#plotting survival with fare

plt.figure(figsize = (20,8))
survived_fare = data[data['Survived'] == 1]['Fare']
dead_fare     = data[data['Survived'] == 0]['Fare']
plt.hist([survived_fare, dead_fare], bins = 50, stacked = True, rwidth = 0.8,
         color = ['g','r'], label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('No. of passengers')
plt.axis([0,550, 0, 350])
plt.grid(True)
plt.legend()
# shows that people who had booked tickets of lower fares, died the most

#plotting age, fare and survival in one chat using scatter plot

plt.figure(figsize = (20,8))
ax1 = plt.subplot()
surv_age  = data[data['Survived'] == 1]['Age']
surv_fare = data[data['Survived'] == 1]['Fare']
dead_age  = data[data['Survived'] == 0]['Age']
dead_fare = data[data['Survived'] == 0]['Fare']
ax1.scatter(surv_age, surv_fare, s = 40, c = 'green')
ax1.scatter(dead_age, dead_fare, s = 40, c = 'red')
ax1.set_xlabel('Age')
ax1.set_ylabel('Fare')
ax1.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


#plotting embarked with survival
surv_embarked = data[data['Survived'] == 1]['Embarked'].value_counts()
dead_embarked = data[data['Survived'] == 0]['Embarked'].value_counts()
df2 = pd.DataFrame([surv_embarked, dead_embarked])
df2.index = ['Survived','Dead']
df2.plot(kind = 'bar', stacked = True, figsize = (10,8))
##clearly no useful conclusion can be made from this plot


#************  phase 2: Feature engineering         ***********************


    
#combining train and test datasets
def get_combined_data():
    train = pd.read_csv('../input/train.csv')
    test  = pd.read_csv('../input/test.csv')
    
    train.drop('Survived',1, inplace = True)
    combined = train.append(test)
    combined.reset_index(inplace = True)
    combined.drop('index',1, inplace = True)
    
    return combined

combined = get_combined_data()

#processing titles
#adding feature title by extracting title from name
#we will use a dictionary to map titles 

def get_titles():
    global combined
    combined['Title'] = combined.Name.map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    titles_dic = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                }
    combined['Title'] = combined.Title.map(titles_dic)
    
get_titles()

#processing ages

def process_age():
    global combined
    
    grouped_median_train = combined.head(891).groupby(['Sex','Pclass','Title']).median()
    grouped_median_test = combined.iloc[891:].groupby(['Sex','Pclass','Title']).median()
    
    def fillAges(row,grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',1,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',1,'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female',1,'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female',1,'Royalty']['Age']    
        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',2,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',2,'Mrs']['Age']
        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',3,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',3,'Mrs']['Age']
        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male',1,'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male',1,'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male',1,'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male',1,'Royalty']['Age']  
        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male',2,'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male',2,'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male',2,'Officer']['Age']
        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male',3,'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male',3,'Mr']['Age']
            
    combined.head(891)['Age'] = combined.head(891).apply(lambda r :
                fillAges(r, grouped_median_train) 
                if np.isnan(r['Age']) else r['Age'], axis = 1)
    combined.iloc[891:]['Age'] = combined.iloc[891:].apply(lambda r :
                fillAges(r, grouped_median_test) 
                if np.isnan(r['Age']) else r['Age'], axis = 1)

process_age()

#processing names
#since using names can create problems in later stages when we will use machine
#learning algorithms, so we will drop the names and keep only the titles as dummies

def process_names():
    global combined
    combined.drop('Name', axis = 1, inplace = True)
    title_dummies = pd.get_dummies(combined['Title'], prefix = 'Title')
    combined = pd.concat([combined,title_dummies],axis = 1)
    combined.drop('Title', axis = 1, inplace = True)
    
process_names()

#processing fare
#there is only one missing value in fare. we will replace it by mean

def process_fare():
    global combined
    combined.Fare.fillna(combined.Fare.mean(), inplace = True)
    
process_fare()

#processing embarked
#we will fill missing values using the most frequent value, i.e, 'S'

def process_embarked():
    global combined
    combined.Embarked.fillna('S', inplace = True)
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix = 'Embarked')
    combined = pd.concat([combined,embarked_dummies], axis = 1)
    combined.drop('Embarked', axis = 1, inplace = True) 
    
process_embarked()

#processing cabin
#the missing values will be replaced by 'U' for unknown

def process_cabin():
    global combined
    combined['Cabin'].fillna('U', inplace = True)
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix = 'Cabin')
    combined = pd.concat([combined,cabin_dummies], axis = 1)
    combined.drop('Cabin', axis = 1, inplace = True)
    
process_cabin()

#processing sex; male as 1 and female as 0

def process_sex():
    global combined
    combined['Sex'] = combined.Sex.map({'male': 1, 'female': 0})

process_sex()

#processing Pclass as dummies

def process_class():
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix = 'Pclass')
    combined = pd.concat([combined,pclass_dummies], axis = 1)
    combined.drop('Pclass', axis = 1, inplace = True) 

process_class()


#process ticket

def process_ticket():
    global combined
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
            
    combined['Ticket'] = combined.Ticket.map(cleanTicket)
    ticket_dummies = pd.get_dummies(combined['Ticket'], prefix = 'Ticket')
    combined = pd.concat([combined,ticket_dummies], axis = 1)
    combined.drop('Ticket', axis = 1, inplace = True) 

process_ticket()


# processing Family (SibSp and Parch)
def process_family():
    global combined
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined.FamilySize.map(lambda s: 1 if s == 1 else 0)
    combined['LargeFamily'] = combined.FamilySize.map(lambda s: 1 if 2<=s<=4 else 0)
    combined['SmallFamily'] = combined.FamilySize.map(lambda s: 1 if s >4 else 0)

process_family()


#before modelling, we will drop the passenger id column as it conveys no meaningful informtion

combined.drop('PassengerId', axis = 1, inplace = True)

combined = combined.apply(lambda r: r/max(r))


# Modelling - Feature Selection

def partition_sets():
    global combined
    train0 = pd.read_csv('../input/train.csv')
    target = train0.Survived
    train = combined.head(891)
    test  = combined.iloc[891:]
    return train, test, target

train, test, target = partition_sets()


rforest = RandomForestClassifier(n_estimators=50, max_features='sqrt')
rforest.fit(train, target)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = rforest.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 15))

model = SelectFromModel(rforest, prefit = True)
train_reduced = model.transform(train)
test_reduced  = model.transform(test)


parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 100, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
model = RandomForestClassifier(**parameters)
model.fit(train, target)
    
pred = model.predict(test).astype(int)

# submitting prediction and writing out to the test set
aux = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": aux["PassengerId"],
        "Survived"   : pred
    })

#merged = submission.merge(aux, on = "PassengerId")
#merged.to_csv('titanic.csv', index=False)

submission.to_csv('titanic.csv', index=False)
