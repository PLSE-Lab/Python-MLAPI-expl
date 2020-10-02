# Starting with the simplest possible solution, lets see if we can improve upon our scores

import random
import numpy  as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math

# Utility Functions
sns_percent = lambda x: sum(x)/len(x)*100


# Load Datasets
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

### 1. Guessing at random is our null hypothesis
### 50% success rate based on no information, strangely there where entries on the leaderboard worst than this
output_random = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : np.random.randint(0,2, size=len(test)) # random number 0 or 1
})
output_random.to_csv('random.csv', index=False); # score 0.51196 (6993/7071)


### 2. Assume everybody died
### 62% success rate based on the fact that the Titanic overall was a disaster

output_everybody_dead = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : 0
})
output_everybody_dead.to_csv('everybody_dead.csv', index=False) # score 0.62679 (6884/7071)


### 3. Women Only 
test["Gender"] = test["Sex"].map({ "male": 0, "female": 1 }).astype(int)

plt.figure(figsize=[2,2])
plt.title('Survived by Gender')
sns.barplot(x="Sex", y="Survived", data=train, estimator=sns_percent) # http://seaborn.pydata.org/generated/seaborn.barplot.html
plt.savefig('survived_by_gender.png')

output_everybody_dead = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : test["Gender"]
})
output_everybody_dead.to_csv('women_only_survive.csv', index=False) # score 0.76555 



### 4. Women and Children First = actually worse than just guessing women
plt.figure(figsize=[13,5])
plt.title('Survived by Age')
sns.distplot(train[train['Survived']==0]['Age'].dropna().values, bins=range(0, 81, 1), kde=False, axlabel='Died')
sns.distplot(train[train['Survived']==1]['Age'].dropna().values, bins=range(0, 81, 1), kde=False, axlabel='Survived')
plt.savefig('survived_by_age.png')

output_women_and_children = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : ((test['Sex'] == 'female') | (test['Age'] <= 12.0)).astype(int)
})
output_women_and_children.to_csv('women_and_children.csv', index=False)  # score 0.75119


# # Data mappings
# for dataset in [test, train]:
#     dataset['Gender']     = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
#     dataset['CabinClass'] = dataset['Cabin'].astype(str).map(lambda x: re.sub('^(\w)?.*', '\\1', x) if x != "nan" else None )
#     dataset['LogFare']    = dataset['Fare'].astype(float).map(lambda x: math.log(x) if x else None)
#     dataset['Title']      = dataset['Name'].astype(str).map(lambda x: re.findall('(\w+)\.', x)[0])
# test