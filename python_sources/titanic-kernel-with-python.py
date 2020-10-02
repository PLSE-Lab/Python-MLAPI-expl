# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

# Plots and Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt 

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import module to split dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics


plt.rc("font", size=14)
# Setup background and colors for 
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Categorical/Ordinal Data Plot
def cat_plot(df, feature_name, target_name, palettemap): 
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
    sns.countplot(x=feature_name, hue=target_name, data=df,
                  palette=palettemap,ax=axis1)
    plt.show()

### Step 2: Importing from data source

# Importing the training dataset
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Consolidating the dataset
full_set = train_data.append(test_data,sort=False)
full_set.set_index('PassengerId', inplace = True)
full_set = full_set.drop('Ticket', axis=1)
print("The titanic dataset has {} samples with {} features each.".format(*train_data.shape))

### Step 3: Data Exploration

# Find survivors and fatalities based "Survived flag"
n_survivors = (train_data['Survived']==1).sum()
n_fatalities = (train_data['Survived']==0).sum()


# Plot the data visualization people survived and fatal
sns.countplot(x='Survived',data=train_data,palette="hls")
plt.show()
plt.savefig('count_plot')

# Plot the data visualization people survived and fatal by Sex
pd.crosstab(train_data.Sex,train_data.Survived).plot(kind='bar')
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('# of people')
plt.savefig('Survival people by sex')

# Find the % of survivors in the fatal titanic accident
survived_percent = float(n_survivors) / len(train_data) * 100
print("Percentage of passengers who survived the titanic disaster: {:.2f}%".format(survived_percent))


print ("People survived by Travel class")
survival_palette = {0: "red", 1: "green"} # Color map for visualization
cat_plot(train_data, 'Pclass','Survived', survival_palette)

print ("People survived by Travelling along with Spouse and Siblings")
survival_palette = {0: "red", 1: "green"} # Color map for visualization
cat_plot(train_data, 'SibSp', 'Survived', survival_palette)


# DATA PRE-PROCESSING

# Replacing the data based on the data related findings from SarahG https://www.kaggle.com/sgus1318 
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)

# Create function to impute the age value if it is null
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
# Apply function to impute the age for missing values
# The age column is at position 0
# The pclass column is at position 1
# axis -> argument refers to columns instead of rows to apply the impute_age function
train_data['Age'] = train_data[['Age', 'Pclass']].apply(impute_age, axis=1)

## Drop insignificant values

train_data.drop('Cabin', axis=1, inplace=True)

# Use the .get_dummies() method to convert categorical data into dummy values
# train['Sex'] refers to the column we want to convert
# drop_first -> argument avoids the multicollinearity problem, which can undermines
# the statistical significance of an independent variable.
sex = pd.get_dummies(train_data['Sex'], drop_first=True)
embark = pd.get_dummies(train_data['Embarked'], drop_first=True)
# Use  .concat() method to merge the series data into one dataframe
train_data = pd.concat([train_data, sex, embark], axis=1)

train_data.drop(['Sex','Embarked','Ticket','Name','PassengerId'], axis=1, inplace=True)


## Training the data model

# Split data into 'X' features and 'y' target label sets
X = train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y = train_data['Survived']

# Split data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
svm = SVC(probability=True)
# Fit to the training data
svm.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = svm.predict(X_test)

print("Score Support Vector Machines on train data : {}".format(svm.score(X_train, y_train)))

