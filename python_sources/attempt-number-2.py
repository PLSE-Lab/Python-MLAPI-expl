import numpy as np
import pandas as pd
import csv as csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
total = pd.concat([train, test]) #The entire set

#Now let's see which all columns have missing values
#print(total.info())

#Now let's fill the missing value of Fare 
#print(total[ total['Fare'].isnull() ]) #HINT: We know there's only one missing value of fare
#We are going to find the median of all fares which have the same Pclass & Embarked as the missing fare
missing_fare_pclass = total[total['Fare'].isnull()]['Pclass'].values[0] #This is the Pclass of the missing fare
missing_fare_embarked = total[total['Fare'].isnull()]['Embarked'].values[0] #this is the Embarked of the missing fare
dtype_missing_fare = total[(total.Pclass == missing_fare_pclass) & (total.Embarked == missing_fare_embarked)] #This is the datatype containing all entries with the matched Pclass & Embarked
missing_fare_median_fare = dtype_missing_fare[["Fare"]].median().values[0] #This is the median of all the relevant fares
total.loc[ total['Fare'].isnull(), 'Fare'] = missing_fare_median_fare #Now we've replaced the missing value with the median

#Now let's fill the missing value of Embarked
#print(total[ total['Embarked'].isnull() ]) #HINT: We know there are two missing values of embarked, but same ticket number implies same Embarked value for both
#We are going to find which embarked is more frequent for the same Pclass & Cabin type, but if the answer is not very significant, then
#We are going to find the median of all fares which have the same Pclass & Cabin type and allocate Embarked for whichever median is closest to the missing Embarked's fare
total['Cabin_type'] = total['Cabin'].str[0] #Creating a new column containing the Cabin type
total['Cabin_type'] = total['Cabin_type'].fillna('X') #To fill the missing Cabin type values as X
missing_embarked_pclass = total[total['Embarked'].isnull()]['Pclass'].values[0] #This is the Pclass of the missing embarked
missing_embarked_cabintype = total[total['Embarked'].isnull()]['Cabin_type'].values[0] #This is the Cabin type of the missing embarked
missing_embarked_fare = total[total['Embarked'].isnull()]['Fare'].values[0] #This is the fare of the missing embarked
dtype_missing_embarked = total[(total.Pclass == missing_embarked_pclass) & (total.Cabin_type == missing_embarked_cabintype)] #This is the datatype containing all entries with the matched Pclass & Cabin type
missing_embarked_count = dtype_missing_embarked[["Embarked", "PassengerId"]].groupby(["Embarked"],as_index=False).count() #Let's see if the frequency of one embarked is significantly more than the other(s)
#print(missing_embarked_count) #Unfortunately, it's not, 32 vs 31
missing_embarked_fare_matrix = dtype_missing_embarked[["Embarked", "Fare"]].groupby(["Embarked"],as_index=False).median() #This is the median of all fares for cases with matching Pclass & Cabin type
missing_embarked_fare_matrix["Fare_diff"] = missing_embarked_fare_matrix["Fare"] - missing_embarked_fare #This is the difference of the median fares for those cases from the actual fare of the missing embarked case
missing_embarked_min_diff_fare = missing_embarked_fare_matrix['Fare_diff'].min() #This is the minimum difference between the median fares and the actual fare
missing_embarked = missing_embarked_fare_matrix[missing_embarked_fare_matrix.Fare_diff == missing_embarked_min_diff_fare]["Embarked"].values[0] #This is the embarked value for the closest median
total.loc[ total["Embarked"].isnull(), "Embarked"] = missing_embarked #Now we've replaced the missing value with the value obtained above

#Now let's extract the title
total['Title'] = total['Name'].str.replace('(.*, )|(\\..*)', '')
#print(total[["Survived", "Title"]].groupby(["Title"],as_index=False).agg(['mean', 'count'])) #Let's look at the survival % grouped by Titles
#Using the above data, we find that there are a lot of duplicates that can be clubbed together
total['Title'] = total['Title'].replace(['Dr', 'Col', 'Major', 'Sir'], 'Officer')
total['Title'] = total['Title'].replace(['Capt', 'Don', 'Jonkheer'], 'JrOfficer')
total['Title'] = total['Title'].replace(['Dona', 'Lady', 'the Countess'], 'Lady')
total['Title'] = total['Title'].replace(['Mlle', 'Ms'], 'Miss')
total['Title'] = total['Title'].replace('Mme', 'Mrs')
#print(total[["Survived", "Title"]].groupby(["Title"],as_index=False).agg(['mean', 'count'])) #Data looks much cleaner now

#To run regression to fill the missing values of age, we need to ensure that dependant variables are categorized as integer
total['Title_num'] = total['Title'].map( {'Rev': 0, 'JrOfficer': 1, 'Mr': 2, 'Officer': 3, 'Master': 4, 'Miss':5, 'Mrs':6, 'Lady':7} ).astype(int) #For Title
total['Sex_num'] = total['Sex'].map( {'male': 0, 'female': 1} ).astype(int) #For Sex
total['Embarked_num'] = total['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int) #For Embarked
total['Fare_bins'] = pd.qcut(total['Fare'], 12, labels=False) #For Fare, we are using bins dividing the fare into 12 quantiles
total['Pclass_num'] = total['Pclass'] - 1

#Now let's fill the missing value of Age
#We will use Random Forest technique to predict the Age using the following variables - Pclass, Title, Sex, Embarked, Fare, Parch, Sibsp
age_train = total[total['Age'].isnull()==False] #This is the training set where Age is present
age_test = total[total['Age'].isnull()] #This is the test set where Age is not present
X_age_train = age_train.drop(['Age', 'Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Cabin_type', 'Title', 'Sex', 'Embarked', 'Fare'],axis=1) #Dropping the irrelevant values
Y_age_train = age_train['Age'].astype(int) #Converting to int, to facilitate randomforest
X_age_test  = age_test.drop(['Age', 'Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Cabin_type', 'Title', 'Sex', 'Embarked', 'Fare'],axis=1).copy() #Dropping the irrelevant values
age_random_forest = RandomForestClassifier(n_estimators=100)
age_forest = age_random_forest.fit(X_age_train, Y_age_train)
Y_age_pred = age_random_forest.predict(X_age_test)
age_test['Age'] = Y_age_pred #Replacing the null values with the predicted values
total = pd.concat([age_train, age_test]).sort(['PassengerId']).reset_index(drop=True) #Concatenating back the training and test set
total['Age_bins'] = pd.qcut(total['Age'], 12, labels=False) #Now dividing age into 12 bins

#Now let's add another variable of familysize
total['FamilySize'] = total['SibSp'] + total['Parch'] + 1 #Adding a variable for Family Size
#print(total[["Survived", "FamilySize"]].groupby(["FamilySize"],as_index=False).agg(['mean', 'count'])) #Singletons and large families have poor survival rate
total['FamilySizeCategories'] = total['FamilySize'] #For creating bins
total.loc[ total['FamilySize'] == 1, 'FamilySizeCategories' ] = 2 #Singletons
total.loc[ (total['FamilySize'] > 1) & (total['FamilySize'] < 4) , 'FamilySizeCategories' ] = 1 #Small families
total.loc[ total['FamilySize'] == 4, 'FamilySizeCategories' ] = 0 #Medium families
total.loc[ total['FamilySize'] > 4, 'FamilySizeCategories' ] = 3 #Large families
#print(total[["Survived", "FamilySizeCategories"]].groupby(["FamilySizeCategories"],as_index=False).agg(['mean', 'count'])) #Data looks much cleaner now

#Now let's look at survival rate by Cabin type
#print(total[["Survived", "Cabin_type"]].groupby(["Cabin_type"],as_index=False).agg(['mean', 'count'])) #Cabin type seems to have an impact on survival
total["Cabin_num"] = total['Cabin_type'].map( {'A': 2, 'B': 0, 'C': 1, 'D': 0, 'E': 0, 'F': 1, 'G': 2, 'T': 4, 'X': 3} ) #To clean the data
#print(total[["Survived", "Cabin_num"]].groupby(["Cabin_num"],as_index=False).agg(['mean', 'count'])) #Data looks much cleaner now

#Now let's drop irrelevant columns
total = total.drop(['Age', 'Cabin', 'Name', 'Ticket', 'FamilySize', 'Embarked', 'Fare', 'Sex', 'Title', 'Cabin_type'], axis=1) #Dropping irrelevant columns

train = total[0:890]
test = total[891:1309]
X_train = train.drop(['Survived', 'PassengerId'],axis=1)
Y_train = train["Survived"]
X_test  = test.drop(['Survived', 'PassengerId'],axis=1).copy()

random_forest = RandomForestClassifier(n_estimators=100)
forest = random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))

output = Y_pred.astype(int)
ids = test['PassengerId'].values
predictions_file = open("titanic_predict.csv", "w") # Python 3
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
