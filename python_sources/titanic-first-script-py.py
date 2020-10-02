import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Print you can execute arbitrary python code
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#print description of training set and testing set
#print (train_df.info())
#print ("++++++++++++++++++++++++++")
#print (test_df.info())

#dropping unnecessary columns in both training and testing dataset
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket'], axis = 1)

#Correcting Embarked column
train_df['Embarked'] = train_df['Embarked'].fillna("S")
sns.factorplot('Embarked', 'Survived', data = train_df, size = 4, aspect = 3)
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)

#group by embarked, and get mean for each value in embarked
embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
#print ("+++++++++++++++++++++++++++++")
#print (embark_perc.head())

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic = pd.get_dummies(train_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

#join newly created Embark 'C' and 'Q' columns to test and train dataset
train_df = train_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)


#drop the Embarked column from train and test dataset
train_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

#filling blanks in Fare test data
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

#converting Fare from float to int
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)

#getting Fare of passengers who survived and didn't survive people
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
#print (fare_survived)
#print ("++++++++++++++++++++++++")
#print (fare_not_survived)

#getting average of survived/ not survived passengers
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

#plotting graph
train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, xlim=(0,50))

#naming the index of datasets
average_fare.index.names = std_fare.index.names = ["Survived"]
#print (average_fare)
#print ("+++++++++++++++++++++++++++++++++")
#print (std_fare)

#plotting std fare
average_fare.plot(yerr=std_fare, kind = 'bar', legend = False)

# Age 
#creating of graphs
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# get average, std, and number of NaN values in titanic_df
average_age_train   = train_df["Age"].mean()
std_age_train       = train_df["Age"].std()
count_nan_age_train = train_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
#print (rand_1)

#convert Age values to int. Remember to drop NaN value while doing so
train_df["Age"].dropna().astype("int").hist(bins=70, ax=axis1)

#fill NaN value in Age column with the random numbers generated
train_df["Age"][np.isnan(train_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

#converting Age values from float to int
train_df["Age"] = train_df["Age"].astype(int)
test_df["Age"] = test_df["Age"].astype(int)

#plotting the new age values
train_df["Age"].hist(bins=70, ax=axis2)

#plotting of Age column
#peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()

#average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18, 4))
average_age = train_df[["Age", "Survived"]].groupby(['Age'], as_index = False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)

#Dropping Cabin values because they contain a lot of NaN values
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

#Family
#combining Parch & SibSp into single column Family
train_df['Family'] = train_df['Parch'] + train_df['SibSp']
train_df['Family'][train_df['Family'] > 0] = 1
train_df['Family'][train_df['Family'] == 0] = 0

test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'][test_df['Family'] > 0] = 1
test_df['Family'][test_df['Family'] == 0] = 0

#dropping Parch and SibSp columns from dataset
train_df.drop(['Parch', 'SibSp'], axis = 1, inplace = True)
test_df.drop(['Parch', 'SibSp'], axis = 1, inplace = True)

#plotting data
fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)

#average of people who survived/not survived with/without family on-board
family_perc = train_df[['Family', 'Survived']].groupby('Family', as_index = False).mean()
#print (family_perc)
#plotting graph
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
axis1.set_xticklabels(['WithFamily', 'Alone'], rotation=0)

#Sex
#Classifying passengers as male, female and child
def get_passenger(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train_df['Person'] = train_df[['Age', 'Sex']].apply(get_passenger, axis = 1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_passenger, axis = 1)

#Dropping Sex column since Person column is created
train_df.drop('Sex', axis = 1, inplace = True)
test_df.drop('Sex', axis = 1, inplace = True)

#creating separate columns for Person variables
#Dropping Male as it has lowest Survival rate
person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child', 'Female', 'Male']
person_dummies_train.drop(['Male'], axis = 1, inplace = True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis = 1, inplace = True)

#joining the newly created columns to the main dataset
train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)

#plotting graphs for the new columns
fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))
sns.countplot(x='Person', data=train_df, ax=axis1)

#find average of suvived child/female/male
person_perc = train_df[['Person', 'Survived']].groupby('Person', as_index = False).mean()
#plotting graph for Person
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order = ['male', 'female', 'child'])

#dropping Person column since new Female and Child columns have been created
test_df.drop('Person', axis = 1, inplace = True)
train_df.drop('Person', axis = 1, inplace = True)
#print (train_df.head())

#Pclass
#plotting graph
sns.factorplot('Pclass', 'Survived', order = [1, 2, 3], data = train_df, size =5)

#creating dummy variables for Pclass and dropping class 3 as it has lowest rate of survival
pclass_dummies_train = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_train.drop('Class_3', axis = 1, inplace = True)

pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.drop('Class_3', axis = 1, inplace = True)

train_df.drop('Pclass', axis = 1, inplace = True)
test_df.drop('Pclass', axis = 1, inplace = True)

train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)

#defining the training and testing dataset
X_train = train_df.drop('Survived', axis = 1)
Y_train = train_df["Survived"]
X_test = test_df.drop('PassengerId', axis = 1).copy()
#print (X_train)
#print ("+++++++++++++++++++++")
#print(X_test)

#logistic regression
logi = LogisticRegression()
logi.fit(X_train, Y_train)
Y_pred_logi = logi.predict(X_test)
print ("Logistic Regression: " + str(logi.score(X_train, Y_train)))

#Support vector machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
print ("SVC: " + str(svc.score(X_train, Y_train)))

#Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
print ("Random Forest: " + str(random_forest.score(X_train, Y_train)))

#KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print ("KNN Classifier: " + str(knn.score(X_train, Y_train)))

#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
print ("Gaussian Classifier: " + str(gaussian.score(X_train, Y_train)))

#print (Y_pred_random_forest)

#get the correlation coefficients for each feature using Logistic Regression
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = pd.Series(logi.coef_[0])

#creating final prediction csv file
submission = pd.DataFrame({"PassengerId":test_df['PassengerId'], 'Survived':Y_pred_random_forest})
submission.to_csv('titanic.csv', index = False)


#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)