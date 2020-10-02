##import required packages
## read csv files
## preview data
##find the relationship among parameters
##drop unnecessary columns
##find all missing values and fill those gaps
##fit to the classifier
##predict

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

##extract csv files into dataframe
train_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
'''
### split name 

train_set["Name"] = train_set["Name"].str.rsplit('.',expand=True)
train_set["Name1"] = train_set["Name"].str.rsplit(' ',n=1).str[1]
train_set["Name2"] = train_set["Name"].str.rsplit(' ',n=1).str[0]


test_set["Name"] = test_set["Name"].str.rsplit('.',expand=True)
test_set["Name1"] = test_set["Name"].str.rsplit(' ',n=1).str[1]
test_set["Name2"] = test_set["Name"].str.rsplit(' ',n=1).str[0]

## preview the data
print(train_set.head())
print(test_set.Name1[168])
'''
# create new column in test and intialize it to zero
#test_set = test_set.copy()

#test_set["Survived"] = 0

# find the relationship among parameters
# drop the uncessary column which are not useful for predictions
train_set = train_set.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch'],axis=1)
test_set = test_set.drop(['Name','Cabin','Ticket','SibSp','Parch'],axis=1)



'''
########## fill missing values in age column #########
train_set["Name1"][train_set["Name1"] == "Master"] = 1
train_set["Name1"][train_set["Name1"] == "Miss"] = 2
train_set["Name1"][train_set["Name1"] == "Mrs"] = 3
train_set["Name1"][train_set["Name1"] == "Mr"] = 4
train_set["Name1"][train_set["Name1"] == "Dr"] = 5

test_set["Name1"][test_set["Name1"] == "Master"] = 1
test_set["Name1"][test_set["Name1"] == "Miss"] = 2
test_set["Name1"][test_set["Name1"] == "Mrs"] = 3
test_set["Name1"][test_set["Name1"] == "Mr"] = 4
test_set["Name1"][test_set["Name1"] == "Ms"] = 5



print(train_set.head())
print(test_set.head())

count_nan = train_set["Name1"][train_set["Name1"] == 2]
null = np.sum(count_nan)
print(null)
# fill missing values with random numbers

rand_train1 = np.random.randint(1,8,size=40)
rand_train2 = np.random.randint(9,34,size=36)
rand_train3 = np.random.randint(24,47,size=17)
rand_train4 = np.random.randint(20,45,size=177)

rand_test_master = np.random.randint(2,12,size=4)
rand_test_miss = np.random.randint(11,32,size=14)
rand_test_mrs = np.random.randint(23,53,size=10)
rand_test_mr = np.random.randint(20,44,size=57)

#sample = np.isnan(train_set["Age"][train_set["Name1"] == 2])

#train_set["Name1"][train_set["Name1"] == 1] = rand_train1

#print(train_set.Age[28])
'''

################ fare  ################
##fill the missing columns in fare
test_set["Fare"].fillna(test_set["Fare"].median(),inplace=True)

'''
## to make computation easy group the fare column
train_set["Fare"][train_set["Fare"] < 10] = 1
train_set["Fare"][(train_set["Fare"] >= 10) & (train_set["Fare"] < 20)] = 2
train_set["Fare"][(train_set["Fare"] >= 20) & (train_set["Fare"] < 30)] = 3
train_set["Fare"][(train_set["Fare"] >= 30) & (train_set["Fare"] < 50)] = 4
train_set["Fare"][(train_set["Fare"] >= 50) & (train_set["Fare"] < 100)] = 5
train_set["Fare"][(train_set["Fare"] >= 100) & (train_set["Fare"] < 200)] = 6
train_set["Fare"][(train_set["Fare"] >= 200)] = 7



test_set["Fare"][test_set["Fare"] < 10] = 1
test_set["Fare"][(test_set["Fare"] >= 10) & (test_set["Fare"] < 20)] = 2
test_set["Fare"][(test_set["Fare"] >= 20) & (test_set["Fare"] < 30)] = 3
test_set["Fare"][(test_set["Fare"] >= 30) & (test_set["Fare"] < 50)] = 4
test_set["Fare"][(test_set["Fare"] >= 50) & (test_set["Fare"] < 100)] = 5
test_set["Fare"][(test_set["Fare"] >= 100) & (test_set["Fare"] < 200)] = 6
test_set["Fare"][(test_set["Fare"] >= 200)] = 7
'''
##convert float to int
#train_set["Fare"] = train_set["Fare"].astype(int)
#test_set["Fare"] = test_set["Fare"].astype(int)



##################### age ######################
## fill the missing columns in age
## get the average,std and NAN values
avg_age_train = train_set["Age"].mean()
std_age_train = train_set["Age"].std()
neg_train = avg_age_train - std_age_train
pos_train = avg_age_train + std_age_train
count_nan_train = train_set["Age"].isnull()
null_train = np.sum(count_nan_train)

avg_age_test = test_set["Age"].mean()
std_age_test = test_set["Age"].std()
neg_test = avg_age_test - std_age_test
pos_test = avg_age_test + std_age_test
count_nan_test = test_set["Age"].isnull()
null_test = np.sum(count_nan_test)

## get random values between (mean – std) and (mean + std)
rand_1 = np.random.randint(neg_train, pos_train, size = null_train)
rand_2 = np.random.randint(neg_test, pos_test, size = null_test)
##fill nan values in age column with random values generated

train_set["Age"][np.isnan(train_set["Age"])] = rand_1
test_set["Age"][np.isnan(test_set["Age"])] = rand_2
## convert from float to int
train_set["Age"] = train_set["Age"].astype(int)
test_set["Age"] = test_set["Age"].astype(int)



###################### Gender ########################
# convert female and male into numerical values ; female=2 and male=1
train_set['Gender'] = train_set['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
test_set['Gender'] = test_set['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
'''
# probability of male and female survived
total_male = np.sum(train_set.Gender == 1)
male_survived = np.sum(train_set["Survived"][train_set["Gender"] == 1])
proportion_male_survived = (male_survived)/(total_male)
print("total_male: %d" %total_male)
print("male_survived: %d" %male_survived)
print("proportion_male_survived: %f" %proportion_male_survived)

total_female = np.sum(train_set["Gender"] == 2)
female_survived = np.sum(train_set["Survived"][train_set["Gender"] == 2])
proportion_female_survived = (female_survived)/(total_female)
print("total_female: %d" %total_female)
print("female_survived: %d" %female_survived)
print("prportion_female_survived: %f" %proportion_female_survived)
'''


#################### Embarked #######################
# fill the missing values in the Embarked column
train_set["Embarked"] = train_set["Embarked"].fillna("S")

#convert S,Q & C to numerical values ; S=1 , Q=2, C=3
train_set['Embarked'] = train_set['Embarked'].map( {'S':1, 'Q':2, 'C':3} ).astype(int)
test_set['Embarked'] = test_set['Embarked'].map( {'S':1, 'Q':2, 'C':3} ).astype(int)
train_set = train_set.drop(["Sex"],axis=1)
test_set = test_set.drop(["Sex"],axis=1)
'''
# proportion of S,Q & C survived

total_S = np.sum(train_set["Embarked"] == 1)
S_survived = np.sum(train_set["Survived"][train_set["Embarked"] == 1])
proportion_S_survived = (S_survived)/(total_S)
print("total_S: %d" %total_S)
print("S_survived: %d" %S_survived)
print("proportion_S_survived: %f" %proportion_S_survived)

total_Q = np.sum(train_set["Embarked"] == 2)
Q_survived = np.sum(train_set["Survived"][train_set["Embarked"] == 2])
proportion_Q_survived = (Q_survived)/(total_Q)
print("total_Q: %d" %total_Q)
print("Q_survived: %d" %Q_survived)
print("proportion_Q_survived: %f" %proportion_Q_survived)


total_C = np.sum(train_set["Embarked"] == 3)
C_survived = np.sum(train_set["Survived"][train_set["Embarked"] == 3])
proportion_C_survived = (C_survived)/(total_C)
print("total_C: %d" %total_C)
print("C_survived: %d" %C_survived)
print("proportion_C_survived: %f" %proportion_C_survived)




####################### pclass ###########################
# proportion of class1,class2 and class3 survived

total_class1 = np.sum(train_set["Pclass"] == 1)
class1_survived = np.sum(train_set["Survived"][train_set["Pclass"] == 1])
proportion_class1_survived = (class1_survived)/(total_class1)
print("total_class1: %d" %total_class1)
print("class1_survived: %d" %class1_survived)
print("proportion_class1_survived: %f" %proportion_class1_survived)

total_class2 = np.sum(train_set["Pclass"] == 2)
class2_survived = np.sum(train_set["Survived"][train_set["Pclass"] == 2])
proportion_class2_survived = (class2_survived)/(total_class2)
print("total_class2: %d" %total_class2)
print("class2_survived: %d" %class2_survived)
print("proportion_class2_survived: %f" %proportion_class2_survived)

total_class3 = np.sum(train_set["Pclass"] == 3)
class3_survived = np.sum(train_set["Survived"][train_set["Pclass"] == 3])
proportion_class3_survived = (class3_survived)/(total_class3)
print("total_class3: %d" %total_class3)
print("class3_survived: %d" %class3_survived)
print("proportion_class3_survived: %f" %proportion_class3_survived)


print(train_set.head())

'''
########################### prediction model##########################
'''
test_set["Survived"][(test_set["Gender"] == 2) & \
                     (((test_set["Age"] < 10) & (test_set["SibSp"] <= 2)) | \
                     (((test_set["Age"] >= 10) & (test_set["Age"] <= 20)) & (test_set["SibSp"] < 2)) | \
                     ((test_set["Age"] > 20) & (test_set["Age"] <= 30)) | \
                     ((test_set["Age"] > 30) & (test_set["Age"] <= 40)) | \
                     ((test_set["Age"] > 40) & (test_set["Age"] <= 50)) | \
                     (test_set["Age"] > 50))] = 1


test_set["Survived"][(test_set["Gender"] == 1) & \
                     (((test_set["Age"] < 10) & (test_set["SibSp"] <= 2)) | \
                     (((test_set["Age"] >= 10) & (test_set["Age"] <= 20)) & (test_set["SibSp"] < 2)) | \
                     ((test_set["Age"] > 20) & (test_set["Age"] <= 30)) | \
                     ((test_set["Age"] > 30) & (test_set["Age"] <= 40)) | \
                     ((test_set["Age"] > 40) & (test_set["Age"] <= 50)) | \
                     (test_set["Age"] > 50))] = 1

'''
'''
test_set["Survived"][((test_set["Gender"] == 2) & (test_set["Pclass"] == 1) & \
                     ((((test_set["Embarked"] == 3) & (test_set["Fare"] > 50)) | \
                     ((test_set["Embarked"] == 3) & (test_set["Age"] < 50))) | \
                     (test_set["Embarked"] == 2) | \
                     ((test_set["Embarked"] == 1) & (test_set["Cabin"] != "C23 C25 C27"))))] = 1
                     

test_set["Survived"][((test_set["Gender"] == 2) & (test_set["Pclass"] == 2) & \
                     ((test_set["Age"] < 20) | \
                     (test_set["Embarked"] == 3) | \
                     ((test_set["Fare"] != 10.5) & (test_set["Fare"] != 13) & (test_set["Fare"] != 21) & (test_set["Fare"] != 26)) | \
                     ((test_set["Fare"] == 26) & (test_set["Age"] < 50)) | \
                     (test_set["Fare"] == 21) | \
                     (test_set["Fare"] == 13) | \
                     (test_set["Fare"] == 10.5)))] = 1

test_set["Survived"][((test_set["Gender"] == 2) & (test_set["Pclass"] == 3) & \
                     ((test_set["Embarked"] == 3) & (test_set["Fare"] != 7.2292)))] = 1

print(test_set.head())
print(test_set.Survived[24])

# proportion of male and female survived test set

total_female_test = np.sum(test_set["Gender"] == 2)
female_survived_test = np.sum(test_set["Survived"][test_set["Gender"] == 2])
proportion_female_survived_test = (female_survived_test)/(total_female_test)
print("total_female_test: %d" %total_female_test)
print("female_survived_test: %d" %female_survived_test)
print("prportion_female_survived_test: %f" %proportion_female_survived_test)

total_male_test = np.sum(test_set.Gender == 1)
male_survived_test = np.sum(test_set["Survived"][test_set["Gender"] == 1])
proportion_male_survived_test = (male_survived_test)/(total_male_test)
print("total_male_test: %d" %total_male_test)
print("male_survived_test: %d" %male_survived_test)
print("proportion_male_survived_test: %f" %proportion_male_survived_test)
'''


## define training and test set
X_train = train_set.drop(["Survived"],axis=1)
Y_train = train_set["Survived"]
X_test = test_set.drop(["PassengerId"],axis=1).copy()
#RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=100,random_state=1)
clf = GaussianNB()
clf.fit(X_train,Y_train)
print(clf.score(X_train,Y_train))
Y_pred = clf.predict(X_test)
output = pd.DataFrame({"PassengerId":test_set["PassengerId"],"Survived":Y_pred})
output.to_csv('output.csv',index=False)


