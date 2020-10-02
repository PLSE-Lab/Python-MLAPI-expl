import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
#Print to standard output, and see the results in the "log" section below after running your script
'''
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
'''

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
#Make Pclass and Survived categorical values
#train['Pclass']=train['Pclass'].astype('category')
#train['Survived']=train['Survived'].astype('category')
#test['Pclass']=test['Pclass'].astype('category')
#test['Survived']=test['Survived'].astype('category')
#train.describe()
#get number of missing values in each column
missing_dict=dict()
columns=train.columns.values
for col in columns:
    if sum(train[col].isnull())>0:
        missing_dict[col]=float(sum(train[col].isnull()))/len(train)
missing_dict

#77% of the values in the cabin field are missing, we can effectively ignore the cabin value for prediction

train=train.drop(['Cabin'],axis=1)
train.columns.values

test=test.drop(['Cabin'],axis=1)
# # 19% of Age values are missing.We can predict age based on remaining features. 
#Understand effect of Pclass on Age
#Get the rows with Pclass and Age where Age is not null
#df=train[train['Age'].notnull()][['Age','Pclass','Name']]
#import matplotlib
# % matplotlib inline

#df.boxplot('Age',by='Pclass')
# * From the above we can observe that, Average age of Passenger travelling in 1st class is 37-38, while that of those travelling in 2nd class is 28-29 and those that travel in 3rd class have an average age < 25
#Extract Surnames from Name
#Name is of the form LastName , SurName. First Name
#df['Name'].head()
def getTitle(name):
    split_name=name.split(",")
    title=split_name[1].split(".")[0].replace(" ","")
    return title
#create a new column called surname is train and test data
#df['Title']=df.apply(lambda row: getTitle(row['Name']), axis=1)
train['Title']=train.apply(lambda row: getTitle(row['Name']),axis=1)
test['Title']=test.apply(lambda row: getTitle(row['Name']),axis=1)

#Effect of SibSp and Parch on Age
#train.boxplot("Age",by="SibSp")
#train.boxplot("Age",by="Parch")
#train['Title']=train['Title'].astype("category")
#test['Title']=test['Title'].astype("category")
# #Observe that SibSp and Parch has an effect on Age
# #Let us apply linear regression using SibSp,Parch,Title,Pclass as predictor variable and Age as the Target Variable
#train.columns.values
#test.columns.values
# combine the test and train data into one entire set. To identify the test and train data,create a field type which indicates
# whether it is train or test data
train['Type']="Train"
test['Type']="Test"
test['Survived']=" "
Survived=train["Survived"]
train=train.drop(['Survived'],axis=1)
train["Survived"]=Survived

#len(full_data)
#train.columns.values

full_data=pd.concat([train,test],axis=0)
full_data.columns.values
#Create a Label Encoder for Title
#Use label encoder to convert Title which is a string to an incremental value

le = LabelEncoder()
target = full_data['Title']
full_data['Title_LabelEncoder'] = le.fit_transform(target)
#Apply Linear Regression using SibSp,Parch,Title_LabelEncoder,PClass to predict missing Age values on full_data

#Divide the full data into test and train to predict the age
test_X_age=full_data[full_data['Age'].isnull()]
train_X_age=full_data[full_data["Age"].notnull() ]

regr = linear_model.LinearRegression()
regr.fit(train_X_age[['SibSp','Parch','Title_LabelEncoder','Pclass']], train_X_age['Age'])
test_X_age['Age']=regr.predict(test_X_age[['SibSp','Parch','Title_LabelEncoder','Pclass']])
new_full_data=pd.concat([train_X_age,test_X_age],axis=0)
#Embarked has 0.2% missing data in train set, replace it by the most common Embarkment place in the new_full_data
df =new_full_data.dropna(subset=['Embarked'])
most_frequent_value=df['Embarked'].value_counts().idxmax()
#most_frequent_value
new_full_data['Embarked']=new_full_data['Embarked'].fillna(most_frequent_value)
#Fare has missing values in test data,replace by average fare 
meanFare = np.mean(new_full_data.Fare)
new_full_data.Fare = new_full_data.Fare.fillna(meanFare)
#Create Label encoder for Sex and Embarked
le = LabelEncoder()
target = new_full_data['Sex']
new_full_data['Sex_LabelEncoder'] = le.fit_transform(target)

target=new_full_data['Embarked']
new_full_data['Embarked_LabelEncoder'] = le.fit_transform(target)
#Write the test and train data after filling the missing values
train_data=new_full_data[new_full_data['Type']=='Train']
test_data=new_full_data[new_full_data['Type']=='Test']
train_data=train_data.drop(['Type'],axis=1)
test_data=test_data.drop(['Type'],axis=1)
train_data.to_csv('train_data_withoutMissingValues.csv', index=False)
test_data.to_csv('test_data_withoutMissingValues.csv', index=False)
train=pd.read_csv("train_data_withoutMissingValues.csv",dtype={"Age": np.float64},)
f=open("submission.csv","w")
f.write("PassengerId,Survived")
f.write("\n")
f.close()
submission=pd.read_csv("submission.csv")
test=test_data
train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]
#Use Logistic Regression to predict Survival Chances
#Predictor variables are Age,Pclass,SibSp,Parch,Title_LabelEncoder,Embarked
features_list=[ 'Pclass','Title_LabelEncoder', 'Sex_LabelEncoder', 'Age', 'Embarked_LabelEncoder','SibSp']
x_train = Train[features_list].values
y_train=Train[['Survived']].values
x_validate=Validate[features_list].values
y_validate=Validate[['Survived']].values
x_test=test[features_list].values
'''
lr=linear_model.LogisticRegression()
lr.fit(x_train,y_train)
predic_validate=lr.predict(x_validate)
lr.score(x_validate,y_validate)
#Get the predicted values for the test data 
submission['PassengerId']=test['PassengerId']
submission['Survived']=lr.predict(x_test)
print(submission.head())
submission.to_csv("submission.csv",index=False)
'''


#Implement Random Forest on titanic data and get the feature importance
submission['PassengerId']=test['PassengerId']
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
print(rf.score(x_validate,y_validate)) #0.808695652174
#print(rf.feature_importances_)
submission['Survived']=rf.predict(x_test)
submission.to_csv("submission.csv",index=False)