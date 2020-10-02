import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
all_data = pd.concat((train.loc[:,'Pclass': 'Embarked'], test.loc[:,'Pclass': 'Embarked']))
getitle=all_data['Name'].str.split('(.*, )|(\\..*)',  expand=True, n =2)
all_data['Title']=getitle[3]
#numericfeats=all_data.dtypes[all_data.dtypes != "object"].index
#all_data[numericfeats]=all_data[numericfeats].fillna(all_data[numericfeats].mean())


miss_title = ["Mlle", "Ms"]
all_data['Title'][all_data['Title'].isin(miss_title)] = "Miss"
all_data['Title'][all_data['Title'] == "Mme"] = "Mrs"
rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
all_data['Title'][all_data['Title'].isin(rare_titles)] = "rare_titles"
all_data['Title'].value_counts()
all_data['Sex'].value_counts()
all_data['Pclass'] = all_data['Pclass'].astype('category')
all_data['SibSp'] = all_data['SibSp'].astype('category')
all_data['Parch'] = all_data['Parch'].astype('category')


all_data2 = all_data[['Age','Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked','Title']]
##fil in the missing values
all_data2['Embarked']=all_data2['Embarked'].fillna("C")
all_data2.Fare=all_data2['Fare'].fillna(all_data2[(all_data2['Pclass'] == 3) & (all_data2['Embarked'] == 'S')]['Fare'].median())
##imput missing age values
all_data2=pd.get_dummies(all_data2)
test_age = all_data2[all_data2['Age'].isnull()]
test_age = test_age.drop('Age', axis = 1)
train_age = all_data2[all_data2['Age'].notnull()]
train_age = train_age.drop('Age', axis = 1)
y_age = all_data.Age[all_data['Age'].notnull()]
##impute ages usinf an rf regressor
from sklearn.ensemble import RandomForestRegressor
rtr = RandomForestRegressor(n_estimators=2000, n_jobs=2)
rtr.fit(train_age, y_age)
pred_age = rtr.predict(test_age)
all_data2.loc[ (all_data2.Age.isnull()), 'Age' ] = pred_age


#split into train and test
X_train = all_data2[:train.shape[0]]
X_test = all_data2[train.shape[0]:]
y = train.Survived

#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



clf = svm.SVC()
clf.fit(X_train, y)
svc_pred=clf.predict(X_test)
solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":svc_pred})
solution.to_csv("svc_predsjan3_titanic.csv", index = False)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y)
rf_pred = random_forest.predict(X_test)
random_forest.score(X_train, y)
solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":rf_pred})
solution.to_csv("randomf_preds_titanic.csv", index = False)

clf2 = KNeighborsClassifier(n_neighbors=7)
clf2.fit(X_train, y)
kn_pred = clf2.predict(X_test)
solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":kn_pred})
solution.to_csv("kn_preds_titanic.csv", index = False)

