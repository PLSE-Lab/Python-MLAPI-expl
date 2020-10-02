# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# read data
test_df = pd.read_csv("../input/test.csv")
train_df = pd.read_csv("../input/train.csv")

print(train_df.info())
print(train_df.describe())

# see missing values
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(12))

# fill embarked nulls by backward fill
# we can also remove the values beacause there are 2 rows only
train_df['Embarked'] = train_df['Embarked'].fillna(method='bfill')
test_df['Embarked'] = test_df['Embarked'].fillna(method='bfill')

# see missing values to check
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#correlation matrix to see the correlation
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True , fmt='.2f');
plt.show()

# see column names
print(train_df.columns.values)

categorical_features_indexes = [ 4  , 2]
independent_variable_indexes_for_feuture_scaling = [ ]
dependent_variable_index = [ 1 ]

# encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder_X = LabelEncoder()
train_df_as_dic = {}

# Encoding categorical_features
for index in categorical_features_indexes:
    temp = train_df.iloc[:,[index]].values

    try :
        temp[:,0] = labelencoder_X.fit_transform(temp[:,0])

    except :
        temp[:,0] = labelencoder_X.fit_transform(temp[:,0].astype(str))

    onehotencoder = OneHotEncoder(categorical_features = [0]) #which colum should be encoded

    temp = onehotencoder.fit_transform(temp).toarray()
    #dummy variable trap
    temp = temp[:,1:]

    # column index as key
    train_df_as_dic.update({ index : temp })

X_initial = train_df_as_dic[list(train_df_as_dic.keys())[0]]
train_df_as_dic.pop(list(train_df_as_dic.keys())[0], None)

for key in train_df_as_dic:
    X_initial = np.column_stack((X_initial , train_df_as_dic[key] ))

for index in independent_variable_indexes_for_feuture_scaling:
    X_initial = np.column_stack((X_initial , train_df.iloc[:,index] ))

X = X_initial
y = train_df.iloc[:,dependent_variable_index]
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train[0])
#Done with data preprocessing

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print("KNN-->" , round(acc_knn,2,), "%")

# DecisionTreeClassifier classifier
classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0 )
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
acc_decision_tree = round(classifier.score(X_train, y_train) * 100, 2)
print('DecisionTreeClassifier-->' ,round(acc_decision_tree,2,), "%")

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Random Forest-->" , round(acc_random_forest,2,), "%")

results = pd.DataFrame({
    'Model': ['KNN','Random Forest','Decision Tree'],
    'Score': [acc_knn,acc_random_forest,acc_decision_tree]
    })

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head())

# data processing for test dataset
# see missing values to check
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
test_df['Fare'] = test_df['Fare'].fillna(method='bfill')


categorical_features_indexes = [ 3 ,  1]
independent_variable_indexes_for_feuture_scaling = [ ]

labelencoder_X = LabelEncoder()
test_df_as_dic = {}

# Encoding categorical_features
for index in categorical_features_indexes:
    temp = test_df.iloc[:,[index]].values

    try :
        temp[:,0] = labelencoder_X.fit_transform(temp[:,0])

    except :
        temp[:,0] = labelencoder_X.fit_transform(temp[:,0].astype(str))

    onehotencoder = OneHotEncoder(categorical_features = [0]) #which colum should be encoded

    temp = onehotencoder.fit_transform(temp).toarray()
    #dummy variable trap
    temp = temp[:,1:]

    # column index as key
    test_df_as_dic.update({ index : temp })

X_initial = test_df_as_dic[list(test_df_as_dic.keys())[0]]
test_df_as_dic.pop(list(test_df_as_dic.keys())[0], None)

for key in test_df_as_dic:
    X_initial = np.column_stack((X_initial , test_df_as_dic[key] ))

for index in independent_variable_indexes_for_feuture_scaling:
    X_initial = np.column_stack((X_initial , test_df.iloc[:,index] ))

X = X_initial

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X)
#Done with data preprocessing

# DecisionTreeClassifier classifier
y_pred_test = (classifier.predict(X_test))

# Export CSV for submission
passenger = test_df.iloc[:,[0]].values
print(passenger[0])
print(y_pred_test[0])
sub = np.column_stack((passenger ,  y_pred_test ))
df = pd.DataFrame(sub)
df.to_csv("submit.csv" , header = [ 'PassengerId' , 'Survived'] , index=False)