# %% [code]
import pandas as pd
import numpy as np

# %% [code]
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()

# %% [code]
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.head()

# %% [code]
df.describe()

# %% [code]
df.info()

# %% [markdown]
# ## Missing Values

# %% [markdown]
# Drop Name as having text values
# 
# Drop Cabin as having 687 missing values

# %% [code]
df.drop(["Name", "Cabin"], axis=1, inplace=True)
test_df.drop(["Name", "Cabin"], axis=1, inplace=True)

# %% [markdown]
# Convert na values in Embarked from Train and Fare from Test Data 

# %% [code]
test_df['Fare'].describe()

# %% [code]
df["Embarked"] = df["Embarked"].fillna('S')
test_df["Fare"]  =test_df["Fare"].fillna('35.627')

# %% [markdown]
# Missing value of Age

# %% [code]
df['Age'].describe()

# %% [code]
data = [df]

for dataset in data:
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df["Age"].astype(int)
df["Age"].isnull().sum()

# %% [code]
test_df['Age'].describe()

# %% [code]
data = [test_df]

for dataset in data:
    mean = test_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = test_df["Age"].astype(int)
test_df["Age"].isnull().sum()

# %% [code]
missing_data = df.isnull().sum()
missing_data

# %% [code]
df.info()

# %% [markdown]
# ##  Converting values to Numeric

# %% [markdown]
# convert Fare to Int

# %% [code]
df['Fare'] = df['Fare'].astype(int)
test_df['Fare'] = df['Fare'].astype(int)

# %% [markdown]
# Converting SEX to Numeric

# %% [code]
sex = pd.get_dummies(df['Sex'])
df["Male"] = sex["male"].astype(int)
df.drop(["Sex"], axis=1, inplace=True)

# %% [code]
sex = pd.get_dummies(test_df['Sex'])
test_df["Male"] = sex["male"].astype(int)
test_df.drop(["Sex"], axis=1, inplace=True)

# %% [code]
df['Ticket'].describe()

# %% [markdown]
# Since Ticket having 681 unique values difficult to convert them so drop the tiket

# %% [code]
df.drop(['Ticket'], axis=1, inplace=True)
test_df.drop(['Ticket'], axis=1, inplace=True)

# %% [markdown]
# Convert Embarked into Numeric

# %% [code]
ports = {"S": 0, "C": 1, "Q": 2}
data = [df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# %% [code]
df.info()

# %% [code]


# %% [markdown]
# Normalise Data

# %% [code]
x_train = df.drop({"PassengerId","Survived"}, axis=1)
y = df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

# %% [code]
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit(x_train).transform(x_train)
X[0:5]

# %% [code]
from sklearn import preprocessing
x_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)
x_test[0:5]

# %% [markdown]
# # Models

# %% [code]
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

# %% [markdown]
# ## KNN

# %% [markdown]
# 

# %% [code]
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Finding Best K

ks = 10
mean_acc =np.zeros((ks-1))
std_acc = np.zeros((ks-1))
Confusionmx = [];
for n in range(1,ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X, y)
    yhat = neigh.predict(X)
    mean_acc[n-1] = metrics.accuracy_score(y, yhat)
    
    std_acc[n-1] = np.std(yhat == y)/np.sqrt(yhat.shape[0])
    
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# %% [code]
k = 1
knn = KNeighborsClassifier(n_neighbors = k).fit(X, y)
yhat = knn.predict(X)

knnscore = metrics.accuracy_score(y, yhat)
f1_knn = f1_score(y, yhat, average='weighted')
j_knn = jaccard_similarity_score(y, yhat)

# %% [markdown]
# ## Decision Tree

# %% [code]
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
tree.fit(X, y)
yhat = tree.predict(X)

treescore = metrics.accuracy_score(y, yhat)
f1_tree = f1_score(y, yhat, average='weighted')
j_tree = jaccard_similarity_score(y, yhat)

# %% [markdown]
# ## Support Vector Machine

# %% [code]
from sklearn import svm
svm = svm.SVC(kernel= 'rbf', gamma = 'scale')
svm.fit(X, y)
yhat = svm.predict(X)

svmscore = metrics.accuracy_score(y, yhat)
f1_svm = f1_score(y, yhat, average='weighted')
j_svm = jaccard_similarity_score(y, yhat)

# %% [markdown]
# ## Logistic Regerassion

# %% [code]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X, y)
yhat = LR.predict(X)

lrscore = metrics.accuracy_score(y, yhat)
f1_lr = f1_score(y, yhat, average='weighted')
j_lr = jaccard_similarity_score(y, yhat)

# %% [markdown]
# ## Random Forest

# %% [code]
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
yhat = rf.predict(X)

rfscore = metrics.accuracy_score(y, yhat)
f1_rf = f1_score(y, yhat, average='weighted')
j_rf = jaccard_similarity_score(y, yhat)

# %% [markdown]
# ## Best Model ?

# %% [code]
results = pd.DataFrame({'Score': ['KNN', 'Tree', 'SVM', 'LR', 'Random F'], 'Metrics': [knnscore, treescore, svmscore, lrscore, rfscore], 'F1 Score': [f1_knn, f1_tree, f1_svm, f1_lr, f1_rf], 'Jaccard': [j_knn, j_tree, j_svm, j_lr, j_rf]})

result_df = results
result_df = result_df.set_index('Score')
result_df

# %% [code]
randomf = rf.predict(x_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': randomf})
output.to_csv('random2.csv', index=False)
print("Your submission was successfully saved!")

# %% [code]
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
tree.fit(X, y)
yhat = tree.predict(X)

treescore = metrics.accuracy_score(y, yhat)
f1_tree = f1_score(y, yhat, average='weighted')
j_tree = jaccard_similarity_score(y, yhat)

# %% [code]
tree = tree.predict(x_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': tree})
output.to_csv('tree2.csv', index=False)
print("Your submission was successfully saved!")

# %% [code]
from sklearn import svm
svm = svm.SVC(kernel= 'rbf', gamma = 'scale')
svm.fit(X, y)
yhat = svm.predict(X)

svmscore = metrics.accuracy_score(y, yhat)
f1_svm = f1_score(y, yhat, average='weighted')
j_svm = jaccard_similarity_score(y, yhat)

# %% [code]
svm = svm.predict(x_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': svm})
output.to_csv('svm2.csv', index=False)
print("Your submission was successfully saved!")

# %% [code]
