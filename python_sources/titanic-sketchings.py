import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import (LogisticRegression, Perceptron)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import (SVC, LinearSVC)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                            GradientBoostingClassifier, VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, ShuffleSplit

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_id = test["PassengerId"]

full_dataset = [train, test]

for dataset in full_dataset: 
    # Handle incomplete values
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
    # Feature engineering
    dataset['Name_length'] = dataset['Name'].apply(len)
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
    
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Integer mappings
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} )
    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']  = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
    title_mapping = {"Rare": 0, "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    

drop_columns = ['PassengerId', 'Name', 'Cabin', 'Ticket',
                'FareBin', 'AgeBin', 'SibSp']
train.drop(drop_columns, axis=1, inplace = True)
test.drop(drop_columns, axis=1, inplace = True)

# Correlation Heat Matrix
def pearson_correlation_heatmap(dataset):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(dataset.astype(float).corr(), linewidths=0.1, vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)

# Correlation by survival
def correlation_survival(dataset):
    for x in dataset:
        if (dataset[x].dtype != 'float64') and (x != 'Survived'):
            print('Survival Correlation by:', x)
            print(dataset[[x, 'Survived']].groupby(x, as_index=False).mean().sort_values(by='Survived', ascending=False))
            print('-'*10, '\n')


# Prepare data for training
x_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
# print("xtrain, ytrain, test shapes: ", (x_train.shape, y_train.shape, test.shape))

# ### Simple model performances
def logistic_regression(x_train, y_train):      # achieves 0.82940
    # Binary regression is implied by 'ovr' multiclass
    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    logreg.fit(x_train, y_train)
    return logreg.score(x_train, y_train)

def svc(x_train, y_train):                      # achieves 0.86307
    svc_object = SVC(gamma='auto')
    svc_object.fit(x_train, y_train)
    return svc_object.score(x_train, y_train)

def knn_classifier(x_train, y_train, n):        # achieves 0.96408(k=1), 0.87317(k=[2,3])
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(x_train, y_train)
    return knn.score(x_train, y_train)

def naive_bayes(x_train, y_train):              # achieves 0.79797
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    return gaussian.score(x_train, y_train)

def perceptron(x_train, y_train):               # achieves 0.76206
    model = Perceptron(max_iter=5, tol=None)
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def linear_svc(x_train, y_train):               # achieves 0.82154
    model = LinearSVC()
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def decision_tree(x_train, y_train):            # achieves 0.96520
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def random_forest(x_train, y_train):            # achieves 0.96520
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def ada_boost(x_train, y_train):                # achieves 0.83389
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def grad_boost(x_train, y_train):               # achieves 0.88664
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

def xgbc(x_train, y_train):                     # achieves 0.87766
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model.score(x_train, y_train)

# Cross validation folds
kfold = StratifiedKFold(n_splits=10)

# Simple Model Performances
def model_performance_chart():
    classifiers = [ LogisticRegression(solver='liblinear', multi_class='ovr'), 
                    SVC(gamma='auto'), KNeighborsClassifier(n_neighbors = 1), 
                    KNeighborsClassifier(n_neighbors = 2), GaussianNB(),  
                    DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), 
                    GradientBoostingClassifier(), XGBClassifier()]
    
    cv_results = []
    cv_means = []
    cv_std = []
    
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
    
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
        
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
            "Algorithm":["LogisticRegression","SVC","KNeighboors1","KNeighboors2","NaiveBayes",
                "DecisionTree","RandomForest","GradientBoosting","XGBC"]})
    
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, **{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g.set_title("Cross validation scores")

# First Level - Model
classifiers = [ LogisticRegression(solver='liblinear', multi_class='ovr'), 
                SVC(gamma='auto'), KNeighborsClassifier(n_neighbors = 1), 
                KNeighborsClassifier(n_neighbors = 2), DecisionTreeClassifier(), 
                RandomForestClassifier(n_estimators=100), GradientBoostingClassifier()]

classifier_names = ["LogisticRegression","SVC","KNeighboors1","KNeighboors2",
                    "DecisionTree","RandomForest","GradientBoosting"]

# Layer 1 output correlations
def get_layer_corr():
    pred0 = pd.Series(classifiers[0].fit(x_train,y_train).predict(test), name=classifier_names[0])
    pred1 = pd.Series(classifiers[1].fit(x_train,y_train).predict(test), name=classifier_names[1])
    pred2 = pd.Series(classifiers[2].fit(x_train,y_train).predict(test), name=classifier_names[2])
    pred3 = pd.Series(classifiers[3].fit(x_train,y_train).predict(test), name=classifier_names[3])
    pred4 = pd.Series(classifiers[4].fit(x_train,y_train).predict(test), name=classifier_names[4])
    pred5 = pd.Series(classifiers[5].fit(x_train,y_train).predict(test), name=classifier_names[5])
    pred6 = pd.Series(classifiers[6].fit(x_train,y_train).predict(test), name=classifier_names[6])
    
    predictions = pd.concat([pred0,pred1,pred2,pred3,pred4,pred5,pred6],axis=1)
    return sns.heatmap(predictions.corr(),annot=True)

# Second Layer - Model
# Voting
hard_voting = VotingClassifier(
    estimators=[('classifier_names[0]', classifiers[0]), ('classifier_names[1]', classifiers[1]), 
                ('classifier_names[2]', classifiers[2]), ('classifier_names[3]', classifiers[3]), 
                ('classifier_names[4]', classifiers[4]), ('classifier_names[5]', classifiers[5]), 
                ('classifier_names[6]', classifiers[6])], 
    voting='hard', weights=[0.1,0.1,0.2,0.1,0.2,0.2,0.1], n_jobs=4)
hard_voting = hard_voting.fit(x_train, y_train)
print(hard_voting.score(x_train, y_train))

# Stacking
def Stacking(model, train ,y ,test ,n_fold):
    folds = StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred = np.empty((test.shape[0],1),float)
    train_pred = np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val = train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val = y.iloc[train_indices],y.iloc[val_indices]
        
        model.fit(X=x_train, y=y_train)
        train_pred = np.append(train_pred,model.predict(x_val))
        test_pred = np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred

test_pred0 ,train_pred0 = Stacking(model = classifiers[0], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred1 ,train_pred1 = Stacking(model = classifiers[1], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred2 ,train_pred2 = Stacking(model = classifiers[2], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred3 ,train_pred3 = Stacking(model = classifiers[3], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred4 ,train_pred4 = Stacking(model = classifiers[4], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred5 ,train_pred5 = Stacking(model = classifiers[5], n_fold=10, 
                            train=x_train, test=test, y=y_train)
test_pred6 ,train_pred6 = Stacking(model = classifiers[6], n_fold=10, 
                            train=x_train, test=test, y=y_train)

train_pred0=pd.DataFrame(train_pred0)                            
train_pred1=pd.DataFrame(train_pred1)
train_pred2=pd.DataFrame(train_pred2)                            
train_pred3=pd.DataFrame(train_pred3)
train_pred4=pd.DataFrame(train_pred4)
train_pred5=pd.DataFrame(train_pred5)                            
train_pred6=pd.DataFrame(train_pred6)
test_pred0=pd.DataFrame(test_pred0)
test_pred1=pd.DataFrame(test_pred1)
test_pred2=pd.DataFrame(test_pred2)
test_pred3=pd.DataFrame(test_pred3)
test_pred4=pd.DataFrame(test_pred4)
test_pred5=pd.DataFrame(test_pred5)
test_pred6=pd.DataFrame(test_pred6)

df = pd.concat([train_pred0, train_pred1, train_pred2, train_pred3, train_pred4, train_pred5, train_pred6], axis=1)
df.columns = classifier_names
df_test = pd.concat([test_pred0, test_pred1, test_pred2, test_pred3, test_pred4, test_pred5, test_pred6], axis=1)
df_test.columns = classifier_names

model = XGBClassifier()
model.fit(df, y_train)
model.score(df, y_train)

test_output = pd.Series(hard_voting.predict(test), name="Survived")
results = pd.concat([test_id,test_output],axis=1)
results.to_csv("titanic.csv",index=False)

# ###Training set descriptions
# train.info()
# train.sample(1)
# train.head(5)
# train.describe()
# train['Age'].describe()

# ###Training set descriptions
# test.info()
# test.head(5)
# test.describe()

# print(train['Title'].value_counts())
# print(train.isnull().sum())
# print(test.isnull().sum())