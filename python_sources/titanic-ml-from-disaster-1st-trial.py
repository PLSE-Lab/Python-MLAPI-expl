# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

""" explore data """

# read data
df_train = pd.read_csv('../input/train.csv')
df_train['Flag'] = 'Train' 

df_test = pd.read_csv('../input/test.csv')
df_test['Flag'] = 'Test'

# combine data (Test starts with passengerid 892)
df_all = pd.concat([df_train,df_test]) 

""" explore training data """

# visualize data
sns.countplot(x='Survived', data=df_train)
sns.boxplot(x='Survived', y='Age', hue='Sex',data=df_train, palette="coolwarm")
sns.barplot(x='Pclass', y='Survived', hue='Sex',data=df_train,estimator=np.sum)

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# summarize data
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



""" missing data """

# check for missing data
pd.isnull(df_all).sum().sum()
sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# impute missing data - Age
df_all.groupby(['Sex','Pclass']).median()['Age']

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):
        if Sex == 'female':
            if Pclass == 1:
                return 36
            elif Pclass == 2:
                return 28
            else:
                return 22       
        else:
            if Pclass == 1:
                return 42
            elif Pclass ==2:
                return 29.5
            else:
                return 25       
    else:
        return Age

df_all.Age = df_all[['Age','Pclass','Sex']].apply(impute_age,axis=1)

# impute missing data - Embarked
sns.countplot(x='Embarked',data=df_all)
df_all.Embarked = df_all.Embarked.fillna(value='S')

# impute missing data - Fare
df_all['Fare'].groupby(df_all.Embarked).mean()

def impute_fare(cols):
    Fare = cols[0]
    Embarked = cols[1]
    
    if pd.isnull(Fare):
        if Embarked == 'C':
            return 62.34
        elif Embarked == 'Q':
            return 12.41
        else:
            return 27.53       
    return Fare

df_all.Fare = df_all[['Fare','Embarked']].apply(impute_fare,axis=1)

# impute missing data - Cabin
def simplify_cabin(df):
    df.Cabin = df.Cabin.fillna('NA')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

df_all = simplify_cabin(df_all)
sns.countplot(x='Cabin', data=df_all)


""" feature engineering - part 1 """

def simplify_age(df):
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_fare(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df
    
def extract_title(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.')
    return df   
     
def format_pclass(df):
    df.Pclass = df.Pclass.apply(lambda x: str(x))
    return df   

def create_famsize(df):
    df['Famsize'] = df.SibSp + df.Parch
    return df
    

def drop_features(df):
    return df.drop(['Ticket','Name','SibSp','Parch'], axis=1) 

def transform_features(df):
    df = simplify_age(df)
    df = simplify_fare(df)
    df = extract_title(df)
    df = format_pclass(df)
    df = create_famsize(df)
    df = drop_features(df)
    
    return df 

df_all = transform_features(df_all)


""" feature engineer - part 2 """

# format 'Title'
abc = df_all.Title.value_counts()
ax = abc.plot(kind='barh')
ax.invert_yaxis()

df_all.Title = df_all.Title.replace('Mlle','Miss')
df_all.Title = df_all.Title.replace('Ms','Miss')
df_all.Title = df_all.Title.replace('Mme','Mrs')
df_all.Title = df_all.Title.replace(['Rev','Dr','Col',
                                   'Major','Don','Lady',
                                   'Dona','Countess','Jonkheer',
                                   'Capt','Sir'],'Rare')
 
# format 'Alone'
df_all['Alone'] = 0
df_all.loc[df_all['Famsize'] == 1, 'Alone'] = 1    
df_all = df_all.drop('Famsize',axis=1)
    
# encode categorical features
df_all = pd.get_dummies(df_all,
                          columns=['Age','Cabin','Embarked','Fare','Pclass','Sex','Title'], 
                          prefix=['Age','Cabin','Embark','Fare','Class','Gender','Title'], 
                          drop_first=True)

""" split data """

# set training set
df_train_new = df_all[df_all.Flag =='Train']
df_train_new = df_train_new.drop('Flag', axis=1)

# split train into subsets: train and test sets
X = df_train_new.drop(['PassengerId','Survived'], axis=1)
y = df_train_new.Survived

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# scale features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


""" build logistic regression model """

# training the model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# testing the model
pred_log = logmodel.predict(X_test)

# evaluate the model 
#from sklearn.metrics import confusion_matrix, classification_report
#print(confusion_matrix(y_test, pred_log))
#print(classification_report(y_test,pred_log))

""" build decision tree """

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree.fit(X_train, y_train)

# Predicting the Test set results
pred_dt = dtree.predict(X_test)

# evaluate the model 
#print(confusion_matrix(y_test, pred_dt))
#print(classification_report(y_test,pred_dt))


""" build random forest """

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
trees = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
trees.fit(X_train, y_train)

# testing the model
pred_rfc = trees.predict(X_test)

# evaluate the model 
#print(confusion_matrix(y_test, pred_rfc))
#print(classification_report(y_test,pred_rfc))


""" build kernel support vector machine """

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
kernel = SVC(kernel = 'rbf', random_state = 0)
kernel.fit(X_train, y_train)

# testing the model
pred_svmk = kernel.predict(X_test)

# evaluate the model
#print(confusion_matrix(y_test, pred_svmk))
#print(classification_report(y_test,pred_svmk))


""" build linear support vector machine """

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
linear = SVC(kernel = 'linear', random_state = 0)
linear.fit(X_train, y_train)

# testing the model
pred_svml = linear.predict(X_test)

# evaluate the model
#print(confusion_matrix(y_test, pred_svml))
#print(classification_report(y_test,pred_svml))


""" build gaussian naive bayes  """

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)

# Predicting the Test set results
pred_gnb = naive.predict(X_test)

# evaluate the model
#print(confusion_matrix(y_test, pred_gnb))
#print(classification_report(y_test,pred_gnb))

""" evaluate models """

# compute accuracy scores, (60% is acceptable, 70% is good, 85% is very good)
acc_log = round(logmodel.score( X_test , y_test ) * 100, 2)
acc_dt = round(dtree.score( X_test , y_test ) * 100, 2)
acc_rfc = round(trees.score( X_test , y_test ) * 100, 2)
acc_svmk = round(kernel.score( X_test , y_test ) * 100, 2)
acc_svml = round(linear.score( X_test , y_test ) * 100, 2)
acc_gnb = round(naive.score( X_test , y_test ) * 100, 2)

models = pd.DataFrame({
        'Model':['Logistic Regression','Decision Tree','Random Forest','Kernel SVM', 'Linear SVM','Naive Bayes'],
        'Score':[acc_log, acc_dt, acc_rfc, acc_svmk, acc_svml, acc_gnb]
        })

models.sort_values(by='Score', ascending=False)
    

""" predict the actual Test data """

# set test data and label IDs
df_test_new = df_all[df_all.Flag =='Test']
df_test_new = df_test_new.drop(['Flag','Survived'], axis=1)
ids = df_test_new['PassengerId']

# scale features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_new = sc.fit_transform(df_test_new.drop('PassengerId',axis=1))

# run best model (random forest)
predictions = logmodel.predict(X_new)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.head()
#output.to_csv('titanic-predictions.csv', index = False)