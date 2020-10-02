''' This code will import the relevant files for data analysis purposes.
    The data file is located in ../input/train.csv '''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # package to embellish graphs / plots
import matplotlib.pyplot as plt
import pylab as pl

# Import the train.csv file into a dataframe
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

# Get an overview of the various statistics of titanic_train
print(titanic_train.describe())

# Identify the missing entries for the various indices in order to better assess what data needs to be cleansed
print(titanic_train.info())

def clean_names(df):
    df['LastName'], df['FirstName'] = df['Name'].str.split(', ', 1).str
    df['Title'], df['FirstName'] = df['FirstName'].str.split('. ', 1).str
    del df['Name']
    print('Split name columns in FirstName / LastName / Title')
    return df

# Remove passengers that didn't embark (i.e. resulting in survivor rate higher than expected)
def clean_unembarkedPassengers(df):
    numPassengers = len(df)
    df = df.drop(titanic_train.Embarked.notnull())
    print('Removed {} passengers that have not embarked'.format(numPassengers - len(df)))
    return df

def clean_cabin(cabin):

    cabinString = str(cabin)
    if cabinString == 'nan':
        return 'Z'
    else:
        return cabinString[0]

def standardize_titles(title):
    if title == 'Mr':
        return 'Mr'
    elif title == 'Mrs':
        return 'Mrs'
    elif title == 'Mme':
        return 'Mme'
    elif title == 'Miss':
        return 'Miss'
    else:
        return 'Special'

# Run data cleanup
#titanic_train = clean_names(titanic_train)
#titanic_train = clean_unembarkedPassengers(titanic_train)
#titanic_train['Title'] = titanic_train['Title'].apply(standardize_titles)
#titanic_train['Cabin'] = titanic_train['Cabin'].apply(clean_cabin)
#titanic_test = clean_names(titanic_test)
#titanic_test = clean_unembarkedPassengers(titanic_test)
#titanic_test['Title'] = titanic_test['Title'].apply(standardize_titles)
#titanic_test['Cabin'] = titanic_test['Cabin'].apply(clean_cabin)
#print('Standardized all titles')
#print('Cleansed cabin names')

def calculateSurvivorRatebyCriteria(df, criteria):
    print('Overall survival rate is: {}'.format(df['Survived'].mean()))
    survivalRateByCriteria = titanic_train.groupby(criteria).sum()['Survived'] / titanic_train.groupby(criteria).count()['Survived']
    #print survivalRateByCriteria
    ax = survivalRateByCriteria.plot(kind='barh',title='Survival rate by: {}'.format(criteria))

def CompareWithPieCharts(df1,df2, desc1, desc2):
    survivalRate1 = df1['Survived'].mean()
    survivalRate2 = df2['Survived'].mean()
    pieData = [{desc1: survivalRate1, desc2: survivalRate2}, {desc1: 1-survivalRate1, desc2: 1-survivalRate2}]

    df = pd.DataFrame(pieData, index=['Survived', 'Deceased'])
    return df.plot.pie(subplots = True,figsize=(8,4),
                       title = 'Survival rate of {} vs. {} (in %)'.format(desc1,desc2), colormap='Pastel2',#colors=['g','r'],
                      legend = False, autopct='%1.1f%%')

cherbourg_data = titanic_train.loc[titanic_train['Embarked'] == 'C']
CompareWithPieCharts(cherbourg_data, titanic_train, 'Cherbourg Passengers', 'All Passengers')

def plotTwoVars(clf, X_test, y_test):
    x_min = 0.0; x_max = 600.0
    y_min = 0.0; y_max = 100.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("Fare")
    plt.ylabel("Class")

    #plt.savefig("test.png")
    return plt

#########################################################
### your code goes here ###
def classifyML(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    #from sklearn import svm
    from sklearn import ensemble
    from sklearn import tree
    dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(features_train, labels_train)

    return ensemble.AdaBoostClassifier(base_estimator=dt_stump, n_estimators=100, learning_rate=0.75).fit(features_train, labels_train)
    #return svm.SVC(kernel='linear').fit(features_train, labels_train)

def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for GaussianNB
    from sklearn import svm

    ### create classifier
    clf = svm.SVC()

    ### fit the classifier on the training features and labels
    #t0 = time()
    clf.fit(features_train, labels_train)
    #print("training time: {}s".format(round(time()-t0,3)))

    ### use the trained classifier to predict labels for the test features
    #t0 = time()
    pred = clf.predict(features_test)
    #print("training time: {}s".format(round(time()-t0,3)))

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    return accuracy

#########################################################

titanic_train_men = titanic_train.loc[titanic_train['Sex'] == 'male']
titanic_train_women = titanic_train.loc[titanic_train['Sex'] == 'female']
titanic_test_men = titanic_test.loc[titanic_test['Sex'] == 'male']
titanic_test_women = titanic_test.loc[titanic_test['Sex'] == 'female']

#women_data = titanic_train.loc[titanic_train['Sex'] == 'female']
#age_data = titanic_train['Age'].fillna(titanic_train['Age'].mean())


features_train_men = [[data1, data2] for data1, data2 in zip(titanic_train_men['Pclass'], titanic_train_men['Fare'].fillna(titanic_train_men['Fare'].mean()))]
labels_train_men = [data3 ==1 for data3 in titanic_train_men['Survived']]

features_train_women = [[data1, data2] for data1, data2 in zip(titanic_train_women['Pclass'], titanic_train_women['Fare'].fillna(titanic_train_women['Fare'].mean()))]
labels_train_women = [data3 ==1 for data3 in titanic_train_women['Survived']]


features_test_men = [[data1, data2] for data1, data2 in zip(titanic_test_men['Pclass'], titanic_test_men['Fare'].fillna(titanic_test_men['Fare'].mean()))]
features_test_women = [[data1, data2] for data1, data2 in zip(titanic_test_women['Pclass'], titanic_test_women['Fare'].fillna(titanic_test_women['Fare'].mean()))]


clf_men = classifyML(features_train_men, labels_train_men)
clf_women = classifyML(features_train_women, labels_train_women)

#print("Accuracy {}".format(SVMAccuracy(features_train, labels_train, features_train, labels_train)))

#print features_test
pred_men = clf_men.predict(features_test_men)
titanic_test_men['Survived'] = pred_men * 1

pred_women = clf_women.predict(features_test_women)
titanic_test_women['Survived'] = pred_women * 1

#titanic_test.loc[titanic_test['Sex'] == 'male']['Survived'] = pred_men * 1
#titanic_test.loc[titanic_test['Sex'] == 'female']['Survived'] = pred_women * 1
frames = [titanic_test_men, titanic_test_women]
titanic_test_results = pd.concat(frames).sort(['PassengerId'], ascending=True)

#print(titanic_test_women)
#print(titanic_test)

del titanic_test_results['Pclass']
del titanic_test_results['Name']
del titanic_test_results['Sex']
del titanic_test_results['Age']
del titanic_test_results['SibSp']
del titanic_test_results['Parch']
del titanic_test_results['Ticket']
del titanic_test_results['Fare']
del titanic_test_results['Cabin']
del titanic_test_results['Embarked']

#predictions_file = open("../csv/genderclassmodel.csv", "wb")
#p = csv.writer(predictions_file)
#p.writerow(["PassengerId", "Survived"])
titanic_test_results.to_csv("MarkusPredictions.csv", header=True, index=False)


#print NBAccuracy(features_train, labels_train, features_test, labels_test)

#plt.figure()
#plt = plotTwoVars(clf, features_train, labels_train)
#plt.show()
