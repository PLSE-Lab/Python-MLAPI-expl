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
titanic_train = clean_names(titanic_train)
titanic_train = clean_unembarkedPassengers(titanic_train)
titanic_train['Title'] = titanic_train['Title'].apply(standardize_titles)
titanic_train['Cabin'] = titanic_train['Cabin'].apply(clean_cabin)
titanic_test = clean_names(titanic_test)
titanic_test = clean_unembarkedPassengers(titanic_test)
titanic_test['Title'] = titanic_test['Title'].apply(standardize_titles)
titanic_test['Cabin'] = titanic_test['Cabin'].apply(clean_cabin)
print('Standardized all titles')
print('Cleansed cabin names')

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
def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    from sklearn.naive_bayes import GaussianNB
    return GaussianNB().fit(features_train, labels_train)

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

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

#men_data = titanic_train.loc[titanic_train['Sex'] == 'male']
#women_data = titanic_train.loc[titanic_train['Sex'] == 'female']
#age_data = titanic_train['Age'].fillna(titanic_train['Age'].mean())

features_train = [[data1, data2 == 'male'] for data1, data2 in zip(titanic_train['Pclass'], titanic_train['Sex'])]
labels_train = [data3 ==1 for data3 in titanic_train['Survived']]

features_test = [[data1, data2 == 'male'] for data1, data2 in zip(titanic_test['Pclass'], titanic_test['Sex'])]

clf = classify(features_train, labels_train)

print("Accuracy {}".format(NBAccuracy(features_train, labels_train, features_train, labels_train)))

#print features_test
pred = clf.predict(features_test)


titanic_test['Survived'] = pred

#print(titanic_test)

#predictions_file = open("../csv/genderclassmodel.csv", "wb")
#p = csv.writer(predictions_file)
#p.writerow(["PassengerId", "Survived"])
titanic_test.to_csv("MarkusPredictions.csv", header=True) 


#print NBAccuracy(features_train, labels_train, features_test, labels_test)

#plt.figure()
#plt = plotTwoVars(clf, features_train, labels_train)
#plt.show()
