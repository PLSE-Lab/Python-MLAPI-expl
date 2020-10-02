import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn import ensemble
import re
#functions
def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
def IsAlone(familySize):
    "determines if a person who is in a family of familySize people is alone"
    return int(familySize==0)
def replaceNaByAverage(colonne):
    moyenne = int(colonne.mean())
    colonne = colonne.fillna(moyenne)
    return colonne
def ExtractTitle(name):
    title_search = re.search('[A-Za-z]+\.', name)
    if title_search:
        return title_search.group()
    else:
        return None
def MapTitle(title):
    if (title=='Mr.'):
        return 0
    elif (title=='Miss.'):
        return 1
    elif (title=='Mrs.'):
        return 2
    elif(title== 'Master.'):
        return 3
    else:
        return 4
def MapTicket(ticket):
    return len(ticket)
def MapCabin(cabin):
    if (cabin == 0):
        return 0
    else:
        return 1
def featureEngineering(dataFrame, normalize):
    "changes the data for a better approach"
    dataFrame['TicketSize'] = dataFrame['Ticket'].apply(MapTicket)
    dataFrame['Cabin'] = dataFrame['Cabin'].fillna(0).apply(MapCabin)  
    dataFrame['NameLength'] = dataFrame['Name'].apply(len)
    dataFrame['FamilySize'] = dataFrame['SibSp'] + dataFrame['Parch']
    dataFrame['IsAlone'] = dataFrame['FamilySize'].apply(IsAlone)
    dataFrame['Title'] = dataFrame['Name'].apply(ExtractTitle)
    dataFrame['Title'] = dataFrame['Title'].apply(MapTitle)
    dataFrame['Embarked'] = dataFrame['Embarked'].fillna('S')
    dataFrame['Embarked'] = dataFrame['Embarked'].map( {'C': 0, 'S': 1, 'Q': 2} ).astype(int)
    dataFrame['Sex'] = dataFrame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataFrame['Age'] = replaceNaByAverage(dataFrame['Age'])
    dataFrame['Fare'] = replaceNaByAverage(dataFrame['Fare'])
    del dataFrame['Name']
    del dataFrame['Ticket']
    del dataFrame['IsAlone']
    #del dataFrame['SibSp']
    #del dataFrame['Parch']
    del dataFrame['TicketSize']
    del dataFrame['Cabin']
    if (normalize==True):
        dataFrame = (dataFrame - dataFrame.mean()) / (dataFrame.max() - dataFrame.min()) #normalize
    return;
def nullValues(dataFrame):
    "for each column of a dataset, outputs the number of null/NaN values"
    sommeNulls = dataFrame.isnull().sum()
    nombreNulls = pd.DataFrame(
            {'column' : dataFrame.columns[:],
             'null_values' : sommeNulls,
             })
    return  nombreNulls
    
#magic numbers
SVM_C = 2
PREDICT = True
LEARNING_CURVES = True
FEATURE_SCALING = False
CROSS_VALIDATION_SIZE = 200

#load the data
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

if (PREDICT == True):
    y = train['Survived']
    X = train.ix[:,'Pclass':] 
    Xtest = test.ix[:,'Pclass':]
    
    featureEngineering(X,FEATURE_SCALING)
    featureEngineering(Xtest,FEATURE_SCALING)
    
    #train a classifier on the transformed data
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X,y)
    
    #use the classifier to make a prediction
    yTrainPredict = pd.DataFrame(clf.predict(X))
    yTestPredict = pd.DataFrame(clf.predict(Xtest))
    yTestPredict.columns = ['Survived']
    prediction = pd.concat([test['PassengerId'],yTestPredict], axis=1)
    
    features = X.columns.values.tolist()
    #importances = clf.feature_importances_
    #print('feature importances :')
    #for i in range(0,len(features)):
    #    line = str(features[i])+ ' : '+str(importances[i])
    #    print(line)
    #export prediction
    prediction.to_csv('prediction.csv', index=False)
    #X.to_csv('X.csv',index=False)
else:
    y = train['Survived']
    X = train.ix[:,'Pclass':] 
    Xtest = test.ix[:,'Pclass':]
    
    featureEngineering(X,FEATURE_SCALING)
    featureEngineering(Xtest,FEATURE_SCALING)
    print(X['Title'].value_counts())

    X.to_csv('X.csv',index=False)
if (LEARNING_CURVES == True):
    
    shuffle(train)
    trainCv = train.iloc[:CROSS_VALIDATION_SIZE]   #cross validation set
    train = train.iloc[CROSS_VALIDATION_SIZE:]    #training set
    
    del train['PassengerId']
    del trainCv['PassengerId']
    
    featureEngineering(train,FEATURE_SCALING)
    featureEngineering(trainCv,FEATURE_SCALING)
    
    train.to_csv("train.csv",index = True)
    trainCv.to_csv("trainCv.csv",index = True)
    
    plot_m = list()
    testError = list()
    crossValidationError = list()
    
    #train a classifier on the transformed data
    clf = ensemble.GradientBoostingClassifier()
    
    for m in range(4,CROSS_VALIDATION_SIZE):
        
        train_temp = train.iloc[CROSS_VALIDATION_SIZE:(CROSS_VALIDATION_SIZE+m)]    #train_temp is a subset of the training set (blue curve)
        y_temp = train_temp['Survived']
        X_temp = train_temp.ix[:,'Pclass':]
            
        trainCv_temp = trainCv.iloc[:m]                                             #trainCv_temp is a subset of the cross validation set, for testing (red curve)
        yCv_temp = trainCv_temp['Survived']
        XCv_temp = trainCv_temp.ix[:,'Pclass':] 
        
        clf.fit(X_temp,y_temp)
        ypredict = pd.DataFrame(clf.predict(X_temp))
        yCvpredict = pd.DataFrame(clf.predict(XCv_temp))
    
        testError.append(metrics.mean_squared_error(ypredict,y_temp))
        crossValidationError.append(metrics.mean_squared_error(yCvpredict,yCv_temp))
        plot_m.append(m)
    plt.plot(plot_m,crossValidationError, color = 'red',label = 'crossValidationError')
    plt.plot(plot_m,testError, color = 'blue',label = 'testError')
    plt.xlabel('m : size of the training set')
    plt.ylabel('J : squared error of the predictions')
    plt.title('The test set error is blue, the cross validation set error is red',fontsize=8)
    plt.suptitle('Learning curves')
    plt.savefig('fig.png')
#export results
#X.to_csv('copy_of_the_training_data.csv', index=False)












