# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from  pandas.core.series import Series
from sklearn.model_selection import cross_val_score


class Titanic():
    def __init__(self, fileTrain, fileTest, runMode):
        '''
        Here in this constructor we initialize the variables and
        run our model_selection to test the
        Parameters :-
            filename - Name of the Data file to read
            runMode  - Mode of model creation either "full" or "semi" train
        '''
        self.fileTrain = fileTrain
        self.fileTest = fileTest
        self.runMode = runMode
        self.model = self.training()
        self.prediction = self.predict()
        self.accuracy()


    def training(self):
        '''
        Here the model is trained with the specified mode of running.
        Before the training of model the data cleaning is done.
        '''
        df = pd.read_csv(self.fileTrain)
        x = self.cleaning(df)
        x = df[['Pclass', 'Sex', 'Parch', 'Fare', 'Embarked']]
        y = df['Survived']
        if(self.runMode == "semi"):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
            self.total_test_rows = len(y_test)
            self.x_test = x_test
            self.y_test = y_test
            model = XGBClassifier().fit(x_train, y_train)
            return(model)
        else:
            model = XGBClassifier().fit(x, y)
            return(model)


    def cleaning(self, df):
        '''
        Here the data cleaning is done. By keeping only the most important features.
        Parameters :-
            df - DataFrame of the read data
        '''
        lt1 = []
        lt2 = []
        for row1, row2 in zip(df['Embarked'], df['Sex']) :
            if(row1 == 'S'):
                lt1.append(1.0)
            else:
                lt1.append(0.0)
            if(row2 == 'male'):
                lt2.append(1.0)
            else:
                lt2.append(0.0)
        df['Embarked'] = lt1
        df['Sex'] = lt2
        return(df)


    def predict(self):
        '''
        Here the testing of our model on the test Data
        '''
        if(self.runMode == "semi"):
            prediction = self.model.predict(self.x_test)
            return(prediction)
        else:
            df = pd.read_csv(self.fileTest)
            self.fullTestData = self.cleaning(df)
            x = self.fullTestData[['Pclass', 'Sex', 'Parch', 'Fare', 'Embarked']]
            prediction = self.model.predict(x)
            return(prediction)


    def accuracy(self):
        count = 0
        if(self.runMode == "semi"):
            for yPred,y_Actual in zip(self.prediction, self.y_test):
                if(yPred == y_Actual):
                    count+=1
            accuracy = (count/self.total_test_rows)
            print('Accuracy of our prediction Model is %d', accuracy*100)
        else:
            Id = self.fullTestData['PassengerId']
            pred = self.prediction
            with open('output.csv', 'w') as writer:
                writer.write("PassengerId,Survived\n")
                for i, j in zip(Id, pred):
                    writer.write(str(i)+","+str(j)+"\n")
            writer.close()


if(__name__ == "__main__"):
    try:
        print("Please enter the train filename")
        #trainFileName = input()
        trainFileName = '../input/train.csv'
        print("Please enter the test filename")
        #testFileName = input()
        testFileName = '../input/test.csv'
        print("In which mode do want the ML model to run : semi or full ?")
        #runMode = input()
        runMode = "full"
        Titanic(trainFileName, testFileName, runMode)
    except:
        raise TypeError("Passed parameters are not correct please rerun the python file with correct parametrs .")