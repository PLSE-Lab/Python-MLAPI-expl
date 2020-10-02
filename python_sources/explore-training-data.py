import numpy as np
import pandas as pd

def exploredata(df):
    # survival ratio for different classes
    pclasses = df['Pclass'].unique()
    #survivor_pclasses = np.zeros(len(pclasses))
    print ('\npclass\tpassenger\tsurvivor\tratio')
    for i in range(len(pclasses)):
        #survivor_pclasses[i] = len(df[df['Pclass']== pclasses[i]])
        num_passenger = len(df[df['Pclass'] == pclasses[i]])
        num_survivor = len(df[(df['Pclass'] == pclasses[i]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (pclasses[i], '\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different sexes
    sexes = df['Sex'].unique()
    print ('\nsex\tpassenger\tsurvivor\tratio')
    for i in range(len(sexes)):
        num_passenger = len(df[df['Sex'] == sexes[i]])
        num_survivor = len(df[(df['Sex'] == sexes[i]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (sexes[i], '\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different age groups
    #agebreaks = [-1,10,20,30,40,50,60,99999]
    agebreaks = [-1,16,25,40,60,99999]
    print ('\nage group\tpassenger\tsurvivor\tratio')
    for i in range(len(agebreaks)-1):
        num_passenger = len(df[(df['Age'] > agebreaks[i]) & (df['Age'] <= agebreaks[i+1])])
        num_survivor = len(df[(df['Age'] > agebreaks[i]) & (df['Age'] <= agebreaks[i+1]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (agebreaks[i],'-',agebreaks[i+1], '(inclusive)\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different sibling/spouse numbers
    #breaks = [-1,0,1,2,999999]
    breaks = [-1,0,99999]
    print ('\n#siblings/spouses\tpassenger\tsurvivor\tratio')
    for i in range(len(breaks)-1):
        num_passenger = len(df[(df['SibSp'] > breaks[i]) & (df['SibSp'] <= breaks[i+1])])
        num_survivor = len(df[(df['SibSp'] > breaks[i]) & (df['SibSp'] <= breaks[i+1]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (breaks[i],'-',breaks[i+1], '(inclusive)\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different parents/children numbers
    #breaks = [-1,0,1,2,999999]
    breaks = [-1,0,99999]
    print ('\n#parents/children\tpassenger\tsurvivor\tratio')
    for i in range(len(breaks)-1):
        num_passenger = len(df[(df['Parch'] > breaks[i]) & (df['Parch'] <= breaks[i+1])])
        num_survivor = len(df[(df['Parch'] > breaks[i]) & (df['Parch'] <= breaks[i+1]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (breaks[i],'-',breaks[i+1], '(inclusive)\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different fares
    #breaks = [-1,0,1,2,999999]
    breaks = [df['Fare'].min(), df['Fare'].quantile(0.25), df['Fare'].quantile(0.5), df['Fare'].quantile(0.75), df['Fare'].max()]
    print ('\n#fare\tpassenger\tsurvivor\tratio')
    for i in range(len(breaks)-1):
        num_passenger = len(df[(df['Fare'] > breaks[i]) & (df['Fare'] <= breaks[i+1])])
        num_survivor = len(df[(df['Fare'] > breaks[i]) & (df['Fare'] <= breaks[i+1]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (breaks[i],'-',breaks[i+1], '(inclusive)\t', num_passenger, '\t', num_survivor, '\t', ratio)
    
    # survival ratio for different ports of embarkation
    breaks = df[df['Embarked'].notnull()]['Embarked'].unique()
    print ('\n#Embarkation\tpassenger\tsurvivor\tratio')
    for i in range(len(breaks)):
        num_passenger = len(df[df['Embarked'] == breaks[i]])
        num_survivor = len(df[(df['Embarked'] == breaks[i]) & (df['Survived'] == 1)])
        ratio = num_survivor * 1.0 / num_passenger
        print (breaks[i], '\t', num_passenger, '\t', num_survivor, '\t', ratio)

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
exploredata(train)