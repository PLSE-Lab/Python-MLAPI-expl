import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train_data.head())

print("\n\nSummary statistics of training data")
print(train_data.describe())

print("\n\nTraining number")
train_num = train_data.shape[0]
print(train_num)

#Any files you save will be available in the output tab below
train_data.to_csv('copy_of_the_training_data.csv', index=False)

full_data = train_data.append(test_data, ignore_index = True)
titanic_data = full_data[:train_num]

# prepare feature
from sklearn.preprocessing import LabelEncoder

# sex and embarked feature
sex = pd.Series( np.where( full_data.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
embarked = pd.get_dummies( full_data.Embarked , prefix='Embarked' )
le = LabelEncoder()

# age and fare feature
imputed = pd.DataFrame()
imputed[ 'Age' ] = full_data.Age.fillna( full_data.Age.mean() )
imputed[ 'Fare' ] = full_data.Fare.fillna( full_data.Fare.mean() )
imputed[ 'Parch' ] = full_data.Parch.fillna(full_data.Parch.mean())
imputed[ 'SibSp' ] = full_data.SibSp.fillna(full_data.SibSp.mean())

 
# title feature
title = pd.DataFrame()
title['Title'] = full_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip() )

Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Dr":         "Officer",
                    "Rev":        "Officer",                    
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Lady":       "Royalty",                    
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr":         "Mr",
                    "Mrs":        "Mrs",
                    "Miss":       "Miss",
                    "Master":     "Master",
                    }

title['Title'] = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)

# cabin feature
cabin = pd.DataFrame()
cabin['Cabin'] = full_data.Cabin.fillna('U')
cabin['Cabin'] = cabin.Cabin.map(lambda c: c[0])
cabin = pd.get_dummies(cabin.Cabin, prefix = 'Cabin')
print(cabin.head())


