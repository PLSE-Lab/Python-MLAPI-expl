
import csv
import numpy
import pandas
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

train_file = "../input/train.csv"
test_file = "../input/test.csv"

def data_from_file(file_name):
    df = pandas.read_csv(file_name)
    #print df.info()
    #print df.head()

    title_mapping = {
                    "Capt":       "Important",
                    "Col":        "Important",
                    "Major":      "Important",
                    "Jonkheer":   "Important",
                    "Don":        "Important",
                    "Sir" :       "Important",
                    "Dr":         "Important",
                    "Rev":        "Important",
                    "the Countess":"Important",
                    "Dona":       "Important",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Important",
                    "Lady" :      "Important"
                    } 

    df['Title'] = df['Name'].apply(lambda x: title_mapping[x.split(',')[1].split('.')[0].strip()])

    def ticket_prefix(s):
        s=s.split()[0]
        if s.isdigit():
            return 'ZZ'
        else:
            return s.replace('.','').replace('/','')[0:2]

    df['TicketPrefix'] = df['Ticket'].apply(ticket_prefix)

    mask_age = df.Age.notnull()
    age_title_gender_pclass = df.loc[mask_age, ["Age", "Title", "Sex", "Pclass"]]
    age_inserts = age_title_gender_pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()
    age_inserts = age_inserts.Age.unstack(level = -1).unstack(level = -1)

    mask_age = df.Age.isnull()
    missing_age = df.loc[mask_age, ["Title", "Sex", "Pclass"]]

    def estimate_age(row):
        if row.Sex == "female":
            age = age_inserts.female.loc[row["Title"], row["Pclass"]]
            return age
    
        elif row.Sex == "male":
            age = age_inserts.male.loc[row["Title"], row["Pclass"]]
            return age
    
    missing_age["Age"]  = missing_age.apply(estimate_age, axis = 1)   

    df["Age"] = pandas.concat([age_title_gender_pclass["Age"], missing_age["Age"]])    

    df['Embarked'] = df['Embarked'].fillna(value='C')
    df['Cabin'] = df['Cabin'].fillna(value='Z')
    df['Cabin'] = df['Cabin'].apply(lambda c : c[0])
    df['Fare'] = df['Fare'].fillna(value=df.Fare.mean())
    df = df.drop(['Ticket', 'Name'], axis=1)

    df.Sex = preprocessing.LabelEncoder().fit_transform(df.Sex)
    df.Embarked = preprocessing.LabelEncoder().fit_transform(df.Embarked)
    df.Cabin = preprocessing.LabelEncoder().fit_transform(df.Cabin)
    df.Title = preprocessing.LabelEncoder().fit_transform(df.Title)
    df.TicketPrefix = preprocessing.LabelEncoder().fit_transform(df.TicketPrefix)
    return df.values

train_data = data_from_file(train_file)
test_data = data_from_file(test_file)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,2::],train_data[0::,1])
predictions = forest.predict(test_data[0::,1::])
