import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tflearn

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

def preprocess(data, columns_to_ignore):
    processed_data = data.copy()
    processed_data['Family_Size']= processed_data['SibSp']+processed_data['Parch']
    processed_data['Gender'] = processed_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    processed_data['Embarked'].fillna('S', inplace=True)
    processed_data['C'] = processed_data['Embarked'].map(lambda s: 1 if s == 'C' else 0).astype(int)
    processed_data['Q'] = processed_data['Embarked'].map(lambda s: 1 if s == 'Q' else 0).astype(int)
    processed_data['S'] = processed_data['Embarked'].map(lambda s: 1 if s == 'S' else 0).astype(int)
    processed_data['Singleton'] = processed_data['Family_Size'].map(lambda s: 1 if s == 1 else 0)
    processed_data['SmallFamily'] = processed_data['Family_Size'].map(lambda s: 1 if 2<=s<=4 else 0)
    processed_data['LargeFamily'] = processed_data['Family_Size'].map(lambda s: 1 if 5<=s else 0)
    processed_data['Young'] = processed_data['Age'].map(lambda s: 1 if s<=16 else 0)
    processed_data['Middle'] = processed_data['Age'].map(lambda s: 1 if s>16 and s<=32 else 0)
    processed_data['Middle_2'] = processed_data['Age'].map(lambda s: 1 if s>30 and s<=40 else 0)
    processed_data['Old'] = processed_data['Age'].map(lambda s: 1 if s>40 and s<=60 else 0)
    processed_data['Very Old'] = processed_data['Age'].map(lambda s: 1 if s> 80 else 0)
    
    processed_data['Title'] = processed_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    processed_data['Title'] = processed_data.Title.map(Title_Dictionary)
    processed_data['Fare'].fillna(processed_data['Fare'].mean(), inplace=True)
    processed_data['Cabin'].fillna('U', inplace=True)
    processed_data['Cabin'] = processed_data['Cabin'].map(lambda c : c[0])
    cabin_dummies = pd.get_dummies(processed_data['Cabin'], prefix='Cabin')
    processed_data = pd.concat([processed_data,cabin_dummies], axis=1)
    if not 'Cabin_T' in processed_data: processed_data['Cabin_T'] = 0
    processed_data = pd.concat([processed_data, pd.get_dummies(processed_data['Pclass'], prefix='Pclass')], axis=1)
    processed_data = pd.concat([processed_data, pd.get_dummies(processed_data['Title'], prefix='Title')], axis=1)
    processed_data["Age"] = processed_data.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x:  x.fillna(x.median()))
    processed_data.drop(columns_to_ignore, errors='ignore', axis=1, inplace=True)
    print(list(processed_data.columns.values))

    #processed_data.dropna(axis=0, how='any', inplace=True)
    return processed_data
to_ignore=['Sex','PassengerId','Pclass','Ticket','Name','Survived','Family_Size', 'Embarked', 'Title', 'Cabin', 'Fare']
labels = data_train['Survived']
bin_labels = np.zeros((len(labels), 2))
bin_labels[np.arange(len(labels)),labels] = 1.
clean_data = preprocess(data_train, to_ignore)

net = tflearn.input_data(shape=[None, clean_data.shape[1]])
net = tflearn.fully_connected(net, 30)
net = tflearn.dropout(net, 0.75)
net = tflearn.fully_connected(net, 30)
net = tflearn.dropout(net, 0.75)
net = tflearn.fully_connected(net, 16)
net = tflearn.dropout(net, 0.75)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(np.array(clean_data, dtype=np.float32), bin_labels, n_epoch=17, batch_size=100,validation_set=0.05,show_metric=True)
output = pd.concat([data_test['PassengerId'], pd.DataFrame(model.predict_label(np.array(preprocess(data_test,to_ignore), dtype=np.float32)), columns=['Survived', 'Not_Survived'])['Survived']], axis=1)
output.to_csv("titanic.csv", index=False)