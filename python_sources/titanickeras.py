'''
import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
'''

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Missing values
train.set_value(train.Embarked.isnull(), 'Embarked', 'C')
test.set_value(test.Fare.isnull(), 'Fare', 8.05)
full = pd.concat([train, test], ignore_index=True)
full.set_value(full.Cabin.isnull(), 'Cabin', 'U0')

# Feature Engineering
import re
names = full.Name.map(lambda x: len(re.split(' ', x)))
_ = full.set_value(full.index, 'Names', names)
del names

# Add title feature
title = full.Name.map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
title[title=='Mme'] = 'Mrs'
title[title.isin(['Ms','Mlle'])] = 'Miss'
title[title.isin(['Don', 'Jonkheer'])] = 'Sir'
title[title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
title[title.isin(['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev'])] = 'Officer'
_ = full.set_value(full.index, 'Title', title)
del title

# Add deck feature
deck = full[~full.Cabin.isnull()].Cabin.map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
deck = pd.factorize(deck)[0]
_ = full.set_value(full.index, 'Deck', deck)
del deck

# Add room feature
checker = re.compile("([0-9]+)")
def roomNum(x):
    nums = checker.search(x)
    if nums:
        return int(nums.group())+1
    else:
        return 1
rooms = full.Cabin.map(roomNum)
_ = full.set_value(full.index, 'Room', rooms)
del checker, roomNum
full['Room'] = full.Room/full.Room.sum()

full['Group_num'] = full.Parch + full.SibSp + 1

full['Group_size'] = pd.Series('M', index=full.index)
_ = full.set_value(full.Group_num>4, 'Group_size', 'L')
_ = full.set_value(full.Group_num==4, 'Group_size', 'B')
_ = full.set_value(full.Group_num==3, 'Group_size', 'E')
_ = full.set_value(full.Group_num==1, 'Group_size', 'S')

# Normalize fare
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
full['NorFare'] = pd.Series(scaler.fit_transform(full.Fare.values.reshape(-1,1)).reshape(-1), index=full.index)

def setValue(col):
    _ = train.set_value(train.index, col, full[:891][col].values)
    _ = test.set_value(test.index, col, full[891:][col].values)

for col in ['Deck', 'Room', 'Group_size', 'Group_num', 'Names', 'Title']:
    setValue(col)

full.drop(labels=['PassengerId', 'Name', 'Cabin', 'Survived', 'Ticket', 'Fare'], axis=1, inplace=True)
full = pd.get_dummies(full, columns=['Embarked', 'Sex', 'Title', 'Group_size'])

# Predict age
from sklearn.model_selection import train_test_split
X = full[~full.Age.isnull()].drop('Age', axis=1)
y = full[~full.Age.isnull()].Age
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from keras.models import Sequential
from keras.layers import Dense

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1, input_dim=X_train.shape[1]))

# choose loss function and optimizing method
model.compile(loss='mae', optimizer='sgd')

# training
print('Training -----------')
for step in range(1001):
    cost = model.train_on_batch(X_train.as_matrix(), y_train.as_matrix())

# evaluate
'''
cost = model.evaluate(X_test.as_matrix(), y_test.as_matrix())
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)
'''

y_pred = model.predict(X_test.as_matrix())
full.set_value(full.Age.isnull(), 'Age', y_pred)

full['NorAge'] = pd.Series(scaler.fit_transform(full.Age.values.reshape(-1,1)).reshape(-1), index=full.index)
full['NorNames'] = pd.Series(scaler.fit_transform(full.Names.values.reshape(-1,1)).reshape(-1), index=full.index)
full['Group_num'] = pd.Series(scaler.fit_transform(full.Group_num.values.reshape(-1,1)).reshape(-1), index=full.index)

for col in ['NorAge', 'NorFare', 'NorNames', 'Group_num']:
    setValue(col)

train.Sex = np.where(train.Sex=='female', 0, 1)
test.Sex = np.where(test.Sex=='female', 0, 1)

def to_num(x):
    try:
        return int(x)
    except:
        return 0
'''
ticket1 = train.Ticket.map(to_num)
train.set_value(train.index, 'Ticket_', ticket1)
train['Ticket_'] = pd.Series(scaler.fit_transform(train.Ticket_.values.reshape(-1,1)).reshape(-1), index=train.index)
ticket2 = test.Ticket.map(to_num)
test.set_value(test.index, 'Ticket_', ticket2)
test['Ticket_'] = pd.Series(scaler.fit_transform(test.Ticket_.values.reshape(-1,1)).reshape(-1), index=test.index)
'''

train['Child'] = pd.Series(0, index=train.index)
train.set_value(train.Age<10, 'Child', 1)
test['Child'] = pd.Series(0, index=test.index)
test.set_value(test.Age<10, 'Child', 1)

train['Mother'] = pd.Series(0, index=train.index)
train.set_value((test.Sex==0)&(train.Parch+full.SibSp>0)&(train.Age>18), 'Mother', 1)
test['Mother'] = pd.Series(0, index=train.index)
test.set_value((test.Sex==0)&(test.Parch+full.SibSp>0)&(test.Age>18), 'Mother', 1)

# Drop unused column
train.drop(labels=['PassengerId', 'Name', 'Names', 'Cabin', 'Age', 'Ticket', 'Fare'], axis=1, inplace=True)
test.drop(labels=['Name', 'Names', 'Cabin', 'Age', 'Ticket', 'Fare'], axis=1, inplace=True)

train = pd.get_dummies(train, columns=['Embarked', 'Pclass', 'Title', 'Group_size'])
test = pd.get_dummies(test, columns=['Embarked', 'Pclass', 'Title', 'Group_size'])
test['Title_Sir'] = pd.Series(0, index=test.index)


##############################
trainX = train.drop(['Survived'], axis=1)
trainY = train.Survived

# Create model
from keras.layers import Activation
from keras.optimizers import RMSprop
import keras.optimizers as opt

# build a neural network from the 1st layer to the last layer
model = Sequential([
    Dense(2, input_dim=trainX.shape[1]),
    Activation('softmax')
])

# choose loss function and optimizing method
#model.compile(loss='mae', optimizer='rmsprop')

# Another way to define your optimizer
#optobj = RMSprop(lr=0.05)#, rho=0.9)#, epsilon=1e-08, decay=0.0)
optobj = opt.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
#optobj = opt.Adagrad(lr=0.05, epsilon=1e-08, decay=0.0)
# We add metrics to get more results you want to see
model.compile(optimizer=optobj,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

from keras.utils.np_utils import to_categorical
y_binary = to_categorical(trainY.as_matrix())

print ('### shape of trainX %s ###' % str(trainX.shape))

# training
model.fit(trainX.as_matrix(), y_binary, batch_size=32, nb_epoch=10)


##############################
PassengerId = test.PassengerId
test.drop('PassengerId', axis=1, inplace=True)

def to_one_dim(rst):
  s=[]
  for r in rst:
      if r[0] >= r[1]:
          s.append(0)
      else:
          s.append(1)
  return s

#rst = model.predict(test.as_matrix())
#print (to_one_dim(rst))

def submission(model, fname, X):
    ans = pd.DataFrame(columns=['PassengerId', 'Survived'])
    ans.PassengerId = PassengerId
    rst = to_one_dim(model.predict(X))
    ans.Survived = pd.Series(rst, index=ans.index)
    ans.to_csv(fname, index=False)
    return rst
    
rst = submission(model, 'test_submission.csv', test.as_matrix())
print ('PassengerId,Survived')
i = 0
for n in PassengerId:
    print ('%d,%d' % (n, rst[i]))
    i = i + 1

