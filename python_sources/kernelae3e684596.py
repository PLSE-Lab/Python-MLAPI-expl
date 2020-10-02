# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

 
import pandas_datareader
import matplotlib.pyplot as plt

training = pd.read_csv(" titanic/train.csv")
testing = pd.read_csv(" titanic/test.csv")

Y = pd.DataFrame()
Y = training.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training, Y, test_size=0.2, random_state=0)

data=X_train
data=data.drop('Survived', axis=1)
test_data = X_test.drop('Survived', axis=1)


def Preprocess(data):


    from sklearn.preprocessing import   MinMaxScaler,   LabelEncoder,OneHotEncoder

    from sklearn.impute import SimpleImputer
    LE = LabelEncoder()

    # Preprocess gender catagory
    data['Sex'] = LE.fit_transform(data['Sex'])
    Sex = data.iloc[:,3].values
    Sex= np.reshape(Sex,[-1,1])

    sc= MinMaxScaler()
    SQ = SimpleImputer(missing_values=np.nan, strategy='median')

    Fare= data.iloc[:,8].values
    Fare = np.reshape(Fare,[-1,1])
    Fare = SQ.fit_transform(X=Fare)
    Fare= np.reshape(Fare,[-1,1])
    Fare= sc.fit_transform(X=Fare)


    # Preprocess  name category
    onehot = OneHotEncoder(drop='first')
    rows = data.shape[0]


    expendable = ['Mr' ]
    precious = ['Mrs', 'Miss', 'Master', 'Mme', 'Ms', 'Mlle']
    specials = ['Sir', 'Don', 'Countess', 'Jonkheer']
    military = [ 'Major', 'Capt', 'Col']
    medical = ['Dr']
    god = ['Rev']

    people = [expendable,precious, specials, military, medical,god]

    for group in range(len(people)):
        for row in range(rows):
            for title in range(len(people[group])):
                try:
                    x = people[group]
                    if data.iloc[row, 2].find(x[title]) > 0:
                        data.iloc[row, 2] = group

                except:

                    AttributeError
    name = data.iloc[:,2].values
    name = np.reshape(name,[-1,1])
    namedata= onehot.fit_transform(name).toarray()


    Age = data.iloc[:,4].values
    Age = np.reshape(Age, [-1,1])

    isAgenull = np.isnan(Age)

    for row in range (rows):
        if isAgenull[row,0]:
            if data.iloc[row,3]==0:
                Age[row,0] = data.loc[data['Sex']==0][data['Pclass']==data.iloc[row,1]]['Age'].median()
                print(Age[row,0])
            else:
                Age[row,0]= data.loc[data['Sex']==1][data['Pclass']==data.iloc[row,1]]['Age'].median()
                print(Age[row,0])
    Age = sc.fit_transform(X=Age)

    classdata = data.iloc[:,1].values
    classdata = np.reshape(classdata,[-1,1])
    classdata = onehot.fit_transform(classdata).toarray()


    sib = data.iloc[:,5].values
    sib = np.reshape(sib,[-1,1])
    sib = sc.fit_transform(sib)

    parch = data.iloc[:,6].values
    parch = np.reshape(parch,[-1,1])
    parch  = sc.fit_transform(parch)


    SI = SimpleImputer(strategy='most_frequent')
    LE = LabelEncoder()

    embark = data.iloc[:,-1].values
    embark = np.reshape(embark, [-1, 1])
    embark =SI.fit_transform(embark)
    embark = np.reshape(embark,[-1,1])
    embark = np.ravel(embark)
    embark = LE.fit_transform(embark)
    embark = np.reshape(embark,[-1,1])
    MM = OneHotEncoder(drop='first')
    embark = MM.fit_transform(embark).toarray()

    values = [Sex, Fare, namedata,classdata,sib, parch,Age,embark]
    newdata = np.concatenate(values, axis=1)
    return newdata

trainingX = Preprocess(data)

testingX = Preprocess(test_data)




from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(trainingX, y_train)


# Predicting the Test set results
y_pred1 = classifier.predict(testingX)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

testing1= Preprocess(testing)

submission = classifier.predict(Preprocess(testing))
submission = np.reshape(submission,[-1,1])
submission =  pd.DataFrame(submission)

np.savetxt("submission1", submission, delimiter=',')

# Make a NN

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

optimizer = Adam(learning_rate=0.0001)

predictor = Sequential ()

predictor.add(Dense(units=300, kernel_initializer='uniform',  input_dim=14))
predictor.add(Dense(units=300, kernel_initializer='uniform', activation='relu'))
predictor.add(Dense(units=200, kernel_initializer='uniform', activation='relu' ))
predictor.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
predictor.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
predictor.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
predictor.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = predictor.fit(trainingX, y_train, batch_size=600,epochs=1000, )
pred = predictor.predict(testing1)
evaluate = predictor.evaluate(testingX,y_test,batch_size=180)
"""pred= pd.DataFrame(pred)
for x in range (pred.shape[0]):
    if pred.iloc[x,0]>=0.5:
        pred.iloc[x,0] =1
    else:
        pred.iloc[x,0] =0
pred.to_csv('submission.csv')

pred = (pred>0.5)
y_test = (y_test==1)"""


from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, pred)


print(history.history.keys())

plt.plot(history.history['loss'], label= 'train')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper left')
plt.show()

plt.matshow(cm1)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.show()