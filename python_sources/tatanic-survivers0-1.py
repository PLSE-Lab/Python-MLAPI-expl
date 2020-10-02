import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

X_train = train.iloc[:,1:]
y_train = train.iloc[:,:1]
X_test = test.iloc[:,:]

X_train = pd.get_dummies(X_train,drop_first=True)
X_test = pd.get_dummies(X_test, drop_first = True)

X_train = X_train.fillna(X_train['Age'].mean())
X_test = X_test.fillna(X_test['Age'].mean())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(12, kernel_initializer='uniform', activation='relu', input_dim=8))
model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

y_predict = model.predict(X_test)
y_predict = (y_predict > 0.5)
def output_as_bool(output):
    result = []
    for out in range(output.shape[0]):
        if output[out]>0.50:
            result.append(1)
        else:
            result.append(0)
    return result
y_predict = output_as_bool(y_predict)
print(y_predict)


test = pd.read_csv('../input/test.csv')

sol = {'PassengerId': test['PassengerId'],
         'Survived': y_predict}
sol = pd.DataFrame(sol)
sol.to_csv('Submission.csv', index = False)

# from sklearn.metrics import confusion_matrix


# # Cross-Validation and parameter tuning
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score


# def build_classifier(optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(12, kernel_initializer='uniform', activation='relu', input_dim=8))
#     model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# classifier = KerasClassifier(build_fn=build_classifier, epochs=100, batch_size=10)
# accuracies = cross_val_score(classifier, X_train, y_train, cv=10)
# print(accuracies)

# # Parameter tuning with Grid Search CV
# from sklearn.model_selection import GridSearchCV

# params = {'epochs': [50, 100, 500],
#           'batch_size': [25, 32],
#           'optimizer': ['adam', 'rmsprop']}
# grid_cv = GridSearchCV(estimator=classifier, param_grid=params, scoring=['accuracy'], cv=10, refit = False)
# grid_cv = grid_cv.fit(X_train, y_train)
# best_params = grid_cv.best_params_
# best_accuracy = grid_cv.best_score_