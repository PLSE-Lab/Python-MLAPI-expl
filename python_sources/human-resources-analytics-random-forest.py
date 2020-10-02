import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../input/HR_comma_sep.csv')
X = dataset.drop(['left'], axis=1)
y = dataset['left'].values

X['salary'] = X['salary'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['sales'] = labelencoder_X.fit_transform(X['sales'])
onehotencoder = OneHotEncoder(categorical_features=[7])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
    
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
classifier.fit(X_train, y_train)
    
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
precision = cm[1][1] / (cm[0][1] + cm[1][1])
recall = cm[1][1] / (cm[1][0] + cm[1][1])
print('Accuracy : {}'.format(accuracy))
print('Precision : {}'.format(precision))
print('Recall : {}'.format(recall))
print('F1 Score : {}'.format(2 * precision * recall / (precision + recall)))
print('\n')


out = open('output.csv', "w")
out.write("PersonId,Left\n")

rows = ['']*y_pred.shape[0]
for num in range(0, y_pred.shape[0]):
    rows[num] = "%d,%d\n"%(num+1,y_pred[num])

out.writelines(rows)
out.close()