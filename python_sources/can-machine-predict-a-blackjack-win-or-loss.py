
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
#from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../input/blkjckhands.csv')
blkjck_hnds = df[['card1','card2','sumofcards','dealcard1','winloss']].copy()
#blkjck_hnds['winloss'] = blkjck_hnds['winloss'].apply(lambda x : 'Win' if x == 'Push' else x)

le = preprocessing.LabelEncoder()
blkjck_hnds['winloss'] = le.fit_transform(blkjck_hnds['winloss'])
target_data = blkjck_hnds['winloss'].values
training_data = blkjck_hnds.drop('winloss',axis=1).values
y=target_data
X=training_data
X_scal = scale(training_data)
X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(X_scal,y,test_size=0.25,random_state=10,stratify=y)
#parameters = {'n_neighbors':np.arange(7,9,1)}
#knn = KNeighborsClassifier()
#cv = GridSearchCV(knn,parameters,cv=5)
#cv.fit(X_train_new,y_train_new)
#y_pred = cv.predict(X_test_new)
#print("K-nearest-Neighbors accuracy : ",accuracy_score(y_test_new, y_pred).round(2))
#print("Classification Report : ",classification_report(y_test_new, y_pred))
#print(cv.best_score_)


C_param = [0.001,0.01,0.1,1,10,100]
acc=[]
for i in C_param:
    lr = LogisticRegression(penalty = 'l2', C = i)
    lr.fit(X_train_new,y_train_new)
    y_pred = lr.predict(X_test_new)
    accuracy = accuracy_score(y_test_new, y_pred).round(2)
    acc.append(accuracy)
    
print(acc)