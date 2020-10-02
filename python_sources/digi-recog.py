import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
dataset=pd.read_csv('../input/train.csv')
X=dataset.iloc[:,3:784].values
Y=dataset.iloc[:,0].values

dataset_test=pd.read_csv('../input/test.csv')
X_test=dataset_test.iloc[:,2:783].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)         
X_test=sc_X.transform(X_test)   

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
classifier=SVC(kernel='linear',decision_function_shape='ovr',random_state=0)

classifier.fit(X,Y)

y_pred=classifier.predict(X_test)

np.savetxt('result.csv', np.c_[range(1,len(X_test)+1),y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

