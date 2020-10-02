# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
dataset=pd.read_csv('../input/diamonds.csv')
#print(dataset.columns)
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
X = dataset.iloc[:,[1,2,3,4,5,6,8,9,10] ].values
y = dataset.iloc[:, 7].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVR
classifier = SVR(kernel = 'rbf',C=1000.0,gamma=0.01,epsilon=1.0)#, random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
#y_pred.dtype='int32'
#y_pred=np.array(y_pred,dtype='int')


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


#f1= sklearn.metrics.f1_score(y_test,y_pred,average=None)
plt.title("Regressor Accuracy")
plt.xlabel('True Output')
#plt.ylabel('True Output')
plt.scatter(y_test,y_pred,s=3,color='k',label='Predicted Values')
plt.scatter(y_test,y_test,color='r',label='Original Values')

#plt.show()


from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error
z= r2_score(y_test, y_pred)
z1=explained_variance_score(y_test,y_pred)
z2=mean_absolute_error(y_test,y_pred)
print("r-squared test score : " + str(z))

import scipy
zrho=scipy.stats.pearsonr(y_pred,y_test)
plt.show()
plt.savefig('visualization.png')