import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
%matplotlib inline 
import matplotlib.pyplot as plt  # Matlab-style plotting 
import seaborn as sns 
color = sns.color_palette() 
import warnings 
warnings.filterwarnings('ignore') #Supress unnecessary warnings for readabilit y and cleaner presentation 

pd.set_option('display.float_format', lambda x: '%.3f' % x) #Limiting floats o utput to 3 decimal points 

train = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\X_train.csv') 
train.head(5)

train1 = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\y_train.csv') 
train1.head(5)

alldata = train.merge(train1)
alldata.shape

test = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\X_test.csv') 
test.head(5)

test1 = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\y_train.csv') 
test1.head(5)


alldatatest = test.merge(test1)
alldatatest.shape

tr = alldata['series_id']
tr.shape

te = alldatatest['series_id']
te.shape


alldata.drop([ "group_id", "measurement_number", "row_id"], axis=1, inplace=True)
alldatatest.drop(["group_id", "measurement_number", "row_id"], axis=1, inplace=True)

alldatatest.drop(["surface", "series_id"], axis=1, inplace=True)
alldata.drop(["surface", "series_id"], axis=1, inplace=True)

alldata.drop(["orientation_W", "orientation_X", "orientation_Y", "orientation_Z"], axis=1, inplace=True)
alldatatest.drop(["orientation_W", "orientation_X", "orientation_Y", "orientation_Z"], axis=1, inplace=True)

from sklearn.preprocessing import scale
x_scale = scale(alldata)
x_scale= pd.DataFrame(x_scale, columns=alldata.columns)
x_scale.head()

 
from sklearn.preprocessing import scale
x_scal = scale(alldatatest)
x_scal= pd.DataFrame(x_scal, columns=alldatatest.columns)
x_scal.head()

y = train1["surface"]
y.dtype

obj_train1 = train1.select_dtypes(include=['object']).copy()
obj_train1.head()

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
obj_train1["surfaces"] = lb_make.fit_transform(obj_train1["surface"])
obj_train1[["surface", "surfaces"]]

y1 =obj_train1[["surfaces", "surface",]]
y1

surf = pd.DataFrame(y1)
training= pd.concat([x_scale, surf], axis=1, sort=False)


testing = pd.concat([x_scal, surf], axis=1, sort=False)
training["surfaces"].fillna(0, inplace=True)


training["surface"].fillna(0, inplace=True)

testing["surface"].fillna(0, inplace=True)
testing["surfaces"].fillna(0, inplace=True)

training.drop(["surface"], axis=1, inplace=True)
testing.drop(["surface"], axis=1, inplace=True)


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(XTRAIN,YTRAIN)
X_test = lda.transform(XTEST)

XTEST = testing.iloc[:487680,1:6]
YTEST = training.iloc[:, -1]
XTRAIN = training.iloc[:487680,1:6]
YTRAIN = training.iloc[:, -1]
#from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_train,YTRAIN)
PREDI = OVR.predict(X_test)
 
print('Accuracy of SVC linear test set: {:.2f}'.format(OVR.score(X_test, YTRAIN)))
Accuracy of SVC linear test set: 0.99
%matplotlib inline
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(YTRAIN, PREDI))
('Accuracy:', 0.9925750492125984)
sub_df = pd.DataFrame({"series_id": te})
sub_df["surface"] = PREDI
sub_df.to_csv("subm.csv", index=False)







