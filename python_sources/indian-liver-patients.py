# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#Importing csv
df = pd.read_csv('../input/indian_liver_patient.csv')
X = df.iloc[:,0:10].values
Y = df.iloc[:,10].values
#Analyzing Data
df.describe()
#Finding Null values
df.isnull().sum()
#Albumin_and_Globulin_ratio has 4 null values

#Replacing null values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,9:10])
X[:,9:10] = imputer.transform(X[:,9:10])


#Encoding Gender
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
X[:,1] = label.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X =  onehotencoder.fit_transform(X).toarray()

#Data Splitting into training and test set
from sklearn.model_selection import train_test_split
Xtrain , Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=.1,random_state=0)

#Feature Scaling is needed as there are wide range of values
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

#Applying Principal Component Analysis for finding attributes responsible for maximum variance
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
max_variance = pca.explained_variance_ratio_
print(max_variance)
#[2.54824798e-01 1.82722236e-01 1.78439127e-01 1.25638466e-01
 #8.67079930e-02 7.38895593e-02 6.09837244e-02 1.96840040e-02
 #1.19156008e-02 5.19449213e-03 5.78106817e-33]

 #Maximum variance is from first four attributes
 #Attributes chosen:
 #Age , Gender , Total_Bilirubin , Direct_Bilirubin
 
#Using Logistic Regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(Xtrain,Ytrain)
ypred = classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest,ypred)

#[[43  2]
# [11  3]]
# Accuracy percent = 77.9%

# Any results you write to the current directory are saved as output.