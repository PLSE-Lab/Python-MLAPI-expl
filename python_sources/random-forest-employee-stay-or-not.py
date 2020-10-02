# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


dataset=pd.read_csv("../input/HR_comma_sep.csv")
X=dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y=dataset.iloc[:,6]

# encoding categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:,8]=le.fit_transform(X[:,8])
X[:,7]=le.fit_transform(X[:,7])
X=X.astype(dtype='float')
ohe_sales=OneHotEncoder(categorical_features=[7])
X=ohe_sales.fit_transform(X).toarray()
X=X[:,1:]  # to avoid dummy variable trap
ohe_salary=OneHotEncoder(categorical_features=[16])
X=ohe_salary.fit_transform(X).toarray()
X=X[:,1:] # to avoid dummy variable trap
y=y.as_matrix()

# spliting test and train sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# applying random forest classifier

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier.fit(X_test,y_test)

y_pred=classifier.predict(X_test)


#evaluation

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#print the successful results

results=cm[1,1]+cm[0,0]

from pandas import Series,DataFrame

data=DataFrame([y_test,y_pred],index=["Ground Truth","Predictions"],
               columns=np.arange(3000))

print("Successful predictions:{} out of 3000".format(results))
print(data)

sns.pairplot(data=dataset,hue='salary')


