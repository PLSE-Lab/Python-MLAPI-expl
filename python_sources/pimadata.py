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
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import  SMOTE
import pandas as pd ,numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import count

def voter(listOfModel,testdata,testlabel):
    finalresult=np.zeros((len(listOfModel),len(testlabel)))
    counter=0;
    for eachModel in listOfModel:
        finalresult[counter,:]=eachModel.predict(testdata)
        counter +=1;
    print (finalresult);
    final=pd.DataFrame.mode(pd.DataFrame(finalresult),axis=0)
    return final.values[0],accuracy_score(testlabel,final.values[0])

path=("../input/diabetes.csv")
data=pd.read_csv(path)
classlabel=data.Outcome
data=data.drop('Outcome',axis=1)
inter=[(lambda v: v/5 if (v%5) == 0  else (v/5)+1) (val) for val in np.around(data.BMI)]     
data.BMI=np.around(inter).astype(np.int)
inter1=[(lambda v: v/0.05 if (v%0.05) == 0  else (v/0.05)+1) (val) for val in np.around(data.DiabetesPedigreeFunction,decimals=2)]     
data.DiabetesPedigreeFunction=np.around(inter1).astype(np.int)

data=np.array(data);#Numpy array
smote=SMOTE(kind='regular');
X_resampled,Y_resampled=smote.fit_sample(data,classlabel)
train_data,test_data,train_label,test_label=train_test_split(X_resampled,Y_resampled,test_size=.20)
kfold=KFold(train_data.shape[0],n_folds=10)
models=[];
for trainIndex,testIndex in kfold:
    #model=svm.SVC(kernel='linear')
    model=svm.SVC(kernel='rbf')
    #model=svm.SVC(kernel='poly')
    #model=svm.SVC(kernel='sigmoid')
    x=train_data[trainIndex]
    y=np.array(train_label)[trainIndex]
    y1=np.array(train_label)[testIndex]
    model.fit(x, y)
    predicts=model.predict(train_data[testIndex])
    print ('accuracy:',accuracy_score(y1, predicts))
    models.append(model)
   

result,accuracy=voter(models,test_data,test_label)
print ("Predicated Class:",result)
print ("Actual Class :",test_label)
print ("Accuracy:",accuracy)
