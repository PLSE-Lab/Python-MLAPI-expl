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

import os 
dir_name=os.path.abspath(os.path.dirname(__file__))
data_file='../input/data.csv'
data_file_path=os.path.join(dir_name, data_file)
import pandas
import numpy
table=pandas.read_csv(data_file_path)
features=numpy.array(table.iloc[:,2:32])
label1=numpy.array(table.iloc[:,1])
from sklearn import preprocessing 
encoder = preprocessing.LabelEncoder( ) 
encoder.fit(label1)
list(encoder.classes_)
labels=encoder.transform(label1)
print (labels)
# (2) Data Split
from sklearn.model_selection import train_test_split

'Split 1'
features_train_val, features_test, labels_train_val, labels_test =(
                            train_test_split(features, labels, test_size=0.2)) 
'Split 2'
features_train, features_val, labels_train, labels_val =(
                            train_test_split(features_train_val, 
                            labels_train_val, test_size=0.25))  
from sklearn import svm
from sklearn.metrics import accuracy_score
accuracy=[]
C_val=[0.001,0.01,0.1,1,10,100]
for c in C_val:
    Classifier= svm.LinearSVC(C=c) 
    Classifier=Classifier.fit(features_train,labels_train)
    predicted_labels=Classifier.predict(features_val)
    crnt_accuracy=accuracy_score(labels_val,predicted_labels)
    accuracy.append(crnt_accuracy)

# (4) Select Model 
print('SVM_Linear')
print (accuracy) 
best_accuracy=max(accuracy)
best_model_inx=accuracy.index(max(accuracy))
#print (Bestmodel), best_model_inx+1
#print (best_C), C_val[best_model_inx]

# (5) Final Training and Reporting 
Classifier=svm.LinearSVC(C=C_val[best_model_inx])
Classifier=Classifier.fit(features_train_val,labels_train_val)
predicted_labels=Classifier.predict(features_test)
final_accuracy=accuracy_score(labels_test,predicted_labels)
print (final_accuracy) 



# (4) SVMs - Ploynomial kernel 

  #(4-2) Training - SVMs
from sklearn import svm

from sklearn.metrics import accuracy_score
accuracy=[]
C_val=[0.001,0.01,0.1,1,10,100]
for c in C_val:
    Classifier= svm.SVC(C=c, kernel= 'poly', degree=1) 
    Classifier=Classifier.fit(features_train,labels_train)
    predicted_labels=Classifier.predict(features_val)
    crnt_accuracy=accuracy_score(labels_val,predicted_labels)
    accuracy.append(crnt_accuracy)
    Best_accuracy=max(accuracy)

# (4) Select Model 
print('SVM_Poly')
print (accuracy)
best_accuracy=max(accuracy)
best_model_inx=accuracy.index(max(accuracy))
print (best_accuracy) 
Best_model=best_model_inx+1
#print (best_C), C_val[best_model_inx]

# (5) Final Training and Reporting 
Classifier=svm.LinearSVC(C=C_val[best_model_inx])
Classifier=Classifier.fit(features_train_val,labels_train_val)
predicted_labels=Classifier.predict(features_test)
final_accuracy=accuracy_score(labels_test,predicted_labels)
print (final_accuracy)
#---------------------------------------------
#RBF kenel
# (4-3) Training - SVMs
from sklearn import svm
from sklearn.metrics import accuracy_score
accuracy=[]
C_Val=[10,1,0.1,0.01]
Gamma_val=[1,0.1,0.01,0.001]
best_accuracy=0
best_c=0
best_gamma=0

for c in C_Val:
    for g in Gamma_val:
        Classifier= svm.SVC(C=c,kernel= 'rbf', gamma=g) 
        Classifier=Classifier.fit(features_train,labels_train)
        predicted_labels=Classifier.predict(features_val)
        crnt_accuracy=accuracy_score(labels_val,predicted_labels)
        accuracy.append(crnt_accuracy)
        if crnt_accuracy > best_accuracy:
            best_accuracy=crnt_accuracy
            best_c=c 
            best_gamma=g

# (4) Select Model 
print('SVM_RBF')
print (accuracy) 
best_accuracy=max(accuracy)
best_model_inx=accuracy.index(max(accuracy))
print (best_accuracy)
best_model=best_model_inx+1
best_gamma=g 
best_c=c 

# (5) Final Training and Reporting 
Classifier= svm.SVC(C=best_c,kernel='rbf',gamma=best_gamma)
Classifier=Classifier.fit(features_train_val,labels_train_val)
predicted_labels=Classifier.predict(features_test)
final_accuracy=accuracy_score(labels_test,predicted_labels)
print (final_accuracy) 
print(best_gamma)
print(best_c)

#---------------------------------------------
