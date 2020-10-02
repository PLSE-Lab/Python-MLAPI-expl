# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/creditcard.csv')
# Any results you write to the current directory are saved as output.
df_time=df.set_index("Time") # for drop the Time coloumn
amnt=df.loc[:,"Amount"]
cls_lvl=df.loc[:,"Class"]
frud=[]
othrws=[]

v = pd.value_counts(df["Class"], sort= True)

frud_cls_lvl=np.array( df[df["Class"]== 1].index)

othrs_cls_lvl= np.array(df[df["Class"]==0].index)


features = df.loc[:,"Time":"V28"]

frud_features= df.loc[frud_cls_lvl,"Time":"V28"]

othrs_features= df.loc[othrs_cls_lvl,"Time":"V28"]

# training and testing set preparion (67:33)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, cls_lvl, test_size=0.33, random_state=42)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


# Learn to predict each class against the other

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)

roc_auc= auc(fpr, tpr)
print('Area under the curve (AUC) is:' % roc_auc*100,'%')  
import matplotlib.pyplot as plt

plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# ROC curve area is 0.96

# best true positive rate and corresponding false positive rate
tpr_indx=np.argmax(tpr)
tpr_value=tpr[tpr_indx]
print('Best true positive rate is:',tpr_value*100,'%')
fpr_value= fpr[tpr_indx]
print('Best false positive rate is:',fpr_value*100,'%')