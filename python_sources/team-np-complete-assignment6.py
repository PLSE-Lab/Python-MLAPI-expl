# Assignment 6
# NP Complete
# 01FB15ECS364 - Nikitha Rao
# 01FB15ECS365 - Piyush Surana


# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix

# creating output file
outputFileObject = open("NP_Complete_Output.txt", "w")
outputFileObject.write("""

 Assignment 6
 NP Complete
 01FB15ECS364 - Nikitha Rao
 01FB15ECS365 - Piyush Surana

""")



# read file 
df = pd.read_csv('../input/Indian Liver Patient Dataset (ILPD).csv')

# fill all NAs with suitable value
df = df.fillna(method = 'ffill')

# correlation plot 
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

sns.heatmap(df.corr())
plt.title('Correlation Plot')

plt.subplots_adjust(left=0.2, bottom=0.25, right=1, top=0.95, wspace=0, hspace=0)
plt.show()
plt.savefig('Graph.png')

# shuffles data
df = shuffle(df)

# function to compute various measures of confusion matrix 
def confusionMatrix(actual,predicted):
    
    cm = confusion_matrix(actual,predicted)
    outputFileObject.write('\nConfusion Matrix : \n'+ str(cm))

    total=cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0]

    accuracy=(cm[0,0]+cm[1,1])/total
    outputFileObject.write ('\nAccuracy : '+ str(accuracy))

    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    outputFileObject.write('\nSensitivity : '+ str(sensitivity ))
    
    precision = cm[0,0]/(cm[0,0]+cm[1,0])
    outputFileObject.write('\nPrecision: ' + str(precision))

    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    outputFileObject.write('\nSpecificity : ' + str(specificity))
    
    fscore = 2*precision*sensitivity/(precision+sensitivity)
    outputFileObject.write('\nF-Score : ' + str(fscore))


# create matrix X and target vector y
X = np.array(df.iloc[:, 0:9])
y = np.array(df['is_patient'])



# labels for gender
# male = 1 female = 0
for i in X:
    if i[1] == "Female":
        i[1] = 0
    else:
        i[1] = 1


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# outputFileObject.write template for confusion matrix
outputFileObject.write('\nTemplate for Confusion Matrix: \n')
template_list = [["TP","FN"],["FP","TN"]]
outputFileObject.write('\n\t\tPredicted')
outputFileObject.write('\nGround Truth\t' + str(template_list[0]))
outputFileObject.write('\n\t\t' + str(template_list[1]))

# SVM Classifier

clf = svm.SVC(shrinking=True,kernel = 'rbf')
clf.fit(X_train,y_train)
pred_svm = clf.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm) * 100
outputFileObject.write('\n\n\nUsing SVM Classifier')
outputFileObject.write('\n\nThe accuracy of the SVM Classfier is %d%%' % acc_svm)
confusionMatrix(y_test, pred_svm)

# KNN Classifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
outputFileObject.write('\n\n\nUsing KNN Classifier')
acc_knn = accuracy_score(y_test, pred_knn) * 100
outputFileObject.write('\n\nThe accuracy of the knn classfier is %d%%' % acc_knn)
confusionMatrix(y_test, pred_knn)



# Logistic Regression

reg = LogisticRegression()
reg.fit(X_train, y_train)
pred_reg = reg.predict(X_test)
acc_reg = accuracy_score(y_test, pred_reg) * 100
outputFileObject.write('\n\n\nUsing Logistic Regression')
outputFileObject.write('\n\nThe accuracy of Logistic Regression is %d%%' % acc_reg)
confusionMatrix(y_test, pred_reg)


outputFileObject.write("""


NOTE:

1. Reasons for choice of algorithms:
		- Supervised Learning: Lables are given.
		- Classification based on features.
		- Classification algorithms to classify into respective classes.
2. As we can see from the results, the sensitivity for the SVM classifier is very high (close to 1),
   we can conclude that it correctly classifies the people who have the disease and are patients.
   This measure is relatively lower in KNN and Logistic Regression. (Logistic being higher than KNN).

   KNN has max Specificity so it has the best ability to classify those who dont have the disease and are not patients.

   Other measures are roughly the same and hence you cant make any conclusions based on them.
   
   Also, SVM works extremely well for binary classification.
   """)