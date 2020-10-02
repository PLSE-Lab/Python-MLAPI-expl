# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import TomekLinks
from collections import Counter
import pandas as pd,numpy as np
from sklearn import svm 
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,train_test_split, cross_val_score
from timeit import timeit
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#for evalution on Test Data
def voter(model,testdata,test_label):
    result=np.zeros((len(model),test_label.shape[0]))
    row_counter=0
    for classifier in model:
        pre=classifier.predict(testdata)
        result[row_counter,:]=pre;
        row_counter += 1;
    final_result=pd.DataFrame.mode(pd.DataFrame(result),axis=0)
    finalresult=final_result.values[0]
    "print accuracy_score(test_label, finalresult)"
    return finalresult,test_label,accuracy_score(test_label, finalresult)
    
    
    
#plt.style.use('ggplot')
path="../input/Iris.csv"
data=pd.read_csv(path)

data.Species=data.Species.map({'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}).astype(np.int)

label=data.Species.astype(np.int);
data=data.drop('Species',axis=1)

data=np.array(data)
#sm=SMOTE(kind='regular')
sm=SMOTE(kind='svm')
#sm=SMOTE(kind='borderline1')
#sm=SMOTE(kind='borderline2')
#sm=RandomOverSampler();
#sm=TomekLinks(random_state=42)
X_resampled,Y_resampled=sm.fit_sample(data,label);
train,test,train_label,test_label=train_test_split(X_resampled,Y_resampled,test_size=.20)
kfold=KFold(train.shape[0],n_folds=10,shuffle=True)
model=[]
i=0
for trainIndex ,testIndex in kfold:
    #print trainlabel[trainIndex];
    #classifier=svm.SVC(kernel='sigmoid')
    classifier=svm.SVC(kernel='linear')
    #classifier=svm.SVC(kernel='rbf')
    #classifier=svm.SVC(kernel='poly')
    #classifier=DecisionTreeClassifier(criterion='gini',max_depth=5)
    #classifier=RandomForestClassifier(n_estimators=1400, max_depth=4,criterion='entropy')
    x=train[trainIndex] 
    y=np.array(train_label)[trainIndex] 
    y1=np.array(train_label)[testIndex]
    #classifier.fit(train[trainIndex],train_label[trainIndex]);
    classifier.fit(x,y);
    output=classifier.predict(train[testIndex])
    model.append(classifier)
    accuracy=accuracy_score(y1, output)
    print ('Acuracy of model %d = %f'%(i,accuracy))
    i=i+1;
    #print 'FeatureImportance',classifier.feature_importances_
   
final,test_label,acc=voter(model, test,test_label)
print ("Predicted:",final)
print ("Actual:",test_label)
print ("Accuracy:",acc)

fig=plt.figure("Confusion Matrix and ROC Curve",figsize=(5,5))
ax=fig.add_subplot(111)

cm=confusion_matrix(test_label,final)
ax.plot(.3,.6,label='%s'%(cm))
ax.legend(loc='lower right',prop={'size':8})
plt.imshow(cm,interpolation='nearest')
plt.yticks([0,2])
plt.xticks([0,2])
plt.colorbar()
#This is for Voted model



df=data;
pca=PCA(n_components=3)
df=pca.fit_transform(data)
fig2=plt.figure("Projection of Datasets on 2D and 3D:Original dataset",figsize=(5,5))
fig2.subplots_adjust(left=0,right=.95,top=.95,bottom=0)
ax2=fig2.add_subplot(221)
for i in range(len(label)):
    if label[i]==1:
        ax2.scatter(df[i][0],df[i][1],c='r',marker='^')
    elif label[i]==2:
        ax2.scatter(df[i][0],df[i][1],c='b',marker='s')
    else:
        ax2.scatter(df[i][0],df[i][1],c='g',marker='o')
ax2.plot(.2,.6,marker='o',c='red',label='Red:%d'%(Counter(label)[1]))
ax2.plot(.3,.6,marker='o',c='blue',label='Blue:%d'%(Counter(label)[2]))
ax2.plot(.3,.6,marker='o',c='green',label='Green:%d'%(Counter(label)[3]))
ax2.legend(loc='lower right',prop={'size':8}) 
ax3=fig2.add_subplot(222,projection='3d')
for i in range(len(label)):
    if label[i]==1:
        ax3.scatter(df[i][0],df[i][1],df[i][2],c='r',marker='^')
    elif label[i]==2:
        ax3.scatter(df[i][0],df[i][1],df[i][2],c='b',marker='s')
    else:
        ax3.scatter(df[i][0],df[i][1],df[i][2],c='g',marker='o')

X_resampled=pca.fit_transform(X_resampled)

ax4=fig2.add_subplot(223)
for i in range(len(Y_resampled)):
    if Y_resampled[i]==1:
        ax4.scatter(X_resampled[i][0],X_resampled[i][1],c='r',marker='^')
    elif Y_resampled[i]==2:
        ax4.scatter(X_resampled[i][0],X_resampled[i][1],c='b',marker='s')
    else:
        ax4.scatter(X_resampled[i][0],X_resampled[i][1],c='g',marker='o')
ax4.plot(.2,.6,marker='o',c='red',label='Red:%d'%(Counter(Y_resampled)[1]))
ax4.plot(.3,.6,marker='o',c='blue',label='Blue:%d'%(Counter(Y_resampled)[2]))
ax4.plot(.3,.6,marker='o',c='green',label='Green:%d'%(Counter(Y_resampled)[3]))
ax4.legend(loc='lower right',prop={'size':8}) 

ax5=fig2.add_subplot(224,projection='3d')
for i in range(len(Y_resampled)):
    if Y_resampled[i]==1:
        ax5.scatter(X_resampled[i][0],X_resampled[i][1],X_resampled[i][2],c='r',marker='^')
    elif label[i]==2:
        ax5.scatter(X_resampled[i][0],X_resampled[i][1],X_resampled[i][2],c='b',marker='s')
    else:
        ax5.scatter(X_resampled[i][0],X_resampled[i][1],X_resampled[i][2],c='g',marker='o')


plt.show()
        