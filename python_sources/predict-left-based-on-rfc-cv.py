# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import heapq

from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn import tree,preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import f1_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

#Read the analytics csv file and store our dataset into a dataframe called "df"
df=pd.read_csv("../input/HR_comma_sep.csv",index_col=None)

#Feature extraction
#unprdered feature
df["sales"] = df["sales"].astype('category').cat.codes
tmp=pd.get_dummies(df['sales'])
df.drop(labels=['sales'],axis=1,inplace=True)
#ordered feature
df=pd.concat([tmp,df],axis=1) 
df["salary"]=df["salary"].astype('category').cat.set_categories(['low','medium','high'],ordered=True)
df["salary"]=df["salary"].cat.codes

#goal 
target_label = df['left']
df.drop(labels=['left'],axis=1,inplace=True)
df.insert(0,'left',target_label)
df.insert(1,'augmentation',1)

X=df.drop('left',axis=1)
y=df['left']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,stratify=y) #stratify keep the ratio of classes
	
#preprocessing
scaler=preprocessing.StandardScaler().fit(X_train) #Standardization
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("Random Forest Classification:")
model=RandomForestClassifier()
#estimatros_r=np.linspace(20,120,50).astype(int)
estimatros_r=[93]
#features_r=['auto','sqrt','log2']
features_r=['log2']
#depth_r=np.linspace(40,80,40).astype(int)
depth_r=[56]
param_grid=dict(n_estimators=estimatros_r,max_features=features_r,max_depth=depth_r)
grid=GridSearchCV(model,param_grid=param_grid,cv=5,n_jobs=4)
grid.fit(X_train_scaled,y_train)
print("The best parameters are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_))
ACU_test=grid.score(X_test_scaled,y_test)
print("Test acurracy:%.4f"%ACU_test)
y_pred=grid.predict(X_test_scaled)
fsc=f1_score(y_test,y_pred)
print("F_measure on testing set:%.4f"%fsc)

#Feature Selection
clf=tree.DecisionTreeClassifier()
clf.fit(X_train_scaled,y_train)
tmp=clf.feature_importances_
index=heapq.nlargest(5,range(len(tmp)),tmp.take)
index=sorted(index)
print("Top Important Features: "+str(X.columns[index]))

