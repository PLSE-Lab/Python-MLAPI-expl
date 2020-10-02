# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

#check the dataset
train.head()
test.head()

#drop missing values
train=train.dropna()
test=test.dropna()

#Create new columns encoding sex into integer (0 & 1). Use get_dummies for one hot encoding.
train=pd.get_dummies(train,prefix={'Sex':'SexEnc'},columns=['Sex'])
test=pd.get_dummies(test,prefix={'Sex':'SexEnc'},columns=['Sex'])

#Select columns to use for training the model
cols_to_select=['Pclass','SexEnc_male','SexEnc_female','Age','SibSp','Parch','Fare','Embarked']
x_train=train[cols_to_select]
x_test=test[cols_to_select]

#This is our Y. We want to build a classification model to predict this.
y_train=train['Survived']

#Use hot encoding for Embarked data column. Not using get_dummies here.
#Step 1 - Simple encoding of strings into integers
#Step 2 - Use OneHotEncoder method on the output of Step 1
#Step 3 - Bring the new encoded Embark columns into train dataframe
#Probably this is what get_dummies does inside...
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X=pd.DataFrame(x_train['Embarked'])
X_2 = X.apply(le.fit_transform)
enc=OneHotEncoder(n_values='auto')
enc.fit(X_2)
x_train_embarked=pd.DataFrame(enc.transform(X_2).toarray())
x_train=x_train.reset_index()
x_train=x_train.drop('index',axis=1)
x_train1=pd.concat([x_train,x_train_embarked],axis=1,ignore_index=False)
x_train1=x_train1.rename(columns={0:'EmbarkC',1:'EmbarkQ',2:'EmbarkS'})
x_train1=x_train1.drop('Embarked',axis=1)
x_train=x_train1
#
#Apply OneHotEncoding on test data
X=pd.DataFrame(x_test['Embarked'])
X_2 = X.apply(le.fit_transform)
enc.fit(X_2)
x_test_embarked = pd.DataFrame(enc.transform(X_2).toarray())
x_test=x_test.reset_index()
x_test=x_test.drop('index',axis=1)
x_test1=pd.concat([x_test,x_test_embarked],axis=1,ignore_index=False)
x_test1=x_test1.rename(columns={0:'EmbarkC',1:'EmbarkQ',2:'EmbarkS'})
x_test1=x_test1.drop('Embarked',axis=1)
x_test=x_test1

#Check columns in training and test datasets
x_train.columns
x_test.columns

### We are now done with data processing. Let's try some classification techniques.

##
#xgboost method
from xgboost import XGBClassifier
xg=XGBClassifier(n_estimators=500,learning_rate=0.05,random_state=99)
xg.fit(x_train,y_train)
xg_predictions=xg.predict(x_test)

##
#Random forest method
from sklearn.ensemble import RandomForestClassifier
rnf=RandomForestClassifier(n_estimators=500,random_state=99)
rnf.fit(x_train,y_train)
rnf_predictions=rnf.predict(x_test)

##
#support vector machine method
from sklearn import svm
sv=svm.SVC(probability=True)   #We need probability for Soft Voting option down in the code
sv.fit(x_train,y_train)
sv_predictions=sv.predict(x_test)

#Ensemble of different classifiers
from sklearn.ensemble import VotingClassifier
vclf=VotingClassifier(estimators=[('xg',xg),('rf',rnf),('sv',sv)],voting='hard')
vclf.fit(x_train,y_train)
hard_predictions=vclf.predict(x_test)
#
#soft voting
vclf=VotingClassifier(estimators=[('xg',xg),('rf',rnf),('sv',sv)],voting='soft')
vclf.fit(x_train,y_train)
soft_predictions=vclf.predict(x_test)
#
#Compare predictions between hard voting and soft voting
df=pd.DataFrame([hard_predictions,soft_predictions]).T
df=df.rename(columns={0:'Hard',1:'Soft'})
df['compare_flag']=['TRUE' if h==s else 'FALSE' for h,s in zip(df['Hard'],df['Soft'])]

print("Number, % of same predictions between Hard & Soft Votings= ",sum(df['compare_flag']=='TRUE'),round(sum(df['compare_flag']=='TRUE')*100/len(df),2))

#Let's compare predictions from XGBoost, RandomForest, SVM against VotingClassifier (Hard Voting), and see which one is contributing most.
df=pd.DataFrame([xg_predictions,rnf_predictions,sv_predictions,hard_predictions]).T
df=df.rename(columns={0:'XGB',1:'RNF',2:'SVC',3:'HardV'})
df['XGB_HV']=['TRUE' if i==j else 'FALSE' for i,j in zip(df['XGB'],df['HardV'])]
df['RNF_HV']=['TRUE' if i==j else 'FALSE' for i,j in zip(df['RNF'],df['HardV'])]
df['SVC_HV']=['TRUE' if i==j else 'FALSE' for i,j in zip(df['SVC'],df['HardV'])]

XGB_contri=round(sum(df['XGB_HV']=='TRUE')*100/len(df),2)
RNF_contri=round(sum(df['RNF_HV']=='TRUE')*100/len(df),2)
SVC_contri=round(sum(df['SVC_HV']=='TRUE')*100/len(df),2)

print("Contribution in % by XGB, RNF & SVC is ",XGB_contri," ",RNF_contri," ",SVC_contri," respectively")