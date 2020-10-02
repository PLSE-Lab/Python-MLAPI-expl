'''
Created on 06-Mar-2017

@author: biren
'''
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling.random_over_sampler import RandomOverSampler
import pandas as pd
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm;
from matplotlib import pyplot as plt
from sklearn.cross_validation import KFold,cross_val_score
import numpy as np,random
from sklearn.metrics import roc_curve,confusion_matrix,auc,accuracy_score,f1_score
from collections import Counter




'''   This Function does work of voting the classifier models.
      All the models predicts the result on test dataset
      in binary labels :
      [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
      [0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]
      [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1]
       .
       .
       .
      [1,0,1,0,1,0,1,0,1,0,1,1,1,0,1]
    Now the Final result is based on the majority vote on number of occurance of class
    Label in columwise 
      [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1]  
'''

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


df=pd.read_csv('../input/glass.csv')

label=df['Type']
col_name=df.columns
label=label.unique()
dataframe=list(pd.DataFrame())
cl_list=list()
for each_class in label:
    dataframe.append(df[df.Type==each_class])
    cl_list.append(df[df.Type==each_class].shape[0])

df.drop('Type',inplace=True,axis=1)
dataframe2=list(pd.DataFrame())
dataframe_label=list(pd.DataFrame())
count=0;

max_instance_in_data=cl_list[0]

list1=list()
real_df=pd.DataFrame()
daf=dataframe[1]
label1=pd.DataFrame(daf)
label1=label1.Type.values

dataframe_label.append(label1)
daf.drop("Type",inplace=True,axis=1)
dataframe2.append(daf.values) #For Class 15

for i in range(0,6):
    if i==1 :
        pass;
    else:
      
        daf=dataframe[1]
        label2=pd.DataFrame(dataframe[i])
        label2=label2.Type.values
        sm=SMOTE()
        dataframe[i].drop("Type",inplace=True,axis=1)
        X,Y=sm.fit_sample(daf.append(dataframe[i]),np.concatenate((label1,label2),axis=0))
      
        dataframe2.append(X[76:])
        dataframe_label.append(Y[76:])
        


d=pd.DataFrame(dataframe2[0])
d1=pd.DataFrame(dataframe2[1])
d2=pd.DataFrame(dataframe2[2])
d3=pd.DataFrame(dataframe2[3])
d4=pd.DataFrame(dataframe2[4])
d5=pd.DataFrame(dataframe2[5])

df= pd.concat([d[:],d1[:],d2[:],d3[:],d4[:],d5[:]],axis=0)
cl_d=pd.DataFrame(dataframe_label[0])
cl_d1=pd.DataFrame(dataframe_label[1])
cl_d2=pd.DataFrame(dataframe_label[2])
cl_d3=pd.DataFrame(dataframe_label[3])
cl_d4=pd.DataFrame(dataframe_label[4])
cl_d5=pd.DataFrame(dataframe_label[5])

df_cl=pd.concat([cl_d[:],cl_d1[:],cl_d2[:],cl_d3[:],cl_d4[:],cl_d5[:]])

#print dataframe2,dataframe_label
df=np.array(df)

train,test,train_label,test_label=train_test_split(df,df_cl,test_size=.20)
model=[]

kf=KFold(train.shape[0],n_folds=10)#because 
fold=0
train1=train.copy()
feature=list()
for train_index,test_index in kf:
    train=train.copy()
    #clf= GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=1, random_state=0)
    #clf=svm.SVC(kernel='linear')
    #clf=xgb.XGBClassifier(gamma=1)
    clf=RandomForestClassifier(n_estimators=10)

    sub_train=np.array(train)[train_index]
    sub_test=np.array(train)[test_index]
    sub_train_label=np.array(train_label)[train_index]
    sub_test_label=np.array(train_label)[test_index]

    clf.fit(sub_train,sub_train_label)
    output=clf.predict(sub_test)
    
    #np.concatenate(feature_importances(clf, sub_train, sub_train_label))
    sub_test_lable=sub_test_label
    
    fold=fold+1
    acc=accuracy_score(sub_test_label,output)
    print ("Accuracy: %f " %(acc))
    print ("Accuracy F1: %f " %(f1_score(sub_test_label,output,average='macro')))
    model.append(clf)

#voter_for_features(feature)

test_label=np.array(test_label)
output,test_label,accuracy=voter(model,np.array(test),test_label)

print ("Final Accuracy:",accuracy)
print ("Final Accuracy F1: %f " %(f1_score(test_label,output,average='micro')))

  