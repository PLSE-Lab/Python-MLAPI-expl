import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


df=pd.read_csv('../input/HR_comma_sep.csv')
#reading the data from the HR csv file
X=[]
Y=[]
#Cleaning the data
#Here, df is the dataframe, and departments are converted to a one-hot portion of the feature vector of an entry,
#so that each department is treated by the learning algorithm as a seperate dimension.
for i in range(int(len(df)/2)):
    if(i%10==0):
        print(i,len(df))
    features=[]
    features.append(df.T[i]['satisfaction_level'])
    features.append(df.T[i]['last_evaluation'])
    features.append(df.T[i]['number_project'])
    features.append(df.T[i]['average_montly_hours'])
    features.append(df.T[i]['time_spend_company'])
    features.append(df.T[i]['Work_accident'])
    features.append(df.T[i]['promotion_last_5years'])
    
    if(df.T[i]['sales']=='sales'):
        features.extend([1,0,0,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='marketing'):
        features.extend([0,1,0,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='management'):
        features.extend([0,0,1,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='product_mng'):
        features.extend([0,0,0,1,0,0,0,0,0,0])
    if(df.T[i]['sales']=='technical'):
        features.extend([0,0,0,0,1,0,0,0,0,0])
    if(df.T[i]['sales']=='support'):
        features.extend([0,0,0,0,0,1,0,0,0,0])
    if(df.T[i]['sales']=='RandD'):
        features.extend([0,0,0,0,0,0,1,0,0,0])
    if(df.T[i]['sales']=='accounting'):
        features.extend([0,0,0,0,0,0,0,1,0,0])
    if(df.T[i]['sales']=='hr'):
        features.extend([0,0,0,0,0,0,0,0,1,0])
    if(df.T[i]['sales']=='IT'):
        features.extend([0,0,0,0,0,0,0,0,0,1])
        
    if(df.T[i]['salary']=='medium'):
        features.append(2)
    if(df.T[i]['salary']=='high'):
        features.append(3)
    if(df.T[i]['salary']=='low'):
        features.append(1)
    X.append(features)
    Y.append(df.T[i]['left'])
    
print('-------')
print(len(X),len(Y))
#A decision tree is computed using scikit learn
clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)


pos=0
neg=0
fpos=0
fneg=0
#data is split into two halves with training and testing data
#testing data is used to print out the number of false positives and negatives and determine the acuracy.
for i in range(1+int(len(df)/2),len(df)):
    if(i%10==0):
        print(i,len(df))
    features=[]
    features.append(df.T[i]['satisfaction_level'])
    features.append(df.T[i]['last_evaluation'])
    features.append(df.T[i]['number_project'])
    features.append(df.T[i]['average_montly_hours'])
    features.append(df.T[i]['time_spend_company'])
    features.append(df.T[i]['Work_accident'])
    features.append(df.T[i]['promotion_last_5years'])
    
    if(df.T[i]['sales']=='sales'):
        features.extend([1,0,0,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='marketing'):
        features.extend([0,1,0,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='management'):
        features.extend([0,0,1,0,0,0,0,0,0,0])
    if(df.T[i]['sales']=='product_mng'):
        features.extend([0,0,0,1,0,0,0,0,0,0])
    if(df.T[i]['sales']=='technical'):
        features.extend([0,0,0,0,1,0,0,0,0,0])
    if(df.T[i]['sales']=='support'):
        features.extend([0,0,0,0,0,1,0,0,0,0])
    if(df.T[i]['sales']=='RandD'):
        features.extend([0,0,0,0,0,0,1,0,0,0])
    if(df.T[i]['sales']=='accounting'):
        features.extend([0,0,0,0,0,0,0,1,0,0])
    if(df.T[i]['sales']=='hr'):
        features.extend([0,0,0,0,0,0,0,0,1,0])
    if(df.T[i]['sales']=='IT'):
        features.extend([0,0,0,0,0,0,0,0,0,1])
    
    if(df.T[i]['salary']=='medium'):
        features.append(2)
    if(df.T[i]['salary']=='high'):
        features.append(3)
    if(df.T[i]['salary']=='low'):
        features.append(1)
    X.append(features)
    Y.append(df.T[i]['left'])
    pr=clf.predict([features])
    if(pr[0]==1 and Y[-1]==1):
        pos+=1
    if(pr[0]==0 and Y[-1]==0):
        neg+=1
    if(pr[0]==1 and Y[-1]==0):
        pos+=1
        fpos+=1
    if(pr[0]==0 and Y[-1]==1):
        neg+=1
        fneg+=1
        
#results are displayed
print('false positives: '+str(fpos)+'/'+str(pos))
print('false negatives: '+str(fneg)+'/'+str(neg))
print('accuracy : '+str(100*(pos+neg-fpos-fneg)/(pos+neg))+' %.')