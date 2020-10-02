# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:34:53 2019

@author: vivek
"""
###############################################################################
'''-------------------importing necessary libraries--------------------------'''

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression#logisticregression 
from sklearn.neighbors import KNeighborsClassifier#KNN classifier
from sklearn.tree import DecisionTreeClassifier#Descisiontree classifier
from sklearn.ensemble import RandomForestClassifier#Random forest classifier
from sklearn.naive_bayes import GaussianNB #gaussian classifier
from sklearn.svm import SVC #support vector machine
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix #accuracy_score
from sklearn.ensemble import AdaBoostClassifier#Adaboost classifer for boosting
from sklearn.feature_selection import RFE #one of wrapper method uses a model and find important features for the model
from sklearn.preprocessing import  MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder,PowerTransformer
from sklearn.decomposition import PCA #principalcomponentanalysis
from sklearn.model_selection import cross_val_score,KFold #cross validation score and Kfold
from sklearn.model_selection import GridSearchCV #grid searchcv for best parameter analysis

'''----------------feature descrptions---------------------------------------''' 

''' 
member_id               :identification number
loan_amnt               :amount
funded_amnt             :funded amount
funded_amnt_inv         :funded amount invested
term                    :time period to repay loan
int_rate                :interest rate
installment             :installment 
grade                   :grade
emp_length              :unknown
home_ownership'         :ownership of home
annual_inc              :annual income of lender
verification_status     :verified or not
purpose                 :purpose of taking the loan
dti                     :Debit to income ratio
pub_rec                 :Not available
revol_bal               :Not available
revol_util              :Not available
total_pymnt             :total payment
total_pymnt_inv         :total
total_rec_prncp         :total recovery principal
total_rec_int           :total recovery interest
recoveries              :recovery
last_pymnt_amnt         :last payment amount
policy_code             :policy code
loan_status             :fullypaid or not  
'''

'''-------------------Important varibales used-------------------------------

df= Training dataset without any preprocessing 
valid=Test dataset without any preprocessing
X=Training dataset without target 
Xtrain=Training dataset after preprocessing
ytrain=Training target (loan_status where 0 :default ,1 :fully paid)
Xtest=preprocessed testing data
ytest=Testing target  (loan_status where 0 :default ,1 :fully paid)
results=A list of tuple with first value as model accuracy and second value as name

'''

'''------------------reading the training and evaluation data----------------''' 
df=pd.read_csv('../input/bank-data/DefaultData.csv') #reading training dataset
valid=pd.read_csv('../input/bank-data/DefaultData_eval.csv',header=None) #reading evaluation dataset
valid.columns=df.columns #assigning headers to evaluation data

'''--------------------- Understanding data --------------------------------'''
def understandingdata():
 
#------head and tail values of data    
    print(df.head(10))#first 10 rows
    print(df.tail(10))#last 10 rows

#------checking dimensions
    print(df.shape) #number of rows and columns
    print(df.columns) #names of columns present in training dataset
    print(df.info())#dataset information
    print(df.describe(include='all'))#dataset description
    print(df.dtypes.value_counts()) #types data in training dataset

#--------finding missing values
    print(df.isna().sum())
    
'''---------------------plotting training data------------------------------'''   
def plotdata(df):
    
#---------pie plot for finding the percentage of fully paid and default 
   df['loan_status'].value_counts().plot(kind='pie',autopct='%1.1f%%')
   continousvalues=[]
   categoricalvalues=[]
   for i in df.columns:
        if df[i].dtypes=='float64' or df[i].dtypes == 'int64':
                  continousvalues.append(i)
        else :
             categoricalvalues.append(i)
          
   
#----------distribution plot for finding  distribution of continous values .if it is skewed we can apply log or any other function to make it as gaussian distribution  
   for i in continousvalues:
        sns.distplot(df[i].dropna())
        plt.show()

#----------Boxplot for numerical features to find any relation between the target and to find any outliers
   for i in continousvalues:
             plt.figure()
             sns.boxplot(x=i ,y='loan_status',data=df)#including memberid
             plt.plot()
   
#----------Countplot for categerical data to find any relation  
   for i in categoricalvalues:
          plt.figure(figsize=(20,5))
          sns.countplot(x=i,hue='loan_status',data=df)
          plt.plot()

#----------ploting correlation heatmap on continous values
          plt.figure(figsize=(20,10))
          sns.heatmap(df.replace({'loan_status':{'Default':0,'Fully Paid':1}}).corr(),annot=True,fmt='1.1f',cmap='RdBu') #finding correlation between different value
          plt.plot()
'''--------------------------- preprocessing data----------------------------'''

def preprocessing(df):
    
    Xdata=df.replace({'loan_status':{'Default':0,'Fully Paid':1}})#converting target column to numerical valuese

#-------finding missing values
    Xdata.isna().sum()#missing value sum featurewise

#-------removing unimportant features 

    Xdata=Xdata.drop(['member_id','policy_code','emp_length'],axis=1)#removing member_id has no relation with the loan_status and plicy_code has no relation with most of yhe continous vaslues and target 

#---------------selecting numercial feautres
    continousvalues=[]
   
    for i in Xdata.columns:
        if Xdata[i].dtypes=='float64' or Xdata[i].dtypes == 'int64':
                  continousvalues.append(i) #adding continous values to list

#-------label encoding Grade
    le = LabelEncoder() #creating object
    le.fit(['A','B','C','D','E','F','G']) #fitting
    Xdata['grade']=le.transform(Xdata['grade']) #transform

#-------encoding using getdummies from pandas column home_ownership
    ownership=pd.get_dummies(Xdata['home_ownership'],prefix='HOME_') #applying get dummies 
    Xdata=pd.concat([Xdata,ownership],axis=1) #adding the resultnt along columns 
    Xdata.drop('home_ownership',axis=1,inplace=True) # deleting the column

#-------encoding using getdummies from pandas column verification status
    owner=pd.get_dummies(Xdata['verification_status'],prefix='verfication_') #applying get dummies 
    Xdata=pd.concat([Xdata,owner],axis=1) #adding the resultnt along columns 
    Xdata.drop('verification_status',axis=1,inplace=True) # deleting the column

#--------encoding using getdummies from pandas  column purpose
    owner=pd.get_dummies(Xdata['purpose'],prefix='purpose_')  #applying get dummies 
    Xdata=pd.concat([Xdata,owner],axis=1) #adding the resultnt along columns 
    Xdata.drop('purpose',axis=1,inplace=True) # deleting the column

#---------Training Target and data
    X=Xdata.drop('loan_status',axis=1)
    ytrain=Xdata['loan_status']#training dataset

#---------Scaling data usin standard scaler
    scale=StandardScaler()
    col=X.columns
    X[col]=scale.fit_transform(X[col])
   
#------principal compnenent analysis     
    covar_matrix = PCA(n_components = 29) #helps to avoid overfitting and dimension reduction
    c=covar_matrix.fit(X)
    variance = covar_matrix.explained_variance_ratio_ 
    tran=c.transform(X)
    Xtrain=pd.DataFrame(tran,columns=col[:29])#train data

#--------plotting 29 i,portant features
    print(np.cumsum(np.round(variance,decimals=3)*100)) #cumilative sum of variance
    plt.ylabel('% Variance')
    plt.xlabel('no Features')
    plt.title('PCA Analysis')
    plt.ylim(30,100.5)
    plt.style.context('seaborn-whitegrid')
    plt.plot(np.cumsum(np.round(variance,decimals=3)*100))#plotting 
#-----------powertransformation    
    pt=PowerTransformer()
    for i in continousvalues[0:17]:  #dropping loanstatus
        Xtrain[i]=pt.fit_transform(Xtrain[[i]])#applying power tranform to get gaussian distribution
    
    return Xtrain,ytrain,X  #Xtrain -train data ,ytrain - target data for training,X-train data without scaling

'''------------ Evalution data importing and manipulating -------------------'''

def preprocessingevalutiondata(valid,X,df):

#-----dropping unimportant columns
    evl=valid.drop(['member_id','policy_code','emp_length'],axis=1)
    
#------continous values to list
    continousvaluese=[]
    for i in evl.columns:
         if evl[i].dtypes=='float64' or evl[i].dtypes == 'int64':
                  continousvaluese.append(i)     
    
#-------label encoding Grade
    le = LabelEncoder()
    le.fit(['A','B','C','D','E','F','G'])
    evl['grade']=le.transform(evl['grade'])

#-------encoding using getdummies from pandas column home_ownership    
    owner=pd.get_dummies(evl['home_ownership'],prefix='HOME_')
    evl=pd.concat([evl,owner],axis=1)
    evl.drop('home_ownership',axis=1,inplace=True)

#-------encoding using getdummies from pandas column home_ownership   
    owner=pd.get_dummies(evl['verification_status'],prefix='verfication_')
    evl=pd.concat([evl,owner],axis=1)
    evl.drop('verification_status',axis=1,inplace=True)

#-------encoding using onehot encoder    
    oe=OneHotEncoder()
    oe.fit(df[['purpose']])
    s=oe.transform(evl[['purpose']])
    name=oe.get_feature_names()
    purposes =pd.DataFrame(s.toarray())
    purposes.columns=name
    evl=pd.concat([evl,purposes],axis=1)
    evl.drop('purpose',axis=1,inplace=True)

#------replacing loanstatus values    
    evl=evl.replace({'loan_status':{'Default':0,'Fully Paid':1}})
    
    evldata=evl.drop('loan_status',axis=1)
    ytest=evl['loan_status'] #ytest 
    col=evldata.columns

#-------applying standard scaler    
    scale=StandardScaler()
    evldata[col]=scale.fit_transform(evldata[col])

#--------applying principal component analysis    
    covar_matrix = PCA(n_components = 29)
    c=covar_matrix.fit(X)
    Xtest=pd.DataFrame(c.transform(evldata),columns=col[:29])
#------------------applying powertransform    
    pt=PowerTransformer()
    for i in continousvaluese[0:17]:
        Xtest[i]=pt.fit_transform(Xtest[[i]])

#-------returning Xtest for prediction and y(target) for prediction analysis  
    return Xtest,ytest #Xtest -Data for testing model ,ytest -Data for comparing the predicted outpu

'''-------------------------selecting best model-----------------------------'''
def bestmodelchoosing(Xtrain,ytrain):

    models=[] #models 
    results=[]#average accuracy score
    models.append(('knn',KNeighborsClassifier()))
    models.append(('LR',LogisticRegression(C=5,solver='lbfgs')))
    models.append(('DT',DecisionTreeClassifier()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('GB',GaussianNB()))
    models.append(('SVC',SVC(C=.2,kernel='sigmoid')))

    for name,model in models:
        kfold=KFold(n_splits=10,random_state=7)
        v=cross_val_score(model,Xtrain,ytrain,cv=kfold,scoring='accuracy')
        results.append((sum(v)/len(v),name))
    return results
    
'''----------------------model fit and prediction---------------------------'''    

def modelfittingprediction(Xtrain,ytrain,Xtest,ytest):
    
#-------gridsearchCV for getting better results
    model=LogisticRegression(max_iter=115)
    param={'C':range(1,15),'solver':['liblinear','newton-cg','lbfgs']}#parameters for for slelcting
    grid=GridSearchCV(estimator=model,param_grid=param,cv=10)
    grid.fit(Xtrain,ytrain)
    ypred=grid.predict(Xtest)
    print('Best parameters for the model = ',grid.best_params_) #best parameters for the model
    print('Accuracy score = ',accuracy_score(ytest,ypred)) #accuracy score 
    print('Confusion matrix:\n',confusion_matrix(ytest,ypred))#confusion matrix with predicted and actual values
    print('classification report:\n',classification_report(ytest,ypred))# classification report with precision ,recall,accuracy etc
  


understandingdata()
plotdata(df)#plotting data without any scaling to find relations between them.
Xtrain,ytrain,X=preprocessing(df)
Xtest,ytest=preprocessingevalutiondata(valid,X,df)
print('accuracy_score,modelname :\n',bestmodelchoosing(Xtrain,ytrain))
modelfittingprediction(Xtrain,ytrain,Xtest,ytest)


