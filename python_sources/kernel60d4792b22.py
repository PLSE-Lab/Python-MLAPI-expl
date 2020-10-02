#!/usr/bin/env python
# coding: utf-8

# ##                          Bank Telemarketing Campaign Case Study

# ### Objective: To classify the customers who may subscribe to the products on telemarketing campaign

# #### Data Description
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

# In[ ]:


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import  DecisionTreeClassifier
from scipy.stats import zscore

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading the data-pandas dataframe
bankdata=pd.read_csv("/kaggle/input/portuguese-bank-marketing-data-set/bank-full.csv",sep=';')
bankdata.head() #first 5 records of file for sample


# In[ ]:


print("Shape",bankdata.shape) #Shape- no of rows and columns
print("Size",bankdata.size) #Size- number of elements in the data file


# #### File Attributes
# 1. age - Numerical Variable - Customers' Age
# 2. job - Categorical Variable - Customer's Job Type
# 3. marital - Categorical Variable - Customer's Marital Status
# 4. education - Categorical Variable - Customer's Education Level
# 5. default  - Categorical Variable - Customer's credit default status
# 6. balance  - Numerical Variable - Customer's average yearly balance in Euros (numeric)
# 7. housing - Categorical Variable - Customer's housing loan status 
# 8. loan - Categorical Variable - Customer's housing personal loan status   loan?
# 9. contact - Categorical Variable - Customer's prferbale communication mode
# 10. day - Numerical Variable -  last contact day of the month 
# 11. month  - Numerical Variable -  last contact month of year
# 12. duration - Numerical Variable -  last call duration, in seconds
# 13. campaign - Numerical Variable - number of calls performed during the campaign 
# 14. pdays - Numerical Variable - number of days that passed by after the client was last contacted from a previous campaign
# 15. previous - Numerical Variable -  number of calls performed before this campaign and for this client
# 16. poutcome - Categorical Variable - outcome of the previous marketing campaign
# 17. target - Binary Variable - has the client subscribed a term deposit? (binary: "yes","no") 
#  

# In[ ]:


bankdata.info() #file attributes and metadata details


# We see that few variables are of type object and we change them to categorical for further analysis

# In[ ]:


for col in bankdata.columns:
    if bankdata[col].dtype=='object':
        bankdata[col]= pd.Categorical(bankdata[col])
bankdata =bankdata.rename(columns={'y': 'Target'})


# In[ ]:


bankdata.info()


# In[ ]:


# checking for null values in the files
bankdata.isna().sum()


# In[ ]:


bankdata.describe() # Summary of numerical attributes of the file


# In[ ]:


bankdata.head(20) #check data for anamolies


# Though there are no null values or NaN values in the file, we see that are some fields with 'unknown' value, that needs to be treated

# In[ ]:


# Missing values and categorical data treatment with convenient data for analysis

replace_struct={"marital": {"single":0 ,"married":1,"divorced":2},
                "contact": {"unknown":0,"telephone":1,"cellular":2},
                "poutcome":{"other":-1,"unknown":0,"success":1,"failure":2},
                "month": {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12},
                "default":{"yes":1, "no":0},
                "loan":{"yes":1, "no":0},
                "housing":{"yes":1, "no":0},
                "Target": {"no":0,"yes":1}
                }
                
df1=bankdata.replace(replace_struct)
df1


# In[ ]:


#checking the correlation between few columns of interest
corr_data= bankdata[['age','balance','day','duration','campaign','pdays','previous','Target']]
corr_data.corr()


# In[ ]:


df1.corr() # checking correlation between data on missing values and relevant data replacement


# In[ ]:


# checking the correlation 
corr_data= df1[['age','balance','duration','campaign','month','previous','Target']]
corr_data.corr()


# In[ ]:


# correlation matrix/ graph
plt.figure(figsize=(10,8))
sns.heatmap(corr_data.corr(), annot=True, fmt='0.3f', center=0,linewidths=.5)


# According to the correlation heatmap, we see that there are no great correlation between dependent variables and a Target variable, however, we see that there is some good correlation between Target and duration of contact, also previous contact variable and balance have some visible impact on the target variable.

# ### EDA, Outliers Treatment and Modelling

# In[ ]:


#checking for outliers
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('age')
plt.boxplot(df1['age'])
plt.subplot(1,2,2)
plt.title('balance')
plt.boxplot(df1['balance'])


# We see that there are many outlieres in balance, and we need to normalize the data.

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('duration')
plt.boxplot(df1['duration'])
plt.subplot(1,2,2)
plt.title('campaign')
plt.boxplot(df1['campaign'])


# Outlier treatment

# In[ ]:


df1['balance_zscore']=df1['balance']


# In[ ]:


df1


# In[ ]:


df1['balance_zscore']=zscore(df1['balance_zscore'])


# In[ ]:


df1.sample(12)


# In[ ]:


df1=df1.drop(df1[(df1['balance_zscore']>3)|(df1['balance_zscore']<-3)].index, axis=0, inplace=False)


# In[ ]:


print(df1.shape)


# In[ ]:


corr_data= df1[['age','balance','duration','campaign','month','previous','Target']]
print(corr_data.corr())
plt.figure(figsize=(10,8))
sns.heatmap(corr_data.corr(), annot=True, fmt='0.3f', center=0,linewidths=.5)


# #### Univariate Analysis

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('age')
plt.hist(df1['age'],bins=8)
plt.subplot(1,2,2)
plt.title('balance')
plt.hist(df1['balance'],bins=8)


# Age is normally distributed, customers range between age of 18 and 90.However, a majority of customers are in the age group of 30-45.
# 
# Balance is skewed, after dropping outliers in balance, the range of balance is negative, giving a range from -2500 to 12000 euros. 

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('duration')
plt.hist(df1['duration'])
plt.subplot(1,2,2)
plt.title('campaign')
plt.hist(df1['campaign'])


# In[ ]:


df1['campaign'].describe()


# In[ ]:


df1['duration'].describe()


# As observed from the box plot and summary, the duration of contact has a median of around 180 seconds, the left-skewed boxplot indicates that most calls are relatively short.
# 
# The distribution of campagin, most of the customers have been reached by the bank for one to three times,some clients have been contacted by as high as 63 times, which is not normal. 

# In[ ]:


sns.distplot(df1['month'])


# #### Bi-variate and Multivariate Analysis 

# From the plot, we can see that most of the contact were made in month of may

# In[ ]:


sns.scatterplot(x='age',y='balance',data=df1)


# From the Scatterplot, there is no clear relationship between age and balance.
# Nevertheless, over the age of 60, customers tend to have a significantly lower balance.

# In[ ]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='age',y='balance', data=df1, hue='Target')


# From the above graph, we see that there existed customers among all age groups, however. we see that most concentrated with age between 25-65.
# Also, the balance were more for customers of age group between 30-60, and above 60 held low balance, probably because they retired from work

# In[ ]:


plt.figure(figsize=(8,8))
 
sns.countplot(x='poutcome', hue='Target' , data=df1)


# In[ ]:


df2=df1.drop('balance_zscore',axis=1 )


# In[ ]:


df2


# In[ ]:


sns.countplot(x='default', hue='Target' , data=df2) #drop default


# In[ ]:


df2=df2.drop('default',axis=1)


# In[ ]:


df2


# In[ ]:


sns.countplot(x='loan', hue='Target' , data=df2)


# In[ ]:


sns.countplot(x='housing', hue='Target' , data=df2)


# In[ ]:


sns.countplot(x='contact', hue='Target' , data=df2)


# In[ ]:


plt.figure(figsize=(6,8))
sns.countplot(x='month', hue='Target' , data=df2)


# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(x='month',y='duration' , data=df1)
plt.subplot(1,2,2)
sns.barplot(x='month',y='duration', hue='Target', data=df1)


# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(x='month',y='campaign' , data=df1)
plt.subplot(1,2,2)
sns.barplot(x='month',y='campaign', hue='Target', data=df1)


# In[ ]:


print(df2.shape)
df2=df2.drop(df2[df2['education']=='unknown'].index, axis=0)
print(df2.shape)


# In[ ]:


df2['education'].unique()


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(x='job', hue='Target' , data=df2) # can drop unknown values


# In[ ]:


print(df2.shape)
df2=df2.drop(df2[df2['job']=='unknown'].index, axis=0)
print(df2.shape)
df2['job'].unique()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='marital', hue='Target' , data=df2)


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x='day', y='duration' ,hue='Target' , data=df2)


# In[ ]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='campaign',y='duration', data=df1, hue='Target')


# From the scatter plot, we could infer that the customers who subscribed to products were contacted between 2-10 times, and almost call duration ranged between some 60 sec to 2000 seconds

# In[ ]:


df2


# In[ ]:


df2=df2.drop(['pdays','poutcome'],axis=1) #not significant in classifying the customer


# In[ ]:


df2.shape


# In[ ]:


oneHotcode =['job','education']
df2=pd.get_dummies(df2,columns=oneHotcode)


# In[ ]:


df2.shape


# In[ ]:


bankdataset=df2.drop(['job_unknown','education_unknown'],axis=1) 
bankdataset.shape


# In[ ]:


bankdataset.columns


# In[ ]:


x=bankdataset.drop(['Target'], axis=1)
Y=bankdataset['Target']
x_train, x_test, Y_train, Y_test = train_test_split(x,Y, train_size = 0.7, test_size = 0.3, random_state = 1)


# #### Logistic Regression Model
# Logistic Regression is a supervised learning model that predicts a non linear relationship between the dependent and independent variables. It is classification technique

# In[ ]:


LogRegModel= LogisticRegression(solver='sag',max_iter=10000)
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))


# In[ ]:


LogRegModel= LogisticRegression(solver='lbfgs', max_iter=10000)
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))


# In[ ]:


LogRegModel= LogisticRegression(solver='liblinear')
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))


# In[ ]:


pred=LogRegModel.predict(x_test)
print(LogRegModel.intercept_)
print(LogRegModel.coef_)
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(ConfMat_DF, annot=True )


# In[ ]:


#AUC ROC curve
logit_roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, LogRegModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, LogRegModel.predict_proba(x_test)[:,1])
print("Logistic Regression AUC Score:",auc_score)


# #### KNN Model
# KNN stands for K Nearest Neighbour ALogorithm, its one of the classification methods that predicts the output variable based on the nearest dependent variables

# In[ ]:


myList = list(range(1,200))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))
# empty list that will hold accuracy scores
ac_scores = []
rl_scores =[]

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, Y_train)
    # predict the response
    Y_pred = knn.predict(x_test)
    # evaluate accuracy
    scores = accuracy_score(Y_test, Y_pred)
    ac_scores.append(scores)
   # changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)


# In[ ]:


KNNModel = KNeighborsClassifier(n_neighbors = 9)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))


# In[ ]:


KNNModel = KNeighborsClassifier(n_neighbors = 23)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))


# In[ ]:


KNNModel = KNeighborsClassifier(n_neighbors = 41)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))


# In[ ]:


KNNModel = KNeighborsClassifier(n_neighbors =95)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True )


# In[ ]:


knn_roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, KNNModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('KNN_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, KNNModel.predict_proba(x_test)[:,1])
print("KNN AUC Score:",auc_score)


# In[ ]:


NBModel = GaussianNB()
NBModel.fit(x_train, Y_train)
pred= NBModel.predict(x_test)
print(NBModel.score(x_train, Y_train))
print(NBModel.score(x_test, Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, NBModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Baye''s (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Naive_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, NBModel.predict_proba(x_test)[:,1])
print("Naive Bayes AUC Score:",auc_score)


# #### Support Vector Machine
# SVM is an supervised learning model that analyzes the linearly spearable planes and helps resolve the classification problems and regression problems.
# hey are widely used for Classification problems.

# In[ ]:


# Building a Support Vector Machine on train data
SVCModel = SVC(C= .1, kernel='rbf', gamma= 'auto')
SVCModel.fit(x_train, Y_train)

pred= SVCModel.predict(x_test)
# check the accuracy on the training set
print(SVCModel.score(x_train, Y_train))
# check the accuracy on the test set
print(SVCModel.score(x_test, Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# #### Decision Trees
# Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# In[ ]:


dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(x_train, Y_train)


# In[ ]:


print(dTree.score(x_train, Y_train))
print(dTree.score(x_test, Y_test))


# The above DTree model is overfitting, and hence we regularize it by setting the maximum depth of DTree for model

# In[ ]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state=1)
#(The importance of a feature is computed as the normalized total reduction of the criterion brought by that feature. 
#It is also known as the Gini importance )

dTreeR.fit(x_train, Y_train)
print(dTreeR.score(x_train, Y_train))
print(dTreeR.score(x_test, Y_test))


# In[ ]:


# importance of features in the tree building 
print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = x_train.columns))


# In[ ]:


print(dTreeR.score(x_test , Y_test))
pred = dTreeR.predict(x_test)

ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


# Ensemble Techniques

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


bgcl = BaggingClassifier(base_estimator=dTreeR, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


bgcl = BaggingClassifier(base_estimator=LogRegModel, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


abcl = AdaBoostClassifier(base_estimator=dTree,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


abcl = AdaBoostClassifier(base_estimator=dTreeR,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


abcl = AdaBoostClassifier(base_estimator=LogRegModel,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# In[ ]:


rfcl = RandomForestClassifier(n_estimators = 50, random_state=1,max_depth=5)
rfcl = rfcl.fit(x_train, Y_train)
pred= rfcl.predict(x_test)
print(rfcl.score(x_test , Y_test))


# In[ ]:


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


# ![](http://)By applying logistic regression or Decision Tree algorithms, classification and estimation model were built with 89.5% score. With either of these models, the bank will be able to predict if the customer would subscribe to the product as a response to the Bank's telemarketing campaign before calling this customer. 
