#!/usr/bin/env python
# coding: utf-8

# <div align='center' style='font-size:25px;color:green;font-weight:bold'>Classification Algorithms</div>
# <div style='font-size:16px;padding-top:20px '>
# We are starting with some simple algorithms to advanced algorithms for classification.And also boosting our skills on the classification problems on real world datasets.
# </div>
# 
# <div style='padding-top:25px'><p style='font-weight:bold;font-size:17px' >Algorithms I am using are :</p>
# <ul style='font-size:15px'>
# <li>Support Vector Classifier</li>
# <li>K-nearest Neighbors Classifier</li>
# <li>Logistic Regression</li>
# <li>DecisionTree Classifier</li>
# <li>RandomForest Classifier</li>
# <li>GradientBoosting Classifier</li>
# </ul>
# </div>

# In[ ]:


import pandas as pd  #for loading the dataset as DataFrame
import numpy as np   #for handling multi-D arrays and mathematical computations
import seaborn as sb #highly interactive visualization of dataset
from matplotlib import pyplot as plt #visualization the data
from sklearn.model_selection import train_test_split  # split the data into trian and test sets
# different algorithms for comparisons
from sklearn.ensemble import RandomForestClassifier # also used for feature selection
from sklearn.ensemble import GradientBoostingClassifier #boosting algorithm
from sklearn.tree import DecisionTreeClassifier     
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC   # support vector classifier (SVC)


# <div style='font-size:20px'>Reading the Dataset :</div>

# In[ ]:


data = pd.read_csv('../input/loan-prediction.csv') 
#our dataset is about Loan Status of different applicants with several features


# In[ ]:


data.head() #overview of our dataset


# <div style='font-size:20px'>Start Analysing :</div>

# In[ ]:


data.isna().sum() #check how many number of Nan are present in each column


# In[ ]:


data.shape   #dimension of our dataset


# <div style='font-size:20px'>Handling the Nan Values :</div>

# In[ ]:


data.fillna(method='bfill',inplace=True) # here we are use backward filling to remove our Nan from Dataset


# <div style='font-size:20px'>Visualization of data set :</div>

# In[ ]:


#countplot of different gender on the basis of there loan status
sb.countplot(x='Gender',
             data=data,
             hue='Loan_Status',
             palette="GnBu_d") 


# In[ ]:


#here we are clearly obeserve the what is applicantIncome and whether he/she is self_employed or not,
#with there Loan Status
sb.catplot(x='Gender',
           y='ApplicantIncome',
           data=data,
           kind ='bar',
           hue='Loan_Status',
           col='Self_Employed')


# In[ ]:


#here we are clearly obeserve the what is co-applicantIncome and whether he/she is self_employed or not,
#with there Loan Status
sb.catplot(x='Gender',
           y='CoapplicantIncome',
           data=data,
           kind ='bar',
           hue='Loan_Status',
           col='Self_Employed')


# In[ ]:


data['ApplicantIncome'].plot(kind='hist',bins=50) #histogram of Applicant-Income
# we see that most of them are in between 0-10000


# In[ ]:


data['CoapplicantIncome'].plot(kind='hist',bins=50)
#histogram of coappicantIncome which almost similar to the ApplicantIncome's histogram


# In[ ]:


#it is more useful to use ApplicantIncome and CoapplicantIncome as one featue i.e, Total_Income
#for reducing the feature set
#So, I am creating new column Total_Income as the sum of ApplicantIncome and CopplicantIncome
data['Total_Income']=(data['ApplicantIncome']+data['CoapplicantIncome'])
data['Total_Income'].plot(kind='hist',bins=50) #histogram of Total_Income which is almost similar with above two
data.drop(columns=['ApplicantIncome','CoapplicantIncome'],inplace=True) 


# In[ ]:


sb.countplot(data.Dependents,data=data,hue='Loan_Status')
#count of different dependents with respect to there Loan_status


# In[ ]:


sb.countplot(data.Education,data=data,hue='Loan_Status',palette='Blues')
#count of graduated or non-graduated with respect to there Loan_status


# In[ ]:


sb.countplot(data.Married,data=data,hue='Loan_Status')
#count of Married or non-Married applicant with respect to there Loan_status


# In[ ]:


sb.barplot(x='Credit_History',y='Property_Area',data=data,hue='Loan_Status',orient='h')
# relation of credit history in different Property Area with respect to there Loan_Status


# In[ ]:


sb.barplot(x='Loan_Amount_Term',y='LoanAmount',data=data,hue='Loan_Status',palette='Blues')
#visualizing LoanAmount on the basis of LoanAmountTerm with respect to Loan_Status


# <div style='font-size:20px'>Handling Categorical Columns :</div>

# In[ ]:


#As above we observe that our there are so many columns with categorical values.
#which are useful feature for predicting our Loan Status at the end
#for the sake of simplicity I am coverting these categorical values in to numeric values.
x = pd.Categorical(data['Gender'])               # Male=1,Female=0
data['Gender']=x.codes

x = pd.Categorical(data['Married'])              # Yes=1,No=0
data['Married']=x.codes

x = pd.Categorical(data['Education'])            #Graduate=0,Non-graduated=1
data['Education']=x.codes

x = pd.Categorical(data['Self_Employed'])        #Yes=1,No=0
data['Self_Employed']=x.codes

x = pd.Categorical(data['Property_Area'])        # Rural=0,SemiUrban=1,Urban=2
data['Property_Area']=x.codes

x = pd.Categorical(data['Loan_Status'])          #Y=1,N=0
data['Loan_Status'] = x.codes

#in dependent column we clearly see that there is + sign for dependents more than 3
#which makes it column of object data type 
#So, I am going to remove this sign and convert it into numeric value
data['Dependents'] = data['Dependents'].str.replace('+','')     
data['Dependents'] = pd.to_numeric(data['Dependents'])


# In[ ]:


plt.figure(figsize=(10,7))
sb.heatmap(data.corr(),cmap='Greens',annot=True)
#Visualizing the correlation matrix using heatmap 


# <div style='font-size:20px'>Feature Selection Process :</div>
# <div style='font-size:15px'>There many methods for selecting some of the specific features but, now I am using RandomForest Classifier for feature selection.
# </div>
# 
# <div style='font-size:20px;padding-top:20px'>
# Steps are :
# <ul style='font-size:15px'>
# <li>Fit training data set to RandomForest Classifier</li>
# <li>Use feature_importance to see the importance of each feature for predicting</li>
# <li>Then pick few features having high feature_importance value.</li>
# </ul>
# </div>

# In[ ]:


#We are going to predict the Loan_Status
Y=data['Loan_Status']
X=data.drop(columns=['Loan_Status','Loan_ID']) #X is all columns except Loan_Status and Loan_ID
# split the train and test dataset where test set is 30% of original dataset
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3) 


# <div style='font-size:18px;color:brown;font-weight:bold'>RandomForest Classifier :</div>
# <p style='font-size:15px'>Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees. </p>

# In[ ]:


clf = RandomForestClassifier(n_estimators=400,max_depth=5) #defining RandomForest Classifier


# In[ ]:


clf = clf.fit(xtrain,ytrain)  #fitting our train dataset


# In[ ]:


clf.score(xtest,ytest)       #score on our test dataset


# In[ ]:


pd.Series(clf.feature_importances_,xtrain.columns).sort_values(ascending=False)
#feature importance in descending order
#So, I am using only top 4 features as my input


# In[ ]:


#Respliting the trianing and testing dataset
Y=data['Loan_Status']
X=data[['Credit_History','Total_Income','LoanAmount']] #X is top 3 feature having more feature importance values
# split the train and test dataset where test set is 30% of original dataset
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3) 


# In[ ]:


#Re-applying the RandomForest Classifiers
clf = RandomForestClassifier(n_estimators=400,max_depth=5) 
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)
#we can clearly observe that it increases the accuracy percentage


# <div style='font-size:18px;color:brown;font-weight:bold'>Logistic Regression :</div>
# <p style='font-size:15px'>The logistic model is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist.</p>

# In[ ]:


clf = LogisticRegression()  #defining Logistic Regression
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)


# <div style='font-size:18px;color:brown;font-weight:bold'>Support Vector Classifier :</div>
# <p style='font-size:15px'>Support-Vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.A support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.</p>

# In[ ]:


clf = SVC()  #defining Support Vector Classifier
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)


# <div style='font-size:18px;color:brown;font-weight:bold'>K-nearest Neighbors Classifier :</div>
# <p style='font-size:15px'>An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.</p>

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=3)  #defining K-nearest Neighbors(KNN) Classifier
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)


# <div style='font-size:18px;color:brown;font-weight:bold'>DecisionTree Classifier :</div>
# <p style='font-size:15px'>A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.</p>

# In[ ]:


clf = DecisionTreeClassifier(max_depth=3)  #defining DecisionTree Classifier
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)


# <div style='font-size:18px;color:brown;font-weight:bold'>GradientBoosting Classifier :</div>
# <p style='font-size:15px'>Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.</p>

# In[ ]:


clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=2)  #defining Logistic Regression
clf = clf.fit(xtrain,ytrain) 
clf.score(xtest,ytest)

