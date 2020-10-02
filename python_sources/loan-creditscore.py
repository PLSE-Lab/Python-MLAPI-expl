#!/usr/bin/env python
# coding: utf-8

# 
# **Dictionary for BK digital loan data**
# 
# 
# **Customer Id**             :Customer's identification number
# 
# **Date of Birth**           :Customer's date of birth
# 
# **Province**               :Province in which the national Id was issued
# 
# **District**                :District in which the national Id was issued
# 
# **Customer Branch**         :Branch in which the account was opened
# 
# **Principal amt**           :Disbursed loan amount
# 
# **Paid principal**          :Amount paid of the disbursed amount
# 
# **Paid Interest**           :Interest paid on the loan
# 
# **Paid Penalty**            :Penalty paid due to late payment
# 
# **Total Remaining prinipal**:Total outstanding amount of the loan
# 
# **Remaining principal**     :Outstanding amount minus amount for the                                         nextinstallement
# 
# **Amount due**            :Total amount due for the current installemnet (due                              principal plus interest)
# 
# **Due Principal**          :Due amount of the principal for the current                                     installement (does not include interest)
# 
# **Due interest**            :Interest amount due for the current installement
# 
# **Due pen interest**        :Due penality interest for the current installement 
# 
# **Due Fee**                 :Processing fee paid on disbursement (1% of the principal amount on loans disbursed after 26th April 2019)
# 
# **Paid Fee**                :Same as Due fee
# 
# **Overdue Days**            :Days passed without paying due amount for each 
#                              installement
#                              
# **Effective Date**          :Loan value date
# 
# **Maturity Date**           :Date  after which the loan expires if not paid
# 
# **CreditScoreGroup**        :Credit score to which a given customer belongs to.
# 
# **PaymentStatus**           :The status of the customer's current loan
# 
# **Duration**               :Loan duration in months
# 
# **ReturningCustomer**     :Indicates whether a custoner is returnig or new
# 
# **Class**             :Customer's class depending on the number of overdue days
# 
#                              *overdue days they have*                        
# 					Overdue Days >= 1 & Overdue Days < 30: Acceptable Risk     
# 					Overdue Days >= 30 & Overdue Days < 90: Special Mention
# 					Overdue Days >= 90 & Overdue Days < 180: Substandard
# 					Overdue Days >= 180  & Overdue Days < 360: Doubtful
# 					Overdue Days >= 360: Loss
# 					Overdue Days = 0: Normal
# <br />
# 
# 
# 

# ## The below Report is divided into 3 main sections namely:
# 
# ### 1. Initialize Libraries, Load Data & Preprocess
# ### 2. Exploratory Data Analysis and Visualization
# ### 3. Predictive Modeling(predicting credit score)

# - ### Goal of the study is to create a machine learning based model that predicts credit score of a client.
# - ### This is a Supervised regression problem. Where credit score A(0), B(1) and C(2) is the dependant variable

# To find the predictability of a creditscore our main objective is to find what features can play a role to predict a CreditscoreGroup? Therefore we need to find answers to some questions like:
# - **1.** Is the % of CreditScoreGroup significantly different from Paid Penalty,?
# - **2.** How does Duration effect the proportion of CreditScoreGroup ?
# - **3.** Does the ReturningCustomer play a role in the % of CreditScoreGroup?
# - **4.** Which age group constitutes for higher proportion of CreditScoreGroup ?
# - **5.** Is the number of CreditScoreGroup correlated with Due pen interest  ?
# - **6.** Is there a pattern in PaymentStatusstatuses which can help predict probability of a CreditScoreGroup ?
# - **7.** Does the Total Remaining prinipal amount has a correlation with the % of CreditScoreGroup?

# #### Import the required Packages 

# In[ ]:


import scipy

import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from IPython.core.interactiveshell import InteractiveShell
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore') # to supress seaborn warnings
pd.options.display.max_columns = None # Remove pandas display column number limit
#InteractiveShell.ast_node_interactivity = "all" # Display all values of a jupyter notebook cell
import sys
#import savReaderWriter as sav
#import the evaluatation metric
from sklearn.metrics import balanced_accuracy_score
#pandas library for reading data
import pandas as pd
#numpy library for computation with matrices and arrays
import numpy as np
#matplotlib library for visualization
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
#command for displaying visualizations within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix, recall_score, precision_score

import seaborn as sns


# In[ ]:


# Read the data into DataFrames.
LOANDATA=pd.read_csv("../input/LOANDATA1.csv")
# Breif look at the data
LOANDATA.head()


# In[ ]:


#Check feaures of the data , data types and number of rows for each column
LOANDATA.info()


# In[ ]:


#View some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values
LOANDATA.describe()


# In[ ]:


#Check total number of rows and columns within dataframe
LOANDATA.shape


# ### Exploratory data analysis of a Pandas Dataframe with pandas profiling

# In[ ]:


import pandas_profiling as pp
pp.ProfileReport(LOANDATA)


# From above data analysis explored using pandas profiling shows that Due TAX,Due interest,Due Principal,Paid Fee and Paid Tax are highly correlated to each other, but very less correlation to target label 'CreditScoreGroup '. When data is huge to save computational resource, such features can be dropped without losing significant prediction power.
# 
# Class has Missing values 49613 (%) which is equal 85.4% of total data thus we can drop column since since the missing values is quite significant.
# 
# 

# In[ ]:


#Checking further infomation about CreditScoreGroup
LOANDATA.CreditScoreGroup.value_counts()


# In[ ]:


#Checking further infomation about Province
LOANDATA.Province.value_counts()


# In[ ]:


#Checking further infomation about Class
LOANDATA.Class.value_counts()


# ### Checking the total number of missing values from each column within the dataframe 

# In[ ]:


LOANDATA.isna().sum()


# #### Graphical representation of data where the individual values contained in a matrix are represented using heatmap to check missing vallues

# In[ ]:


sns.heatmap(LOANDATA.isnull(),cmap="winter_r")


# We can see alot of missing values wthin class column and some few missing values in province and district columns

# In[ ]:


#Making a copy of loandata so that we can utilize the copy for further feature enginneering
LOANDATA1=LOANDATA.copy()


# In[ ]:



cor = LOANDATA1.corr()
plt.figure(figsize=(18,18))
sns.heatmap(cor, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
            xticklabels=cor.columns.values,
            yticklabels=cor.columns.values)


# In[ ]:


#replacing spaced words in culumns (' ', '_')  with underscore for ease 
LOANDATA1.columns =LOANDATA1.columns.str.strip().str.replace(' ', '_')
LOANDATA1.head(1)


# In[ ]:


#LOANDATA1['Unnamed:_0']
LOANDATA1.drop('Unnamed_0', axis=1,inplace=True)


# In[ ]:


LOANDATA1.set_index(['Customer_Id'],inplace=True)


# In[ ]:


LOANDATA1.Class.value_counts()


# In[ ]:


LOANDATA1.head()


# ### Checking class Imbalance

# In[ ]:


Creditscore_count =LOANDATA1.CreditScoreGroup.value_counts()
print('Class 0:', Creditscore_count[0])
print('Class 1:', Creditscore_count[1])
print('Class 2:', Creditscore_count[2])
print('Proportion:', round(Creditscore_count[0] / (Creditscore_count[1]+Creditscore_count[2]), 3), ': 1')

Creditscore_count.plot(kind='bar', title='Count (Creditscore)');


# **From above class imbalance check we can see that prorpotion is 0.72 which 72% this shows that data fairly distributed thus may not couse advesre effect on accuracy**

# **Dropping the column of  class has Missing values 49613 (%) which is equal 85.4% of total data thus we can drop column since since the missing values is quite significant**

# In[ ]:


LOANDATA1.drop('Class',axis=1,inplace=True)


# ### Converting  Categorical data to Numerical

# In[ ]:


LOANDATA1["CreditScoreGroup"]=LOANDATA1["CreditScoreGroup"].map({"A": 0, "B": 1, "C": 2}).astype(int)


# In[ ]:


LOANDATA1=LOANDATA1.fillna(value={"Province":"Umujyi wa Kigali"})


# In[ ]:


LOANDATA1["Province"]=LOANDATA1["Province"].map({"Umujyi wa Kigali": 0, "Iburasirazuba": 1, "Iburengerazuba": 2,'Amajyepfo':3,"Amajyaruguru":4,"Diaspora - A":5}).astype(int)


# In[ ]:


LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].str.strip().str.replace(' ', '_')


# In[ ]:


LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].map({"Completely_Repaid":0,"Partially_Repaid":1,"In_arrears":2,"Not_yet":3}).astype(int)


# In[ ]:


LOANDATA1["ReturningCustomer"]=(LOANDATA1["ReturningCustomer"]).astype(int)


# In[ ]:


LOANDATA4=LOANDATA1.drop(['District','DOB','Effective_Date','Maturity_Date','Date_of_Birth'], axis=1)


# ### Visualize Data with t-SNE
# 
# t-SNE is a technique for dimensionality reduction that is well suited to visualise high-dimensional datasets. Lets have a first look on the map that will set some expectations for the prediction accuracy i.e. if our dataset has many overlaps it would be good if our model achieves an accuracy of 60-70%.!

# In[ ]:


#Set df4 equal to a set of a sample of 1000 deafault and 1000 non-default observations.
df1 = LOANDATA4[LOANDATA1.CreditScoreGroup == 0].sample(n = 1000)
df2 = LOANDATA4[LOANDATA1.CreditScoreGroup == 1].sample(n = 1000)
df3 = LOANDATA4[LOANDATA1.CreditScoreGroup == 2].sample(n = 1000)
df4 = pd.concat([df1,df2,df3], axis = 0)

#Scale features to improve the training ability of TSNE.
standard_scaler = StandardScaler()
df4_std = standard_scaler.fit_transform(df4)

#Set y equal to the target values.
y = df4.CreditScoreGroup

tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(df4_std)

Creditscore_count =LOANDATA1.CreditScoreGroup.value_counts()
print('Class 0:', Creditscore_count[0])
print('Class 1:', Creditscore_count[1])
print('Class 2:', Creditscore_count[2])
print('Proportion:', round(Creditscore_count[0] / (Creditscore_count[1]+Creditscore_count[2]), 3), ': 1')

Creditscore_count.plot(kind='bar', title='Count (Creditscore)');


# ### Build the scatter plot to show dataset distribution and model  accuracy  

# In[ ]:


#Build the scatter plot with the three types of transactions.
color_map = {0:'red', 1:'blue',2:'yellow'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x = x_test_2d[y==cl,0], y = x_test_2d[y==cl,1], c = color_map[idx], label = cl)
    #plt.scatter(x = x_test_2d[y==cl,0], y = x_test_2d[y==cl,2], c = color_map[idx], label = cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper right')
plt.title('t-SNE visualization of train data')
plt.show()


# **Plotting histogram od loan data to check data distribution**

# In[ ]:


plt.figure(figsize=(10,10));
LOANDATA4.hist(figsize=(10,10));


# **The plot reveals a rather mixed up dataset which means we should not expect very accurate model.**

# - **Now let us check the correlation between different features**

# In[ ]:



cor = LOANDATA4.corr()
plt.figure(figsize=(18,18))
sns.heatmap(cor, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
            xticklabels=cor.columns.values,
            yticklabels=cor.columns.values)


# From above data analysis explored using pandas profiling shows that Due TAX,Due interest,Due Principal,Paid Fee and Paid Tax are highly correlated to each other, but very less correlation to target label 'CreditScoreGroup '. When data is huge to save computational resource, such features can be dropped without losing significant prediction power.
# 
# 'Overdue Days' show highest contribution to the CreditscoreGroup label

# - **We can see above that Overdue Days,Previous Days,Customer Branch...have high positive correlation to 'CreditScoreGroup ' and ReturningCustomer has pretty high negative correlation**

# ### Feature Engineering

# ## 1. Is the % of CreditScoreGroup significantly different between Completely Repaid	Partially Repaid	In arrears	Not yet customers ?
# 

# In[ ]:


PaymentStatus_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['PaymentStatus'], margins=True, normalize=False)
new_index = {0: 'A', 1:'B',2: 'C', }
new_columns = {0:"Completely_Repaid",1:"Partially_Repaid",2:"In_arrears",3:"Not_yet"}
PaymentStatus_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
PaymentStatus_crosstab/PaymentStatus_crosstab.loc['All']


# **We can see customers in A whose payment is in arrears has lowest credit score while customers who have partially paid has highest credit score**
# 
# **We can see customers in B whose payment status is Completely Repaid has higher credit score while customers who have partially paid has lowest credit score**
# 
# **We can see customers in C whose payment status is In arrearsd has higher credit score while customers who have partially paid has lower credit score**
# 
# 

# In[ ]:


ReturningCustomer_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['ReturningCustomer'], margins=True, normalize=False)
new_index = {0: 'A', 1:'B',2: 'C', }
new_columns = {0: 'False', 1:'True'}
ReturningCustomer_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
ReturningCustomer_crosstab/ReturningCustomer_crosstab.loc['All']


# **We can see Returning customers in A higher credit score while customers who are not returning has lowest credit score**
# 
# **We can see Returning customers in B higher credit score while customers who are not returning has lowest credit score**
# 
# **We can see Returning customers in C has lowest credit score while customers who are not returning  higher credit score**
# 

# In[ ]:


pen_interest_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['Due_pen_interest'], margins=True, normalize=False)
new_index = {0: 'A', 1:'B',2: 'C', }
new_columns = {}
pen_interest_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
pen_interest_crosstab/pen_interest_crosstab.loc['All']


# **From above table customers in A whose Due penality interest is zero has highest credit score than those customers with Due penality interest**
# 
# **customers in B shows some mixed credit score based on  Due penality interest but we can see on most cases credit score is higher on section with Due penality interest**
# 
# **customers in C whose Due penality interest is zero has lower credit score than those customers with Due penality interest since with Due penality interest highest credit score**
# 

# In[ ]:


age_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['age'], margins=True, normalize=False)
new_index = {0: 'A', 1:'B',2: 'C', }
new_columns = {}
age_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
age_crosstab/age_crosstab.loc['All']


# **We can see customers in A whose  age is between 28 to 36 have high credt score**  
# 
# **We can see customers in B whose  age is between 19 to 62 have high credt score**
# 
# **In C customer at age of 18 has high credit score than the rest , also credit score reduces has age increases**

# In[ ]:


Duration_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['Duration'], margins=True, normalize=False)
new_index = {0: 'A', 1:'B',2: 'C', }
new_columns = {}
Duration_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
Duration_crosstab/Duration_crosstab.loc['All']


# * **We can see customers in A whose  Loan duration are 6  months have high credt score**  
# 
# **We can see customers in B whose  Loan duration are 12  months have high credt score** 
# 
# **We can see customers in C whose  Loan duration are 1  month have high credt score** 

# #### Import the Machine learning Packages  to enhance credit scoring

# In[ ]:


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import preprocessing, metrics
from xgboost import XGBClassifier
warnings.filterwarnings('ignore') # to supress warnings
from sklearn.model_selection import learning_curve, GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import preprocessing, metrics
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore') # to supress warnings
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[ ]:





# **Splitting data into train and test set and checking the accuracy**

# In[ ]:


x= LOANDATA4.drop(columns=['CreditScoreGroup','Due_Principal','Due_interest',
                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)
Y = LOANDATA4.CreditScoreGroup
scaler=StandardScaler()
X=scaler.fit(x).transform(x)
# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)


# In[ ]:





# ******Here we are first trying out below listed classification models to get the first look at accuracy**

# In[ ]:


# list of different classifiers we are going to test
clfs = {
'LogisticRegression' : LogisticRegression(),
'GaussianNB': GaussianNB(),
'RandomForest': RandomForestClassifier(),
'DecisionTreeClassifier': DecisionTreeClassifier(),
'SVM': SVC(),
'KNeighborsClassifier': KNeighborsClassifier(),
'GradientBoosting': GradientBoostingClassifier(),
'XGBClassifier': XGBClassifier()
}


# In[ ]:


# code block to test all models in clfs and generate a report
models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])

for clf, clf_name in zip(clfs.values(), clfs.keys()):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.score(x_test,y_test)
    
    #print('Calculating {}'.format(clf_name))
    t = pd.Series({ 
                     'Model': clf_name,
                     'Precision_score': metrics.precision_score(y_test, y_pred,average='macro'),
                     'Recall_score': metrics.recall_score(y_test, y_pred,average='macro'),
                     'F1_score': metrics.f1_score(y_test, y_pred,average='macro'),
                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}
                   )

    models_report = models_report.append(t, ignore_index = True)

models_report


# ### Ensemble Framework Overview
# Ensemble learning is a machine learning paradigm in which a number of learners are trained
# to solve the same problem with the goal of obtaining better predictive accuracy than could have
# been achieved from any of the constituent learning models alone [Zhou, 2015]. It is a well-established
# and widely employed methodology designed to enhance the generalizable signal by averaging out
# noise from a diverse set of models.

# ### Function to optimize model using gridsearch

# In[ ]:


# Function to optimize model using gridsearch 
def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):
    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)
    gs.fit(x_train, y_train)
    print ('Best params: ', gs.best_params_)
    print ('Best AUC on Train set: ', gs.best_score_)
    print( 'Best AUC on Test set: ', gs.score(x_test, y_test))

# Function to generate confusion matrix
def confmat(pred, y_test):
    conmat = np.array(confusion_matrix(y_test, pred, labels=[0,1,2]))
    conf = pd.DataFrame(conmat, index=['A', 'B','C'],
                             columns=['Predicted A', 'Predicted B','Predicted C'])
    print( conf)

# Function to plot roc curve
def roc(prob, y_test):
    y_score = prob
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])
    plt.figure(figsize=[7,7])
    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)
    plt.plot([1,0], [1,0], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive rate', fontsize=15)
    plt.ylabel('True Positive rate', fontsize=15)
    plt.title('ROC curve for Credit Default', fontsize=16)
    plt.legend(loc='Lower Right')
    plt.show()
    
def model(md, x_train, y_train,x_test, y_test):
    md.fit(x_train, y_train)
    pred = md.predict(x_test)
    #prob = md.predict_proba(x_test)[:,1]
    print( ' ' )
    print ('Accuracy on Train set: ', md.score(x_train, y_train))
    print( 'Accuracy on Test set: ', md.score(x_test, y_test))
    print( ' ')
    print(classification_report(y_test, pred))
    print( ' ')
    print('Confusion Matrix',confmat(pred, y_test))
    
    #roc(prob, y_test)
    return md


# **Splitting data into train test and validation set and checking the accuracy**

# In[ ]:


x= LOANDATA4.drop(columns=['CreditScoreGroup','Due_Principal','Due_interest',
                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)
Y = LOANDATA4.CreditScoreGroup
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10
pca = PCA(n_components = 23)
kfold = 5

scaler=StandardScaler()
X=scaler.fit(x).transform(x)
# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
StratifiedKFold(n_splits=kfold, random_state=42)
# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

#print(x_train, x_val, x_test


# In[ ]:


clfs = {
'LogisticRegression' : LogisticRegression(),
'GaussianNB': GaussianNB(),
'RandomForest': RandomForestClassifier(),
'DecisionTreeClassifier': DecisionTreeClassifier(),
'SVM': SVC(),
'KNeighborsClassifier': KNeighborsClassifier(),
'GradientBoosting': GradientBoostingClassifier(),
'XGBClassifier': XGBClassifier()
}


# In[ ]:



 # code block to test all models in clfs and generate a report
models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])

for clf, clf_name in zip(clfs.values(), clfs.keys()):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_val)
    y_score = clf.score(x_val,y_val)
    
    #print('Calculating {}'.format(clf_name))
    t = pd.Series({ 
                     'Model': clf_name,
                     'Precision_score': metrics.precision_score(y_val, y_pred,average='macro'),
                     'Recall_score': metrics.recall_score(y_val, y_pred,average='macro'),
                     'F1_score': metrics.f1_score(y_val, y_pred,average='macro'),
                     'Accuracy': metrics.accuracy_score(y_val, y_pred)}
                   )

    models_report = models_report.append(t, ignore_index = True)

models_report


# **The accuracy for train test **

# In[ ]:


# code block to test all models in clfs and generate a report
models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])

for clf, clf_name in zip(clfs.values(), clfs.keys()):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.score(x_test,y_test)
    
    #print('Calculating {}'.format(clf_name))
    t = pd.Series({ 
                     'Model': clf_name,
                     'Precision_score': metrics.precision_score(y_test, y_pred,average='macro'),
                     'Recall_score': metrics.recall_score(y_test, y_pred,average='macro'),
                     'F1_score': metrics.f1_score(y_test, y_pred,average='macro'),
                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}
                   )

    models_report = models_report.append(t, ignore_index = True)

models_report


# # Discussion and Insights :
# - Logistic Regression produced results with a lower accuracy but overall performance is lower than other models. 
# <br>
# - Decision Tree dominated over Logistic Regression in all cases.
# <br>
# 
# -  GradientBoosting model with feature selector f_classf would be the best method to use because it has the Highest RECALL value and highest accuracy
# <br>
# -  XGBClassifier model with feature selector f_classf would be the second best method to use because it has the second Highest RECALL value  and second highest accuracy
# 
# - Obviously,changing the threshold affects the performance of the model and this can be observed in the next section.
# <br>
# - This can be further extended by Resampling of the data to increase the RECALL score

# **Since GradientBoosting and XGBClassifier have higher accuracy,we will select the two for further algorthm tunning to improve accuracy **

# **We can improve accuracy  in selected models through algorithm tuning know that machine learning algorithms are driven by parameters. These parameters majorly influence the outcome of learning process.Since algorithm tuning find the optimum value for each parameter to improve the accuracy of the model**

# In[ ]:


# Function to optimize model using gridsearch 
def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):
    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)
    gs.fit(x_train, y_train)
    print ('Best params: ', gs.best_params_)
    print ('Best AUC on Train set: ', gs.best_score_)
    print( 'Best AUC on Test set: ', gs.score(x_test, y_test))

# Function to generate confusion matrix
def confmat(pred, y_test):
    conmat = np.array(confusion_matrix(y_test, pred, labels=[0,1,2]))
    conf = pd.DataFrame(conmat, index=['A', 'B','C'],
                             columns=['Predicted A', 'Predicted B','Predicted C'])
    print( conf)

# Function to plot roc curve
def roc(prob, y_test):
    y_score = prob
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])
    plt.figure(figsize=[7,7])
    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)
    plt.plot([1,0], [1,0], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive rate', fontsize=15)
    plt.ylabel('True Positive rate', fontsize=15)
    plt.title('ROC curve for Credit Default', fontsize=16)
    plt.legend(loc='Lower Right')
    plt.show()
    
def model(md, x_train, y_train,x_test, y_test):
    md.fit(x_train, y_train)
    pred = md.predict(x_test)
    #prob = md.predict_proba(x_test)[:,1]
    print( ' ' )
    print ('Accuracy on Train set: ', md.score(x_train, y_train))
    print( 'Accuracy on Test set: ', md.score(x_test, y_test))
    print( ' ')
    print(classification_report(y_test, pred))
    print( ' ')
    print('Confusion Matrix',confmat(pred, y_test))
    
    #roc(prob, y_test)
    return md


# **Improving accuracy  in through algorithm tuning **

# In[ ]:


# feature selection with the best model from grid search
gb = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 8,n_estimators=1000, max_features = 0.9,min_samples_leaf = 100)
model_gb = model(gb, x_train, y_train,x_test, y_test)


# In[ ]:


# Use gridsearch to fine tune the parameters
xgb = XGBClassifier()
xgb_params = {'n_estimators':[200,300],'learning_rate':[0.05,0.02], 'max_depth':[4],'min_child_weight':[0],'gamma':[0]}
gridsearch(xgb, xgb_params,x_train, x_test, y_train, y_test,5)


# In[ ]:


# feature selection with the best model from grid search
xgb = XGBClassifier(
 learning_rate =0.2,
 n_estimators=200,
 max_depth=7,
 eta=0.025,
 min_child_weight=10,
 gamma=0.65,
 max_delta_step=1.8,
 subsample=0.9,
 colsample_bytree=0.4,
 objective= 'binary:logistic',
 nthread=1,
 scale_pos_weight=1,
 thresh = 0.5,
 reg_lambda=1,
 booster='gbtree',
 n_jobs=1,
 num_boost_round=700,
 silent=True,
 seed=30)
model_xgb = model(xgb, x_train, y_train,x_test, y_test)


# The classification metrics of iterest for this fairly imbalanced dataset are: 
# - precision = tp / (tp + fp)
# - recall = tp / (tp + fn)
# - f1 = 2(precision)(recall) / (precision + recall)
# - Roc curve area
# 
# Depending upon banks operational costs & ideology a large bank may follow the principal that fewer False Positives are preferable over a few more False Negatives to be able to lend more & spend less on investigations on the contrary a conservative approach would go with the opposite i.e more accuracy.
# 
# ### Therefore we see that XGBoost trains with little higher accuracy and auc score than GradientBoost. We will use XGBoost for final predictions. i.e. fewer False Positives are preferable over a few more False Negatives

# **We will use model_gb3 of XGboost since it has highest accuracy **

# In[ ]:


LOANDATA4['PREDICTED_STATUS']=np.int_(model_gb.predict(LOANDATA4.drop(['CreditScoreGroup','Due_Principal','Due_interest',
                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)))
LOANDATA4.index.names =['Customer_Id']


# In[ ]:


LOANDATA4[20:30]


# **One can check credit score of specific  customer using customer ID which is unique identifier on Jupiter Notebook**

# In[ ]:


LOANDATA4.loc[851947]


# ### Viewing credit score of any randomly selected customer using customer ID

# **Save the output to csv file in desired format**

# In[ ]:


#LOANDATA4['PREDICTED_STATUS'].to_csv("LOANDATA4_Predict.csv")


# **XGBoost  have the ability to evaluate loan applications based on a large set of variables and learn from actual outcomes of loans to provide a likely outcome. By building its own algorithm through experience, machine learning technology such as XGBoost  can eliminate bias, and enable underwriters to assess debt-to-income ratio in an accurate manner. When presented with hard facts, it can advise underwriters and auditors on the best decision to take, and identify loans that are more likely to default through predictive credit score**

# **Our solution is unique since we are using local data which has not been used before**
