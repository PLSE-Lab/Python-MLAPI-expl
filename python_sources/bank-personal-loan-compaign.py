#!/usr/bin/env python
# coding: utf-8

# <h4><font color='#F55905'> <u>BUSINESS PROBLEM:</u></font><center><br><br><font color='#15657F'>REDUCE THE CAMPAIGN COST FOR THE PRODUCT(PERSONAL LOAN) OF IDBI BANK</font></center></h4>
# 
# This case is about a bank (IDBI Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). 
# 
#    ***A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success.  This has encouraged the retail marketing department to devise campaigns to better target marketing to increase the success ratio with a minimal budget.***
#    
# The department wants to build a model that will help them identify the potential customers who have a higher probability of purchasing the loan. 
# This will increase the success ratio while at the same time reduce the cost of the campaign.
# 
# The file given below contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). 
# Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign
# 
# <h4><font color='#F55905'><u>MACHINE LEARNING PROBLEM:</u></font><center><br><br><font color='#15657F'>USE A CLASSIFICATION MODEL TO PREDICT THE LIKELYHOOD OF A LIABILITY CUSTOMER BUYING PERSONAL LOANS</font></center></h4>

# <h4><font color='#F55905'><u>ATTRIBUTE DESCRIPTION:</u></font></h4>

# <table cellspacing="0" border="0">
# 	<tr>
# 		<td height="17" align="left">ID</td>
# 		<td align="left">Customer ID</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Age</td>
# 		<td align="left">Customer's age in completed years</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Experience</td>
# 		<td align="left">#years of professional experience</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Income</td>
# 		<td align="left">Annual income of the customer</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">ZIPCode</td>
# 		<td align="left">Home Address ZIP code.</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Family</td>
# 		<td align="left">Family size of the customer</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">CCAvg Avg.</td>
# 		<td align="left">spending on credit cards per month</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Education</td>
# 		<td align="left">Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Mortgage</td>
# 		<td align="left">Value of house mortgage if any</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Personal Loan</td>
# 		<td align="left">Did this customer accept the personal loan offered in the last campaign?</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Securities Account</td>
# 		<td align="left">Does the customer have a securities account with the bank?</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">CD Account</td>
# 		<td align="left">Does the customer have a certificate of deposit (CD) account with the bank?</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">Online</td>
# 		<td align="left">Does the customer use internet banking facilities?</td>
# 	</tr>
# 	<tr>
# 		<td height="17" align="left">CreditCard</td>
# 		<td align="left">Does the customer uses a credit card issued by UniversalBank?</td>
# 	</tr>
# </table>

# <h4><font color='#F55905'><u> EVALUATION METRIC:</u></font></h4> <br>Recall for the positive class (Correctly classify people who opt for personal loan)

# In[ ]:


import os

from sklearn.metrics import classification_report


def classifcation_report_train_test(y_train, y_train_pred, y_test, y_test_pred):

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TRAIN DATA
            =========================================
            ''')
    print(classification_report(y_train, y_train_pred))

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TEST DATA
            =========================================
            ''')
    print(classification_report(y_test, y_test_pred))


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ## Few important parameters of read_csv
# 
# ### skiprows = Line numbers to skip  or we can specify the set of row names to exclude 
# #####                    way 1: pd.read_csv('Bank.csv',skiprows=4)
# #####                   way 2: pd.read_csv('Bank.csv',skiprows=[0,1,2,3,6])
# 
# 
# ### skipfooter = number of lines to skip from the bottom
# #####  pd.read_csv('Bank.csv',skiprows=[0,1,2,3,6],skipfooter=0)
# 
# 
# ### nrows = no of rows to read from a file. useful for big datasets
# 
# ### na_values = additional strings to recognise as NA/NAN values
# #### pd.read_csv('Bank.csv',skiprows=[0,1,2,3],na_values={'Age':['Null']},nrows=10) its for feature specific
# #### pd.read_csv('Bank.csv',skiprows=[0,1,2,3],na_values=['Null'],nrows=10)
# 
# 
# ### index_col = used to set the column as a row label
# #### pd.read_csv('Bank.csv',skiprows=[0,1,2,3],na_values=['Null'],nrows=10,index_col='ID')
# 
# ### usecols = used to subset the set of columns to read 
# #### pd.read_csv('Bank.csv',skiprows=[0,1,2,3],na_values=['Null'],nrows=10,usecols=['Age','Income'])

# In[ ]:


# Load the Bank.csv
data=pd.read_csv("../input/bank-personal-loan-campaign/Bank.csv",skiprows=4,skipfooter=3,na_values=['Null'])


# In[ ]:


# Check the head and tail of data
data.head(6)


# In[ ]:


data.tail(6)


# The variable ID,ZIP Code does not add any interesting information. There is no association between a person's customer ID ,ZIP Code and loan.So We Can Drop ID ,ZipCode

# In[ ]:


data=data.drop(columns=['ID','ZIP Code'],axis=1)
data.head()


# In[ ]:


#converting the column names to Lower case
data.columns = map(str.lower, data.columns)
data.columns


# In[ ]:


#replace spaces in column names with _
data.columns = [x.replace(' ', '_') for x in data.columns]
data.columns


# In[ ]:


data.head()


# In[ ]:


#check for null values
data.apply(lambda x : sum(x.isnull()))


# In[ ]:


#find unique levels in each column\
data.apply(lambda x: len(x.unique()))


# # Another way to find unique levels in each column

# In[ ]:



def myfunc(x):
    return len(x.unique())

data.apply(myfunc)


# In[ ]:


data.apply(lambda x: len(x.unique())).sort_values()


# In[ ]:


# check the statistics of Dataframe
data.describe()


# # For Better Understanding Transpose The Above Matrix (by using T) 

# In[ ]:


data.describe().T


# In[ ]:


#Check is there any customers with negative experience, if yes remove those rows from data 
#data[data['experience']<0].count()
print('People Having Negative Experience:',data[data['experience'] < 0]['experience'].count())
print('People Having Positive Experience:',data[data['experience'] > 0]['experience'].count())


# In[ ]:


#Check is there any customers with negative experience, if yes remove those rows from data 
data.drop(data[data['experience']<0].index,inplace=True)


# In[ ]:


data.experience.value_counts()


# there are 52 records with negative experience.so we don't want negative experience better remove those from  data

# In[ ]:


df = data.copy()
df.head()


# In[ ]:





# In[ ]:


for col in df.columns:
    if len(df[col].unique()) < 10:
        print(col, df[col].unique())


# In[ ]:





# In[ ]:





# In[ ]:


# Check the no of levels of all categorical columns

num_col=['age','experience','income',"ccavg",'mortgage']
cat_col=df.columns.difference(num_col)
cat_col


# In[ ]:


df[cat_col] = df[cat_col].apply(lambda x: x.astype('category'))
df[num_col] = df[num_col].apply(lambda x: x.astype('float'))
df.dtypes


# In[ ]:


df[cat_col]


# In[ ]:


for i in df[cat_col]:
    #print([i],':',df[cat_col[i]].unique())
    print(i,':',df[i].nunique())
    


# In[ ]:


num_data = data.loc[:,num_col]
cat_data = data.loc[:,cat_col]
num_data.head()


# ## Imputations

# In[ ]:


# check is there any NA values present, and if present impute them 
cat_data.isna().sum()


# In[ ]:


num_data.isna().sum()


# In[ ]:


#num_data.fillna()
num_data.fillna(num_data['age'].mean(), inplace = True) 


# In[ ]:


num_data.isna().sum()


# In[ ]:


full_data = pd.concat([num_data,cat_data],axis=1)
full_data.head()


# ## Visualisations

# In[ ]:


# check the Personal loan statistics with plot which is suited
full_data['personal_loan'].value_counts().plot(kind='bar')


# Vectors of data represented as lists, numpy arrays, or pandas Series objects passed directly to the x, y, and/or hue parameters.
# The x, y, and hue variables will determine how the data are plotted

# In[ ]:


# Influence of income and education on personal loan and give the observations from the plot
sns.boxplot(x='education',y='income',hue='personal_loan',data=full_data)


# Observation : It seems the customers whose education level is 1 is having more income. However customers who has taken the personal loan have the same income levels

# In[ ]:


# Influence of Credict card usage and Personal Loan  and give observations from the plot
sns.boxplot(y='ccavg',x='personal_loan',data=full_data)


# Observation : personal_loan =1, having more 'ccavg' than personal_loan=0
#      .the outliers are more in personal_loan=0 than 1

# In[ ]:


# Influence of education level on persoanl loan and give the insights
full_data.groupby(['education','personal_loan']).size().plot(kind='bar')
plt.xticks(rotation=30)                                                      


# In[ ]:





# palette : Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette().

# In[ ]:


# Influence of family size on persoanl loan and suggest the insights
sns.countplot(x='family',data=full_data,hue='personal_loan',palette='Set3')


# Observation : Family size does not have any impact in personal loan. But it seems families with size of 3 are more likely to take loan. When considering future campaign this might be good association.

# In[ ]:


# Influence of deposit account on personal loan and give the insights
sns.countplot(x='cd_account',data=full_data,hue='personal_loan')


# Observation : Customers who does not have cd_account , does not have loan as well. This seems to be majority. But almost all customers who has cd_account has loan as well

# In[ ]:


# Influence of Security account on personal loan and give the insights
sns.countplot(x="securities_account", data=full_data,hue="personal_loan")


# Observation : Majority of customers who does not have loan have securities account

# In[ ]:


# Influence of Credit card on Persoanl Loan and give insights
sns.countplot(x="creditcard", data=full_data,hue="personal_loan")


# Observation: The graph show persons who have personal loan have a higher credit card average.

# In[ ]:


#median
print('Non-Loan customers: ',full_data[full_data.personal_loan == 0]['ccavg'].median()*1000)
print('Loan customers    : ', full_data[full_data.personal_loan == 1]['ccavg'].median()*1000)


# In[ ]:


#mean
print('Non-Loan customers: ',full_data[full_data.personal_loan == 0]['ccavg'].mean()*1000)
print('Loan customers    : ', full_data[full_data.personal_loan == 1]['ccavg'].mean()*1000)


# In[ ]:


#family income personalloan realtionship
sns.boxplot(x='family',y='income',data=full_data,hue='personal_loan')


# Observation : families with income less than 100K are less likely to take loan,than families with high income

# In[ ]:


# Correlation with heat map
import matplotlib.pyplot as plt
import seaborn as sns
corr = data.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})


# In[ ]:





# In[ ]:


# Correlation with heat map
import matplotlib.pyplot as plt
import seaborn as sns
corr = data.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# Observed : 
# Income and CCAvg is moderately correlated.
# Age and Experience is highly correlated

# In[ ]:


#mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True


# In[ ]:


# creating` dummies for columns which have more than two levels
for col in cat_data.columns:
    if len(cat_data[col].unique()) > 2:
        print(col, cat_data[col].unique())


# In[ ]:


df = pd.get_dummies(data=full_data, columns=['family', 'education'], drop_first=True)
df.head()


# In[ ]:


# Train test split
## Split the data into X_train, X_test, y_train, y_test with test_size = 0.20 using sklearn
X = df.copy().drop("personal_loan",axis=1)
y = df["personal_loan"]

## Split the data into X_train, X_test, y_train, y_test with test_size = 0.20 using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

## Print the shape of X_train, X_test, y_train, y_test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


X_train.head()


# In[ ]:


X_train.isna().sum()


# In[ ]:


# after split check the proportion of target levels - train
print(y_train.value_counts(normalize=True)) 


# In[ ]:


# after split check the proportion of target levels - test
print(y_test.value_counts(normalize=True))


# In[ ]:


#you can also plot this 

import matplotlib.pyplot as plt
y_test.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


# Implement ***SVM CLASSIFIER*** with grid search 


# In[ ]:


# Predict


# In[ ]:


# Apply the follwing models and show a data frame with the all the model performances
#    1. Logistic Regression - We haven't given the code, you need to explore!
#    2. Decision trees
#    3. K-nn 
    
# Please ensure you experiment with multiple hyper parameters for the each of the above algorithms


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


lrc = LogisticRegression()

lrc.fit(X_train,y_train)

y_train_pred_lrc_be = lrc.predict(X_train)
y_test_pred_lrc_be = lrc.predict(X_test)


# In[ ]:


svc = SVC()

svc.fit(X_train,y_train)

y_train_pred_svc_be = svc.predict(X_train)
y_test_pred_svc_be = svc.predict(X_test)


# In[ ]:


dtc = DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_train_pred_dt_be = dtc.predict(X_train)
y_test_pred_dt_be = dtc.predict(X_test)


# In[ ]:


knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_train_pred_knn_be = dtc.predict(X_train)
y_test_pred_knn_be = dtc.predict(X_test)
#classifcation_report_train_test(y_train,y_train_pred_knn_be,y_test, y_test_pred_knn_be)


# In[ ]:


naive_model = GaussianNB()
naive_model.fit(X_train, y_train)

y_train_pred_nv_be = naive_model.predict(X_train)
y_test_pred_nv_be =naive_model.predict(X_test)


# In[ ]:


from sklearn.metrics import recall_score

print("Recall of DecisionTrees:",recall_score(y_test, y_test_pred_dt_be))
print("Recall of LogisticRegression:",recall_score(y_test, y_test_pred_lrc_be))
print("Recall of SupportVectorMachines:",recall_score(y_test, y_test_pred_svc_be))
print("Recall of KNearestNeighbours:",recall_score(y_test, y_test_pred_knn_be))
print("Recall of naiibeys:",recall_score(y_test, y_test_pred_nv_be))


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, lrc,scoring='recall')
plt.show()


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, svc,scoring='recall')
plt.show()


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, dtc,scoring='recall')
plt.show()


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, knn,scoring='recall')
plt.show()


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, naive_model,scoring='recall')
plt.show()


# # Standardization

# In[ ]:


#Scale the numeric attributes

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train[num_col])

X_train[num_col] = scaler.transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])
X_train.head()


# In[ ]:


#Build svc Classifier
from sklearn.svm import SVC


# In[ ]:


## Create an SVC object and print it to see the default arguments

svc = SVC(class_weight='balanced')
svc


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


y_train_pred_svc = svc.predict(X_train)
y_test_pred_svc = svc.predict(X_test)


# In[ ]:


#from sklearn.metrics import classifcation_report_train_test
classifcation_report_train_test(y_train, y_train_pred_svc, y_test, y_test_pred_svc)


# In[ ]:


get_ipython().system('pip install mlxtend')


# # trial 1 Hyperpameter tuning

# In[ ]:


## Use Grid Search for parameter tuning

from sklearn.model_selection import GridSearchCV

svc_grid = SVC(class_weight='balanced')
 

param_grid = {

'C': [0.001, 0.01, 0.1, 1, 10],
'gamma': [0.001, 0.01, 0.1, 1], 
'kernel':['linear', 'rbf']}

 
svc_cv_grid = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)


# In[ ]:


svc_cv_grid.fit(X_train, y_train)


# In[ ]:


# Display the best estimator
svc_cv_grid.best_estimator_


# In[ ]:


#predicting using best_estimator
y_train_pred_svc_best = svc_cv_grid.best_estimator_.predict(X_train)
y_test_pred_svc_best = svc_cv_grid.best_estimator_.predict(X_test)


# In[ ]:


#classification reprot by using base function created on first cell
classifcation_report_train_test(y_train, y_train_pred_svc_best, y_test, y_test_pred_svc_best)


# # trial 2 Hyperparameter tuning

# In[ ]:


## Use Grid Search for parameter tuning

from sklearn.model_selection import GridSearchCV

svc_grid = SVC(class_weight='balanced')
 

param_grid = {

'C': [0.6,0.7,0.8,0.9,1,1.5],
'gamma': [1,2,3,4,5], 
'kernel':['linear', 'rbf']}

 
svc_cv_grid2 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)


# In[ ]:


svc_cv_grid2.fit(X_train, y_train)


# In[ ]:


svc_cv_grid2.best_estimator_


# In[ ]:


y_train_pred_svc_best2 = svc_cv_grid2.best_estimator_.predict(X_train)
y_test_pred_svc_best2 = svc_cv_grid2.best_estimator_.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, y_train_pred_svc_best2, y_test, y_test_pred_svc_best2)


# 
# # Trial 3

# In[ ]:


## Use Grid Search for parameter tuning

from sklearn.model_selection import GridSearchCV

svc_grid = SVC(class_weight='balanced')
 

param_grid = {

'C': [0.6,0.7,0.8,0.9,1,1.5],
'gamma': [0.6,0.7,0.8],  
'kernel':['linear', 'rbf']}

 
svc_cv_grid3 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)


# In[ ]:


svc_cv_grid3.fit(X_train, y_train)


# In[ ]:


svc_cv_grid3.best_estimator_


# In[ ]:


y_train_pred_svc_best1 = svc_cv_grid3.best_estimator_.predict(X_train)
y_test_pred_svc_best1 = svc_cv_grid3.best_estimator_.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, y_train_pred_svc_best1, y_test, y_test_pred_svc_best1)


# # Trial 4

# In[ ]:


## Use Grid Search for parameter tuning

from sklearn.model_selection import GridSearchCV

svc_grid = SVC(class_weight='balanced')
 

param_grid = {

'C': [0.9,1,1.2,1.3,1.4],
'gamma': [0.6,0.7,0.8], 
'kernel':['linear', 'rbf']}

 
svc_cv_grid4 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)


# In[ ]:


svc_cv_grid4.fit(X_train, y_train)


# In[ ]:


svc_cv_grid4.best_estimator_


# In[ ]:


svc_cv_grid4.best_params_


# In[ ]:


y_train_pred_svc_best2 = svc_cv_grid4.best_estimator_.predict(X_train)
y_test_pred_svc_best2 = svc_cv_grid4.best_estimator_.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, y_train_pred_svc_best2, y_test, y_test_pred_svc_best2)


# # Learning Curves

# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, svc_cv_grid4.best_estimator_,scoring='recall')


# # KNN random tuning

# In[ ]:


knn1 = KNeighborsClassifier(n_neighbors= 3 , weights = 'uniform', metric='euclidean')
knn1.fit(X_train, y_train)    
predicted = knn1.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predicted)
print(acc)


# In[ ]:


knn1 = KNeighborsClassifier(n_neighbors= 5 , weights = 'uniform', metric='euclidean')
knn1.fit(X_train, y_train)    
predicted = knn1.predict(X_test)
from sklearn.metrics import accuracy_score
acc = recall_score(y_test, predicted)
print(acc)


# In[ ]:


knn2 = KNeighborsClassifier(n_neighbors= 7, weights = 'uniform', metric='euclidean')
knn2.fit(X_train, y_train)    
predicted2 = knn2.predict(X_test)
from sklearn.metrics import accuracy_score
acc = recall_score(y_test, predicted2)
print(acc)


# 
# # KNN tuning using GridSearchCV

# In[ ]:


grid_params ={
    'n_neighbors':[3,5,7,9],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan','minkowski']
}
gs_results=GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1,cv=5,n_jobs=-1)
gs_results.fit(X_train,y_train)


# In[ ]:


gs_results.best_estimator_


# In[ ]:


gs_results.best_params_


# In[ ]:


gs_results.best_score_


# In[ ]:


knn_predict_train=gs_results.best_estimator_.predict(X_train)
knn_predict_test=gs_results.best_estimator_.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, knn_predict_train, y_test, knn_predict_test)


# # Choosing a K Value

# Let's go ahead and use the elbow method to pick a good K Value:

# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid()
plt.show()


# Observation : at k=3 we are getting less error.....and we can also get k=1 less error but it is overfitting case.So,better to avoid k=1.

# # Decision Tree Tuning using GridSearchCV()

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


#  make an array of depths to choose from, say 1 to 20

# In[ ]:


depths = np.arange(1, 21)
depths


# In[ ]:


#num_leafs = [1, 5, 10, 20, 50, 100]
#criterion = ['gini', 'entropy']


# In[ ]:


param_grid = { 'criterion':['gini','entropy'],'max_depth': depths}
dtree_model=DecisionTreeClassifier()


# In[ ]:


gs = GridSearchCV(estimator=dtree_model, param_grid=param_grid, cv=10)


# In[ ]:


gs = gs.fit(X_train, y_train)


# In[ ]:


gs.best_estimator_


# In[ ]:


gs.best_params_


# In[ ]:


dt_predict_train=gs.best_estimator_.predict(X_train)
dt_predict_test=gs.best_estimator_.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, dt_predict_train, y_test, dt_predict_test)


# We got max_depth=6 .So, use this max_depth value in DTClassifier 

# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, gs.best_estimator_,scoring='recall')
plt.show()


# In[ ]:


from sklearn import tree
dt1=tree.DecisionTreeClassifier(max_depth=6)
dt1.fit(X_train,y_train)


# In[ ]:


dt1.get_depth


# In[ ]:


dt1.get_params


# In[ ]:


#dt1.predict(X_test)
dt1_predict_train=dt1.predict(X_train)
dt1_predict_test=dt1.predict(X_test)


# In[ ]:


classifcation_report_train_test(y_train, dt1_predict_train, y_test, dt1_predict_test)


# In[ ]:





# In[ ]:


tree.plot_tree(dt1.fit(X_train, y_train)) 


# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:





# In[ ]:


dt1.feature_importances_


# In[ ]:


f_imp = pd.Series(dt1.feature_importances_, index = X_train.columns)


# In[ ]:


## Sort importances  
f_imp_order= f_imp.nlargest(n=10)
f_imp_order


# In[ ]:


## Plot Importance
get_ipython().run_line_magic('matplotlib', 'inline')
f_imp_order.plot(kind='barh')
plt.show()


# # Logistic Regression With GridSearchCV()

# l1,l2 : Create regularization penalty space

# C : Create regularization hyperparameter space

# In[ ]:


# Grid search cross validation
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:


logreg2=LogisticRegression(C=10,penalty="l2")
logreg2.fit(X_train,y_train)
print("score",logreg2.score(X_test,y_test))


# In[ ]:


reg_predict_train=logreg2.predict(X_train)
reg_predict_test=logreg2.predict(X_test)
classifcation_report_train_test(y_train, reg_predict_train, y_test, reg_predict_test)


# # Why Grid Search is not performed for Naive Bayes Classifier?

#    there is no any hyperparameter to tune

# # RF Classifier

# In[ ]:


# Create first pipeline for base without reducing features.

#pipe = Pipeline([('classifier' , RandomForestClassifier())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create param grid.
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [240,245,250],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [12,13,14],
    'criterion' :['gini', 'entropy']
}
# Create grid search object
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[ ]:


CV_rfc.best_estimator_


# In[ ]:


rf_predict_train=CV_rfc.best_estimator_.predict(X_train)
rf_predict_test=CV_rfc.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, rf_predict_train, y_test, rf_predict_test)


# In[ ]:


from mlxtend.plotting import plot_learning_curves
plot_learning_curves(X_train, y_train, X_test, y_test, CV_rfc.best_estimator_,scoring='recall')
plt.show()


# In[ ]:


CV_rfc.best_estimator_.feature_importances_


# In[ ]:


feat_importances = pd.Series(CV_rfc.best_estimator_.feature_importances_, index = X_train.columns)


# In[ ]:


feat_importances_ordered = feat_importances.nlargest(n=10)
feat_importances_ordered


# In[ ]:


## Plot Importance
get_ipython().run_line_magic('matplotlib', 'inline')
feat_importances_ordered.plot(kind='barh')
plt.show()


# In[ ]:





# In[ ]:


from sklearn.metrics import recall_score

scores = pd.DataFrame(columns=['Model','Train_Recall','Test_Recall'])

def get_metrics(train_actual,train_predicted,test_actual,test_predicted,model_description,dataframe):
    
    train_recall   = recall_score(train_actual,train_predicted)
    test_recall   = recall_score(test_actual,test_predicted)
    dataframe = dataframe.append(pd.Series([model_description,train_recall,
                                            test_recall],
                                           index=scores.columns ), ignore_index=True)
    return(dataframe)


# In[ ]:


scores = get_metrics(y_train,y_train_pred_dt_be,y_test,y_test_pred_dt_be,'DecisionTrees basic model',scores)
scores = get_metrics(y_train,y_train_pred_lrc_be,y_test,y_test_pred_lrc_be,'LogisticRegression basic model',scores)
scores = get_metrics(y_train, y_train_pred_svc_be,y_test, y_test_pred_svc_be,'SupportVectorMachines basic model',scores)
scores = get_metrics(y_train, y_train_pred_knn_be,y_test, y_test_pred_knn_be,'KNearestNeighbours basic model',scores)
scores = get_metrics(y_train, y_train_pred_nv_be,y_test, y_test_pred_nv_be,'naiibeys basic model',scores)
scores = get_metrics(y_train,dt_predict_train,y_test,dt_predict_test,'Decision Tree with GridSearchCV()',scores)
scores = get_metrics(y_train,reg_predict_train,y_test,reg_predict_test,'logistic regression with GridSearchCV()',scores)
scores = get_metrics(y_train,y_train_pred_svc_best2,y_test,y_test_pred_svc_best2,'SVC using GridSearchCV()',scores)
scores = get_metrics(y_train,knn_predict_train,y_test,knn_predict_test,'KNN using GridSearchCV(),Where k=5',scores)
scores = get_metrics(y_train,rf_predict_train,y_test,rf_predict_test,'random forest using GridSearchCV',scores)


# In[ ]:


scores


# In[ ]:


scores.insert(3, "Best Tuning Parametrs",['','','','','','{criterion: gini, max_depth: 6}', '{C: 10.0, penalty: l2}', '{C: 1, gamma: 0.7, kernel: rbf}', '{metric: euclidean, n_neighbors: 3, weights: distance}','max_depth=13,max_features=auto,n_estimators=240,criterion=entropy'], True)
scores
#df.insert(2, "Age", [21, 23, 24, 21], True)


# In[ ]:




