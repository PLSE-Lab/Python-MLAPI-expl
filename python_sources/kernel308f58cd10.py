#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# LOAN PREDICTION - MACHINE LEARNING 
# IMPORT NECESSARY LIBRARY  
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sa


# READING THE GIVEN DATA SET 

# In[ ]:


loandf=pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")

loandf.head(10)


# checking the data set value 

# In[ ]:



print(loandf.info())
print(loandf.describe())
loandf.head(5)


# In[ ]:


print(loandf.isnull().sum())


# NOW WE ARE GOING TO FILL THE NULL VALUE WITH MODE(FOR CATAGORICAL DATA ) AND MEAN (FOR NUMERIC DATA ) 

# In[ ]:



print(loandf['Gender'].value_counts())
print(loandf['Married'].value_counts())
print(loandf['Dependents'].value_counts())
print(loandf['Self_Employed'].value_counts())
print(loandf['Loan_Amount_Term'].value_counts())
print(loandf['Credit_History'].value_counts())


# In[ ]:


loandf.Gender=loandf.Gender.fillna("Male")
loandf.Married=loandf.Married.fillna("Yes")
loandf.Dependents=loandf.Dependents.fillna(0)
loandf.Self_Employed=loandf.Self_Employed.fillna("No")
loandf.Loan_Amount_Term=loandf.Loan_Amount_Term.fillna(360)
loandf.Credit_History=loandf.Credit_History.fillna(1)


# In[ ]:


m1=loandf['LoanAmount'].mean()
loandf.LoanAmount=loandf.LoanAmount.fillna(m1)


# Now we have to encode the catagorical value in to numerical value
#  1) splitting the dependented and in dependent value 
# 

# In[ ]:


loandf.replace(to_replace="3+", value=3, inplace=True, limit=None, regex=False)
loandf.head(10)


# encoding the catagorical data in to numerical values 

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l = LabelEncoder()
#for i in range(1,6):
loandf.iloc[ : ,1]=l.fit_transform(loandf.iloc[ : ,1])
loandf.iloc[ : ,2]=l.fit_transform(loandf.iloc[ : ,2])
loandf.iloc[ : ,4]=l.fit_transform(loandf.iloc[ : ,4])
loandf.iloc[ : ,5]=l.fit_transform(loandf.iloc[ : ,5])
for i in range(10,13):
  loandf.iloc[ : ,i]=l.fit_transform(loandf.iloc[ : ,i])


# In[ ]:


loandf.head(10)


# NOW DROP THE UNWANTED DATA (LOAN_ID) AND SPLIT THE X AND Y

# In[ ]:



loandf.drop(["Loan_ID"],axis=1,inplace=True)
list1=loandf.columns
n=len(list1)
x_tr=loandf[list1[0:n-1]]
y_tr=loandf[list1[n-1]]
Ans=list(y_tr)
print (Ans)


# In[ ]:


loandf.corr()


# NOW WE HAVE TO FIT THE CORRECT MODULE 
# 1) LOGISTIC REGRESSION 

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 0)
LR.fit(x_tr, y_tr)


# NOW TEST THE VALUE 

# In[ ]:


x_ts=pd.read_csv("../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")


# In[ ]:


x_ts


# In[ ]:


x_ts.info()
print(x_ts['Gender'].value_counts())
print(x_ts['Married'].value_counts())
print(x_ts['Dependents'].value_counts())
print(x_ts['Self_Employed'].value_counts())
print(x_ts['Loan_Amount_Term'].value_counts())
print(x_ts['Credit_History'].value_counts())


# In[ ]:


x_ts.Gender=x_ts.Gender.fillna("Male")
x_ts.Dependents=x_ts.Dependents.fillna(0)
x_ts.Self_Employed=x_ts.Self_Employed.fillna("No")
x_ts.Loan_Amount_Term=x_ts.Loan_Amount_Term.fillna(360)
x_ts.Credit_History=x_ts.Credit_History.fillna(1)
x_ts.LoanAmount=x_ts.LoanAmount.fillna(x_ts['LoanAmount'].mean())
x_ts.replace(to_replace="3+", value=3, inplace=True, limit=None, regex=False)
x_ts.columns


# In[ ]:


x_ts.drop(["Loan_ID"],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l = LabelEncoder()
for i in range(0,2):
  x_ts.iloc[ : ,i]=l.fit_transform(x_ts.iloc[ : ,i])
for i in range(3,6):
  x_ts.iloc[ : ,i]=l.fit_transform(x_ts.iloc[ : ,i])
for i in range(9,11):
  x_ts.iloc[ : ,i]=l.fit_transform(x_ts.iloc[ : ,i])


# In[ ]:


x_ts


# NOW CHECK THE VALUE FOR OUR TEST DATA  using LOGISTIC regression 

# In[ ]:


pred=LR.predict(x_ts)
pred


# In[ ]:


y_pre=LR.predict(x_tr)
y_pre


# CHECK THE ACCURACY OF LOGISTIC REGRESSION 

# In[ ]:


from sklearn import metrics
print('The score of Logistic Regression is: ', metrics.accuracy_score(y_pre, y_tr))


# USING KNN ALGORITHM 
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN.fit(x_tr, y_tr)


# In[ ]:


y_tr1=KNN.predict(x_tr)
y_tr1


# In[ ]:


print('The score of KNN is: ', metrics.accuracy_score(y_tr1, y_tr))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_tr1, y_tr)
cm1


# SVM ALGORITHM
# 

# In[ ]:


from sklearn.svm import SVC
SVM= SVC(kernel = 'linear', random_state = 0)
SVM.fit(x_tr, y_tr)


# In[ ]:


y_tr2=SVM.predict(x_tr)
y_tr2


# In[ ]:


print("The score of SVM is :",metrics.accuracy_score(y_tr2, y_tr))


# DECISION TREE ALGORITHM 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DTREE = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTMOD=DTREE.fit(x_tr, y_tr)


# In[ ]:


pred3=DTREE.predict(x_ts)
pred3


# In[ ]:


y_tr3=DTREE.predict(x_tr)
Ans1=list(y_tr3)
print(Ans1)


# In[ ]:


print("The score of DECISSION TREE:",metrics.accuracy_score(y_tr3, y_tr))


# so decission tree gives the max score (1) so we can go with decission tree for this problem 
# now check the value for accuracy 

# In[ ]:


count=0
missvalue=0
for i in range(len(Ans)):
    if Ans[i]==Ans1[i] :
        count+=1
    else :
        missvalue+=1
print("The count of correctly predicted value by decision tree out of "+str(len(Ans))+"is:",count )


# so the decision tree gives the 100% correct value 

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_tr,y_tr3))


# NOW GET THE ANS FOR TEST DATA 

# In[ ]:


y_ans_for_test_data=DTREE.predict(x_ts)
print(y_ans_for_test_data)


# NOW CHECK THE CORRELATION AND DATA VISUALIZATION 
# before making decision tree we have to remove index of x_tr
# 

# In[ ]:


from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_tr, y_tr)
tree.plot_tree(clf.fit(x_tr,y_tr)) 


# NOW CHECK THE IMPORTANCES OF FEATURE 

# In[ ]:


list2=(DTMOD.feature_importances_)
pd.DataFrame(list2,[x_tr.columns])


# The above feature shows that Credit_History, Applicantincome and Loan amount decides the most importent decision  

# In[ ]:


DecisionTreeClassifier(class_weight=0.2, criterion='entropy', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=5, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort=False,
                      random_state=0, splitter='best')
DTMOD.max_features
DTMOD.get_n_leaves


# In[ ]:


plt.scatter(x_tr['LoanAmount'],x_tr["ApplicantIncome"])


# ABOVE PLOT SHOWS THAT APPLICANT INCOME LESSTHAN 20000 SEEK for loan less than 300

# In[ ]:



sa.catplot(x="Married", y="LoanAmount", data=x_tr)


# from the above graph we can say MARRIED persons want more loan than unmarried person 

# In[ ]:


sa.catplot(x="Education", y="LoanAmount", data=x_tr)


# graduate people want more loan 

# In[ ]:


sa.catplot(x="LoanAmount", y="Loan_Status", data=loandf)


# THERE IS NO CORR BET LOAN AMOUNT AND LOAN STATUS 

# In[ ]:


sa.catplot(x="Loan_Status", y="ApplicantIncome", data=loandf)


# APPLICATION INCOME not decide loan status 

# In[ ]:


sa.catplot(x="Loan_Status", y="LoanAmount", data=loandf)


# In[ ]:


sa.catplot(data=loandf, orient="Dependents", kind="box")


# THANKS FOR WATHING 

# In[ ]:




