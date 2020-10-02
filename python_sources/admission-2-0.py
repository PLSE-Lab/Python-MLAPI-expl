#!/usr/bin/env python
# coding: utf-8

# Step 1: **Import library**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score


# Step 2 :**Import the dataset**

# In[ ]:


Admission = pd.read_csv("../input/Admission_Predict.csv",sep = ",")


# Step 3: **Explore the data**

# In[ ]:


Admission.head()


# In[ ]:


Admission.describe()


# About the data :
# 
# 1. There are no missing value in the data
# 2. There are 400 records in the dataset

# Step 4: **Remove the unwanted columns**

# In[ ]:


Admission.drop(['Serial No.'],axis=1,inplace=True)


# In[ ]:


Admission.columns


# Step 5: **Building the model**

# In[ ]:


#check the correlation
plt.figure(figsize=(10,5))
sns.heatmap(Admission.corr(),annot=True)


# **About the Heat map **
# 
# 1. CGPA ,university rating  and GRE score,TOEFL score  are most important parameters for admission
# 2. The 3 least important features for admission  are  Research, LOR, and SOP

# In[ ]:


#Divide the dataset into source and target for prediction
Admission=Admission.rename(columns = {'Chance of Admit ':'Chance of Admit'})

# normalization
y = Admission["Chance of Admit"].values
x_data= Admission.drop(["Chance of Admit"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# Train and Test split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

#making target as 0 or 1  for logistic calculation

y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01  = [1 if each > 0.8 else 0 for each in y_test]

# list to array
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)


# In[ ]:


#check the new variables shape
print("Train_x shape:",x_train.shape)
print("Test_x shape:",x_test.shape)
print("Train_y shape:",y_train.shape)
print("Test_y shape:",y_test.shape)


# Step 6:**Different model **

# 1. **Linear Regression Model**
# 

# In[ ]:


Lr=LinearRegression()
Lr.fit(x_train,y_train)
y_head_Lr=Lr.predict(x_test)

print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(Lr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(Lr.predict(x_test.iloc[[2],:])))

print("r_square score: ", r2_score(y_test,y_head_Lr))
y_head_Lr_train = Lr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_Lr_train))


# 2. **Logistic Regression Model**

# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train_01,x_train)
result=logit_model.fit()
print(result.summary2())


# The p-values for most of the variables are smaller than 0.05 except SOP  so we remove it

# In[ ]:


Admission.drop(['SOP'],axis=1,inplace=True)


# In[ ]:


#Divide the dataset into source and target for prediction
Admission=Admission.rename(columns = {'Chance of Admit ':'Chance of Admit'})

# normalization
y = Admission["Chance of Admit"].values
x_data= Admission.drop(["Chance of Admit"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# Train and Test split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

#making target as 0 or 1  for logistic calculation

y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01  = [1 if each > 0.8 else 0 for each in y_test]

# list to array
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)


# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train_01,x_train)
result=logit_model.fit()
print(result.summary2())


# In[ ]:


lrc = LogisticRegression()
lrc.fit(x_train,y_train_01)
print("score: ", lrc.score(x_test,y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(lrc.predict(x_test.iloc[[1],:])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(lrc.predict(x_test.iloc[[2],:])))


# In[ ]:


cm_lrc = confusion_matrix(y_test_01,lrc.predict(x_test))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_lrc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

from sklearn.metrics import precision_score, recall_score
print("precision_score: ", precision_score(y_test_01,lrc.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,lrc.predict(x_test)))

from sklearn.metrics import f1_score
print("f1_score: ",f1_score(y_test_01,lrc.predict(x_test)))


# Confusion Matrix:
# 
# Predicted 1: 22
# Predicted 0: 7
# According to Confusion Matrix, the model predicted that 23 candidate's Chances of Admit are greater than 80%. In reality, 22 of them have a Chance of Admit greater than 80%. In total, 29 candidate's Chances of Admit are greater than 80%.
# 
# 
# Predicted 1: 1
# Predicted 0:50
# 
# According to Confusion Matrix, the model predicted that 57 candidate's Chances of Admit are less than or equal to 80%. In reality, 50 of them have a Chance of Admit less than or equal to 80%. In total, 51 candidate's Chances of Admit are less than or equal to 80%.

# 3. **Random Forest Regression model**

# In[ ]:


rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train,y_train)
y_head_rfr = rfr.predict(x_test)

print("r_square score: ", r2_score(y_test,y_head_rfr))
print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(rfr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(rfr.predict(x_test.iloc[[2],:])))


y_head_rf_train = rfr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_rf_train))


# **3. Support vector machine prediction**

# In[ ]:


svm = SVC(random_state = 1)
svm.fit(x_train,y_train_01)
print("score: ", svm.score(x_test,y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(svm.predict(x_test.iloc[[1],:])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(svm.predict(x_test.iloc[[2],:])))


# In[ ]:


# confusion matrix
cm_svm = confusion_matrix(y_test_01,svm.predict(x_test))

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_svm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

print("precision_score: ", precision_score(y_test_01,svm.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,svm.predict(x_test)))

from sklearn.metrics import f1_score
print("f1_score: ",f1_score(y_test_01,svm.predict(x_test)))


# 4. **Decision Tree Regression prediction**

# In[ ]:


dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train,y_train)
y_head_dtr = dtr.predict(x_test)

print("r_square score: ", r2_score(y_test,y_head_dtr))
print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[2],:])))

y_head_dtr_train = dtr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_dtr_train))


# Gaussian Naive Bayes Prediction

# In[ ]:


nb = GaussianNB()
nb.fit(x_train,y_train_01)
print("score: ", nb.score(x_test,y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(nb.predict(x_test.iloc[[1],:])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(nb.predict(x_test.iloc[[2],:])))


# In[ ]:


cm_nb = confusion_matrix(y_test_01,nb.predict(x_test))

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_nb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

print("precision_score: ", precision_score(y_test_01,nb.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,nb.predict(x_test)))

from sklearn.metrics import f1_score
print("f1_score: ",f1_score(y_test_01,nb.predict(x_test)))


# Step 7: **Comparison of Regression Algorithms**

# In[ ]:


y = np.array([r2_score(y_test,y_head_Lr),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])
x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()


# Linear regression and random forest regression algorithms were better than decision tree regression algorithm.

# In[ ]:


y = np.array([lrc.score(x_test,y_test_01),svm.score(x_test,y_test_01),nb.score(x_test,y_test_01)])
#x = ["LogisticRegression","SVM","GaussianNB","DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier"]
x = ["LogisticReg.","SVM","GNB"]

plt.bar(x,y)
plt.title("Comparison of Classification Algorithms")
plt.xlabel("Classfication")
plt.ylabel("Score")
plt.show()


# All classification algorithms achieved around 90% success. The most successful one is Gaussian Naive Bayes with 96% score.

# **Ensemble Model **

# **Voting**
# 
# Hard voting is where a model is selected from an ensemble to make the final prediction by a simple majority vote for accuracy.
# 
# Soft Voting can only be done when all your classifiers can calculate probabilities for the outcomes. Soft voting arrives at the best result by averaging out the probabilities calculated by individual algorithms.

# In[ ]:


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc',svm_clf)],voting = 'hard')
voting_clf.fit(x_train, y_train_01)


# In[ ]:


print("score: ", voting_clf.score(x_test,y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(voting_clf.predict(x_test.iloc[[1],:])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(voting_clf.predict(x_test.iloc[[2],:])))


# In[ ]:


cm_vot = confusion_matrix(y_test_01,voting_clf.predict(x_test))


# In[ ]:


f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_vot,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

print("precision_score: ", precision_score(y_test_01,voting_clf.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,voting_clf.predict(x_test)))

from sklearn.metrics import f1_score
print("f1_score: ",f1_score(y_test_01,voting_clf.predict(x_test)))


# In[ ]:


y = np.array([lrc.score(x_test,y_test_01),svm.score(x_test,y_test_01),nb.score(x_test,y_test_01),
             voting_clf.score(x_test,y_test_01)])
#x = ["LogisticRegression","SVM","GaussianNB","DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier"]
x = ["LogisticReg.","SVM","GNB","Ensemble"]

plt.bar(x,y)
plt.title("Comparison with ensemblem voting")
plt.xlabel("Classfication")
plt.ylabel("Score")
plt.show()


# In[ ]:




