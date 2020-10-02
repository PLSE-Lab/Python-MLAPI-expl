#!/usr/bin/env python
# coding: utf-8

# <b>Problem Statement:</b>
#     
#     The main goal of this problem is to predict the 'Chance of Admit' of a student in a 
#     perticular university given various parameters such as:
# *         GRE Scores(out of 340)
# *         TOEFL Scores(out of 120)
# *         University Rating(out of 5)
# *         Statement of Purpose and Letter of Recommendation Strength(out of 5)
# *         Undergraduate GPA(out of 10)
# *         Research Experience(either 0 or 1)
#     

# <b>Algorithms Considered:</b>
# *      Linear Regression
# *      Logistic Regression
# *      Support Vector Machine
# *      K Nearest Neighbours
# *      Decision Tree Regressor

# <b>Goal:</b>
# *      The main aim of this kernel is to predict the 'Chance of Admit' with high accuracy by applying various ML Algorithms 
#         and then comparing their scores..
# *      Compare different models to check for best model depending on r_squared score and accuracy score

# <b>----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# <b>Importing Python Packages:</b>
# *      Pandas
# *      Numpy
# *      Matplotlib
# *      Seaborn
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# <b>Reading Data as CSV File:</b>

# In[ ]:


df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# <b>Making basic Insights about given data:</b>

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# <b>Checking for any Linear Relationship between the given Parameters:</b>
# *      By constructing Pairplot
# *      By constructing Correlation heatmap 

# In[ ]:


sns.pairplot(df)


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True)


# <b>From above the Parameters with High Corelation against 'Chance of Admit' are:</b>
# *      GRE Score
# *      TOEFL Score
# *      CGPA

# <b>----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# <b>Distribution of Parameters:</b>

# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
sns.distplot(df['GRE Score'], color='Orange')
plt.grid(alpha=0.5)

plt.subplot(2,2,2)
sns.distplot(df['TOEFL Score'], color='Orange')
plt.grid(alpha=0.5)

plt.subplot(2,2,3)
sns.distplot(df['University Rating'], color='Orange')
plt.grid(alpha=0.5)

plt.subplot(2,2,4)
sns.distplot(df['CGPA'], color='Orange')
plt.grid(alpha=0.5)


# <b>-----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# <b>Analysis on Research Column:</b>

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['Research'])
plt.grid(alpha=0.5)
plt.xlabel('Students')
plt.show()


# In[ ]:


print("Total number of students with Research : ",(df['Research']==1).sum())
print("Total number of students with-out Research : ",len(df)-(df['Research']==1).sum())
print("Percentage of students with Research : ",round(((df['Research']==1).sum()/len(df))*100,2),'%')


# <b>Analysis on University Ranking:</b> 

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['University Rating'])
plt.grid(alpha=0.5)
plt.xlabel('Students Count')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
uni_influence = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts()
uni_influence.plot(kind='barh')
plt.grid(alpha=0.5)
plt.xlabel('Student Count')
plt.ylabel('University Rating')
plt.show()


# In[ ]:


print('From given University Rating each university has a Student count of:')
print('University Rating 1 : ',(df['University Rating']==1).sum())
print('University Rating 2 : ',(df['University Rating']==2).sum())
print('University Rating 3 : ',(df['University Rating']==3).sum())
print('University Rating 4 : ',(df['University Rating']==4).sum())
print('University Rating 5 : ',(df['University Rating']==5).sum())


# In[ ]:


print('From given University Rating and Student count in each university, number of Students having chance >75% of Admit:')
print('University Rating 1 : ',uni_influence.iloc[4])
print('University Rating 2 : ',uni_influence.iloc[3])
print('University Rating 3 : ',uni_influence.iloc[2])
print('University Rating 4 : ',uni_influence.iloc[0])
print('University Rating 5 : ',uni_influence.iloc[1])


# <b>------------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# <b>Mean and Standard Deviation of 'GRE Score', 'TOEFL Score', 'CGPA' has been accounted to compare students easily as the 
# students with positive standard deviation tend to perform well as compared to majority of people i.e scores better than
# that of mean value</b>

# In[ ]:


gre_avg = df['GRE Score'].mean()
gre_std = df['GRE Score'].std()
print("Maximum GRE Score : 340")
print("Average GRE Score : ",gre_avg)
print("Standard Deaviation : ",gre_std)

diff = df['GRE Score']-gre_avg
df['SD_GRE'] = diff/gre_std


# In[ ]:


toefl_avg = df['TOEFL Score'].mean()
toefl_std = df['TOEFL Score'].std()
print("Maximum TOEFL Score : 120")
print("Average TOEFL Score : ",toefl_avg)
print("Standard Deaviation : ",toefl_std)

diff = df['TOEFL Score']-toefl_avg
df['SD_TOEFL'] = diff/toefl_std


# In[ ]:


cgpa_avg = df['CGPA'].mean()
cgpa_std = df['CGPA'].std()
print("Maximum CGPA Score : 10")
print("Average CGPA Score : ",cgpa_avg)
print("Standard Deaviation : ",cgpa_std)

diff = df['CGPA']-cgpa_avg
df['SD_CGPA'] = diff/cgpa_std


# In[ ]:


df.head()


# <b>Constructing Pairplot for new parameters against 'Chance of Admit':</b> 

# In[ ]:


sns.pairplot(df, x_vars=['GRE Score','TOEFL Score','CGPA','SD_GRE','SD_TOEFL','SD_CGPA'], y_vars='Chance of Admit')


# <b>Constructing Heatmap of Corelation for new parameters against 'Chance of Admit':</b> 

# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(), annot=True)


# <b>------------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Splitting Data For Training & Testing

# In[ ]:


x = df.drop(['Chance of Admit'], axis=1)
y = df['Chance of Admit']


# In[ ]:


x.info()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)


# <b>----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Applying Linear Regression Model:

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


lr.fit(x_train, y_train)


# In[ ]:


coef = pd.DataFrame(lr.coef_, x_test.columns, columns = ['Co-efficient'])


# <b>The Co-efficients for the following parameters are:</b>

# In[ ]:


coef


# <b>From above we can infer that :</b>
# - If GRE Score increases by 1 then Chance of Admit will be affected by 0.002092
# - If TOEFL increases by 1 then Chance of Admit will be affected by 0.003529
# - If University Rating increases by 1 then Chance of Admit will be affected by 0.008793
#   <br>and so on...</br>

# In[ ]:


y_pred_mlr = lr.predict(x_test)


# In[ ]:


len(x_test)


# <b>Plotting Actual vs Predicted Values:</b>

# In[ ]:


fig = plt.figure()
c = [i for i in range(1,101,1)]
plt.plot(c,y_test, color = 'green', linewidth = 2.5, label='Test')
plt.plot(c,y_pred_mlr, color = 'orange', linewidth = 2.5, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
fig.suptitle('Actual vs Predicted')


# <b>Calculating Error Terms:</b>

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred_mlr)
r_square_score = r2_score(y_test, y_pred_mlr)


# In[ ]:


print('Mean Square Error = ',mse)
print('R_Square Score = ',r_square_score)


# In[ ]:


fig = plt.figure()
plt.plot(c,y_test-y_pred_mlr, color = 'orange', linewidth = 2.5)
plt.grid(alpha = 0.3)
fig.suptitle('Error Terms')


# <b>-----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Statistical Information using Statsmodels:

# In[ ]:


import statsmodels.api as sm


# In[ ]:


x_train_sm = x_train
x_train_sm = sm.add_constant(x_train_sm)
lml = sm.OLS(y_train, x_train_sm).fit()
lml.params


# In[ ]:


print(lml.summary())


# <b>Re-Valuating the Data:</b>

# If 'p > 0.05' for a 95% level of confidence:
# - Ho : Value is not significant
# - H1 : Value is significant
# Since in GRE p(0.065) > 0.05 so 'we fail to reject Ho' 

# In[ ]:


x_new = df.drop(['Serial No.','University Rating','SOP','Chance of Admit'], axis=1)
y_new = df['Chance of Admit']


# In[ ]:


x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new,y_new, train_size = 0.7, random_state = 100)


# In[ ]:


lr.fit(x_train_new, y_train_new)


# In[ ]:


y_pred_new = lr.predict(x_test_new)


# In[ ]:


len(x_test_new)


# In[ ]:


# Actual vs Predicted after removing GRE
fig = plt.figure()
c = [i for i in range(1,121,1)]
plt.plot(c,y_test_new, color = 'green', linewidth = 2.5, label='Test')
plt.plot(c,y_pred_new, color = 'orange', linewidth = 2.5, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
fig.suptitle('Actual vs Predicted')


# In[ ]:


mse_new = mean_squared_error(y_test_new, y_pred_new)
r_square_score_new = r2_score(y_test_new, y_pred_new)
print('Mean Square Error = ',mse_new)
print('R_Square Score = ',r_square_score_new)


# In[ ]:


fig = plt.figure()
plt.plot(c,y_test_new-y_pred_new, color = 'orange', linewidth = 2.5)
plt.grid(alpha = 0.3)
fig.suptitle('Error Terms')


# <b>-----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Applying Logistic Regression Model:

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# Since Logistic Regression is a Classification model, 'Continuous Data' will not help to classify the output. Hence we need
# to categorize the data into:
#     - Label 1 for Chance of Admit greater or equal to 80%
#     - Label 0 for Chance of Admit lesser than 80%

# In[ ]:


y_train_label = [1 if each > 0.8 else 0 for each in y_train]
y_test_label  = [1 if each > 0.8 else 0 for each in y_test]


# In[ ]:


logmodel.fit(x_train, y_train_label)


# In[ ]:


y_pred_log = logmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_label, y_pred_log))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test_label, y_pred_log)


# In[ ]:


sns.heatmap(cm_log, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_log))
print("precision_score: ", precision_score(y_test_label,logmodel.predict(x_test)))
print("recall_score: ", recall_score(y_test_label,logmodel.predict(x_test)))
print("f1_score: ",f1_score(y_test_label,logmodel.predict(x_test)))


# <b>------------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Applying Support Vector Machine Model: 

# In[ ]:


from sklearn.svm import SVC
svmmodel = SVC()


# In[ ]:


svmmodel.fit(x_train,y_train_label)


# In[ ]:


y_pred_svm = svmmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test_label, svmmodel.predict(x_test))


# In[ ]:


sns.heatmap(cm_svm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_svm))
print("precision_score: ", precision_score(y_test_label,svmmodel.predict(x_test)))
print("recall_score: ", recall_score(y_test_label,svmmodel.predict(x_test)))
print("f1_score: ",f1_score(y_test_label,svmmodel.predict(x_test)))


# <b>------------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Applying Decision Tree Regressor Model:

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train,y_train)


# In[ ]:


y_pred_dt = dt_model.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score
print('R_Squared Score = ',r2_score(y_test, y_pred_dt))


# <b>-------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Applying KNN Model:

# In[ ]:


from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


import math
math.sqrt(len(y_test_label))


# In[ ]:


knnc = KNeighborsClassifier(n_neighbors = 11, p=2, metric = 'euclidean')


# In[ ]:


knnc.fit(x_train, y_train_label)


# In[ ]:


y_pred_knn = knnc.predict(x_test)


# In[ ]:


y_pred_knn


# In[ ]:


cm = confusion_matrix(y_test_label, y_pred_knn)
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")


# In[ ]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_knn))
print("precision_score: ", precision_score(y_test_label,knnc.predict(x_test)))
print("recall_score: ", recall_score(y_test_label,knnc.predict(x_test)))
print("f1_score: ",f1_score(y_test_label,knnc.predict(x_test)))


# <b>--------------------------------------------------------------------------------------------------------------------------------------------------------------------</b>

# # Comparision Between Models:

# <b>Comparing Regression Models:</b>

# In[ ]:


x = ["Linear_Reg","Decision_Tree_Reg"]
y = np.array([r2_score(y_test,y_pred_mlr),r2_score(y_test,y_pred_dt)])
plt.barh(x,y, color='#225b46')
plt.xlabel("R_Squared_Score")
plt.ylabel("Regression Models")
plt.title("Best R_Squared Score")
plt.grid(alpha=0.5)
plt.show()


# <b>Comparing Regression Models:</b>

# In[ ]:


x = ["KNN","SVM","Logistic_Reg"]
y = np.array([accuracy_score(y_test_label, y_pred_knn),accuracy_score(y_test_label, y_pred_svm),accuracy_score(y_test_label, y_pred_log)])
plt.barh(x,y, color='#225b46')
plt.xlabel("Accuracy Score")
plt.ylabel("Classification Models")
plt.title("Best Accuracy Score")
plt.grid(alpha=0.5)
plt.show()


# # Conclusion:

# <b>By analyzing the data and by applying ML model:</b>
#     - In Classification
#         - Logistic Regression was better with accuracy score of 97.0%
#         - K Nearest Neighbour was better with accuracy score of 84.0%
#         - Support Vector Machine was better with accuracy score of 70.0%
#         
#     - In Regression
#         - Linear Regression was better with r_squared score of 82.14%
#         - Decision Tree Regressor was better with r_squared score of 65.81%

# In[ ]:




