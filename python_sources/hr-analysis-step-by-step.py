#!/usr/bin/env python
# coding: utf-8

# <b>Importing Necessary Libraries</b>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[ ]:


# Loading the data
hr=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
hr.head()


# In[ ]:


hr.shape


# <b>Observation:</b>
# There are 1470 rows and 35 columns in the dataset.

# In[ ]:


hr.dtypes


# In[ ]:


# Removing unnecessary columns in the dataset
import warnings
warnings.filterwarnings("ignore")
hr.drop(["EmployeeNumber","Over18","EmployeeCount","StandardHours"],axis=1,inplace=True)


# In[ ]:


hr.shape


# In[ ]:


numerical_data= hr.select_dtypes(include=["int64"])
numerical_data.shape


# There are 23 attributes which are of int datatype.

# In[ ]:


categorical_data= hr.select_dtypes(include=["O"])
categorical_data.shape


# There are 8 attributes which are of object datatype.

# <b>Checking Missing Values</b>

# In[ ]:


hr.isnull().sum()


# <b>Observation:</b>
# There are no missing values in the dataset.

# # Exploratory Data Analysis

# <b>Univariate Analysis</b>

# In[ ]:


sns.countplot(hr["Attrition"])


# From the above visualization, we can see that target variable is imbalanced.

# In[ ]:


sns.countplot(hr["BusinessTravel"],palette="Set2")


# <b>Observation:</b>
# Most of the Employees Travel Rarely.

# In[ ]:


sns.countplot(hr["Department"])


# There are three departments i.e. Sales, Research & Development, Human Resources. Out of which there are more employees who is in Research & Development.

# In[ ]:


sns.countplot(hr["Gender"])


# There are more number of employees who are Male.

# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(hr["JobRole"])


# <b>Observation:</b>
# There are more employee who works as a Sales Executive.

# In[ ]:


sns.countplot(hr["MaritalStatus"])


# <b>Observation:</b>
# Most of the employees who are working are Married.

# In[ ]:


plt.figure(figsize=(8,4))
plt.title("Distribution of age")
a = sns.distplot(hr["Age"], color = 'orange')


# <b>Observation:</b>
# The average Age of employees is 35.

# In[ ]:


plt.figure(figsize=(8,4))
plt.title("Distribution of Daily Rate ")
b= sns.distplot(hr["DailyRate"], color = 'green')


# <b>Observation:</b>
# The average DailyRate is ~750.

# In[ ]:


plt.figure(figsize=(8,4))
plt.title("Distribution of Hourly Rate ")
c= sns.distplot(hr["HourlyRate"], color = 'lime')


# <b>Observation:</b>
# The Hourly Rate of employees are ~ 70.

# In[ ]:


plt.figure(figsize=(8,4))
plt.title("Distribution of Percentage Salary Hike ")
d= sns.distplot(hr["PercentSalaryHike"], color = 'green')


# <b>Observation:</b>
# The average percentage salary hike of employees are 12.5

# In[ ]:


plt.figure(figsize=(8,4))
plt.title("Distribution of Years at Company ")
e= sns.distplot(hr["YearsAtCompany"], color = 'springgreen')


# <b>Observation:</b>
# The average years of employees at company is 5 years.

# <b>Bi-Variate Analysis</b>

# In[ ]:


sns.countplot(x="BusinessTravel", hue="Attrition",data=hr,palette="Set2")
plt.show()


# <b>Observation:</b>
# From the above Visualization, it is clear that employees who travel rarely have high attrition.

# In[ ]:


sns.countplot(x="Department", hue="Attrition",data=hr,palette="Set1")
plt.show()


# <b>Observation:</b>
# From the above visualization, it is clear that employees who are in Research & Development have high attrition.

# In[ ]:


sns.countplot(x="Gender", hue="Attrition",data=hr,palette="Set3")
plt.show()


# <b>Observation:</b>
# From the above visualization, Males have high attrition.

# In[ ]:


sns.countplot(x="MaritalStatus", hue="Attrition",data=hr)
plt.show()


# <b>Observation:</b>
# Employees who are single have high attrition.

# In[ ]:


sns.countplot(x="OverTime", hue="Attrition",data=hr,palette="Set1")
plt.show()


# <b>Observation:</b>
# Employees who do over time have high attrition.

# In[ ]:


sns.barplot(x="YearsAtCompany", y="Attrition",data=hr,palette="Set2")
plt.show()


# <b>Observation:</b>
# Employees who are at the company and have experience <=5 have high attrition.

# In[ ]:


sns.countplot(x="YearsSinceLastPromotion", hue="Attrition",data=hr,palette="Set1")
plt.show()


# <b>Observation:</b>
# Employees who have less experience since last promotion have high attrition.

# <b> Multi-Variate Analysis</b>

# In[ ]:


hr.hist(figsize=(18,18),grid=True,bins='auto');


# In[ ]:


hr.corr()


# <b>Checking Skewness</b>

# In[ ]:


hr.skew(axis=0)


# In[ ]:


# Treating the skewness in the dataset
for index in hr.skew().index:
    if hr.skew().loc[index]>0.5:
        hr[index]=np.log1p(hr[index])


# In[ ]:


hr.skew(axis=0)


# <b>Label Encoder</b>

# In[ ]:


# Lets convert the target variable
from sklearn.preprocessing import LabelEncoder
LE= LabelEncoder()
hr["Attrition"]=LE.fit_transform(hr["Attrition"])
hr["BusinessTravel"]=LE.fit_transform(hr["BusinessTravel"])
hr["Department"]=LE.fit_transform(hr["Department"])
hr["EducationField"]=LE.fit_transform(hr["EducationField"])
hr["Gender"]=LE.fit_transform(hr["Gender"])
hr["JobRole"]=LE.fit_transform(hr["JobRole"])
hr["MaritalStatus"]=LE.fit_transform(hr["MaritalStatus"])
hr["OverTime"]=LE.fit_transform(hr["OverTime"])


# <b>Checking Outliers</b>

# In[ ]:


from scipy.stats import zscore
z_score=abs(zscore(hr))
print("The shape of dataset before removing outliers",hr.shape)
hr=hr.loc[(z_score<3).all(axis=1)]
print("The shape of dataset after removing outliers",hr.shape)


# <b>Dividing the input and output variables</b>

# In[ ]:


X= hr.drop(["Attrition"],axis=1)
y= hr["Attrition"]


# In[ ]:


# Lets bring the dataset features into same scale
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X= scaler.fit_transform(X)


# <b>Splitting into training and testing</b>

# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.30)


# In[ ]:


# We will use auc_roc score as the metrics because target variable has imbalance dataset
def max_auc_roc_sc(w,X,y):
    max_auc_roc_sc=0
    for r_state in range(42,100):
        X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.30, random_state=r_state,stratify=y)
        w.fit(X_train,y_train)
        y_pred= w.predict(X_test)
        auc_roc=roc_auc_score(y_test,y_pred)
        if auc_roc>max_auc_roc_sc:
            max_auc_roc_sc=auc_roc
            a_score=r_state
    print("Maximum AUC_ROC Score corresponding to:",a_score," and it is :",round((max_auc_roc_sc),3))


# # Machine Learning Models

# As Target variable(Attrition) is binary, its classification problem, we will use KNN, Decision Tree Classifier, Gradient Boosting Classifier and Random Forest Classifier.

# <b>KNN Classifier</b>

# In[ ]:


knn= KNeighborsClassifier()
neighbors={"n_neighbors":range(1,30)}
knn= GridSearchCV(knn, neighbors, cv=5,scoring="roc_auc")
knn.fit(X,y)
knn.best_params_


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=29)
max_auc_roc_sc(knn,X,y)


# In[ ]:


pred_knn= knn.predict(X_test)
m1= knn.score(X_test, y_test)
print("The accuracy of the KNN Model is:",round((m1),3))
print(confusion_matrix(y_test,pred_knn))


# <b>Observations:</b>
# <li> There are 357 observations which are predicted Positive as TP(True Positive) and it is true.</li>
# <li> There are 1 observations which are predicted Negative as TN(True Negative) and it is True.</li>
# <li> There are 0 observations which are predicted Negative as FN(False Negative) and it is False. </li>
# <li> There are 64 observation which are predicted Positive as FP(False Positive) and it is False.</li>

# In[ ]:


print(classification_report(y_test,pred_knn))


# In[ ]:


from sklearn.model_selection import cross_val_score
mean_knn_auc=cross_val_score(knn, X,y,cv=5,scoring="roc_auc").mean()
print("Mean AUC_ROC Score after cross validation", cross_val_score(knn, X,y,cv=5,scoring="roc_auc").mean())
st_knn_auc= cross_val_score(knn, X,y,cv=5,scoring="roc_auc").std()
print("standard deviation for KNN from mean AUC_ROC score is",cross_val_score(knn, X,y,cv=5,scoring="roc_auc").std())


# In[ ]:


y_pred_prob= knn.predict_proba(X_test)[:,0]
tpr,fpr, thresholds= roc_curve(y_test, y_pred_prob)


# Plot
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="KNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN")
plt.show()


# In[ ]:


a_c1=roc_auc_score(y_test, knn.predict(X_test))
a_c1


# <b>Decision Tree Classifier</b>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy',max_depth=50)

max_auc_roc_sc(dtc,X,y)


# In[ ]:


pred_dtc= dtc.predict(X_test)
dtc1= dtc.score(X_test, y_test)
print("The accuracy of the Decision Tree Model is:",round((dtc1),3))
print(confusion_matrix(y_test,pred_dtc))


# <b>Observations:</b>
# <li> There are 343 observations which are predicted Positive as TP(True Positive) and it is true.</li>
# <li> There are 52 observations which are predicted Negative as TN(True Negative) and it is True.</li>
# <li> There are 14 observations which are predicted Negative as FN(False Negative) and it is False. </li>
# <li> There are 13 observation which are predicted Positive as FP(False Positive) and it is False.</li>

# In[ ]:


print(classification_report(y_test,pred_dtc))


# In[ ]:


from sklearn.model_selection import cross_val_score
mean_dtc_auc=cross_val_score(dtc, X,y,cv=5,scoring="roc_auc").mean()
print("Mean AUC_ROC Score Score after cross validation", cross_val_score(dtc, X,y,cv=5,scoring="roc_auc").mean())
s_dtc_auc= cross_val_score(dtc, X,y,cv=5,scoring="roc_auc").std()
print("standard deviation for Decision Tree Classifier from mean AUC_ROC score is",cross_val_score(dtc, X,y,cv=5,scoring="roc_auc").std())


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_pred_prob= dtc.predict_proba(X_test)[:,0]
tpr,fpr, thresholds= roc_curve(y_test, y_pred_prob)


# Plot
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree")
plt.show()


# In[ ]:


a_c2=roc_auc_score(y_test, dtc.predict(X_test))
a_c2


# <b>Gradient Boosting Classifier</b>

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
parameters= {'learning_rate': [0.01,0.03,0.05], 'subsample': [0.1, 0.5,0.3], 'n_estimators': [10,50,100], 'max_depth': [2,4,8]}
gb= GridSearchCV(estimator=gb, param_grid= parameters, cv=5, n_jobs=-1)
gb.fit(X,y)
gb.best_params_


# In[ ]:


gb = GradientBoostingClassifier(learning_rate=0.05,max_depth=2,n_estimators=100,subsample=0.1)
max_auc_roc_sc(gb,X,y)


# In[ ]:


pred_gb= gb.predict(X_test)
gb1= gb.score(X_test, y_test)
print("The accuracy of the Grading Boosting Model is:",round((gb1),3))


# In[ ]:


print(confusion_matrix(y_test,pred_gb))


# <b>Observations:</b>
# <li> There are 352 observations which are predicted Positive as TP(True Positive) and it is true.</li>
# <li> There are 20 observations which are predicted Negative as TN(True Negative) and it is True.</li>
# <li> There are 5 observations which are predicted Negative as FN(False Negative) and it is False. </li>
# <li> There are 45 observations which are predicted Positive as FP(False Positive) and it is False.</li>

# In[ ]:


print(classification_report(y_test,pred_gb))


# In[ ]:


from sklearn.model_selection import cross_val_score
mean_gb_auc=cross_val_score(gb, X,y,cv=5,scoring="roc_auc").mean()
print("Mean AUC_ROC Score after cross validation", cross_val_score(gb, X,y,cv=5,scoring="roc_auc").mean())
std_gb_auc= cross_val_score(gb, X,y,cv=5,scoring="roc_auc").std()
print("standard deviation for Gradient  Boosting from mean AUC_ROC score is",cross_val_score(gb, X,y,cv=5,scoring="roc_auc").std())


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_pred_prob= gb.predict_proba(X_test)[:,0]
tpr,fpr, thresholds= roc_curve(y_test, y_pred_prob)


# Plot
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Gradient Boosting")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Gradient Boosting")
plt.show()


# In[ ]:


a_c3=roc_auc_score(y_test, gb.predict(X_test))
a_c3


# <b>Random Forest Classifier</b>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()
parameters= {'n_estimators':[4,6,8],'max_features':['log2','sqrt','auto'],'criterion':['entropy','gini'],'max_depth':[2,5,10],'min_samples_split':[2,3,5],'min_samples_leaf':[3,5,7]}
rfc= GridSearchCV(rfc,parameters)
rfc.fit(X,y)
rfc.best_params_


# In[ ]:


rfc=RandomForestClassifier(criterion='gini', max_depth=10,max_features='sqrt',min_samples_leaf=5,min_samples_split=3,n_estimators=8)
max_auc_roc_sc(rfc,X,y)


# In[ ]:


pred_rfc= rfc.predict(X_test)
rf= rfc.score(X_test, y_test)
print("The accuracy of the Random Forest Classifier is:",round((rf),3))


# In[ ]:


print(confusion_matrix(y_test,pred_rfc))


# <b>Observations:</b>
# <li> There are 353 observations which are predicted Positive as TP(True Positive) and it is true.</li>
# <li> There are 19 observations which are predicted Negative as TN(True Negative) and it is True.</li>
# <li> There are 4 observations which are predicted Negative as FN(False Negative) and it is False. </li>
# <li> There are 46 observations which are predicted Positive as FP(False Positive) and it is False.</li>

# In[ ]:


print(classification_report(y_test,pred_rfc))


# In[ ]:


from sklearn.model_selection import cross_val_score
mean_rfc_auc=cross_val_score(rfc, X,y,cv=5,scoring="roc_auc").mean()
print("Mean AUC_ROC Score after cross validation", cross_val_score(rfc, X,y,cv=5,scoring="roc_auc").mean())
std_rfc_auc= cross_val_score(rfc, X,y,cv=5,scoring="roc_auc").std()
print("standard deviation for Random Forest Classifier from mean AUC_ROC score is",cross_val_score(rfc, X,y,cv=5,scoring="roc_auc").std())


# In[ ]:


y_pred_prob= rfc.predict_proba(X_test)[:,0]
tpr,fpr, thresholds= roc_curve(y_test, y_pred_prob)

# Plot
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest Classifier")
plt.show()


# In[ ]:


a_c4=roc_auc_score(y_test, rfc.predict(X_test))
a_c4


# # Evaluation:

# In[ ]:


#Lets initialise the data frame with columns model and f1_score
data= [["KNN", m1, mean_knn_auc,st_knn_auc],["Decision Tree Classifier",dtc1,mean_dtc_auc,s_dtc_auc],["Gradient Boosting Classifier", gb1,mean_gb_auc, std_gb_auc],["Random Forest Classifier",rf,mean_rfc_auc,std_rfc_auc]]
comparsion_table= pd.DataFrame(data, columns=["Model Name", "Accuracy","Mean AUC Score"," Std from mean AUC Score"], index=[1,2,3,4])
comparsion_table


# <b>Observations:</b>
# <li> From the above models, Decision Tree Classifier performed well with 93.60% accuracy.</li>
# <li>As the data was imbalanced, we used AUC ROC for model evaluation and calculated Mean AUC Score and Standard Deviation mean AUC Score</li>

# <b>Saving the Prediction</b>

# As the Decision Tree Classifier performed well, we are saving the prediction.

# In[ ]:


np.savetxt('HR.csv',pred_dtc,delimiter=',')


# In[ ]:


#Lets save the above model
from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(dtc, 'hr.pkl')

