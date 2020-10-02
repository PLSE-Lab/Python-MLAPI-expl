#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split,cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


# number of nulls are drawn using heatmap we can also print the values ( caution may not show small values like in this case )
sns.heatmap(data.isnull(),yticklabels=False,cbar=False) 
plt.show()


# In[ ]:


sns.countplot(x="Property_Area",data=data)
plt.show()


# In[ ]:


sns.countplot(x="Gender",data=data)


# In[ ]:


sns.countplot(x="Loan_Status",hue="Married",data=data)
# it can be seen that married people are more likely to be given loan


# In[ ]:


sns.countplot(x="Loan_Status",hue="Credit_History",data=data,palette = "rainbow")
plt.show()
# it can be seen that credit history  is one of the most important feature for loan status


# In[ ]:


sns.countplot(x="Loan_Status",data=data)


# In[ ]:


data.hist(bins=50,figsize=(10,10),grid=False)
plt.tight_layout()
plt.show()


# In[ ]:


sns.boxplot(x="Credit_History",y="LoanAmount",data=data,palette="winter")


# In[ ]:


sns.countplot(x="Dependents",data=data)
plt.show()


# In[ ]:


sns.countplot(x="Self_Employed",data=data)
plt.show()


# In[ ]:


sns.countplot(x="Education",hue="Loan_Status",data=data)
plt.show()


# NAN removing according to the plots we have plotted above
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


# from the boxplot i can see that mean loan amount for both credit history is approximately same so i will fill nan using mean 
# value only
data.LoanAmount.fillna(data.LoanAmount.mean(),inplace=True)


# In[ ]:


# from the countplot of gender w can see that male is  more than female so we have high chances of nan being male 
data.Gender.fillna("Male",inplace=True)


# In[ ]:


data.Dependents.fillna('0',inplace=True)


# In[ ]:


data.Self_Employed.fillna("No",inplace=True)


# In[ ]:


data.Credit_History.fillna(0.0,inplace=True)


# In[ ]:


data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mode()[0],inplace=True)
data.Married.fillna(data.Married.mode()[0],inplace = True)


# some extra pre processing 

# In[ ]:


# ouliers in loan amount dataset can be log transformed so that effect of outliers can be removed
data["log_LoanAmount"] = np.log(data.LoanAmount)


# In[ ]:


data.log_LoanAmount.hist(bins=50)
plt.show()


# In[ ]:


# rather than using differnet values of the Income combining both to get a general income and than taking log to remove 
# effect of outliers
data["totalIncome_log"] = np.log(data.ApplicantIncome + data.CoapplicantIncome)


# In[ ]:


data.totalIncome_log.hist()


# In[ ]:


data.head()


# In[ ]:


data = data.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount"],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
categorical_column = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in categorical_column:
    data[i] = le.fit_transform(data[i])
data.head()


# In[ ]:


data.info()


# In[ ]:


data.info()


# In[ ]:


train = data
train.head()


# In[ ]:


train.Loan_Status.value_counts()


# # clearly dataset is very imbalance therefore using upsampling to make datatset balanced

# In[ ]:


from sklearn.utils import resample
data_major = train[train.Loan_Status==1]
data_minor = train[train.Loan_Status==0]
data_upscale = resample(data_minor,replace= True,n_samples=422)
train=pd.concat([data_major,data_upscale])


# In[ ]:


train.shape


# In[ ]:


def roc_curve_do(c_name,classifier,x_test,y_test):
    probs = classifier.predict_proba(x_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs)
    plt.plot(fper, tper)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC curve'.format(c_name))
  # show the plot
    plt.show()

def kfold(classifier,X,Y,cv):
    score=cross_val_score(classifier,X,Y,cv=cv)
    print("Individual Score:",score)
    print("Mean Score:",score.mean()*100,"%")
  # plot
    plt.plot(np.arange(cv), score, 'o-', linewidth=1)
    plt.title("Accuracy: %f%% and Deviation (%f%%)" % (score.mean()*100, score.std()*100))
    plt.xlabel('number of Folds')
    plt.ylabel('Accuracy score')
    plt.show()

def all_score(classifier,x_test,y_test,x_train,y_train):
    predict=classifier.predict(x_test)
    print("testing accuracy:",accuracy_score(y_test,predict))
    print("training accuracy:",accuracy_score(y_train,classifier.predict(x_train)))
    print(confusion_matrix(y_test,predict))
    print("Classification report:\n",classification_report(y_test,predict))


# In[ ]:


xtrain = train.drop(["Loan_Status"],axis=1)
ytrain = train["Loan_Status"]


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(xtrain,ytrain,test_size=0.2)


# In[ ]:


model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)


# In[ ]:


roc_curve_do("Logistic Regression",model,x_test,y_test)
kfold(model,xtrain,ytrain,5)
all_score(model,x_test,y_test,x_train,y_train)


# In[ ]:


model1 = RandomForestClassifier(n_estimators=200)
model1.fit(x_train,y_train)


# In[ ]:


roc_curve_do("Logistic Regression",model1,x_test,y_test)
kfold(model1,xtrain,ytrain,5)
all_score(model1,x_test,y_test,x_train,y_train)


# feature selection using random forest

# In[ ]:


parameters =  ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','log_LoanAmount','totalIncome_log']
featimp = pd.Series(model1.feature_importances_, index=parameters).sort_values(ascending=False)
print (featimp)


# In[ ]:


new_train = train.drop(['Property_Area','Married','Education','Gender','Self_Employed','Dependents','Loan_Amount_Term'],axis=1)


# In[ ]:


new_train.head()


# In[ ]:


new_xtrain = new_train.drop(["Loan_Status"],axis=1)
new_ytrain = new_train["Loan_Status"]


# In[ ]:


new_xtrain.head()


# In[ ]:


x_newtrain,x_newtest,y_newtrain,y_newtest = train_test_split(new_xtrain,new_ytrain,test_size=0.2)


# In[ ]:


model2 = RandomForestClassifier(n_estimators=150)


# In[ ]:


model2.fit(x_newtrain,y_newtrain)


# In[ ]:


roc_curve_do("Random Forest",model2,x_newtest,y_newtest)
kfold(model2,new_xtrain,new_ytrain,5)
all_score(model2,x_newtest,y_newtest,x_newtrain,y_newtrain)


# In[ ]:




