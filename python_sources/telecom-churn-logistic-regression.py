#!/usr/bin/env python
# coding: utf-8

# Out of the given Customer Churn dataset, Build a Logistic Regression model to predict Customer Churns.

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# For import error with scipy 1.3. downgraging to scipy 1.1

# In[ ]:


get_ipython().system('pip uninstall statsmodels --yes')
get_ipython().system('pip install statsmodels==0.10.0rc2 --pre')


# In[ ]:


get_ipython().system('pip install --upgrade scipy==1.1.0')
get_ipython().system('pip install statsmodels==0.10.0rc2 --pre')


# In[ ]:


import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_excel('../input/Churn_data.xlsx')
df.head()


# In[ ]:


#checking columns datatypes
df.dtypes


# All datatypes are floor and int

# In[ ]:


#data summary, checking for outliers 
df.describe().transpose()


# In[ ]:


#checking for any null values
df.isnull().any().any()


# In[ ]:


#checking dependent variable classes
import numpy as np
class_freq=np.bincount(df.Churn)
pChurn=class_freq[1]/sum(class_freq) #will use it later
print("probabilities:")
print("No Churn: "+str(class_freq[0]/sum(class_freq)))
print("Churn: "+str(class_freq[1]/sum(class_freq)))


# Dataset is very unbalanced.
# 14.45% of the total customers churn.

# In[ ]:


#Writing a function to calculate the VIF values
import statsmodels.formula.api as sm
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)


# Checking for Correlation of independent variables with Churn

# In[ ]:


#correlation graph
sns.set(style='darkgrid',palette="muted")
#fig, ax=plt.subplots(figsize=dims)
df.corr()['Churn'][1:].sort_values(ascending = False).plot(kind='bar')
plt.xlabel("Dependent Variables")
plt.ylabel("Correlation to Churn")
plt.title("Correlation to Churn")


# CustServCalls, DayMins have high positive correlation with Churn where as Contract renewal has high negative correlation.
# We will investigate further.

# In[ ]:


#Calculating VIF values using that function
vif_cal(input_data=df, dependent_col="Churn")


# VIF(Multicollinearity) is high for a lot of variables.
# VIF should ideally be 1, but anythin under 5 is good and under 10 can be considered okay.
# But here they are quite high.
# We will try and lower it down

# In[ ]:


# Acceptable vif columns: AccountWeeks, Contract Renewal, CustServCalls, DayCalls, RoamMins
vif_cal(df.drop(columns=['MonthlyCharge','DataUsage']),dependent_col="Churn")
#Removing Monthly Charge and DataUsage leads to very good improvement in vif.


# All VIFs are 1 now. Thats ideal, we can proceed with these variables

# Lets check Interaction of Churn with these variables

# In[ ]:


ax=sns.kdeplot(df.CustServCalls[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.CustServCalls[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Customer case calls')
ax.set_title('Distribution of customer service calls by churn')


# In[ ]:


ax=sns.kdeplot(df.AccountWeeks[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.AccountWeeks[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('AccountWeeks')
ax.set_title('Distribution of AccountWeeks by churn')


# there is less interaction with AccountWeeks, as seen from correlation plot

# In[ ]:


ax=sns.countplot("ContractRenewal",data=df,palette="rainbow",hue="Churn")
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('ContractRenewal')
ax.set_title('Distribution of ContractRenewal by churn')


# In[ ]:


ax=sns.countplot("DataPlan",data=df,palette="rainbow",hue="Churn")
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('DataPlan')
ax.set_title('Distribution of DataPlan by churn')


# In[ ]:


ax=sns.kdeplot(df.DayMins[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.DayMins[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('DayMins')
ax.set_title('Distribution of DayMins by churn')


# In[ ]:


ax=sns.kdeplot(df.DayCalls[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.DayCalls[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('DayCalls')
ax.set_title('Distribution of DayCalls by churn')


# In[ ]:


ax=sns.kdeplot(df.OverageFee[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.OverageFee[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('OverageFee')
ax.set_title('Distribution of OverageFee by churn')


# In[ ]:


ax=sns.kdeplot(df.RoamMins[(df['Churn']==1)],color="blue",shade=True)
ax=sns.kdeplot(df.RoamMins[(df['Churn']==0)],color="red",shade=True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('RoamMins')
ax.set_title('Distribution of RoamMins by churn')


# **Moving to Logistic regression**

# In[ ]:


#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Removing Columns with high VIF from the model

# In[ ]:


X = df.drop(columns = ['Churn','MonthlyCharge','DataUsage'])
y = df['Churn'].values


# Creating Test/Train sets in 70-30

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


#    Running the model on Training data

# In[ ]:


model = LogisticRegression()
result = model.fit(X_train, y_train)


# Predicting on the Test Set

# In[ ]:


prediction_test = model.predict(X_test)
# Print the prediction accuracy
metrics.accuracy_score(y_test, prediction_test)


# Model have an accuracy of 86.6% that seems good, But as the set is imbalanced and we are interested in predicting Churns, this may be a bad way of accessing the model.

# Looking for Model Coefficients(Betas)

# In[ ]:


model.coef_[0]


# Plotting the coefficients

# In[ ]:


weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# The results are similar to our Exploratory Data Analysis. DayMins, DayCalls and AccountWeeks have less weightage.

# Lets look at the confusion matrix

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# Lets look at Precision and Recall

# In[ ]:


metrics.precision_recall_fscore_support(y_test,prediction_test)


# Precision : .468
# Recall: .16
# We are more interested in imporving Recall(% of Churns predicted out of actual Churns)

# Cheking out predicted probabilities

# In[ ]:


predicted_proba=model.predict_proba(X_test)
predicted_proba


# Lets draw some graphs

# In[ ]:


#plot precicion, recall and thresholds
#predicted_proba[:,1]
def plotPrecisionRecallThreshold(y_test, pred_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 
   #retrieve probability of being 1(in second column of probs_y)
    pr_auc = metrics.auc(recall, precision)
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])
    
def plotROC(y_test,pred_prob):
    fpr, tpr, threshold=metrics.roc_curve(y_test,pred_prob)
    plt.title("ROC Curve")
    sns.lineplot(x=fpr,y=tpr,palette="muted")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    
def areaUnderROC(y_test, pred_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 
    return metrics.auc(recall, precision)


# In[ ]:


plotPrecisionRecallThreshold(y_test, predicted_proba[:,1])


# In[ ]:


plotROC(y_test, predicted_proba[:,1])


# In[ ]:


areaUnderROC(y_test, predicted_proba[:,1])


# Area Under ROC Curve: .37, Not impressive but again given unbalanced dataset, thats what we got.

# Now time to tweek the Model.
# Remember the data that we have, only 14.5% of the records are Churn. Its unfair to have prediction threshold of .5
# Updating threshold to pChurn
# 

# In[ ]:


import numpy as np
import math
pred=np.empty(1000)
probsChurn= predicted_proba[:,1]
pred=np.empty(1000)
thresh=pChurn
for i in range(0, probsChurn.size):
    if probsChurn[i]>thresh:
        pred[i]=1
    else:
        pred[i]=0
        


# Prediction with New threshold

# In[ ]:


metrics.precision_recall_fscore_support(y_test,pred)


# Yoohoo!
# Recall: 77% (Model is able to predict 77% of the actual Churns correctly)
# Precision: 33% (Out of all Churns we predict, 33% are actually Churns)
# However the Precision is not quite upto the mark, but we do manage to Predict 77% of Churns, that is pretty good number as compared to the default rate of 14% in the given dataset.

# And The Confusion Matrix

# In[ ]:


arr=metrics.confusion_matrix(y_test,pred)
df_cm = pd.DataFrame(arr, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1.2)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 15},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .145")


# In[ ]:


metrics.accuracy_score(y_test, pred)


# The accuracy of our model has gone down to 76%, but that should not be a matter of much concern because predicting No Churns doesnot help. What we are interested in is predicting Churns.

# Hope, you find this helpful.
# This is my first Kernel !!
# Any feedback appreciated.

# In[ ]:




