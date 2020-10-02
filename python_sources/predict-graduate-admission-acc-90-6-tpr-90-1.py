#!/usr/bin/env python
# coding: utf-8

# ## Packages Used in Project

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For Plotting and Visualization
import seaborn as sns # For Plotting and Visualization
from sklearn.model_selection import train_test_split # For Splitting the Dataset into Training and Test Dataset
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics # Metrics Calculation
import statsmodels.api as sm # For Building Logistic Regression Model
import os
print(os.listdir("../input"))


# ## Classes and Functions Used

# Created a Class which contains reusable functions used for Calculating `Performance Evaluation Metrics` from the `Built Logistic Regression Model`

# In[ ]:


class CalculateMetrics():
        def __init__(self,x,y):
            self.actualvalue=x
            self.predictedvalue=y
            
# Calculate the Accuracy of the Model
        def accuracy(self):
            return metrics.accuracy_score(self.actualvalue,self.predictedvalue)

# Calculate the Other Performance Evaluation Metrics 
        def eval_metrics(self):
            confusionmatrix=metrics.confusion_matrix(self.actualvalue,self.predictedvalue)
            # Created a Dictionary which will hold the values of Evaluation Metrics as Key-Value Pairs
            metrics_dict={ "ConfusionMatrix":confusionmatrix
                          ,"TruePositiveRate":confusionmatrix[1,1]
                          ,"FalsePositiveRate":confusionmatrix[0,1]
                          ,"TrueNegativeRate":confusionmatrix[0,0]
                          ,"FalseNegativeRate":confusionmatrix[1,0]
                          ,"Sensitivity":(confusionmatrix[1,1]/float(confusionmatrix[1,1]+confusionmatrix[1,0]))
                          ,"Specificity":(confusionmatrix[0,0]/float(confusionmatrix[0,0]+confusionmatrix[0,1]))
                         }   
            return metrics_dict


# **Function to Print Evaluation Metrics**

# In[ ]:


def print_evalmetrics(metrics_obj):
    items=(('Accuracy of the Model:',metrics_obj.accuracy()),('Sensitivity:',metrics_obj.eval_metrics().get("Sensitivity")),('Specificity:',metrics_obj.eval_metrics().get("Specificity")),('TruePositiveRate:',metrics_obj.eval_metrics().get("TruePositiveRate")),('FalsePositiveRate:',metrics_obj.eval_metrics().get("FalsePositiveRate")),('TrueNegativeRate:',metrics_obj.eval_metrics().get("TrueNegativeRate")),('FalseNegativeRate:',metrics_obj.eval_metrics().get("FalseNegativeRate")))
    for item in items:
        print(item[0],item[1])


# **Function to Plot ROC Curve**

# In[ ]:


def plot_roc(actualvalue,probabilityvalue):
    fpr,tpr,thresholds=metrics.roc_curve(actualvalue,probabilityvalue,drop_intermediate=False)
    #Calculate the Area Under Curve Score
    auc_score=metrics.roc_auc_score(actualvalue,probabilityvalue)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr,label='ROC Curve (area=%0.2f)'% auc_score)
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return None
    


# In[ ]:


# Read the input dataset

df_input=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df_input.head()


# In[ ]:


# Read the input dataset

df_input=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df_input.head()


# In[ ]:


# Column Information of the Dataset

df_input.info()


# ## Assumption

# - In order to build a Logistic Regression Model ,we need a output variable which is said to be binary in nature i.e having a value of either 0 or 1
# - Currently Chance of Admit has a Probability Value of Student getting an Admission
# - I will set a threshold of value of 0.75 i.e Students whose Chance of Admit value is greater than 0.75 will get an admission (i.e value will be 1) and value less than 0.75 will be treated as 0

# In[ ]:


# Convert Chance of Admit Column into 0 and 1 as this will be considered as response variable
df_input.loc[df_input['Chance of Admit ']>0.75,'Chance of Admit ']=1
df_input.loc[df_input['Chance of Admit ']<0.75,'Chance of Admit ']=0
df_input['Chance of Admit ']=df_input['Chance of Admit '].astype(np.int64)
df_input.head()


# In[ ]:


# Understand the Correlation between the Columns in the dataset
sns.set(style='ticks',color_codes=True)
sns.pairplot(df_input)
plt.show()


# `GRE Score` is said to be highly correlated with `TOEFL Score`, `CGPA`

# In[ ]:


# Check for Missing Values in any of the Columns in the dataset

df_input.isnull().sum()


# In[ ]:


# Drop Unnecessary Columns from the Dataset

df_input=df_input.drop(['Serial No.'],axis=1)


# - There is no missing values in any of the columns in the dataset.
# - we have necessary data to build the Logistic Regression Model.

# ## Dataset Train - Test Split

# In[ ]:


# Divide the input dataset into train and test dataset

# Putting Feature Variable into X

X=df_input.drop(['Chance of Admit '],axis=1)
X.head()


# In[ ]:


# Putting Response Variable into Y

Y=df_input['Chance of Admit ']
Y.head()


# In[ ]:


# Splitting the data into train and test 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=100)


# ## Model Building

# ## Logistic Regression Model 1

# In[ ]:


X_train_sm=sm.add_constant(X_train)
logm1=sm.GLM(Y_train,X_train_sm,family=sm.families.Binomial())
res=logm1.fit()
res.summary()


# - From the results, predictor variables like `SOP`,`LOR`,`Research` has high p-values
# - Let us drop this predictor variables one by one and notice the behaviour of the Model

# ## Logistic Regression Model 2

# In[ ]:


# Drop the SOP variable which has high p-value

X_train=X_train.drop(['SOP'],axis=1)
X_train_sm=sm.add_constant(X_train)
logm2=sm.GLM(Y_train,X_train_sm,family=sm.families.Binomial())
res2=logm2.fit()
res2.summary()


# ## Logistic Regression Model 3

# In[ ]:


# Drop the Research Variable which has high p-value

X_train=X_train.drop(['Research'],axis=1)
X_train_sm=sm.add_constant(X_train)
logm3=sm.GLM(Y_train,X_train_sm,family=sm.families.Binomial())
res3=logm3.fit()
res3.summary()


# ![](http://)- `LOR` and `TOEFL Score` Predictor variables has moderate p-value and can still be considered as predictor varaibles for Model Prediction

# In[ ]:


# Get the Predicted Values from the training set

Y_train_pred=res3.predict(X_train_sm)
Y_train_pred[:10]


# In[ ]:


Y_train_pred=Y_train_pred.values.reshape(-1)
Y_train_pred[:10]


# In[ ]:


# Creating a data frame with response variable and predicted probabilities

Y_train_pred_final=pd.DataFrame({'Admission':Y_train.values,'Admission_Probability':Y_train_pred})
Y_train_pred_final['StudentID']=Y_train.index
Y_train_pred_final.head()


# - We have to find an optimal cut-off probability which decide whether a student gets an admission into Graduate School.
# - Once we find the optimal cut-off value ,then if the Admission_Probability is greater than optimal cut-off value it is highly possible the Student gets an admission

# ## Finding Optimal Cut-Off Point

# In[ ]:


# Let us create columns with different ranges of probabilities

numbers=[float(x)/10 for x in range(10)]
for i in numbers:
    Y_train_pred_final[i]=Y_train_pred_final.Admission_Probability.map(lambda x:1 if x>i else 0)
Y_train_pred_final.head()


# ## Accuracy,Sensitivity and Specificity

# In[ ]:


# let us identify accuracy,sensitivity and specificity for different proability cut-off

cutoff_df=pd.DataFrame(columns=['prob','accuracy','sensitivity','specificity'])

# Confusion Matrix metrics are derived below

num=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:
    cm1=confusion_matrix(Y_train_pred_final.Admission,Y_train_pred_final[i])
    total1=sum(sum(cm1))
    #Accuracy of the Model
    accuracy=(cm1[0,0]+cm1[1,1])/total1
    #Sensitivity of the Model
    sensitivity=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    #Specificity of the Model
    specificity=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    cutoff_df.loc[i]=[i,accuracy,sensitivity,specificity]
print(cutoff_df)


# In[ ]:


## Plotting Accuracy,Sensitivity and Specificity for Various Probabilities
cutoff_df.plot.line(x='prob',y=['accuracy','sensitivity','specificity'])
plt.show()


# It is eveident from the plot that `0.4 is the ideal proability cut off point` which balances the sensitivity,specifciity and accuracy.

# ## Model Prediction

# In[ ]:


# Create the Predicted Column which will be assigned a value 1 if the Probability of Admission is greater than ideal probability cut off point

Y_train_pred_final['Predicted']=Y_train_pred_final.Admission_Probability.map(lambda x:1 if x>0.4 else 0)
Y_train_pred_final.head()


# In[ ]:


metrics_results=CalculateMetrics(Y_train_pred_final.Admission,Y_train_pred_final.Predicted)
print_evalmetrics(metrics_results)


# In[ ]:


## Plotting the ROC Curve for the train set
plot_roc(Y_train_pred_final.Admission,Y_train_pred_final.Admission_Probability)


# ## Predictions on the Test Set

# In[ ]:


# Select Columns Necessary for Prediction

X_test=X_test.drop(['SOP','Research'],axis=1)
X_test.info()


# In[ ]:


X_test_sm=sm.add_constant(X_test)


# In[ ]:


Y_test_pred=res3.predict(X_test_sm)


# In[ ]:


Y_test_pred=Y_test_pred.values.reshape(-1)
Y_test_pred[:10]


# In[ ]:


# Creating a data frame with response variable and predicted probabilities

Y_test_pred_final=pd.DataFrame({'Admission':Y_test.values,'Admission_Probability':Y_test_pred})
Y_test_pred_final['StudentID']=Y_test.index
Y_test_pred_final.head()


# In[ ]:


# Create the Predicted Column which will be assigned a value 1 if the Probability of Admission is greater than ideal probability cut off point

Y_test_pred_final['Predicted']=Y_test_pred_final.Admission_Probability.map(lambda x:1 if x>0.4 else 0)
Y_test_pred_final.head()


# In[ ]:


## Metrics Results of the test set
metrics_results=CalculateMetrics(Y_test_pred_final.Admission,Y_test_pred_final.Predicted)
print_evalmetrics(metrics_results)


# In[ ]:


# Plotting the ROC Curve for the test set
plot_roc(Y_test_pred_final.Admission,Y_test_pred_final.Admission_Probability)


# ## Conclusion

# - Model built predicted the results of the test data with an `Accuracy` of `90.6%` and a `high sensitivity` of `90.1%`
# - Moreover the more the ROC plot follows the left hand border i.e with high Sensitivity which says the Model built is accurate.
