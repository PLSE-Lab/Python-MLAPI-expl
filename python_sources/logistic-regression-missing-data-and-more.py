#!/usr/bin/env python
# coding: utf-8

# # Pima Indians: Logistic regression, handling missing data, feature reduction, and interpretations and insights from the model.

# # 1. Introduction

# In this notebook we will create a machine learning model with logistic regression, use a feature reduction technique to eleminate less useful features, interpret the coefficients of each feature, and interpret the outcomes of our machine learning model. 

# ## The Data Set

# This is a data set from the National Institute of Diabetes and Digestive and Kidney Diseases. All patients here are females at least 21 years old of pima indian heritage. 

# ## The Features

# <b>Pregnancies</b>: Number of times pregnant
# 
# <b>Glucose</b>: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
# 
# <b>BloodPressure</b>: Diastolic blood pressure (mm Hg)
# 
# <b>SkinThickness</b>: Triceps skin fold thickness (mm)
# 
# <b>Insulin</b>: 2-Hour serum insulin (mu U/ml)
# 
# <b>BMI</b>: Body mass index (weight in kg/(height in m)^2)
# 
# <b>DiabetesPedigreeFunction</b>: Diabetes pedigree function
# 
# <b>Age</b>: Age (years)
# 
# <b>Outcome</b>: Class variable (0 or 1) 268 of 768 are 1, the others are 0

# # 2. Data Exploration and Cleaning

# ## Exploring the data

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# In[ ]:


diabetes_df=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
diabetes_df.head(10)


# In[ ]:


diabetes_df.describe()


# In[ ]:


#Checking for Null Values in each Feature
diabetes_df.isnull().sum()


# So we have no null values.

# In[ ]:


#Drawing a histogram of each feature
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='deepskyblue')
        ax.set_title(feature+" Distribution",color = 'black')
        
    fig.tight_layout()
    plt.show()

draw_histograms(diabetes_df, diabetes_df.columns,4,3)


# They seem to have replaced missing data with the number 0 in several categores: Glucose, BloodPressure, SkinThickness, Insulin, BMI.

# In[ ]:


#Checking the outcome counts
diabetes_df.Outcome.value_counts()


# In[ ]:


sn.countplot(x='Outcome', data=diabetes_df)


# There are 500 without diabetes and 268 with diabetes.

# In[ ]:


#Looking for correlation between the different features
sn.pairplot(data=diabetes_df)


# In[ ]:


plt.figure(figsize=(12,10))
sn.heatmap(diabetes_df.corr(), annot=True,cmap ='coolwarm', vmax=.6)


# ## Cleaning the Data

# In logistic regression, colinear features can reduce the accuracy of the estimate coefficients for the features in the model. There appears to be some correlation between the different features. Several features will likely be removed. A good way to check for colinearity is to look at the VIF (variance inflation factor) values of each feature. If a feater has a VIF value greater than 5, it is safe to remove it.

# In[ ]:


import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif[0:-1])


# None of our VIF values are large enough to drop a feature. Instead, we will clean up the missing values in our data, and then use a technique called Recursive feature elimination to decide which features to train our model on.

# In[ ]:


#adding a constant value to the dataframe. This will not influence the accuracy of our model.
from statsmodels.tools import add_constant as add_constant
diabetes_df_constant = add_constant(diabetes_df)


# In[ ]:


#running a logistic regression model in order to look at the p values of each feature
import scipy.stats as st
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=diabetes_df_constant.columns[:-1]
model = sm.Logit(diabetes_df.Outcome,diabetes_df_constant[cols])
result = model.fit()
result.summary()


# Time to clean the data. We will assume that a value of 0 means that the value is actually missing, and was defaulted to 0. We need to drop the SkinThickness Feature because there is too much missing data, and we need to replace the 0(missing) values for the glucose, insulin, BMI, and blood pressure features with the median value.

# In[ ]:


#dropping the SkinThickness feature
diabetes_df_drop = diabetes_df_constant.drop(['SkinThickness'], axis=1)


# In[ ]:


#Replacing the 0 values of the Glucose, Insulin, BMI, and BloodPressure features with their median values
median_glucose = diabetes_df_drop['Glucose'].median(skipna=True)
median_Insulin = diabetes_df_drop['Insulin'].median(skipna=True)
median_BMI = diabetes_df_drop['BMI'].median(skipna=True)
median_bp = diabetes_df_drop['BloodPressure'].median(skipna=True)
diabetes_df_drop['Glucose']=diabetes_df_drop.Glucose.mask(diabetes_df_drop.Glucose == 0,median_glucose)
diabetes_df_drop['Insulin']=diabetes_df_drop.Insulin.mask(diabetes_df_drop.Insulin == 0,median_Insulin)
diabetes_df_drop['BMI']=diabetes_df_drop.BMI.mask(diabetes_df_drop.BMI == 0,median_BMI)
diabetes_df_drop['BloodPressure']=diabetes_df_drop.BloodPressure.mask(diabetes_df_drop.BloodPressure == 0,median_bp)


# In[ ]:


#Feature histograms after replacing missing data
draw_histograms(diabetes_df_drop, diabetes_df_drop.columns,4,3)


# In[ ]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=diabetes_df_drop.columns[:-1]
model = sm.Logit(diabetes_df.Outcome,diabetes_df_drop[cols])
result = model.fit()
result.summary()


# After cleaning the data, some of our P values are still high. We will now use recursive feature elimination. With this technique, we will test the importance of each feature, remove the feature with the lowest performance, then run the test again untill all of our features p values fall below our alpha of 0.05.

# In[ ]:


def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)


# In[ ]:


#using feature elimination function given above
result=back_feature_elem(diabetes_df_drop,diabetes_df_drop.Outcome,cols)


# In[ ]:


result.summary()


# After running recursive feature elimination, we removed the Insulin, BloodPressure, and Age features. We are left with the Pregnancies, Glucose, BMI, and DiabetesPedigreeFunction features.

# # 3. Building the Model

# ## Interpreting the values of model's coefficients

# In[ ]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


# Our model shows that, holding all other reatures constant, the odds of getting diagnosed with diabetes for females 21+ years old:
# 
# 1. For every pregnancy the woman has, her odds of being diagnosed with diabetes increases by <b>15.4%</b>.
# 2. For every 1 mg/dL increase in Glucose, her odds of being diagnosed with diabetes increased by <b>3.76%</b>.
# 3. For every unit increase in BMI, her odds of being diagnosed with diabetes increases by <b>9.28%</b>.
# 4. For every unit increase in the Diabetes Pedigree Function, her odds of being diagnosed with diabetes increases by <b>241.55%</b>
# 

# ## Training our machine learning model

# In[ ]:


import sklearn
new_features = diabetes_df[['Pregnancies','Glucose','BMI','DiabetesPedigreeFunction','Outcome']]


# In[ ]:


#Creating the training and test sets
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)


# In[ ]:


#Running our machine learning model
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[ ]:


#Our Accuracy score
sklearn.metrics.accuracy_score(y_test,y_pred)


# Our model predicted correctly 81% of the time.

# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
hm = sn.heatmap(conf_matrix, annot=True,fmt='d',cmap='coolwarm',vmax=50, cbar=False)
hm.set_title("Model Confusion Matrix")
hm


# True Positives: 35
# True Negatives: 90
# False Positives: 10
# False Negatives: 19

# In[ ]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[ ]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# <b>Sensitivity</b> is the true positive rate.
# 
# <b>Specificity</b> is the true negative rate.
# 
# <b>Positive predictive value</b> is the probability that somebody predicted to be positive with diabetes actually has diabetes.
# 
# <b>Negative predictive value</b> is the probability that somebody predicted to be negative with diabetes, doesn't actually have diabetes.
# 
# <b>Positive likelihood ratio</b> gives the change in odds of having a diagnosis in women with a positive test. Our positive likelihood ratio indicates a 6.9 fold increase in the odds of having diabetes in a woman with a positive test result.
# 
# <b>Negative likelihood ratio</b> gives the change in the odds of having a diagnosis in women with a negative test. Our negative likelihood ratio indicates a 2.6 fold decrease in the odds of having diabetes in a woman with a negative test result.
# 
# citation: https://www.uws.edu/wp-content/uploads/2013/10/Likelihood_Ratios.pdf
# 

# In[ ]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Diabetes classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[ ]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# # Conclusion

# Our logistic regression model predicted correctly 81% of the time, and our AUC score was 87%. We are pretty happy with those results. We were also able to get some interesting information about the likelyhood of our different features affecting a diabetes diagnosis.

# ## Credits

# A big thanks to my partner on this project Jordan King for all of his help with concepts of logistic regression and interpretations of the data. 
# 
# We took a lot of inspiration from the kaggle user "Nisha" and her approach to interpreting logistic regression models on her "Heart Disease and Prediction using Logistic Regression" project found here: https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression.
