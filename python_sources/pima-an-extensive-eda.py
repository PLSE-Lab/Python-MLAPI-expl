#!/usr/bin/env python
# coding: utf-8

# # Working on Pima Indians Diabetes

# ## Introduction

# The dataset we have taken is on Pima Indians Diabetes data consisting of diagnostic measurements of females of atleast 21 years of age . We are asked to predict based on these diagnostic measurements whether a person is diabetic or not .In this work , I have done an extensive EDA on all the variables to understand how each variable is related to the predictor variable . I will implement Random forest algorithm to model the classification problem and compare the metrics before and after feature selection .

# ## Loading the required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import itertools
warnings.filterwarnings("ignore")


### Modelling
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix,accuracy_score,make_scorer,f1_score,auc
from sklearn.tree import export_graphviz
from IPython.display import display


# ## Reading the dataset 

# In[ ]:


kaggle=1
if kaggle==0:
    diab=pd.read_csv("diabetes.csv")
else:
    diab=pd.read_csv("../input/diabetes.csv")


# ## Exploratory Data Analysis

# In[ ]:


print("Number of rows:{} and Number of column:{}".format(diab.shape[0],diab.shape[1]))


# In[ ]:


diab.info()


# In[ ]:


diab.describe()


# The description of the variable is as follows:
# 
# *Pregnancies* - Number of times pregnant
# 
# *Glucose* - Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# *BloodPressure* - Diastolic blood pressure (mm Hg)
# 
# *SkinThickness* - Triceps skin fold thickness (mm)
# 
# *Insulin* - 2-Hour serum insulin (mu U/ml)
# 
# *BMI* - Body mass index (weight in kg/(height in m)^2)
# 
# *DiabetesPedigreeFunction* - Diabetes pedigree function
# 
# *Age* - Age (years)
# 
# *Outcome* - Class variable (0 or 1) 
# 
# 
# From the summary of the data ,we understand that all the columns are numeric . The predictor variable  Outcome is binary (0,1).

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.countplot(diab['Outcome'])
ax.set_xlabel("Outcome")
ax.set_ylabel('Count')
ax.set_title("Countplot of Predictor Variable")


# From the countplot , it is seen that there are more values of '0' than '1' which means the dataset has more non-diabetic diagnostic related information than diabetic related information .

# In[ ]:


count_non_diab=len(diab[diab['Outcome']==0])
count_diab=len(diab[diab['Outcome']==1])
print("The dataset has {} % of non-diabetic cases and {} % of diabetic cases".format(round(count_non_diab/len(diab['Outcome'])*100,2),round(count_diab/len(diab['Outcome'])*100,2)))


# In[ ]:


diab.isnull().values.any()


# There data doesnt seem to have any missing information.

# Given the nature of the dataset it may be wise to plot histogram and boxplot inorder to determine which factors will have maximum influence in predicting the outcome of the model .Therefore a function is created first to plot histogram and outcome.

# In[ ]:


## Defining a function to plot histogram for all the independent variable
def plot_variables(variable):

    f, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    sns.distplot(diab[variable][diab['Outcome']==1],ax=axes[0],color='green')
    axes[0].set_title(r"Distribution of {}-Diabetic Outcome".format(variable))
    sns.distplot(diab[variable][diab['Outcome']==0],ax=axes[1],color='red')
    axes[1].set_title(r"Distribution of {}-Non Diabetic Outcome".format(variable))
    sns.boxplot(x=diab['Outcome'],y=diab[variable],ax=axes[2],palette="Set3")
    axes[2].set_title(r"Boxplot of {}".format(variable))
    f.tight_layout()


# ### Age Vs Outcome:

# In[ ]:


plot_variables('Age')


# From the histogram it is seen that the distribution is non-normal .The histogram for age -diabetic outcome seems to be multimodal with modes centered around 20-30 years and again at 50 years whereas the histogram for age-non diabetic outcome is at 20 years .The distribution is skewed towards right for age-non diabetic outcome .There is a significant difference between the outcome and age as seen from the boxplot .The median age for a diabetic is higher compared to a non-diabetic .There are also many outliers in age for non-diabetic outcome compared to a diabetic outcome.

# ### Pregnancies Vs Outcome 

# In[ ]:


plot_variables('Pregnancies')


# Going by the first look ,it seems that for a diabetic outcome , the data is skewed towards right .The median number of pregnancies is higher for a diabetic when compared to a non-diabetic outcome .Thus the outcome with respect to pregnancies is also significant.

# ### Glucose Vs Outcome

# In[ ]:


plot_variables('Glucose')


# * I did a quick google search to understand more about this variable .According to the data describtion the values in the column represent Plasma glucose concentration a 2 hours in an oral glucose tolerance test . According to [Wikipedia](https://en.wikipedia.org/wiki/Glucose_tolerance_test#Results) the glucose tolerance test is a medical test in which glucose is given and blood samples taken afterward to determine how quickly it is cleared from the blood.The test is usually used to test for diabetes, insulin resistance, impaired beta cell function,and sometimes reactive hypoglycemia and acromegaly, or rarer disorders of carbohydrate metabolism. In the most commonly performed version of the test, an oral glucose tolerance test (OGTT), a standard dose of glucose is ingested by mouth and blood levels are checked two hours later.
# 
# * The normal blood glucose level (tested while fasting) for non-diabetics, should be between 3.9 and 7.1 mmol/L (70 to 130 mg/dL).Blood sugar levels for those without diabetes and who are not fasting should be below 6.9 mmol/L (125 mg/dL).
# 
# * A reading of more than 200 mg/dL (11.1 mmol/L) after two hours indicates diabetes.
# 
# * Thus going by our distribution plot , we see that the median value of glucose levels after 2 hour period for a non-diabetic person is around 100 mg/dL whereas for a diabetic it has been close to 140 mg/dL.There are also some outliers in the non-diabetic where the glucose levels were on the extreme end.

# ###  Blood Pressure Vs Outcome

# In[ ]:


plot_variables('BloodPressure')


# * There are two blood pressures - Systolic and Diastolic . The reading provided here is the diastolic blood pressure .Diastolic blood pressure is the pressure in the arteries when the heart rests between beats. This is the time when the heart fills with blood and gets oxygen.A normal diastolic blood pressure is lower than 80. A reading of 90 or higher means you have high blood pressure.
# 
# * From the histogram it is understood that the are values where the blood pressure is 0 !!! . The mode for diabetic outcome is on the higher side when compared to non-diabetic outcome . The median value is slightly different between the outcomes . 
# 
# * This variable is also significant in predicting the outcome.
# 

# ### Skin Thickness Vs Outcome

# In[ ]:


plot_variables('SkinThickness')


# Distribution of skin thickness shows that it is multimodal .The median values are close to each other for both the outcomes.This might not have a good influence in predicting the outcomes .

# ### Insulin Vs Outcome

# In[ ]:


plot_variables('Insulin')


# The distribution for each of the outcomes seem to be similar with the histograms skewed towards right . There are outliers for both the outcomes .The median value of insulin is close to zero in case of diabetic outcome.There is a significant difference between the outcomes with respect to insulin levels.Therefore this might be a good predictor of outcome.

# ### BMI Vs Outcome

# In[ ]:


plot_variables('BMI')


# The histogram shows that the BMI for a diabetic outcome is higher compared to a non-diabetic outcome . There are zero values in both cases.The median BMI value for a non-diabetic outcome is higher compared to a non-diabetic outcome .A significant difference between the two is seen . 

# ### DiabetesPedigreeFunction Vs Outcome

# In[ ]:


plot_variables('DiabetesPedigreeFunction')


# Distribution patterns for both the outcomes seem to be similar - skewed towards right . The median is almost equal between the two . There are visible outliers between the two .

# ### Modelling

# We split the dataset into train and test dataset by 80-20 ratio.This is not a stratified split and it is a random split .During our EDA we saw that there are some 0 values which might be some data error .We can impute those variables with median values in the column.

# In[ ]:


mut_vari=['BloodPressure','Glucose','SkinThickness','BMI']
for i in mut_vari:
    print('Imputing column {} with median value {} \n'.format(i,diab[i].median()))
    diab[i]=diab[i].replace(0,value=diab[i].median())


# In[ ]:


X=diab.drop('Outcome',axis=1)
y=diab['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# ### Random Forest

# We use the Random Forest algorithm for our modelling and use the confusion matrix,accuracy score , f1 score as our scoring metrics.Since it is a unbalanced dataset , using accuracy score will not be a good metric . Hence we give f1 score more importance.

# In[ ]:


### Function to plot confusion matrix.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


prediction={}  ### Create a dictionary of prediction for comparison.


# In[ ]:


def print_score(m):
    res = {'F1 Score for Train:':f1_score(m.predict(X_train), y_train),'F1 Score for Test:':f1_score(m.predict(X_test), y_test),
           'Accuracy Score for Train:':m.score(X_train, y_train),'Accuracy Score for Test:':m.score(X_test, y_test)}
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


###https://www.kaggle.com/tunguz/just-some-simple-eda

#param_grid = {
           # 'n_estimators': [100, 200, 500],
           # 'max_features': [2, 3, 4],
           # 'min_samples_leaf': [1, 2, 4],
           # 'min_samples_split': [2, 5, 10]
          #  }
rfm=RandomForestClassifier(random_state=100,n_jobs=-1,n_estimators=40,min_samples_leaf=5,max_features=0.5)
rfm.fit(X_train,y_train)


# In[ ]:


estimator=rfm.estimators_[5]


# In[ ]:


prediction['RF']=rfm.predict(X_test)


# In[ ]:


print_score(rfm)


# From the scores , we understand that the F1 score for test is 0.55 whereas the accuracy score is 72 %.

# In[ ]:


## Following code is taken from fast.ai libraries function by Jermey Howard .Thanks Jermey for the awesome package .

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[ ]:


class_names = set(diab['Outcome'])
cnf_matrix = confusion_matrix(y_test, prediction['RF'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Random Forest Model(W/O CV)')


# In[ ]:


###https://stats.stackexchange.com/questions/117654/what-does-the-numbers-in-the-classification-report-of-sklearn-mean
###https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
print(classification_report(y_test,prediction['RF']))


# The classification report here shows that the model has a precision score of 0.61 and a recall value of 0.51 .The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The recall is intuitively the ability of the classifier to find all the positive samples.

# Lets plot the feature importance of the model and check which variables are contributing to the model accuracy.

# In[ ]:


fi=rf_feat_importance(rfm,X_train)
fi


# From the feature importance we see that to determine the outcome , glucose , BMI ,AGE and diabetes pedigree function are considered as most important features.Lets consider only those variables and see if we can improve the accuracy of the model .

# In[ ]:


to_keep=fi[fi['imp']>0.05].cols;len(to_keep)


# We have 5 columns whose importance are more than 0.05 . We recreate the dataset with these columns.

# In[ ]:


diab_keep=diab[to_keep]


# In[ ]:


X=diab_keep
y=diab['Outcome']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=100)
print("Dimensions of train set:{} and Dimensions of test set:{}".format(X_train.shape,X_test.shape))


# In[ ]:


rfm=RandomForestClassifier(random_state=100,n_jobs=-1,n_estimators=40,min_samples_leaf=5,max_features=0.5)
rfm.fit(X_train,y_train)


# In[ ]:


prediction['RF_FI']=rfm.predict(X_test)


# In[ ]:


print_score(rfm)


# The accuracy of the model has marginally improved from 0.72 to 0.77 .But the F1 score has improved from 0.55 to 0.66.Lets print the confusion matrix and check the classification report .

# In[ ]:


class_names = set(diab['Outcome'])
cnf_matrix = confusion_matrix(y_test, prediction['RF_FI'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Random Forest Model(After Feature Selection)')


# In[ ]:


print(classification_report(y_test,prediction['RF_FI']))


# The model has done a good job in improving the f1 score . Lets plot the ROC curve and check the AUC .

# In[ ]:


#Borrowed from https://www.kaggle.com/gpayen/building-a-prediction-model
cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,predicted)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Thus after selecting the variables after feature importance , we see that there is an improvement in the AUC curve from 0.48 to 0.74 .

# ### References

# * [fastais Machine Learning class](https://course.fast.ai/ml)
# * [Glucose Tolerance Test](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451)
# * [Insulin](https://emedicine.medscape.com/article/2089224-overview)
# * [Stackexchange discussion on classification metrics](https://stats.stackexchange.com/questions/117654/what-does-the-numbers-in-the-classification-report-of-sklearn-mean)
# * [Medium blog on classification metrics](https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)
# 
# **Kaggle Kernels:**
# 
# * [Guillaume Payen Kernel - Building a prediction model](https://www.kaggle.com/gpayen/building-a-prediction-model)
# * [Bojan Tunguz Kernel - Just some simple EDA](https://www.kaggle.com/tunguz/just-some-simple-eda)

# Thanks for reading my kernel . Pls provide your valuable feedback through comments/upvotes.
