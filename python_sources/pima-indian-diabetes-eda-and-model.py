#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


diabetes_df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
print(diabetes_df.shape)
diabetes_df.head()


# We have 768 patients in the dataset and 9 columns. Lets discuss each column one by one.
# 
# Pregnancies is the number of times a person was pregnant. It's important to know the number of times a woman has become pregnant because each pregnancy has the risk of developing gestional diabetes (GDM) which itself is a risk factor for diabetes (DM). 
# 
# Glucose is the measure of plasma (or blood) glucose 2 hours after a oral glucose test. The elevated levels of glucose after 2 hours indicates impaired insulin function which can be a diagnostic for diabetes.
# 
# Blood pressure itself can't be used to determine if one has diabetes or not, but high blood pressure usually coexists with diabetes.
# 
# SkinThickness is how thick the tricep skin fold is. This test is a surrogate measure for body fat composition. Diabetics usually have high body fat.
# 
# Insulin is the insulin level, 2 hours after ingesting oral glucose.
# 
# BMI is body mass index. This column informs us if the patient is overweight for their height. Being overweight is a risk factor for diabetes.
# 
# Age is self-explanatory. Diabetes is more common in the elderly due to various reasons.
# 
# Outcome is the target variable. It tells us if someone has diabetes or not.
# 
# ## EDA
# 
# Lets explore each column and see how well correlated they are to the outcome.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(diabetes_df.corr(), annot=True, linewidths=1, cmap='YlGnBu')
plt.show()


# As we can see most of the variables are waekly correlated with Outcome. Glucose highly correlates with the outcome. Insulin surprisingly correlates more with SkinThickness than with Outcome. This could be due to the fact since high body fat is known to cause increased insulin resistance which leads to the pancreas to release more insulin. Increased SkinThickness leads to increased body fat.
# 
# Lets plot each column with outcome and deeply explore their relationship
# 
# 
# ### What is the distribution of number of pregnancies?
# ### What is the effect of number of pregnancies on the outcome?

# In[ ]:


sns.distplot(diabetes_df['Pregnancies'], kde=False, bins=range(0,17))
plt.show()

pregnancies_group = diabetes_df.groupby(['Pregnancies'], as_index=False)
pregnancies_group_count = pregnancies_group.count()['Outcome']
pregnancies_group_sum = pregnancies_group.sum()['Outcome']
pregnancies_group_percentage = pregnancies_group_sum / pregnancies_group_count * 100

plt.bar(x=range(0,17), height=pregnancies_group_percentage, yerr=pregnancies_group_percentage.std(), tick_label=range(0,17))
plt.title("Number of Pregnancies vs Diabetic Outcome")
plt.xlabel("Number of Pregnancies")
plt.ylabel("% Diabetic Outcome")
plt.show()


# We can see that most females have 0-2 babies. We can also so see that having high number of pregnancy increases the risk of diabetes significantly.
# 
# ### What is the effect of BMI on the Outcome?

# In[ ]:


def get_bmi_groups(bmi):
    if bmi >= 16 and bmi <18.5:
        return "Underweight"
    elif bmi >= 18.5 and bmi < 25 :
        return "Normal weight"
    elif bmi >= 25 and bmi < 30:
        return "Overweight"
    elif bmi >= 30 and bmi < 35:
        return "Obese Class I (Moderately obese)"
    elif bmi >= 35 and bmi < 40:
        return "Obese Class II (Severely obese)"
    elif bmi >= 40 and bmi < 45:
        return "Obese Class III (Very severely obese)"
    elif bmi >= 45 and bmi < 50:
        return "Obese Class IV (Morbidly Obese)"
    elif bmi >= 50 and bmi < 60:
        return "Obese Class V (Super Obese)"
    elif bmi >= 60:
        return "Obese Class VI (Hyper Obese)"


diabetes_df['bmi_groups'] = diabetes_df['BMI'].apply(get_bmi_groups)

bmi_groups_groupby = diabetes_df.groupby(['bmi_groups'])
bmi_groups_groupby_count = bmi_groups_groupby.count()['Outcome']
bmi_groups_groupby_sum = bmi_groups_groupby.sum()['Outcome']
bmi_groups_groupby_percentage = bmi_groups_groupby_sum / bmi_groups_groupby_count * 100
plt.figure(figsize=(16,4))
plt.bar(x=range(0,9), height=bmi_groups_groupby_percentage, yerr=bmi_groups_groupby_percentage.std(), tick_label=["Normal Weight", "Class 1", "Class 2", 
                                                                        "Class 3", "Class 4", "Class 5", "Class 6", 
                                                                        "Overweight", "Underweight"])
plt.title("BMI class vs Diabetic Outcome")
plt.xlabel("BMI classes")
plt.ylabel("Diabetic Outcome")
plt.show()


# We can see that the more overweight a person is, the more likely they are diabetic. Next we are going to create some features to do additional analysis.

# ## Feature engineering
# 
# Large waist circumference (WC) is an important risk factor for cardiovascular disease. Large WC can also be an indicator for diabetes. WC measures abdominal obesity, and abdominal obesity is linked to numerous poor health outcomes such as metabolic disease, dyslipidemia, and high blood pressure.
# 
# I have already talked about how to derive WC from BMI and Age in another notebook: Healthcare Dataset Stroke Data EDA and Modeling (https://www.kaggle.com/gundamb2/cardio-eda-and-ensemble-methods). Please check notebook out for a more indepth explanation.
# 
# Paper where this equation was obtained from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3441760/

# In[ ]:


def predicted_women_waist(bmi, age):
    c0 = 28.81919
    c1BMI = 2.218007*(bmi)
    age_35 = 0
    if age > 35:
        age_35 = 1
    
    c2IAGE35 = -3.688953 * age_35
    IAGE35 = -0.6570163 * age_35
            
    c3AGEi = 0.125975*(age)
    
    return (c0 + c1BMI + c2IAGE35 + IAGE35 + c3AGEi)

diabetes_df['waist circumference'] = diabetes_df.apply(lambda row: predicted_women_waist(row['BMI'], row['Age']), axis=1)

# Lets apply the same cut off from the previous paper and visualize the results

diabetes_df['waist_cut_off'] = diabetes_df['waist circumference'].apply(lambda size: 1 if size > 88 else 0)

waist_group = diabetes_df.groupby(['waist_cut_off'])
waist_group_count = waist_group.count()['Outcome']
waist_group_sum = waist_group.sum()['Outcome']
waist_group_percentage = waist_group_sum / waist_group_count * 100

plt.bar(x=[0,1], height=waist_group_percentage, yerr=waist_group_percentage.std(), tick_label=['Under Cut Off', 'Over Cut Off'])
plt.title("Waist Cut off vs Diabetic Outcome")
plt.xlabel("Waist Cut off")
plt.ylabel("% Diabetic Outcome")
plt.show()


# We can see that people above the cut off have a higher probability of having diabetes.
# 
# ## Model Prediction
# 
# For modelling we are going to keep it simple and use XGB classifier without any ensembling or hyperparamter optimization. We are not going to use any cross validation k-fold sets due to the very low number of inputs in this dataset.

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                 'DiabetesPedigreeFunction', 'Age', 'waist circumference']]
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

data_dmatrix = xgb.DMatrix(data=X, label=y)
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

predictions = xgb_clf.predict(X_test)

f1 = f1_score(y_test, predictions)
print("The F1 score is {}: ".format(f1))


# We can see that our F score is some where around ~0.57-0.70. However the F score is hard to interpet and doesnt tell us if our model's sensitivity or specificity. Lets instead use a AUC-ROC curve to get more details of our model.

# In[ ]:


import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.1f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# We can see from the curve that our model is somewhat good in it's prediction. AUC is above the red line where AUC = 0.5, meaning that the classifer has a slightly higher true positive rate than false positive rate. However we ideally want our ROC-AUC closer to the top left corner where F would equal 1. In this cases the classifer would correctly classify all entries correctly. This unrealistic standard to acheive but we can still try to improve our model through feature selection.
# 
# ## Feature Selection
# 
# For feature selection we are going to use the properties of xgb_clf.feature_importances_ to choose the top 3 features to use in out model.

# In[ ]:


feature_importance = pd.DataFrame()
feature_importance['columns'] = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                                 'DiabetesPedigreeFunction', 'Age', 'waist circumference']
feature_importance['importances'] = xgb_clf.feature_importances_
feature_importance


# We can see that XGB choose features that correlates closely with the Outcome. Lets pick the top 3 features (Glucose, BMI, and Age) and see if that improves out ROC-AUC curve.

# In[ ]:


X = diabetes_df[['Glucose', 'BMI', 'Age']]
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

data_dmatrix = xgb.DMatrix(data=X, label=y)
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

predictions = xgb_clf.predict(X_test)

f1 = f1_score(y_test, predictions)
print("The F1 score is {}: ".format(f1))

fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.1f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

