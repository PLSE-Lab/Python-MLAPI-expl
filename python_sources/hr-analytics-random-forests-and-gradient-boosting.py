#!/usr/bin/env python
# coding: utf-8

# **HR EMPLOYEE ATTRITION DATASET.**
# 
# This is a fictional data set created by IBM data scientists. We need to explore the dataset, understanding the algorithms and techniques which can be applied on it. We' ll try to gain meaningful insights from the dataset, like what are the factors which have an impact on Employee Attrition.

# In[ ]:


# Import Desired libraries.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# Importing the Dataset. After which an important step is to understand our data.

# In[ ]:


data=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
attrition=data
print(data.columns)
print(data.shape)


# So there are 35 columns and 1470 rows.
# Next let's check how many categorical variables and numerical variables:

# In[ ]:


# Differentiate numerical features (minus the target) and categorical features
categorical_features = data.select_dtypes(include=['object']).columns
categorical_features

numerical_features = data.select_dtypes(exclude = ["object"]).columns
print(categorical_features.shape)
print(categorical_features)
print(numerical_features)


# So there are 9 categorical features which includes our target  variable "Attrition".
# These will need to be encoded using one of the following ways:
# 1. Dummy Encoding
# 2.One Hot Encoder
# 3.Label Encoder
# Rest are numerical features. Before going any furthur let's check for any NULLS in our dataset.

# In[ ]:


print(data.isnull().values.any())


# Since there are no nulls we Do not need to worry about this anymore. 
# To get a better grasp of our datset. I will execute the next line of my code.

# In[ ]:


data.describe() # this creates a kind of summary of the datset withh various statistical features.


# We need to specify our target variable which is Attrition in this case. Also since Attrition is a categorical feature we will map it to numerical values.

# **DATA VISUALIZATION**  
# Data Visualization is one of the core step before building any model. Python offers numerous libraries for this purpose.I have used Seaborn library for this porpose. First and foremost I want to see how my Target variable is distributed across the dataset.

# In[ ]:


sns.countplot("Attrition",data=data)
plt.show()


# **COUNTPLOT**:-The above plot shows the distribution of our target variable.As can be seen clearly, the graph represents imbalanced dataset.
# So we' ll need to balance this dataset.
# Imbalanced class distributions are an issue when anamoly detection like fraud cases, identification of rare diseases, or cases similiar to the above are present. In such scenarios we are more interested in the minority class and the factors that contibute to the occurrence of them. 
# 
# Various techniques are available for handling imbalanced Dataset like Undersampling the Majority class and Oversampling the Minority class.
# In simple terms, it is decreasing instances of majority classes or increasing instances of minority classes to result in a balanced dataset.
# We will deal with this issue while building our model.Each has its own set of pros and cons.

# To see the correlation with our target Variable which is Attrition we will convert it into numerical values.
# I will be using Replace function for this.

# In[ ]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,annot=True,fmt= '.1f',vmax=.8, square=True)


# Statistical relationship between two variables is referred to as their ** correlation**. The performance of some algorithms can deteriorate if two or more variables are tightly related, called multicollinearity.This is of special importance in Regression.
# From the above correlation matrix , we find most of the features are uncorrelated.But, there is a correlation (0.8) between Performance Rating and Performance Salary Hike. We need to look into the white lines shown by EmployeeCount and Standard Hours.TotalWorkingYears with JobLevel also has high correlation(0.8).

# An important feature in our datset is Gender. I ' ll verify this with help of plot to understand who is more likely for Job Attrition if we only consider Gender as the factor.

# In[ ]:


x, y, hue = "Attrition", "prop", "Gender"
f, axes = plt.subplots(1,2)
sns.countplot(x=x, hue=hue, data=data, ax=axes[0])
prop_df = (data[x]
           .groupby(data[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())
sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])


# One important thing to note here is we cannot make direct inferences from first countplot. The second barplot is made in accordance with proportion.
# One can clearly infer from the plot above that higher proportion of males are likely for Attrition as compared to females.

# In[ ]:


x, y, hue = "Attrition", "prop", "Department"
f, axes = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x=x, hue=hue, data=data, ax=axes[0])
prop_df = (data[x]
           .groupby(data[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())
sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])


# I have tried to find out the relation between** Department and Attrition**. It can be infered that employees from Sales Department have higher possibility of Attrition whereas Research and Development have higher proportion on the No Attrition side.

# In[ ]:


sns.pairplot(data=data,x_vars=['MonthlyIncome'], y_vars=['Age'],height=5, hue='Attrition',palette="prism")


# The above plot shows those with lesser age and Lower income groups upto 5000 have higher possibility of Attrition as green dots are more concentrated in that region more.

# In[ ]:


sns.set()
cols=['Age','DailyRate','Education','JobLevel','DistanceFromHome','EnvironmentSatisfaction','Attrition']
sns.pairplot(data[cols],hue='Attrition',height=2.5,palette="hls")


# Above are the pairplots between various numerical variables.
# 1. The first distribution plot for age shows that employees which fall in lower age group are more likely for attrition, However, as the age increases the blue curve goes up.
# 2. For the Job Level, there is a sharp peak at lower job levels between 0-2 which shows Atrrition is more likely.
# 3. The plot between DistancefromHome and Age shows that again employees with less ages aand living at a distance greater than 15 have higher chances of Attrition.
# 4. Employees with lower level of EnvironmentSatisfaction indicates higher chances of Attrition as the Red curve goes above the blue curve for lower levels of EnvironmentSatisfaction.

# To understand the counts of different values in each feature I have used value_counts method of Pandas. 

# In[ ]:


#for c in data.columns:
    #print("---- %s ---" % c)
    #print(data[c].value_counts())


# Since attrition is a categorical Variable , one needs to convert it into numerical form. I am replacing Yes with 1 and No with 0.

# In[ ]:


data1=data
di={"Yes": 1, "No": 0}
data1["Attrition"].replace(di,inplace=True)


# Since attrition value to be classified, assigning it to the target variable

# In[ ]:


attrition=data
data1.shape
target=data.iloc[:,1]
print(target.head(5))


# In[ ]:


print(target.dtypes)
target=pd.DataFrame(target)
print(target.dtypes)


# In[ ]:


print(data1.columns)


# Since Attrition is the target variable we do not need it in our predictor variables.
# Apart from these from value_counts of each variable we can see that 'Over18', 'StandardHours', 'EmployeeCount' are all same values and can be dropped without loss of information.
# 

# In[ ]:


data1.head(5)
data1.drop(["Attrition","Over18","StandardHours","EmployeeCount","EmployeeNumber"],axis=1,inplace=True)


# Converting categorical features to dummy variables. I have used get_dummies method of pandas for the same.

# In[ ]:


categorical=data1.select_dtypes(include=['object']).columns
data1.shape
print(data1.columns)
print(categorical)
Prediction=data1##copy paste


# In[ ]:


print(data1.columns)
dummie=pd.get_dummies(data=data1, columns=['OverTime','BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole','MaritalStatus'])
dummie=pd.DataFrame(dummie)
new_data=pd.concat([data1, dummie], axis=1)
# print(new_data.columns)


# **MODEL BUILDING :****
# 
# I have used two models for this classification problem -Random Forests & Gradient Boosting Algorithm.
# 
# First , I have applied both the algorithms on the imbalanced dataset and then on the Balanced Dataset. 
# For the purpose of Balancing of our datset, I have used SMOTE(Synthetic MInority Over Sampling Technique).
# 
# **RANDOM FORESTS:**-  This is an ensemble method used for building predictive models for  both classification and regression problems.
# First I am applying Random Forests Algorithm for this Binary classification problem of Employee Attrition.This is a bagging Ensemble Model.
# Bagging is a simple ensembling technique in which we build many independent predictors/models/learners and combine them using some model averaging techniques.
# 
# **GRADIENT BOOSTING:-** Another way of ensembling to build predictive models is through Boosting. Boosting is a technique in which learners/predictors are combined sequentially, rather than independently. This means each predictor learns from the mistakes of their previous predictors. 

# In[ ]:


print(target.head(5))


# In[ ]:


new_data.drop(['OverTime','BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole','MaritalStatus'],axis=1,inplace=True)
# Since we have already created dummy variables so we can drop the columns with categorical features.


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(new_data,target,test_size=0.33,random_state=7)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# In[ ]:


#  importing Libraries for our model 
# Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=1000)
forest.fit(x_train,y_train.values.ravel())


# In[ ]:


predicted= forest.predict(x_test)


# Next using Gradient Boosting For our dataset classification.

# **EVALUATION OF MODEL:-**
# There are several metrics for evaluation of any machine learning model. The most common is accuracy.
# **ACCURACY**- This is simply the ratio of TOTAL CORRECT PREDICTIONS to TOTAL NO OF PREDICTIONS.
# However, this metric is useful especially if there are equal no of samples belonging to each class.
# This is certainly not the case in our Dataset. This is because if 90% of data belongs to one particular class say class X and 10% to class Y,then our model will get 90% accuracy even if predicts that entire sample belongs to class X.
# 
#  In such cases , a classification report is better way to check the quality of classification algorithm predictions. 
#  This report gives several classifcation metrics like recall , precision, & f1 score.
#  
# **** PRECISION:**- it is defined as the ratio of true positives to the sum of true and false positives.  It is the accuracy in our postive predictions.
# 
# **RECALL:-** It is defined as the ratio of True Positives to sum of True Positives and False Negatives.,which is how many positives we have identified out of the total positives that are actually present in the dataset.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(x_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(x_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(x_test, y_test)))
    print()


# In[ ]:


# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

gb = GradientBoostingClassifier(n_estimators=20,learning_rate = 0.5,random_state = 7)
gb.fit(x_train, y_train)
predictions = gb.predict(x_test)

print("Confusion Matrix for Gradient boosting:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report for Gradient Boosting")
print(classification_report(y_test, predictions))


# In[ ]:


print("Accuracy score (validation): {0:.3f}".format(forest.score(x_test, y_test)))
print("Confusion Matrix for Random Forests:")
print(confusion_matrix(y_test, predicted))
print()
print("Classification Report for Random Forests")
print(classification_report(y_test, predicted))


# Here we applied two methods Gradient Boosting and Random Forests and checked their accuracy as well as confusion matrix.
# But since this data was not balanced ,next we will see what effect occurs on accuracy after using techniques for imbalanced data.

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=7, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
forest_sm = RandomForestClassifier(n_estimators=500, random_state=7)
forest_sm.fit(x_train_res, y_train_res.ravel())
prediction2 = forest_sm.predict(x_test)
print("Accuracy score (validation): {0:.3f}".format(forest_sm.score(x_test, y_test)))
print("Confusion Matrix for Random Forests:")
print(confusion_matrix(y_test, prediction2))
print()
print("Classification Report for Random Forests")
print(classification_report(y_test, prediction2))


# Next  fitting our model using Gradient Boosting after using SMOTE for handling imbalanced Data.

# In[ ]:


gb_sm = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 7)
gb_sm.fit(x_train_res, y_train_res.ravel())
prediction3 = gb_sm.predict(x_test)

print("Confusion Matrix for Gradient boosting:")
print(confusion_matrix(y_test, prediction3))
print()
print("Classification Report for Gradient Boosting")
print(classification_report(y_test, prediction3))
print("Accuracy score (validation): {0:.3f}".format(gb_sm.score(x_test, y_test)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




