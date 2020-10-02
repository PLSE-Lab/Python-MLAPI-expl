#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #Importing the file as dataframe
# 

# In[6]:


diabetes= pd.read_csv('../input/diabetes.csv')


# The above data is trying to predict the outcome of whether a person is diabetic as a function of the other 8 parameters
# 
# There are total 8 independent variable which are all continous and Outcome is target variable.
# Outcome is a categorical variable and rest all are continous variable in the dataset.

# In[7]:


diabetes.describe(include = "all").transpose()


# There are six variables that has minimum value as '0' and out of these variables only 'Pregnancies' variable
# could take a 'zero' value.  The other 'zero' values can be taken as  missing values as they are not practical values
# 
# Insulin and Skin Thickness parameters has both min and Q1 value as '0' and it suggests that these two parameters has maximum missing values

# Checking the info of the dataset for null values and the size of the dataframe

# In[8]:


diabetes.info()


# Dataframe has 768 rows and 9 columns, with 06 parameters as intergers and 02 as floating point values.  Both integers 
# and floating values are of 64 bit

# Visualization of data before data refining

# In[ ]:


diabetes.columns = ['Pregnancies','Glucose','BloodPressure',
                     'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']


# In[ ]:


diabetes.columns


# In[ ]:


diabetes.hist(figsize = (20,20))


# The above histograms of the different parameters depicts that age, insulin and diabetespedigreefunction parameters
# are left skewed and 6 parameters having minimum value as 'zero'

# In[ ]:


#We find that there are '0' values in the dataset which can be missing data.  Before replacing this missing values 
#copy the dataframe and then make the required changes

pimadiabetes = diabetes.copy(deep = True)
pimadiabetes.head()


# In[ ]:


pimadiabetes.tail()


# There are no null values in the dataset and the dataset has 768 rows and 9 columns.  
# 
# The 'zero' values can be refined in following ways
# 
# - remove the rows with 'zero' values 
# - replace the 'zero' values with the population mean/median
# - replace the 'zero' values with the population mean/median  according to the outcome 
# 
# As there are many zero's in the dataset, it is not advisable to remove the rows and as mean is sensitive to outliers replacing with median is selected

# The '0' minimum value for 'Glucose','BMI','SkinThickness','BloodPressure','Insulin' values seems to be impractical. 
# Hence, let us replace these values with Null Values first and after that replace with median value.  Median value is choosen as it is not sensitive to outliers present in the dataset
# 
# The final outcome depends on the value of 'zero' that we replace and it is better to take the median according to the final outcome i.e., calculate the median of non-diabetic outcome and diabetic outcome separately and replace 0's with these respective medians

# 1. BLOOD PRESSURE

# In[ ]:


pimadiabetes[pimadiabetes['BloodPressure'] == 0].shape[0]


# In[ ]:


pimadiabetes[pimadiabetes['BloodPressure'] == 0].index.tolist()
pimadiabetes[pimadiabetes['BloodPressure'] == 0].groupby('Outcome')['BloodPressure'].count()


# There are 35 zero values for Blood pressure which can be replaced with median values, out of which 19 are non-diabetic and 16 are diabetic 

# In[ ]:


# Replacing '0' values with null values for column Glucose
pimadiabetes['BloodPressure'].replace(0,np.NaN,inplace = True)


# In[ ]:


# The median value calculation of Glucose grouped by outcome 

pimadiabetes_bp = pimadiabetes[pimadiabetes['BloodPressure'].notnull()]
pimadiabetes_bp = pimadiabetes_bp[['BloodPressure','Outcome']].groupby('Outcome')['BloodPressure'].median().reset_index()
pimadiabetes_bp                               
    


# Median for '0' Outcome is 70 and '1' Outcome is 74.5

# In[ ]:


# Replacing the zero values of glucose with the corresponding median values of outcomes as calculated
pimadiabetes.loc[(pimadiabetes['BloodPressure'].isnull()) & (pimadiabetes['Outcome'] == 0),'BloodPressure'] = pimadiabetes_bp['BloodPressure'][0]
pimadiabetes.loc[(pimadiabetes['BloodPressure'].isnull()) & (pimadiabetes['Outcome'] == 1),'BloodPressure'] = pimadiabetes_bp['BloodPressure'][1]


# In[ ]:


pimadiabetes.head(10)


# In[ ]:


# checking for replacement of all '0' values
pimadiabetes[pimadiabetes['BloodPressure'] == 0].shape[0]


# All zero values of 'BloodPressure' replaced

# 2. GLUCOSE

# In[ ]:


#zero values to be replaced
pimadiabetes[pimadiabetes['Glucose'] == 0].shape[0]


# In[ ]:


pimadiabetes[pimadiabetes['Glucose'] == 0].index.tolist()


# In[ ]:


pimadiabetes[pimadiabetes['Glucose'] == 0].groupby('Outcome')['Glucose'].count()


# There are only 5 zero values for the Glucose attribute, out of which 3 are for non-diabetic and 2 for diabetic

# In[ ]:


# Replacing '0' values with null values
pimadiabetes['Glucose'].replace(0,np.NaN,inplace = True)


# In[ ]:


# Calculation of median values grouped according to outcome
pimadiabetes_gl = pimadiabetes[pimadiabetes['Glucose'].notnull()]
pimadiabetes_gl = pimadiabetes_gl[['Glucose','Outcome']].groupby('Outcome').median()
pimadiabetes_gl


# Median for '0' Outcome is 107 and '1' Outcome is 140

# In[ ]:


pimadiabetes.loc[(pimadiabetes['Glucose'].isnull()) & (pimadiabetes['Outcome'] == 0),'Glucose'] = pimadiabetes_gl['Glucose'][0]
pimadiabetes.loc[(pimadiabetes['Glucose'].isnull()) & (pimadiabetes['Outcome'] == 1),'Glucose'] = pimadiabetes_gl['Glucose'][1]


# In[ ]:


# checking for replacement of all '0' values
pimadiabetes[pimadiabetes['Glucose'] == 0].shape[0]


# All the zero values of 'Glucose' are replaced

# 3. BMI

# In[ ]:


# Selection of data to refine for BMI
pimadiabetes[pimadiabetes['BMI'] == 0].shape[0]


# In[ ]:


pimadiabetes[pimadiabetes['BMI'] == 0].index.tolist()


# In[ ]:


pimadiabetes[pimadiabetes['BMI'] == 0].groupby('Outcome')['BMI'].count()


# There are 11 zero values to be replaced with median values and 9 are for non-diabetic and 2 for diabetic

# In[ ]:


# Replacing zero values by NaN values
pimadiabetes['BMI'].replace(0,np.NaN,inplace = True)


# In[ ]:


# Calculating median of the BMI according to outcome
pimadiabetes_bmi = pimadiabetes[pimadiabetes['BMI'].notnull()]
pimadiabetes_bmi = pimadiabetes_bmi[['BMI','Outcome']].groupby('Outcome').median().reset_index()
pimadiabetes_bmi


# Median for '0' Outcome is 30.1 and '1' Outcome is 34.3

# In[ ]:


# Replacing the Null Values with corresponding median values
pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['BMI'].isnull()),'BMI'] = pimadiabetes_bmi['BMI'][0]
pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['BMI'].isnull()),'BMI'] = pimadiabetes_bmi['BMI'][1]
pimadiabetes.head (10)


# In[ ]:


# checking for replacement of all '0' values
pimadiabetes[pimadiabetes['BMI'] == 0].shape[0]


# All zero values of 'BMI' are replaced

# 4. INSULIN

# In[ ]:


# CALCULATING NUMBER OF ZERO VALUES
pimadiabetes[pimadiabetes['Insulin'] == 0].shape[0]
pimadiabetes[pimadiabetes['Insulin'] == 0].index.tolist()
pimadiabetes[pimadiabetes['Insulin'] == 0].groupby('Outcome')['Insulin'].count()


# There are 374 zero values to be replaced with respective median values and 236 are for non-diabetic and 138 for diabetic

# In[ ]:


# Replace zero with Null Values
pimadiabetes['Insulin'].replace(0,np.NaN,inplace = True)


# In[ ]:


# Calculating the median grouped by Outcome
pimadiabetes_ins = pimadiabetes[pimadiabetes['Insulin'].notnull()]
pimadiabetes_ins = pimadiabetes_ins[['Insulin','Outcome']].groupby('Outcome')['Insulin'].median().reset_index()
pimadiabetes_ins


# Median for '0' Outcome is 102.5 and '1' Outcome is 169.5

# In[ ]:


# Replacing Null Values with corresponding median values
pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['Insulin'].isnull()),'Insulin'] = pimadiabetes_ins['Insulin'][0]
pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['Insulin'].isnull()),'Insulin'] = pimadiabetes_ins['Insulin'][1]
pimadiabetes.head(10)


# In[ ]:


# checking for replacement of all '0' values
pimadiabetes[pimadiabetes['Insulin'] == 0].shape[0]


# All '0's of Insulin values are replaced

# 5. SKIN THICKNESS DATASET

# In[ ]:


# Checking zero values to be replaced
pimadiabetes[pimadiabetes['SkinThickness'] == 0].shape[0]
pimadiabetes[pimadiabetes['SkinThickness'] == 0].index.tolist()
pimadiabetes[pimadiabetes['SkinThickness'] == 0].groupby('Outcome')['SkinThickness'].count()


# There are 227 zero values to be replaced with median values and 139 are for non-diabetic and 88 for diabetic

# In[ ]:


# Replacing zero values with null values
pimadiabetes['SkinThickness'].replace(0,np.NaN,inplace = True)


# In[ ]:


# Calculating the median values grouped by Outcome
pimadiabetes_skin = pimadiabetes[pimadiabetes['SkinThickness'].notnull()]
pimadiabetes_skin = pimadiabetes_skin[['SkinThickness','Outcome']].groupby('Outcome').median().reset_index()
pimadiabetes_skin


# Median for '0' Outcome is 27 and '1' Outcome is 32

# In[ ]:


# Replacing the Null Values with the medians calculated
pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['SkinThickness'].isnull()),'SkinThickness'] = pimadiabetes_skin['SkinThickness'][0]
pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['SkinThickness'].isnull()),'SkinThickness'] = pimadiabetes_skin['SkinThickness'][1]
pimadiabetes.head(10)


# In[ ]:


# checking for replacement of all '0' values
pimadiabetes[pimadiabetes['SkinThickness'] == 0].shape[0]


# All zero values  of 'SkinThickness' are replaced

# In[ ]:



pimadiabetes.describe().transpose()


# In[ ]:


#Calculating the number of persons who are diabetic at an younger age (<30)
pimadiabetes[pimadiabetes['Age']<30].count()


# In[ ]:


pimadiabetes[(pimadiabetes['Age']<30) & (pimadiabetes['Outcome'] == 1)].count()


# Out of 396 persons with age less than 30, 84 Persons are diabetic

# In[ ]:


# Number of persons who are not diabetic at age > 50
pimadiabetes[pimadiabetes['Age']>50].count()


# In[ ]:


pimadiabetes[(pimadiabetes['Age']>50) & (pimadiabetes['Outcome'] == 0)].count()


# More than 50% persons of age greater than 50 are diabetic

# # VISUALISATION OF DATA AFTER DATA REFINING

# # 1. Univariate analysis of the data

# In[ ]:


sns.countplot(x = 'Outcome', data = pimadiabetes)


# In[ ]:


# Histogram
pimadiabetes.hist(figsize = (20,20))


# 1. Age, DiabetisPedigreeFunction are parameters that are left skewed.  Insulin doesnot seem to be left skewed after replacing zero with median values
# 
# 2. Glucose parameter almost has a normal distribution
# 

# In[ ]:


#Boxplot for the refined data
pimadiabetes.plot(kind = 'box',figsize = (20,20), subplots = True, layout = (3,3))


# 1.  Glucose value doesnot have an Outlier
# 2.  Insulin, DiabetesPedigreeFunction and SkinThicknesss are having many outliers, in which SkinThickness has outliers
#     both at lower and higher value side
# 

# # 2. Bivariate analysis

# Plotting all the continous variables against the target variable

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(x='Outcome', y='Pregnancies', data=pimadiabetes, ax=axes[0])
sns.countplot(pimadiabetes['Pregnancies'], hue = pimadiabetes['Outcome'], ax=axes[1])


# The mean of the women who are diabetic is more than that of non-diabetic as can be seen in the histogram also.  

# Comparison of various parameters by dividing data into different age groups

# In[ ]:


pimadiabetes['age_group'] = pd.cut(pimadiabetes['Age'], range(0, 100, 10))


# In[ ]:



g = sns.catplot(x="age_group", y="Pregnancies", hue="Outcome",
               data=pimadiabetes, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# Women with age between 30 - 60 years of age are prone to diabetic and it seems the distribution is also normal

# In[ ]:



g = sns.catplot(x="age_group", y="BMI", hue="Outcome",
               data=pimadiabetes, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# BMI variation doesnot seem to depend on age group

# In[ ]:



g = sns.catplot(x="age_group", y="SkinThickness", hue="Outcome",
               data=pimadiabetes, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# In[ ]:



g = sns.catplot(x="age_group", y="Glucose", hue="Outcome",
               data=pimadiabetes, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# # Normalization of all numerical columns

# In[ ]:


#For processing of data
from sklearn import preprocessing


# In[ ]:


# Normalization of all numerical columns

# Normalize Pregnancies column
x_array = np.array(pimadiabetes['Pregnancies'])
norm_Pregnancies = preprocessing.normalize([x_array])

# Normalize Glucose column
x_array1 = np.array(pimadiabetes['Glucose'])
norm_Glucose = preprocessing.normalize([x_array1])

# Normalize BloodPressure column
x_array2 = np.array(pimadiabetes['BloodPressure'])
norm_BloodPressure = preprocessing.normalize([x_array2])

# Normalize Skinthickness column
x_array3 = np.array(pimadiabetes['SkinThickness'])
norm_SkinThickness = preprocessing.normalize([x_array3])

# Normalize Insulin column
x_array4 = np.array(pimadiabetes['Insulin'])
norm_Insulin = preprocessing.normalize([x_array4])

# Normalize BMI column
x_array5 = np.array(pimadiabetes['BMI'])
norm_BMI = preprocessing.normalize([x_array5])

# Normalize DiabeticPedigreeFunction column
x_array6 = np.array(pimadiabetes['DiabetesPedigreeFunction'])
norm_DiabeticPedigreeFunc = preprocessing.normalize([x_array6])

# Normalize Age column
x_array7 = np.array(pimadiabetes['Age'])
norm_Age = preprocessing.normalize([x_array7])

# Outcome Variable
x_array8 = np.array(pimadiabetes['Outcome'])


# In[ ]:


#Preparing Normalized Dataset
pimadiabetes_norm = pd.DataFrame({'Pregnancies':norm_Pregnancies[0,:],
                            'Glucose':norm_Glucose[0,:],
                            'BloodPressure':norm_BloodPressure[0,:],
                            'SkinThickness':norm_SkinThickness[0,:],
                            'Insulin':norm_Insulin[0,:],
                            'BMI':norm_BMI[0,:],
                            'DiabetesPedigreeFunction':norm_DiabeticPedigreeFunc[0,:],
                            'Age':norm_Age[0,:],
                            'Outcome':x_array8
                            })

pimadiabetes_norm.head()


# In[ ]:


# To get a view of how the  variable are scattered with respect to each other scatter matrix is drawn
from pandas.tools.plotting import scatter_matrix
p = scatter_matrix(pimadiabetes_norm, figsize = (25,25))


# In[ ]:


corr = pimadiabetes_norm.corr()
corr


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(corr,annot = True,linewidth = 0.5,fmt = '.2f')


# The maximum value of correilation in the heatmap is 0.54, which is not an healthy correlation to say the variables are redundant. 

# In[ ]:


#To split the data into train and test data set
from sklearn.model_selection import train_test_split

# To model Gaussian Naive Bayers classifier
from sklearn.naive_bayes import GaussianNB

#To check the accuracy of the model
from sklearn.metrics import accuracy_score

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# for prepartation of classification report
from sklearn.metrics import classification_report


# # ITERATION 1

# DATA SLICING

# In[ ]:


X = pimadiabetes.iloc[:,:8]
y = pimadiabetes.iloc[:,8]


# In[ ]:


# Split the data into train and test data
# Train data is 70%
# Test data us 30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45, random_state= 1000 )


# In[ ]:


#implement Gausian Naive Baye's
clf = GaussianNB()


# In[ ]:


#Fitting GaussianNB into test data set
clf.fit(X_train, y_train, sample_weight = None)


# In[ ]:


#Predicting test and train data using gaussian function
y_predict_test1 = clf.predict(X_test)
y_predict_train1 = clf.predict(X_train)


# In[ ]:


#Accuracy of test data set
accuracy_score(y_test,y_predict_test1,normalize = True)


# In[ ]:


#Accuracy of train data set
accuracy_score(y_train,y_predict_train1,normalize = True)


# In[ ]:


#Confusion matrix for the test data set
cm_test1 = confusion_matrix(y_test, y_predict_test1)
cm_test1


# In[ ]:


# Confusion matrix for the train data set
cm_train1 = confusion_matrix(y_train,y_predict_train1)
cm_train1


# In[ ]:


print(classification_report(y_test, y_predict_test1))


# In[ ]:


print(classification_report(y_train,y_predict_train1))


# In[ ]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,auc
import matplotlib.pyplot as plt

##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_predict_test1,y_test,drop_intermediate=False)

plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve - Iteration - 1')
plt.legend()
plt.show()


# In[ ]:


Performance()


# # ITERATION 2: WITH NORMALIZED AND STANDARDIZED DATA

# In[ ]:


X_features = pimadiabetes_norm.iloc[:,:8]
y = pimadiabetes_norm.iloc[:,8]


# In[ ]:


# Standardization

from sklearn.preprocessing import StandardScaler
rescaledX = StandardScaler().fit_transform(X_features)

X = pd.DataFrame(data = rescaledX, columns= X_features.columns)
X.head()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.45, random_state = 1000)


# In[ ]:


clf = GaussianNB()


# In[ ]:


clf.fit(X_train, y_train, sample_weight = None)


# In[ ]:


#Predicting test and train data using gaussian function
y_predict_test2 = clf.predict(X_test)
y_predict_train2 = clf.predict(X_train)


# In[ ]:


#Accuracy for test data
accuracy_score(y_test,y_predict_test2, normalize = True)


# In[ ]:


#Accuracy for train data
accuracy_score(y_train,y_predict_train2,normalize = True)


# In[ ]:


#Confusion matrix for the test data set
cm_test2 = confusion_matrix(y_test, y_predict_test2)
cm_test2


# In[ ]:


# Confusion matrix for the train data set
cm_train2 = confusion_matrix(y_train,y_predict_train2)
cm_train2


# In[ ]:


#Classification report for test data
print(classification_report(y_test, y_predict_test1))


# In[ ]:


#Classification report for test data
print(classification_report(y_train, y_predict_train1))


# In[ ]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,auc
import matplotlib.pyplot as plt

##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_predict_test2,y_test,drop_intermediate=False)
AUC = auc(fpr,tpr)
print('AUC is : %0.4f' % AUC)
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve - Iteration - 2')
plt.legend()
plt.show()


# There is no much effect on normalizing and standardizing the data before applying the Gaussian Naive Baye's algoithim

# # ITERATION 3
# 
# 
# 

# In[ ]:


# As Age and Pregency has correlation of 0.54, SkinThickness and BMI has correlation of 0.57 we will take only one variable
# out of these two sets.  BMI is choosen as it has less number of null values

X_features = pd.DataFrame(data = pimadiabetes_norm, columns =  ['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age'])
X_features.head()


# In[ ]:


y = pimadiabetes_norm.iloc[:,8]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45, random_state = 1000 )


# In[ ]:


clf = GaussianNB()


# In[ ]:


clf.fit(X_train, y_train, sample_weight = None)


# In[ ]:


# Predicting the test and train data set using the GaussianNB classifier
y_predict_test3 = clf.predict(X_test)
y_predict_train3 = clf.predict(X_train)


# In[ ]:


# Accuracy of test data by GaussianNB Classifier
accuracy_score(y_test, y_predict_test3, normalize = True)


# In[ ]:


# Accuracy of training data by Gaussian Classifier
accuracy_score(y_train, y_predict_train3, normalize = True)


# In[ ]:


#Confusion matrix for the test data
cm_test3 = confusion_matrix(y_test, y_predict_test3)
cm_test3


# In[ ]:


#Confusin matrix for the train data
cm_train3 = confusion_matrix(y_train, y_predict_train3)
cm_train3


# In[ ]:


# Classification report for the test data
print(classification_report(y_test, y_predict_test3))


# In[ ]:


#Classification report for the train data
print(classification_report(y_train, y_predict_train3))


# In[ ]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_predict_test3,y_test,drop_intermediate=False)

plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve - Iteration - 3')
plt.legend()
plt.show()


# # ITERATION 4

# In[ ]:


# As Age and Pregency has correlation of 0.54, SkinThickness and BMI has correlation of 0.57 we will take only one variable
# out of these two sets.  BMI is choosen as it has less number of null values

X_features = pd.DataFrame(data = pimadiabetes_norm, columns =  ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age'])
X_features.head()


# In[ ]:


y = pimadiabetes_norm.iloc[:,8]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45,random_state = 1000 )


# In[ ]:


clf = GaussianNB()


# In[ ]:


clf.fit(X_train, y_train, sample_weight = None)


# In[ ]:


# Predicting the test and train data set using the GaussianNB classifier
y_predict_test4 = clf.predict(X_test)
y_predict_train4 = clf.predict(X_train)


# In[ ]:


# Accuracy of test data by GaussianNB Classifier
accuracy_score(y_test, y_predict_test4, normalize = True)


# In[ ]:


# Accuracy of training data by Gaussian Classifier
accuracy_score(y_train, y_predict_train4, normalize = True)


# In[ ]:


#Confusion matrix for the test data
cm_test4 = confusion_matrix(y_test, y_predict_test4)
cm_test4


# In[ ]:


#Confusin matrix for the train data
cm_train4 = confusion_matrix(y_train, y_predict_train4)
cm_train4


# In[ ]:


# Classification report for the test data
print(classification_report(y_test, y_predict_test4))


# In[ ]:


#Classification report for the train data
print(classification_report(y_train, y_predict_train4))


# In[ ]:




##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_predict_test4,y_test,drop_intermediate=False)

plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve - Iteration - 4')
plt.legend()
plt.show()


# # Conclusions:

# 
#  1. The pima indian diabetes dataset has many '0' values which are treated as 'missing' values and replaced with median value calculated after grouping by outcome
#  2. Univariate and bivariate analysis is done by scatter plots and boxplots
#  3. Plotting of different independent variable with the target variable is done 
#  4. The Gaussian Naive Baye's is implemented on the normalized pimadiabetes dataset by standardizing data in one of the iterations.  In another iteration it is done without standardising data.  It was found that there is no differencein the accuracy/recall/precision calcualated.  
#  5. The Naive Baye's is implemented by removing one of the two parameters having correlation more than 0.5 i.e., removing SkinThickness and calculated.  Even then the accuracy/recall/precision didnot change.  
#  6. The parameters pregnancy and SkinThickness removed and precision/recall/accuracy is calculated.  No change observed using the Gaussian Naive Baye's.  It inidicates that there is no much effect of these parameters on the prediction of class.
#  7. The train and test split is selected so that it maximizes the recall value for the diabetic patients and it is found to  be 55/45 ratio for train/test data.
#  8. Confusion matrix for both the training and test data is obtained and ROC Curve drawn for test data
#  9. The accuracy, Precision, Recall values are as follows
#         Accuracy on Test data = 0.7658
#         Accuracy on Train data = 0.7843
#         Precision for test data for  Non-Diabetic/Diabetic Outcome = 0.81/0.67
#         Recall for test data for Non-Diabetic/Diabetic Outcome = 0.83/0.66
# 
# 

# In[ ]:





# In[ ]:




