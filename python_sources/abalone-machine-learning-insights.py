#!/usr/bin/env python
# coding: utf-8

# ######  Abalone Data-set - > Extension of EDA Kernel by [Rageeni Sah](https://www.kaggle.com/ragnisah/eda-abalone-age-prediction)
# 
# - Model Insights 
# - Different Classification Algorithms used,
# - Work Done by [Sriram Arvind Lakshmanakumar](https://www.kaggle.com/sriram1204), [Nikhita Agarwal](https://www.kaggle.com/nikhitaagr)

# In[ ]:


''' Library Import'''
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


''' SK-Learn Library Import'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLasso,LassoLarsCV
from sklearn.exceptions import ConvergenceWarning 
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
import sklearn.datasets 


# In[ ]:


'''Scipy, Stats Library'''
from scipy.stats import skew


# In[ ]:


''' To Ignore Warning'''
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


''' To Do : Inline Priting of Visualizations '''
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


''' Importing Data : from the Archive Directly'''
df = pd.read_csv(r'../input/abalone.csv')


# In[ ]:


'''Display The head --> To Check if Data is Properly Imported'''
df.head()


# In[ ]:


''' Feature Information of the DataSet'''
df.info()


# ##### According to the Infomation: 
# 
# - 1)No-Null data
# - 2)1 - Object Type
# - 3)7 - Float Type
# - 4)1 - Int Type

# In[ ]:


'''Feature Distirbution of data for Float and Int Data Type'''
df.describe()


# ###### According to Described Information: 
# 
# - 1)No Feature has Minimum Value = 0, except *Height*
# - 2)All Features are not Normally Distributed, ( Theortically if feature is normally distributed, Mean = Median = Mode ).
# - 3)But Features are close to Normality
# - 4)All numerical, Except Sex
# - 5)Each Feature has Different Scale

# In[ ]:


'''Numerical Features and Categorical Features'''
nf = df.select_dtypes(include=[np.number]).columns
cf = df.select_dtypes(include=[np.object]).columns


# In[ ]:


'''List of Numerical Features'''
nf


# In[ ]:


''' List of Categorical Features'''
cf


# In[ ]:


'''Histogram : to see the numeric data distribution'''
df.hist(figsize=(20,20), grid = True, layout = (2,4), bins = 30)


# In[ ]:


'''After Seeing Above Graph of Data Distribution, I feel the Data is skewed, So checking for Skewness '''
skew_list = skew(df[nf],nan_policy='omit') #sending all numericalfeatures and omitting nan values
skew_list_df = pd.concat([pd.DataFrame(nf,columns=['Features']),pd.DataFrame(skew_list,columns=['Skewness'])],axis = 1)


# In[ ]:


skew_list_df.sort_values(by='Skewness', ascending = False)


# ###### According to the rules
# - For a normally Distributed Data, Skewness should be greater than 0
# - Skewness > 0 , More weight is on the right tail of the distribution
# 

# In[ ]:


'''Missing Values '''
mv_df = df.isnull().sum().sort_values(ascending = False)
pmv_df = (mv_df/len(df)) * 100
missing_df = pd.concat([mv_df,pmv_df], axis = 1, keys = ['Missing Values','% Missing'])


# In[ ]:


missing_df


# In[ ]:


'''Target Column Analysis'''
print("Value Count of Rings Column")
print(df.Rings.value_counts())
print("\nPercentage of Rings Column")
print(df.Rings.value_counts(normalize = True))


# ###### No of Classes In Target

# In[ ]:


print(len(df.Rings.unique()))


# ### Visualization

# In[ ]:


'''Sex Count of Abalone, M - Male, F - Female, I - Infant'''
sns.countplot(x='Sex', data = df)


# In[ ]:


'''Sex Ratio in Abalone'''
print("\nSex Count in Percentage")
print(df.Sex.value_counts(normalize = True))
print("\nSex Count in Numbers")
print(df.Sex.value_counts())


# In[ ]:


'''Small Feature Engineering, Deriving Age from Rings Column, Age = Rings + 1.5'''
df['Age'] = df['Rings'] + 1.5
df['Age'].head(5)


# In[ ]:


'''Sex and Age Visulization'''
plt.figure(figsize = (20,7))
sns.swarmplot(x = 'Sex', y = 'Age', data = df, hue = 'Sex')
sns.violinplot(x = 'Sex', y = 'Age', data = df)


# ###### According to The above Graph
# - Male : Majority Between 7.5 to 19
# - Female : Majority Between 8 to 19
# - Infant : Majority Between 6 to < 10

# In[ ]:


df.groupby('Sex')[['Length', 'Diameter', 'Height', 'Whole weight', 
                   'Shucked weight','Viscera weight', 'Shell weight', 'Age']].mean().sort_values(by = 'Age',ascending = False)


#    ###### Preprocessing Data for the Model

# In[ ]:


'''LabelEnconding the Categorical Data'''
df['Sex'] = LabelEncoder().fit_transform(df['Sex'].tolist())


# In[ ]:


'''One Hot Encoding for Sex Feature '''
transformed_sex_feature = OneHotEncoder().fit_transform(df['Sex'].values.reshape(-1,1)).toarray()
df_sex_encoded = pd.DataFrame(transformed_sex_feature, columns = ["Sex_"+str(int(i)) for i in range(transformed_sex_feature.shape[1])])
df = pd.concat([df, df_sex_encoded], axis=1)


# In[ ]:


df.head()


# ###### Data Splitting for Model
# - Learning Features
# - Predicting Feature
# - Train & Test Split

# In[ ]:


'''Learning Features and Predicting Features'''
Xtrain = df.drop(['Rings','Age','Sex'], axis = 1)
Ytrain = df['Rings']


# In[ ]:


'''Train Test Split , 70:30 Ratio'''
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)


# ###### Simple Logistic Regression Model
# No of Classes : 28

# In[ ]:


'''Creating Object of LogisticRegression'''
logreg = LogisticRegression()
'''Learning from Training Set'''
logreg.fit(X_train, Y_train)
'''Predicting for Training Set'''
Y_pred = logreg.predict(X_test)
'''Accuracy Score'''
result_acc = accuracy_score(Y_test,Y_pred) 


# In[ ]:


'''For Both, LabelEncoding and OneHotEncoding -> The accuracy is 25 %'''
result_acc


# ###### Simple Logistic Regression Model
# 
# - No of Classes : 2
# - 1 - Rings > 10
# - 0 - Rings <= 10 

# In[ ]:


'''Creating New Target Variable '''
df['newRings'] = np.where(df['Rings'] > 10,1,0)


# In[ ]:


'''Learning Features and Predicting Features'''
Xtrain = df.drop(['newRings','Rings','Age','Sex'], axis = 1)
Ytrain = df['newRings']


# In[ ]:


'''Train Test Split , 70:30 Ratio'''
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)


# In[ ]:


'''Creating Object of LogisticRegression'''
logreg = LogisticRegression()
'''Learning from Training Set'''
logreg.fit(X_train, Y_train)
'''Predicting for Training Set'''
Y_pred = logreg.predict(X_test)
'''Accuracy Score'''
result_acc = accuracy_score(Y_test,Y_pred) 


# In[ ]:


result_acc


# ##### Note : If you have Binary Classification, Logistic Regression is able to Boost to Higher Accuracy
# 

# ##### So to Handle Multi-Class Classification, We can Try SVM Model, as it works well for multi-class and multi-label Classification

# ###### Multi-Class Classification : When you have one target Column with 3 or more discreet values to predict, you state the problem as multi-class classification.

# ###### We WIll first try with all the 28 classes in the target column, using linear kernel , Regularization parameter value as 1, and gamma 1

# In[ ]:


'''Importing SVM from SK-Learn'''
from sklearn import svm


# In[ ]:


'''Learning Features and Predicting Features'''
Xtrain = df.drop(['Rings','Age','Sex'], axis = 1)
Ytrain = df['Rings']


# In[ ]:


'''Train Test Split , 70:30 Ratio'''
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)


# In[ ]:


'''Creating Object of SVM'''
svmModel = svm.SVC(kernel='linear', C=1, gamma=1) 
'''Learning from Training Set'''
svmModel.fit(X_train, Y_train)
'''Predicting for Training Set'''
Y_pred = svmModel.predict(X_test)
'''Accuracy Score'''
result_acc = accuracy_score(Y_test,Y_pred) 


# In[ ]:


result_acc


# ###### We can see, the Model Accuracy has increased with SVM, it is now 37 percent.
# - Lets Try to tweak the model Learning Process and see if the accuracy is increases or not.

# In[ ]:


'''Creating Object of SVM'''
svmModel = svm.SVC(kernel='rbf', C=1, gamma=100) 
'''Learning from Training Set'''
svmModel.fit(X_train, Y_train)
'''Predicting for Training Set'''
Y_pred = svmModel.predict(X_test)
'''Accuracy Score'''
result_acc = accuracy_score(Y_test,Y_pred) 


# In[ ]:


result_acc


# ###### We can see, the Model Accuracy has increased with Tweaking SVM parameters, it is now 38 percent.
# - Lets Try to reduce the number of classes and see how the model is performing

# In[ ]:


'''Making a Copy of the primary DataSet'''
new_df = df.copy()


# In[ ]:


'''Feature Engineering , class 1 - 1-8, class 2 - 9-8, class 3 - 11 >'''
new_df['newRings_1'] = np.where(df['Rings'] <= 8,1,0)
new_df['newRings_2'] = np.where(((df['Rings'] > 8) & (df['Rings'] <= 10)), 2,0)
new_df['newRings_3'] = np.where(df['Rings'] > 10,3,0)


# In[ ]:


new_df['newRings'] = new_df['newRings_1'] + new_df['newRings_2'] + new_df['newRings_3']


# In[ ]:


'''Learning Features and Predicting Features'''
Xtrain = new_df.drop(['Rings','Age','Sex','newRings_1','newRings_2','newRings_3'], axis = 1)
Ytrain = new_df['newRings']


# In[ ]:


'''Train Test Split , 70:30 Ratio'''
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)


# In[ ]:


'''Creating Object of SVM'''
svmModel = svm.SVC(kernel='rbf', C=1, gamma=100) 
'''Learning from Training Set'''
svmModel.fit(X_train, Y_train)
'''Predicting for Training Set'''
Y_pred = svmModel.predict(X_test)
'''Accuracy Score'''
result_acc = accuracy_score(Y_test,Y_pred) 


# In[ ]:


result_acc


# ###### Final Conclusion : we have not removed Outliers ( as we ad to capture all the type of different shapes and weights of abalone ), But with Less number of classes, SVM is giving an accuracy of 98% ( not Fully Tested ). 

# ###### References Used:
# 
# - SVM -> https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
# - Abalone DataSet -> https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/
# - Logistic Regression -> Andrews Ng
# 
# ###### Things To DO: 
# - Outlier Data Handling ( to be kept or Removed)
# - More Visulization for Outlier Data
# - Model Output VIsualization
# - More Tweaks on C and Gamma Parameter
