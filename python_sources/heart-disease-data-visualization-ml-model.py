#!/usr/bin/env python
# coding: utf-8

# # $$Introduction$$
# 
# ### Heart Disease data analysis & Machine Learning model implementation
# 
# ## Objective
# I develop a Machine Learning Model which is predict a person is affected or not by **"HEART DISEASES"** based on previous heart disease data.
# 
# ###  Data Information
# - Number of rows 303
# - Number of columns 14
# 
#  - Age(int)
#     - The maximum Value is 77 
#     - The minimum Value is 29 
#     - The number of unique Values is 41 
#     
#   - Sex(int)
#     - The maximum Value is 1 
#     - The minimum Value is 0 
#     - The number of unique Values is 2 
#     - The nunique Values is [0 1]
#        - Female
#        - male
#     
#   - Cp(int) = chest pain type
#      - The maximum Value is 3 
#      - The minimum Value is 0 
#      - The number of unique Values is 4 
#      - The nunique Values is [0 1 2 3]
#         - typical angina
#         - atypical angina
#         - non-anginal pain
#         - asymptomatic
#      
#   - trestbps(int) = resting blood pressure (in mm Hg on admission to the hospital)
#     - The maximum Value is 200 
#     - The minimum Value is 94 
#     - The number of unique Values is 49 
#     
#   - Chol(int) = serum cholestoral in mg/dl
#     - The maximum Value is 564 
#     - The minimum Value is 126 
#     - The number of unique Values is 152 
#     
#   - Fbs(int) = fasting blood sugar > 120 mg/dl
#     - The maximum Value is 1 
#     - The minimum Value is 0 
#     - The number of unique Values is 2 
#     - The nunique Values is [0 1]
#       - False
#       - True
#     
#   - Restecg(int) = restecg: resting electrocardiographic results
#      - The maximum Value is 2 
#      - The minimum Value is 0 
#      - The number of unique Values is 3 
#      - The nunique Values is [0 1 2]
#        - normal 
#        - having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >0.05  mV      
#        - showing probable or definite left ventricular hypertrophy by Estes' criteria]
#     
#   - thalach(int) = maximum heart rate achieved
#     - The maximum Value is 202 
#     - The minimum Value is 71 
#     - The number of unique Values is 91
#     
#   - exang(int) = exercise induced angina
#     - The maximum Value is 1 
#     - The minimum Value is 0 
#     - The number of unique Values is 2 
#     - The nunique Values is [0 1] 
#        - No
#        - Yes
#     
#   - oldpeak(float) = ST depression induced by exercise relative to rest
#     - The maximum Value is 6.2 
#     - The minimum Value is 0.0 
#     - The number of unique Values is 40
#     
#   - slope(int) = the slope of the peak exercise ST segment
#     - The maximum Value is 2 
#     - The minimum Value is 0 
#     - The number of unique Values is 3 
#     - The nunique Values is [0 1 2]
#        - upsloping
#        - flat
#        - downsloping
#      
#   - Ca(int) = number of major vessels (0-3) colored by flourosopy
#     - The maximum Value is 4 
#     - The minimum Value is 0 
#     - The number of unique Values is 5 
#     - The nunique Values is [0 2 1 3 4] 
#     
#   - Thal(int)
#     - The maximum Value is 3 
#     - The minimum Value is 0 
#     - The number of unique Values is 4 
#     - The nunique Values is [0 1 2 3]
#        - normal
#        - fixed
#        - defect
#        - reversable
#     
#   - Target(int)
#     - The maximum Value is 1 
#     - The minimum Value is 0 
#     - The number of unique Values is 2 
#     - The nunique Values is [0 1]
#        - Not disease
#        - disease

# ## $$Exploratory$$ $$Data$$ $$Analysis$$

# ## Necessery Liberary Import

# In[ ]:


import pandas as pd  # Load data
import numpy as np # Scientific Computing
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns  # Data Visualization
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix, classification_report
import missingno as msno  # showing null values by bar graph
import warnings  # Ignore Warnings
warnings.filterwarnings("ignore")
sns.set()   # Set Graphs Background


# ## Data Load

# In[ ]:


data = pd.read_csv('../input/heartdata1/heart.csv')
data.head()


# ## Data Information
# - There are 303 rows
# - There are 14 columns
# - No Null Values Present
# - Only oldpeak is float data type otherwise all is int data type

# In[ ]:


# shape showing how many data rows & columns have
data.shape


# In[ ]:


# info() showing rows & columns number column name data type Non-null count
data.info()


# ## Mising Value Checking
# - I use five function for mising value checking
#    - insull()
#       - If any value is null return True
#       - Otherewise return False
#       
#    - isnull().any()
#       - If any columns have null value return True
#       - Otherewise return False
#       
#    - isnull().sum()
#       - If any columns have null value return how many null values have
#       - If no null value present return 0
#       
#    - missingno()
#       - Showing values by bar graph
#       
#    - Heatmap()
#       - Showing values by graph

# In[ ]:


# isnull() check null value
data.isnull()


# In[ ]:


# any() check null values by columns
data.isnull().any()


# - All columns return False
# - The dataset no null value present

# In[ ]:


# innull().sum() show total null values 
data.isnull().sum()


# - All columns return 0
# - The dataset no null value present

# In[ ]:


# missingno() showing null values by bar graph
msno.bar(data, figsize=(12,6))
plt.show()


# - Total rows have 303 & every bar is full
# - There have no null values present

# In[ ]:


# heatmap() showing null values
sns.heatmap(data.isnull(), yticklabels=False,cbar=False, cmap='viridis')
plt.show()


# - Graph is full clean
# - There have no null values

# ## Statistical Information

# In[ ]:


# describe() statistical information
data.describe()


# - Values should be closer each other.
# - There is no incompatible values

# ## Histogram 

# In[ ]:


# hist() histogram 
data.hist(figsize = (15,12))
plt.show()


# ## Sex Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.sex.value_counts())
sns.countplot(x='sex', data=data)
plt.show()


# - There are 207 male
# - There are 96 female

# ## Ca Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.ca.value_counts())
sns.countplot(x='ca', data=data)
plt.show()


# ## Fbs Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.fbs.value_counts())
sns.countplot(x='fbs', data=data)
plt.show()


# - There are 258 False values
# - There are 45 True values

# ## Cp Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.cp.value_counts())
sns.countplot(x='cp', data=data)
plt.show()


# - typical angina 143
# - atypical angina 50
# - non-anginal pain 87
# - asymptomatic 23

# ## Exang Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.exang.value_counts())
sns.countplot(x='exang', data=data)
plt.show()


# - Exang No 204
# - Exang Yes 99

# ## Restecg Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.restecg.value_counts())
sns.countplot(x='restecg', data=data)
plt.show()


# - normal 147
# - having ST-T wave abnormality 152
# - left ventricular hypertrophy by Estes' criteria 4

# ## Slope Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.slope.value_counts())
sns.countplot(x='slope', data=data)
plt.show()


# - upsloping 21
# - flat 140
# - downsloping 142

# ## Thal Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.thal.value_counts())
sns.countplot(x='thal', data=data)
plt.show()


# - normal 2
# - fixed 18
# - defect 166
# - reversable 117

# ## Target Unique Value Counts & Plot

# In[ ]:


# value_counts() total unique value count
print(data.target.value_counts())
sns.countplot(x='target', data=data)
plt.show()


# - Yes 165
# - No 138

# ## Age VS Chol Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['age'],data['chol'])
plt.title('Age VS Chol', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Chol', fontsize=20)
plt.show()


# - There are some value outlier

# ## Age VS Trestbps Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['age'],data['trestbps'])
plt.title('Age VS Trestbps', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Trestbps', fontsize=20)
plt.show()


# - There are some value outlier

# ## Age VS Thalach Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['age'],data['thalach'])
plt.title('Age VS Thalach', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Thalach', fontsize=20)
plt.show()


# - There are some value outlier

# ## Trestbps VS Chol Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['trestbps'],data['chol'])
plt.title('Trestbps VS Chol', fontsize=20)
plt.xlabel('Trestbps', fontsize=20)
plt.ylabel('Chol', fontsize=20)
plt.show()


# - There are some value outlier

# ## Trestbps VS Thalach Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['trestbps'],data['thalach'])
plt.title('Trestbps VS Thalach', fontsize=20)
plt.xlabel('Trestbps', fontsize=20)
plt.ylabel('Thalach', fontsize=20)
plt.show()


# - There are some value outlier

# ## Chol VS Thalach Scatter Plot

# In[ ]:


# scatter() relation between two columns
plt.figure(figsize=(10,8))
plt.scatter(data['chol'],data['thalach'])
plt.title('Chol VS Thalach', fontsize=20)
plt.xlabel('Chol', fontsize=20)
plt.ylabel('Thalach', fontsize=20)
plt.show()


# - There are some value outlier

# ## Box Plot

# In[ ]:


# boxplot() showing outlier
box = data[['age','trestbps','chol','thalach']]
plt.figure(figsize=(12,8))
sns.boxplot(data=box)
plt.show()


# - Now data has outlier
# - Trestbps , Chol ,Thalach Some outlier present

# ### Outlier Remove

# In[ ]:


data_iqr = box
Q1 = data_iqr.quantile(0.25)
Q3 = data_iqr.quantile(0.75)
iqr = Q3 - Q1

data_iqr_clean = data_iqr[~((data_iqr < (Q1 - 1.5*iqr)) | (data_iqr > (Q3 + 1.5*iqr))).any(axis=1)]


# In[ ]:


# boxplot() showing outlier
box = data_iqr_clean[['age','trestbps','chol','thalach']]
plt.figure(figsize=(12,8))
sns.boxplot(data=box)
plt.show()


# - Now data has no outlier

# ## Distibution Plot

# In[ ]:


# distplot() same as histogram
fig, ax = plt.subplots(2,2, figsize=(10,8))
sns.distplot(data_iqr_clean.age, bins = 20, ax=ax[0,0]) 
sns.distplot(data_iqr_clean.trestbps, bins = 20, ax=ax[0,1]) 
sns.distplot(data_iqr_clean.chol, bins = 20, ax=ax[1,0])
sns.distplot(data_iqr_clean.thalach, bins = 20, ax=ax[1,1])
plt.show()


# ## Heatmap

# In[ ]:


# corr() relation with data
corr=data.corr()

plt.figure(figsize=(14,8))

sns.heatmap(corr, vmax=.8, linewidths=0.01,annot=True,cmap='summer',linecolor="black")
plt.title('Correlation between features')
plt.show()


# ## Machine Learning Model Implementation

# In[ ]:


x = data.iloc[:,0:-1].values # All rows & columns present except Target column
y = data.iloc[:,-1].values # Only target column present


# ## Data Split
# - train_test_split() use for data divided
# - test size use for ratio split
# - random_state is random seed

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=4)


# ## After split data shape

# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Support vector machine object create
# - fit() tarin the model

# In[ ]:


svm = SVC(kernel='rbf',random_state=0)
svm.fit(x_train,y_train)


# In[ ]:


svm.score(x_test,y_test)


# - svm score 67.21%

# ## Use StandardScaler
# - StandardScaler() use for value transform
# - fit_transform() use for train data transform
# - transform() use for test data transform

# In[ ]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


svm = SVC(kernel='rbf',random_state=0,probability=True) #probability for predict_proba
svm.fit(x_train,y_train)


# In[ ]:


svm.score(x_test,y_test)


# - After StandardScaler the score is 86.88%

# ## Hyperparameter Tuning

# In[ ]:


param_grid = {'C':[1,10,100,200],
              'kernel':['rbf','poly','linear','sigmoid'],
              'degree':[1,2,4,6],
              'gamma':[0.01,0.1,0.5,1]}

grid=GridSearchCV(SVC(), param_grid=param_grid, cv=4)
grid.fit(x_train,y_train)

y_pred = grid.predict(x_test)

print("Accuracy: {}".format(grid.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(grid.best_params_))


# In[ ]:


svm = SVC(C=1,kernel='poly',degree=1,gamma=0.5,probability=True)
svm.fit(x_train,y_train)


# In[ ]:


svm.score(x_test,y_test)


# - After Hyperparameter Tuning use accuracy is 90.16%

# ## Confusion Matrix

# In[ ]:


y_pred = svm.predict(x_test)
cm = confusion_matrix(y_pred,y_test)
print('Confusion Matrix \n',cm)


# - False Positive 21
# - False Negative 2
# - True Negative 4
# - True Positive 34

# ## Confusion Matrix Heatmap

# In[ ]:


plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()


# ## Classification Report

# In[ ]:


cr = classification_report(y_pred,y_test)
print('Classification Report\n',cr)


# ## ROC Curve with Support Vector Machine

# In[ ]:


y_prob = svm.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test,y_prob)
Auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr,linestyle='-',label='(auc=%0.3f)' %Auc)
plt.plot([0,1],[0,1])
plt.title('ROC CURVE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# ## Model complexity

# In[ ]:


num = np.arange(1, 30)
train_accuracy = []
test_accuracy = []
for i, k in enumerate(num):
    svm = SVC(C=k)
    svm.fit(x_train,y_train)
    train_accuracy.append(svm.score(x_train, y_train))
    test_accuracy.append(svm.score(x_test, y_test))

# Plot
plt.figure(figsize=(10,6))
plt.plot(num, test_accuracy, label = 'Testing Accuracy')
plt.plot(num, train_accuracy, label = 'Training Accuracy')
plt.legend(loc=10)
plt.title('value VS Accuracy')
plt.xlabel('Number of C')
plt.ylabel('Accuracy')
plt.xticks(num)
plt.show()
print("Best accuracy is {} with C = {}".format(np.max(test_accuracy),
                                               1+test_accuracy.index(np.max(test_accuracy))))


# ## Get Dummies Use

# In[ ]:


data1 = data.copy() # copy data
data1.head()


# ## Thal column get dummies

# In[ ]:


data1_thal = pd.get_dummies(data['thal'],prefix='thal')
data1_thal.head()


# ## Slope column get dummies

# In[ ]:


data1_slope = pd.get_dummies(data['slope'],prefix='slope')
data1_slope.head()


# ## Restecg column get dummies

# In[ ]:


data1_restecg = pd.get_dummies(data['restecg'],prefix='restecg')
data1_restecg.head()


# ## Cp column get dummies

# In[ ]:


data1_cp = pd.get_dummies(data['cp'],prefix='cp')
data1_cp.head()


# ## Ca column get dummies

# In[ ]:


data1_ca = pd.get_dummies(data['ca'],prefix='ca')
data1_ca.head()


# ## Get dummies data concat 

# In[ ]:


data2 = pd.concat([data1_cp,data1_restecg,data1_slope,data1_ca,data1_thal],axis='columns')


# In[ ]:


data2.head()


# ## Original & dummies data concat

# In[ ]:


data3 = pd.concat([data1,data2],axis='columns')
data3.head()


# ## Drop main dummies columns with target column

# In[ ]:


data3 = data3.drop(['cp','restecg','slope','thal','ca','target'], axis=1)
data3.head()


# ## concat target column with data3

# In[ ]:


data3 = pd.concat([data3,data.target],axis=1)
data3.head()


# ## Data3 separate into input columns & output column

# In[ ]:


x = data3.iloc[:,:-1].values # All rows & columns present except Target column
y = data3.iloc[:,-1].values # Only target column present


# ## separate data split 80:20

# In[ ]:


xx_train,xx_test,yy_train,yy_test = train_test_split(x,y, test_size=0.2, random_state=4)


# ## Apply SVC

# In[ ]:


svm = SVC(C=1,kernel='poly',degree=1,gamma=0.5,probability=True)
svm.fit(xx_train,yy_train)


# ## DecisionTreeClassifier

# In[ ]:


param_grid = {'ccp_alpha':[0.0,0.1,0.2,0.3,0.4,1],
              'criterion':['gini','entropy'],
              'max_depth':[5,10,50,100,200],
              'max_leaf_nodes':[5,10,50,100,200],
              'random_state':[2,5,10,20,42]}

grid=GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=4)
grid.fit(xx_train,yy_train)

print("Tuned Model Parameters: {}".format(grid.best_params_))


# In[ ]:


dtc = DecisionTreeClassifier(criterion='gini',max_depth=10,max_leaf_nodes=10,
                            ccp_alpha=0.0,random_state=2)
dtc.fit(xx_train,yy_train)


# ## LogisticRegression

# In[ ]:


param_grid = {'C':[1.0,2.0,5.0,10.0,20.0],
              'penalty':['l1','l2','none','elasticnet'],
              'max_iter':[50,100,200,300,500],
              'multi_class':['auto','ovr','multinomial']}

grid=GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=4)
grid.fit(xx_train,yy_train)

print("Tuned Model Parameters: {}".format(grid.best_params_))


# In[ ]:


lg = LogisticRegression(C=5.0,max_iter=50,multi_class='multinomial',penalty='l2')
lg.fit(xx_train,yy_train)


# ## RandomForestClassifier

# In[ ]:


param_grid = {'n_estimators':[50, 100,150,200,300],
              'criterion':['gini','entropy'],
              'max_depth':[5,10,50,100,200]}

grid=GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=4)
grid.fit(xx_train,yy_train)

print("Tuned Model Parameters: {}".format(grid.best_params_))


# In[ ]:


rfc = RandomForestClassifier(criterion='gini',max_depth=5,n_estimators=50)
rfc.fit(xx_train,yy_train)


# In[ ]:


print("Support Vector Machine Accuracy: {}".format(svm.score(xx_test, yy_test)))
print("DecisionTreeClassifier Accuracy: {}".format(dtc.score(xx_test, yy_test)))
print("LogisticRegression Accuracy: {}".format(lg.score(xx_test, yy_test)))
print("RandomForestClassifier Accuracy: {}".format(rfc.score(xx_test, yy_test)))


# - Support Vector Machine Accuracy: 91.80%
# - DecisionTreeClassifier Accuracy: 80.32%
# - LogisticRegression Accuracy: 88.52%
# - RandomForestClassifier Accuracy: 93.44%
# 
# - So we can decide that LogisticRegression is the best model for this dataset & its accuracy 93.44%

# In[ ]:




