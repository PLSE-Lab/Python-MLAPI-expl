#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Data = pd.read_excel("../input/UCI_Credit_Card.csv")


# In[ ]:


Data.head()


# In[ ]:


Data.columns
# We have to change the column name in Meaningful format.


# In[ ]:


Data.shape
Data = Data.drop('ID', axis=0) # Droping the ID column as it is not useful for us.


# In[ ]:


Data.describe()


# In[ ]:


# Check any number of columns with NaN
print(Data.isnull().any().sum(),'/', len(Data.columns))

# Check any number of data points with NaN
print(Data.isnull().any(axis=1).sum(), '/', len(Data))


# In[ ]:


# Renaming the columns name with meaningful format
Data = Data.rename(columns={'X1':"LIMIT_BAL",'X2':"SEX","X3":"EDUCATION","X4":"MARRIAGE",'X5':"AGE",'X6':"PAY_1","X7":"PAY_2","X8":"PAY_3",'X9':"PAY_4","X10":"PAY_5","X11":"PAY_6","X12":"bill_amt1","X13":"bill_amt2","X14":"bill_amt3","X15":"bill_amt4","X16":"bill_amt5","X17":"bill_amt6","X18":"PAY_AMT1","X19":"PAY_AMT2","X20":"PAY_AMT3","X21":"PAY_AMT4","X22":"PAY_AMT5","X23":"PAY_AMT6","Y":"default"})


# In[ ]:


Data.info()


# ## Wow!! I can see there are  no null values and I am going to chnage the variable type.

# In[ ]:


Data[['LIMIT_BAL','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6',
      'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','AGE']] = Data[['LIMIT_BAL','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','AGE']].apply(pd.to_numeric)


# In[ ]:


Data.info()


# In[ ]:


Data.describe()


# ## Now I am going to perfrom Univariate Analysis:

# In[ ]:


# Some features are in Upper case and some are lower case: gooing to convert all in lower case:
Data.columns = Data.columns.str.lower()


# In[ ]:



Data.columns # Now we can see all column are in same case i.e. Lower case


# In[ ]:


Sex_count = Data.sex.value_counts()
print(Sex_count)
Data.sex.value_counts().plot(kind = "bar")


# In[ ]:


Data["education"] = Data['education'].map({0: np.NaN, 1:1, 2:2, 3:3, 4:np.NaN,5: np.NaN, 6: np.NaN})


# In[ ]:


education_count = Data.education.value_counts()
print(education_count)
Data.education.value_counts().plot(kind = "bar")


# In[ ]:


Data['marriage'] = Data['marriage'].map({0:np.NaN, 1:1, 2:0, 3:np.NaN})


# In[ ]:


marriage_count = Data.marriage.value_counts()
print(marriage_count)
Data.marriage.value_counts().plot(kind= "bar")


# In[ ]:


Data.loc[Data["education"].isnull(), "education"] = Data["education"].mean()
Data.loc[Data["marriage"].isnull(), "marriage"] = Data["marriage"].mean()


# In[ ]:


Data.age.plot.hist()


# In[ ]:


# limit balance
fig = plt.figure()
fig.set_size_inches(10,5)
Data.limit_bal.plot.hist()


# In[ ]:


sns.distplot(Data["limit_bal"])


# In[ ]:


Payment_Type = Data.loc[:,"pay_1":"pay_6"]


# In[ ]:


sns.boxplot(Payment_Type)


# ## 	Analyze the trend on outstanding amount for the bank 
# 
# ## Overall outstanding amount trends
# 

# In[ ]:


Bill_amount = Data.loc[:,"bill_amt1":"bill_amt6"]


# In[ ]:


Bill_amount.describe()


# In[ ]:


sns.boxplot(Bill_amount)


# In[ ]:


sns.pairplot(Bill_amount)


# o	Number of customers with outstanding amount (in different outstanding amount buckets)

# In[ ]:


Payment_amount = Data.loc[:,"pay_amt1":"pay_amt6"]


# In[ ]:


Payment_amount.describe()


# In[ ]:


# Let's see how AGE compares across default and non-default observations.

print ("Default :")
print (Data.age[Data.default == 1].describe())
print ()
print ("NO default :")
print (Data.age[Data.default == 0].describe())


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



ax1.hist(Data.age[Data.default == 1], bins = 20)
ax1.set_title('Default')

ax2.hist(Data.age[Data.default == 0], bins = 20)
ax2.set_title('No Default')

plt.xlabel('Age')
plt.ylabel('Number of Observations')
plt.show()


# ## looks here are more defaults observed between the age of 25 and 35 

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))



ax1.hist(Data.limit_bal[Data.default == 1], bins = 20)
ax1.set_title('Default')

ax2.hist(Data.limit_bal[Data.default == 0], bins = 20)
ax2.set_title('No Default')

plt.xlabel('Limit Balance ')
plt.ylabel('Number of Observations')
plt.show()


# ## Defaulting by various demographics

# In[ ]:


Data['age_cat'] = pd.cut(Data['age'], range(0, 100, 10), right=False)
fig, ax = plt.subplots(1,4)
fig.set_size_inches(20,5)
fig.suptitle('Defaulting by absolute numbers, for various demographics')

d = Data.groupby(['default', 'sex']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[0])

d = Data.groupby(['default', 'marriage']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[1])

d = Data.groupby(['default', 'age_cat']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[2])

d = Data.groupby(['default', 'education']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[3])


# ## Developing a predictive model to predict chance of default (credit score of customer) in next month based on available information.

# In[ ]:


from sklearn import preprocessing
import random
random.seed(90)
from sklearn.neural_network import MLPClassifier
print('Random',random.random())
from sklearn.preprocessing import StandardScaler  
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,cross_validation,svm,neighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


predictors = Data.columns.drop(['default','age_cat'])

x = np.asarray(Data[predictors])
y = np.asarray(Data['default'])
y=y.astype('int')

X = x
Y =y
model = LogisticRegression()


# In[ ]:


def run_classification_algorithms(X1, features_list, num_of_feaures):
    row = {}
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=0, stratify=y)
    X_train=preprocessing.robust_scale(X_train, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    X_test=preprocessing.robust_scale(X_test, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    svm.LinearSVC(C=1.0,random_state=0)
    row['features']=features_list
    row['NumOfFeatures']=num_of_feaures
    clf=svm.LinearSVC(C=1.0,random_state=0)
    row['SVM']=clf.fit(X_train,y_train).score(X_test,y_test)
    clf=DecisionTreeClassifier(random_state=0)
    row['DT'] = clf.fit(X_train,y_train).score(X_test,y_test)
    clf= RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=1, verbose=0, warm_start=False, class_weight=None)
    row['RF'] = clf.fit(X_train,y_train).score(X_test,y_test)
    clf = KNeighborsClassifier(n_neighbors=7)
    row['KNN']=clf.fit(X_train,y_train).score(X_test,y_test)
    clf=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    beta_1=0.9, beta_2=0.999, early_stopping=False,
    epsilon=1e-08, hidden_layer_sizes=(5,2), learning_rate='constant',
    learning_rate_init=0.001, max_iter=300, momentum=0.9,
    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    solver='adam', tol=0.001, validation_fraction=0.1, verbose=False,
    warm_start=False)
    row['ANN']=clf.fit(X_train,y_train).score(X_test,y_test)
    return row

#select top best features from regression fit then procede on classification
def select_features(if_select):
    selected_array = []
    for i in range(if_select.size):
        if(if_select[i]):
            selected_array.append(i)
    return selected_array

#create df to hold values for the accuracy in 'SVM_SVC', 'Decision Tree', 'Random Forest','KNN Accuracy', 'ANN or MultiLayerPerceptron'
acc_columns = ['NumOfFeatures','SVM', 'DT', 'RF','KNN', 'ANN']
df1 =  pd.DataFrame(columns=['features'], dtype=str)
df2 =  pd.DataFrame(columns=acc_columns, dtype=float)
results = pd.concat([df1, df2], axis=1)

for i in range(len(predictors)):
    rfe = RFE(model, i+1)
    fit = rfe.fit(X, Y)
    selected_features = select_features(fit.support_)
    features_list = [predictors[i] for i in selected_features]
    print("Num Features:",fit.n_features_)
    print("Selected Features:",features_list)
    print("Feature Ranking: ",fit.ranking_)
    X1=pd.DataFrame(X)
    X1 = X1[selected_features]
    results=results.append(run_classification_algorithms(X1, features_list,fit.n_features_), ignore_index=True)


# ## Developing dashboard to present model statistics

# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 13, 8      
plt.plot(results['NumOfFeatures'],results['SVM'],'bo-', linewidth=4, label='SVM')
plt.plot(results['NumOfFeatures'],results['DT'],'ro-', linewidth=4, label='Decison Tree')
plt.plot(results['NumOfFeatures'],results['RF'],'go-', linewidth=4, label='Random Forest')
plt.plot(results['NumOfFeatures'],results['KNN'],'yo-', linewidth=4, label='KNN')
plt.plot(results['NumOfFeatures'],results['ANN'],'mo-', linewidth=4, label='ANN')

plt.title('Performances over number of features')
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.axis([0, len(predictors)+1, 0.6, 0.85])
plt.legend(loc='lower center')
plt.rcParams.update({'font.size': 12})
plt.grid(True)
plt.tight_layout
plt.show()


# In[ ]:




