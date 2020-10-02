#!/usr/bin/env python
# coding: utf-8

# ### <font color = blue>Brief Description : 
# **We need to classify the persons as 1,2,3 or 4 class based on the Costa Rican household characteristics in this dataset.**

# In[ ]:


# Basic essential libraries >>>
import numpy as np
import pandas as pd
import os 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import time
import sys


# In[ ]:


# Accessing the data >>> 
train = train = pd.read_csv("../input/train-costa-rica/train.csv")
print('Shape : ',train.shape)
pd.options.display.max_columns = None
train.head(2)


# # <font color = darkorange> Part - 1 : Data Analysis and Preprocessing 

# <font color = dodgerblue>**Handing Missing Values**

# In[ ]:


Null_Df = pd.DataFrame({'Null_Count':train.isnull().sum(),
                        'Null_Perc %':round(train.isnull().sum()*100/train.shape[0],2)})
Null_Df[Null_Df.Null_Count!=0]


# **<font color=red>Handling missing values :**`v2a1_num --> Monthly rent payment`

# In[ ]:


print('Mean -->',train.v2a1_num.mean())
print('Median -->',train.v2a1_num.median())
print('Std -->',train.v2a1_num.std())
sns.distplot(train[train.v2a1_num.notnull()].v2a1_num)
plt.show()


# In[ ]:


# As we can see that all the data is centred at on point only hence we can replace the missing values with average values.
print('Mean BEFORE Change : ',train.v2a1_num.mean())
train.v2a1_num.fillna(train.v2a1_num.mean(),inplace=True)
print('Mean AFTER Change : ',train.v2a1_num.mean())
print('\nMissing Values :')
train.v2a1_num.isnull().value_counts()


# **<font color=red>Handling missing values :**`v18q1_num  --> number of tablets household owns`

# In[ ]:


print('Mean -->',train.v18q1_num.mean())
print('Median -->',train.v18q1_num.median())
print('Std -->',train.v18q1_num.std())

sns.distplot(train[train.v18q1_num.notnull()].v18q1_num)
plt.show()


# **`Important :`** As we can see that all the data is centred at one point however we can not take average in this case because this column represents "Number of tablets per household" and if we replace by average it will be in decimals but a physical object can not be in decimals. 
# 
# Hence we will replace the missing values by **`median`**

# In[ ]:


train.v18q1_num.fillna(train.v18q1_num.median(),inplace=True)
print('\nMissing Values :')
train.v18q1_num.isnull().value_counts()


# **<font color=red>Handling missing values :** `rez_esc_num  --> Years behind in school`
# 

# In[ ]:


print('Mean -->',train.rez_esc_num.mean())
print('Median -->',train.rez_esc_num.median())
print('Std -->',train.rez_esc_num.std())

sns.distplot(train[train.rez_esc_num.notnull()].rez_esc_num)
plt.show()


# In[ ]:


# "Unlike" v18q1_num, this number can be in decimals hence we will replace the missing values with mean value >>>  
print('Mean BEFORE Change : ',train.rez_esc_num.mean())
train.rez_esc_num.fillna(train.rez_esc_num.mean(),inplace=True)
print('Mean AFTER Change : ',train.rez_esc_num.mean())
print('\nMissing Values :')
train.rez_esc_num.isnull().value_counts()


# **<font color=red>Handling Missing values :**  `"meaneduc_num" and "SQBmeaned_num"`
# 
# As the number of values missing are very less hence we will use method of 'ffill' which means the data from just previous cell will be copied to the next missing cells.

# In[ ]:


train.meaneduc_num.fillna(method='ffill',inplace=True)
train.SQBmeaned_num.fillna(method='ffill',inplace=True)


# In[ ]:


Null_Df = pd.DataFrame({'Null_Count':train.isnull().sum(),
                        'Null_Perc %':round(train.isnull().sum()*100/train.shape[0],2)})
Null_Df[Null_Df.Null_Count!=0]


# <font color=red>***So No Null values***

# # <font color=royalblue> Plan Of Attack:
#     
# 1. Divide the train data into Numerical, Categorical and Dependent var
# 2. Categorical : We don't need to encode categorical columns as the columns are already encoded as 0 or 1
# 3. Numerical : We have to scale them. But we will scale them using both methods "StandardScaler" and "MinMaxScaler" so that we can use any of the scaled data as per requirement to increase accuracy. 
# 4. After scaling, we will combine both Categorical and Numerical
# 5. Next we will split this combined data and dependent data into train and test datasets as we need to calculate the accuracy of our prediction model. 
# 6. Then we will try to apply differnt methodology like KNN, RandomForest and lastly ANN (if required)
# 7. We will also try to use the Validation method K-Fold and dimension reduction method PCA (if required)
# 8. Finally after the model is prepared, we will apply on test data provided with this project and will generate the submission file. 

# ### <font color =darkmagenta>Split the data - Numerical, Categorical and Dependent

# In[ ]:


train.head(2)


# In[ ]:


# Below code is to generate the separate lists of numerical and categorical columns.  
# This is required as in this dataset, categorical and numerical columns have same type of values i.e. int or float instead of str value for...
# ... categorical values which is usual for other datasets. Hence in those datasets, it is easy to filter the numerical and categorical columns...
# ... using dtype. Hence for this dataset, I renamed the column names manually in excel by checking the column description given on the project's website. 

numerical = []
categorical = []
j = 0
for i in train.columns:
    if train.columns.str.contains('_num',regex=True)[j]:
        numerical.append(i)
    else:
        categorical.append(i)
    j+=1
print('Numerical Columns : %s\n'%len(numerical))
print('Categorical Columns : %s\n'%len(categorical))
print('Total Columns :%s'%train.shape[1])


# ### <font color=green> Cleaning the data
# The data contains mixed values means along with the numerical values, the data contains string values within the same columns. Hence we have to clean this data

# In[ ]:


temp_numerical = train.loc[:,numerical]
temp_categorical = train.loc[:,categorical]


# In[ ]:


obj_num = []
for i in temp_numerical.columns:
    if str(temp_numerical[i].dtype) =='object':
        obj_num.append(i)
print('Numerical columns with str values : ',obj_num)        
obj_cat = []
for i in temp_categorical.columns:
    if str(temp_categorical[i].dtype) =='object':
        obj_cat.append(i)
print('Categorical columns with str values : ',obj_cat) 


# **`Cleaning Categorical`** : Well we really would not include the two columns found with str in categorical category as those columns can not contribute in the training of model because they are just IDs/Unique Values hence can not predict any pattern. 

# In[ ]:


temp_categorical.drop(['Id', 'idhogar'],axis=1,inplace=True)


# **`Cleaning Numerical`** : For this, lets plot the histogram and see how the numerical values are spread so that we can decide how to replace the str values. 

# In[ ]:


# Run below to know what are the str values >>> 
#test_temp_numerical.dependency_num.value_counts().index
#temp_numerical.edjefe_num.value_counts().index
#temp_numerical.edjefa_num.value_counts().index


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.distplot(temp_numerical[(temp_numerical.edjefe_num != 'yes')& (temp_numerical.edjefe_num != 'no')].edjefe_num.astype('int'))

plt.subplot(1,3,2)
sns.distplot(temp_numerical[(temp_numerical.edjefa_num != 'yes')& (temp_numerical.edjefa_num != 'no')].edjefa_num.astype('int'))

plt.subplot(1,3,3)
sns.distplot(temp_numerical[(temp_numerical.dependency_num != 'yes')& (temp_numerical.dependency_num != 'no')].dependency_num.astype('float'))

plt.show()


# **<font color = steelblue>Data above is spread almost in a similar way. So we will replace the values with method : Mean of values :-)**

# In[ ]:


t1 = time.time()
mean_edjefe = temp_numerical[(temp_numerical.edjefe_num != 'yes')& (temp_numerical.edjefe_num != 'no')].edjefe_num.astype('int').mean()
mean_edjefa = temp_numerical[(temp_numerical.edjefa_num != 'yes')& (temp_numerical.edjefa_num != 'no')].edjefa_num.astype('int').mean()
mean_dependency = temp_numerical[(temp_numerical.dependency_num != 'yes')& (temp_numerical.dependency_num != 'no')].dependency_num.astype('float').mean()
edjefe = []
edjefa = []
dependency = []
for i in range(temp_numerical.shape[0]):
    if temp_numerical.edjefe_num[i] in ['yes','no']:
        edjefe.append(mean_edjefe)
    else:
        edjefe.append(temp_numerical.edjefe_num[i])
    
    if temp_numerical.edjefa_num[i] in ['yes','no']:
        edjefa.append(mean_edjefa)
    else:
        edjefa.append(temp_numerical.edjefa_num[i])
        
    if temp_numerical.dependency_num[i] in ['yes','no']:
        dependency.append(mean_dependency)
    else:
        dependency.append(temp_numerical.dependency_num[i])
    
    if i%100==0:        # Small patch just to keep track of progress of loop. 
        print('Rows Processed --> %s'%i,end='\r')

temp_numerical.edjefe_num = edjefe
temp_numerical.edjefa_num = edjefa
temp_numerical.dependency_num = dependency

# Changing the type of the columns >>> 

temp_numerical.edjefe_num = temp_numerical.edjefe_num.astype('int64')
temp_numerical.edjefa_num = temp_numerical.edjefa_num.astype('int64')
temp_numerical.dependency_num = temp_numerical.dependency_num.astype('float64')

t2 = time.time()
print('Time Elapsed : %s'%(t2-t1))


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.distplot(temp_numerical.edjefe_num)
plt.subplot(1,3,2)
sns.distplot(temp_numerical.edjefa_num)
plt.subplot(1,3,3)
sns.distplot(temp_numerical.dependency_num)
plt.show()


# After replacing the str values with numerical, the data is quite skewed to one point for all three columns however there was no other option but to replace the values or otherwise just remove the whole columns themselves. 
# 
# We will try to remove the columns if we do not get good results. (good accuracy in the model). Lets move further for the scaling exercise >>> 

# In[ ]:


# Making the copies of the above datasets >>> 
# WHY making copies ? Because we have processed the two types of data separately. Hence now if we make any changes to ...
# ...the copies and want fresh data again then we can just run this cell >>> 

x_train_numerical = temp_numerical.copy()
x_train_categorical = temp_categorical.copy()

# This data would have the dependent variable as well hence we need to drop that >>> 
x_train_categorical.drop('Target',axis=1,inplace=True)

y_train_full = train.loc[:,'Target']


# ### <font color =darkblue> Scaling the Numerical data
#     
# We will scale using both methods in separate cells hence whichever type is required, just run that cell !!!

# In[ ]:


# Indicator var >>> Run This Cell >>>
std_scal = 0
MinMax =0

# Hence after running any of the scaling method's cell, run "# Scaling method Used >>> " cell as well to mention which...
# ... method being used for scaling. 


# In[ ]:


# StandardScaler >>> 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled = sc.fit_transform(x_train_numerical)

# Converting array type back to dataframe >>> 
x_train_scaled = pd.DataFrame(scaled)
x_train_scaled.columns = x_train_numerical.columns
std_scal+=1


# In[ ]:


# MinMaxScaler >>> 

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
scaled = sc.fit_transform(x_train_numerical)

# Converting array type back to dataframe >>> 
x_train_scaled = pd.DataFrame(scaled)
x_train_scaled.columns = x_train_numerical.columns
MinMax+=1


# In[ ]:


# Scaling method Used >>> 
if MinMax >= std_scal:
    print('Scaling Method Utilized : "MinMaxScaler"')
else:
    print('Scaling Method Utilized : "StandardScaler"')


# ### <font color = tomato>Combining Data : Numerical - Scaled + Categorical

# In[ ]:


print('Shape of Numerical : %s'%x_train_numerical.shape[1])
print('Shape of Categorical : %s'%x_train_categorical.shape[1])
x_train_full = x_train_scaled.merge(x_train_categorical,left_index=True,right_index=True)
print('Shape of Combined : %s'%x_train_full.shape[1])


# # <font color = darkblue> Part - 2 : Creating Prediction Model 

# ## <font color = limegreen> Algorithm : KNN

# In[ ]:


t1 = time.time()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Creating the copies of data to be inserted >>> 
x_train = x_train_full.copy()
y_train = y_train_full.copy()

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=123)

# Find optimal neigbors >>> 
A = []
B = []
for i in range(11,500,40):
    classifier = KNeighborsClassifier(n_neighbors=i,metric='euclidean').fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    A.append(i)
    B.append(round(accuracy_score(y_test,y_pred)*100,3))
    
table = pd.DataFrame({'Neighbors':A,'Accuracy':B})
plt.figure(figsize=(15,5))
plt.plot(table.Neighbors,table.Accuracy,marker='^',mec='black',mfc='r',ms=10,color='b')
plt.xticks(np.arange(11,500,40))
plt.grid()
plt.show()

t2 = time.time()
print('Time Elapsed : %s'%(t2-t1))


# **<font color=royalblue>With above run, we can conclude that the highest accuracy we can achieve with neighbors = 11**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Creating the copies of data to be inserted >>> 
x_train = x_train_full.copy()
y_train = y_train_full.copy()

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=123)

# Run KNN >>> 
classifier = KNeighborsClassifier(n_neighbors=10,metric='euclidean').fit(x_train,y_train)
y_pred = classifier.predict(x_test)

accuracy_knn = round(accuracy_score(y_test,y_pred)*100,3)
print('Accuracy via KNN : ',accuracy_knn)


# ## <font color=fuchsia>Algorithm : RandomForest
# Lets run RandomForest Algorithm and see how it performs

# In[ ]:


t1 = time.time()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Creating the copies of data to be inserted >>> 
x_train = x_train_full.copy()
y_train = y_train_full.copy()

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=123)

# Find optimal number of trees  >>> 
A = []
B = []
for i in range(11,500,40):
    classifier = RandomForestClassifier(n_estimators=i,criterion='entropy',random_state=123).fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    A.append(i)
    B.append(round(accuracy_score(y_test,y_pred)*100,3))
    
    print('Processing Estimators : %s'%i,end='\r')
    
table = pd.DataFrame({'Neighbors':A,'Accuracy':B})
plt.figure(figsize=(15,5))
plt.plot(table.Neighbors,table.Accuracy,marker='^',mec='black',mfc='b',ms=10,color='r',linestyle='--')
plt.xticks(np.arange(11,500,40))
plt.grid()
plt.show()

t2 = time.time()
print('Time Elapsed : %s'%(t2-t1))


# **<font color=royalblue>Lets create the model using the estimators = 211**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Creating the copies of data to be inserted >>> 
x_train = x_train_full.copy()
y_train = y_train_full.copy()

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=123)

# Run RandomForest >>> 
classifier = classifier = RandomForestClassifier(n_estimators=211,criterion='entropy',random_state=123).fit(x_train,y_train)
y_pred = classifier.predict(x_test)

accuracy = round(accuracy_score(y_test,y_pred)*100,3)
print('Accuracy via RandomForest : ',accuracy)


# ## <font color=blueviolet>Model Validation : K-Fold

# In[ ]:


# Importing essential libraries >>> 

from sklearn.model_selection import cross_val_score

# run K-Fold >>>
accuracies = cross_val_score(estimator = classifier,X = x_train,y=y_train,cv=10)

# Results >>>
# -------------
print('Actual Accuracy : %s'%round((accuracies.mean()*100),2)+' %')
print('Standard Deviation of Accuracy : %s'%round((accuracies.std()*100),2)+' %')
print('RandomFt Accuracy :%s'%accuracy+' %')


# ### <font color = royalblue> Remarks for K-Fold :
# <font color = red>**From above, it is clear that the model accuracy is stable as the mean accuracy doesn't have high difference from the accuracy calculated by RandomForest model directly.**
# <font color = red\>  
#     
# <font color =black>Now we will apply this model on the test data
# ---
#     

# In[ ]:


# Accessing the test data >>> 
test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
print('Shape : ',test.shape)
test.head(2)


# In[ ]:


test.drop(['Id','idhogar'],axis=1,inplace=True)
print('Shape : ',test.shape)
test.head(2)


# **Tagging of "Numerical" columns**

# In[ ]:


# Instead of manually renaming the columns, for test data I will try to add suffix "_num" to the numerical columns based on their values >>> 
numerical = []
categorical = []
columns = test.columns
for i in columns:
    if len(test[i].unique()) >2:
        test.rename(columns={i:i+'_num'},inplace=True)
        numerical.append(i+'_num')
    else:
        categorical.append(i)
        
print('Numerical Columns : %s\n'%len(numerical))
print('Categorical Columns : %s\n'%len(categorical))
print('Total Columns :%s'%test.shape[1])


# **Finding Missing Data (if any)**

# In[ ]:


Null_Df = pd.DataFrame({'Null_Count':test.isnull().sum(),
                        'Null_Perc %':round(test.isnull().sum()*100/test.shape[0],2)})
Null_Df[Null_Df.Null_Count!=0]


# **Hanlding the missing values same as train data**

# In[ ]:


test.v2a1_num.fillna(test.v2a1_num.mean(),inplace=True)
test.v18q1_num.fillna(test.v18q1_num.median(),inplace=True)
test.rez_esc_num.fillna(test.rez_esc_num.mean(),inplace=True)
test.meaneduc_num.fillna(method='ffill',inplace=True)
test.SQBmeaned_num.fillna(method='ffill',inplace=True)


# In[ ]:


Null_Df = pd.DataFrame({'Null_Count':test.isnull().sum(),
                        'Null_Perc %':round(test.isnull().sum()*100/test.shape[0],2)})
Null_Df[Null_Df.Null_Count!=0]


# <font color=red>***So NO Null Values !!!***

# In[ ]:


test_temp_numerical = test.loc[:,numerical]
test_temp_categorical = test.loc[:,categorical]


# **Validating if all the train data columns match test data columns**

# In[ ]:


columns = temp_numerical.columns
num = []
for i in test_temp_numerical.columns:
    if i not in columns:
        num.append(i)
print('Column not matching : ',num)
        
cat=[]
columns = temp_categorical.columns
for i in test_temp_categorical.columns:
    if i not in columns:
        cat.append(i)
print('Column not matching : ',cat)


# <font color=red>***So No mismatched columns!!!***

# **Clearing Data : Handling Mixed Values**

# In[ ]:


obj_num = []
for i in test_temp_numerical.columns:
    if str(test_temp_numerical[i].dtype) =='object':
        obj_num.append(i)
print('Numerical columns with str values : ',obj_num)        
obj_cat = []
for i in test_temp_categorical.columns:
    if str(test_temp_categorical[i].dtype) =='object':
        obj_cat.append(i)
print('Categorical columns with str values : ',obj_cat) 


# In[ ]:


# Run below to know what are the str values >>> 
#test_temp_numerical.dependency_num.value_counts().index
#test_temp_numerical.edjefe_num.value_counts().index
#test_temp_numerical.edjefa_num.value_counts().index


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.distplot(test_temp_numerical[(test_temp_numerical.edjefe_num != 'yes')& (test_temp_numerical.edjefe_num != 'no')].edjefe_num.astype('int'))

plt.subplot(1,3,2)
sns.distplot(test_temp_numerical[(test_temp_numerical.edjefa_num != 'yes')& (test_temp_numerical.edjefa_num != 'no')].edjefa_num.astype('int'))

plt.subplot(1,3,3)
sns.distplot(test_temp_numerical[(test_temp_numerical.dependency_num != 'yes')& (test_temp_numerical.dependency_num != 'no')].dependency_num.astype('float'))

plt.show()


# **<font color = steelblue>Just like the train data, we will handle str values for test data**

# In[ ]:


t1 = time.time()
mean_edjefe = test_temp_numerical[(test_temp_numerical.edjefe_num != 'yes')& (test_temp_numerical.edjefe_num != 'no')].edjefe_num.astype('int').mean()
mean_edjefa = test_temp_numerical[(test_temp_numerical.edjefa_num != 'yes')& (test_temp_numerical.edjefa_num != 'no')].edjefa_num.astype('int').mean()
mean_dependency = test_temp_numerical[(test_temp_numerical.dependency_num != 'yes')& (test_temp_numerical.dependency_num != 'no')].dependency_num.astype('float').mean()
edjefe = []
edjefa = []
dependency = []
for i in range(test_temp_numerical.shape[0]):
    if test_temp_numerical.edjefe_num[i] in ['yes','no']:
        edjefe.append(mean_edjefe)
    else:
        edjefe.append(test_temp_numerical.edjefe_num[i])
    
    if test_temp_numerical.edjefa_num[i] in ['yes','no']:
        edjefa.append(mean_edjefa)
    else:
        edjefa.append(test_temp_numerical.edjefa_num[i])
        
    if test_temp_numerical.dependency_num[i] in ['yes','no']:
        dependency.append(mean_dependency)
    else:
        dependency.append(test_temp_numerical.dependency_num[i])
    
    if i%100==0:        # Small patch just to keep track of progress of loop. 
        print('Rows Processed --> %s'%i,end='\r')

test_temp_numerical.edjefe_num = edjefe
test_temp_numerical.edjefa_num = edjefa
test_temp_numerical.dependency_num = dependency

# Changing the type of the columns >>> 

test_temp_numerical.edjefe_num = test_temp_numerical.edjefe_num.astype('int64')
test_temp_numerical.edjefa_num = test_temp_numerical.edjefa_num.astype('int64')
test_temp_numerical.dependency_num = test_temp_numerical.dependency_num.astype('float64')

t2 = time.time()
print('Time Elapsed : %s'%(t2-t1))


# **Making Copies of the datasets**

# In[ ]:


# Making the copies of the above datasets >>> 
# WHY making copies ? Because we have processed the two types of data separately. Hence now if we make any changes to ...
# ...the copies and want fresh data again then we can just run this cell >>> 

x_test_numerical = test_temp_numerical.copy()
x_test_categorical = test_temp_categorical.copy()


# **Scaling the Numerical Data**

# In[ ]:


# MinMaxScaler >>> 

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
scaled = sc.fit_transform(x_test_numerical)

# Converting array type back to dataframe >>> 
x_test_scaled = pd.DataFrame(scaled)
x_test_scaled.columns = x_test_numerical.columns


# **Combining Numerical + Categorical**

# In[ ]:


print('Shape of Numerical : %s'%x_test_numerical.shape[1])
print('Shape of Categorical : %s'%x_test_categorical.shape[1])
x_test_full = x_test_scaled.merge(x_test_categorical,left_index=True,right_index=True)
print('Shape of Combined : %s'%x_test_full.shape[1])


# ## Fitting Model & Making Prediction 

# In[ ]:


y_test_pred = classifier.predict(x_test_full)


# In[ ]:


data_test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv") 
# Accessing raw test again to get the Id column which is dropped in the original data. 


# In[ ]:


submission_file = pd.DataFrame({'Id':data_test.Id,'Target':y_test_pred})
submission_file.Target.value_counts()


# In[ ]:


submission_file.head()


# In[ ]:


submission_file.to_csv('submission.csv',index=False)

