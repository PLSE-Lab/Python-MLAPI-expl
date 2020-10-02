#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler


# In[ ]:


train_df = pd.read_csv('../input/medical-treatment-dataset/trainms.csv')
test_df = pd.read_csv('../input/medical-treatment-dataset/testms.csv')
target = pd.read_csv('../input/medical-treatment-dataset/samplems.csv')
train_df.head(3)


# In[ ]:


#dataframe.drop(['<columns/rows>'], axis = 0/1, inplace = False/true)
target.drop(['s.no'], axis = 1, inplace = True)
test_df = pd.concat([test_df, target], axis = 1)


# In[ ]:


test_df.head()


# In[ ]:


print("Train data dim : ", train_df.shape, "\nTest data dim : ", test_df.shape)


# In[ ]:


df = pd.concat([train_df, test_df], sort = True)
df.reset_index(drop=True, inplace = True)
df


# In[ ]:


df.drop( 's.no', inplace = True, axis = 1)
df.head(3)


# # DATA PREPROCCESING

# In[ ]:


#finding Missing data
#dataframe.isnull()
null_df = df.isnull()
null_df.head()


# In[ ]:


for col in null_df.columns:
    print(null_df[col].value_counts())
    print("_______________________")


# **From the above list, it is clear that the feature 'state' contain more null values that cannot even filled, since some countries are almost null. So we can drop the column state**

# In[ ]:


df.drop(columns = ['Timestamp', 'state'], inplace = True)
df.head()


# In[ ]:


df['self_employed'].value_counts()


# In[ ]:


#since 'No' is most common in train_df['self_employed'], we replace NaN with 'No'. 
df['self_employed'].fillna('No', inplace = True)
train_df.head()


# In[ ]:


df['work_interfere'].value_counts()


# In[ ]:


df['work_interfere'].fillna('Sometimes', inplace = True)
df['work_interfere'].value_counts()


# In[ ]:


def clean_data(result):

    #Gender value cleaning
    #Male
    #dataframe.loc[dataframe['<column>'] == '<value>', '<column>'] = <value>
    result.loc[result['Gender'] == 'M','Gender'] = 'Male'
    result.loc[result['Gender'] == 'male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Mail','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Mal','Gender'] = 'Male'
    result.loc[result['Gender'] == 'msle','Gender'] = 'Male'
    result.loc[result['Gender'] == 'm','Gender'] = 'Male'
    result.loc[result['Gender'] == 'maile','Gender'] = 'Male'
    result.loc[result['Gender'] == 'mal','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Male-ish','Gender'] = 'Male'
    result.loc[result['Gender'] == 'ostensibly male, unsure what that really means','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Cis Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'something kinda male?','Gender'] = 'Male'
    result.loc[result['Gender'] == 'make','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Make','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Cis Male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis Male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Malr','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Male ','Gender'] = 'Male'




    #Female
    result.loc[result['Gender'] == 'F','Gender'] = 'Female'
    result.loc[result['Gender'] == 'female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'f','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Cis Female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Femake','Gender'] = 'Female'
    result.loc[result['Gender'] == 'cis-female/femme','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Female (cis)','Gender'] = 'Female'
    result.loc[result['Gender'] == 'cis female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Woman','Gender'] = 'Female'
    result.loc[result['Gender'] == 'woman','Gender'] = 'Female'
    result.loc[result['Gender'] == 'femail','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Female ','Gender'] = 'Female'
    
    
    #Transgender
    result.loc[result['Gender'] == 'Trans woman','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Female (trans)','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Female (trans)','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Trans-female','Gender'] = 'Other'
    result.loc[result['Gender'] == 'non-binary','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Nah','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Enby','Gender'] = 'Other'
    result.loc[result['Gender'] == 'fluid','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Genderqueer','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Androgyne','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Agender','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Guy (-ish) ^_^','Gender'] = 'Other'
    result.loc[result['Gender'] == 'male leaning androgynous','Gender'] = 'Other'
    result.loc[result['Gender'] == 'Neuter','Gender'] = 'Other'
    result.loc[result['Gender'] == 'queer','Gender'] = 'Other'
    result.loc[result['Gender'] == 'A little about you','Gender'] = 'Other'
    result.loc[result['Gender'] == 'p','Gender'] = 'Other'


# In[ ]:


clean_data(df)
df['Gender'].value_counts()


# # Outlier Detection and Removing

# In[ ]:


sns.boxplot(y = df['Age'])


# In[ ]:


df['Age'].describe()


# In[ ]:


outlierIndex = []
outlierIndex.append(df.loc[df['Age'] == -1726.000000 ].index.values[0])
outlierIndex.append(df.loc[df['Age'] == 329.000000 ].index.values[0])


# In[ ]:


df.drop(outlierIndex, inplace = True)


# In[ ]:


sns.boxplot(y = df['Age'])


# In[ ]:


df['Age'].quantile([.01, .98])


# In[ ]:


outlierIndex = []
for idx in df.loc[df['Age'] < 18 ].index.values:
    outlierIndex.append(idx)
for idx in df.loc[df['Age'] > 50 ].index.values:
    outlierIndex.append(idx)
    
print(outlierIndex)


# In[ ]:


df.drop(outlierIndex, inplace = True)
df.shape


# In[ ]:


sns.boxplot(y = df['Age'])


# # Categorical Data Encoding

# In[ ]:


data = df


# In[ ]:


y = data['treatment']
y = pd.DataFrame(y)
data.drop(['treatment'], axis = 1, inplace = True)
data.head()


# In[ ]:


#Encoding target variable
y.loc[y['treatment'] == 'Yes','treatment'] = 1
y.loc[y['treatment'] == 'No','treatment'] = 0
y.head()


# ### 1. Finding and encoding High Cardinality Features.
# 
# * High Cardinality means that, features with lots of unique values

# In[ ]:


data['Country'].unique().shape


# * Both features are high cardinality features(k<10k). Hence we cannot apply one-hot encoding since sparse matrix affect the performance of our model.
# 
# * The best two options are,
#     * Binary Encoding 
#     * Feature Hashing
#     * Target Encoding
#     
# * Binary encoding is most suitable for Ordinal data. Here 'Country' and 'Gender are nominal data. But its worth trying it too.
# 
# * Hash encoding may results in Collision, but effect of collision is much less on model prediction.
# 
# * Here I continue with target encoding, which I feel more suitable after trying other encodings 
# 
# [Referance - Categorical encoding ](https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512)
# 
# [Reference - Hash Ticking](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087)

# In[ ]:


#Target encoding on features ['Country', 'Gender']
ce_df = ce.TargetEncoder(cols = ['Country'])
data = ce_df.fit_transform(data, y['treatment'])
data.head()


# ### 2. Encoding Categorical variables with Yes/No values
# 
# * The best method to encode categorical variables with values Yes/No is replacing them with 1/0
#     * Yes - 1
#     * No - 0

# In[ ]:


features = ['self_employed', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']
data[features]

def encodeYesNo(feature):
    data.loc[data[feature] == 'Yes', feature] = 1
    data.loc[data[feature] == 'No', feature ] = 0
    
for feature in features:
    encodeYesNo(feature)
    
data[features].head()


# In[ ]:


data.head()


# ### Encoding Categories with a few unique values.
# * Here most of the categorical features are nominal

# In[ ]:


ordinal_features = ['Gender', 'work_interfere', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical']
ce_ord = ce.TargetEncoder(cols = ordinal_features)
data = ce_ord.fit_transform(data, y['treatment'])
data.head()


# ### 4.Encoding Interval type categorical values.
# 
# * one of the good approch for interval type values are split them into lower and upper values.

# In[ ]:


data['no_employees'].value_counts()


# In[ ]:


data.loc[data['no_employees'] == 'More than 1000', 'no_employees'] = '1000-10000'
data.head()


# In[ ]:


lower = []
upper = []
for val in data['no_employees']:
    values = val.split('-')
    lower.append(int(values[0]))
    upper.append(int(values[1]))
    
    
data['no_employees_lower'] = lower
data['no_employees_upper'] = upper
data[['no_employees', 'no_employees_lower', 'no_employees_upper']].head()


# In[ ]:


data.drop('no_employees', axis = 1, inplace = True)
data.head()


# In[ ]:


data.drop(['comments'], axis = 1, inplace = True)


# In[ ]:


data.head()


# ### 5. Scaling numerical Variables
# 
# * scaling numerical variables using min-max scaler function

# In[ ]:


scaled_df = data

minmax_scaler = StandardScaler()
scaled = minmax_scaler.fit_transform(data[['Age','no_employees_lower', 'no_employees_upper']]) 


scaled_df['scaled_age'] = scaled[:,0]
scaled_df['scaled_no_employees_lower'] = scaled[:,1]
scaled_df['scaled_no_employees_upper'] = scaled[:,2]
scaled_df.drop(['Age', 'no_employees_lower', 'no_employees_upper'], axis = 1, inplace = True)
scaled_df.head()


# In[ ]:


plt.title("Data distribution before scalling")
plt.xlim(-6, 6)

for col in ['scaled_age', 'scaled_no_employees_lower', 'scaled_no_employees_upper']:
    sns.kdeplot(data[col],)

plt.legend(bbox_to_anchor=(2.2, 1), ncol=2)


# In[ ]:


X = scaled_df
print(X.shape, "\n\n", y.shape)


# # Feature Selection

# In[ ]:


proccessed_data = scaled_df
proccessed_data['treatment'] = y['treatment'].values


# In[ ]:


corr = proccessed_data.corr(method = 'pearson')['treatment']
corr


# In[ ]:


plt.figure(figsize=(30, 20))
sns.heatmap(proccessed_data.corr(method = 'pearson'), annot = proccessed_data.corr(), square = True)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


extraTree = ExtraTreesClassifier()
extraTree.fit(X, y)
print(extraTree.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[ ]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(extraTree.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 important features")
plt.show()


# In[ ]:


const = np.ones(y.shape)
X.insert(loc = 0, column = 'x0', value = const)


# In[ ]:


imp_features = ['x0']
for feature in feat_importances.nlargest(15).keys():
    if feature != 'treatment':
        imp_features.append(feature)
        
X[imp_features]


# # Model Development

# ### Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split( X[imp_features], y, test_size=0.2, random_state=42)
print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)


# In[ ]:


model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_train, y_train)


# In[ ]:


yhat = model.predict(x_test)
yhat


# In[ ]:


confusion_mtx = confusion_matrix(y_test, yhat)
print(confusion_mtx)


# In[ ]:


print("Test Accuracy = ", f1_score(y_test, yhat))

