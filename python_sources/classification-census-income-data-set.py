#!/usr/bin/env python
# coding: utf-8

# ## Importing the data and first look

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/us-adult-income-update/train.csv')
cleaned_census = pd.read_csv('/kaggle/input/us-adult-income-update/census.csv')
census = pd.read_csv('/kaggle/input/us-adult-income-update/census.csv')
test = pd.read_csv('/kaggle/input/us-adult-income-update/test.csv')


# In[ ]:


#census.profile_report()


# In[ ]:


train.head()


# In[ ]:


census.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)
print(census.shape)


# In[ ]:


print(train.shape[0]+test.shape[0])
print(test.shape[0]/census.shape[0])


# ## Census data EDA

# In[ ]:


census['income_above_50K']=census['income'].map({'<=50K':0, '>50K':1})


# In[ ]:


census.head()


# In[ ]:


def plot(column):
    if census[column].dtype != 'int64':
        f, axes = plt.subplots(1,1,figsize=(15,5))
        sns.countplot(x=column, hue='income_above_50K', data = census)
        plt.xticks(rotation=90)
        plt.suptitle(column,fontsize=20)
        plt.show()
    else:
        g = sns.FacetGrid(census, row="income_above_50K", margin_titles=True, aspect=4, height=3)
        g.map(plt.hist,column,bins=100)
        plt.show()
    plt.show()


# In[ ]:


plot('age')


# The graph is as expected in terms of distribution of those earning above 50K have a normal distribution and majority at eraly age are earning below 50K. A look at education levels should further support this as those who have taken a higher education will start full time work at a later age but will get a hiigher salary.

# In[ ]:


plot('workclass')


# I don't think I am going to use column in my final model, but will still test whether it affects the model, positively or negatively. For data cleaning purpose, I am going to keep the **Private** column and convert the rest to **Others**

# In[ ]:


plot('fnlwgt')


# Not sure how to interpret the above plot, but I will keep it as it is.

# In[ ]:


plot('education')


# I will group the above labels as below and perform a one hot encoding.
# 
# 1. Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad as **'level_1_ed'**
# 2. Prof-school < Assoc-acdm < Assoc-voc < Some-college as **'level_2_ed'**
# 3. Bachelors < Masters < Doctorate as **'level_3_ed'**

# In[ ]:


plot('education-num')


# In[ ]:


plot('marital-status')


# I am going to transform this column by changing **Married-civ-spouse** and **Married-AF-spuse** to **is_couple**.

# In[ ]:


plot('occupation')


# In[ ]:


over_50k_count_by_occ=pd.DataFrame(census[census['income_above_50K']==1]['occupation'].value_counts())
count_by_occ=pd.DataFrame(census['occupation'].value_counts())
merged=pd.merge(over_50k_count_by_occ,count_by_occ,left_index=True,right_index=True)
merged.rename(columns={'occupation_x':'income_over_50K','occupation_y':'Total pop'}, inplace=True)
merged['percent_of_above_50K']=merged['income_over_50K']/(merged['Total pop'])
merged=merged.sort_values(by='percent_of_above_50K',axis=0,ascending=False)
merged


# I am going to group this into 3 parts:
# 1. All above 30% as level_3_occ
# 2. Between 20% and 30% as level_2_occ
# 3. Below 20% as level_1_occ
# 

# In[ ]:


plot('relationship')


# This column looks like it gives the same information as the marital status. I am going to drop this column for my initial model. Maybe I will try adding it or replacing it with the marital-status for any test run to see how it affects the model.

# In[ ]:


f, axes = plt.subplots(1,1,figsize=(15,5))
sns.countplot(data=census,x='relationship',hue='marital-status')
plt.show()


# In[ ]:


plot('race')


# For this column, I am going to do a simple **is_white** transformation

# In[ ]:


plot('sex')


# A very low proportion of females have income above 50K. I am going to change the values to **is_female** column with 1's and 0's

# In[ ]:


plot('native-country')


# Again, a simple transformation of **is_US-native** should be sufficient here.

# In[ ]:


plot('capital-gain')


# In[ ]:


plot('capital-loss')


# Note sure on what to do with these two column, but going to keep them since they are numerical anyway.

# In[ ]:


plot('hours-per-week')


# I will also leave this as it it

# ## Data cleaning function

# In[ ]:


def data_cleaner(data):
    data['is_female']=data['sex'].map({'Male':0, 'Female':1})
    
    data['is_private']=data['workclass'].map({'Private':1})
    
    data['is_private'].fillna(0, inplace=True)
    
    data['education'] = data['education'].map(
        {'Preschool':'level_1_ed','1st-4th':'level_1_ed','5th-6th':'level_1_ed','7th-8th':'level_1_ed','9th':'level_1_ed','10th':'level_1_ed','11th':'level_1_ed','12th':'level_1_ed','HS-grad':'level_1_ed',
        'Prof-school':'level_2_ed','Assoc-acdm':'level_2_ed','Assoc-voc':'level_2_ed','Some-college':'level_2_ed',
        'Bachelors':'level_3_ed','Masters':'level_3_ed','Doctorate':'level_3_ed'})
    
    data['is_couple']=data['marital-status'].map({'Married-civ-spouse':1,'Never-married':0,'Divorced':0,'Separated':0,'Widowed':0,'Married-spouse-absent':0,'Married-AF-spouse':1})
    
    occupation_level_map={
    'Exec-managerial':'level_3_occ',
    'Prof-specialty':'level_3_occ',
    'Armed-Forces':'level_3_occ',
    'Protective-serv':'level_3_occ',
    'Tech-support':'level_2_occ',
    'Sales':'level_2_occ',
    'Craft-repair':'level_2_occ',
    'Transport-moving':'level_2_occ',
    'Adm-clerical':'level_1_occ',
    'Machine-op-inspct':'level_1_occ',
    'Farming-fishing':'level_1_occ',
    '?':'level_1_occ',
    'Handlers-cleaners':'level_1_occ',
    'Other-service':'level_1_occ',
    'Priv-house-serv':'level_1_occ'}
    data['occupation']=data['occupation'].map(occupation_level_map)
    
    race_map={'Black':0, 'White':1, 'Asian-Pac-Islander':0, 'Other':0,'Amer-Indian-Eskimo':0}
    data['is_white']=data['race'].map(race_map)
    
    native_country_map=pd.DataFrame(data=census['native-country'].unique(),columns=['Country'])
    native_country_map['map']=native_country_map['Country'].apply(lambda x:1 if x=='United-States' else 0)
    native_country_map=dict(zip(native_country_map['Country'],native_country_map['map']))
    data['is_US_native']=data['native-country'].map(native_country_map)
    
    data = pd.get_dummies(data,columns =['education','occupation'], dtype = int, drop_first=True)
    
    data['income_above_50K']=data['income'].map({'<=50K':0, '>50K':1})
    
    data.drop(labels=['income','sex','workclass','marital-status','relationship','race','native-country'],axis=1,inplace=True)
    
    return data


# In[ ]:


data_cleaner(cleaned_census).head()


# ## Train and test data cleaning

# In[ ]:


cleaned_train=data_cleaner(train)


# In[ ]:


cleaned_train.info()


# In[ ]:


cleaned_test=data_cleaner(test)


# In[ ]:


cleaned_test.info()


# I will assume the data cleaner function is working as expected as it didn't return any null values.

# ## Building the model

# In[ ]:


X=cleaned_train.drop(labels=['income_above_50K'],axis=1)
y=cleaned_train['income_above_50K']

test_X=cleaned_test.drop(labels=['income_above_50K'],axis=1)
test_y=cleaned_test['income_above_50K']


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('XGBClassifier', XGBClassifier()))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
#models.append(('MLPClassifier', MLPClassifier()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


model=GradientBoostingClassifier(random_state=0,learning_rate=0.2,n_estimators=200)
model.fit(X,y)
pred_y=model.predict(test_X)

from sklearn import metrics

cm = metrics.confusion_matrix(test_y, pred_y)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(test_y,pred_y))
plt.title(all_sample_title, size = 15);
plt.show()
print(metrics.classification_report(test_y,pred_y))


# In[ ]:


print(census['income_above_50K'].value_counts())
print(census['income_above_50K'].value_counts()[0]/sum(census['income_above_50K'].value_counts()))


# However, it is still working well than just guessing that everyone is below 50K by 13%

# TO DO:
# 1. Experiment with removing some columns - Capital gain/loss.
# 2. Adjust the train data such that it has balanced split between above and below 50K.
