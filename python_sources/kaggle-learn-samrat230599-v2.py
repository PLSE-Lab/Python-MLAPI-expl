#!/usr/bin/env python
# coding: utf-8

# In[ ]:


X.head()


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


from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , RandomForestClassifier , ExtraTreesClassifier , GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

#bring thte data to the dataframe
data = pd.read_csv('/kaggle/input/learn-together/train.csv')

#describe the data
data.describe()


# In[ ]:


#as id is not a feature or something, we can drop it 
data.drop(['Id'], axis = 1 , inplace = True)

data.describe()


# In[ ]:


data.columns


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

y = data['Cover_Type']
X = data.drop(['Cover_Type'] , axis = 1)
X = X.apply(lambda x : abs(x))
#apply SelectKBest class to extract top 16 best features
#16 has been taken after a bit of experimentation with different values
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 16 best features


# In[ ]:


y = data['Cover_Type']
print(type(y))


# In[ ]:


train = data
train.describe()
train = train.drop('Cover_Type' , axis = 1)


# In[ ]:


test = pd.read_csv('/kaggle/input/learn-together/test.csv')
id_ = test['Id']
test = test.drop('Id' , axis = 1)
test.describe()


# In[ ]:


features_chi = ['Horizontal_Distance_To_Roadways' , 'Horizontal_Distance_To_Fire_Points' ,
     'Elevation' , 'Horizontal_Distance_To_Hydrology' , 'Vertical_Distance_To_Hydrology' ,
     'Aspect' , 'Hillshade_3pm' , 'Hillshade_9am' , 'Slope' , 'Wilderness_Area4' , 'Soil_Type3' , 
     'Soil_Type10' , 'Soil_Type38' , 'Wilderness_Area1' , 'Soil_Type39' , 'Soil_Type40' , 
    'Soil_Type30' , 'Soil_Type29' , 'Hillshade_Noon' , 'Wilderness_Area3' ]


# In[ ]:


X = data[features_chi]
print(X.shape)
temp = pd.concat([X , data['Cover_Type']] , axis = 1)
#temp = X + data['Cover_Type']
print(temp.shape)


# Now we will check the distribution of train and test set.. As we can infer from the size of the datasets, that they are bound to have different distributions, so we will try to normalize both the data..

# In[ ]:





# In[ ]:


sns.set(rc={'figure.figsize':(12.7,10.27)})
sns.heatmap(temp.corr() , cmap = 'gist_ncar' , annot = True)


# In v_1, the combination of BaggingClassifier with base estimator as ExtraTreeClassifier worked the best.We will start off with that.

# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X , y , stratify = y , random_state = 5)


# In[ ]:


base = ExtraTreeClassifier(max_depth = 500)

for n in range(100 , 1500 ,100):
    clf = BaggingClassifier(base_estimator = base , n_estimators=n , random_state = 0)
    clf.fit(X_train , y_train)
    
    y_pred = clf.predict(X_test)
      
    print("For n_estimator =  " + str(n) , end = ' ')
    print(accuracy_score(y_pred , y_test))


# To go with this one, best combination 
# n_estimator = 800
# max_depth = 500

# In[ ]:


get_ipython().run_cell_magic('time', '', '#RandomForestClassifier\nfor n in range(100 , 1500 ,50):\n    clf = RandomForestClassifier(n_estimators=n , random_state = 0)\n    clf.fit(X_train , y_train)\n    \n    y_pred = clf.predict(X_test)\n      \n    print("For Max n_estimator =  " + str(n) , end = \' \')\n    print(accuracy_score(y_pred , y_test))')


# In[ ]:


test = test[features_chi]
print(test.shape)


# In[ ]:


for i in X.columns:
    #if i not in test.columns:
    print(type(X[i]))


# In[ ]:


sns.set(rc={'figure.figsize':(5,3.27)})
sns.distplot(train['Elevation'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Elevation'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


from sklearn.preprocessing import MinMaxScaler , StandardScaler
mms = MinMaxScaler()
ssc = StandardScaler()

mu1 = train[["Elevation"]].values.astype('float')
mu2 = test[["Elevation"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Slope'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Slope'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Aspect'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Aspect'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = train[["Slope"]].values.astype('float')
mu2 = test[["Slope"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Horizontal_Distance_To_Roadways'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Horizontal_Distance_To_Roadways'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = train[["Horizontal_Distance_To_Roadways"]].values.astype('float')
mu2 = test[["Horizontal_Distance_To_Roadways"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Horizontal_Distance_To_Fire_Points' ] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Horizontal_Distance_To_Fire_Points' ] , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = train[["Horizontal_Distance_To_Fire_Points"]].values.astype('float')
mu2 = test[["Horizontal_Distance_To_Fire_Points"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Horizontal_Distance_To_Hydrology'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Horizontal_Distance_To_Hydrology'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Vertical_Distance_To_Hydrology'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Vertical_Distance_To_Hydrology'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Hillshade_3pm'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Hillshade_3pm'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = train[["Hillshade_3pm"]].values.astype('float')
mu2 = test[["Hillshade_3pm"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Hillshade_9am'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Hillshade_9am'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = train[["Hillshade_9am"]].values.astype('float')
mu2 = test[["Hillshade_9am"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


sns.distplot(train['Hillshade_Noon'] , hist = False , norm_hist = True , color = "y")
sns.distplot(test['Hillshade_Noon'] , hist = False , norm_hist = True , color = "b")


# In[ ]:


#for i in train.columns:
    #print(type(train[i].iloc[0]))


# In[ ]:


mu1 = train[["Hillshade_Noon"]].values.astype('float')
mu2 = test[["Hillshade_Noon"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

sns.distplot(mu_1 , hist = False , norm_hist = True , color = "y")
sns.distplot(mu_2 , hist = False , norm_hist = True , color = "b")


# In[ ]:


mu1 = X[["Hillshade_9am"]].values.astype('float')
mu2 = test[["Hillshade_9am"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Hillshade_9am'] , axis = 1)
test = test.drop(['Hillshade_9am'] , axis = 1) 

X[['Hillshade_9am']] = pd.DataFrame(mu_1)
test[['Hillshade_9am']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Hillshade_Noon"]].values.astype('float')
mu2 = test[["Hillshade_Noon"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)

X = X.drop(['Hillshade_Noon'] , axis = 1)
test = test.drop(['Hillshade_Noon'] , axis = 1) 

X[['Hillshade_Noon']] = pd.DataFrame(mu_1)
test[['Hillshade_Noon']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Hillshade_3pm"]].values.astype('float')
mu2 = test[["Hillshade_3pm"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Hillshade_3pm'] , axis = 1)
test = test.drop(['Hillshade_3pm'] , axis = 1) 

X[['Hillshade_3pm']] = pd.DataFrame(mu_1)
test[['Hillshade_3pm']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Horizontal_Distance_To_Fire_Points"]].values.astype('float')
mu2 = test[["Horizontal_Distance_To_Fire_Points"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Horizontal_Distance_To_Fire_Points'] , axis = 1)
test = test.drop(['Horizontal_Distance_To_Fire_Points'] , axis = 1) 

X[['Horizontal_Distance_To_Fire_Points']] = pd.DataFrame(mu_1)
test[['Horizontal_Distance_To_Fire_Points']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Horizontal_Distance_To_Roadways"]].values.astype('float')
mu2 = test[["Horizontal_Distance_To_Roadways"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Horizontal_Distance_To_Roadways'] , axis = 1)
test = test.drop(['Horizontal_Distance_To_Roadways'] , axis = 1) 

X[['Horizontal_Distance_To_Roadways']] = pd.DataFrame(mu_1)
test[['Horizontal_Distance_To_Roadways']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Slope"]].values.astype('float')
mu2 = test[["Slope"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Slope'] , axis = 1)
test = test.drop(['Slope'] , axis = 1) 

X[['Slope']] = pd.DataFrame(mu_1)
test[['Slope']] = pd.DataFrame(mu_2)


# In[ ]:


mu1 = X[["Elevation"]].values.astype('float')
mu2 = test[["Elevation"]].values.astype('float')

mu_1 = ssc.fit_transform(mu1)
mu_2 = ssc.fit_transform(mu2)
print(mu_1.shape)

X = X.drop(['Elevation'] , axis = 1)
test = test.drop(['Elevation'] , axis = 1) 

X[['Elevation']] = pd.DataFrame(mu_1)
test[['Elevation']] = pd.DataFrame(mu_2)


# In[ ]:


model = BaggingClassifier(base_estimator = base , n_estimators=800 , random_state = 0)

print("Training")
model.fit(X , y)
print("Finished!!!")
print("Predicting")
pred = model.predict(test)


# In[ ]:


submission = pd.DataFrame({ 'Id': id_,
                            'Cover_Type': pred })
submission.to_csv("submission_example.csv", index=False)


# In[ ]:


submission['Cover_Type'].value_counts()


# In[ ]:




