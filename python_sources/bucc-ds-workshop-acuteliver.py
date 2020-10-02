#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
#visualizations
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#algorithms
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#score metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/liverfailure/ALF_Data.csv')
copy_df=df
df.head(10)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#dropping samples that dont have value fore ALF
df = df.dropna(axis = 0, subset=['ALF'])


# In[ ]:


total_missingvalues = df.isnull().sum()
total_missingvalues


# In[ ]:





# In[ ]:


#selecting a sample of the features for easier understanding; we do not do this while actual implementation.
df = df[['Age','Gender','Region','Weight','Height','Body Mass Index','Obesity','Waist',
         'Maximum Blood Pressure','Minimum Blood Pressure','Good Cholesterol','Bad Cholesterol',
         'Total Cholesterol','Dyslipidemia','PVD','ALF']]

df.head()


# In[ ]:


#refer to slide for heat map 
# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[ ]:


df.corr()


# In[ ]:


y = df['ALF']
df = df.drop('ALF',axis=1)
df.head()


# In[ ]:


total_missingvalues = df.isnull().sum()
total_missingvalues


# In[ ]:


#Taking care of missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN',strategy = 'median',axis=0)
imputer = imputer.fit(df.iloc[:,3:13]) #SELECTING THE COLUMN WITH MISSING VALUES
df.iloc[:,3:13] = imputer.transform(df.iloc[:,3:13])


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


#checking number of classes in the categorical feature
df['Region'].unique()


# In[ ]:


df['Gender'].unique()


# In[ ]:


#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
df.iloc[:,1] = labelencoder_X.fit_transform(df.iloc[:,1]) #SELECTING THE COLUMN WITH OBJECT TYPE

df=pd.get_dummies(df, columns=["Region"], prefix=["Region"])



# In[ ]:


df.head()


# In[ ]:


#dropping Region_west because the model can infer the values for this from the other 3 columns
df = df.drop('Region_west',axis = 1)
df.head()
X = df


# In[ ]:


#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


#splitting our dataset into training sets and teset sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y)


# In[ ]:


from keras import Sequential
from keras.layers import Dense


# In[ ]:


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=X_train.shape[1]))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


# In[ ]:


#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[ ]:


#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=10)


# In[ ]:


y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)


# In[ ]:


xgb = XGBClassifier(random_state=10)


# In[ ]:


xgb.fit(X_train,y_train)


# In[ ]:


pred = xgb.predict(X_test)


# In[ ]:


count = 0
for i in range( len(y_test) ):
    if pred[i] != y_test.iloc[i]: 
        count = count + 1


# In[ ]:


error = count/len(pred)
print( "Error for XGBoost= %f " % (error*100) + '%' )
accuracy = (1-error)
print( "Accuracy for XGBoost = %f " % (accuracy*100) + '%' )


# In[ ]:


rf = RandomForestClassifier(random_state=10)
rf.fit(X_train,y_train)
pred_rf = rf.predict(X_test)
count = 0
for i in range( len(y_test) ):
    if pred_rf[i] != y_test.iloc[i]: 
        count = count + 1
error = count/len(pred_rf)
print( "Error for RF = %f " % (error*100) + '%' )
accuracy = (1-error)
print( "Accuracy for RF = %f " % (accuracy*100) + '%' )


# In[ ]:


cv_results = cross_val_score(rf, X,y, cv = 4, scoring='neg_log_loss', n_jobs = -1)
cv_results


# In[ ]:


prt_string = "Log Loss: %f " % (-1*cv_results.mean())
                                                        
print(prt_string)


# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 2)
CV_rfc.fit(X_train, y_train)


# In[ ]:


best_param = CV_rfc.best_params_


# In[ ]:


rf = RandomForestClassifier(**best_param)


# In[ ]:


rf.fit(X_train,y_train)
pred_rf = rf.predict(X_test)
count = 0
for i in range( len(y_test) ):
    if pred_rf[i] != y_test.iloc[i]: 
        count = count + 1
error = count/len(pred_rf)
print( "Error for RF = %f " % (error*100) + '%' )
accuracy = (1-error)
print( "Accuracy for RF = %f " % (accuracy*100) + '%' )


# In[ ]:


print(copy_df.shape)
#dropping samples that dont have value fore ALF
copy_df = copy_df.dropna(axis = 0, subset=['ALF'])

y = copy_df['ALF']
df = copy_df.drop('ALF',axis=1)
X=df


# In[ ]:





# In[ ]:


X = X.drop(['Gender','Region','Source of Care'], axis=1)
X.head()


# In[ ]:


# imputer = Imputer(missing_values = 'NaN',strategy = 'median',axis=0)
# imputer = imputer.fit(X.iloc[:,:]) #SELECTING THE COLUMN WITH MISSING VALUES
# X.iloc[:,:] = imputer.transform(X.iloc[:,:])


# In[ ]:





# In[ ]:


# rf = RandomForestClassifier(**best_param)
# rf.fit(X, y)
# print(rf.feature_importances_)


# In[ ]:


# variable = [ ]
# name=[]
# for i in range(len(rf.feature_importances_)):
    
#     if (rf.feature_importances_[i] >=0.03):
#         variable.append(i)
#         name.append(rf.feature_importances_[i])
# print(variable)
# print(name)

# print(len(variable))


# In[ ]:


# X=X.iloc[:,variable]


# In[ ]:


# X.head()


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y)
# rf = RandomForestClassifier(**best_param)
# rf.fit(X_train,y_train)
# pred_rf = rf.predict(X_test)
# count = 0
# for i in range( len(y_test) ):
#     if pred_rf[i] != y_test.iloc[i]: 
#         count = count + 1
# error = count/len(pred_rf)
# print( "Error for RF = %f " % (error*100) + '%' )
# accuracy = (1-error)
# print( "Accuracy for RF = %f " % (accuracy*100) + '%' )


# In[ ]:


# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
# test = SelectKBest(score_func=f_classif, k=10)
# kbestfit = test.fit(X, Y)
# kbestbool=kbestfit.get_support()
# count=0
# kbestchi_feature=[]
# for i in kbestbool:
#     if i:
#         kbestchi_feature.append(count)
#     count=count+1
# print(len(kbestchi_feature))
# print(kbestchi_feature)

