#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival with Python
#         
#    **Objective is to develop a classification model to predict, if a passenger survives or perishes. Let's begin our understanding of the dataset followed by widely used classification algorithm.**
# 
# ## Importing Libraries
# 
#    Let's import libraries to get started!

# In[ ]:


import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics as met
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Dataset - Train and Test Dataset. Find columns in testing and training set and Number of records available in each set.
# 
# Import the train and test dataset and verify the columns available.

# In[ ]:


training_set = pd.read_csv('../input/train.csv')
testing_set = pd.read_csv('../input/test.csv')
pID = testing_set['PassengerId']


# In[ ]:


print(training_set.shape)


# In[ ]:


print(testing_set.shape)


# In[ ]:


print(training_set.columns)


# In[ ]:


print(testing_set.columns)


#   # Data Exploration :
#   
#   ### 'Survived' column is the target variable which needs to be predicted and is not present in testing set.
#   

# In[ ]:


training_set.head()


# In[ ]:


training_set.describe()


# **By Using .info() command, we can notice that "Age" and "Cabin" column have missing values.**
# 
# **Roughly 20 percent values in Age column is missing. We can make a reasonable impution on Age column. But Cabin column, we are missing approx 80% values. We'll drop the cabin column.**

# ## Plot the Data

# **Performing basic visulization with the help of Seaborn.**

# In[ ]:


sns.heatmap(training_set.isnull(),yticklabels=False,cbar=False,cmap='Dark2')


# In[ ]:


sns.heatmap(testing_set.isnull(),yticklabels=False,cbar=False,cmap='Dark2')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=training_set,palette='RdBu_r')


# **The plot between Gender and Target variable, clearly suggest that more men have suffered the fate of Jack from Titanic.**

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=training_set,palette='RdBu_r')
plt.title("Gender vs Survived")
plt.legend(loc = 'top left')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=training_set,palette='rainbow')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=training_set,palette='Dark2')
plt.legend(loc = 'top left',bbox_to_anchor=(1.2, 1.2))


# In[ ]:


sns.distplot(training_set['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=training_set)


# In[ ]:


training_set['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ## Data Cleaning
# 
# Fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However, we can be smarter about this and check the average age by passenger class.

# In[ ]:


print(training_set.isnull().sum(),"\n")
print(testing_set.isnull().sum())


# **Feature Engineering - https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy**

# In[ ]:


for dataset in [training_set,testing_set]:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
training_set.drop(drop_column, axis=1, inplace = True)
testing_set.drop(drop_column, axis=1, inplace = True)
print(training_set.isnull().sum())
print("-"*10)
print(testing_set.isnull().sum())


# In[ ]:


###CREATE: Feature Engineering for train and test/validation dataset
for dataset in [training_set,testing_set]:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    #dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


    
#cleanup rare title names
# #print(data1['Title'].value_counts())
# stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
# title_names = (training_set['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

# #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
# training_set['Title'] = training_set['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# print(training_set['Title'].value_counts())
# print("-"*10)


#preview data again
training_set.info()
testing_set.info()
training_set.sample(10)


# In[ ]:


training_set[training_set["Name"].str.contains("Master")]


# In[ ]:


#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

#code categorical data
label = LabelEncoder()
for dataset in [training_set,testing_set]:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    #dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
training_set_x = ['Sex','Pclass', 'Embarked','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
training_set_x_calc = ['Sex_Code','Pclass', 'Embarked_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
training_set_xy =  Target + training_set_x
print('Original X Y: ', training_set_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
training_set_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
training_set_xy_bin = Target + training_set_x_bin
print('Bin X Y: ', training_set_xy_bin, '\n')


#define x and y variables for dummy features original
training_set_dummy = pd.get_dummies(training_set[training_set_x],drop_first=True)
training_set_x_dummy = training_set_dummy.columns.tolist()
training_set_xy_dummy = Target + training_set_x_dummy
print('Dummy X Y: ', training_set_xy_dummy, '\n')

training_set_dummy.head()


# In[ ]:


y = training_set['Survived']
X = training_set_dummy


# In[ ]:


testing_set_dummy = pd.get_dummies(testing_set[training_set_x],drop_first=True)


# In[ ]:


ss = MinMaxScaler()
#ss = StandardScaler()
training_set_dummy_ss= ss.fit_transform(training_set_dummy)
testing_set_dummy_ss= ss.fit_transform(testing_set_dummy)


# In[ ]:


# pca = PCA(n_components=6)
# X_train_pca = pca.fit_transform(training_set_dummy)
# X_test_pca = pca.transform(testing_set_dummy)


# In[ ]:


# transformer = FastICA()
# X_train_ica = transformer.fit_transform(training_set_dummy)


# In[ ]:


# X_test_ica = transformer.transform(testing_set_dummy)


# # Building Machine learning Models :

# In[ ]:


# Models
classifiers = {'Gradient Boosting Classifier':GradientBoostingClassifier(),'Adaptive Boosting Classifier':AdaBoostClassifier(),'RadiusNN':RadiusNeighborsClassifier(radius=40.0),
               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(), 'GaussianNB': GaussianNB(), 'BerNB': BernoulliNB(), 'KNN': KNeighborsClassifier(),
               'Random Forest Classifier': RandomForestClassifier(min_samples_leaf=10,min_samples_split=20,max_depth=4),'Decision Tree Classifier': DecisionTreeClassifier(),'Logistic Regression':LogisticRegression(), "XGBoost": xgb.XGBClassifier()}


# ## Sampling Data

# ### Train test split

# In[ ]:


X_training, X_validating, y_training, y_validating = train_test_split(training_set_dummy, y, test_size=0.20, random_state=11)


# In[ ]:


base_accuracy = 0
for Name,classify in classifiers.items():
    classify.fit(X_training,y_training)
    y_predictng = classify.predict(X_validating)
    print('Accuracy Score of '+str(Name) + " : " +str(met.accuracy_score(y_validating,y_predictng)))
    if met.accuracy_score(y_validating,y_predictng) > base_accuracy:
        predictions_test = classify.predict(testing_set_dummy)
        base_accuracy = met.accuracy_score(y_validating,y_predictng)
    else:
        continue

# Generate Submission File 
predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
                        'Survived': predictions_test })
predicted_test_value.to_csv("PredictedTestScore.csv", index=False)


# ### Stratified KFold Sampling

# In[ ]:


# skfold = StratifiedKFold(n_splits=2,random_state=42,shuffle=True)
# for Name,classify in classifiers.items():
#     for train_KF, test_KF in skfold.split(X,y):
#         X_train,X_test = X.iloc[train_KF], X.iloc[test_KF]
#         y_train,y_test = y.iloc[train_KF], y.iloc[test_KF]
#         classify.fit(X_train,y_train)
#         y_pred = classify.predict(X_test)
#         print('Accuracy Score of '+str(Name) + " : " +str(met.accuracy_score(y_test,y_pred)))
#         print(classification_report(y_test,y_pred))


# ### Stratified Shuffle Split

# In[ ]:


# sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=1)
# for Name,classify in classifiers.items():
#     for train_KF, test_KF in sss.split(X,y):
#         X_train,X_test = X.iloc[train_KF], X.iloc[test_KF]
#         y_train,y_test = y.iloc[train_KF], y.iloc[test_KF]
#         classify.fit(X_train,y_train)
#         y_pred = classify.predict(X_test)
#         print('Accuracy Score of '+str(Name) + " : " +str(met.accuracy_score(y_test,y_pred)))
#         print(classification_report(y_test,y_pred))


# ### GridSearchCV

# **GridSearchCV for SVC**

# In[ ]:


# param_grid = {'C':[5000],'gamma':[0.0001]}
# gscv = GridSearchCV(SVC(),param_grid)
# gscv.fit(X_training,y_training)
# predictions = gscv.predict(X_validating)
# print(met.accuracy_score(y_validating,predictions))
# print(gscv.best_params_)
# print(gscv.best_score_)


# **GridSearchCV for Gradient Boosting Classifier**

# In[ ]:


# param_grid = {'learning_rate':[0.1],"n_estimators":[40],'min_samples_leaf':[15],'min_samples_split':[45],"max_depth":[3],'loss': ['deviance'],"max_features":["auto"]}
# gbccv = GridSearchCV(GradientBoostingClassifier(),param_grid)
# gbccv.fit(X_training,y_training)
# predictions_train = gbccv.predict(X_validating)
# print(met.accuracy_score(y_validating,predictions_train))
# print(gbccv.best_params_)
# print(gbccv.best_score_)


# **GridSearchCV for XGBoost**

# In[ ]:


# param_grid = {'learning_rate':[0.1],'gamma':[0.4],"n_estimator":[10],"max_depth":[3]}
# xgbcv = GridSearchCV(xgb.XGBClassifier(),param_grid)
# xgbcv.fit(X_training,y_training)
# predictions_train = xgbcv.predict(X_validating)
# print(met.accuracy_score(y_validating,predictions_train))
# print(xgbcv.best_params_)
# print(xgbcv.best_score_)


# **GridSearchCV for Random Forest Classifier**

# In[ ]:


# param_grid = {'min_samples_leaf':[10],'min_samples_split':[20],"max_depth":[5]}
# xgbcv = GridSearchCV(RandomForestClassifier(),param_grid)
# xgbcv.fit(X_training,y_training)
# predictions_train = xgbcv.predict(X_validating)
# print(met.accuracy_score(y_validating,predictions_train))
# print(xgbcv.best_params_)
# print(xgbcv.best_score_)


# In[ ]:


cbr = xgb.XGBClassifier()#logging_level='Silent'
cbr.fit(X_training,y_training)
predictions_train = cbr.predict(X_validating)
print(met.accuracy_score(y_validating,predictions_train))
#print(QDA.best_params_)
#print(QDA.best_score_)


# In[ ]:


clf1 = GradientBoostingClassifier()
#clf2 = CatBoostClassifier(logging_level='Silent')
clf3 = LinearDiscriminantAnalysis()
clf4 = LogisticRegression()
clf5 = xgb.XGBClassifier()
exTreeClf = VotingClassifier(estimators=[('svc', clf1), ('gbc', clf3),('lr',clf4),('lda',clf5)])
exTreeClf.fit(X_training,y_training)
# y_pred = exTreeClf.predict(X_validating)
# print(met.accuracy_score(y_validating,y_pred))


# In[ ]:



# predictions_test = exTreeClf.predict(testing_set_dummy)
# predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
#                         'Survived': predictions_test })
# predicted_test_value.to_csv("PredictedTestScore.csv", index=False)


# **Deep Learning Model**

# In[ ]:


# model = Sequential()
# model.add(Dense(32, input_dim=10, activation='selu',kernel_initializer='uniform'))
# model.add(Dropout(rate=0.4))
# model.add(Dense(16, activation='selu'))
# model.add(Dropout(rate=0.4))
# model.add(Dense(8, activation='selu'))
# model.add(Dropout(rate=0.4))
# model.add(Dense(1, activation='sigmoid'))
# opt = keras.optimizers.Adadelta()
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[ ]:


# import keras
# keras.optimizers.
# keras.activations.


# In[ ]:


# #fit the keras model on the dataset
# model.fit(X_training, y_training, epochs=500, batch_size=50,validation_data=(X_validating,y_validating),verbose=1)


# In[ ]:


# y_pred = model.predict_classes(X_validating)
# met.accuracy_score(y_validating,y_pred)


# In[ ]:


# predicted_test = []
# for x in model.predict_classes(testing_set_dummy):
#     predicted_test.append(x[:][0])


# In[ ]:


# predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
#                         'Survived': predicted_test })
# predicted_test_value.to_csv("PredictedTestScore.csv", index=False)


# ### Pseudo-Labeling Technique - https://towardsdatascience.com/simple-explanation-of-semi-supervised-learning-and-pseudo-labeling-c2218e8c769b

# In[ ]:


# xgboost = CatBoostClassifier()
# xgboost.fit(training_set_dummy,y)


# In[ ]:


#test_index_with_80p = list(np.argwhere(xgboost.predict_proba(testing_set_dummy)>0.75)[:,0])


# In[ ]:


#y_pred_with_80p = pd.Series(list(np.argwhere(xgboost.predict_proba(testing_set_dummy)>0.75)[:,1]))


# In[ ]:


# for idx in test_index_with_80p:
#     training_set_dummy = training_set_dummy.append(testing_set_dummy.iloc[idx],ignore_index=True)


# In[ ]:


#y = y.append(y_pred_with_80p,ignore_index=True)


# In[ ]:


# lda = CatBoostClassifier()
# lda.fit(training_set_dummy,y)


# In[ ]:


predicted_test = []
for x in exTreeClf.predict(testing_set_dummy):
    predicted_test.append(x)
predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
                        'Survived': predicted_test })
predicted_test_value.to_csv("PredictedTestScore.csv", index=False)


# ### Feature Importance using Permutation Importance - https://www.kaggle.com/dansbecker/permutation-importance

# In[ ]:


# perm = PermutationImportance(xgboost, random_state=1).fit(X_validating, y_validating)
# eli5.show_weights(perm, feature_names = X_validating.columns.tolist())


# In[ ]:


# training_set_dummy.drop(columns=["Fare","SibSp","IsAlone","Embarked_Q"],inplace=True)
# testing_set_dummy.drop(columns=["Fare","SibSp","IsAlone","Embarked_Q"],inplace=True)


# In[ ]:


# exTreeClf.fit(training_set_dummy,y)
# predicted_test = []
# for x in exTreeClf.predict(testing_set_dummy):
#     predicted_test.append(x)
# predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
#                         'Survived': predicted_test })
# predicted_test_value.to_csv("PredictedTestScore.csv", index=False)


# In[ ]:




