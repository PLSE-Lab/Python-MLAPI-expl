#!/usr/bin/env python
# coding: utf-8

# # 1. Importing libraries and data

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#import data
train = pd.read_csv('../input/train.csv',index_col=0)
test = pd.read_csv('../input/test.csv',index_col=0)

#Check what it looks like
print(train.shape)
print(test.shape)
train.head(5)


# In[ ]:


# Combine the train and test for pre-processing
df_all = pd.concat([train.drop('Survived',axis=1),test])
print(df_all.shape)
df_all.head(5)


# In[ ]:


# Save y_train for later
y_train = train.Survived


# # 2. Data exploratory

# In[ ]:


#Number and types of columns
df_all.info()


# In[ ]:


#Look at statistics
df_all.describe()


# In[ ]:


#Look for missing data
plt.figure(figsize=[20,10])
sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False)


# In[ ]:


# No. of nulls
df_all.isnull().sum(axis = 0)


# In[ ]:


# Looking at train data - Comparing survival by features
plt.figure(figsize=[20,5])
plt.subplot(1,3,1)
sns.countplot(x="Sex",data=train, hue="Survived")
plt.title('Gender comparison')

plt.subplot(1,3,2)
sns.countplot(x='Pclass',data=train, hue="Survived")
plt.title('Class comparison')

plt.subplot(1,3,3)
sns.countplot(x='Embarked',data=train, hue="Survived")
plt.title('Class comparison')


# In[ ]:


#Looking at spreads in data
plt.figure(figsize=[20,10])
plt.subplot(2,2,1)
sns.distplot(df_all['Age'].dropna(),kde=True,bins=10)
plt.xlim(0)
plt.title('Age spread')

plt.subplot(2,2,2)
sns.distplot(df_all['Fare'].dropna(),kde=True,bins=50)
plt.xlim(0)
plt.title('Fare spread')

plt.subplot(2,2,3)
sns.countplot(x='SibSp',data=df_all)
plt.title('Sibling spread')

plt.subplot(2,2,4)
sns.countplot(x='Parch',data=df_all)
plt.title('Parent spread')


# # 3. Data cleaning/ feature creation

# ## 3.1 Predicting age for missing values

# In[ ]:


#Comparing age by features
plt.figure(figsize=[15,5])
plt.subplot(1,4,1)
sns.boxplot(x='Pclass',y='Age',data=df_all)
plt.title('Class comparison')

plt.subplot(1,4,2)
sns.boxplot(x='Sex',y='Age',data=df_all)
plt.title('Sex comparison')

plt.subplot(1,4,3)
sns.boxplot(x='SibSp',y='Age',data=df_all)
plt.title('Sibling comparison')

#Parents not used
plt.subplot(1,4,4)
sns.boxplot(x='Parch',y='Age',data=df_all)
plt.title('Parent comparison')


# In[ ]:


#Considered fare but not progressed with this as no strong correlation
plt.figure(figsize=[20,5])
sns.lmplot(x='Age',y='Fare',data=df_all)


# In[ ]:


#Creating a feature and observation set for a simple prediction model
df_all_age = df_all.drop(['Name','Parch','Ticket','Fare','Cabin','Embarked'],axis=1).dropna()
df_all_age_features = df_all_age.drop('Age',axis=1)
df_all_age_observations=df_all_age['Age']
print(df_all_age_features.shape)
print(df_all_age_observations.shape)
df_all_age_features.head(5)


# In[ ]:


# For the prediction, these categories need to be turned into numerical dummies

#Split passenger classes
pclasses = pd.get_dummies(df_all_age_features['Pclass'],drop_first=True)

#Split sex
genders = pd.get_dummies(df_all_age_features['Sex'],drop_first=True)

#e.g. for genders (we only need to know if it's a male, female is base or 0))
genders.head(5)         


# In[ ]:


# Remove categorical values
df_all_age_features.drop(['Pclass','Sex'],axis=1,inplace=True)

#Concatinate the dummie columns
df_all_age_features_2 = pd.concat([df_all_age_features,pclasses,genders],axis=1)
df_all_age_features_2.head(5)


# In[ ]:


# As we know what the observations are, we can fit a simple model
df_all_age_observations.head(5)


# In[ ]:


# Set up the linear model and fit the data
from sklearn.linear_model import LinearRegression
predict_age = LinearRegression()
predict_age.fit(df_all_age_features_2,df_all_age_observations)

# Check that the coefficients and intercept look appropriate
print(df_all_age_features_2.columns.values)
print(predict_age.coef_)
print(predict_age.intercept_)


# In[ ]:


#define predictive function

def add_age(mylist):
    Age = mylist['Age']
    Pclass = mylist['Pclass']
    SibSp = mylist['SibSp']
    if mylist['Sex'] == 'Male':
        SibSp = 1
    else:
        Sex = 0
    
    # If the age value is missing, predict age
    if pd.isnull(Age):
        if Pclass == 1:
            predicted_age = predict_age.predict([[SibSp,0,0,Sex]]).round()[0]

        elif Pclass == 2:
            predicted_age = predict_age.predict([[SibSp,1,0,Sex]]).round()[0]

        else:
            predicted_age = predict_age.predict([[SibSp,0,1,Sex]]).round()[0]

    # If the age value is present, then keep
    else:
        predicted_age = Age
    
    # There may be predictions that are below 0. We know this is impossible, so return a suitably low age e.g. 1.
    if predicted_age < 0:
        return 1
    else: 
        return predicted_age


# In[ ]:


# Apply this function to our data to predict ages
df_all['Age_predict'] = df_all[['Age','Pclass','SibSp','Sex']].apply(add_age, axis =1)


# In[ ]:


#Check age distibution of passengers (known vs predicted)
plt.figure(figsize=[20,5])
sns.distplot(df_all['Age_predict'],kde=True,bins=20,color='Red',label="Predicted ages")
sns.distplot(df_all['Age'].dropna(),kde=True,bins=20,color='Blue',label="Known ages")
plt.xlim(-5)
plt.legend()


# We can see that more are predicted nearer the average (~25) which is to be expected. This is fine for this purpose.

# In[ ]:


#Showing predicted ages has no nulls now (we will remove age later)
plt.figure(figsize=[20,5])
sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False)


# In[ ]:


# Looking at data again, notice the new column
df_all.head(5)


# In[ ]:


# Last sense check of predicted ages
print(df_all['Age'].min())
print(df_all['Age'].max())
print(df_all['Age_predict'].min())
print(df_all['Age_predict'].max())


# In[ ]:


df_all.head(5)


# In[ ]:


df_all.drop(['Age'],axis=1,inplace=True)


# ## 3.2 Creating features from Cabin field

# In[ ]:


# Exploratory
df_all['Cabin'].describe()


# In[ ]:


#Extract cabin numbers and letters from Cabin field
df_all['Cabin_deck_letter'] =df_all.Cabin.str.extract('(\D+)')


# In[ ]:


# Looking at number of unique values
df_all['Cabin_deck_letter'].unique()


# In[ ]:


# Fill blanks with unknown
df_all['Cabin_deck_letter'].fillna('Unknown', inplace = True)


# In[ ]:


# Look at if there is any correlation between cabin deck letter and survived
plt.figure(figsize=[20,5])
sns.countplot(x='Cabin_deck_letter',data=pd.concat([df_all[:train.shape[0]],y_train],axis=1), hue='Survived')


# Hmm, nothing spectacular, just a lot more chance of surviving if you had a cabin

# In[ ]:


# To keep the features simpler we're just going to include cabin yes/no
df_all['Cabin_yes'] = df_all['Cabin_deck_letter'].apply(lambda x: 1 if x is not 'Unknown' else 0)
df_all['Cabin_yes'].head(5)


# In[ ]:


# Confirm there is a correlation between having a cabin and surviving
plt.figure(figsize=[15,5])
sns.countplot(x='Cabin_yes',data=pd.concat([df_all[:train.shape[0]],y_train],axis=1), hue='Survived')


# In[ ]:


# Drop the others
df_all.drop(['Cabin','Cabin_deck_letter'],axis=1,inplace = True)


# ## 3.3 Creating features from name column

# In[ ]:


# Checking what we're dealing with
df_all['Name'].describe()


# How strange there are two Miss Kate Connollys 

# In[ ]:


# Reminder what it looks like
df_all['Name'].head()


# In[ ]:


# Create some features
df_all['Title'] = df_all.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0]).str.replace(" ","") #remove whitespace
df_all['First_names'] = df_all.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[1])
df_all['Surname'] = df_all.Name.str.split(',').apply(lambda x: x[0])


# In[ ]:


# Let's have a look
df_all[['Title','First_names','Surname']].head(5)


# In[ ]:


#Exploring titles
df_all.Title.unique()


# In[ ]:


#Exploring titles
df_all.Title.value_counts()


# In[ ]:


# A lot of these are similar so will combine
df_all.Title.replace(['Mme','Ms','Mlle'], 'Miss',inplace=True)
df_all.Title.replace(['Rev','Capt'], 'Mr',inplace=True)
df_all.Title.replace(['Dona','theCountess'], 'Lady',inplace=True)
df_all.Title.replace(['Col','Major','Jonkheer','Don'], 'Sir',inplace=True)

# Check now
df_all.Title.value_counts()


# In[ ]:


# Survival chances vs title
plt.figure(figsize=[20,5])
plt.subplot(2,2,1)
sns.countplot(x='Title',hue='Survived',data=pd.concat([df_all[:train.shape[0]],y_train],axis=1))
plt.ylim(0)

plt.subplot(2,2,2)
sns.countplot(x='Title',hue='Survived',data=pd.concat([df_all[:train.shape[0]],y_train],axis=1))
plt.ylim(0,10)
plt.title('Zoomed')


# However, we do need to be careful about correlation with sex. We'll sort this out later.

# In[ ]:


# Can't do anthing too insightful with names (possibly combine families) so will drop
df_all.drop(['Name','First_names','Surname'],axis=1,inplace = True)


# ## 3.4 Ticket

# In[ ]:


# Not sure how to use this, will drop
df_all.drop('Ticket',axis=1,inplace=True)


# In[ ]:


df_all.isnull().sum(axis = 0)


# ## 3.5 Filling the last gaps

# In[ ]:


#Replace missing fare with mean
df_all.Fare = df_all.Fare.fillna(value = df_all.Fare.mean())

# Replace embarked with most common port
df_all.Embarked = df_all.Embarked.fillna(value = df_all.Embarked.value_counts().index[0])


# In[ ]:


#Checking no more missing
df_all.isnull().sum(axis = 0)


# ## 3.6 Checking for skewness/ Homoscedasticity

# In[ ]:


from scipy.stats import skew
from scipy.stats import yeojohnson

# Calculate skewness
skewed_feats = df_all[['Fare','Age_predict']].apply(lambda x: skew(x)) 
skewed_feats.sort_values()


# In[ ]:


# Looking at transforming it to reduce skewness

plt.figure(figsize=[20,5])

# Histogram plot (normal)
plt.subplot(1,2,1)
sns.distplot(df_all.Fare)
plt.title('Standard')

# Histogram plot (log)
plt.subplot(1,2,2)
sns.distplot(np.log1p(df_all.Fare))
plt.title('Log transformation')

# Skewness and kurtosis
print("Skewness (before): %f" % df_all.Fare.skew())
print("Kurtosis (before): %f" % df_all.Fare.kurt())

print("Skewness (log): %f" % np.log1p(df_all.Fare).skew())
print("Kurtosis (log): %f" % np.log1p(df_all.Fare).kurt())


# In[ ]:


# We'll log transform fare
df_all.Fare = np.log1p(df_all.Fare)


# # 4. Preparing data for predictions

# ## 4.1 Categoric variables into dummies

# In[ ]:


df_all.head(5)


# In[ ]:


# Add dummies for categoric variables
Pclass_dummy = pd.get_dummies(df_all['Pclass'],drop_first = False)
Sex_dummy = pd.get_dummies(df_all['Sex'],drop_first = True)
Embarked_dummy = pd.get_dummies(df_all['Embarked'],drop_first = False)
Title_dummy = pd.get_dummies(df_all['Title'],drop_first = False)


# In[ ]:


# Drop categoric columns
df_all.drop(['Title','Pclass','Sex','Embarked'],axis=1,inplace=True)


# In[ ]:


# Add dummy columns. This is the feature dataframe.
df_all = pd.concat([df_all,Pclass_dummy,Sex_dummy,Embarked_dummy,Title_dummy],axis=1)
df_all.head(10)


# ## 4.2 Split into Train and Test

# In[ ]:


# Split back
X_train = df_all[:train.shape[0]]
X_test = df_all[train.shape[0]:]

# Check they are the same shape as started. They are, great.
print(X_train.shape)
print(X_test.shape)


# In[ ]:


# The observation series
print(y_train.shape)
y_train.head(5)


# ## 4.3 Scale data

# In[ ]:


X_train.head(5)


# In[ ]:


# Fit scaler (using RobustScaler to reduce effect of outliers) to training set mean and variance
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train)


# In[ ]:


# Transform both the training and testing sets
scaled_features_train = scaler.transform(X_train)
scaled_features_test = scaler.transform(X_test)

# Put scaled data back into a pandas dataframe
X_train = pd.DataFrame(scaled_features_train,columns = X_train.columns)
X_test = pd.DataFrame(scaled_features_test,index = X_test.index, columns = X_test.columns)
X_train.head(5)


# ## 4.4 Checking co-linearity

# In[ ]:


#Check the correlation between variables
plt.figure(figsize=[20,10])
Z=pd.concat([X_train,y_train],axis=1)
sns.heatmap(Z.corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")


# There is a strong argument that we should drop 'male' as this information is inferred from the title.
# Likewise 1st class and having a cabin.

# # 5. Predictions

# In[ ]:


from sklearn.model_selection import cross_val_score


# ## 5.1 Split the training data into a t_train and a t_test

# In[ ]:


from sklearn.model_selection import train_test_split
Xt_train, Xt_test, yt_train, yt_test, = train_test_split(X_train, y_train, test_size = 0.3,random_state = 27)


# ## 5.2 Define general evaluation method

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model_list = []
accuracy_list = []

## Define a testing procedure

def testing_model(model):
    #Fit model on t_train sets (we'll be using all data for final model, but should be fairly reflective)
    model.fit(Xt_train,yt_train)
    #Make prediction
    predictions = model.predict(Xt_test)
    
    # Evaluation
    print(model)
    model_list.append(model)
    # Print predictive accuracy
    print('\nPredictive accuracy: ' + str(round(accuracy_score(yt_test,predictions),3)))
    accuracy_list.append(round(accuracy_score(yt_test,predictions),3))
    # Return confusion matrix
    return confusion_matrix(yt_test,predictions, labels=None, sample_weight=None)


# ## 5.3 Define general improvement method

# In[ ]:


# Optimise the model parameters using a parameter grid and cross-validation

from sklearn.model_selection import GridSearchCV

# Requires parameters in form e.g. {'learning_rate': [0.01,0.03,0.1],'max_depth': [3,5],'min_child_weight': [2,4,6]}

def optimise_model(model, para_dict):

    model_grid = GridSearchCV(model,
                            para_dict,
                            cv = 5,
                            n_jobs = 8,
                            verbose=True)

    # This can be done on the entire training set as it just returns parameter tuning values, should not overfit once fitted after
    model_grid.fit(X_train,y_train)
    
    # Return best model
    print('CV best accuracy: ' + str(round(model_grid.best_score_,2)))
    print(model_grid.best_params_)
    return model_grid.best_params_


# ## 5.4 Logistic regression

# In[ ]:


# import the class
from sklearn.linear_model import LogisticRegression


# ### 5.4.1 Default parameters

# In[ ]:


# Instantiate model
logreg = LogisticRegression()

# Evaluate model
testing_model(logreg)


# ### 5.4.2 Parameter hypertuning

# In[ ]:


# Test the following parameters
test_param = {'solver': ['liblinear','lbfgs'],'C': np.logspace(start=-1,stop=2,base=10,num=10)}

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(logreg,test_param)

# Puts these optimised parameters in the new model
logreg_2 = LogisticRegression(solver=opt_param['solver'],C=opt_param['C'])


# In[ ]:


# Compare the following score with before
testing_model(logreg_2)


# In[ ]:


len(X_train.columns)


# In[ ]:


a=[[1,2],[3,4]]
len(a)


# In[ ]:


# Have a look at feature importances
pd.DataFrame(data={'Feature': X_train.columns, 'Importance': logreg_2.coef_[0]}).sort_values(by=['Importance'], ascending=False)


# Nope. Turns out it couldn't be improved (looking at the parameters I tried)

# ## 5.5 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Instantiate model
rfc = RandomForestClassifier(max_depth=None)

# Evaluate model
testing_model(rfc)


# In[ ]:


# Test the following parameters
test_param = {'max_depth': [None,4,8,12,16],'max_features': ['auto',4,8,12,16],'max_leaf_nodes': [None,4,8,12,16]}

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(rfc,test_param)

# Puts these optimised parameters in the new model
rfc_2 = RandomForestClassifier(max_depth=opt_param['max_depth'],max_features=opt_param['max_features'],max_leaf_nodes=opt_param['max_leaf_nodes'])


# In[ ]:


# Compare the following score with before
testing_model(rfc_2)


# In[ ]:


# Have a look at feature importances
pd.DataFrame(data={'Feature': X_train.columns, 'Importance': rfc_2.feature_importances_}).sort_values(by=['Importance'], ascending=False)


# ## 5.6 Support vector machines

# In[ ]:


from sklearn.svm import SVC

# Instantiate model
svc = SVC(probability=True)

# Evaluate model
testing_model(svc)


# In[ ]:


# Test the following parameters
test_param = {'kernel': ['linear','poly','rbf','sigmoid'],'C': [0.01,0.1,1],'gamma': ['auto',0.01, 0.1, 1]}

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(svc,test_param)

# Puts these optimised parameters in the new model
svc_2 = SVC(kernel=opt_param['kernel'],C=opt_param['C'],gamma=opt_param['gamma'],probability=True) #probability required for stacking 'soft'


# In[ ]:


# Compare the following score with before
testing_model(svc_2)


# ## 5.7 K-nearest neighbour

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# Instantiate model
knn = KNeighborsClassifier()

# Evaluate model
testing_model(knn)


# In[ ]:


# Test the following parameters
test_param = {'n_neighbors': [4,8,12,16,20],'weights': ['uniform','distance']}

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(knn,test_param)

# Puts these optimised parameters in the new model
knn_2 = KNeighborsClassifier(n_neighbors=opt_param['n_neighbors'],weights=opt_param['weights'])


# In[ ]:


# Compare the following score with before
testing_model(knn_2)


# ## 5.8 XGBClassifier

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Instantiate model
xgbc = xgb.XGBClassifier()

# Evaluate model
testing_model(xgbc)


# In[ ]:


# Test the following parameters
test_param = {'learning_rate': [0.01,0.03,0.1],
              'max_depth': [3,5,7],
              'min_child_weight': [4,6,8],
              'n_estimators': [1000]}

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(xgbc,test_param)

# Puts these optimised parameters in the new model
xgbc_2 = XGBClassifier(learning_rate=opt_param['learning_rate'],max_depth=opt_param['max_depth'],min_child_weight=opt_param['min_child_weight'],n_estimators=opt_param['n_estimators'])


# In[ ]:


# Compare the following score with before
testing_model(xgbc_2)


# # 6. Stacking models

# In[ ]:


df_eval = pd.DataFrame(data={'model': model_list, 'Accuracy score': accuracy_list})
df_eval


# In[ ]:


from sklearn.ensemble import VotingClassifier
estimators=[ ('xgbc',xgbc_2),('lr', logreg_2), ('svc', svc_2), ('rf', rfc_2), ('knn',knn_2)]

vc = VotingClassifier(estimators=estimators, voting='hard')

# Evaluate model
testing_model(vc)


# In[ ]:


# Test the following parameters
test_param = {'voting': ['hard','soft'],
    'weights': [[1,1,1,1,1],[2,1,1,1,1],[1,2,1,1,1],[1,1,2,1,1],[1,1,1,2,1],[2,2,1,1,1],[3,2,1,1,1],[2,3,1,1,1]],
              'n_jobs':[8]
             }

# Runs the CVgrid function earlier defined. Returns a dictionary of best parameters
opt_param = optimise_model(vc,test_param)

# Puts these optimised parameters in the new model
vc_2 = VotingClassifier(estimators=estimators,voting=opt_param['voting'],weights=opt_param['weights'])


# In[ ]:


# Compare the following score with before
testing_model(vc_2)


# Looks pretty good, let's test on the real thing

# # 7. Testing

# In[ ]:


final_predict = vc_2.predict(X_test)


# In[ ]:


# Put into dataframe
d = {'PassengerId': X_test.index, 'Survived': final_predict}
predictions_df = pd.DataFrame(data=d)
predictions_df.head(10)


# In[ ]:


# Export for testing
predictions_df.to_csv('output.csv', index=False)

