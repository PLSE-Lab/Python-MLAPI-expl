#!/usr/bin/env python
# coding: utf-8

# # The H2O package from H2O.AI and AutoML function
# H2O.AI is the company which provide this package h2o, it runs on both R and python.The package has an function Automl which run GBM, Distributed Random Forest , DNN and combination of these stacked models. It compares all the models based on lowest error rate/accuracy this can be used to generate decent models with above average performance in short time.
# 
# There is detailed explanation of the Automl function from fellow Kaggler : Parul 
# 
# https://towardsdatascience.com/a-deep-dive-into-h2os-automl-4b1fe51d3f3e

# # Data loading and EDA

# In[ ]:


import pandas as pd
titanic_train=pd.read_csv('../input/titanic/train.csv')
titanic_test=pd.read_csv('../input/titanic/test.csv')
titanic_train.describe()


# Here are few basic inferences we can get from numeric data:
# 1. the median age is around 28 which suggest that majority passengers were young.
# 2. The median fare is 14 pound while 75 percentile is 31 suggesting only very few passengers have paid for super luxury like above 500 pounds.
# 3. Majority of passengers are from class 3

# # Checking for missing values

# In[ ]:


titanic_train.isnull().sum()


# In[ ]:


titanic_test.isnull().sum()


# Imputation with median for numeric variables and mode for character variables

# In[ ]:


titanic_train['Age'].fillna(titanic_train['Age'].median(), inplace = True)
titanic_test['Age'].fillna(titanic_test['Age'].median(), inplace = True)
titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode(),inplace=True)
titanic_test['Fare'].fillna(titanic_test['Fare'].median(), inplace = True)


# # Feature Engineering 

# If you are single and male the chances of survival would be low, as historical account of the incident women and children were the first one to be rescued, hence we will create a column solo traveller. We create a family size using siblings/spouse and parents/child column , if family size=1 then passenger is the solo traveller

# In[ ]:


titanic_train['familysize']=titanic_train['SibSp']+titanic_train['Parch']+1
titanic_train['Solo']=(titanic_train['familysize'] >1 ).astype(int)
titanic_train['Solo'],titanic_train['familysize']


# In[ ]:


titanic_test['familysize']=titanic_test['SibSp']+titanic_test['Parch']+1
titanic_test['Solo']=(titanic_test['familysize'] >1 ).astype(int)
titanic_train['Solo'].value_counts()


# # Age - creating bins as we know young were last to be rescued, so lets explore this relation

# In[ ]:


Age_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Age']).sum())
Age_wise_dist=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Age']).count())
Age_wise_dist


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(Age_wise_survival,label="Survived")
plt.plot(Age_wise_dist,label="Total")
plt.legend()


# As we suspected the survived blue line has a gap from total passengers if you are between 15-35 , so we have to create bins for child ,adult and olds.Let us create 5 bins in which middle age is slightly more divided , since few above age 40 had families together and they might have been rescued too.

# In[ ]:


titanic_train['AgeBin'] = pd.cut(titanic_train['Age'].astype(int), 5)
titanic_train['AgeBin'].values


# We have 5 categories cut off from 16, then at 32, 48, 64.

# In[ ]:


titanic_test['AgeBin'] = pd.cut(titanic_test['Age'].astype(int), 5)


# # Let us explore gender wise survival too though we will use column as it is

# In[ ]:


Gender_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Sex']).sum())
Gender_wise_count=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Sex']).count())
Gender_wise_survival/Gender_wise_count


# Only 19% male passengers survived as compared to 74% female

# # Using the title = Master to create a column VIP
# 
# I am assuming that master title has some signifance as it is used for children and that increases there chance of survival, let me treat them as VIP's :)

# In[ ]:


titanic_train['vip'] = [1 if x =='Master' else 0 for x in titanic_train['Name']] 
titanic_test['vip'] = [1 if x =='Master' else 0 for x in titanic_test['Name']] 


# Exploring Fare and creating categories

# In[ ]:


Fare_wise_dist=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Fare']).count())
Fare_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Fare']).sum())
Fare_wise_s_per=(Fare_wise_survival/Fare_wise_dist)*100
import matplotlib.pyplot as plt
plt.plot(Fare_wise_s_per,'bo',label="% of survived by fare")
plt.legend()


# Well the person who paid 500 pound did survive while the relationship is not linear but we do have slightly high chance of survival if we paid more than 100 pounds. I am making three catgeories with equal width

# In[ ]:


titanic_train['farebin'] = pd.cut(titanic_train['Fare'].astype(int), 3)
titanic_train['farebin'].values


# In[ ]:


titanic_test['farebin'] = pd.cut(titanic_test['Fare'].astype(int), 3)


# # Lets prepare the data for H2O automl and select only important columns
# 
# H2o do not take pandas dataframe directly , it has its own dataframe so we will convert pandas datframe to H2o dataframe, one key diffrence is that in train data you must have the dependent column and this need not be passed like seperate like in GBM and NN

# In[ ]:


y="Survived"
train_x=titanic_train[['Pclass','Sex','familysize','Solo','AgeBin','vip','farebin','Survived']]
test_x=titanic_test[['Pclass','Sex','familysize','Solo','AgeBin','vip','farebin']]


# Encoding the variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train_x=train_x.apply(lambda col: label.fit_transform(col), axis=0, result_type='expand')
test_x=test_x.apply(lambda col: label.fit_transform(col), axis=0, result_type='expand')
train_x


# Loading the H2O package

# In[ ]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()


# In[ ]:


trframe=h2o.H2OFrame(train_x)


# In[ ]:


teframe=h2o.H2OFrame(test_x)


# H2OAutoml has two primary requirements one y variable and training dataframe , and two options, one max_runtime in seconds , this closes the algorithm within that time limit and max models which limits the number of models excluding stacked models, that automl generates if we dont specify this option it will generate as long as it encounters max runtime , i think both are some type of hyperparameters to reduce overfitting , if we give large run time probably this may lead to vanishing gradient problem , I am trying with 120 seconds.
# 
# More documentation is here:
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# In[ ]:


titanic_model = H2OAutoML(max_runtime_secs = 120, seed = 1, project_name = "titanic_kaggle_nishant")
titanic_model.train(y = y, training_frame = trframe)


# # Using the Leader board option to arrive at best model.

# In[ ]:


titanic_model.leaderboard


# The XGBoost_grid__1_AutoML_20200513_144044_model_2 gives us the lowest root mean square error and this the best fitted model at this run time. If you go to the local server page http://127.0.0.1:54321/flow/index.html you can find the option in getmodel in assistance.Search this model and you get plethora of details about model. Let me higlight few of them here

# # PARAMETERS DETAILS

# In[ ]:


titanic_model.leader.params.keys()


# In[ ]:


titanic_model.leader.params['colsample_bytree'],titanic_model.leader.params['stopping_rounds']


# The training deviance graph by number of trees This feature unfortunately is only availble through H2o flow server page  
# 
# Kindly check the graph there

# # Variable importance, cross validation details and model stats

# In[ ]:


lb=titanic_model.leader


# In[ ]:


m = h2o.get_model(lb)


# # predicting using h2o.predict

# In[ ]:


pred_h2o = titanic_model.leader.predict(teframe)
pred_h2o


# In[ ]:


pred_pandas=pred_h2o.as_data_frame(use_pandas=True)
pred_pandas


# let us use 50% above as cutoff for saying survived 

# In[ ]:


pred_pandas['Survived'] = [1 if x > 0.5 else 0 for x in pred_pandas['predict']] 
pred_pandas


# In[ ]:


output= titanic_test.merge(pred_pandas['Survived'], left_index=True, right_index=True)
output


# In[ ]:


output_final=output[['PassengerId','Survived']]
output.to_csv('GBM_NISHANT.csv',index="FALSE")

