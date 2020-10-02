#!/usr/bin/env python
# coding: utf-8

# # Welcome to my Notebook!

# ## Objective

# Given this dataset, the task is to classify whether a credit card transaction is fraud or not, based on features that were transformed using PCA and hidden for confidentiality purposes. Our objective is to have a fast,reliable and unbiased model that can accurately classify transactions as being fraud or not fraud.
# 
# This data gives many challenges, as there is a high imbalance between non-fraud and fraud transactions. I will use two technique in this notebook to try and get the best results:
# 
# 1. XGBoostClassifier
# 2. RandomForestClassifier using the `RandomUnderSampler` from the  `imblearn` module

# ## Import the usual libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the CSV File and adjust grid style with seaborn

# In[ ]:


sns.set_style('darkgrid')
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# ## Grab the Head of the Data

# In[ ]:


df.head()


# ## Some info on the data to check if there is any missing values

# In[ ]:


df.info()


# ## Describe the Data to get a sense of the data we are working with

# In[ ]:


df.describe()


# # Exploratory Data Analysis 

# ## Histogram of the Time feature with KDE graph

# In[ ]:


sns.distplot(df['Time'],bins=30)
plt.title('Histogram of Time feature')
plt.show()


# ## Histogram of Amount feature

# In[ ]:


sns.distplot(df['Amount'],bins=10)
plt.title('Histogram of Amount feature')
plt.show()


# ## Heatmap to check for any correlation

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.show()


# ### Here we can see that some features did have correlations.
# 
# We can see that some features were indeed correlated with the Class feature, but the correlations were insignificant because of two primary factors:
# 
# 1. The data was transformed using PCA
# 2. There is a high imbalance in the data, which might reduce correlations with the Class feature

# ## Countplot of class labels

# In[ ]:


sns.countplot(df['Class'])
plt.show()


# It is clearly evident from this plot that an imbalance is evident. I will try different methods to reduce this imbalance. Training this without evening out the balance will result in your model heavily overfitting, therefore we will have to use an imbalance technique to even out the balance

# # Preparing the Model

# For this model, I will split the data using Sklearn's `StratisfiedKfold`, as it effectively keeps the distribution of the data when performing cross validations splits

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ## Initialize StandardScaler

# I am going to use `StandardScaler` to scale out the `Time` and `Amount` features. It is important to have all your features at a relatively similiar scale and close to normal distribution.

# In[ ]:


scaler = StandardScaler()

df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))


# ## Define our features and target

# In[ ]:


X = df.drop('Class',axis=1)
y = df['Class']


# ## Split data with StratifiedKFold

# It is a good habit to set a `random_state` as this ensures repeatability in your data. If you do not do so, each time you split will result in different train and test values, which is not ideal 

# In[ ]:


kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)


# In[ ]:


for train,test in kf.split(X,y):
    X_train,X_test = X.loc[train],X.loc[test],
    y_train,y_test = y.loc[train],y.loc[test]


# # The Model

# The model I am going to be using is `XGBClassifier`. This is a very popular classification model used by many Kagglers to win many Data Science Competitions. It is essnetially a gradient booster that builds off the erros of previous models. Most of its parameters are best optimized using a Grid Search or Randomized Search. I myself ran 3 instances of `RandomizedSearchCV` to find the optimal parameters

# ## Import the Model

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(n_estimators=200,learning_rate=0.3,gamma=0.2)


# ## Fit the model

# In[ ]:


xgb.fit(X_train,y_train)


# ## Predict on test set

# In[ ]:


pred = xgb.predict(X_test)


# # Evaluating the models
# 
# ## I used the following to evalute my models:
# ### 1. Classification Report
# ### 2. Confusion Matrix
# ### 3. ROC & AUC

# I have created a function for evaluation that includes all of the evaluation methods listed above. I did not use `accuracy_score` here as accuracy tends to be very misleading and is not a good evaluation when deealing with imbalanced datasets

# ## Import Evaluation metrics

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve
from imblearn.metrics import classification_report_imbalanced


# ## Evaluation function

# In[ ]:


def evaluate(prediction,test,imb=False):
    sns.heatmap(confusion_matrix(prediction,test),annot=True)
    plt.show()
    if imb == True:
        print(classification_report(prediction,test))
    else:
        print(classification_report_imbalanced(test,prediction))
    score = roc_auc_score(test,prediction)
    print('AUC Score: ' + str(score))
    fp,tp,_ = roc_curve(test,prediction)

    plt.figure(figsize=(8,8))
    plt.plot(fp,tp,linestyle='dashed',c='orange',label='XGBoostClassifier')
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Visualisation')
    plt.legend()
    plt.show()


# ## Call the function

# In[ ]:


evaluate(pred,y_test)


# Well, looking at the code here we can see that our `XGBClassifier` did a pretty good job, reaching 93% recall. We will now use Undersampling and the `RandomForestClassifier` to view its performance

# # Undersampling with RF

# Now, since we have imbalanced data, we have a few options that we can use at hand:
# 
# 1. Undersampling
# 2. Oversampling
# 3. SMOTE
# 
# I will be using Imblearn's `RandomUnderSampler` to undersample the majority class(Non Fraud) and make it match with the minority class

# I decided to use Random Forests as I believe that they perform slightly better than `XGBoosterClassifier` for this problem.

# ## Import the library

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler


# ## Instantiate RandomUnderSampler and fit it to the data

# In[ ]:


rus = RandomUnderSampler(sampling_strategy=1)

X_rus,y_rus = rus.fit_sample(X,y)


# ## Concactinate the two DataFrames together and visualize using CountPlot

# In[ ]:


rus_df = pd.concat([X_rus,y_rus],axis=1)
sns.countplot(rus_df['Class'])
plt.show()


# Now we have removed the imbalance by randomly selecting samples to match the minority class. Thus, we can begin modelling!

# ## Define Features and Target

# In[ ]:


undersample_X = rus_df.drop('Class',axis=1)
undersample_y = rus_df['Class']

for train,test in kf.split(undersample_X,undersample_y):
    X_train_rus,X_test_rus = undersample_X.loc[train],undersample_X.loc[test],
    y_train_rus,y_test_rus = undersample_y.loc[train],undersample_y.loc[test]


# I successfully used imblearn's `RandomUnderSampler` to undersample the majority class.
# Note the `sampling_strategy` was set to 1. of I was to set the strategy to 0.9, the minority class was still slightly smaller than the majority class, but the imbalance is practically gone. Now that the imbalance is gone,we can start training the data.

# ## Import RandomForestClassifier and initialise

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200,max_features=2)


# In[ ]:


rf.fit(undersample_X,undersample_y)


# In[ ]:


pred_rf = rf.predict(X_test)


# ## Evaluate Model

# In[ ]:


evaluate(pred_rf,y_test,imb=True)


# In[ ]:




