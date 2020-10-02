#!/usr/bin/env python
# coding: utf-8

# # Telco Customer Churn 
# 
# ## Part 1 
# ### Data Cleaning 
# - [Cleaning](#cleaning) 
# 
# ## Part 2
# ### Exploratory Analysis 
# - [Exploratory Analysis](#explo)
# 
# For EDA, I do some very simple exploratory analysis to try and pull out interesting factors that may be affecting customer churn. 
# 
# 
# 
# ## Part 3
# - [Machine Learning](#ml)
# 
# I then fit a decision tree classifier and a random forest classifier to classify the data with highest possible accuracy without overfitting (maintaing a good bias variance tradeoff !)
# 
# 
#    - Decision Tree Classifier 
#    - Random Forest Classifier
#    - Logistic Regression  
# 
# 
# ## Recommendation
# 
# ### Demographic 
# - One thing I noted is that individuals without partners/dependents are more likely to leave the company. In my opinion, individuals 'with' partners and dependents are more likely  to remain as most telecom companies offer special deals for families, couples etc.. As a recommendation, I would recommend the telecom company to maybe have a special promotion/marketing campaign targeted towards individuals without partners/dependents. 
# 
# 
# ### Service-Specific
# - Based off the feature importances and the EDA, I can conclude that the company needs to relaunch/remodel their FiberOptic Service. 
# 
# 
# - On the other hand, a lot of customers that are leaving the company seem to also not be registered for any support-like services (security, protection etc..) I would recommend the company to start offering these as bundles with services as customers who are signed up for these security-like services are more likely to remain with the company. (or make it mandatory to have these or just make them included) 
# 
# ### Payment-Specific
# - No recommendations 

# In[ ]:


import pandas as pd 
import os
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head(5)


# The dataset is very clean, there are no null values. Here, I replaced the binary outcome variable from Yes and No into 1 and 0. 

# <a id='cleaning' ></a>

# ### Cleaning 

# In[ ]:


print(df.isnull().sum())
df['Churn'].replace('Yes',1,inplace=True)
df['Churn'].replace('No',0,inplace=True)


# Farther down, I ran into an issue with the Total Charges Column as highlighted below. In the next few cells, I just adjust this column by removing these empty strings and reformatting the column to a numeric type

# In[ ]:


#There is an issue with the Total Charges colummns (the data is stored as a string)
print(' The data type for the Total Charges Column is:',type(df['TotalCharges'].loc[4]))
#While attempting to convert this to a numeric type, ran into another problem at some positions,empty strings
print(df['TotalCharges'][(df['TotalCharges'] == ' ')])


# In[ ]:


# Drop rows where there is no value for Total Charges 
index = [488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]
for i in index: 
    df.drop(i,axis=0,inplace=True)


# In[ ]:


# Convert from str to float
df['TotalCharges'].apply(float)


# <a id='explo' ></a>
# 

# ### Exploratory Analysis
# 
# - For my EDA, I split up the data into 3 categories. The first categroy is demographic which contains features like gender, partner etc.. 

#    #### Demographic
# Few things that stood out that seem to be quite different at the demographic level: 
# - Number of Senior Citizens 
# - Number of Individuals with Dependents 
# 
# In terms of churn, one interesting thing stands out, is that although the distribution of individuals with partners is equal, individuals without a partner are more likely to leave the company while individuals with a partner are more likely to remain as customers. This is a factor that is out of the company's control
# 
# 

# In[ ]:


# Inspecting frequency in the different demographic variables that are not related to the service
dem = ['gender','SeniorCitizen','Partner','Dependents']

for i in dem: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'],)
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())


#  #### Service-Specific 
# 
# - One thing that stood out may seem to uncover a service that may be causing high churn rates. Looking at the Internet Service, it seems that people with Fiber Optic service are more likely to leave the company. Although individuals with fiber optic services make up a large proportion of internet service customers, DSL customers have a much lower churn rate and make up almost the same number of individuals/customers.
#    - Proportion of individuals with DSL leaving the company 459/2416 * 100 = 19%
#    - Proportion of individuals with Fiber Optic leaving the company 1297/3096 * 100 = 42%
#  
#  
# - OnlineSecurity seems to be another factor causing high churn rates 
#   - Proportion of individuals with Online Security leaving the company = 295/3497 *100 = 8.4%
#   - Proportion of individuals without Online Security leaving the company = 1461/2015*100= 73%
#   
#   
# - OnlineBackup,Device Protection and Tech Support also follow the same pattern as online security where individuals without these services are more likely to leave the company. 
# 
# 
# 
# ##### Streaming Services 
#   - More than 40% of individuals with streaming TV and streaming movies service are also unsubscribing which may indicate that the streaming services could also be improved

# In[ ]:


cat = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for i in cat: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'],)
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())
    


# #### Payment-Specific 
#    - One thing that stood out was that the majority of customers pay for Telco services on a month to month contract basis. 
# 
# 

# In[ ]:


pay = ['Contract','PaperlessBilling','PaymentMethod']

for i in pay: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'])
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())


# <a id='ml' ></a>

# <a id='ml' ></a>

# ### Machine Learning
# - Start off converting variables to their appropriate format. Convert Binary Variables to 1 for Yes and 0 for no. For variables with more than 2 categories I used pd.get_dummies.

# In[ ]:


# Convert Binary Categories to 0's and 1's
df['Partner'].replace('Yes',1,inplace=True)
df['Partner'].replace('No',0,inplace=True)
df['Dependents'].replace('Yes',1,inplace=True)
df['Dependents'].replace('No',0,inplace=True)
df['gender'].replace('Male',1,inplace=True)
df['gender'].replace('Female',0,inplace=True)
df['PhoneService'].replace('Yes',1,inplace=True)
df['PhoneService'].replace('No',0,inplace=True)
df['PaperlessBilling'].replace('Yes',1,inplace=True)
df['PaperlessBilling'].replace('No',0,inplace=True)


# In[ ]:


## Prepare Categorical Variables with more than 2 categories
cat_X = df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',           'Contract','PaymentMethod']]
# Dummy Categorical Variables 
for i in cat_X: 
    cat_X = pd.concat([cat_X,pd.get_dummies(cat_X[str(i)],                                            drop_first=True,prefix=str(i))],axis=1)


cat_X = cat_X.drop(columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',           'Contract','PaymentMethod'])


# In[ ]:


features = pd.concat([df[['tenure','Partner','Dependents','gender','PhoneService',                          'PaperlessBilling','MonthlyCharges','TotalCharges']],cat_X],axis=1)


# The first tree is just a proof of concept to get an idea of which features really minimize confusion during the learning process for the tree (gini impurity). 
#    
#    - Tenure and Fiber Optic Services seem to be really important features to classify whether a customer will leave or remain. 

# ### Decision Tree Classifier

# In[ ]:


# Used stratified split as the classes are imbalanced
X=features
y= df['Churn']
X_train, X_test, y_train, y_test = train_test_split(features, df['Churn'],                                                     test_size=0.33, random_state=42,stratify=y)
my_DT = tree.DecisionTreeClassifier(max_depth=3)
my_DT.fit(X_train, y_train)
dot_data = StringIO()
export_graphviz(my_DT, out_file=dot_data,feature_names=features.columns,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Cross-Validation to find optimal value for the max_depth

# In[ ]:


X = features
y = df['Churn']
depth_range = np.arange(1,50,1)
val_scores = []
for d in depth_range:
    my_DT = tree.DecisionTreeClassifier(max_depth=d)
    scores = cross_val_score(my_DT, X, y, cv=10, scoring='accuracy')
    val_scores.append(scores.mean())
print(val_scores)


# In[ ]:


#Plot results from cross-validation
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(10,5))
ax1.plot(depth_range, val_scores)
ax1.set_xlabel('Max_Depth Values')
ax1.set_ylabel('Cross-Validated Accuracy Scores')

# A more zoomed in version of the first plot
ax2.plot(depth_range,val_scores)
ax2.set_xlim(1,15)
ax2.set_xlabel('Max_Depth Values')
ax2.set_ylabel('Cross-Validated Accuracy Scores')


# Decision Tree with Optimized Hyperparameters

# In[ ]:


my_DT = tree.DecisionTreeClassifier(max_depth=3)
my_DT.fit(X_train,y_train)
print(my_DT.score(X_train,y_train))
print(my_DT.score(X_test,y_test))


# In[ ]:


# What are the 10 most important features for classification ? 
imp = pd.DataFrame(my_DT.feature_importances_).sort_values(by=0,ascending=False).head(10).index.values
imp_vals = pd.DataFrame(my_DT.feature_importances_).sort_values(by=0,ascending=False).head(10)

for i,j in zip(imp,imp_vals[0]):
    print(features.columns[i],j)
    


# #### Confusion Matrix Decision Tree Classifier
#   - The x axis shows the predicted values and the y axis show the true class values. On the top right we have the "false positives" which are the X values that are actually 0s but were classified (predicted) as 1s. On the bottom left we have the "false negatives" which are values that are actually 1s (churn) but were classified (predicted) as 0s.   
#   
#   - The F1 score shown above the confusion matrix represents the harmonic mean of precision and recall. It receives equal contribution from precision and recall, hence the higher the score (closer to 1) the better the model is at classifying. 

# In[ ]:


y_pred = my_DT.predict(X_test)
def cm(pred):
    cm = confusion_matrix(y_test, pred)
    fig = plt.plot(figsize=(8,5))
    sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f1_score(y_test,pred))
    return plt.show()
cm(y_pred)


# #### ROC-AUC Decision Tree Classifier
#  - The red line (random predictor) is used as a baseline to see whether the model is useful.
#  - The blue line demonstrates the TPR and FPR at varying thresholds. 
#  - The greater the Area under the Curve (AUC), the better the model is at classifying. 

# In[ ]:


y_proba_DT = my_DT.predict_proba(X_test)


# In[ ]:


def roc_auc(prediction,model):
    fpr, tpr, thresholds = metrics.roc_curve(y_test,prediction)
    auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic '+str(model))
    plt.plot(fpr, tpr, color='blue', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'--',color='red')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt.show()


# In[ ]:


roc_auc(y_proba_DT[:, 1],'Decision Tree Classifier')


# ### Random Forest Regressor 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train,y_train)
print(RF.score(X_train,y_train))
print(RF.score(X_test,y_test))


# In[ ]:


warnings.filterwarnings('ignore')
# Number of trees in random forest
n_estimators = np.arange(10,1000,10)
# Number of features to consider at every split
# Maximum number of levels in tree
max_depth = np.arange(1,25,2)
# Minimum number of samples required to split a node
min_samples_split = [2,4,8]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = RF, param_distributions = grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)



# In[ ]:


print(rf_random.best_score_)
print(rf_random.best_params_)


# In[ ]:


RFR=RandomForestClassifier(bootstrap= True, max_depth= 11, min_samples_split= 2, n_estimators=30,min_samples_leaf= 4)
RFR.fit(X_train,y_train)
print(RFR.score(X_train,y_train))
print(RFR.score(X_test,y_test))
y_pred_rf = RFR.predict(X_test)


# The random forest scores slightly better and brings down the number of false negatives from 419 to 321 but .. results in twice the amount of false positives. 

# #### Confusion Matrix Random Forest Classifier

# In[ ]:


cm(y_pred_rf)


# #### ROC-AUC Random Forest Classifier

# In[ ]:


y_proba_rf = RFR.predict_proba(X_test)


# In[ ]:


roc_auc(y_proba_rf[:,1],'Random Forest Classifier')


# ### Logistic Regression

# In[ ]:


lr = LogisticRegression()
print(lr.fit(X_train,y_train))
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))


# In[ ]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0)
grid_model = clf.fit(X_train, y_train)


# In[ ]:


print('Best Penalty:', grid_model.best_estimator_.get_params()['penalty'])
print('Best C:', grid_model.best_estimator_.get_params()['C'])


# In[ ]:


print(grid_model.score(X_train,y_train))
print(grid_model.score(X_test,y_test))
y_pred_lr = grid_model.predict(X_test)


# #### ROC-AUC Curve Logistic Regression

# In[ ]:


cm(y_pred_lr)


# #### ROC-AUC Curve Logistic Regression

# In[ ]:


y_proba_lr = grid_model.predict_proba(X_test)


# In[ ]:


roc_auc(y_proba_lr[:,1],'Logistic Regression')

