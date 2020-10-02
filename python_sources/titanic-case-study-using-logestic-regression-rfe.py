#!/usr/bin/env python
# coding: utf-8

# ## Titanic: Machine Learning from Disaster

# ### Step 1: Importing Data

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Pandas and NumPy
import pandas as pd
import numpy as np


# In[3]:


# Importingdatasets
train = pd.read_csv("../input/train.csv")
train.head()


# In[4]:


# Importingdatasets
test = pd.read_csv("../input/test.csv")
test.head()


# In[5]:


test1 = test['PassengerId']
test1.head()


# In[6]:


test1 = test1.to_frame(name=None)


# In[7]:


test1 = test1.set_index('PassengerId')


# In[8]:


train.columns


# In[9]:


# Let's check the dimensions of the dataframe
train.shape


# In[10]:


test.shape


# In[11]:


# let's look at the statistical aspects of the dataframe
train.describe()


# In[12]:


# Let's see the type of each column
train.info()


# ### Step 3: Data Preparation

# #### Converting some binary variables to 1/0

# In[13]:


# List of variables to map

varlist1 =  ['Sex']

# Defining the map function
def binary_map(x):
    return x.map({'male': 1, "female": 0})

# Applying the function to the train list
train[varlist1] = train[varlist1].apply(binary_map)


# In[14]:


train.head()


# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[15]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(train[['Embarked', 'Pclass']], drop_first=True)

# Adding the results to the dataframe
train = pd.concat([train, dummy1], axis=1)


# In[16]:


train.head()


# #### Dropping the repeated variables

# In[17]:


# We have created dummies for the below variables, so we can drop them
train = train.drop(['Embarked', 'Pclass','Name', 'Ticket', 'Fare', 'Cabin'], 1)


# In[18]:


train.info()


# #### Checking for Missing Values and Inputing Them

# In[19]:


# Adding up the missing values (column-wise)
train.isnull().sum()


# In[20]:


# Checking the percentage of missing values
round(100*(train.isnull().sum()/len(train.index)), 2)


# In[21]:


train["Age"] = train["Age"].fillna(value=train["Age"].mean()) #replace NaN by mean


# In[22]:


# Adding up the missing values (column-wise)
train.isnull().sum()


# Now we don't have any missing values

# ### Step 4: Test-Train Split

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


# Putting feature variable to X
X = train.drop(['Survived','PassengerId'], axis=1)

X.head()


# In[25]:


# Putting response variable to y
y = train['Survived']

y.head()


# In[26]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Step 5: Feature Scaling

# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


scaler = StandardScaler()

X_train[['Age']] = scaler.fit_transform(X_train[['Age']])

X_train.head()


# In[29]:


### Checking the survival Rate
survived = (sum(train['Survived'])/len(train['Survived'].index))*100
survived


# ### Step 6: Looking at Correlations

# In[30]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(train.corr(),annot = True)
plt.show()


# #### Checking the Correlation Matrix

# In[32]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# ### Step 7: Model Building
# Let's start by splitting our data into a training set and a test set.

# #### Running Your First Training Model

# In[33]:


import statsmodels.api as sm


# In[34]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ### Step 8: Feature Selection Using RFE

# In[35]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[36]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 5)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)


# In[37]:


rfe.support_


# In[38]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[39]:


col = X_train.columns[rfe.support_]


# In[40]:


X_train.columns[~rfe.support_]


# ##### Assessing the model with StatsModels

# In[41]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[42]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[43]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ##### Creating a dataframe with the actual survival flag and the predicted probabilities

# In[44]:


y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'survival_prob':y_train_pred})
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()


# ##### Creating new column 'predicted' with 1 if survival_prob > 0.5 else 0

# In[45]:


y_train_pred_final['predicted'] = y_train_pred_final.survival_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[46]:


from sklearn import metrics


# In[47]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)


# In[48]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[49]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))


# #### Checking VIFs

# In[50]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[51]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'PhoneService' has the highest VIF. So let's start by dropping that.

# In[52]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[53]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[54]:


y_train_pred[:10]


# ##### Let's check the VIFs again

# In[55]:


# Let's drop TotalCharges since it has a high VIF
col = col.drop('Parch')
col


# In[56]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[57]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[58]:


y_train_pred[:10]


# The accuracy is still practically the same.

# In[59]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
confusion


# In[60]:


# Actual/Predicted     not_survived    survived
        # not_churn        3269      366
        # churn            595       692  


# In[61]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted)


# ## Metrics beyond simply accuracy

# In[62]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[63]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[64]:


# Let us calculate specificity
TN / float(TN+FP)


# In[65]:


# Calculate false postive rate
print(FP/ float(TN+FP))


# In[66]:


# positive predictive value 
print (TP / float(TP+FP))


# In[67]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Step 9: Plotting the ROC Curve

# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[68]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[69]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.survival_prob, drop_intermediate = False )


# In[70]:


draw_roc(y_train_pred_final.Survived, y_train_pred_final.survival_prob)


# ### Step 10: Finding Optimal Cutoff Point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[71]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.survival_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[72]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[73]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[74]:


y_train_pred_final['final_predicted'] = y_train_pred_final.survival_prob.map( lambda x: 1 if x > 0.25 else 0)

y_train_pred_final.head()


# In[75]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[76]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )
confusion2


# In[77]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[78]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[79]:


# Let us calculate specificity
TN / float(TN+FP)


# In[80]:


# Calculate false postive rate
print(FP/ float(TN+FP))


# In[81]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[82]:


# Negative predictive value
print (TN / float(TN+ FN))


#  

# ## Precision and Recall

# In[83]:


#Looking at the confusion matrix again


# In[84]:


confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
confusion


# ##### Precision
# TP / TP + FP

# In[85]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[86]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# Using sklearn utilities for the same

# In[87]:


from sklearn.metrics import precision_score, recall_score


# In[88]:


precision_score


# In[89]:


precision_score(y_train_pred_final.Survived, y_train_pred_final.predicted)


# In[90]:


recall_score(y_train_pred_final.Survived, y_train_pred_final.predicted)


# ### Precision and recall tradeoff

# In[91]:


from sklearn.metrics import precision_recall_curve


# In[92]:


y_train_pred_final.Survived, y_train_pred_final.predicted


# In[93]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.survival_prob)


# In[94]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Step 11: Making predictions on the splitted test set

# In[95]:


X_test[['Age']] = scaler.transform(X_test[['Age']])


# In[96]:


X_test = X_test[col]
X_test.head()


# In[97]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[98]:


y_test_pred = res.predict(X_test_sm)


# In[99]:


y_test_pred[:10]


# In[100]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[101]:


# Let's see the head
y_pred_1.head()


# In[102]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[103]:


# Putting PassengerId to index
y_test_df['PassengerId'] = y_test_df.index


# In[104]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[105]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[106]:


y_pred_final.head()


# In[107]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'survival_prob'})


# In[108]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex_axis(['PassengerId','Survived','survival_prob'], axis=1)


# In[109]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[110]:


y_pred_final['final_predicted'] = y_pred_final.survival_prob.map(lambda x: 1 if x > 0.4 else 0)


# In[111]:


y_pred_final.head()


# In[112]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)


# In[113]:


confusion2 = metrics.confusion_matrix(y_pred_final.Survived, y_pred_final.final_predicted )
confusion2


# In[114]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[115]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[116]:


# Let us calculate specificity
TN / float(TN+FP)


# # checking on the given test set

# In[117]:


test.isnull().sum()


# In[118]:


test["Age"] = test["Age"].fillna(value=test["Age"].mean()) #replace NaN by mean


# In[119]:


test.isnull().sum()


# In[120]:


# List of variables to map

varlist =  ['Sex']

# Defining the map function
def binary_map(x):
    return x.map({'male': 1, "female": 0})

# Applying the function to the train list
test[varlist] = test[varlist].apply(binary_map)


# In[121]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy = pd.get_dummies(test[['Embarked', 'Pclass']], drop_first=True)

# Adding the results to the dataframe
test = pd.concat([test, dummy], axis=1)


# In[122]:


# We have created dummies for the below variables, so we can drop them
test = test.drop(['Embarked', 'Pclass','Name', 'Ticket', 'Fare', 'Cabin'], 1)


# In[123]:


test[['Age']] = scaler.transform(test[['Age']])


# In[124]:


test = test[col]
test.head()


# In[125]:


test_sm = sm.add_constant(test)


# In[126]:


test_pred = res.predict(test_sm)


# In[127]:


test_pred[:10]


# In[128]:


# Converting y_pred to a dataframe which is an array
pred_1 = pd.DataFrame(test_pred)
# Let's see the head
pred_1.head()


# In[129]:


# Converting y_test to dataframe
test_df1 = pd.DataFrame(test)


# In[130]:


# Putting PassengerId to index
test_df1['PassengerId'] = test1.index


# In[131]:


# Removing index for both dataframes to append them side by side 
pred_1.reset_index(drop=True, inplace=True)
test_df1.reset_index(drop=True, inplace=True)


# In[132]:


# Appending test_df and pred_1
pred_final1 = pd.concat([test_df1, pred_1],axis=1)
pred_final1.head()


# In[133]:


# Renaming the column 
pred_final1= pred_final1.rename(columns={ 0 : 'survival_prob'})


# In[134]:


# Rearranging the columns
pred_final1 = pred_final1.reindex_axis(['PassengerId','survival_prob'], axis=1)
# Let's see the head of pred_final1
pred_final1.head()


# In[135]:


pred_final1['Survived'] = pred_final1.survival_prob.map(lambda x: 1 if x > 0.4 else 0)
pred_final1.head()


# In[151]:


gender_submission = pred_final1.drop('survival_prob', axis=1)
gender_submission.head()


# In[156]:


submission = pred_final1[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)


# In[ ]:




