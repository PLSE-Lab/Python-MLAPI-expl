#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction
# ### In this notebook I will analyze the data set from Loan prediction III problem from Analytics Vidhya and build models to predict the loan status
# 
# 
# #### About Dataset
# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.
# 
# 
# #### Problem
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set
# 
# 
# 
# ## Load the data and look at summary statistics

# In[43]:



import pandas as pd
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[44]:


loan = pd.read_csv("../input/loan_predict_train.csv")


# In[45]:


loan.shape


# In[46]:


loan.info()


# In[47]:


loan.describe()


# In[48]:


loan.isnull().sum()


# #### Target Variable is Loan_Status - let's see its distribution

# In[49]:


loan.Loan_Status.value_counts()


# #### The data is moderately unbalanced - minority class is about 31%.

# ## Exploratory Analysis
# #### Is there any trend in loan status based on LoanAmount or Income of applicant/coapplicant ?

# In[50]:


fig, ax = plt.subplots(figsize=(8, 8))
plt.style.use('seaborn-whitegrid')
sns.set(palette="muted")
ax.set_ylim(0,500)
ax = sns.boxplot(x="Loan_Status", y="LoanAmount", data=loan)


# In[51]:


fig, ax = plt.subplots(figsize=(8, 10))
ax.set_ylim(0,20000)
ax = sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=loan)


# In[52]:


fig, ax = plt.subplots(figsize=(8, 10))
ax.set_ylim(0,20000)
ax = sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=loan)


# - ###### We can see from above plot that mean coapplicant income is higher for approved loans

# ### How is approval rate among various categories ?

# In[53]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,20))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.3)
rows = 3
cols = 3
categorical_col = ['Gender', 'Married', 'Education', 'Property_Area', 'Dependents', 'Self_Employed']
for i, column in enumerate(categorical_col):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set(xticks=[])
    ax = pd.crosstab(loan[categorical_col[i]], loan.Loan_Status, normalize='index').plot.bar(ax=ax)


# - ###### We can see that SemiUrban properties have relatively higher rate of approval

# ### Is there any gender bias?

# In[54]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
pd.crosstab([loan.Married, loan.Gender, loan.Dependents], loan.Loan_Status, normalize='index').plot.barh(ax=ax);


# - It can be noted from above plot that Unmarried Females with 3+ dependents have very high rate of rejection - let's dive into the details - from the data it doesn't show any gender bias as there are only three samples out of which only one has good credit history

# In[55]:


loan.loc[(loan.Gender == 'Female') & (loan.Married == 'No') & (loan.Dependents == '3+')]


# ## Data Preprocessing

# ### Missing data
# - Out of the 13 columns we will exclude Loan_Status and Loan_ID and arrive at 11 predictor variblles. Out of these seven variables have varying degrees of missing values. We need to impute these appropriately.

# In[56]:


def do_preprocess(data):
    table = data.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
    def fage(x):
        return table.loc[x['Self_Employed'],x['Education']]
    # Replace missing values
    data['Self_Employed'].fillna('No', inplace=True)
    data['LoanAmount'].fillna(data[data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)


do_preprocess(loan)


# ### Feature Engineering
# - Performance of a model will be greatly influenced by carefully crafted features.
# - It is reasonable to assume a loan application is weighed by the ability to repay it. This ability will depend on the applicant's income. Hence we can add a feature of the ratio of there two terms.
# - We can come up with the below features
#     - ApplicantIncomeByLoanAmount
#     - IncomeByLoanAmount - here Income is the total income of applicant and co-applicant

# In[57]:


def add_new_features(data):
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['IncomeByLoanAmount'] = data['TotalIncome'] / data['LoanAmount']
    data['AplIncomeByLoanAmount'] = data['ApplicantIncome'] / data['LoanAmount']
   
add_new_features(loan)


# ### Encoding Predictors
# - As the machine learning algorithms internally operate on matrices of numerical data we need to encode our categorical variables into numberic form.
# - The most widely used encoding is the OneHotEncoding and pandas readily provides this functionality

# In[58]:


def doOneHotEncoding(data, cols):
    for var in cols:
        one_hot = pd.get_dummies(data[var], prefix = var)
        # Drop column B as it is now encoded
        data = data.drop(var,axis = 1)
        # Join the encoded data
        data = data.join(one_hot)
    return data

loan = doOneHotEncoding(loan, ['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'])
loan.Loan_Status = loan.Loan_Status.map(dict(Y=1,N=0))


# In[59]:


loan.Loan_Status.value_counts()


# ## Building Model
# #### We will begin with building a linear model because of its interpretability. Will use statsmodels api to fit the data as it gives detailed statistics on the fitted model. Let's do Logistic Regression on this data.

# In[60]:


outcome_var = "Loan_Status"
#exclude baseline categorical variables
predictor_var = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', #'TotalIncome',
       'IncomeByLoanAmount', 'AplIncomeByLoanAmount', #'EMI',
       'Gender_Male', 'Married_No', 'Dependents_0', 'Dependents_1', 'Dependents_3+',
       'Education_Not Graduate', 'Self_Employed_Yes',
       'Property_Area_Semiurban', 'Property_Area_Urban']

logit = sm.Logit(loan[outcome_var], loan[predictor_var])
result = logit.fit_regularized()


# In[61]:


result.summary2()


# In[62]:


pt = result.pred_table()
pt


# In[63]:


print ("Accuracy : %s" % "{0:.3%}".format((pt[0,0]+pt[1,1])/pt.sum()))


# - Based on p-values we got 6 out of 16 variables as significant
# - Let's fit the model with only those variables

# In[64]:


predictor_var = ['Loan_Amount_Term', 'Credit_History', 'Property_Area_Semiurban',
                 'Married_No', 'Dependents_1', 'Education_Not Graduate']


# In[65]:


logit = sm.Logit(loan[outcome_var], loan[predictor_var])
result = logit.fit_regularized()


# In[66]:


result.summary2()


# In[67]:


pt = result.pred_table()
pt


# In[68]:


print ("Accuracy : %s" % "{0:.3%}".format((pt[0,0]+pt[1,1])/pt.sum()))


# - Dropping insignificant variables slightly improved the accuracy.

# ### Interpretation
# #### - Having good credit history changes the log odds of loan approval by 3.4846
# #### - For every unit change in Loan Amount Term the log odds of loan approval decreases by 0.0057 units
# #### - Buying a property located in SemiUrban area as compared to a Rural area changes the log odds of loan approval by 0.7526 units

# ### Model Evaluation
# #### A single fit is not sufficient and we need to Evaluate the model to ensure that it generalizes well. We need scikit learn api for cross validation.
# 
# #### In classification, Accuracy does not tell the whole story particularly in unbalanced datasets - we need to look at the precision and recall of the classifier on each class to understand the how well the classfier is doing its job.

# Let's fit Logistic Regression classfier from scikit-learn

# In[69]:


from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression(random_state=42)


# In[70]:


def classify_and_report_metrics(model, data, predictors, outcome):
    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[outcome], random_state=42, stratify=loan[outcome_var])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy on test set: %s' % '{0:.3%}'.format(model.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))


# In[71]:


##adding all non-base levels of categorical variables that were found significant
predictor_var += ['Dependents_0', 'Dependents_3+', 'Property_Area_Urban']
predictor_var_logistic = predictor_var


# ##### Split the data into train and test - fit the model on train dataset and test on the test set

# In[72]:


classify_and_report_metrics(model_logistic, loan, predictor_var, outcome_var)


# ##### Now perform a Five fold cross validation

# In[73]:


def fit_and_validate(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

    cm = confusion_matrix(data[outcome], predictions)
    print(cm)
    print(classification_report(data[outcome], predictions))    
    
    #Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    if (isinstance(model, (RandomForestClassifier))):
            #Create a series with feature importances:
            featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
            print (featimp)
            if (isinstance(model, RandomForestClassifier)):
                if model.get_params()['oob_score'] == True:
                    print('OOB Score %f' % (1 - model.oob_score_))
                else:
                    print('OOB Score False')


# In[74]:


fit_and_validate(model_logistic, loan, predictor_var, outcome_var)


# ### Let's see if we can improve the accuracy. We will use Random Forest classifier as it,
# 
# #### - Works well for classification
# #### - Robust to outliers in the data
# #### - Relatively easy to tune
# #### - Provides out of bag error rate indicative of the unbiased nature of the fit
# 
# #### For fitting a RandomForest model, we will include,
# - Variables identified as significant by Linear Model
# - Two additonal Features created earlier
# - ApplicantIncome and LoanAmount which has many outliers

# In[75]:


predictor_var = [
 'Credit_History', 'IncomeByLoanAmount', 'AplIncomeByLoanAmount',
    'LoanAmount', 'Loan_Amount_Term',
    'Property_Area_Semiurban', 'ApplicantIncome','Married_No','Dependents_1',
    'Education_Not Graduate'
]


# - Fit the model using the tuned parameters

# In[76]:


model_rf = RandomForestClassifier(random_state=42, n_estimators=200, bootstrap= True, oob_score=True)


# - Split the data, train the model and verify on the test set

# In[77]:


classify_and_report_metrics(model_rf, loan, predictor_var, outcome_var)


# - Perform 5-fold cross validation

# In[78]:


fit_and_validate(model_rf, loan, predictor_var, outcome_var)


# #### Cross validation score is slightly lower than Logistic Regression. Let's use the top 6 important features and fit again

# In[79]:


predictor_var = [
 'Credit_History', 'IncomeByLoanAmount', 'AplIncomeByLoanAmount',
    'LoanAmount','Property_Area_Semiurban', 'ApplicantIncome'
]
predictor_var_rf = predictor_var


# #### Let's find the best hyperparameters for the RandomForestClassifier

# In[80]:


import pprint as pp
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 400, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pp.pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(loan[predictor_var], loan[outcome_var])
rf_random.best_params_


# #### Create the model with tuned parameters

# In[81]:


model_rf = RandomForestClassifier(random_state=42, n_estimators = rf_random.best_params_['n_estimators'], 
                                  min_samples_split = rf_random.best_params_['min_samples_split'], 
                                  min_samples_leaf = rf_random.best_params_['min_samples_leaf'], 
                                  max_features = rf_random.best_params_['max_features'], 
                                  max_depth = rf_random.best_params_['max_depth'],
                                  bootstrap = rf_random.best_params_['bootstrap'],
                                  oob_score = rf_random.best_params_['bootstrap'])


# In[82]:


fit_and_validate(model_rf, loan, predictor_var, outcome_var)


# In[83]:


print(rf_random.best_params_['n_estimators']), print(rf_random.best_params_['max_features']), 
print(rf_random.best_params_['max_depth']), print(rf_random.best_params_['min_samples_split']),
print(rf_random.best_params_['min_samples_leaf']), print(rf_random.best_params_['bootstrap'])


# In[ ]:





# In[84]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(y_test, logistic_model.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(X_test)[:,1])
X_train, X_test, y_train, y_test = train_test_split(loan[predictor_var_logistic], loan[outcome_var], random_state=42, stratify=loan[outcome_var])
model_logistic.fit(X_train, y_train)
logit_roc_auc = roc_auc_score(y_test, model_logistic.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_logistic.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')

X_train, X_test, y_train, y_test = train_test_split(loan[predictor_var_rf], loan[outcome_var], random_state=42, stratify=loan[outcome_var])
model_rf.fit(X_train, y_train)
logit_roc_auc = roc_auc_score(y_test, model_rf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_rf.predict_proba(X_test)[:,1])
# plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ### Conclusion

# #### We can see that Random Forest gives the best accuracy on this dataset.

# ##### That's it ! We have reached the end of this kernel. Thank's for your time !
# If you find this kernel useful please UP VOTE ! If anything can be improved please share your valuable feedback !
