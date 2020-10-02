#!/usr/bin/env python
# coding: utf-8

# # **Predicting loan defaults**
# Purpose of this work is to build machine learning models for loan default prediction. I will use LendingClub dataset to train the models.
# 
# 
# **LendingClub**
# 
# LendingClub is an American lending company, headquartered in San Francisco. Company offers loans between 1,000 and 40,000 for standard loan period of 3 years.
# 
# **Dataset**
# 
# The dataset contains complete loan data for all loans issued through the 2007-2019. Each row represents a loan, dataset has 145 columns with all possible variables including the current loan status (Current, Late, Fully Paid, etc.), wich will be the target value for prediction. Total number of loans is 2,2M.
# 

# At first, lets see what we are working with:

# In[ ]:


import pandas as pd
PATH = '../input/lending-club-loan-data/loan.csv'
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


raw_data = pd.read_csv(PATH)
pd.options.display.max_columns = 2000
raw_data.head()


# # **Data Preparation**
# As mentioned, dataset has 145 columns. We won't need all of them for the model, lets select the necessary ones:

# In[ ]:


data = raw_data[['loan_amnt', 'term', 'int_rate', 'installment', 
                 'grade', 'sub_grade', 'emp_length', 
                 'home_ownership', 'annual_inc', 'verification_status', 
                 'purpose', 'dti', 'delinq_2yrs', 'delinq_amnt', 
                 'chargeoff_within_12_mths',  'tax_liens',  
                 'acc_now_delinq', 'inq_last_12m', 'open_il_24m', 
                 'loan_status']]
data.head()


# **Null values**
# 
# Lets see the number of null values in each column we selected:

# In[ ]:


data.isnull().sum()


# Some columns have few rows with NaNs, we will drop these rows:

# In[ ]:


data = data.dropna(axis = 'index', 
                   subset = ['annual_inc', 'dti', 'delinq_2yrs', 
                             'delinq_amnt', 'chargeoff_within_12_mths', 
                             'tax_liens', 'acc_now_delinq'])
data.isnull().sum()


# On the other hand, columns **emp_length**, **inq_last_12m** and **open_il_24m** have too many NaNs to drop them out, we will now change their type to string.
# Also, **inq_last_12m** and **open_il_24m** have same amount of nulls, lets remember to sort this out later.

# In[ ]:


data = data.fillna(value = {'emp_length' : 'no_info', 
                            'inq_last_12m' : 'no_info', 
                            'open_il_24m':'no_info'}) 


# **Correlation Matrix**
# 
# Now lets build a correlation matrix to show dependencies between columns:

# In[ ]:


import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

data_for_corr = data.assign(term = data.term.astype('category').cat.codes,
                            grade = data.grade.astype('category').cat.codes,
                            sub_grade = data.sub_grade.astype('category').cat.codes,
                            emp_length = data.emp_length.astype('category').cat.codes,
                            home_ownership = data.home_ownership.astype('category').cat.codes,
                            verification_status = data.verification_status.astype('category').cat.codes,
                            purpose = data.purpose.astype('category').cat.codes,
                            loan_status = data.loan_status.astype('category').cat.codes,
                            inq_last_12m = data.inq_last_12m.astype('category').cat.codes,
                            open_il_24m = data.open_il_24m.astype('category').cat.codes
                            )

corr_matrix = data_for_corr.corr()
plt.figure(figsize=(18,14))
sn.heatmap(corr_matrix, annot=True, cmap = 'Blues',vmin=-0.1, vmax=1)
plt.title('Correlation matrix')
plt.show()


# As we can see, some columns have a strong correlation:
# 1. **loan_amnt** with **installment**
# 2. **int_rate** with **grade** and **sub_grade**
# 3. And as expected **inq_last_12m** with **open_il_24m** 
# 
# We will drop this columns to prevent multicollinearity:

# In[ ]:


data = data.drop(columns = ['installment', 'grade', 'sub_grade', 'open_il_24m'])


# Wow, we also see *some* correlation between **inq_last_12m** and **loan_status**! Lets inspect this, maybe this column can be very useful for prediction.
# 
# Lets compare percent of **loan_status** in subset, where **inq_last_12m** is null and in whole dataset:

# In[ ]:


data_inq_nan = data[data.inq_last_12m == 'no_info']
pd.DataFrame({'where inq_last_12m=NaN': data_inq_nan['loan_status'].value_counts()/len(data_inq_nan), 
              'all dataset'           : data['loan_status'].value_counts()/len(data)}
             ).style.format('{:.2f}')


# We see, that loans with *i**nq_last_12m**=NaN* have few Current statuses, this may be explained by the fact that these are old loans, they are either Charged Off or Fully Paid.
# 
# Now lets check if their default rate is any different:

# In[ ]:


data_3_statuses = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Charged Off') | 
                 (data.loan_status == 'Default')]
data_inq_nan = data_3_statuses[data_3_statuses.inq_last_12m == 'no_info']
pd.DataFrame({'where inq_last_12m=NaN': 
              data_inq_nan['loan_status'].value_counts()/len(data_inq_nan), 
              'all dataset': 
              data_3_statuses['loan_status'].value_counts()/len(data_3_statuses)}
             ).style.format('{:.2f}')


# Actually it is similar. The correlation is explained by low number of Current statuses in these loans. Giving the fact, that this column has 866k nulls with not much useful information in it, we will drop it.

# In[ ]:


data = data.drop(columns = 'inq_last_12m')


# **Loan status**
# 
# Now lets inspect **loan_status** column.

# In[ ]:


data['loan_status'].value_counts()


# We can only use Fully Paid, Charged Off or Default loans for prediction. Others need to be omitted. 
# 
# Lets also encode **loan_status** with binary number, where 1 indicates Default loan.

# In[ ]:


data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Charged Off') | 
                 (data.loan_status == 'Default')]
data['loan_status'] = data['loan_status'].replace(to_replace = ['Fully Paid', 'Charged Off', 'Default'], 
                                                       value = [0, 1, 1])
data['loan_status'].value_counts()


# We see, that dataset is inbalanced, this could be a problem with modeling. Lets come back to this later.

# A short summary:

# In[ ]:


data.describe().style.format('{:.2f}')


# For machine learning model we will need to split up the dataset into **X** and **y**, where **y** is a target feature array, and **X** is dataset without a target feature.

# In[ ]:


X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values


# Categorical columns **term** and **emp_length** have *comparable* values, they need to be encoded with LabelEncoder:

# In[ ]:


from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])


# Columns **home_ownership**, **verification_status** and **purpose** on the other hand have *incomparable* values, encode them with ColumnTransformer:

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 6, 7])],
                       remainder='passthrough')
X = ct.fit_transform(X)


# Split for train and test partitions:

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# And perform scaling (y_train and y_test dont need scaling, because they are binary anyway):

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# For easy of use, lets put this all together in a function:

# In[ ]:


def prepare_data(data):
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values
    
    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 6, 7])],
                       remainder='passthrough')
    X = ct.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# In[ ]:


#X_train, X_test, y_train, y_test = prepare_data(data)


# # **Modeling**
# 
# We will use Naive Bayes, Random Forest and Logistic Regression for our prediction.
# 
# Lets start with Naive Bayes model.

# In[ ]:


from sklearn.naive_bayes import GaussianNB

classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)
y_pred = classifier_naive_bayes.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

def build_conf_matrix(title):
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='pred'))
    sn.heatmap(conf_matrix, annot=True, cmap='Blues',vmin=0, vmax=1)
    plt.title(title)
    plt.show()
    
build_conf_matrix(title='Naive Bayes, imbalanced dataset')


# As we see, our model puts everything into 0. Lets balance data and try again:

# In[ ]:


data_loan_status_1 = data[data['loan_status'] == 1]
data_loan_status_0 = data[data['loan_status'] == 0].sample(n=len(data_loan_status_1))
data_balanced = data_loan_status_1.append(data_loan_status_0) 


# In[ ]:


X_train, X_test, y_train, y_test = prepare_data(data_balanced)


# In[ ]:


classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)
y_pred = classifier_naive_bayes.predict(X_test)


# In[ ]:


build_conf_matrix(title='Naive Bayes, balanced dataset')


# Much better, although accuracy is not that great.
# 
# Random Forest model:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier_rand_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_rand_forest.fit(X_train, y_train)
y_pred = classifier_rand_forest.predict(X_test)


# In[ ]:


build_conf_matrix(title='Random Forest, 10 trees')


# Lets use GridSearch to find any potential improvement for Random Forest model:

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = [ {'n_estimators':[10, 50, 100, 200], 'criterion':['entropy', 'gini']}]
grid_search = GridSearchCV(estimator = classifier_rand_forest, 
                                 param_grid = parameters,
                                 scoring = 'accuracy',
                                 cv = 2,
                                 n_jobs = -1,
                                 verbose = 5)
grid_search = grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# I should mention that I ran grid search 3 times in different versions of this notebook so far, I got gini 2 times and entropy once, so these criterions are close in terms of accuracy.
# But this time we got model with gini criterion again.

# In[ ]:


classifier_rand_forest = RandomForestClassifier(n_estimators = 200, criterion = 'gini')
classifier_rand_forest.fit(X_train, y_train)
y_pred = classifier_rand_forest.predict(X_test)


# In[ ]:


build_conf_matrix(title='Random Forest, 200 trees, gini')


# But not a big improvement anyway compared to 10 trees model

# Lets try Logistic Regression:

# In[ ]:


from sklearn.linear_model import LogisticRegression

classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)
y_pred = classifier_log_reg.predict(X_test)


# In[ ]:


build_conf_matrix(title='Logistic Regression')


# Slightly better result than other models.
# 
# To summurize models accuracy we will use cross validation score: 

# In[ ]:


from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier_naive_bayes, X = X_train, y = y_train, cv = 5)
('Naive Bayes',{'accuracy':accuracies.mean(), 'std':accuracies.std()})


# In[ ]:


accuracies = cross_val_score(estimator = classifier_log_reg, X = X_train, y = y_train, cv = 5)
('Logistic Regression',{'accuracy':accuracies.mean(), 'std':accuracies.std()})


# In[ ]:


accuracies = cross_val_score(estimator = classifier_rand_forest, X = X_train, y = y_train, cv = 2)
('Random Forest',{'accuracy':accuracies.mean(), 'std':accuracies.std()})


# We got maximum accuracy of 64,5% with Logistic Regression, this is not great by any means. 
# 
# Lets try to make a small 2D dataset so we can see data and model job on the plot. Maybe we could find a way to improve our models. 
# 
# Dataset with 10k rows will be enough to see the general picture and not to overwhelm plot with points.
# 

# In[ ]:


data_loan_status_1 = data[data['loan_status'] == 1].sample(n=10000)
data_loan_status_0 = data[data['loan_status'] == 0].sample(n=len(data_loan_status_1))
data_small = data_loan_status_1.append(data_loan_status_0)


# Lets use **loan_amnt** and **int_rate** columns as X:

# In[ ]:


X = data_small.iloc[:, [0, 2]].values
y = data_small.iloc[:, -1].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# As classifier lets use a Logistic Regression model:

# In[ ]:


classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)
y_pred = classifier_log_reg.predict(X_test)


# In[ ]:


build_conf_matrix(title='Logistic Regression, small dataset with 2 columns')


# Interesting, we got pretty high accuracy even with just 2 columns.

# Plot with data points and classifier separation:

# In[ ]:


from matplotlib.colors import ListedColormap
import numpy as np
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.3, stop = X_set[:, 0].max() + 0.3, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 0.3, stop = X_set[:, 1].max() + 0.3, step = 0.01))
plt.contourf(X1, X2, classifier_log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#024000', '#600000')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#4dcb49', '#ff3b3b'))(i), label = j, s=5)
plt.xlabel('loan_amnt')
plt.ylabel('int_rate')
plt.show()


# **Green and red points** represent paid and default loans, **green and red regions** represent model classification.
# As we see, there is no room for improvement: default and paid loans are well mixed up - the model cant separate them.  Model is actually doing its best to separate major groups of loans.

# # **Conclusion**
# 
# Logistic Regressoin is our best model in terms of accuracy, although its not much better than Naive Bayes or Random Forest models.
# From the plot we saw, that model is doing its best to separate default and paid loans, but we still got maximum accuracy of 64,5%. We need to find different ways to improve the prediction accuracy.
# 
