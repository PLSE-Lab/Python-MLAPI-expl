#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from scipy.stats import norm, skew 
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/train_csv.csv')
test = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/test.csv.csv')
train.head()


# # **Data overview**

# In[ ]:


train = train.drop('Loan_ID',axis = 1)


# In[ ]:


train.shape


# **Data summary**

# In[ ]:


train.describe()


# In order to see statistics on non-numerical features, one has to explicitly indicate data types of interest in the include parameter.

# In[ ]:


train.describe(include = ['object'])


# In[ ]:


print(train.info())


# In[ ]:


train['Dependents'].value_counts()


# In[ ]:


train['Dependents'].value_counts()


# We change the target from Yes, No into logical expression.

# In[ ]:


target = 'Loan_Status'
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train[target] = encoder.fit_transform(train[target])
train.head()


# In[ ]:


cat_cols = train.dtypes =='object'
cat_cols = list(cat_cols[cat_cols].index)
num_cols = train.dtypes != 'object'
num_cols = list(num_cols[num_cols].index)
num_cols.remove('Loan_Status')


# In[ ]:


train[cat_cols].head()


# In[ ]:


train[num_cols].head()


# # EDA

# Get summary of target variable. 

# In[ ]:


train[target].describe()


# # Visualization

# In[ ]:


sns.countplot(x=target, data=train)


# **Relationship with numerical features**

# In the following section, plot boxes will be made to see the relationship of target varible and numerical features.

# In[ ]:


def num_boxplot(col,target,train,y = 80000):
    data =  pd.concat([train[target], train[col]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=str(target), y=col, data=train)
    fig.axis(ymin=0, ymax=y);


# Applicant Income and target

# In[ ]:


#Maximum value in ApplicantIncome is 81000.
num_boxplot(num_cols[0],target,train,81000)
num_boxplot(num_cols[0],target,train,25000)


# The distributions of applicant income on different loan status are similar. Hence, there're lot of outliers.

# CoapplicantIncome and target

# In[ ]:


#Maximum value in CoapplicantIncome is 41667.
num_boxplot(num_cols[1],target,train,41667)
num_boxplot(num_cols[0],target,train,15000)


# The distributions of applicant income on different loan status are similar. Hence, there're lot of outliers.

# LoanAmount and target

# In[ ]:


num_boxplot(num_cols[2],target,train,700)


# Loan_Amount_Term and target

# In[ ]:


#Maximum loan amount trem is 700. Notice that Q1,Q2 and Q3 are equal
num_boxplot(num_cols[3],target,train,1000)


# The distribution are similar.

# Credit history and traget

# In[ ]:


num_boxplot(num_cols[4],target,train,3)


# The distributions are differnet here. Creidit history is more vary in loan that were disapproved.

# **What are average values of numerical features for each loan status?**

# In[ ]:


train[train.Loan_Status == 1].mean()


# In[ ]:


train[train.Loan_Status == 0].mean()


# **Relationship with categorical variables.**

# Convert target variable into numceric form

# In[ ]:


for col in cat_cols:
    sns.barplot(col, target, data=train, color="darkturquoise")
    plt.show()


# # Missing Data/ Data Cleaning

# In[ ]:


#Join the df together handling the missing data together
all_df = pd.concat([train,test.drop('Loan_ID',axis =1)],axis = 0)
#train = all_df.iloc[1:614]


# In[ ]:


test_id = test['Loan_ID']


# In[ ]:


y = train[target]


# In[ ]:


#Drop the target column, it hasn't dropped in test data set. 
all_df = all_df.drop('Loan_Status',axis = 1)
all_df.head()


# In[ ]:


all_cols = list(all_df.columns)
missing_cols = [col for col in all_cols if all_df[col].isnull().any()]
len(missing_cols)


# There 7 columns with missing values, let's go further.

# In[ ]:


#Function to create a data frame with number and percentage of missing data in a data frame
def missing_to_df(df):
    #Number and percentage of missing data in training data set for each column
    total_missing_df = df.isnull().sum().sort_values(ascending =False)
    percent_missing_df = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data_df = pd.concat([total_missing_df, percent_missing_df], axis=1, keys=['Total', 'Percent'])
    return missing_data_df


# In[ ]:


missing_df = missing_to_df(all_df)
missing_df[missing_df['Total'] > 0]


# Missing in credit history might mean the credit history of the clients are not available. Fill the missing data with 2 means the data aren't available.

# In[ ]:


all_df['Credit_History'] = all_df['Credit_History'].fillna(2)


# Missing in self employed can mean a person is not in labor force or retired. So, we give a new categorical to those people.

# In[ ]:


all_df['Self_Employed'] = all_df['Self_Employed'].fillna('Other')


# There are outliers in Loan Amount (maximum value is 700 and Q3 is 162), so the missing value in this column will be filled with median. The remaining columns with missing values will be filled by median value as well. There size are relatively small, it's safe to do so.

# In[ ]:


from sklearn.impute import SimpleImputer

num_missing = ['LoanAmount',  'Loan_Amount_Term']
cat_missing = ['Gender', 'Married','Dependents']


# In[ ]:


median_imputer = SimpleImputer(strategy = 'median')
for col in num_missing:
    all_df[col] = pd.DataFrame(median_imputer.fit_transform(pd.DataFrame(all_df[col])))


# In[ ]:


freq_imputer = SimpleImputer(strategy = 'most_frequent')
for col in cat_missing:
    all_df[col] = pd.DataFrame(freq_imputer.fit_transform(pd.DataFrame(all_df[col])))


# In[ ]:


missing_df = missing_to_df(all_df)
missing_df[missing_df['Total'] > 0]


# There are no more missing data in our data set.

# **Skewed features**

# In[ ]:


numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# **Box Cox Transformation of (highly) skewed features**

# We use the scipy  function boxcox1p which computes the Box-Cox transformation of **\\(1 + x\\)**. 

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_df[feat] = boxcox1p(all_df[feat], lam)


# Getting dummy categorical features

# In[ ]:


all_df.head()


# **Feature engineering**

# In[ ]:


#Adding total income by combining applicant's income and coapplicant's income
all_df['Total_Income'] = all_df['ApplicantIncome'] + all_df['CoapplicantIncome']


# Label Encoding dependets that contain information in their ordering set.
# 
# Convert 3+ in depedents into 3, and convert the column into numeric feature.

# In[ ]:


all_df = all_df.replace({'Dependents': r'3+'}, {'Dependents': 3}, regex=True)


# In[ ]:


# process column, apply LabelEncoder to categorical features
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
lbl.fit(list(all_df["Dependents"].values))
all_df["Dependents"] = lbl.transform(list(all_df["Dependents"].values))
# shape        
print('Shape all_data: {}'.format(all_df.shape))


# In[ ]:


all_df = pd.get_dummies(all_df)
print(all_df.shape)


# Getting the new train and test sets

# In[ ]:


train = all_df.iloc[:614]
print(train.shape)
train.tail()


# In[ ]:


test = all_df[614:]
print(test.shape)
test.tail()


# **Features selection**

# In[ ]:


def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]
    
    return train, valid, test


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
sel_train, valid, _ = get_data_splits(pd.concat([train,y],axis=1))
feature_cols = sel_train.columns.drop(target)

# Keep 5 features
selector = SelectKBest(f_classif, k=8)

X_new = selector.fit_transform(sel_train[feature_cols], sel_train[target])
X_new


# In[ ]:


# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=sel_train.index, 
                                 columns=feature_cols)
selected_features.head()


# In[ ]:


# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]

# Get the valid dataset with the selected features.
valid[selected_columns].head()


# # Modelling

# Import Libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb


# Train test split for model building.

# **Baseline models**

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train,y,random_state = 1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


#Logistic Regression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
acc_log = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc  = round(accuracy_score(y_test,y_pred)*100,2)
acc_svc


# In[ ]:


#kNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state = 1)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc_random_forest = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


#LGBMClassifier

lgbc = lgb.LGBMClassifier()
lgbc.fit(x_train, y_train)
y_pred = lgbc.predict(x_test)
acc_lgbc = round(accuracy_score(y_test,y_pred)*100,2)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron',  
              'Decision Tree','LGBMClassifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
             acc_decision_tree,acc_lgbc]})
models.sort_values(by='Score', ascending=False)


# **Models with selected features**

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train[selected_columns],y,random_state = 1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# **Logistic Regression**

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
acc_log = round(accuracy_score(y_test,y_pred)*100,2)
acc_log


# Now we compute the coefficient of each features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# * Credit History is highest positive coefficient, implying as the credit history value rises, the probability of Loan Status =1 (Loan approved) increases the most.
# 
# * Total income isn't a good artificial feature to model as it is not included in selected features.

# In[ ]:


coeff_df = pd.DataFrame(train[selected_columns].columns.delete(-1))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# **Support Vector Machines**

# Support Vector Machines is a supervised learning models with associated learning algorithms that analyse data used for classification and regression analysis. 
# 
# Note that the model generates a confidence score which is lower than Logistics Regression model.

# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc  = round(accuracy_score(y_test,y_pred)*100,2)
acc_svc


# **kNeighborsClassifier**

# In clustering, the k-Nearest Neighbors algorithm is an unsupervisied learning model. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors

# KNN confidence score is the lowest.

# In[ ]:


#kNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(accuracy_score(y_test,y_pred)*100,2)
acc_knn


# **Naive Bayes Classifiers**

# Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
# 
# Naive Bayes classifiers confidence score is better than SVM but sightly worse than Logisitics Regression.
# 

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(accuracy_score(y_test,y_pred)*100,2)
acc_gaussian


# **Perceptron**

# It is a linear classifier that makes predictions based on a linear predcition function integrating a set of weights with the feature's vector.
# 
# The model generated confidence score is not high.

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(accuracy_score(y_test,y_pred)*100,2)
acc_perceptron


# **Decision Tree**

# The accuaracy of decision tree is not too high.

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(accuracy_score(y_test,y_pred)*100,2)
acc_decision_tree


# **Random Forest**

# Random foresdt is not bad.

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state = 1)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc_random_forest = round(accuracy_score(y_test,y_pred)*100,2)
acc_random_forest


# **LGBMClassifier**

# In[ ]:


#LGBMClassifier

lgbc = lgb.LGBMClassifier()
lgbc.fit(x_train, y_train)
y_pred = lgbc.predict(x_test)
acc_lgbc = round(accuracy_score(y_test,y_pred)*100,2)
acc_lgbc


# **Model Evaluation**

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron',  
              'Decision Tree','LGBMClassifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
             acc_decision_tree,acc_lgbc]})
models.sort_values(by='Score', ascending=False)


# The models built with selected features are more accurate.

# # Cross valudation

# 
# The following section is using **cross validation** strategy to compare performance of differnence models. f1 is used as our target is binary.
# [ The scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html)

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train[selected_columns].values)
    rmse= (cross_val_score(model, train[selected_columns].values, 
                                   y.values, scoring="f1", cv = kf))
    return(rmse)


# Below show how the models above perform on the data by evaluating the cross-validation rmsle error.

# In[ ]:


log_reg_score = rmsle_cv(logreg)
print("\nLogistic Regression score: {:.4f} ({:.4f})\n".format(log_reg_score.mean(), log_reg_score.std()))


# In[ ]:


svc_score = rmsle_cv(svc)
print("\nSupport Vector Machines score: {:.4f} ({:.4f})\n".format(svc_score.mean(), svc_score.std()))


# In[ ]:


knn_score = rmsle_cv(knn)
print("\nkNN score: {:.4f} ({:.4f})\n".format(knn_score.mean(), knn_score.std()))


# In[ ]:


naive_score = rmsle_cv(gaussian)
print("\nGaussian Naive Bayes score: {:.4f} ({:.4f})\n".format(naive_score.mean(), naive_score.std()))


# In[ ]:


perceptron_score = rmsle_cv(perceptron)
print("\nPerceptron score: {:.4f} ({:.4f})\n".format(perceptron_score.mean(), perceptron_score.std()))


# In[ ]:


decision_tree_score = rmsle_cv(decision_tree)
print("\nDecision_tree score: {:.4f} ({:.4f})\n".format(decision_tree_score.mean(), decision_tree_score.std()))


# In[ ]:


random_forest_score = rmsle_cv(random_forest)
print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(random_forest_score.mean(), random_forest_score.std()))


# In[ ]:


lgbc_score = rmsle_cv(lgbc)
print("\nLGB Classificier score: {:.4f} ({:.4f})\n".format(lgbc_score.mean(), lgbc_score.std()))


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron',  
              'Decision Tree','LGBMClassifier'],
    'Mean_Score': [svc_score.mean(), knn_score.mean(), log_reg_score.mean(), 
              random_forest_score.mean(), naive_score.mean(), perceptron_score.mean(), 
             decision_tree_score.mean(),lgbc_score.mean()]})
models.sort_values(by='Mean_Score', ascending=False)


# Svc , logistic regression and Naive Bayes are always the top 3 highest accurate models in this case.

# In[ ]:


svc = SVC()
svc.fit(train[selected_columns], y)
Y_pred = svc.predict(test[selected_columns])


# In[ ]:


submission = pd.DataFrame({
        "Loan_Id": test_id,
        "Loan_Status": Y_pred
    })
submission.head(10)

