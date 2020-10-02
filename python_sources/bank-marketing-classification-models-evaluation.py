#!/usr/bin/env python
# coding: utf-8

# # About the dataset
# * This dataset gives you information about a marketing campaign of a financial institution, which you will have to analyze in order to find ways to look for future strategies in order to improve future marketing campaigns for the bank.
# ## Input variables explained:
# ### bank client data:<br> 
# 1 - age (numeric)<br> 
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br> 
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br> 
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br> 
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')<br> 
# 6 - balance: average yearly balance, in euros (numeric)
# 7 - housing: has housing loan? (categorical: 'no','yes','unknown')<br> 
# 8 - loan: has personal loan? (categorical: 'no','yes','unknown')<br> 
# ### Related with the last contact of the current campaign:
# 9 - contact: contact communication type (categorical: 'cellular','telephone')<br> 
# 10 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br> 
# 11 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br> 
# 12 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br> 
# ### Other attributes:
# 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br> 
# 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br> 
# 15 - previous: number of contacts performed before this campaign and for this client (numeric)<br> 
# 16 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br> 
# ## Output variable (desired target):
# 17 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# * Deposite definition: What is a Term Deposit?
# A Term deposit is a deposit that a bank or a financial institurion offers with a fixed rate (often better  than just opening deposit account) in which your money will be returned back at a specific maturity time. 

# ## Approach
# In order to optimize marketing campaigns with the help of the dataset, we will have to take the following steps:
# * Import data from dataset and perform initial high-level analysis: look at the number of rows, look at the missing values, look at dataset columns and their values respective to the campaign outcome.
# * Clean the data: remove irrelevant columns, deal with missing and incorrect values, turn categorical columns into dummy variables.
# * Here some categorical columns have values "unknown". We are considering it as one category which can influence the deposite status. Hence not removing it.
# * Use machine learning techniques to predict the marketing campaign outcome and to find out factors, which affect the success of the campaign.

# ### Import linear algebra and data manipulation libraries

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#import standard visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


conda install mglearn


# In[ ]:


pip install mglearn


# In[ ]:


df=pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')


# # 1. DATA PREPROCESSING

# ## Checking for Missing values

# In[ ]:


df.isna().count()


# In[ ]:


df.head()


# ## Slicing the dataset instances to 1500

# In[ ]:


df1=df.sample(n=1500,random_state=0)


# In[ ]:


df1=df1.sort_index()
df1


# # Inserting missing values

# * Have inserted 10% missing values into the dataset as the dataset is clean

# In[ ]:


np.random.seed(0)
df2 = df1.mask(np.random.random(df1.shape) < .10)


# In[ ]:


df2


# In[ ]:


missing_values= df2.isna().mean().round(2)
missing_values.sum()


# * we can see that there is nearly 10% of missing values in the data, hence lets explore data by categorical and numerical column wise.

# ## Check for missing values in outcome variable-deposit
# * If the deposit variable has missing values, then it is better to do row deletion.

# In[ ]:


df3 = df2.dropna(how='all', subset=['deposit'])


# ### Fill missing values 
# * with most frequent values in categorical columns
# * with mean in numerical columns(as we have only less than 10% of the data is missing, filling with average should not decrease the variance much to deviate our predictions.)

# In[ ]:


df_cat_imputed=df3.select_dtypes(include='object').fillna(df3.select_dtypes(include='object').mode().iloc[0])
df_cat_imputed


# In[ ]:


df_num_imputed=df3.select_dtypes(exclude ='object').fillna(df3.select_dtypes(exclude='object').mean().iloc[0])


# In[ ]:


df_imputed = pd.concat([df_cat_imputed, df_num_imputed], axis=1)
df_imputed


# In[ ]:


df_imputed.isna().sum()


# * We will check how the categorical columns are distributed

# In[ ]:


cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','poutcome']
fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))
counter = 0
for cat_column in cat_columns:
    value_counts = df_imputed[cat_column].value_counts()
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index)
    
    axs[trace_x, trace_y].set_title(cat_column)
    
    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    counter += 1

plt.show()


# * We will look at the numerical columns

# In[ ]:


num_columns = ['age','balance', 'day','duration', 'campaign', 'pdays', 'previous']
df3_num=df_imputed[num_columns]
df3_num


# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.gca()
df3_num.hist(ax=ax)
plt.show()


# * We can see from the above graphs, that balance,campaign, duration, pdays and previous variables have some outliers. lets look at those columns

# In[ ]:


df3_num.describe()


# ## Handling Outliers
# We can see that duration, pdays have outliers

# In[ ]:


df_imputed.groupby('deposit').mean()


# In[ ]:


len(df_imputed[df_imputed['pdays'] > 400] ) / len(df_imputed) * 100


# In[ ]:


len(df_imputed[df_imputed['pdays'] == -1.0])/len(df_imputed)*100


# Box plot to show outliers in pdays

# In[ ]:


plt.boxplot(df_imputed['pdays'])
plt.show()


# Removing Outliers from pdays column

# In[ ]:


print(df_imputed['pdays'].quantile(0.10))
print(df_imputed['pdays'].quantile(0.90))


# In[ ]:


df_imputed['pdays'] = np.where(df_imputed['pdays'] >185.0, 185.0,df_imputed['pdays'])


# Box plot to show outliers in duration

# In[ ]:


len (df_imputed[df_imputed['duration'] > 1700] ) / len(df_imputed) * 100


# In[ ]:


plt.boxplot(df_imputed['duration'])
plt.show()


# Removing Outliers from duration column

# In[ ]:


print(df_imputed['duration'].quantile(0.10))
print(df_imputed['duration'].quantile(0.90))


# In[ ]:


df_imputed['duration'] = np.where(df_imputed['duration'] >815.1, 815.1,df_imputed['duration'])


# ## Response column(y)-Deposit
# On the diagram we see that counts for 'yes' and 'no' values for 'deposit' are close, so we can use accuracy as a metric for a model, which predicts the campaign outcome.

# In[ ]:


value_counts = df_imputed['deposit'].value_counts()

value_counts.plot.bar(title = 'Deposit value counts')


# In[ ]:


df_imputed.describe()


# In[ ]:


df_imputed.groupby('deposit').mean()


# ## Heatmap to check correlation between variables

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(data=df_imputed.corr(), annot=True, cmap='viridis')


# In[ ]:


# Build correlation matrix
corr = df_imputed.corr()
corr.style.background_gradient(cmap='PuBu')


# * From the above heatmap, it seems like there is no correlation between input numerical variables, hence we do not need to drop any variables

# ## Creating dummies for categorical variables

# In[ ]:


print("Unique levels in 'job' variable:", df_imputed.job.nunique())
print("Unique levels in 'marital' variable:", df_imputed.marital.nunique())
print("Unique levels in 'education' variable:", df_imputed.education.nunique())
print("Unique levels in 'default' variable:", df_imputed.default.nunique())
print("Unique levels in 'housing' variable:", df_imputed.housing.nunique())
print("Unique levels in 'loan' variable:", df_imputed.loan.nunique())
print("Unique levels in 'contact' variable:", df_imputed.contact.nunique())
print("Unique levels in 'month' variable:", df_imputed.month.nunique())
print("Unique levels in 'poutcome' variable:", df_imputed.poutcome.nunique())
print("Unique levels in 'deposit' variable:", df_imputed.deposit.nunique())


# * From the information above, we will create one-hot encoding for categorical variables with > 2 levels. So, 'job',  'marital', 'education', 'contact', 'month' and 'poutcome' variables have levels >2.
# * For 'default', 'housing', 'loan' and 'deposit' variables, we use create label encoding as they have just 2 unique levels.

# In[ ]:


dummy1= pd.get_dummies(df_imputed, columns=['job', 'marital', 'education','contact', 'month','poutcome'],
               drop_first=False, prefix=['job', 'mar', 'edu', 'con', 'mon', 'pout'])


# In[ ]:


dummy2=dummy1.replace(to_replace = ['yes','no'],value = ['1','0'])
#dummy2.info()
df_conv=dummy2.copy()
df_conv.head()


# In[ ]:


y = df_conv.deposit
X = df_conv.drop(['deposit'], axis=1)


# ## Feature Selection

# For selecting feautres we will use random forest method because it is robust, nonlinear

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
X_sc = pd.DataFrame(scaled_features,columns=X.columns)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators = 50, max_depth = 4,random_state=0)

scores = []
num_features = len(X_sc.columns)
for i in range(num_features):
    col = X_sc.columns[i]
    score = np.mean(cross_val_score(clf, X_sc[col].values.reshape(-1,1), y, cv=10))
    scores.append((float(score*100), col))

print(sorted(scores, reverse = True))


# * From the above Cross value scores of the input variables, lets select the top 15 variables with highest scores
# * So the features selected for the classification modeling are: 'duration', 'pdays', 'pout_success', 'pout_unknown', 'previous','age','con_unknown','job_retired','mar_single','housing','con_cellular','mon_apr','mon_may','job_student', 'mon_sep'

# In[ ]:


df_final=X[['duration','pdays','pout_success','pout_unknown','previous',
            'age','con_unknown','job_retired','mar_single','housing', 
            'con_cellular','mon_apr','mon_may','job_student','mon_sep']]
df_final.info()


# # 2. Classification

# ## Split the data into train and test

# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.model_selection import train_test_split

X_train_org, X_test_org, y_train, y_test = train_test_split(df_final,y, random_state = 0)

print("Size of training set: {}  size of test set:"
      " {}\n".format(X_train_org.shape[0],X_test_org.shape[0]))


# ## Feature Scaling
# * As we can see from the graphs of the input varibles, it is clear that they do not have normally distributed data, hence we are using MinMaxScaling. This will be suitable option as we have also removed the outliers

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train2 = pd.DataFrame(scaler.fit_transform(X_train_org))
X_test2= pd.DataFrame(scaler.transform(X_test_org))
X_train2.columns = X_train_org.columns.values
X_test2.columns = X_test_org.columns.values
X_train2.index = X_train_org.index.values
X_test2.index = X_test_org.index.values
X_train = X_train2
X_test = X_test2


# In[ ]:


print("Checking the balance status of y train data set\n",y_train.value_counts())


# * From the above counts of 0 and 1 values for target variable, it looks not that imbalanced. Hence we can proceed.

# ## Grid Search & cross validation applied on classification models

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


# ## Method1: KNN-Grid search with Cross validation

# In[ ]:


knn = KNeighborsClassifier()

knn_param_grid = {'n_neighbors':[1,2,3,5,7,10,15,25],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

#Fit the model 5-fold cross validation
KNN_grid = GridSearchCV(knn, knn_param_grid, cv=5)
best_knn=KNN_grid.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_knn = pd.DataFrame([['KNN_GridCV', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
Results_all=model_results_knn

print("Best Train Accuracy score: ",KNN_grid.best_score_)
print("Best parameters:", KNN_grid.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_n_neighbors', y='mean_test_score', data=pd.DataFrame(KNN_grid.cv_results_))


# ## Method2: Logistic Regression - Grid search with Cross validation

# In[ ]:


lreg=LogisticRegression(random_state = 0)
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = [0.001,0.01,0.1,0.2,0.8,1.2,1.5]
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
GS_lreg = GridSearchCV(lreg, hyperparameters, cv=5, verbose=0)
# Fit grid search
LR_best_model = GS_lreg.fit(X_train, y_train)
y_pred = GS_lreg.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_lr = pd.DataFrame([['Logistic Regression_GridCV', acc, prec, rec, f1]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
Results_all = Results_all.append(model_results_lr, ignore_index = True)

print("Best Train Accuracy score: ",GS_lreg.best_score_)
print("Best parameters:", GS_lreg.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(GS_lreg.cv_results_))


# ## Method3: Linear SVC - Gridsearch with Cross Validation

# In[ ]:


classifier_LinSVM = LinearSVC()
# Grid serach for hyperparameter tuning
param_grid_svm = {'C': [0.001, 0.01, 0.10, 1, 10,100]}  
 
Linsvm_grid = GridSearchCV(classifier_LinSVM, param_grid_svm, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
LinSVM_best_model= Linsvm_grid.fit(X_train, y_train) 
#Predict test data using best model
y_pred=LinSVM_best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')


model_results_Linsvm = pd.DataFrame([['LinearSVM_GridCV', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
Results_all = Results_all.append(model_results_Linsvm, ignore_index = True)

print("Best Train Accuracy score: ",Linsvm_grid.best_score_)
print("Best parameters:", Linsvm_grid.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(Linsvm_grid.cv_results_))


# ## Method4: SVM_lin - Grid search with Cross Validation

# In[ ]:


classifier_SVM_lin = SVC()
# Grid serach for hyperparameter tuning 
param_grid_svm_lin = {'C': [0.001,0.01,0.1, 1, 10, 50,100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  
svm_lin_grid = GridSearchCV(classifier_SVM_lin, param_grid_svm_lin, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
SVMlin_best_model= svm_lin_grid.fit(X_train, y_train) 
#Predict test data using best model
y_pred = SVMlin_best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_svm = pd.DataFrame([['SVM (Linear)_GridCV', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
Results_all = Results_all.append(model_results_svm, ignore_index = True)

print("Best Train Accuracy score: ",svm_lin_grid.best_score_)
print("Best parameters:", svm_lin_grid.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_lin_grid.cv_results_))
#pd.DataFrame(svm_lin_grid.cv_results_)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_lin_grid.cv_results_).mean_test_score).reshape(7, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid_svm_lin['gamma'], ylabel='C', yticklabels=param_grid_svm_lin['C'], cmap="viridis")


# ## Method5: SVM_rbf - Grid Search with Cross Validation

# In[ ]:


classifier_SVM_rbf = SVC()
# Grid search for hyperparameter tuning 
param_grid_svm_rbf = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
svm_rbf_grid = GridSearchCV(classifier_SVM_rbf, param_grid_svm_rbf, refit = True, verbose = 3, cv = 5, scoring='roc_auc') 
  
# fitting the model for grid search 
SVMrbf_best_model= svm_rbf_grid.fit(X_train, y_train) 
# Predicting test data using best model
y_pred = SVMrbf_best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_SVM_rbf = pd.DataFrame([['SVM(RBF)_GridCV', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

Results_all = Results_all.append(model_results_SVM_rbf, ignore_index = True)

print("Best Train Accuracy score: ",SVMrbf_best_model.best_score_)
print("Best parameters:", SVMrbf_best_model.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_rbf_grid.cv_results_))
#pd.DataFrame(svm_rbf_grid.cv_results_)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_rbf_grid.cv_results_).mean_test_score).reshape(5, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid_svm_rbf['gamma'], ylabel='C', yticklabels=param_grid_svm_rbf['C'], cmap="viridis")


# ## Method6: SVM_poly - Grid Search with Cross Validation

# In[ ]:


classifier_SVM_poly = SVC()
# Grid search for hyperparameter tuning 
param_grid_svm_poly = {'C': [0.1, 1, 10, 20, 100], 'degree': [2,3,4],'kernel': ['poly']}  
svm_poly_grid = GridSearchCV(classifier_SVM_poly, param_grid_svm_poly, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
SVMpoly_best_model= svm_poly_grid.fit(X_train, y_train) 
y_pred = SVMpoly_best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_SVM_poly = pd.DataFrame([['SVM(POLY)_GridCV', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

Results_all = Results_all.append(model_results_SVM_poly, ignore_index = True)

print("Best Train Accuracy score: ",svm_poly_grid.best_score_)
print("Best parameters:", svm_poly_grid.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# In[ ]:


sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_poly_grid.cv_results_))
#pd.DataFrame(svm_poly_grid.cv_results_)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_poly_grid.cv_results_).mean_test_score).reshape(3, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='C', xticklabels=param_grid_svm_poly['C'], ylabel='degree', yticklabels=param_grid_svm_poly['degree'], cmap="viridis")


# ## Method7: Decision tree classifier - Grid search with cross validation

# In[ ]:


classifier_dec = DecisionTreeClassifier(random_state = 0)
param_grid_dec = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10],
              "max_depth": [2, 5, 10]
              }
dec_grid = GridSearchCV(classifier_dec, param_grid_dec, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
dec_best_model= dec_grid.fit(X_train, y_train) 

# Predicting Test Set
y_pred = dec_best_model.predict(X_test) 

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_results_dec = pd.DataFrame([['Decision tree_GridCV', acc, prec, rec, f1 ]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
Results_all = Results_all.append(model_results_dec, ignore_index = True)
  
print("Best Train Accuracy score: ",dec_grid.best_score_)
print("Best parameters:", dec_grid.best_estimator_)
print("Best Test Accuracy score :", accuracy_score(y_test, y_pred))


# # Model Comparision1

# In[ ]:


print(Results_all)
Proj1_results=Results_all.copy()


# * Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes as in the above case. 
# * In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on.
# * We can see from the above scores results that, Logistic Regression has better accuracy score and  F1_Score, which is greater than all other models. 
# * We can consider this as the best model after applying Grid search and cross validation techniques.

# ## Performance of Classification models on reduced dataset

# * Reducing the dataset using PCA to retain 95% variance

# In[ ]:


from sklearn.decomposition import PCA
pca_red = PCA(n_components=0.95)
X_train_red = pd.DataFrame(pca_red.fit_transform(X_train))
X_test_red = pd.DataFrame(pca_red.transform(X_test))


# In[ ]:


X_train_red.info()


# ### Method1: KNN-GridSearchCV on reduced dataset

# In[ ]:


knn = KNeighborsClassifier()

knn_param_grid = {'n_neighbors':[1,2,3,5,7,10,15,25],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

#Fit the model 5-fold cross validation
KNN_grid = GridSearchCV(knn, knn_param_grid, cv=5)
KNN_grid.fit(X_train_red, y_train)
y_pred=KNN_grid.predict(X_test_red)
#print("Best Accuracy score: ",KNN_grid.best_score_)
#print("Best parameters:", KNN_grid.best_estimator_)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_knn_grid_red = pd.DataFrame([['KNN classifier(GridSearch)_red', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(KNN_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(KNN_grid.score(X_test_red, y_test)))
print("Best parameters:", KNN_grid.best_estimator_)

#plot
sns.lineplot(x='param_n_neighbors', y='mean_test_score', data=pd.DataFrame(KNN_grid.cv_results_))


# ### Method2: LogisticRegression-GridSearch on reduced dataset

# In[ ]:


from sklearn.linear_model import LogisticRegression
lreg=LogisticRegression(random_state = 0)
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = [0.001,0.01,0.1,0.2,0.8,1.2,1.5]
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 2-fold cross validation
lreg_grid = GridSearchCV(lreg, hyperparameters, cv=5, verbose=0)
# Fit grid search
LR_best_grid = lreg_grid.fit(X_train_red, y_train)
y_pred=LR_best_grid.predict(X_test_red)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_lreg_grid_red = pd.DataFrame([['LogisticRegression(GridSearch)_red', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(LR_best_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(LR_best_grid.score(X_test_red, y_test)))
print("Best parameters:", lreg_grid.best_estimator_)
#PLot
sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(lreg_grid.cv_results_))


# ### Method3: LinearSVC-GridSearchCV on reduced dataset

# In[ ]:


from sklearn.svm import LinearSVC
classifier_LinSVM = LinearSVC()
# Grid search for hyperparameter tuning
param_grid_svm = {'C': [0.001, 0.01, 0.10, 1, 10,100]}  
 
Linsvm_grid = GridSearchCV(classifier_LinSVM, param_grid_svm, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
LinSVM_best_model= Linsvm_grid.fit(X_train_red, y_train) 
#Predicting test dataset
y_pred = LinSVM_best_model.predict(X_test_red) 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred,pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_linsvm_grid_red = pd.DataFrame([['LinearSVM(GridSearch)_red', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(Linsvm_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(Linsvm_grid.score(X_test_red, y_test)))
print("Best parameters:", Linsvm_grid.best_estimator_)
#plot
sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(Linsvm_grid.cv_results_))


# ### Method4: SVM_kernel="linear"-GridSearchCV on reduced dataset

# In[ ]:


from sklearn.svm import SVC
classifier_SVM_lin = SVC(random_state = 0)
# Grid serach for hyperparameter tuning 
param_grid_svm_lin = {'C': [0.001,0.01,0.1, 1, 10, 50,100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  
svm_lin_grid = GridSearchCV(classifier_SVM_lin, param_grid_svm_lin, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
SVMlin_best_model= svm_lin_grid.fit(X_train_red, y_train) 
#Predicting testset
y_pred = SVMlin_best_model.predict(X_test_red)  
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred,pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_svmlin_grid_red = pd.DataFrame([['SVM_lin(GridSearch)_red', acc, prec, rec, f1 ]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(svm_lin_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(svm_lin_grid.score(X_test_red, y_test)))
print("Best parameters:", svm_lin_grid.best_estimator_)

#Plot
sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_lin_grid.cv_results_))


# In[ ]:


# plot the mean cross-validation scores
get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_lin_grid.cv_results_).mean_test_score).reshape(7, 5)
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid_svm_lin['gamma'], ylabel='C', yticklabels=param_grid_svm_lin['C'], cmap="viridis")


# ### Method5: SVM_kernel="rbf"-GridSearchCV on reduced dataset

# In[ ]:


from sklearn.svm import SVC
classifier_SVM_rbf = SVC(random_state = 0)
# Grid search for hyperparameter tuning 
param_grid_svm_rbf = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
svm_rbf_grid = GridSearchCV(classifier_SVM_rbf, param_grid_svm_rbf, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
SVMrbf_best_model= svm_rbf_grid.fit(X_train_red, y_train) 
# Predicting test set
y_pred = SVMrbf_best_model.predict(X_test_red) 

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred,pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_svmrbf_grid_red = pd.DataFrame([['SVM_rbf(GridSearch)_red', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(svm_rbf_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(svm_rbf_grid.score(X_test_red, y_test)))
print("Best parameters:", svm_rbf_grid.best_estimator_)

#plot
sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_rbf_grid.cv_results_))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_rbf_grid.cv_results_).mean_test_score).reshape(5, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid_svm_rbf['gamma'], ylabel='C', yticklabels=param_grid_svm_rbf['C'], cmap="viridis")


# ### Method6: SVM_kernel="poly"-GridSearchCV on reduced dataset

# In[ ]:


from sklearn.svm import SVC
classifier_SVM_poly = SVC(random_state = 0)
# Grid search for hyperparameter tuning 
param_grid_svm_poly = {'C': [0.1, 1, 10, 20, 100],   
              'kernel': ['poly'],
                'degree' :[2,3,4]}  
svm_poly_grid = GridSearchCV(classifier_SVM_poly, param_grid_svm_poly, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
SVMpoly_best_model= svm_poly_grid.fit(X_train_red, y_train) 
y_pred = svm_poly_grid.predict(X_test_red) 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred,pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_svmpoly_grid_red = pd.DataFrame([['SVM_poly(GridSearch)_red', acc, prec, rec, f1 ]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(svm_poly_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(svm_poly_grid.score(X_test_red, y_test)))
print("Best parameters:", svm_poly_grid.best_estimator_)

#PLOT
sns.lineplot(x='param_C', y='mean_test_score', data=pd.DataFrame(svm_poly_grid.cv_results_))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
scores = np.array(pd.DataFrame(svm_poly_grid.cv_results_).mean_test_score).reshape(3, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='C', xticklabels=param_grid_svm_poly['C'], ylabel='degree', yticklabels=param_grid_svm_poly['degree'], cmap="viridis")


# ### Method7: DecisionTreeClassifier-GridSearchCV on reduced dataset

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_dec = DecisionTreeClassifier(random_state = 0)

param_grid_dec = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2,10],
              "max_depth": [2, 5, 10]
              }
dec_grid = GridSearchCV(classifier_dec, param_grid_dec, refit = True, verbose = 3, cv = 5) 
  
# fitting the model for grid search 
dec_best_model= dec_grid.fit(X_train_red, y_train) 

# Predicting Test Set
y_pred = dec_best_model.predict(X_test_red) 
  
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,pos_label='1')
rec = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred,pos_label='1')

model_dec_grid_red = pd.DataFrame([['DecisionTree(GridSearch)_red', acc, prec, rec, f1 ]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print("Accuracy on training set: {:.3f}".format(dec_grid.score(X_train_red, y_train)))
print("Accuracy on test set: {:.3f}".format(dec_grid.score(X_test_red, y_test)))
print("Best parameters:", dec_grid.best_estimator_)


# ## Classification Models prediction scores comparision on original dataset vs. on reduced data

# * The models which were developed on full dataset are compared with models developed on PCA reduced dataset.

# In[ ]:


#Project 2 Models results table
Proj2_models= pd.concat([model_knn_grid_red,
           model_lreg_grid_red,model_linsvm_grid_red,model_svmlin_grid_red,
           model_svmrbf_grid_red,model_svmpoly_grid_red
           ,model_dec_grid_red])


# In[ ]:


#Comparision table
pd.concat([Proj1_results,Proj2_models],ignore_index=True,sort=False)


# * From the above results table we can see that, the accuracy scores of the models that were developed on PCA reduced datasets are slightly smaller compared to the accuracy scores of the models developed on full datasets.But in most of the models, the accuracy is approximately same. 
# 
# * The models have predicted with sligtly better precision on reduced dataset.
# 
# * Hence we can say that even after reducing the dataset to retain 95% variance, the models are predicting the output variable "deposite" with good accuracy.
# 
# * We can conclude from this, that PCA indeed helps in getting good results with faster analysis. But it is always accuracy variance tradeoff, as we loose variance and some information in the data further by reducing the data using PCA.
