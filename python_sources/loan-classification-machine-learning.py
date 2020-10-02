#!/usr/bin/env python
# coding: utf-8

# # Lending Club Loan Data
# ## Loan Category Classifier Machine Learning
# 
# #### by: Albertus Rianto Wibisono - 28 Nov 2019
# 
# data source: https://www.kaggle.com/wendykan/lending-club-loan-data

# This notebook is utilizing data obtained from the link provided above. This is a good notebook for ML amateur pracitioner to learn:
# 1. Visualize data using matplotlib and seaborn
# 2. Creating machine learning model and gradually find ways to increase the ML performance that generates the most accurate classification
# 
# A thing to note here is that this notebook runs for a long time (almost 12 minutes on my laptop) because the size of the dataset itself and the complexity of the ML model.

# ### 1.) Data Import

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

startall = time.time()

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.2f}'.format # to set the displayed data as in the two decimal format


# In[ ]:


start = time.time()

df = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', low_memory = False)

stop = time.time()
duration = stop-start
print('It took {:.2f} seconds to read the entire csv file.'.format(duration))

df.head()


# ### 2.) Exploratory Data Analysis

# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


# Analyzing the missing value in each columns
df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': round(100*df.isnull().sum()/len(df),2)})
df_null[df_null['Count'] != 0] 


# In[ ]:


# Visualize the percentage of missing values in columns that have more than 70% missing values
df_null_70up = df_null[df_null['Percent'] >= 70]
df_null_70up = df_null_70up.sort_values(
    by=['Percent'], 
    ascending=False
)

plt.figure(figsize=(15,8))
barchart = sns.barplot(
    df_null_70up.index, 
    df_null_70up['Percent'],
    palette='Set2'
)

barchart.set_xticklabels(barchart.get_xticklabels(), rotation=45, horizontalalignment='right')


# Columns with total misisng values more than 70% would be unnecessary for further analysis and might eventually lead to a inaccurate result in the final model

# In[ ]:


# Remove columns which missing values > 70%
df_1 = df.dropna(axis=1, thresh=int(0.70*len(df)))
df_1.head()


# In[ ]:


print(
    'The number of columns has reduced from {} to {} columns by removing columns with 70% missing values'.
    format(len(df.columns), len(df_1.columns))
)


# ### 3.) Data Visualization 

# **3.1. Loan Status**

# In[ ]:


plt.figure(figsize = (15,5))
plot1 = sns.barplot(df.loan_status.value_counts().index, df.loan_status.value_counts(), palette = 'Set1')
plt.xticks(rotation = 45, horizontalalignment='right')
plt.yticks(fontsize = 12)
plt.title("Loan Status Distribution", fontsize = 20, weight='bold')
plt.ylabel("Count", fontsize = 15)

total = len(df_1)
sizes = []
for p in plot1.patches:
    height = p.get_height()
    sizes.append(height)
    plot1.text(p.get_x() + p.get_width()/2.,
            height + 10000,
            '{:1.3f}%'.format(height/total*100),
            ha = "center", 
            fontsize = 10) 


# Based on the existing loan_status, I will choose only rows which loan status = ('fully paid', 'default', 'charged off') in order to easily categorize them into *good* or *bad* loans.

# In[ ]:


selected_loan_status = ['Fully Paid', 'Charged Off', 'Default']
df_2 = df_1[df_1.loan_status.isin(selected_loan_status)]
df_2.loan_status = df_2.loan_status.replace({'Fully Paid' : 'Good Loan'})
df_2.loan_status = df_2.loan_status.replace({'Charged Off' : 'Bad Loan'})
df_2.loan_status = df_2.loan_status.replace({'Default' : 'Bad Loan'})


# In[ ]:


print(
    'The number of rows has been reduced from {:,.0f} to {:,.0f} by filtering the data with the correlated loan status'.
    format(len(df_1), len(df_2))     
)


# **3.2. Loan's Term**

# In[ ]:


plt.figure(figsize=(8, 5))
plot2 = sns.countplot(df_2.term, hue = df_2.loan_status)
plt.title("Loan's Term Distribution", fontsize = 20, weight='bold')
plt.ylabel("Count", fontsize = 15)
plt.xlabel("Term", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

total = len(df_2)
sizes = []
for p in plot2.patches:
    height = p.get_height()
    sizes.append(height)
    plot2.text(p.get_x() + p.get_width()/2.,
            height + 10000,
            '{:1.0f}%'.format(height/total*100),
            ha = "center", 
            fontsize = 12) 


# From the graph above, it can be noted that more borrowers in Lending Club are using 36 months loan's term. It is also important to see that while fewer people using 60 months loan's term, but one third of them is categorised as *bad loan* while only 16% of the total borrowers who use 36 months loan's term is considered as *bad*.

# **3.3. Loan Amount**

# In[ ]:


plt.figure(figsize = (10,7))
sns.distplot(df.loan_amnt, bins=20)
plt.title('Loan Amount Distribution', fontsize = 20, weight='bold')
plt.xlabel('Loan Amount', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)


# **3.4. Interest Rate**

# In[ ]:


plt.figure(figsize = (10,7))
sns.countplot(round(df.int_rate, 0).astype(int))
plt.title('Interest Rate Distribution', fontsize = 20, weight='bold')
plt.xlabel('Interest Rate (Rounded)', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)


# It can be inferred that many loan done in Lending Club has interest rate lies between 7-18 %

# **3.5. Grade**

# In[ ]:


plt.figure(figsize = (16,5))
plot3 = sns.countplot(df_2.sort_values(by='grade').grade, hue = df_2.loan_status)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title("Grade Distribution", fontsize = 20, weight='bold')
plt.xlabel("Grade", fontsize = 15)
plt.ylabel("Count", fontsize = 15)

total = len(df_2)
sizes = []
for p in plot3.patches:
    height = p.get_height()
    sizes.append(height)
    plot3.text(p.get_x() + p.get_width()/2.,
            height + 3000,
            '{:1.2f}%'.format(height/total*100),
            ha = "center", 
            fontsize = 10) 


# From the chart above we see that as the grade is degrading, more loans are categorised as *bad* rather than *good*, which is reasonable because a lower grade means that the risk of defaulting is also increasing

# **3.6. Loan Purpose**

# In[ ]:


plt.figure(figsize = (15,5))
plot4 = sns.barplot(df.purpose.value_counts().index, df.purpose.value_counts(), palette = 'Set1')
plt.xticks(rotation = 30, fontsize = 12, horizontalalignment='right')
plt.yticks(fontsize = 12)
plt.title("Loan Purpose Distribution", fontsize = 20, weight='bold')
plt.ylabel("Count", fontsize = 15)

total = len(df_1)
sizes = []
for p in plot4.patches:
    height = p.get_height()
    sizes.append(height)
    plot4.text(p.get_x() + p.get_width()/2.,
            height + 10000,
            '{:1.2f}%'.format(height/total*100),
            ha = "center", 
            fontsize = 10) 


# It can be inferred that almost 80% of borrower in Lending Club has the purpose to re-pay their previous debt

# **3.7. Loan Status Distribution based by Loan Amount and Interest Rate**

# In[ ]:


plt.figure(figsize = (20,11))
sns.boxplot(df_2.loan_status, df_2.loan_amnt, hue = df_2.term, palette = 'Paired')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Loan Status Categories", fontsize = 15)
plt.ylabel("Loan Amount Distribution", fontsize = 15)
plt.title("Loan Status by Loan Amount", fontsize = 20, weight='bold')


# In[ ]:





# A nice thing to note here is that either *bad loan* and *good loan* has relatively the same loan amount (for both loan's terms). Except for good loan with 60 months terms, the amount average is slightly higher compared to the bad loan's average number. This means that the amount of loan alone can't necessarily predict the category of the loan.

# In[ ]:


plt.figure(figsize = (20,11))
sns.boxplot(df_2.loan_status, round(df_2.int_rate, 0).astype(int), hue = df_2.term, palette = 'Paired')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Loan Status Categories", fontsize = 15)
plt.ylabel("Interest Rate Distribution", fontsize = 15)
plt.title("Loan Status by Interest Rate", fontsize = 20, weight='bold')


# The graphic above shows that *bad loan* has relatively higher interest rate compared to *good loan*. This means that with higher interest rate, borrowers are more unlikely to repay their debt, and this is totally reasonable.

# **3.8. Verification Status**

# In[ ]:


plt.figure(figsize=(12, 7))
plot5 = sns.countplot(df_2.verification_status, hue = df_2.loan_status, palette = 'inferno')
plt.title("Verification Status Distribution", fontsize = 20)
plt.xlabel("Verification Status", fontsize = 15)
plt.ylabel("Count", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

total = len(df_2)
sizes = []
for p in plot5.patches:
    height = p.get_height()
    sizes.append(height)
    plot5.text(p.get_x() + p.get_width()/2.,
            height + 5000,
            '{:1.0f}%'.format(height/total*100),
            ha = "center", 
            fontsize = 12)


# By comparing the height of the bar of good and bad loan for each verification status, we can imply that **'verified'** status has the biggest percentage of bad loan (almost 30%). Somehow, this doesn't make sense in the first place. This might create a misleading conclusion that '*verification status'* might not be the best consideration to predict the loan's quality.

# **3.9. Top 20 Job Titles for Each Grades**

# In[ ]:


most_emp_title = df_2.emp_title.value_counts()[:20].index.values  # get the top 20 most frequent employee job title
cm = sns.light_palette("orange", as_cmap=True)

round(pd.crosstab(df_2[df_2['emp_title'].isin(most_emp_title)]['emp_title'], 
                  df_2[df_2['emp_title'].isin(most_emp_title)]['grade'], 
                  normalize='index') * 100,2).style.background_gradient(cmap = cm)


# The crosstab functions builds a cross-tabulation table that can show the frequency with which certain groups of data appear.
# From the visualization above, we can see that for loans with grade **'A'**, the most frequent borrowers are employee with job titles = ('Director', 'Engineer', 'President', 'Vice President') and other high-paying job titles. This means that these job titles have a lower risk of defaulting so that their loans is graded higher, which is reasonable. 

# ### 4.) Machine Learning

# #### 4.A) First Trial

# For the first trial, I'm gonna select data with only potentially related features. To filter the dataframe with this method, one has to really understand the business knowledge behind the data very well. 
# 
# The fewer number of column is, means that the model running time will also be faster. 

# In[ ]:


df_3 = df_2[[
    'loan_status', 'term','int_rate',
    'installment','grade', 'annual_inc',
    'verification_status','dti'  # These features are just initial guess, you can try to choose any other combination
]]
df_3.head()


# In[ ]:


# Find missing values in the chosen columns
df_null = pd.DataFrame({'Count': df_3.isnull().sum(), 'Percent': round(100*df_3.isnull().sum()/len(df_3),2)})
df_null[df_null['Count'] != 0] 


# In[ ]:


# Dropping rows with null values
df_clean = df_3.dropna(axis = 0)


# In[ ]:


print('Number of dropped rows: {} rows'.format(len(df_3)-len(df_clean)))


# In[ ]:


# The next step is to transform categorical target variable into integer
df_clean.loan_status = df_clean.loan_status.replace({'Good Loan' : 1})
df_clean.loan_status = df_clean.loan_status.replace({'Bad Loan' : 0})
df_clean.loan_status.unique()


# We also have to transform categorical feature columns using *one hot encoding*

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df_clean['term'] = label.fit_transform(df_clean['term'])
df_clean['grade'] = label.fit_transform(df_clean['grade'])
df_clean['verification_status'] = label.fit_transform(df_clean['verification_status'])


# Split data into target column (x) and features (y)

# In[ ]:


x = df_clean.drop(['loan_status'], axis=1)
y = df_clean['loan_status']


# Using *OneHotEncoder* to transform categorical columns: loan_term, loan_grade, verification_status

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np 

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,3,5])],      # 0,3,5 refers to the column indexes that need to be transformed      
    remainder = 'passthrough'                               
)                                                         

x = np.array(coltrans.fit_transform(x))


# The next step is, splitting data into training and testing data

# In[ ]:


from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    x,
    y,
    test_size = .2
)


# In[ ]:


print(ytr.value_counts())
print(yts.value_counts())


# If we look to the loan status distribution above, it is clearly noted that the data set is unbalanced, where the amount of *bad* loan is far fewer than than the *good* one.
# 
# An imbalanced data can result in inaccurate / biased classifications of the final output. Therefore, before fitting the data into the machine learning model, we need to rebalance the data with a method called SMOTE (Synthetic Minority Oversampling Technique).

# In[ ]:


from imblearn.over_sampling import SMOTE

smt = SMOTE()
xtr_2, ytr_2 = smt.fit_sample(xtr, ytr)


# In[ ]:


np.bincount(ytr_2)


# Here we can see that after SMOTE function runs, the number of 'good' and 'bad' loan has been balanced.

# **Fitting into Machine Learning Model**

# Random Forest is the most common method used for classification algorithm. Although it has a high risk to overfit the data, but it has been a prominent method to solve cases with classification output.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

start = time.time()

model = RandomForestClassifier()
model.fit(xtr_2, ytr_2)

stop = time.time()
duration = stop-start
print('The training took {:.2f} seconds.'.format(duration))


# In[ ]:


print(round(model.score(xts, yts) * 100, 2), '%')


# While the model gives a good accuracy score, it's better to see the **classification report** and **confusion matrix** to really see how good the model performance is in classifying both classes.

# In[ ]:


y_pred = model.predict(xts)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(yts, y_pred)


# In[ ]:


pd.crosstab(yts, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.metrics import classification_report

target_names = ['Bad Loan', 'Good Loan']
print(classification_report(yts, model.predict(xts), target_names=target_names))


# From the confusion matrix and classification report above, we can really see that the model is biased toward *good loan*. It runs pretty good at predicting *good loan*, but it performs really bad at predicting the *bad loan* (from all 52k actual bad loans, the model predicted only less than half of them right --> 13k)

# #### 4.B.) Second Trial

# In this second trial, I want to try fitting the ML model using the unbalanced dataset (without applying SMOTE) to see if it really impactful to the final performance of the model. Using the same steps from the first trial...

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

start = time.time()

model2 = RandomForestClassifier()
model2.fit(xtr, ytr)

stop = time.time()
duration = stop-start
print('The training took {:.2f} seconds.'.format(duration))


# In[ ]:


print(round(model2.score(xts, yts) * 100, 2), '%')


# In[ ]:


y_pred2 = model2.predict(xts)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(yts, y_pred2)


# In[ ]:


pd.crosstab(yts, y_pred2, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.metrics import classification_report

target_names = ['Bad Loan', 'Good Loan']
print(classification_report(yts, y_pred2, target_names=target_names))


# It seems that the accuracy score gives a slightly higher score compared to the first trial with balanced dataset. But, it also has to be noted that the model shows a worse performance at predicting *good loan* (look at the decreasing classification report scores).

# #### 4.C.) Third Trial

# Another method to deal with unbalanced dataset is by applying NearMiss to perform undersampling in order to also get a balanced dataset.

# In[ ]:


from imblearn.under_sampling import NearMiss

nr = NearMiss()
xtr_3, ytr_3 = nr.fit_sample(xtr, ytr)


# In[ ]:


np.bincount(ytr_3)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

start = time.time()

model3 = RandomForestClassifier()
model3.fit(xtr_3, ytr_3)

stop = time.time()
duration = stop-start
print('The training took {:.2f} seconds.'.format(duration))


# In[ ]:


print(round(model3.score(xts, yts) * 100, 2), '%')


# In[ ]:


y_pred3 = model3.predict(xts)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(yts, y_pred3)


# In[ ]:


pd.crosstab(yts, y_pred3, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.metrics import classification_report

target_names = ['Bad Loan', 'Good Loan']
print(classification_report(yts, y_pred3, target_names=target_names))


# For the classification report generated from this first trial, it seems that undersampling method increase the performance of the ML model to predcit bad loan, but it decreases the performance to predict good loan significantly. As a result, the whole accuracy score is just as bad.
# 
# So far, we haven't been able to increase the performance of the ML model.

# #### 4.D.) Fourth Trial

# Another way to increase ML model performance is by doing hyperparameter tuning, but this is not what I'm gonna do in this notebook.
# 
# Therefore, we could also re-consider the features we have filtered earlier. Beside using domain knowledge, it is also better to choose the features **objectively** by looking at the correlation for each features toward the target variable (loan_status).

# Features can be in the form of number (float/integer) or string (object). For categorical features, we also have to one-hot-encode it first before fitting into the dataset. Choosing a categorical feature which have too many options will only causing the matrix becoming too large and difficult to handle by the computer.

# In[ ]:


# First, by knowing what are the features available in the dataframe
df_4 = df_2


# In[ ]:


# The next step is to transform categorical target variable into integer
df_4.loan_status = df_4.loan_status.replace({'Good Loan' : 1})
df_4.loan_status = df_4.loan_status.replace({'Bad Loan' : 0})


# In[ ]:


df_4.columns.to_series().groupby(df_clean.dtypes).groups


# In[ ]:


# First, dropping categorical features (object type) which have too many options available
df_4 = df_4.drop(['emp_title', 'sub_grade', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'hardship_flag', 'debt_settlement_flag'], axis=1)


# In[ ]:


# Second, to filter numerical features, we can use .corr() function to select only features with high correlation to the target variable
df_4.corr()['loan_status']


# In[ ]:


df_clean = df_4[[
    'loan_status', # target variable
    # features (object):
    'term', 'grade','home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 
    'initial_list_status', 'application_type', 'disbursement_method',
    # features (int/float):
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'recoveries',                   
    'collection_recovery_fee', 'last_pymnt_amnt', 'int_rate'
]]


# In[ ]:


df_null = pd.DataFrame({'Count': df_clean.isnull().sum(), 'Percent': round(100*df_clean.isnull().sum()/len(df_clean),2)})
df_null[df_null['Count'] != 0] 


# It's good that there's no missing values

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df_clean['term'] = label.fit_transform(df_clean['term'])
df_clean['grade'] = label.fit_transform(df_clean['grade'])
# df_clean['emp_length'] = label.fit_transform(df_clean['emp_length'])
df_clean['home_ownership'] = label.fit_transform(df_clean['home_ownership'])
df_clean['verification_status'] = label.fit_transform(df_clean['verification_status'])
df_clean['pymnt_plan'] = label.fit_transform(df_clean['pymnt_plan'])
df_clean['purpose'] = label.fit_transform(df_clean['purpose'])
df_clean['initial_list_status'] = label.fit_transform(df_clean['initial_list_status'])
df_clean['application_type'] = label.fit_transform(df_clean['application_type'])
df_clean['disbursement_method'] = label.fit_transform(df_clean['disbursement_method'])


# In[ ]:


df_clean.head()


# In[ ]:


x = df_clean.drop(['loan_status'], axis=1)
y = df_clean['loan_status']


# In[ ]:


x.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np 

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,1,2,3,4,5,6,7,8])],        
    remainder = 'passthrough'                               
)                                                         

x = np.array(coltrans.fit_transform(x))


# In[ ]:


from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    x,
    y,
    test_size = .2
)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import time

start = time.time()

model = RandomForestClassifier()
model.fit(xtr, ytr)

stop = time.time()
duration = stop-start
print('The training took {:.2f} seconds.'.format(duration))


# In[ ]:


print(round(model.score(xts, yts) * 100, 2), '%')


# In[ ]:


y_pred = model.predict(xts)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(yts, y_pred)


# In[ ]:


pd.crosstab(yts, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.metrics import classification_report

target_names = ['Bad Loan', 'Good Loan']
print(classification_report(yts, model.predict(xts), target_names=target_names))


# Finally, the fourth trial shows the best result from the ML model in predicting loan's classification. Beside generating a good accuracy score, the classification report also shows a magnificent result for both classification.

# In[ ]:


import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(xts)
preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(yts, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# Plotting the ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# It even shows a good AUC ROC score!

# In[ ]:


import math

stopall = time.time()
durationall = stopall-startall
duration_mins = math.floor(durationall/60)
duration_secs = durationall - (duration_mins*60)

print('The whole notebook runs for {} minutes {:.2f} seconds.'.format(duration_mins, duration_secs))

