#!/usr/bin/env python
# coding: utf-8

# # Banking - Loan Payment Data

# ## Are people going to pay the money back? Lets See!
# 
# <img align="Centre" src="https://images.unsplash.com/photo-1580722434936-3d175913fbdc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60" width="600px"> <br />
# 
# Photo Credits: Jules Bss on Unsplash

# ## Introduction

# <br>The current data set includes details of the 500 people who have opted for loan. Also, the data mentions whether the person has paid back the loan or not and if paid, in how many days they have paid.
# In this project, we will try to draw few insights on sample Loan data.
# 
# <br>Please find the details of dataset below which can help to understand the features in it.
# 
# <br>Loan_id : A unique loan (ID) assigned to each loan customers- system generated
# <br>Loan_status : Tell us if a loan is paid off, in collection process - customer is yet to payoff, or paid off after the collection efforts
# <br>Principal : Principal loan amount at the case origination OR Amount of Loan Applied
# <br>terms : Schedule(time period to repay)
# <br>Effective_date : When the loan got originated (started)
# <br>Due_date : Due date by which loan should be paid off
# <br>Paidoff_time : Actual time when loan was paid off , null means yet to be paid
# <br>Past_due_days : How many days a loan has past due date
# <br>Age : Age of customer
# <br>Education : Education level of customer applied for loan
# <br>Gender : Customer Gender (Male/Female)

# In[ ]:


#Libraries used in the project
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Reading the input dataset 
data = pd.read_csv('../input/loandata/Loan payments data.csv')

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

data.head()


# ## Checkpoint 1

# In[ ]:


df = data.copy()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Spelling Correction

# In[ ]:


#Changed the name of a value in 'education' column from 'Bechalor' to 'Bachelor'
df['education']= df['education'].replace('Bechalor','Bachelor')


# # Filling Missing Values / Data Imputation

# In[ ]:


#Find the number of missing values in each columns
df.isna().sum()


# In[ ]:


#Temporarily Filling the empty values in 'past_due_days' as '0'
df['past_due_days'] = df['past_due_days'].fillna(0)


# In[ ]:


#Filling the empty values in 'paid_off_time' as '-1'
df['paid_off_time'] = df['paid_off_time'].fillna(-1)


# In[ ]:


#Find the number of missing values in each columns
df.isna().sum()


# In[ ]:


#Number of unique values in each column
for cat in data.columns:
    print("Number of levels in category '{0}': \b  {1:2.0f} ".format(cat, df[cat].unique().size))


# In[ ]:


#Coverting the following columns to 'datetime'
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['due_date'] = pd.to_datetime(df['due_date'])
df['paid_off_time'] = pd.to_datetime(df['paid_off_time']).dt.date


# In[ ]:


#To Convert the 'paid_off_time' column to datetime64 type
df['paid_off_time'] = pd.to_datetime(df['paid_off_time'])
df.head()


# In[ ]:


df.info()


# ## CheckPoint 2

# In[ ]:


df_fe = df.copy()


# In[ ]:


df_fe.head()


# # Feature Engineering

# #### Replacing values in past_due_days column for PAIDOFF class

# In[ ]:


for i in range(len(df_fe[df_fe['loan_status']=="PAIDOFF"])):
    df_fe['past_due_days'][i] = (df_fe['paid_off_time'][i] - df_fe['effective_date'][i] + pd.Timedelta(days=1)).days - df_fe['terms'][i]
df_fe.head(10)


# In[ ]:


#Records where the difference in the paid_off_time and effective_date is greater than the terms
df_fe[(df_fe['past_due_days']>0)&(df_fe['loan_status']=='PAIDOFF')]


# # Feature Analysis

# ### Loan Status Analysis

# In[ ]:


a = df_fe['loan_status'].value_counts()
pd.DataFrame(a)


# In[ ]:


plt.pie(df_fe['loan_status'].value_counts(),labels=df_fe['loan_status'].unique(),explode=[0,0.1,0],startangle=144,autopct='%1.f%%')
plt.title('Loan Status Distribution',fontsize = 20)
plt.show()


# #### Observations:
# *<b>**20%**</b> of the people have <b>**not repaid**</b> the loan <b>**(COLLECTION)**</b><br>
# *<b>**20%**</b> of the people have <b>**repaid**</b> the loan but <b>lately</b> after due date <b>**(COLLECTION_PAIDOFF)**</b><br>
# *<b>**60%**</b> of the people have <b>**repaid**</b> the loan <b>**on time**</b> <b>**(PAIDOFF)**</b><br>

# ### Gender Analysis

# In[ ]:


b= df_fe['Gender'].value_counts()
pd.DataFrame(b)


# In[ ]:


c = df_fe.groupby(['Gender'])['loan_status'].value_counts()
pd.DataFrame(c)


# In[ ]:


plt.figure(figsize = [10,5])
sns.countplot(df_fe['Gender'],hue=df_fe['loan_status'])
plt.legend(loc='upper right')
plt.title('Gender vs Loan Status',fontsize=20)
plt.xlabel('Gender', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# *Around <b>40%</b> of <b>male</b> population have <b>repaid</b> their loan <b>lately (or yet to pay)</b> <br>
# *Around <b>30%</b> of <b>female</b> population have <b>repaid</b> their loan <b>lately (or yet to pay)</b> <br>

# ### Education Analysis

# In[ ]:


d = df_fe['education'].value_counts()
pd.DataFrame(d)


# In[ ]:


plt.figure(figsize = [10,5])
sns.countplot(df_fe['education'],hue=df_fe['loan_status'])
plt.legend(loc='upper right')
plt.title('Education vs Loan Status',fontsize=20)
plt.xlabel('Education', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# * <b>Majority</b> of the loan takers are from <b>High School</b> or <b>College</b> background<br>
# * <b>Very few</b> people from <b>Masters or above</b> background took loan.

# ### Age Analysis

# In[ ]:


for i in df_fe['loan_status'].unique():
    agemean=df_fe[df_fe['loan_status']==i]['age'].mean()
    agemode=df_fe[df_fe['loan_status']==i]['age'].mode()
    print("average age of people whose loan status is'{0}': \b {1:2.2f} and mode is {2}".format(i,agemean, agemode[0]))


# In[ ]:


plt.figure(figsize = [14,5])
sns.countplot(df_fe['age'],hue=df_fe['loan_status'])
plt.legend(loc='upper left')
plt.title('Age vs Loan Status',fontsize=20)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# * <b>Majority</b> of the people who took loan have <b>age</b> ranging from <b>24 years</b> to <b>38 years</b><br>

# ### Principal Analysis

# In[ ]:


e = df_fe['Principal'].value_counts()
pd.DataFrame(e)


# In[ ]:


plt.figure(figsize = [10,5])
sns.countplot(df_fe['Principal'],hue=df_fe['loan_status'])
plt.legend(loc='upper left')
plt.title('Principal vs Loan Status',fontsize=20)
plt.xlabel('Principal', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# *<b>Majority</b> of the people have opted for <b>Principal</b> of $\$800$ and $\$1000$ <br>

# ### Term Analysis

# In[ ]:


plt.figure(figsize = [10,5])
sns.countplot(df_fe['terms'],hue=df_fe['loan_status'])
plt.legend(loc='upper left')
plt.title('Terms vs Loan Status',fontsize=20)
plt.xlabel('Terms', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# *Only <b>few people</b> have opted loan for <b>7 days term</b> <br>
# *Majority of the <b>late payments</b> are from people who have their loan terms as <b>15 days</b> and <b>30 days</b> </b><br>

# ### Loan Effective Date Analysis

# In[ ]:


g = df_fe.groupby(['effective_date'])['loan_status'].value_counts()
pd.DataFrame(g)


# In[ ]:


plt.figure(figsize = [10,5])
dates = df_fe['effective_date'].dt.date
sns.countplot(x=dates, hue=df_fe['loan_status'])
plt.legend(loc='upper right')
plt.title('Effective Date vs Loan Status',fontsize=20)
plt.xlabel('Effective Date', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# #### Observations:
# *On <b>11th and 12th September</b>, loan was given to <b>many people</b> maybe as part of a drive.<br>
# *Most of people who <b>paid latety(or yet to pay)</b> are from these <b>2 days</b>.

# ### Loan Status Distribution

# In[ ]:


# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,8), dpi=1600)
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
sns.countplot(df_fe['loan_status'],hue=df_fe['education'])
plt.legend(loc='upper right')
plt.title('Loan Status vs Education',fontsize=15)
plt.xlabel(None)
plt.ylabel('Count',fontsize=14)

plt.subplot2grid((2,3),(0,1),rowspan=2)
plt.pie(df_fe['loan_status'].value_counts(),labels=df_fe['loan_status'].unique(),explode=[0,0.1,0],startangle=165,autopct='%1.f%%',)
plt.grid(b=True, which='major', axis='y')
plt.title("Loan Status Distribution",fontsize=20)

ax3 = plt.subplot2grid((2,3),(0,2))
sns.countplot(df_fe['loan_status'],hue=df_fe['terms'])
plt.legend(loc='upper right')
plt.title('Loan Status vs Terms',fontsize=15)
plt.xlabel(None)
plt.ylabel('Count',fontsize=14)

ax4 = plt.subplot2grid((2,3),(1,0))
sns.countplot(df_fe['loan_status'],hue=df_fe['Principal'])
plt.legend(loc='upper right')
plt.title('Loan Status vs Principal',fontsize=15)
plt.xlabel('Loan Status',fontsize=14)
plt.ylabel('Count',fontsize=14)

ax5 = plt.subplot2grid((2,3),(1,2))
sns.countplot(df_fe['loan_status'],hue=df_fe['Gender'])
plt.legend(loc='upper right')
plt.title('Loan Status vs Gender',fontsize=15)
plt.xlabel('Loan Status',fontsize=14)
plt.ylabel('Count',fontsize=14)

plt.show()


# ### Age vs Past Due Days

# In[ ]:


px.scatter(df_fe, x="age", y="past_due_days", size ="terms" ,color="loan_status",
           hover_data=['Gender','Principal'], log_x=True, size_max=8)


# #### Observations:
# *Most of the <b>Elder people</b> (35 - 50 years) have paid back loan <b>on time.</b>

# ### Loan Status vs Past Due Days

# In[ ]:


# Relation between loan_status and past_due_days
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = [9,5])
sns.boxplot(x='loan_status', y='past_due_days', data=df_fe)
plt.xlabel('Loan Status', fontsize=16)
plt.ylabel('Past Due Days', fontsize=16)
plt.show()


# #### Observations:
# *We can infer that if people take <b>more than 25 days after due date</b>, they might end up in taking <b>even more time.</b>

# In[ ]:


df_fe = df_fe.drop(['Loan_ID','effective_date','due_date','paid_off_time'],axis = 1)
df_fe.head()


# In[ ]:


df_fe.info()


# In[ ]:


p = df_fe.groupby(['loan_status'])['Principal'].value_counts()
pd.DataFrame(p)


# ## CheckPoint 3

# In[ ]:


df_fe_Pri = df_fe.copy()


# In[ ]:


df_fe_Pri.head()


# ## Dropping Principal Values [300, 500, 700, 900] & 'terms' = 7 days

# In[ ]:


df_fe_Pri[df_fe_Pri['terms']==7]


# In[ ]:


df_fe_Pri[(df_fe_Pri['Principal']!=800) &(df_fe_Pri['Principal']!=1000)]


# In[ ]:


#Dropping rows where 'Principal' is not equal to 800 and 1000 [12 rows]
df_fe_Pri = df_fe_Pri[(df_fe_Pri['Principal']==800) | (df_fe_Pri['Principal']==1000)]


# In[ ]:


#Dropping rows where 'terms' = 7 days [21 rows]
df_fe_Pri = df_fe_Pri[df_fe_Pri['terms']!=7]


# In[ ]:


df_fe_Pri.head()


# In[ ]:


df_fe_Pri.shape


# ## CheckPoint 4

# In[ ]:


df_clean = df_fe_Pri.copy()


# ### Age Classification

# In[ ]:


def age_classification(age):
    if age.item()<21:
        return 'Young'
    elif age.item()>=21 and age.item()<31:
        return 'MidAge'
    elif age.item()>=31 and age.item()<41:
        return 'Senior'
    else:
        return 'Older'


# In[ ]:


#Categorizing age column
df_clean['age'] = df_clean[['age']].apply(age_classification,axis=1)


# ## One hot encoding - 'terms', 'education', 'Principal', 'age' & 'Gender'

# In[ ]:


df_clean.info()


# In[ ]:


df_clean['terms'] = df_clean['terms'].astype('object')
df_clean['Principal'] = df_clean['Principal'].astype('object')


# In[ ]:


#Select the variables to be one-hot encoded
one_hot_features = ['education','Gender','Principal','age','terms']
# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df_clean[one_hot_features],drop_first=True)
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)
# Convert Categorical to Numerical for default column


# In[ ]:


one_hot_encoded.head()


# In[ ]:


df_encoded = pd.concat([df_clean,one_hot_encoded],axis=1)
df_encoded.head()


# In[ ]:


df_encoded.drop(['terms','education','Gender','age','Principal'],axis=1,inplace = True)
df_encoded.head()


# ## Label Encoding - 'loan_status'

# In[ ]:


df_clean['loan_status'].unique()


# In[ ]:


loan_status_dict = {'PAIDOFF':1,'COLLECTION':2,'COLLECTION_PAIDOFF':3}
df_encoded['loan_status'] = df_encoded.loan_status.map(loan_status_dict)
df_encoded.head()


# ## CheckPoint 5

# In[ ]:


df_model = df_encoded.copy()


# In[ ]:


df_model.info()


# In[ ]:


df_model.head()


# In[ ]:


correlation = df_model[df_model.columns].corr()
plt.figure(figsize=(12, 10))
plot = sns.heatmap(correlation, vmin = -1, vmax = 1,annot=True, annot_kws={"size": 10})
plot.set_xticklabels(plot.get_xticklabels(), rotation=30)


# # Model Building

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split   #splitting data

#Standardize rows into uniform scale

X = df_model.drop(['loan_status','past_due_days'],axis=1)
y = df_model['loan_status']

# scaler = MinMaxScaler()#StandardScaler,MinMaxScaler
# scaler.fit(X_Act)#df_model[cols_to_norm]

# # Scale and center the data
# fdf_normalized = scaler.fit_transform(X_Act)

# # # Create a pandas DataFrame
# fdf_normalized_df = pd.DataFrame(data=fdf_normalized, index=X_Act.index, columns=X_Act.columns)

# X = fdf_normalized_df

##Note: In this case, Scaling is not required

X.head()


# In[ ]:


#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=400,test_size=0.30,stratify = y)


# In[ ]:


from collections import Counter
print("y : ",Counter(y))
print("y_train : ",Counter(y_train))
print("y_test : ",Counter(y_test))


# In[ ]:


# Actual Values(of Majority Class) of y_test
y_test.value_counts()
y_test.value_counts().head(1) / len(y_test)


# ### Function To Run Different Models

# In[ ]:


# metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, recall_score


# In[ ]:


def model_train(model, name):
    model.fit(X_train, y_train)                                          # Fitting the model
    y_pred = model.predict(X_test)                                       # Making prediction from the trained model
    cm = confusion_matrix(y_test, y_pred)                               
    print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix
    print(cm)
    print('-----------------------')
    print('-----------------------')
    cr = classification_report(y_test, y_pred)
    print(name +" Classification Report " +" Validation Data")           # Displaying the Classification Report
    print(cr)
    print('------------------------')
    print(name + " Bias")                                                 # Calculating bias
    bias = y_pred - y_test.mean()
    print("Bias "+ str(bias.mean()))
    
    print(name + " Variance")                                             # Calculate Variance
    var = np.var([y_test, y_pred], axis=0)
    print("Variance " + str(var.mean()) )
#     return auc, rec, model
    return model


# ### Let's try to check the metrics with couple of models.

# ## Logistic Regression

# In[ ]:


# Building the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1000,max_iter=500,class_weight='balanced')    # Set Large C value for low regularization to prevent overfitting
# logreg.fit(X_train, y_train)

dt_model = model_train(logreg, "Logistic Regression")
print('_________________________')
print("Coefficients: ",logreg.coef_)                                            # Coefficients for Logistic Regression
print("Intercepts: ",logreg.intercept_)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini',max_depth = 4, min_samples_leaf =2,random_state=101,class_weight='balanced')

dt_model = model_train(dt, "Decision Tree")


# ### From the above results, we can observe that:
# <br>1.F1 Score using Logistic regression = 0.26
# <br>2.F1 Score using Decision Tree = 0.28
# 
# The results are low due to the <b>imbalance</b> in the class categories.
# 
# Let's try to apply sampling methods to overcome this issue

# In[ ]:


# pip install imbalanced-learn


# ### SMOTE - OverSampling

# In[ ]:


#Let us try some sampling technique to remove class imbalance
from imblearn.over_sampling import SMOTE,KMeansSMOTE,SVMSMOTE
#Over-sampling: SMOTE
#SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, 
#based on those that already exist. It works randomly picking a point from the minority class and computing 
#the k-nearest neighbors for this point.The synthetic points are added between the chosen point and its neighbors.
smote = KMeansSMOTE(sampling_strategy='auto')

X_sm, y_sm = smote.fit_sample(X, y)
print(X_sm.shape, y_sm.shape)


# In[ ]:


#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sm,y_sm,random_state=400,test_size=0.30,stratify = y_sm)#,stratify = y


# In[ ]:


from collections import Counter
print("y : ",Counter(y))
print("y_train : ",Counter(y_train))
print("y_test : ",Counter(y_test))


# In[ ]:


# Actual Values(of Majority Class) of y_test
y_test.value_counts()
y_test.value_counts().head(1) / len(y_test)


# ## Logistic Regression

# In[ ]:


# Building the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1000,max_iter=500,class_weight='balanced')#solver ='lbfgs',class_weight='balanced'    # Set Large C value for low regularization to prevent overfitting
# logreg.fit(X_train, y_train)

dt_model = model_train(logreg, "Logistic Regression")
print('_________________________')
print("Coefficients: ",logreg.coef_)                                            # Coefficients for Logistic Regression
print("Intercepts: ",logreg.intercept_)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini',max_depth = 4, min_samples_leaf =2,random_state=101)

dt_model = model_train(dt, "Decision Tree")


# ## Grid Seach

# In[ ]:


from sklearn.model_selection import GridSearchCV

random_grid = {'n_estimators': range(5,20),
              'max_features' : ['auto', 'sqrt'],
              'max_depth' : [10,20,30,40],
              'min_samples_split':[2,5,10],
              'min_samples_leaf':[1,2,4]}

rf = RandomForestClassifier()

rf_gs = GridSearchCV(rf, random_grid, cv = 3, n_jobs=1, verbose=2)

rf_gs.fit(X_train, y_train)
y_pred = rf_gs.predict(X_test)


# In[ ]:


print(rf_gs.best_estimator_)
print('-----------------------')
print("Grid Search Validation Data")
cm = confusion_matrix(y_test, y_pred)                               
print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix
print(cm)
print('-----------------------')
cr = classification_report(y_test, y_pred)
print("Grid Search Classification Report " +" Validation Data")           # Displaying the Classification Report
print(cr)
print('------------------------')
print("Grid Search Bias")                                                 # Calculating bias
bias = y_pred - y_test.mean()
print("Bias "+ str(bias.mean()))
    
print("Grid Search Variance")                                             # Calculate Variance
var = np.var([y_test, y_pred], axis=0)
print("Variance " + str(var.mean()) )


# ## Model Explainability

# In[ ]:


# Import Eli5 package
import eli5
from eli5.sklearn import PermutationImportance

# Find the importance of columns for prediction
perm = PermutationImportance(dt, random_state=10).fit(X_test,dt.predict(X_test))
eli5.show_weights(perm, feature_names = X.columns.tolist())


# # **Conclusion**
# ## From the Decision tree model, we have acheived an accuracy of 72% after applying SMOTE technique on the imbalanced target class.

# # **Thank You**
