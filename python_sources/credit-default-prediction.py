#!/usr/bin/env python
# coding: utf-8

# # Credit Default Prediction

# **Data Set Information:** The training data contains 22500 observations with the predictor variables as well as the response variable. The test set contains 7500 observations with the response variable removed.
# 
# **Task:** Predict the response variable (default status) for the test data.
# 
# **IMPORTANT:** Please include the variable "ID" in the prediction, so that model accuracy can be evaluated.
# 
# **Variable descriptions:** This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. 
# 
# This study reviewed the literature and used the following 23 variables as explanatory variables: 
# - **X1:** Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
# - **X2:** Gender (1 = male; 2 = female). 
# - **X3:** Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
# - **X4:** Marital status (1 = married; 2 = single; 3 = others). 
# - **X5:** Age (year). 
# - **X6 - X11:** History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: 
# - **X6** = the repayment status in September, 2005; 
# - **X7** = the repayment status in August, 2005; . . .;
# - **X11** = the repayment status in April, 2005. The measurement scale for the repayment status is: 
#  - -1 = pay duly; 
#  - 1 = payment delay for one month; 
#  - 2 = payment delay for two months; . . .; 
#  - 8 = payment delay for eight months; 
#  - 9 = payment delay for nine months and above. 
#  - -2 = indicates no consumption in the month, and a value of 
#  - 0 = indicates the use of revolving credit (equivalent to prepayment)
# - **X12-X17:** Amount of bill statement (NT dollar). 
# - **X12** = amount of bill statement in September, 2005; 
# - **X13** = amount of bill statement in August, 2005; . . .; 
# - **X17** = amount of bill statement in April, 2005. 
# - **X18-X23:** Amount of previous payment (NT dollar). 
# - **X18** = amount paid in September, 2005; 
# - **X19** = amount paid in August, 2005; . . .;
# - **X23** = amount paid in April, 2005. 

# ## The below Report is divided into 3 main sections namely:
# 
# ### 1. Initialize Libraries, Load Data & Preprocess
# ### 2. Exploratory Data Analysis and Visualization
# ### 3. Predictive Modeling

# - ### Goal of the study is to create a model that predicts if a client will default on credit card payment in next month.
# - ### This is a Supervised binary classification problem. Where Defaulter Yes(1) or No(0) is the dependant variable

# To find the predictability of a defaulter our main objective is to find what features can play a role to predict a credit card defaulter? Therefore we need to find answers to some questions like:
# - **1.** Is the % of defaulters significantly different between male & female ?
# - **2.** How does Marital Status effect the proportion of defaulters ?
# - **3.** Does the Level of Education play a role in the % of defaulters ?
# - **4.** Which age group constitutes for higher proportion of defaulters ?
# - **5.** Is the number of defaulters correlated with credit limit ?
# - **6.** Is there a pattern in past repayment statuses which can help predict probability of a defaulter ?
# - **7.** Does the history of credit card bill amount has a correlation with the % of defaulters ?

# # 1. Initialize Libraries, Load Data & Preprocess

# In[ ]:


# Import the required Packages
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from IPython.core.interactiveshell import InteractiveShell
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore') # to supress seaborn warnings
pd.options.display.max_columns = None # Remove pandas display column number limit
#InteractiveShell.ast_node_interactivity = "all" # Display all values of a jupyter notebook cell


# In[ ]:


# Read the data into DataFrames.
train = pd.read_csv("../input/credit_card_default_TRAIN.csv",index_col=0)
test = pd.read_csv("credit_card_default_TEST.csv",index_col=0)


# In[ ]:


# Breif look at the data
train.head()


# In[ ]:


test.head()


# - **Fix the Header of the data**

# In[ ]:


# Fix Header of the data, row 0 serves as more sensible header names
def fix_header(data):
    new_header = data.iloc[0]    # take the first row for the header
    data = data[1:]              # take the data without the header row
    data.columns = new_header    # set the header row as the df header
    data.rename(columns={'default payment next month':'DEFAULTER'}, inplace=True) # change column name
    return data

train = fix_header(train)
test = fix_header(test)


# In[ ]:


# look at the data with fixed header
train.head()


# In[ ]:


test.head()


# In[ ]:


# Check for Null values in the datasets
train.isnull().values.any(),test.isnull().values.any()


# - **DataSets do not have any Null values**

# - **Combine Train and Test Data set for further analysis & preprocessing**

# In[ ]:


# Combine Train and Test Data set for further analysis & preprocessing
train['Type'] = 'Train'
test['Type'] = 'Test'
fulldata = pd.concat([train,test],axis=0) 


# In[ ]:


#fulldata.shape


# In[ ]:


fulldata.describe()


# - **From Above Table we can deduce that some columns have extra values which might not be correct According to the Description given i.e.**
#  - EDUCATION has 7 unique values instead of 4
#  - MARRIAGE has 4 unique values instead of 3

# In[ ]:


fulldata.EDUCATION.value_counts()


# - **According to description we should have values 1,2,3,4 thus we will change 5,6,0 to 4 i.e. others**

# In[ ]:


fulldata.EDUCATION[fulldata.EDUCATION=='0']='4'
fulldata.EDUCATION[fulldata.EDUCATION=='5']='4'
fulldata.EDUCATION[fulldata.EDUCATION=='6']='4'
fulldata.EDUCATION.unique()


# In[ ]:


fulldata.MARRIAGE.value_counts()


# - **According to description we should have values 1,2,3 thus we will change 0 to 3 i.e. others**

# In[ ]:


fulldata.MARRIAGE[fulldata.MARRIAGE=='0']='3'
fulldata.MARRIAGE.unique()


# In[ ]:


fulldata.SEX.unique()


# In[ ]:


# Check for values less than 0
(fulldata.AGE[fulldata.AGE<0].count(),
fulldata.LIMIT_BAL[fulldata.LIMIT_BAL<0].count())


# In[ ]:


# Split back the combined data to train & test
train=fulldata[fulldata['Type']=='Train']
test=fulldata[fulldata['Type']=='Test']


# In[ ]:


# drop the non numeric column
train.drop(['Type'],axis = 1, inplace=True)
test.drop(['Type'],axis = 1, inplace=True)

# Change variables to type float
train = train.astype(float)
test = test.astype(float)


# # 2. Exploratory Data Analysis and Visualization

# In[ ]:


train.DEFAULTER.mean()*100


# - 22.61 % of people are defaulters in the Train data

# ### Visualize Data with t-SNE
# 
# t-SNE is a technique for dimensionality reduction that is well suited to visualise high-dimensional datasets. Lets have a first look on the map that will set some expectations for the prediction accuracy i.e. if our dataset has many overlaps it would be good if our model achieves an accuracy of 60-70%.!

# In[ ]:


#Set df4 equal to a set of a sample of 1000 deafault and 1000 non-default observations.
df2 = train[train.DEFAULTER == 0].sample(n = 1000)
df3 = train[train.DEFAULTER == 1].sample(n = 1000)
df4 = pd.concat([df2, df3], axis = 0)

#Scale features to improve the training ability of TSNE.
standard_scaler = StandardScaler()
df4_std = standard_scaler.fit_transform(df4)

#Set y equal to the target values.
y = df4.DEFAULTER

tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(df4_std)

#Build the scatter plot with the two types of transactions.
color_map = {0:'red', 1:'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x = x_test_2d[y==cl,0], y = x_test_2d[y==cl,1], c = color_map[idx], label = cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper right')
plt.title('t-SNE visualization of train data')
plt.show()


# **The plot reveals a rather mixed up dataset which means we should not expect very accurate model.**

# - **Now let us check the correlation between different features**

# In[ ]:


cor = train.corr()
plt.figure(figsize=(18,18))
sns.heatmap(cor, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
            xticklabels=cor.columns.values,
            yticklabels=cor.columns.values)


# Figure shows that 'BILL_AMTX' are highly correlated to each other, but very less correlation to target label 'DEFAULTER'. When data is huge to save computational resource, such features can be dropped without losing significant prediction power.
# 
# Payment statuses 'PAY' show highest contribution to the defaulter label.

# - **We can see above that PAY_0,PAY_2...have high positive correlation to DEFAULTER and LIMIT_BAL has pretty high negative correlation**

# ### Feature Engineering 
# 
# The regression coefficients are positive i.e. log-odds of defaulters increase as the ratio of  $\left(\frac{\text{bill amount} - \text{pay amount}}{\text{credit limit}}\right)$ increases. Hence we can add below 6 features.

# In[ ]:


train['BILL_PAY_RATIO1'] = (train['BILL_AMT1']-train['PAY_AMT1'])/train['LIMIT_BAL']
train['BILL_PAY_RATIO2'] = (train['BILL_AMT2']-train['PAY_AMT2'])/train['LIMIT_BAL']
train['BILL_PAY_RATIO3'] = (train['BILL_AMT3']-train['PAY_AMT3'])/train['LIMIT_BAL']
train['BILL_PAY_RATIO4'] = (train['BILL_AMT4']-train['PAY_AMT4'])/train['LIMIT_BAL']
train['BILL_PAY_RATIO5'] = (train['BILL_AMT5']-train['PAY_AMT5'])/train['LIMIT_BAL']
train['BILL_PAY_RATIO6'] = (train['BILL_AMT6']-train['PAY_AMT6'])/train['LIMIT_BAL']

test['BILL_PAY_RATIO1'] = (test['BILL_AMT1']-test['PAY_AMT1'])/test['LIMIT_BAL']
test['BILL_PAY_RATIO2'] = (test['BILL_AMT2']-test['PAY_AMT2'])/test['LIMIT_BAL']
test['BILL_PAY_RATIO3'] = (test['BILL_AMT3']-test['PAY_AMT3'])/test['LIMIT_BAL']
test['BILL_PAY_RATIO4'] = (test['BILL_AMT4']-test['PAY_AMT4'])/test['LIMIT_BAL']
test['BILL_PAY_RATIO5'] = (test['BILL_AMT5']-test['PAY_AMT5'])/test['LIMIT_BAL']
test['BILL_PAY_RATIO6'] = (test['BILL_AMT6']-test['PAY_AMT6'])/test['LIMIT_BAL']


# In[ ]:


train.head()


# In[ ]:


# function to make a bar plot
def plot0(col1, col2, tittle, xticks, train):
    dt = train.groupby(col1).agg([np.mean])*100.0
    dt = dt[col2].reset_index()
    f, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=col1, y="mean", data=dt)
    ax.set(xlabel="", ylabel="Defaulter %")
    ax.set_title(label=tittle, fontsize=15)
    ax.set_xticklabels(xticks, fontsize=11)


# ## 1. Is the % of defaulters significantly different between male & female ?
# - _Below we plot the % of Defaulters by Gender._**Apparently we see that males are slightly more likely to default.**
# 
# Observations:
# - Approximately 24.2% of the males defaulted.
# - Approximately 20.8% of the females defaulted.

# In[ ]:


#Crosstab
sex_crosstab = pd.crosstab(train['DEFAULTER'], train['SEX'], margins=True, normalize=False)
new_index = {0: 'Non-default', 1: 'Default', }
new_columns = {1 : 'Male', 2 : 'Female'}
sex_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
sex_crosstab/sex_crosstab.loc['All']


# In[ ]:


#Bar Chart
col1 = "SEX"
col2 = "DEFAULTER"
tittle = "% of Defaulters by Sex"
xticks = ["Male", "Female"]
plot0(col1, col2, tittle, xticks, train)


# ## 2. How does Marital Status effect the proportion of defaulters ?
# - _Below we plot the % of Defaulters by Marital Status._**We see that in the dataset Married people are slightly more likely to default.**
# 
# Observations:
# - Approximately 24.2% of the Married people defaulted.
# - Approximately 21.2% of the Single people defaulted.

# In[ ]:


#Crosstab
marital_crosstab = pd.crosstab(train['DEFAULTER'], train['MARRIAGE'], margins=True, normalize=False)
new_index = {0: 'Non-default', 1: 'Default', }
new_columns = {1 : 'Married', 2 : 'Single', 3:'Others'}
marital_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
marital_crosstab/marital_crosstab.loc['All']


# In[ ]:


#Bar Chart
col1 = "MARRIAGE"
col2 = "DEFAULTER"
tittle = "% of Defaulters by Marital Status"
xticks = ["Married", "Single", "Other"]
plot0(col1, col2, tittle, xticks, train)


# ## 3. Does the Level of Education play a role in the % of defaulters ?**
# - _Below we plot the % of Defaulters by Education._**We can see that higher the education less likely is the person to default.**
# 
# Observations:
# - Approximately 25.8% of defaulters studied upto High School.
# - Approximately 23.7% of defaulters studied upto University.
# - Approximately 19.7% of defaulters studied upto Graduate School.

# In[ ]:


#Crosstab
education_crosstab = pd.crosstab(train['DEFAULTER'], train['EDUCATION'], margins=True, normalize=False)
new_index = {0: 'Non-default', 1: 'Default', }
new_columns = {1 : 'Graduate school', 2 : 'University', 3 : 'High school', 4 : 'Others'}
education_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
education_crosstab/education_crosstab.loc['All']


# In[ ]:


#Bar Chart
col1 = "EDUCATION"
col2 = "DEFAULTER"
tittle = "% of Defaulters by Education Level"
xticks = ["Graduate", "University", "High school", "Others"]
plot0(col1, col2, tittle, xticks, train)


# ## 4. Which age group constitutes for higher proportion of defaulters ?
# - _Below we can see the Defaulters distribution by Age_ **Majority of defaulters fall in the age group of 25 to 35**
# 
# 
# Observations:
# - Defaulters seems to increase from the early 20s to the early 30s.
# - Defaulters seems to decrease from the early 40s onward.

# In[ ]:


defaulters = train[train["DEFAULTER"] == 1]
non_defaulters = train[train["DEFAULTER"] == 0]
defaulters["Defaulter"] = defaulters["AGE"]
non_defaulters["Non Defaulter"] = non_defaulters["AGE"]
f, ax = plt.subplots(figsize=(12, 6))
ax = sns.kdeplot(defaulters["Defaulter"], shade=True, color="r")
ax = sns.kdeplot(non_defaulters["Non Defaulter"], shade=True, color="g")


# ## 5. Is the number of defaulters correlated with credit limit ?
# - _Below we can see the Defaulters distribution by Credit Limit_ **we can see that people with lower credit balance tend to default more**

# In[ ]:


defaulters = train[train["DEFAULTER"] == 1]
non_defaulters = train[train["DEFAULTER"] == 0]
defaulters["Defaulter"] = defaulters["LIMIT_BAL"]
non_defaulters["Non Defaulter"] = non_defaulters["LIMIT_BAL"]
f, ax = plt.subplots(figsize=(12, 6))
ax = sns.kdeplot(defaulters["Defaulter"], shade=True, color="r")
ax = sns.kdeplot(non_defaulters["Non Defaulter"], shade=True, color="b")


# In[ ]:


# function to make a scatter plot
def plot1(label_list, label_dict, data, col,tittle,xlabel,ylabel, ticks):
  df = {}
  for i in label_list:
      df[i] = data.groupby([i, col]).size().unstack()
      df[i] = df[i][df[i].sum(axis=1)>25]
      df[i] = df[i].div(df[i].sum(axis=1), axis='index') # Calculate proportions
      df[i].sort_index(ascending=False, inplace=True)
          
  sns.set_palette(sns.light_palette("red", reverse=True))   # plot
  fig, ax = plt.subplots(1, 1, figsize=(6,4))

  for i in label_list:
      ax.scatter(x=df[i].index, y=df[i][1], label=label_dict.get(i), s=100, edgecolor='k', lw=1)          

  ax.set_ylim([0, 1])
  plt.xticks(ticks, rotation=0)
  ax.xaxis.set_ticks(ticks=ticks, minor=False)
  ax.grid(b=True, which='major', color='0.4', linestyle='--')
  lgd = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
  for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
  for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
  for spine in ax.spines.values():
      spine.set_edgecolor('k')
  sns.set_palette(sns.light_palette("green", reverse=True))
  plt.title(tittle, fontsize=17, y = 1.05) 
  plt.ylabel(xlabel, fontsize=14)
  plt.xlabel(ylabel, fontsize=14)
  plt.show()


# ## 6. Is there a pattern in past repayment statuses which can help predict probability of a defaulter ?
# 
# Observations:
# - The proportion of defaulters in delinquency bucket 2 or more i.e. with payment delay for 2 or more months are much higher.

# In[ ]:


label_list =['PAY_0',  'PAY_2',  'PAY_3',  'PAY_4',  'PAY_5',  'PAY_6']
label_dict ={'PAY_0': 'PAY_0 - Sep, 2005', 
             'PAY_2': 'PAY_2 - Aug, 2005', 
             'PAY_3': 'PAY_3 - Jul, 2005', 
             'PAY_4': 'PAY_4 - Jun, 2005',  
             'PAY_5': 'PAY_5 - May, 2005',  
             'PAY_6': 'PAY_6 - Apr, 2005'}

col = 'DEFAULTER'
tittle = 'Proportion of Defaulters Versus Repayment Status'
xlabel = 'Proportion of Defaulters'
ylabel = 'Repayment Status'
ticks = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,9]
plot1(label_list, label_dict, train, col,tittle,xlabel,ylabel,ticks)


# ## 7. Does the history of credit card bill amount has a correlation with the % of defaulters ?
# 
# 
# Observations:
# - The proportion of defaulters is positively correlated with bill amount in recent months.

# In[ ]:


label_list =['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
label_dict ={'BILL_AMT1': 'BILL_AMT1 - Sep, 2005',  
             'BILL_AMT2': 'BILL_AMT2 - Aug, 2005',
             'BILL_AMT3': 'BILL_AMT3 - Jul, 2005', 
             'BILL_AMT4': 'BILL_AMT4 - Jun, 2005',  
             'BILL_AMT5': 'BILL_AMT5 - May, 2005', 
             'BILL_AMT6': 'BILL_AMT6 - Apr, 2005'}

col = 'DEFAULTER'
tittle = 'Proportion of Defaulters Versus Bill Amount'
xlabel = 'Proportion of Defaulters'
ylabel = 'Bill Amount'
ticks = []
for i in range(0, 3000, 500):
    ticks.append(round(i,1))
plot1(label_list, label_dict, train, col,tittle,xlabel,ylabel,ticks)


# # 3. Predictive Modeling

# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import preprocessing, metrics
from xgboost import XGBClassifier
warnings.filterwarnings('ignore') # to supress warnings


# In[ ]:


x = train.drop(['DEFAULTER'],axis = 1)
y = train.DEFAULTER

# rescale the metrics to the same mean and standard deviation
scaler = preprocessing.StandardScaler()
x = scaler.fit(x).transform(x)

# Further divide the train data into train test split 70% & 30% respectively
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)


# **Here we are first trying out below listed classification models to get the first look at accuracy**

# In[ ]:


# list of different classifiers we are going to test
clfs = {
'LogisticRegression' : LogisticRegression(),
'GaussianNB': GaussianNB(),
'RandomForest': RandomForestClassifier(),
'DecisionTreeClassifier': DecisionTreeClassifier(),
'SVM': SVC(),
'KNeighborsClassifier': KNeighborsClassifier(),
'GradientBoosting': GradientBoostingClassifier(),
'XGBClassifier': XGBClassifier()
}


# In[ ]:


# code block to test all models in clfs and generate a report
models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])

for clf, clf_name in zip(clfs.values(), clfs.keys()):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.score(x_test,y_test)
    
    #print('Calculating {}'.format(clf_name))
    t = pd.Series({ 
                     'Model': clf_name,
                     'Precision_score': metrics.precision_score(y_test, y_pred),
                     'Recall_score': metrics.recall_score(y_test, y_pred),
                     'F1_score': metrics.f1_score(y_test, y_pred),
                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}
                   )

    models_report = models_report.append(t, ignore_index = True)

models_report


# ### From above report we can see that highest accuracy is given by XGboost followed by GradientBoosting, let us compare both.

# In[ ]:


# Function to optimize model using gridsearch 
def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):
    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)
    gs.fit(x_train, y_train)
    print 'Best params: ', gs.best_params_
    print 'Best AUC on Train set: ', gs.best_score_
    print 'Best AUC on Test set: ', gs.score(x_test, y_test)

# Function to generate confusion matrix
def confmat(pred, y_test):
    conmat = np.array(confusion_matrix(y_test, pred, labels=[1,0]))
    conf = pd.DataFrame(conmat, index=['Defaulter', 'Not Defaulter'],
                             columns=['Predicted Defaulter', 'Predicted Not Defaulter'])
    print conf

# Function to plot roc curve
def roc(prob, y_test):
    y_score = prob
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])
    plt.figure(figsize=[7,7])
    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)
    plt.plot([1,0], [1,0], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive rate', fontsize=15)
    plt.ylabel('True Positive rate', fontsize=15)
    plt.title('ROC curve for Credit Default', fontsize=16)
    plt.legend(loc='Lower Right')
    plt.show()
    
def model(md, x_train, y_train,x_test, y_test):
    md.fit(x_train, y_train)
    pred = md.predict(x_test)
    #prob = md.predict_proba(x_test)[:,1]
    print ' ' 
    print 'Accuracy on Train set: ', md.score(x_train, y_train)
    print 'Accuracy on Test set: ', md.score(x_test, y_test)
    print ' '
    print(classification_report(y_test, pred))
    print ' '
    print 'Confusion Matrix'
    confmat(pred, y_test)
    #roc(prob, y_test)
    return md


# ### Parameter tuning
# 
# There are a few parameters that require tuning to improve the performance. I use GridSearchCV method to test model through a series of parameter values.

# ### GradientBoosting

# In[ ]:


# Use gridsearch to fine tune the parameters
gb = GradientBoostingClassifier()
gb_params = {'n_estimators': [100,200,300],'learning_rate' : [0.01, 0.02, 0.05, 0.1]}
gridsearch(gb, gb_params,x_train, x_test, y_train, y_test,5)


# In[ ]:


# feature selection with the best model from grid search
gb = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 7,n_estimators=300, max_features = 0.9,min_samples_leaf = 5)
model_gb = model(gb, x_train, y_train,x_test, y_test)


# ### XGboost

# In[ ]:


# Use gridsearch to fine tune the parameters
xgb = XGBClassifier()
xgb_params = {'n_estimators':[200,300],'learning_rate':[0.05,0.02], 'max_depth':[4],'min_child_weight':[0],'gamma':[0]}
gridsearch(xgb, xgb_params,x_train, x_test, y_train, y_test,5)


# In[ ]:


# feature selection with the best model from grid search
xgb = XGBClassifier(
 learning_rate =0.05,
 n_estimators=200,
 max_depth=4,
 min_child_weight=0,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=1,
 scale_pos_weight=1,
 seed=27)
model_xgb = model(xgb, x_train, y_train,x_test, y_test)


# The classification metrics of iterest for this fairly imbalanced dataset are: 
# - precision = tp / (tp + fp)
# - recall = tp / (tp + fn)
# - f1 = 2(precision)(recall) / (precision + recall)
# - Roc curve area
# 
# Depending upon banks operational costs & ideology a large bank may follow the principal that fewer False Positives are preferable over a few more False Negatives to be able to lend more & spend less on investigations on the contrary a conservative approach would go with the opposite i.e more accuracy.
# 
# ### Therefore we see that XGBoost trains with little higher accuracy and auc score than GradientBoost. We will use XGBoost for final predictions. i.e. fewer False Positives are preferable over a few more False Negatives

# Save the output to csv file in desired format

# In[ ]:


#Predict final values on Test data set
test['PREDICTED_STATUS']=np.int_(model_gb.predict(test.drop(['DEFAULTER'],axis = 1)))
test.index.names = ['ID']


# In[ ]:


#test.head()


# In[ ]:


test['PREDICTED_STATUS'].to_csv("credit_card_default_TRAIN_Predict.csv")


# In[ ]:


test.DEFAULTER.mean()*100


# In[ ]:




