#!/usr/bin/env python
# coding: utf-8

# # HR Analytics - Classifying Employees Attrition
# ## goals:
# The main goal of this project is to get some useful information and insights about what contributes to employees feeling burnout, fatigue and attrition.
# Employees attrition has a great amount of consequences to the company, and having the ability to know beforehand which employee is more proned to leave, can help mitigating some of those negative effects.
# The area of Human Resource Analysis has a big weight in the field of labour market, thus expanding the knowledge about it is important.
# 
# 
# ## data sources:
# the data we're about to analyze is taken from Kaggle: https://www.kaggle.com/vjchoudhary7/hr-analytics-case-study and contains a number of indicators about each Employee(N=4410), as well as his attrition status('Yes' or No).
# The data contained 30 features after cleaning was done.

# In[ ]:


from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error


# ### working stages:
# 
# #### 1) load the data
# 
# #### 2) clean the data
# 
# #### 3) explore the data
# 
# #### 4) handle null values
# 
# #### 5) feature engeneering
# 
# #### 6) model comparison
# 
# #### 7) model selection
# 
# #### 8) tuning the model
# 
# #### note: part 2-3-4 may often come in a different order, depends on the data.

# In[ ]:


import os

emp_data = pd.read_csv("../input/hr-analytics-case-study/employee_survey_data.csv", index_col='EmployeeID')
gen_data = pd.read_csv("../input/hr-analytics-case-study/general_data.csv",index_col='EmployeeID')
manager_data = pd.read_csv("../input/hr-analytics-case-study/manager_survey_data.csv",index_col='EmployeeID')
in_time_data = pd.read_csv("../input/hr-analytics-case-study/in_time.csv")
out_time_data = pd.read_csv("../input/hr-analytics-case-study/out_time.csv")


# In[ ]:


in_time_data.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
in_time_data.set_index('EmployeeID', inplace=True)
in_time_data
out_time_data.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
out_time_data.set_index('EmployeeID', inplace=True)
out_time_data.head()


# In[ ]:


in_time_data = in_time_data.apply(pd.DatetimeIndex)
out_time_data = out_time_data.apply(pd.DatetimeIndex)


# In[ ]:


times = pd.concat([in_time_data, out_time_data], axis=1)


# In[ ]:


times.head()


# In[ ]:


times = times.applymap(lambda x: x.hour+0.01*x.minute)
times['avg_in'] = round(times.iloc[:, :261].mean(axis=1),2)
times['avg_out'] = round(times.iloc[:, 261:].mean(axis=1),2)
times['med_in'] = round(times.iloc[:, :261].median(axis=1),2)
times['med_out'] = round(times.iloc[:, 261:].median(axis=1),2)


# In[ ]:


times.head()


# In[ ]:


times.shape


# In[ ]:


fig, axs = plt.subplots(1,3, figsize = (12,4))
sns.distplot(times.iloc[4, :261], ax=axs[0]).set(xlabel = 'In time', ylabel = 'Frequency',xlim=(7,12))
sns.distplot(times.iloc[72, :261], ax=axs[1]).set(xlabel = 'In time', ylabel = 'Frequency',xlim=(7,12))
sns.distplot(times.iloc[102, :261], ax=axs[2]).set(xlabel = 'In time', ylabel = 'Frequency',xlim=(7,12))
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(10,4))
g = plt.plot(times.iloc[4, :261])


# In[ ]:


plt.figure(figsize=(10,4))
g = plt.plot(times.iloc[25, :261])


# In[ ]:


times['total'] = times['med_out'] - times['med_in']
time_feats = times[['avg_in', 'avg_out', 'med_in','med_out','total']]


# In[ ]:


time_feats.head()


# In[ ]:


in_time_data.isna().sum()


# In[ ]:


emp_data.head()


# In[ ]:


manager_data.isna().sum()


# In[ ]:


emp_data.loc[emp_data['EnvironmentSatisfaction'].isnull()]


# - there's some null values, let's explore the data a little bit to see how can can handle it properly</li>
# 

# In[ ]:


fig, axs = plt.subplots(1,3, figsize=(10,4))
sns.barplot(emp_data['EnvironmentSatisfaction'], emp_data['JobSatisfaction'],ax=axs[0])
sns.barplot(emp_data['WorkLifeBalance'], emp_data['JobSatisfaction'],ax=axs[1])
sns.barplot(emp_data['WorkLifeBalance'], emp_data['EnvironmentSatisfaction'],ax=axs[2])
plt.tight_layout(pad=3)


# In[ ]:


g = sns.FacetGrid(emp_data, col='WorkLifeBalance',size=2.4, aspect=2, col_wrap=2 )
g = g.map(sns.distplot, 'JobSatisfaction', )


# In[ ]:


g = sns.FacetGrid(emp_data, col='JobSatisfaction',row ='WorkLifeBalance', size=2.4, aspect=2 )
g = g.map(sns.distplot, 'EnvironmentSatisfaction' )


# #### No clear connection between the variables, but in order to be on the safe side, we'll handle the nulls by the conditional mode based on the other two columns

# In[ ]:


def set_mode(data, col, col2, col3):
    index_nan = list(data[col][data[col].isnull()].index)
    for i in index_nan:
        cols_mode = data[col].mode()[0]
        mode_fill = data[col][((data[col2] == data.loc[i][col2]) & (data[col3] == data.loc[i][col3]))].mode()[0]
        data[col].loc[i] = mode_fill


    
                    


# In[ ]:


set_mode(emp_data, 'EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance')


# In[ ]:


emp_data.isnull().sum()


# In[ ]:


set_mode(emp_data, 'JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance')


# In[ ]:


emp_data.isnull().sum()


# In[ ]:


set_mode(emp_data, 'WorkLifeBalance','JobSatisfaction','EnvironmentSatisfaction')


# In[ ]:


emp_data.isnull().sum()


# In[ ]:


gen_data.head()


# ### Alright, we can now merge the dataframes and explore the data as a whole

# In[ ]:


df = pd.concat([gen_data,manager_data,emp_data,time_feats], axis=1)


# In[ ]:


for col in df.columns.values:
    if df[col].nunique() == 1:
        df.drop(col, axis=1, inplace=True)


# In[ ]:


df['Attrition'] = np.where(df['Attrition']=='Yes',1,0)


# In[ ]:


df.NumCompaniesWorked.fillna(0, inplace=True)
df.TotalWorkingYears.fillna(0, inplace=True)


# In[ ]:


df.isna().sum()
df.head()


# In[ ]:


g = sns.distplot(df['total']).set(xlabel = 'Total Hours Of Work', ylabel = 'Frequency')


# In[ ]:


## checking the exact number of people by hours of work
print(df['total'][df['total'] <=7].value_counts().sum())
print(df['total'][(df['total'] > 7) & (df['total'] <=8)].value_counts().sum())
print(df['total'][df['total'] > 8].value_counts().sum())


# In[ ]:


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sns.distplot(df['avg_in'], bins=20)
plt.subplot(1,2,2)
s = sns.distplot(df['avg_out'], bins=20)
plt.xticks((range(16,22)))
plt.tight_layout(pad=5)


# In[ ]:


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sns.distplot(df['med_in'], bins=10)
plt.subplot(1,2,2)
s = sns.distplot(df['med_out'], bins=20)
plt.xticks((range(16,22)))
plt.tight_layout(pad=5)


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot, 'total')


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot, 'med_in')


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'med_out')


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'avg_in')


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition' )
g = g.map(sns.distplot , 'avg_out')


# ### both the median and the mean in/out time, as well as the total time, show us that people who work more, but pushing the working hours till late, are more likely to attrit

# In[ ]:


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
g = sns.barplot(df['Attrition'], df['EnvironmentSatisfaction'], hue=df['JobSatisfaction'])
plt.legend(loc ='upper right')
plt.subplot(1,3,2)
s = sns.barplot(df['Attrition'], df['JobInvolvement'], hue=df['JobSatisfaction'])
plt.legend(loc ='upper right')
plt.subplot(1,3,3)
f = sns.barplot(df['Attrition'], df['WorkLifeBalance'], hue=df['JobSatisfaction'])
plt.legend(loc ='upper right')
plt.tight_layout()


# In[ ]:


cat_cols = ['BusinessTravel', 'Department','EducationField', 'Gender',
       'JobRole', 'MaritalStatus','JobInvolvement', 'PerformanceRating',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
plt.figure(figsize=(15,18))
for i in range(len(cat_cols)):
    plt.subplot(4,3,i+1)
    sns.countplot(df[cat_cols[i]], hue=df['Attrition'])
    if len(df[cat_cols[i]].unique()) >= 3:
        plt.xticks(rotation=75)
plt.tight_layout()


# In[ ]:


def Att_ratio(data, col):
    col_values = data[col].unique()
    print('For',col, ':')
    for index, item in enumerate(col_values):
        ratio = len(df.loc[(df[col] == col_values[index]) & (df['Attrition'] == 1)])/len(df.loc[(df[col] == col_values[index]) & (df['Attrition'] == 0)])
        print('The Attrition ratio(Yes/No) under the category %s is %f' %(item, ratio))
    print('-----------------------------------------------------------------------------------------------')

        


# In[ ]:


cat_cols = ['BusinessTravel', 'Department','EducationField', 'Gender',
       'JobRole', 'MaritalStatus','JobInvolvement', 'PerformanceRating',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
for col in cat_cols:
    Att_ratio(df, col)


# In[ ]:


df['Gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
df['BusinessTravel'].replace({'Travel_Rarely': 1,'Travel_Frequently':2,'Non-Travel':0 }, inplace=True)
cat_cols = ['Gender','BusinessTravel',
       'JobInvolvement', 'PerformanceRating',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
for col in cat_cols:
    print('The Attrition ration(Yes/All) For',col+':')
    print(df.groupby([col]).Attrition.agg(['mean']))
    print('----------------------------------------------------------')


# # summary of the categorical variables:
# * BusinessTravel seems to have a bit of an influence on attrition.those who travel frequently have an attrition ratio of 0.331731 - Probably those who travel a lot for business purposes are having more stress and thus willing to leave
# 
# 
# * Department doesn't seem to be correlated strongly but Human Resources attrition ratio is quite high at 0.431818 - maybe that's because of the low salary, we'll check this one later 
# 
# 
# * EducationField seems to be a related to Department in some way, for Human Resources the ratio is 0.687500
# 
# 
# * Gender doesn't seem correlated - Male attrition ratio is 0.200000 - not significantly more than females
# 
# 
# * JobRole - the attrition ratio for Research Director is 0.311475 - that's not surprising considering the fact that the largest department is Research and Development. Research director is a role with a lot of responsibility,and that definetely can contribute to the the overall fatigue and burnout.
# 
# 
# * MaritalStatus - Single status has a slightly bigger ratio with 0.342857 - maybe mediated by low salary/short term job
# 
# 
# * JobInvolvement - category 1 ratio is 0.276923 - not surprising but isn't much higher than the other categories
# 
# 
# * PerformanceRating - category 4 ratio 0.221622 - probably those who work the hardest, but again no significantly higher
# 
# 
# * EnvironmentSatisfaction - here the difference is quite big, the category 1.0 attrition ratio is 0.339117 - it's pretty normal in my opinion. those who aren't satisfied with their work are not going to stay for long
# 
# 
# * JobSatisfaction - category 1.0 ratio is 0.297134 - same explanation but here it's not significantly higher in my opinion
# 
# 
# * WorkLifeBalance - here, simillarly to enviroment satisfaction, the gap is quite big, for category 1.0 the ratio is 0.457317 - the reasonable explanation is that people with low work/life balance are strongly prone to burnout, have more stress, etc..
# 
# 
# * JobLevel - category 2 ratio 0.216401 - very simillar to other categories therefore doesn;t have a strong impact.

# ## Let's move on to the numerical variables:
# ### We'll start with income, then the Age and the seniority
# ### We'll check if there's any correlation between those variables and the prone to leave

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


sns.distplot(df.MonthlyIncome)


# #### got some outliers in terms of income. 
# ### lets check the income correlations with other variables

# In[ ]:


g = sns.FacetGrid(df,col = 'MaritalStatus', row = 'Department')
g = g.map(sns.distplot , 'MonthlyIncome')
g.fig.subplots_adjust(top=1,right=1.4, wspace=1)


# #### * seems like, as i hypothesised earlier, that single workers are paid the least.
# 
# #### * we can also notice that divorced people are paid the most, regardless of department.
# 
# #### * doesn't seem like there is a clear connection between department and salary

# In[ ]:


## we can see it here as well, it's more useful to look at the median, as the distribution of income is skewed to the right
print(df.groupby(['MaritalStatus']).MonthlyIncome.agg(['mean','median']))
print(df.groupby(['Department']).MonthlyIncome.agg(['mean','median']))


# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(14,8))
top_corr = df.corr().nlargest(15, 'MonthlyIncome').index
cm = np.corrcoef(df[top_corr].values.T)
g = sns.heatmap(cm, cbar=True, annot=True, cmap='BrBG',yticklabels = top_corr.values, xticklabels=top_corr.values)


# #### No real connection between salary and any other numerical value
# ### let's have a look at the connection between attrition and salary

# In[ ]:


g = sns.distplot(df['MonthlyIncome'][df['Attrition'] == 0],color='blue')
f = sns.distplot(df['MonthlyIncome'][df['Attrition'] == 1],color='orange')
df.groupby('Attrition').MonthlyIncome.agg(['mean','median'])


# #### from looking at the graph and the table there's seem to be no correlation whatsoever between salary and attrition

# ### let's take a look at the age, the years from last promotion and years at the company

# In[ ]:


plt.figure(figsize=(14,8))
top_corr = df.corr().nlargest(15, 'Age').index
cm = np.corrcoef(df[top_corr].values.T)
g = sns.heatmap(cm, cbar=True, annot=True, cmap='BrBG',yticklabels = top_corr.values, xticklabels=top_corr.values)


# In[ ]:


fig, axs = plt.subplots(1,3, figsize = (12,4))
sns.distplot(df['Age'], ax=axs[0])
sns.distplot(df['YearsSinceLastPromotion'], ax=axs[1])
sns.distplot(df['YearsAtCompany'], ax=axs[2])
plt.tight_layout()


# ### quite normal distributed ages.
# 
# ### skewd to the right distribution of total years at the company
# 
# ### skewd to the right distribution of years since last promotion

# In[ ]:


sns.distplot(df['Age'][df['Attrition'] == 0],color='blue')
sns.distplot(df['Age'][df['Attrition'] == 1],color='orange')
plt.legend(['No','Yes'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'Age')
g.fig.subplots_adjust(top=1,right=1.2)
#plt.tight_layout()
df.groupby('Attrition').Age.agg(['median','mean'])


# seems like a tendency to attrit among younger people,that's interesting. mayber thats because they work longer hours. let's check that

# In[ ]:


sns.scatterplot(df['Age'], df['total'])


# ### doesn't seem like that's the case. probably related to other things like more stress, need to prove yourself etc...

# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'YearsSinceLastPromotion')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').YearsSinceLastPromotion.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'YearsWithCurrManager')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').YearsWithCurrManager.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'YearsAtCompany')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').YearsAtCompany.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'DistanceFromHome')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').DistanceFromHome.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'TotalWorkingYears')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').TotalWorkingYears.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'Education')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').Education.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'PercentSalaryHike')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').PercentSalaryHike.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'NumCompaniesWorked')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').NumCompaniesWorked.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'YearsSinceLastPromotion')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').YearsSinceLastPromotion.agg(['median','mean'])


# In[ ]:


g = sns.FacetGrid(df,col = 'Attrition')
g = g.map(sns.distplot , 'StockOptionLevel')
g.fig.subplots_adjust(top=1,right=1.2)
df.groupby('Attrition').StockOptionLevel.agg(['median','mean'])


# ## Summary of the numerical variables:
# 
# * Looks like seniority plays somewhat of a role in the attrition attribute, along with total working years and years at the company, though there is a strong correlation beteen age and total working years, as well as total working years and years at the company, therefore there's a risk for multicolinearity. we'll handle it when we get to the model building.
# 
# 
# * No observed impact among all the other variables on attrition.

# # Modeling:
# ### 0) A little more feature engineering like creating age groups
# ### 1) seperating x and y
# ### 2) convert strings to dummy variables
# ### 3) scale x features
# ### 4) compare scores of different models and select the two best
# ### 5) perform search grid on the two best models and see if there's an improvement

# Lets first drop some unimportant features and variables with multicolinearity such as med_in avg_in and total etc

# In[ ]:


x = df.drop(['Attrition'], axis=1).reset_index(drop=True)
y = df['Attrition'].values


# In[ ]:


cols_todrop = ['JobLevel','Department','JobRole','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel','YearsWithCurrManager','med_in', 'avg_in','avg_out']


# In[ ]:


x.drop(cols_todrop, axis=1, inplace=True)


# In[ ]:


## creating age groups
x.Age = pd.cut(x.Age, 4)


# In[ ]:


x.Age.unique()


# In[ ]:


## converting categorial variables to dummies
x = pd.get_dummies(x)


# In[ ]:


x_copy = x.copy()


# In[ ]:


## scaling the features

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)


# In[ ]:


x_copy.head()


# In[ ]:


## splitting the sets into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=42)


# ## Alright we are ready to start with the predictions!
# 
# ### As I've mentionted above, our working process is :
# * Compare
# 
# 
# * Select
# 
# 
# * Improve
# 
# 
# * Check the contribution of the features

# # Model Comparison

# In[ ]:


# Defining a function which examines each model based on the score, then show each one's score and STD, as well as graphic comparison
# evaluate each model in turn
def get_scores(score1, score2):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('GradientBooster', GradientBoostingClassifier()))
    models.append(('ExtraTrees', ExtraTreesClassifier()))
    models.append(('RandomForest', RandomForestClassifier()))
    cv_scores = []
    test_scores = []
    names = []
    stds = []
    differences = []
    #res = pd.DataFrame(columns = {'Model',score+('(train)'), 'Std', score+('(test_score)'), 'difference'})
    #res = res[['Model',score+('(train)'), 'Std', score+('(test_score)'), 'difference']]
    res = pd.DataFrame()
    for index, model in enumerate(models):
        kfold = StratifiedKFold(n_splits=7)
        cv_results = cross_val_score(model[1], x_train, y_train, cv=kfold, scoring=score1)
        cv_scores.append(cv_results)
        names.append(model[0])
        model[1].fit(x_train,y_train)
        predictions = model[1].predict(x_test)
        test_score = score2(predictions, y_test)
        test_scores.append(test_score)
        stds.append(cv_results.std())
        differences.append((cv_results.mean() - test_score))
        res.loc[index,'Model'] = model[0]
        res.loc[index,score1+('(train)')] = cv_results.mean()
        res.loc[index,score1+('(test_score)')] = test_score
        res.loc[index,'Std'] = cv_results.std()
        res.loc[index,'difference'] = cv_results.mean() - test_score
    # boxplot algorithm comparison
    fig = plt.figure(figsize = (12,5))
    fig.suptitle('Model Comparison')
    ax = fig.add_subplot(121)
    plt.boxplot(cv_scores)
    ax.set_xticklabels(names, rotation=70)
    axs = fig.add_subplot(122)
    sns.barplot(names,test_scores)
    axs.set_xticklabels(names, rotation=70)
    plt.tight_layout(pad=5)
    return res
    plt.show()

    


# In[ ]:


get_scores('accuracy', accuracy_score)


# # Model Selection And Tuning
# ### seems like our models has a strong predicting power, especially the random forest and extra tree booster. let's check if theres any way  to improve them with random search cv

# In[ ]:


params = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000]}
RandomForest = RandomForestClassifier()
randomgrid_forest = RandomizedSearchCV(estimator=RandomForest, param_distributions = params, 
                               cv=5, n_iter=25, scoring = 'accuracy',
                               n_jobs = 4, verbose = 3, random_state = 42,
                               return_train_score = True)
randomgrid_forest.fit(x_train,y_train)


# In[ ]:


randomgrid_forest.score(x_test, y_test)


# In[ ]:


forest_preds = randomgrid_forest.predict(x_test)
roc_auc_score(forest_preds, y_test)


# In[ ]:


randomgrid_forest.best_estimator_


# #### A little bit of improvement indeed! let's try tuning it a little bit more

# In[ ]:


### I created a function which take a model and scoring method, then shows the cross validation score for each estimator
### and plot it next to the test score.
def estimators_compare(model, cv_score, metrics_score):
    train_scores = []
    test_scores= []
    estimators = [80,100,200,400,600,800,1200]
    res = pd.DataFrame(columns = {'Number Of Estimators', 'train_score', 'test_score'})
    for ind, i in enumerate(estimators):
        mode = model(n_estimators=i)
        kfold = StratifiedKFold(n_splits=7)
        cv_results = cross_val_score(mode, x_train, y_train, cv=kfold, scoring=cv_score)
        mode.fit(x_train, y_train)
        predictions = mode.predict(x_test)
        train_score = cv_results.mean()
        train_scores.append(train_score)
        test_score = metrics_score(predictions, y_test)
        test_scores.append(test_score)
        res.loc[ind,'Number Of Estimators'] = i
        res.loc[ind,'train_score'] = train_score
        res.loc[ind,'test_score'] = test_score

    plt.plot(estimators, train_scores, color='red')
    plt.plot(estimators, test_scores, color='blue')
    legs = ['train', 'test']
    plt.legend(legs)
    return res


# In[ ]:


estimators_compare(RandomForestClassifier, 'accuracy', accuracy_score)


# ### let's compare 100 estimators with 600 like the grid search provided

# In[ ]:


final_random_forest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=40, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
final_random_forest.fit(x_train, y_train)
final_random_forest.score(x_test, y_test)


# ### Well, Looks like our randomized grid search produced simmilar results. before we move on to the extra trees, let's have a look at the contribution of each feature to our prediction

# In[ ]:


featuers_coefficients = final_random_forest.feature_importances_.tolist()
feature_names = x_copy.columns
feats = pd.DataFrame(pd.Series(featuers_coefficients, feature_names).sort_values(ascending=False),columns=['Coefficient'])
feats


# ## same process for the Extra Trees model

# In[ ]:


params2 = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000]}
ExtraTress = ExtraTreesClassifier()
randomgrid_extrees = RandomizedSearchCV(estimator=ExtraTress, param_distributions = params2, 
                               cv=5, n_iter=25, scoring = 'accuracy',
                               n_jobs = 4, verbose = 3, random_state = 42,
                               return_train_score = True)
randomgrid_extrees.fit(x_train,y_train)


# In[ ]:


randomgrid_extrees.score(x_test, y_test)


# In[ ]:


randomgrid_extrees.best_estimator_


# In[ ]:


estimators_compare(ExtraTreesClassifier, 'accuracy', accuracy_score)


# Let's go with 100 estimators!

# In[ ]:


final_extra_trees_A = ExtraTreesClassifier(max_depth=40, max_features='sqrt', n_estimators=600)
final_extra_trees_A.fit(x_train, y_train)
print(final_extra_trees_A.score(x_test, y_test))
final_extra_trees_B = ExtraTreesClassifier(max_depth=40, max_features='sqrt', n_estimators=100)
final_extra_trees_B.fit(x_train, y_train)
print(final_extra_trees_B.score(x_test, y_test))


# #### again, our random grid search had it on point when combining the parameters together. let's check the coefficients than wrap it up.

# In[ ]:


featuers_coefficients = final_extra_trees.feature_importances_.tolist()
feature_names = x_copy.columns
feats = pd.DataFrame(pd.Series(featuers_coefficients, feature_names).sort_values(ascending=False),columns=['Coefficient'])
feats

