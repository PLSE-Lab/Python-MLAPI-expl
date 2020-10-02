#!/usr/bin/env python
# coding: utf-8

# # Income Classification - Gaussian RBF Kernel SVM, EDA
# 
# Today we'll be taking a look at the adult census data from the 1994 Census bureau database. We'll be doing an Exploratory Data Analysis first on the significant findings found, then perform feature engineering and selection, and finally create our model and evaluate the results.
# 
# # 1. Data Preparation

# Let's first load the data and take a look at it.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df


# We can already see a couple of interesting variables that could help us in classifying the income variable. We can also see that we have 32561 rows of data. Let's shuffle it and create a holdout set that will not be part of our data analysis and modeling. This is done so our final model's result won't be influenced by having seen the holdout data before.

# In[ ]:


df = df.sample(frac=1, random_state=10)
holdout = df[:3000]
df = df[:-3000]


# I decided to remove the last 3000 rows of the shuffled dataset and use it as our holdout set for testing later. Let's take a look at the lengths of each dataframe to verify.

# In[ ]:


print(len(df))
print(len(holdout))


# # 2. Exploratory Data Analysis
# 
# During this EDA, the most important observations in the data that was found would be the ones showcased.
# 
# One of the first variables one would notice would be education. It seems like a variable that would correlate a lot to the income variable. Let's take a look at a bar plot of the income variable that's grouped using the education variable.

# In[ ]:


import seaborn as sns
from matplotlib import pyplot
sns.set(style="whitegrid", font_scale=1.2)

a4_dims = (20, 10)
fig, ax = pyplot.subplots(figsize=a4_dims)
g = sns.countplot(y="education", hue="income", data=df, palette="muted")


# There is a growth in the proportion of people whose income is larger than 50k the higher the education attained gets. The growth is almost non-existent for the first few levels from pre-school to 12th, this may indicate that the growth is not entirely linear. For the Masters and Doctorate category, the number of people who earn more than 50k is higher than the people who earn less than 50k. Overall, we can see that there is indeed a correlation between the income variable and the level of education attained.
# 
# Let's then take a look at marital status. Which is another interesting variable.

# In[ ]:


sns.set(style="whitegrid", font_scale=1.2)

# Draw a nested barplot to show survival for class and sex
a4_dims = (20, 10)
fig, ax = pyplot.subplots(figsize=a4_dims)
g = sns.countplot(y="marital.status", hue="income", data=df, palette="muted")


# The number of people whose income is lesser than 50k is significantly higher than the number of people who earn more or equal to 50k for all of the statuses except Married-civ-spouse. For the status Married-civ-spouse, the number of people who earn more or equal to 50k is close to the number of people who earn less. This may be because as a spouse, being able to provide for the family or children would normally require an income higher than 50k. Overall, this variable shows correlation as well to the income variable as there are noticeable differences.
# 
# Lastly, let's look at the numerical variables and their relationships with each other. We'll also one-hot encode our income variable into a numerical one, but we'll only keep one of the newly created income variables as having both of them would not add a lot of information. 

# In[ ]:


import seaborn as sns

df = pd.get_dummies(df, columns=['income'])
df = df.drop(['income_<=50K'], axis=1)
corr = df.corr()
a4_dims = (20, 10)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.heatmap(corr, annot=True)


# The pair of variables with the highest correlation would be the "education_num" and "income_>50k". This makes sense as we've observed earlier that there is indeed a difference in the income variable the higher the level of attained education becomes. Another noticeable pair would be "income_>50k" and "age", this is understandable as the higher your age, the more likely that you've attained a higher position at a company. "hours.per.week" paired with "income_>50k" also has the same correlation as the previous pair. This makes sense as more hours per week renderred could affect the amount of pay or income that you would obtain. Lastly, the pair "capital.gain" and "income_>50K" is also correlated to each other. A higher capital gain could also indicate a higher income so this pair being correlated is also understandable.

# 
# 
# # 3. Feature Engineering and Selection
# 
# Before we start, let's first convert our categorical variables into numerical ones via one-hot encoding.

# In[ ]:


df2 = df.copy()
df = pd.get_dummies(df, columns=['occupation', 'race', 'workclass', 'marital.status', 'relationship'])


# Let's take a look at the top correlated variables as it may give us more information.

# In[ ]:


df.corr().unstack().sort_values().drop_duplicates()


# Something to notice here is that marital.status variables seems to have high correlation with relationship variables. This would make sense as a martial.status that's "Married-civ-spouse" would obviously indicate that the relationship variable is "husband". This could also mean that the relationship variable is already explained by the marital.status variable so keeping both would not contribute a lot to our model. So we'll drop the relationship variable and only keep martial.status.
# 
# Another key thing to notice here is that workclass_? and occupation_? have a near perfect correlation with each other. We will only keep one of the two as well later on.

# In[ ]:


df = df2
df = pd.get_dummies(df, columns=['occupation', 'race', 'workclass', 'marital.status'])


# Earlier we noticed that the relationship between "education" and "income" may not be linear due to the low amount of increase in the first few levels. We'll do a bit of changing with the "education.num" variable as it is the numerical form of education. Let's first take a look at the correlation between "income_>50k" and "education.num".

# In[ ]:


df['income_>50K'].corr(df['education.num'])


# Since we've noticed that the first few levels exhibited an almost unnoticeable growth in the number of people earning more than 50k, we'll group the first 8 educational levels into one.

# In[ ]:


df['new_educ_num'] = df.apply(lambda x: x['education.num']-8 if x['education.num'] >= 9 else 0, axis=1)


# Let's check the correlation of this new variable with "income_>50K"

# In[ ]:


df['income_>50K'].corr(df['new_educ_num'])


# We can see a small increase in correlation as compared to our previous unchanged variable. Note that correlation that is used is Pearson Correlation and it only takes into account the linear relationship between the two variables.
# 
# Let's then perform feature selection. We'll be using Chi-square feature selection to select our independent variables. Using this feature selection method, variables which are highly dependent on the response variable will be the ones selected as our features for the model. In this case we'll select the top 16% highest-scoring variables as our features.

# In[ ]:


from sklearn.feature_selection import SelectPercentile, chi2

X = df.drop(['income_>50K', 'education', 'sex', 'native.country', 'education.num', 'relationship','occupation_?'], axis=1)
y = df[['income_>50K']]

test = SelectPercentile(score_func=chi2, percentile=16)

fit = test.fit(X, y)

X.columns[test.get_support()]


# Now that we've obtained our features, let's normalize our variables first since SVMs are sensitive to the scaling of their features.

# In[ ]:


from sklearn.preprocessing import StandardScaler

X = df[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week',
       'marital.status_Married-civ-spouse', 'new_educ_num']]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# # 4. Modeling and Evaluation
# Now that we have everything set up. Let's first train a Linear SVM classifier first to check how it performs before diving into a kernelized SVM. We'll use Scikit-Learn's LinearSVC to train our model. We'll also set the dual parameter to False which tells the LinearSVC to solve the primal constrained optimization problem instead (this is better when we have n_samples > n_features).

# In[ ]:


from sklearn.svm import SVC, LinearSVC

model = LinearSVC(max_iter = 10000, dual=False)
model.fit(X, y.values.ravel())


# Let's now utilize our holdout set to assess our model's performance. The same steps we took during the feature engineering abd selection phase will be done again for our holdout set.

# In[ ]:


holdout['new_educ_num'] = holdout.apply(lambda x: x['education.num']-8 if x['education.num'] >= 9 else 0, axis=1)

holdout = pd.get_dummies(holdout, columns=['occupation', 'race', 'workclass', 'marital.status', 'income'])

test_X = holdout[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week',
       'marital.status_Married-civ-spouse', 'new_educ_num']]
test_y = holdout[['income_>50K']]

test_X = scaler.transform(test_X)
y_pred = model.predict(test_X)


# Let's look at the classification report to assess our model's performance.

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_y, y_pred, target_names=['Income more than 50K', 'Income less than 50K']))


# With a simple linear SVM, we've achieved an 84% accuracy in our results. Though it seems that it does better at predicting instances when the income is higher than 50K. A noticeable result here is that the recall for "Income less than 50K" is pretty low. Which indicates that there are a lot of data points that were actually "Income less than 50K" but were deemed to be more than 50K.
# 
# Let's now try out a kernelized SVM to see if there's a difference in performance results. We'll be using grid search to find the optimal hyperparameters. C, which indicates the amount of regularization applied(less is more regularization) and Gamma, which indicates how far the influence of the training examples. Gamma is one of the parameter that's used in the Gaussian RBF Kernel, which we'll be using.
# 
# We'll utilize SVC this time instead of LinearSVC as SVC solves the dual problem which supports the kernel trick.

# In[ ]:


from sklearn.model_selection import GridSearchCV

C_range = [0.1, 1, 10, 100]
gamma_vals = [0.001, 0.0001, 'scale']
param_grid = dict(gamma=gamma_vals, C=C_range)

grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=10)
grid.fit(X, y.values.ravel())

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# Let's now train the model using those parameters.

# In[ ]:


model = SVC(kernel='rbf', **grid.best_params_)


# In[ ]:


model


# In[ ]:


model = model.fit(X, y.values.ravel())


# In[ ]:


y_pred = model.predict(test_X)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_y, y_pred, target_names=['Income more than 50K', 'Income less than 50K']))


# Using a kernelized SVM we achieved a similar but slightly better result. "Income less than 50K" still has a low recall. But overall, this is the better model as it has an increase in predictive power on both classes. However, training a kernelized SVM takes a lot longer than training a simple linear SVM.
