#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Prediction
# In this notebook, I seek to predict the income of people based on multiple socio-economic factors, using the Adult Income Census dataset. I will plot several of these factors against income in order to visualize trends and patterns. Finally, I will be using 5 machine learning models in attempts to accurately predict if a person earns a yearly income greater than or less than $50K.
# ### Machine Learning Models
# * XGBclassifier
# * Random Forest Classifier
# * Linear Discriminat Analysis
# * Decision Tree Classifier
# * Logistic Regression
# <a id="top"></a> <br>
# 
# ### 1. [Preparing Libraries & Data](#1)
# ### 2. [Cleaning Data](#2)
# ### 3. [Visualization](#3)
# ### 4. [Modeling](#4)
# ### 5. [Model Tuning](#5)
# ### 6. [Outcome](#6)

# <a id="1"></a> <br>
# ## 1. Preparing Libraries & Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')
sns.set_context("poster")

dataset = pd.read_csv("../input/adult.csv")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Setup Complete")


# <a id="2"></a> <br>
# ## 2. Cleaning Data

# In[ ]:


dataset.head(5)


# In[ ]:


dataset.isnull().sum()
#checking for missing values


# In[ ]:


#Object in the case = Text
#Int64 = Numbers
dataset.dtypes


# <a id="3"></a> <br>
# ## 3. Visualization

# ### Barplot of  Workclass vs Income

# In[ ]:


dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})
dataset["workclass"] = dataset["workclass"].replace(["?"],'Unknown')
fig, ax = plt.subplots(figsize=(25,7))
sns.set_context("poster")
current_palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark")

fig = sns.barplot(x='workclass',y='income',data=dataset,palette=current_palette)

fig.set_ylabel("Income >50K Probability [%]")
fig.set_xlabel("Workclass")


# From the barplot of Workclass vs Income we see from high to low that **self-emp-inc** has the highest chance of making over 50k with a **54%** chance, while the last known workclass(**private**) has a** 22%** of making over 50k.
# * Self-emp-inc : 54%
# * Federal-gov : 38%
# * Local-gov : 29%
# * Self-emp-not-inc : 28%
# * State-gov : 26%
# * Private : 22%
# * Unknown : 11%

# ### Barplot of Marital Status vs Income

# In[ ]:


fig, axe = plt.subplots(2,1,figsize=(27,12))
current_palette =sns.cubehelix_palette(8)
plt.style.use('seaborn-whitegrid')
fig = sns.barplot(x='marital.status',y='income',data=dataset,ax=axe[0],order =['Never-married','Separated','Widowed','Divorced','Married-spouse-absent','Married-AF-spouse','Married-civ-spouse'],palette=current_palette)

fig.set_ylabel("Income >50K Probability [%]")
fig.set_xlabel("Marital Status")


datasetCopy= dataset.copy()
datasetCopy["marital.status"] = datasetCopy["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], '1')
datasetCopy["marital.status"] = datasetCopy["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], '0')
datasetCopy["marital.status"] = datasetCopy["marital.status"].astype(int)

sns.set_context("poster")
fig = sns.barplot(x='marital.status',y='income',data=datasetCopy,ax=axe[1], order=[1,0],palette=['indigo','silver'])

fig.set(xticklabels =['Married','Single'])
fig.set_ylabel("Income >50K Probability [%]")
fig.set_xlabel("Marital Status")


# From the first barplot of Marital Status vs Income, we see that, from high to low,  **Married-civ-spouse** has the highest probability of earning an income of over 50k with a **43%** chance, while **Never-married** has the lowest, with a **5%** chance of earning over 50k.
# * Married-civilian-spouse : 45%
# * Married-Armed Forces-spouse : 42% 
# * Divorced  : 10%
# * Widowed : 8%
# * Married-spouse-absent : 7%
# * Separated : 6%
# * Never-married : 5%
# 
# From the second barplot ofMarital Status vs Income plot, we can see a wide discrepancy in probability of earning a higher income based on marital status.  **Married** adults on average have a **45%** probability of earning over 50k , while **single** adults possess only a  **7%** probability of earning over 50k. 
# * Married : 45%
# * Single : 7%

# ### Barplot of Relationship vs Income

# In[ ]:


fig, ax = plt.subplots(figsize=(25,7))
sns.set_context("poster")
current_palette = sns.color_palette("Blues")

fig = sns.barplot(x='relationship',y='income',data=dataset, order=['Own-child','Other-relative','Unmarried','Not-in-family','Husband','Wife'], palette=current_palette)

fig.set_ylabel("Income >50K Probability [%]")
fig.set_xlabel("Relationship")


# From the barplot of Relationship vs Income, we see that, from high to low,  **Wife** has the highest probability of earning an income of over 50k with a **47%** chance, while **Own-child** has the lowest, with a **1%** chance of earning over 50k.
# * Wife : 47%
# * Husband : 44% 
# * Not-in-family  : 11%
# * Unmarried : 5%
# * Other-relative : 3%
# * Own-child : 1%
# 
# This plot clearly shows that wives and husbands consecutively have the highest probabilities of earning incomes over 50k. This data agrees with the outcome of the previous barplot of **Marital Status vs Income**, which is that, in general,  married people are more likely to earn higher incomes of over 50k, while single people are significantly less likely to make over 50k. 

# ### Correlation Matrix for Numeric Data

# In[ ]:


numeric_columns = ['marital.status','education.num','age','hours.per.week','capital.gain','capital.loss','income']

fig, axe = plt.subplots(figsize=(15,15))
sns.set_context("poster")
sns.set(font_scale=2)
map1 = sns.color_palette("RdBu_r", 7)
sns.heatmap(datasetCopy[numeric_columns].corr(),annot=True, fmt='.2f',linewidths=2,cmap = map1)


# The heatmap indicates that the most reliably predictors of earning an income over 50k are:
# * Marital Status : 43%
# * Years of Continuous Education : 34%
# * Age : 23%
# * Hours Per Week : 23% 
# 
# This furthermore confirms the previously plotted data that marital status is indeed an crucial factor in determining adults' likelihood of earning a high income. 

# ### Violinplot of Age vs Income vs Sex 
# 

# In[ ]:


fig, axe = plt.subplots(figsize=(25,8))
sns.set_context("poster")

g=sns.violinplot(x='age',y='income',hue='sex',hue_order=["Male","Female"],data=dataset,orient="h",palette=["cornflowerblue","indianred",])

g.set_ylabel("Income")
g.set_xlabel("Age")
g.set(yticklabels =['<=50k','>50k'])
setThis = g.legend(loc='center right')
plt.xlim(0,100)
axe.xaxis.set_major_locator(ticker.MultipleLocator(5))


# The Violinplot of Age vs Income vs Sex shows a strong correlation between age and income, and a loose correlation between sex and income. This indicates that younger people, regardless of the gender, consistently earn less than their older counterparts. Thus, we can deduce that a person's age has a greater impact on income earned than their gender. 
# 

# ### Violinplot of Age vs Marital Status vs Income 

# In[ ]:


dataset = pd.read_csv("../input/adult.csv")
fig, axe = plt.subplots(figsize=(25,8))
sns.set_context("poster")

g=sns.violinplot(x='age',y='marital.status',hue='income',data=datasetCopy,orient="h",palette=["c",'seagreen'])

g.set_ylabel("Marital Status")
g.set_xlabel("Age")
setThis = g.legend(loc='center right')
setThis.get_texts()[0].set_text("<=50k")
setThis.get_texts()[1].set_text(">50k")
g.set(yticklabels =['Single','Married'])
plt.xlim(0,100)
axe.xaxis.set_major_locator(ticker.MultipleLocator(5))


# The Violin-plot of Age vs Marital Status vs Income shows that people who are **younger** and **single** are more likely to make less than 50k, while people who are **older** and **single** have a much higher probability of making over 50k. 
# 
# People who are **younger** and **married** have a slightly higher probability to earn over 50k compared to their single counterparts. Those who are **older** and **married** have the highest probability to make over 50k as we can see from the width of the violin plot.

# ### Violinplot/Lineplot Education year vs Age vs Income

# In[ ]:


fig, axe = plt.subplots(figsize=(31,9))
sns.set_context("poster")

g=sns.violinplot(x="education.num",y='age',hue='income',data=dataset,palette=["skyblue",'mediumseagreen'],ax = axe)
sns.lineplot(x="education.num",y='age',hue='income',data=dataset,palette=["c",'green'],ax=axe)


plt.ylim(0,100)
g.set_xlabel("Years of Continuous Education")
g.set_ylabel("Age")
setThis = g.legend(bbox_to_anchor=(.906, 1),loc=2)
setThis.get_texts()[0].set_text("Income")
axe.yaxis.set_major_locator(ticker.MultipleLocator(10))


#  |Education numbers | Meaning |     
#  | -----------------------|:-----------:|
#  |1 |preschool|
#  |2-8| <HS-grad
#  |9|HS-grad|
#  |10|some-college|
#  |11-12|Associate
#  |13|Bachelors|
#  |14|Masters|
#  |15|Prof-School|
#  |16|Doctorate|
#  
#  ---------------------
# This violin plot graphs Education year vs Age vs Income It indicates that adults who are older have greater probabilities of earning over 50k, even when compared to younger adults with higher levels of education. However, higher education levels still yield higher probabilities of higher income for younger adults. 
# 
# The line plot was overlaid to help visualize the overall trend. We can see that the median of older adults (indicated by the white dots within each violin) remains higher than the median of younger adults all throughout the different levels of education. 
# This data can be further contextualized when viewed in conjunction with the heatmap data, which showed that marital status, age, as well as years of continuous education (which positively correlate with one's age) are key factors in determining one's income. 

# ### lmplot Age vs Hours-per-week vs Income

# In[ ]:


dataset = pd.read_csv("../input/adult.csv")
sns.set_context("poster")
g=sns.lmplot(x="age",y='hours.per.week',hue='income',col="income",data=dataset,markers= ['x','o'],palette=["c",'seagreen'],height=15,line_kws={'color': 'darkslategray'})

g= (g.set_axis_labels("Age","Hours Per Week"))


# This linear regression plots show the correlation between age and length of work weeks among adults who earn under and over 50k, respectively. On average, adults who earn under 50k have a tendency to work more hours as they age, however many people remain working approximately 30-45 hour weeks, and a substantial fraction of much older adults drastically decrease the length of their work weeks. Adults who earn over 50k tend to decrease their working hours as they age (with a decrease rate larger than the increase rate of the work weeks of adults earning less than 50k). Many adults earning over 50k  continue to work 30-60 hour work weeks until they become much older.

# <a id="4"></a> <br>
# ## 4. Modeling

# In[ ]:


#Before we can begin to model are dataset, we first have to drop any categorical data and convert the one's we want to keep into binary:: Yes (1) or No (0)
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
dataset["marital.status"] = dataset["marital.status"].map({"Married":0, "Single":1})
dataset["marital.status"] = dataset["marital.status"]
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})
dataset.drop(labels=["sex","workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)

dataset.head(5)


# In[ ]:


numeric_columns = ['marital.status','age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
X=dataset[numeric_columns]
Y=dataset.income
train_X, val_X, train_y, val_y = train_test_split(X,Y,test_size=0.21,random_state=0)

outcome = []
Modelnames = []
models = []
models.append(('Random Forest Classifier', RandomForestClassifier(n_estimators=50, max_features=4)))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Logistic Regression', LogisticRegression(solver='lbfgs')))

kfoldCV = StratifiedKFold(n_splits=5, random_state=0)
xgb_model = XGBClassifier(n_estimators=250)
results = cross_val_score(xgb_model, train_X, train_y, cv=kfoldCV)
print("XGBClassifier: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
outcome.append(results)
Modelnames.append("XGBClassifier")

for name, model in models:
    kfoldCV = KFold(n_splits=5, random_state=0)
    cv_r = cross_val_score(model, train_X, train_y, cv=kfoldCV, scoring='accuracy')
    outcome.append(cv_r)
    Modelnames.append(name)
    print("%s: %.2f%% (%.2f%%)" % (name, cv_r.mean()*100, cv_r.std()*100))


# In[ ]:


fig, axe = plt.subplots(figsize=(27,10))
data1 ={'Names': Modelnames,'Results': outcome}
fig.suptitle('Model Accuracy Comparison')
current_palette = sns.color_palette("RdBu_r", 5)
sns.set_context("poster")
sns.boxplot(x='Names',y='Results',data=data1,palette = current_palette)


# <a id="5"></a> <br>
# ## 5. Model Tuning

# In[ ]:


eval_set=[(val_X,val_y)]
for i in [50,100,200,400,800,1600]:
    xgb_model = XGBClassifier(n_estimators=i,learning_rate=0.05).fit(train_X,train_y,early_stopping_rounds=i-i*.75,eval_set=eval_set, verbose=False)
    results = xgb_model.predict(val_X)
    predictions = [round(value) for value in results]
    accuracy = accuracy_score(val_y, predictions)
    print("Accuracy: %.2f%% --- %.2f N_estimators" % (accuracy * 100.0,i))


# <a id="6"></a> <br>
# ## 6. Outcome

# In[ ]:


xgb_model = XGBClassifier(n_estimators=1600,learning_rate=0.05).fit(train_X,train_y,early_stopping_rounds=i-i*.75,eval_set=eval_set, verbose=False)
results = xgb_model.predict(val_X)
print("Accuracy: %s%%" % (100*accuracy_score(val_y, results)))
print(classification_report(val_y, results))


# ##### [Go to top](#top)
