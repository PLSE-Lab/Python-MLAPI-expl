#!/usr/bin/env python
# coding: utf-8

# # Default of Credit Card - Kaggle
# 
# Kaggle task: https://www.kaggle.com/gpreda/default-of-credit-card-clients-predictive-models
# 

# There are **25** variables:
# 
# ID: ID of each client
# 
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# 
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment 
# delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# 
# PAY_3: Repayment status in July, 2005 (scale same as above)
# 
# PAY_4: Repayment status in June, 2005 (scale same as above)
# 
# PAY_5: Repayment status in May, 2005 (scale same as above)
# 
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# 
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# 
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# 
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# 
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# 
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# 
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# 
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# 
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# default.payment.next.month: Default payment (1=yes, 0=no)

# # 1. Importing libraries#
# 
# 1. Pandas 
# 2. Seaborn - visualisations

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Reading file###

# In[ ]:


df = pd.read_csv("../input/UCI_Credit_Card.csv")


# # 3. Preprocessing ###
# 
# #### 3.1 Checking for number of rows and columns in the dataframe for a general overview.
# 

# In[ ]:


df.shape


# We have 30,000 rows and 25 columns. 

# In[ ]:


df.head(1)


# #### 3.2 Checking for null values

# In[ ]:


df.isnull().sum()


# The data does not contain any missing value.

# #### 3.3 Renaming where necessary
# 
# We will rename two columns. 
# 
# 
# 1. The **BILL_AMT** & **PAY_AMT** range from 1 to 6 for the six given months. However, a different column **PAY** ranges starts from 0 and then continues from 2 to 6. To introduce uniformity in the data, we will rename PAY_0 to PAY_1.
# 
# 
# 2. The column '**default.payment.next.month**' is a little too long. Let us change it to something simpler like **IS_DEFAULT**. It can take two values. 1 denotes true and 0 denotes false.

# In[ ]:


df = df.rename(columns={'default.payment.next.month': 'IS_DEFAULT', 
                        'PAY_0': 'PAY_1'})
df.head(1)


# #### 3.4 Checking for unique values#### 
# 
# Checking distinct values of some columns. Skipping id, Limit, Bill amount, pay amount

# In[ ]:


list_of_cols = list(df)
for col in list_of_cols:
    if col.startswith('LIMIT') or col.startswith('BILL') or col.startswith('PAY_A'):
        pass
    else:
        print(str(col) + ": "+ str(df[col].unique()))


# Two things stand out. 
# 
# 1. Education can have a value 0 which was not known before. Grouping 0,4,5,6 as unknowns. Replacing any education greater than or equal to four with 0
# 
# 
# 
# 2. Only two marital statuses are known (1 and 2). If the marital status is 0 or 3, we group them as unknown. Replacing any marital status with 3 as its value is assigned value 0.

# In[ ]:


df.loc[df.EDUCATION >= 4, 'EDUCATION'] = 0
df.loc[df.MARRIAGE == 3, 'MARRIAGE'] = 0


# ### Summarising data preprocessing
# 
# Here is a quick recap of preprocessing manipulations we did.
# 
# 1. Renamed default.payment.next.month to IS_DEFAULT
# 2. Renamed PAY_0 to Pay_1 
# 3. Grouped unknown education categories (0,4,5,6) and re-assigned them 0
# 4. Grouped unknown marital categories (0,3) and re-assigned them 0

# # 4. Exploratory Data Analysis###
# 
# #### 4.1 Let us start by having a look at the distribution of defaulters and non-defaulters.

# In[ ]:


fig = sns.countplot(x = 'IS_DEFAULT', data = df)
fig.set_xticklabels(["No Default", "Default"])


# In[ ]:


number_of_defaulters = len(df[df.IS_DEFAULT == 1]) 
number_of_non_defaulters = len(df) - number_of_defaulters
percentage_of_defaulters = number_of_defaulters/number_of_non_defaulters * 100
round(percentage_of_defaulters, 2) #28.4%


# The chart and data above show that the number of non-defaulters is significantly greater than the number of defaulters.There

# #### 4.2 Limit balance

# In[ ]:


df['LIMIT_BAL'].describe()


# As the max value of LIMIT_BAL is 10,00,000. We group the limit values in to 5 groups. We assign the LIMIT_VAL to these groups

# In[ ]:


bins = [0, 200000, 400000, 600000, 800000, 1000000]
df['LIMIT_GROUP'] = pd.cut(df['LIMIT_BAL'], bins,include_lowest=True)


# In[ ]:


# df_2 = df.LIMIT_GROUP.groupby(df.IS_DEFAULT)
# df_2.IS_DEFAULT
# # axis = df_2.LIMIT_GROUP.value_counts(sort = False).plot.bar(rot=0, color="r", figsize=(6,4))


# #### 4.3 Sex
# 
# 

# In[ ]:


#Computing percentage
number_of_male_card_holders = (df.SEX == 1).sum() #11,888
number_of_female_card_holders = (df.SEX == 2).sum() #18,112

number_of_male_defaulters = (df[df.SEX == 1].IS_DEFAULT == 1).sum() #2,873
number_of_female_defaulters = (df[df.SEX == 2].IS_DEFAULT == 1).sum() #3,763

percentage_of_male_def = round((number_of_male_defaulters/number_of_male_card_holders) * 100,2) #24.17%
percentage_of_female_def = round((number_of_female_defaulters/number_of_female_card_holders) * 100,2) #20.78%
temp_df = pd.DataFrame({"non-defaulters":{"male":100 - percentage_of_male_def, "female":100 - percentage_of_female_def},"defaulters":{"male":percentage_of_male_def, "female":percentage_of_female_def}})

#Plotting chart
fig = temp_df.plot(kind = 'bar')
fig.set_title("Percentage of male and female non-defaulters vs defaulters")
fig.set_ylabel("Percentage")


# There isn't a major difference in the distribution of default corresponding to gender. 24.17% of male card holders are defaulters compared to 20.78% of female card holders.

# #### 4.4 EDUCATION

# In[ ]:


#Computing percentage
number_of_unknown_edu_card_holders = (df.EDUCATION == 0).sum() #468
number_of_grad_edu_card_holders = (df.EDUCATION == 1).sum() #10,585
number_of_uni_card_holders = (df.EDUCATION == 2).sum() #14,030
number_of_high_school_card_holders = (df.EDUCATION == 3).sum() #4,917

number_of_unknown_edu_defaulters = (df[(df.EDUCATION == 0)].IS_DEFAULT == 1).sum() #33
number_of_grad_defaulters = (df[(df.EDUCATION == 1)].IS_DEFAULT == 1).sum() #2036
number_of_uni_defaulters = (df[(df.EDUCATION == 2)].IS_DEFAULT == 1).sum() #3330
number_of_high_school_defaulters = (df[(df.EDUCATION == 3)].IS_DEFAULT == 1).sum() #1237

percentage_of_unknown_def = round((number_of_unknown_edu_defaulters/number_of_unknown_edu_card_holders) * 100,2) #7.05
percentage_of_grad_def = round((number_of_grad_defaulters/number_of_grad_edu_card_holders) * 100,2) #19.23
percentage_of_uni_def = round((number_of_uni_defaulters/number_of_uni_card_holders) * 100,2) #23.73
percentage_of_high_school_def = round((number_of_high_school_defaulters/number_of_high_school_card_holders) * 100,2) #25.16
temp_df = pd.DataFrame({"non-defaulters":{"Unknown":100 - percentage_of_unknown_def, "Graduates":100 - percentage_of_grad_def, "University":100 - percentage_of_uni_def, "High school":100 - percentage_of_high_school_def},"defaulters":{"Unknown": percentage_of_unknown_def, "Graduates": percentage_of_grad_def, "University": percentage_of_uni_def, "High school":percentage_of_high_school_def}})

#Plotting chart
fig = temp_df.plot(kind = 'bar')
fig.set_title("Percentage of non-defaulters & defaulters based on education level")
fig.set_ylabel("Percentage")


# People with high school education and people with university education had the largest portion of defaulters. (23.73% and 25.16% respectively.)

# #### 4.5 MARRIAGE

# In[ ]:


number_of_others_card_holders = (df.MARRIAGE == 0).sum() #377
number_of_married_card_holders = (df.MARRIAGE == 1).sum() #13,659
number_of_unmarried_card_holders = (df.MARRIAGE == 2).sum() #15,964

number_of_others_def = (df[(df.MARRIAGE == 0)].IS_DEFAULT == 1).sum() #89
number_of_married_def = (df[(df.MARRIAGE == 1)].IS_DEFAULT == 1).sum() #3,206
number_of_ummarried_def = (df[(df.MARRIAGE == 2)].IS_DEFAULT == 1).sum() #3,341

percentage_of_others_def = round(number_of_others_def/number_of_others_card_holders * 100,2) #23.61
percentage_of_married_def = round(number_of_married_def/number_of_married_card_holders * 100,2) #23.47
percentage_of_ummarried_def = round(number_of_ummarried_def/number_of_unmarried_card_holders * 100,2) #20.93


temp_df = pd.DataFrame({"non-defaulters":{"Unknown":100 - percentage_of_others_def, "Married":100 - percentage_of_married_def, "Unmarried":100 - percentage_of_ummarried_def},
                        "defaulters":{"Unknown":percentage_of_others_def, "Married":percentage_of_married_def, "Unmarried": percentage_of_ummarried_def}})
fig = temp_df.plot(kind = 'barh')
fig.set_title("Percentage of non-defaulters & defaulters based on education level")
fig.set_xlabel("Percentage")


# #### 4.6 Age

# In[ ]:


sns.set(rc={'figure.figsize':(12,5)})
fig = sns.countplot(x = 'AGE', data = df, hue = 'IS_DEFAULT')
fig.legend(title='Is Default?', loc='upper right', labels=["Not Default", "Default"])
fig.set_title("Defaulters based on education level")


# #### 4.7 Correlation 

# In[ ]:


sns.set(rc={'figure.figsize':(25,8)})
sns.set_context("talk", font_scale=0.7)
sns.heatmap(df.iloc[:,1:].corr(), cmap='Greens', annot=True)


# Limit shows a healthy negative correlation with IS_DEFAULT. Similarly, there is a significant

# # 5. Prediction

# Selecting Useful features

# In[ ]:


X_train = df.iloc[:,[0,2,5,6,7,8,9,10]]
Y_train = df.iloc[:,[23]]


# Importing machine learning libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# **Decision Tree**

# In[ ]:


clf = tree.DecisionTreeClassifier()
model = clf.fit(X_train, Y_train)
acc = round(model.score(X_train, Y_train) * 100, 2)
train_pred = model_selection.cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs = -1)
c_acc = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)
print(acc)
print(c_acc)

