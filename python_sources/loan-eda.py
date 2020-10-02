#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[ ]:


url = '/kaggle/input/loan-prediction-problem-dataset'
train_url = url + '/train_u6lujuX_CVtuZ9i.csv'
test_url = url +'test_Y3wMUE5_7gLdaTN.csv'


# # Identify variables

# *info* is a very good and compact function which says
# 1. shape of the data
# 2. Type of each column and whether or not it has nulls

# In[ ]:


train_data = pd.read_csv(train_url)
print(train_data.info())


# In[ ]:


train_data.drop(['Loan_ID'], axis=1,inplace=True)


# let us plot the number of classes in different categorical variables

# In[ ]:


def cateInfo(data):
    categorical = set(data.select_dtypes(include='object')) - {'Loan_ID'}
    no_of_classes = {col:len(set(data[col]) - {np.nan}) for col in categorical}
    
    no_of_classes = pd.DataFrame(no_of_classes.values(), index = no_of_classes.keys(), columns=['n_classes']).sort_values(by='n_classes')
    no_of_classes.plot(kind='bar')
    plt.show()


# In[ ]:


cateInfo(train_data)


# **Variable Info**
# 
# > Numerical Variables: 
#     1. ApplicantIncome,
#     2. CopplicantIncome,
#     3. LoanAmount,
#     4. Loan_Amount_Term,
#     5. Credit_History
# 
# > Categorical Variables:
#     1. Nominal: 
#         Gender, Married, Self_Employed, Loan_Status(Target)
#     2. Ordinal:
#         Dependents, Property_Area, Education
# 
# Nominal variables are those which doesn't have ordering among its classes i.e., one isn't superior to other
# whereas ordinal has (for eg. quality of car -- worse < good < excellent)
# 
# Identifying the variables allow us to plan univariate and bivariate analysis.

# # Univariate Analysis
# 
# After having looked at just the variable types its time to delve into each variable to understand its distribution

# In[ ]:


def catePlot(feature):
    plt.title(feature.name)
    feature.value_counts(normalize=True).plot(kind='bar')
    plt.show()

for col in train_data.select_dtypes(include='object'):
    catePlot(train_data[col])


# **Inferences**
# 
# *Gender*:-  
# >data looks highly biased on gender as 80% of the applicants are male
#     
# *Married*:-  
# >~65% of the applicants are married
# 
# *Dependents*:-  
# >Most do not have dependents (might be insightful, since we also have coapplicant income)
# 
# *Education*:-  
# >~80% are graduates, it might be that either most ppl tend seek after they are graduated or its just the way data has been collected
# 
# *Self_Employed*:-  
# >~85% are salaried persons (quite intuitive)
# 
# *Property_Area*:-
# >balanced
# 
# *Loan_Status*:-  
# >Among all the applicants around 70% got their loan approved, also evident that we have imbalanced data, might have to apply some techniques to find better decision boundary

# In[ ]:


def contPlot(feature):
    _, axs = plt.subplots(1,2)
    axs[0].set_title(feature.name)
    axs[0].boxplot(feature)
    axs[1].hist(feature, bins=50)
    plt.show()

for col in train_data.select_dtypes(exclude='object'):
    contPlot(train_data[col].dropna())


# **Inferences**
# 
# *Loan_Amount_Term*:-  
# >This is numeric but doesn't seems to be continuous
# 
# *Credit_History*:-  
# >Same ,Infact this is binary either 0 or 1

# In[ ]:




