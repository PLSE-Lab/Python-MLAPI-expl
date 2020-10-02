#!/usr/bin/env python
# coding: utf-8

# # Medical Cost Personal Dataset
# **Columns**
# 
# age: age of primary beneficiary
# 
# sex: insurance contractor gender, female, male
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to 
# height,objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# children: Number of children covered by health insurance / Number of dependents
# 
# smoker: Smoking
# 
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# charges: Individual medical costs billed by health insurance
# 
# In notebook I am trying determine predictive variables by analysing the dataset visualy

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/insurance/insurance.csv')
df.head()


# # Finding correlation between Age and Charges by ploting a scatter plot.

# In[ ]:


plt.title('Relation between Age and Charges')
sns.scatterplot(x=df['age'],y=df['charges'])
plt.show()
plt.title('Regression between Age and Charges')
sns.regplot(x=df['age'],y=df['charges'])
plt.show()


# We can notice that older people tend to pay slightly more but to make it more clearer we can draw a regression link. The regression line shows positive correlation. Hence, Age plays small role in predicting insurance price.

# # Finding correlation between BMI and Charges by ploting a scatter plot.

# In[ ]:


plt.title('Relation between BMI and Charges')
sns.scatterplot(x=df['bmi'],y=df['charges'])
plt.show()
plt.title('Relation between BMI and Charges')
sns.regplot(x=df['bmi'],y=df['charges'])
plt.show()


# The scatterplot above suggests that body mass index (BMI) and insurance charges are positively correlated, where customers with higher BMI typically also tend to pay more in insurance costs. The positive regression line proves it.

# # Finding correlation between Smokers and Charges by ploting a scatter plot.

# In[ ]:


sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'])


# The scatter plot shows that while nonsmokers do tend to pay slightly more with increasing BMI, smokers pay much more. To further emphasize this fact, I have added two regression lines, corresponding to smokers and nonsmokers.

# In[ ]:


sns.lmplot(x="bmi", y="charges", hue="smoker", data=df)


# We can notice that the regression line for smokers has a much steeper slope, relative to the line for nonsmokers. Lets conclude with categorical scatter plot.

# In[ ]:


sns.swarmplot(x=df['smoker'],y=df['charges'])


# On average, non-smokers are charged less than smokers, and the customers who pay the most are smokers whereas the customers who pay the least are non-smokers. Hence, smoking habits determine the insurance charges.

# # Finding correlation between Children and Charges by ploting a scatter plot.

# In[ ]:


plt.figure(figsize=(14,6))
plt.title('Relation between Age and Charges')
#sns.regplot(x=df['children'],y=df['charges'])
sns.barplot(x=df['children'], y=df['charges'])


# We can easily say that person having 2 and 3 children tend to pay more. Surprisingly person having 5 children pays the least insurance charges. Hence, definately children is a predictor variable.

# # Finding correlation between Sex and Charges by ploting a scatter plot.

# In[ ]:


sns.swarmplot(x=df['sex'],y=df['charges'])
plt.show()
sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['sex'])
plt.show()
sns.barplot(x=df['sex'], y=df['charges'])
plt.show()


# We cannot find much difference between cost paid by mail and female in first plot and sex is alomst equally scatteret.The bar chart also shows that there is very little difference between average cost paid by mail and female. Hence, sex of the person does not necessarily determine the insurance charges one pays.

# # Finding correlation between Region and Charges by ploting a scatter plot.

# In[ ]:


sns.swarmplot(x=df['region'],y=df['charges'])
plt.show()
sns.barplot(x=df['region'], y=df['charges'])
plt.show()


# First plot does give some information but second graph clearly shows region of the person is important bscause on an average people from southeast and northeast tend to pay more.
# 
# Hence, the predictor variables are:
# * Age
# * BMI
# * Somker
# * Children
# * Region
