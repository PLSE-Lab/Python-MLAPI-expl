#!/usr/bin/env python
# coding: utf-8

# # Medical Insurance Cost Analysis
# 
# Hey guys, so this is my first notebook on Kaggle. I'm still a beginner when it comes to data analysis/machine learning field,and I just wanted to try to make an exploratory data analysis on this dataset. If you have any suggestions/critics/recommendations for me in order to improve this work or my skills please go ahead and post it in the comments. Thanks! 
# 
# Let's get into this!
# 
# 
# ![](https://mk0nationalecze819jj.kinstacdn.com/wp-content/uploads/2017/05/health-care--e1495141013610.jpg)
# 
# 
# In this notebook, I am going to try and examine the diffrent relations between every variable and the charges variable in order to know how every variable affect the medical charges costs.
# 
# **Table of contents:**
# 
# &emsp; <a href='#1'> 1. Relation between age and charges </a><br><br>
# &emsp; <a href='#2'> 2. Relation between sex and charges </a><br><br>
# &emsp; <a href='#3'> 3. Relation between BMI and charges </a><br><br>
# &emsp; <a href='#4'> 4. Relation between number of children and charges </a><br><br>
# &emsp; <a href='#5'> 5. Relation between smoking and charges </a><br><br>
# &emsp; <a href='#6'> 6. Relation between regions and charges </a><br><br>
# &emsp; <a href='#7'> 7. Findings </a><br><br>

# In[ ]:


#Importing needed packages
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Importing data 
dataset = pd.read_csv('../input/insurance/insurance.csv')

#Taking a look at our data
dataset.head()


# In this notebook I'm going to check if there is any relation between the variables we have, especially the relations between the charges and other variables. By this I mean if being a female or a male has any effect on the charges, or maybe if being from a specific region will increase or decrease your medical charges.  
# 
# 
# First, let's check the distribution of the medical charges across all the dataset.

# In[ ]:


#Making a distplot of the charges
plt.figure(figsize=(12,4))
sns.distplot(a=dataset['charges'], color='deepskyblue', bins=100)
plt.title('Distribution of the medical charges\n across all the dataset', size='23')
plt.xlabel('Charges',size=18)
plt.show()


# As we can see our charges are right skewed meaning that most of our individuals have between 2k and 15k as medical costs billed by health insurance, let's check this through a countplot.

# In[ ]:


#Creating another column containing bins of charges
dataset['charges_bins'] = pd.cut(dataset['charges'], bins=[0, 15000, 30000, 45000, 60000, 75000])

dataset.head()


# As you can see, we divided our charges into five diffrent categories. Let's create a countplot based on these bins.

# In[ ]:


#Creating a countplot based on the amount of charges
plt.figure(figsize=(12,4))
sns.countplot(x='charges_bins', data=dataset, palette='husl') 
plt.title('Number of people paying x amount\n for each charges category', size='23')
plt.xticks(rotation='25')
plt.ylabel('Count',size=18)
plt.xlabel('Charges',size=18)
plt.show()


# As we can see, most of the people pay less than 15k for medical costs.
# Let's move now to check if there is any correlation between diffrent variables that we have. 
# 
# ## <a id='1'> 1. Relation between age and charges: </a>
# 
# We must check the distribution of our variable in the beginning:

# In[ ]:


#Making a distplot for the age variable
plt.figure(figsize=(12,4))
sns.distplot(a=dataset['age'], color='darkmagenta', bins=100) 
plt.title('Ages distrubution', size='23')
plt.xlabel('Age',size=18)
plt.show()


# For the age variable we have almost a uniform distribution, with a few exceptions at both ends of the values. 
# Now let's check if there is any relation between the two variables age and charges. 

# In[ ]:


#Making a lineplot to check if there is any correlation between age and charges
plt.figure(figsize=(12,4))
sns.lineplot(x='age', y='charges', data=dataset, color='mediumvioletred')
plt.title('Charges according to age', size='23')
plt.ylabel('Charges',size=18)
plt.xlabel('Ages',size=18)
plt.show()


# We can spot that there is a clear increase in charges as the the age increases, which is normal since older people need more medical care. Let's check how much people are paying approximatively according to their age categories. 

# In[ ]:


#Making bins for the ages
dataset['age_bins'] = pd.cut(dataset['age'], bins = [0, 20, 35, 50, 70])

#Creating boxplots based on the amount of diffrent age categories
plt.figure(figsize=(12,4))
sns.boxplot(x='age_bins', y='charges', data=dataset, palette='RdPu') 
plt.title('Charges according to age categories', size='23')
plt.xticks(rotation='25')
plt.grid(True)
plt.ylabel('Charges',size=18)
plt.xlabel('Age',size=18)
plt.show()


# We can still spot that older people pay more for their health charges. Individuals between 50-70 years old pay the most: The mean of their medical expenses is aroud 13k. 50% of these people pay between 12k and 22k as medical charges. 
# 
# ## <a id='2'> 2. Relation between sex and charges: </a>
# 
# First let's check the number of males and females in our dataset:

# In[ ]:


#Countplot of males/females
plt.figure(figsize=(12,4))
sns.countplot(x='sex', data=dataset, palette='PuBu') 
plt.title('Number of males/females', size='23')
plt.ylabel('Count',size=18)
plt.xlabel('Sex',size=18)
plt.show()


# Great! As we can see there is almost an equal number of men/women in our dataset let's check if the sex has any influence on the charges paied by indivduals 

# In[ ]:


#Cheking the charges distributions for males and females
x1 = sns.FacetGrid(dataset, row='sex', height=4, aspect=3.5)
x1 = x1.map(sns.distplot, 'charges', color='cornflowerblue')
plt.show()


# As we can see the two distributions are almost the same for both women/men, so we can affirm that there is no influence on the medical charges when it comes to the sex variable. 
# 
# ## <a id='3'> 3. Relation between BMI and charges: </a>
# 
# Let's see the distribution of our BMI variable first of all:

# In[ ]:


#Making a distplot for our BMI variable 
plt.figure(figsize=(12,4))
sns.distplot(a=dataset['bmi'], color='mediumseagreen', bins=100)
plt.title('Distribution of the BMI variable\n across all the dataset', size='23')
plt.xlabel('BMI',size=18)
plt.show()


# We have a beautiful normal distribution in our dataset with no outliers! So let's jump into seeing if there is any correlation between the BMI and medical charges. 

# In[ ]:


#Scatterplot to check for correlation 
plt.figure(figsize=(12,4))
sns.scatterplot(x='bmi', y='charges', data=dataset, color='seagreen')
plt.title('Charges according to BMI', size='23')
plt.ylabel('Charges',size=18)
plt.xlabel('BMI',size=18)
plt.show()


# We can't really spot if there is a correlation between the two variables, so look at the data from another angle.  

# In[ ]:


#Making bins and labels for the BMI
bins = [0, 18.5, 25, 30, 35, 40, 60]
labels = ['Underweight', 'Average', 'Overweight', 'Obese 1', 'Obese 2', 'Obese 3']
dataset['bmi_bins'] = pd.cut(dataset['bmi'], bins=bins, labels=labels)

#Checking the charges according to BMI 
plt.figure(figsize=(12,4))
sns.barplot(x='bmi_bins', y='charges', data=dataset, palette='Greens')
plt.title('Charges according to BMI categories', size='23')
plt.ylabel('Charegs',size=18)
plt.xlabel('BMI categories',size=18)
plt.show()


# As we can see there is a clear increase in the medical charges as we go up in classes. The classes 'Obese 2' and 'Obese 3' are the ones who pay the most at around 17.5k. What surprised me here is that the 'Underweight' category pays less than the 'Average' category! 
# 
# ## <a id='4'> 4. Relation between number of children and charges: </a>
# 
# Let's first see how individuals we have in each category:

# In[ ]:


#Countplot for diffrent 'number of children' categories
plt.figure(figsize=(12,4))
sns.countplot(x='children', data=dataset, palette='YlGnBu') 
plt.title('Number of pepople having x children', size='23')
plt.ylabel('Count',size=18)
plt.xlabel('Number of children',size=18)
plt.show()


# Okay, so most of our individuals don't have any children! We can notice that each time we increase the number of children by 1 child the count of individuals decreases. Maybe having many children isn't a trend nowadays! xD 
# 
# Let's move to check for any relation between the children and cthe amount of charges.

# In[ ]:


#Creating a violinplot for each category
plt.figure(figsize=(12,4))
sns.violinplot(x='children', y='charges', data=dataset, hue='sex', palette='YlGnBu')
plt.title('Charges according to number of children', size='23')
plt.ylabel('Charges',size=18)
plt.xlabel('Number of children',size=18)
plt.show()


# As we can see, almost all categories have the same range and mean of costs also the distributions are very similar, except for the people who have 5 children. This might be because of the small size of the sample of this kind of people! 
# 
# Let's move to the next variable, which for me is considered as the most interesting one which is smoker. 
# 
# ## <a id='5'> 5. Relation between smoking and charges: </a>
# 
# Let's first see the number of smokers in our dataset:

# In[ ]:


#Countplot to compare the number of smokers and non-smokers
plt.figure(figsize=(12,4))
sns.countplot(x='smoker', data=dataset, hue='sex', palette='YlOrBr') 
plt.title('Number of smokers and non-smokers', size='23')
plt.ylabel('Count',size=18)
plt.xlabel('Smoker',size=18)
plt.show()


# The number of non-smokers in our dataset is way bigger than those who smoke. Also, male smokers are more present than female smokers. Let's see if smoking has any effect on the charges variable.

# In[ ]:


#Creating boxplots to compare charges distributions for smokers and non-smokers
plt.figure(figsize=(12,12))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))
sns.boxplot(x='charges', y='sex' ,data=dataset[dataset['smoker']=='yes'], palette='YlOrBr', ax=ax1)
ax1.set_title('Somker',size='23')
ax1.set_ylabel('Sex',size=18)
ax1.set_xlabel('Charges',size=18)
sns.boxplot(x='charges', y='sex' ,data=dataset[dataset['smoker']=='no'], palette='YlOrBr', ax=ax2)
ax2.set_title('Non-somker', size='23')
ax2.set_ylabel('Sex',size=18)
ax2.set_xlabel('Charges',size=18)
plt.tight_layout()
plt.show()


# As we can see charges for smokers are much higher than charges for non-smokers. Sex doesn't have any effect on charges when you are a smoker, both men and women have very elevated charges. 
# 
# Now lets inspect the effect of the smoker and bmi variables on the charges. 

# In[ ]:


#Creating a FacetGrid to compare charges of smokers to non-smoker charges with diffrent BMI categories
x2 = sns.FacetGrid(dataset, row='smoker', height=4, aspect=3.5)
x2 = x2.map(sns.barplot, 'bmi_bins', 'charges', palette='YlOrBr', order=labels)
plt.show()


# As we can see smoking itself can increase your medical bills significantly. Furthemore, when we combine smoking with a high BMI can lead to big medical bills and important health issues.  
# 
# ## <a id='6'> 6. Relation between regions and charges: </a>
# 
# Let's check first how many people do we have from each region:

# In[ ]:


#Countplot to compare the number of individuals from diffrent regions
plt.figure(figsize=(12,4))
sns.countplot(x='region', data=dataset, palette='husl') 
plt.title('Number of individuals from diffrent regions', size='23')
plt.ylabel('Count',size=18)
plt.xlabel('Region',size=18)
plt.show()


# Almsot all the regions have equal number of individuals, now let's see if living in a particular region that has an impact on the medical charges. 

# In[ ]:


#Creating distplots to compare charges distributions for diffrent regions and the overall dsitribution of charges
plt.figure(figsize=(12,12))
ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
ax2 = plt.subplot2grid((3,2),(1,0))
ax3 = plt.subplot2grid((3,2),(1,1))
ax4 = plt.subplot2grid((3,2),(2,0))
ax5 = plt.subplot2grid((3,2),(2,1))
sns.distplot(a=dataset['charges'], ax=ax1,  color='lime')
ax1.set_title('Overall distribution',size='23')
ax1.set_xlabel('Charges',size=18)

axis = [ax2, ax3, ax4, ax5]
regions = ['southwest', 'southeast', 'northwest', 'northeast']
for axe, region in zip(axis, regions):
    data = dataset[dataset['region']==region]
    sns.distplot(a=data['charges'], ax=axe,  color='darkorchid')
ax2.set_title('Southwest', size='23')
ax2.set_xlabel('Charges',size=18)
ax3.set_title('Southeast', size='23')
ax3.set_xlabel('Charges',size=18)
ax4.set_title('Northwest', size='23')
ax4.set_xlabel('Charges',size=18)
ax5.set_title('Northeast', size='23')
ax5.set_xlabel('Charges',size=18)
plt.tight_layout()
plt.show()


# As you can see all regions have approximatively the same distributions for charges and they are similar to overall distribution of charges. So we can safely say that the region you live in has no effect on your medical bills.
# 
# ## <a id='7'> 7. Findings: </a>
# 
# After analysing all the relations between the diffrent variables and the 'charges' variable, we got diffrent results for each feature:
#    1. **age:** this variable has an impact on the charges, when a person is older the health costs are larger.
#    2. **sex:** the 'sex' variable doesn't affect the charges variable, it doesn't matter if you are a men or a women your health bills won't change.
#    3. **bmi:** for the BMI we found out after we grouped it to diffrent classes that when the weight increases, the health care charges increase along.
#    4. **children:** the number of children doesn't affect the medical costs billed by health insurance.
#    5. **smoker:** if you are a smoker you must expect some huge medical charges compared to non-smokers. Especially for people who have high BMI values (>35) it will result very serious health care charges. 
#    6. **region:** no matter where you live, this won't have any impact on your medical insurance bills.
