#!/usr/bin/env python
# coding: utf-8

# We are beginners in the Machine Learning.  
# We are trying some notebooks of the Kaggle to understand how to work with a data, predict the outcome, visualize it, and many more.  
# Any advice is greatly welcome.

# Now, we will import neccessary libraries, prepare the plot style, ignoring some filters.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sb
plt.style.use('fivethirtyeight')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Import training and test data.

# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at the training data
train.describe(include="all")
train.shape
train.head(10)


# Check test data.

# In[ ]:


test.head()


# Let get the list of columns from training data.

# In[ ]:


#get a list of the features within the dataset
print(train.columns)


# Count the NULL values from train data.  
# We can see 177 missing values for Age, 687 for Cabin, and 2 for Embarked columns.

# In[ ]:


train.isnull().sum()


# Let see a plot of surived and not.

# In[ ]:


sb.countplot('Survived',data=train)
plt.show()


# Let's see a plot of survival between genders.

# In[ ]:


train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sb.countplot('Sex', hue='Survived', data=train,)
plt.show()


# Let see about the scatter plot between Age and Pclass attributes.  
# We can see all classes share similar ages, and likely the Pclass 3.00 had more children, adults than Pclass 1.0 and 2.0.

# In[ ]:


plt.scatter(train['Pclass'], train['Age'])
plt.show()


# How about survival between ages?

# In[ ]:


plt.scatter(train['Survived'], train['Age'])
plt.show()


# How about the distribution of survival with genders for different ages?

# In[ ]:


train.boxplot(column='Age', by=['Survived', 'Sex'])


# How about the distribution of survival with genders for different ages and class?

# In[ ]:


train.boxplot(column='Age', by=['Survived','Pclass'])


# Let see a summary between Sibling/Spouse and Pclass, how it'd been arranged.

# In[ ]:


pd.crosstab(train.SibSp,train.Pclass).style.background_gradient('summer_r')


# Next, let see one-hot encoding for the Pclass. In the data, it's 1.00, 2.00, 3.00.  
# So we will have an array of 1, 2, 3 (like categorical) for the classes after encoding.  
# When we transform the data, the Pclass one-hot encoding will apply for each record, the proper value of the Pclass will be shown as 1.  
# For example:
# - person A, Pclass 1.00, its data will be [1, 0, 0]
# - person B, Pclass 3.00, it will be [0, 0, 1]

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
one_hot = LabelBinarizer()
one_hot.fit_transform(train['Pclass'])


# In[ ]:


one_hot.classes_


# Next, I will go thru the imputation and few others.

# In[ ]:


train.info()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,5))
sb.heatmap(train.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


# In[ ]:


cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sb.countplot(train[cols[i]], hue=train["Survived"], ax=ax)
        ax.set_title(cols[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center') 
        
plt.tight_layout()   


# Of the 891 passengers in df_test, less than 350 survive.  
# Much more women survive than men.  
# Also, the chance to survive is much higher in Pclass 1 and 2 than in Class 3.  
# Survival rate for passengers travelling with SibSp or Parch is higher than for those travelling alone.  
# Passengers embarked in C and Q are more likely to survie than those embarked in S.

# In[ ]:


bins = np.arange(0, 80, 5)
g = sb.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sb.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()  


# Best chances to survive for male passengers was in Pclass 1 or being below 5 years old.  
# Lowest survival rate for female passengers was in Pclass 3 and being older than 40.  
# Most passengers were male, in Pclass 3 and between 15-35 years old.

# In[ ]:


train['Fare'].max()


# In[ ]:


bins = np.arange(0, 550, 50)
g = sb.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sb.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()


# **Bar and Box plots**
# 
# Default mode for seaborn barplots is to plot the mean value for the category.  
# Also, the standard deviation is indicated.  
# So, if we choose Survived as y-value, we get a plot of the survival rate as function of the categories present in the feature chosen as x-value.

# In[ ]:


sb.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass")
plt.show()


# As we know from the first Titanic kernel, survival rate decreses with Pclass.  
# The hue parameter lets us see the difference in survival rate for male and female.

# In[ ]:


sb.barplot(x='Sex', y='Survived', hue='Pclass', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival between Pclass and Sex")
plt.show()


# Highest survival rate (>0.9) for women in Pclass 1 or 2.  
# Lowest survival rate (<0.2) for men in Pclass 3.

# In[ ]:


sb.barplot(x='Embarked', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Embarked Port")
plt.show()


# Passengers embarked in "S" had the lowest survival rate, those embarked in "C" the highest.  
# Again, with hue we see the survival rate as function of Embarked and Pclass.

# In[ ]:


sb.barplot(x='Embarked', y='Survived', hue='Pclass', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Embarked Port")
plt.show()


# Survival rate alone is not good beacuse its uncertainty depends on the number of samples.  
# We also need to consider the total number (count) of passengers that embarked.

# In[ ]:


sb.countplot(x='Embarked', hue='Pclass', data=train)
plt.title("Count of Passengers as function of Embarked Port")
plt.show()


# Passengers embarked in "C" had largest proportion of Pclass 1 tickets.  
# Almost all Passengers embarked in "Q" had Pclass 3 tickets.  
# For every class, the largest count of Passengers embarked in "S".

# In[ ]:


sb.boxplot(x='Embarked', y='Age', data=train)
plt.title("Age distribution as function of Embarked Port")
plt.show()


# In[ ]:


sb.boxplot(x='Embarked', y='Fare', data=train)
plt.title("Fare distribution as function of Embarked Port")
plt.show()


# In[ ]:


cm_surv = ["darkgrey" , "lightgreen"]
fig, ax = plt.subplots(figsize=(13,7))
sb.swarmplot(x='Pclass', y='Age', hue='Survived', split=True, data=train , palette=cm_surv, size=7, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()


# Here, the high survival rate for kids in Pclass 2 is easily observed.  
# Also, it becomes more obvious that for passengers older than 40 the best chance to survive is in Pclass 1, and smallest chance in Pclass 3.

# In[ ]:


fig, ax = plt.subplots(figsize=(13,7))
sb.violinplot(x="Pclass", y="Age", hue='Survived', data=train, split=True, bw=0.05 , palette=cm_surv, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()


# In[ ]:


g = sb.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=train, kind="swarm", split=True, palette=cm_surv, size=7, aspect=.9, s=7)


# In[ ]:


g = sb.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=train, kind="violin", split=True, bw=0.05, palette=cm_surv, size=7, aspect=.9, s=7)

