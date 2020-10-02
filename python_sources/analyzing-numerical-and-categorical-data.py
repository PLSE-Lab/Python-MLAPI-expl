#!/usr/bin/env python
# coding: utf-8

# ## Analysing Numerical and Categorical data
# 
# This notebook is meant to be a follow-up to my videos on Analysing numerical data at https://youtu.be/w3USlEccVy8  and analysing categorical data at https://youtu.be/De24HyoMBog . I have taken the familiar titanic dataset for this write up. I hope this is a good introduction to anyone who is new to the Data Analysis.

# In[ ]:


#Load Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


# #### Loading Data and processing missing values
# 
# Loading Titanic Dataset from the seaborn library here. There are null in the Deck and age column. We are just going to 'forward fill' the null values.

# In[ ]:


#Load Titanic Dataset
titanic = pd.read_csv("../input/train.csv") #sns.load_dataset('titanic')
titanic.isnull().sum()


# In[ ]:


titanic = titanic.fillna(method='ffill')
print(titanic.shape)


# #### Data Analysis
# 
# There are three good function defined on Pandas dataframe to review the data. They are
# 1. dtypes() - describes the datatypes of individual columns
# 2. head() - lists the first 5 rows of the dataset. To see how the data values look like
# 3. describe() - Provides summary of numerical data in the data frame

# In[ ]:


print(titanic.dtypes)


# In[ ]:


print(titanic.head())


# In[ ]:


titanic.describe()


# #### Exploratory Data Analysis - Histograms
# 
# Exploratory Data analysis is done using the following common plots
# 1. Histogram - Good, simple plot for univariate data analysis. Usually meant for categorical data. But a numerical data will be automatically binned and plotted. 
# 2. Hollow Histogram - To compare the trend of a numerical data across two or more groups. Hence provides a two-dimensional view of data.

# In[ ]:


#single histogram
# magic command. sets up Matplotlib to display the plots inline that are just static images.
#%matplotlib notebook - This would have created interactive plots
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(titanic['Age']) 
plt.title("Passengers By Age")
plt.xlabel("Passenger Age")
plt.ylabel("Passenger Count")


# In[ ]:


#hollow histogram
agebins = np.linspace(0,80,20) # we can specify the number of bins, there by controlling the shape of the historgram. 
plt.hist(titanic[titanic.Sex == 'male'].Age, agebins, alpha = 0.5, label = 'male' )
plt.hist(titanic[titanic.Sex == 'female'].Age, agebins, alpha = 0.5, label = 'female')
plt.legend(loc = 'upper right')
plt.title("Passengers By Age and Gender")
plt.xlabel("Passenger Age")
plt.ylabel("Passenger Count")


# #### Exploratory data analysis - boxplot
# 
# 1. Boxplot - My favorite plot. Univariate analysis of numerical data. Provides median, IQR (InterQuartile Range) and presence of outliers in the data distribution.
# 
# 2. Side-by-side Boxplot - Bivariate analysis. Provides information on data distribution of numerical data across two or more groups

# In[ ]:


#single boxplot
plt.boxplot(titanic['Age'])
plt.title("Passengers By Age")
plt.ylabel("Age")


# In[ ]:


#paired boxplot 1
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')
titanic.boxplot(column = 'Age', by = 'Survived', figsize= (6,5))
plt.title('Survival By Age')
plt.ylabel("Passenger Age")


# In[ ]:


#paired boxplot 2

with sns.axes_style(style='ticks'):
    g = sns.factorplot("Sex", "Age", "Pclass", data=titanic, kind="box")
    g.set_titles("Passenger Age across gender and class traveled")
    g.set_axis_labels("Sex", "Age");


# In[ ]:


#paired boxplot 3
titanic['alive'] = np.where(titanic.Survived == 1, 'yes','no')
with sns.axes_style(style='ticks'):
    g = sns.factorplot("Sex", "Age", "alive", data=titanic, kind="box")
    g.set_axis_labels("Sex", "Age");


# #### Exploratory Data Analysis - Scatterplots
# 
# 1. Scatterplot - performs bivariate analysis of two numerical features(columns). Provides information on possible correlation between these features.
# 2. Scatterplot with (color and scale) - Provide additional dimension to the relationship between the numerical features through color and size of the points in the scatter plot. This is probably one of the most popular plots used in data analysis.

# In[ ]:


#scatterplot
plt.scatter(x=titanic.Age, y=titanic.Fare)
plt.title("Passenger Age Vs fare")
plt.xlim(0,80)
plt.ylim(0,500)


# In[ ]:


#scatterplot 2 
pclass = titanic.Pclass.astype('category')
plt.scatter(x=titanic.Age, y=titanic.Fare, c=pclass, alpha = 0.5, cmap = 'viridis')
plt.xlim(0,80)
plt.ylim(0,500)
plt.colorbar(ticks = range(4) ,label = 'Passenger class')


# ### Analysis of Categorical data
# 
# Categorical data are discrete data with values from a set of defined categories. In addition to using Histogram to analyse them, there are two common tables that are frequently used to describe and analyse them. They are
# 
# 1. Frequency table - Describes data distribution across one category. 
# 2. Contingency table - Describes data distribution across two categories. This is one of the most common tables for data analysis. Especially in Panda these tables can have multiple levels of indexes along both rows and columns there by increasing the number of dimensions the data is visualized at. This idea of multi-index is one of the powerful features of Pandas

# In[ ]:


#frequency table1 - passengers survived vs died
pd.crosstab(titanic.Survived, columns='count')


# In[ ]:


'''frequency table 2 - Number of passengers across the age distribution. Age is a numeric column. Hence it is converted into
categorical data by using pd.cut function. This creates a set number of bins for age'''
agebins = pd.cut(titanic.Age, 5)
agetab =  pd.crosstab(agebins, columns='count')
agetab.index = ["Under 16", "16 to 32", "32 to 48", "48 to 64", "over 64"]
agetab


# In[ ]:


#contingency table 1 - Passenger survival across the class of travel
classbysurvivalTab =   pd.crosstab(titanic.Pclass, columns=titanic.alive, margins= True)
classbysurvivalTab.columns = ["Died", 'Survived', 'RowTotal']
classbysurvivalTab.index = ["First Class", "Second Class", "Third Class", "ColTotal"]
classbysurvivalTab


# In[ ]:


#contingency table 2 - Data presented as proportions. This provides a better visual understanding than raw numbers above
classbysurvivalTab.div(classbysurvivalTab["RowTotal"], axis=0)# /classbysurvivalTab.ix["ColTotal"]


# 

# 
