#!/usr/bin/env python
# coding: utf-8

# # **Income Classification Model**

# ## Introduction

# The income dataset was extracted from 1994 U.S. Census database.
# 
# ### The importance of census statistics 
# The census is a special, wide-range activity, which takes place once a decade in the entire country. The purpose is to gather information about the general population, in order to present a full and reliable picture of the population in the country - its housing conditions and demographic, social and economic characteristics. The information collected includes data on age, gender, country of origin, marital status, housing conditions, marriage, education, employment, etc.
# 
# This information makes it possible to plan better services, improve the quality of life and solve existing problems. Statistical information, which serves as the basis for constructing planning forecasts, is essential for the democratic process since it enables the citizens to examine the decisions made by the government and local authorities, and decide whether they serve the public they are meant to help.
# 
# Read more: [Use of Census Data](http://www.cbs.gov.il/census/census/pnimi_sub_page_e.html?id_topic=1&id_subtopic=5)
# 
# ### Objective of the porject
# The goal of this machine learning project is ** to predict whether a person makes over 50K a year ** or not given their demographic variation. To achieve this, several classification techniques are explored and the random forest model yields to the best prediction result.
# 
# * Source: 
#  -  [adult data set](https://archive.ics.uci.edu/ml/datasets/adult/)
#  -  [Income dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)
# 
# 

# ![image.png](http://www.jbencina.com/blog/wp-content/uploads/2015/07/2013-Household-Income-County-Inequality.png)

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><span><a href="#The-importance-of-census-statistics" data-toc-modified-id="The-importance-of-census-statistics-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>The importance of census statistics</a></span></li><li><span><a href="#Objective-of-the-porject" data-toc-modified-id="Objective-of-the-porject-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Objective of the porject</a></span></li></ul></li><li><span><a href="#Fetching-Data" data-toc-modified-id="Fetching-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Fetching Data</a></span><ul class="toc-item"><li><span><a href="#Import-Package-and-Data" data-toc-modified-id="Import-Package-and-Data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Import Package and Data</a></span></li><li><span><a href="#Data-Dictionary" data-toc-modified-id="Data-Dictionary-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Data Dictionary</a></span></li></ul></li><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Cleaning</a></span><ul class="toc-item"><li><span><a href="#Dealing-with-Missing-Value" data-toc-modified-id="Dealing-with-Missing-Value-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Dealing with Missing Value</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Predclass" data-toc-modified-id="Predclass-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Predclass</a></span></li><li><span><a href="#Education" data-toc-modified-id="Education-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Education</a></span></li><li><span><a href="#Marital-status" data-toc-modified-id="Marital-status-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Marital-status</a></span></li><li><span><a href="#Occupation" data-toc-modified-id="Occupation-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Occupation</a></span></li><li><span><a href="#Workclass" data-toc-modified-id="Workclass-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Workclass</a></span></li><li><span><a href="#age" data-toc-modified-id="age-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>age</a></span></li><li><span><a href="#Race" data-toc-modified-id="Race-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Race</a></span></li><li><span><a href="#Hours-of-Work" data-toc-modified-id="Hours-of-Work-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Hours of Work</a></span></li><li><span><a href="#Create-a-crossing-feature:-Age-+-hour-of-work" data-toc-modified-id="Create-a-crossing-feature:-Age-+-hour-of-work-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Create a crossing feature: Age + hour of work</a></span></li></ul></li><li><span><a href="#EDA" data-toc-modified-id="EDA-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>EDA</a></span><ul class="toc-item"><li><span><a href="#Pair-Plot" data-toc-modified-id="Pair-Plot-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Pair Plot</a></span></li><li><span><a href="#Correlation-Heatmap" data-toc-modified-id="Correlation-Heatmap-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Correlation Heatmap</a></span></li><li><span><a href="#Bivariate-Analysis" data-toc-modified-id="Bivariate-Analysis-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Bivariate Analysis</a></span></li><li><span><a href="#Occupation-vs.-Income-Level" data-toc-modified-id="Occupation-vs.-Income-Level-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Occupation vs. Income Level</a></span></li><li><span><a href="#Race-vs.-Income-Level" data-toc-modified-id="Race-vs.-Income-Level-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Race vs. Income Level</a></span></li></ul></li><li><span><a href="#Building-Machine-Learning-Models" data-toc-modified-id="Building-Machine-Learning-Models-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Building Machine Learning Models</a></span><ul class="toc-item"><li><span><a href="#Feature-Encoding" data-toc-modified-id="Feature-Encoding-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Feature Encoding</a></span></li><li><span><a href="#Train-test-split" data-toc-modified-id="Train-test-split-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Train-test split</a></span></li><li><span><a href="#Principal-Component-Analysis-(PCA)" data-toc-modified-id="Principal-Component-Analysis-(PCA)-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Principal Component Analysis (PCA)</a></span></li><li><span><a href="#Classification-Models" data-toc-modified-id="Classification-Models-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Classification Models</a></span><ul class="toc-item"><li><span><a href="#Perceptron-Method" data-toc-modified-id="Perceptron-Method-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>Perceptron Method</a></span></li><li><span><a href="#Gaussian-Naive-Bayes" data-toc-modified-id="Gaussian-Naive-Bayes-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Gaussian Naive Bayes</a></span></li><li><span><a href="#Linear-Support-Vector-Machine" data-toc-modified-id="Linear-Support-Vector-Machine-6.4.3"><span class="toc-item-num">6.4.3&nbsp;&nbsp;</span>Linear Support Vector Machine</a></span></li><li><span><a href="#Radical-Support-Vector-Machine" data-toc-modified-id="Radical-Support-Vector-Machine-6.4.4"><span class="toc-item-num">6.4.4&nbsp;&nbsp;</span>Radical Support Vector Machine</a></span></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.4.5"><span class="toc-item-num">6.4.5&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-6.4.6"><span class="toc-item-num">6.4.6&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#K-Nearest-Neighbors" data-toc-modified-id="K-Nearest-Neighbors-6.4.7"><span class="toc-item-num">6.4.7&nbsp;&nbsp;</span>K-Nearest Neighbors</a></span></li></ul></li><li><span><a href="#Cross-Validation" data-toc-modified-id="Cross-Validation-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Cross Validation</a></span><ul class="toc-item"><li><span><a href="#GridSearch" data-toc-modified-id="GridSearch-6.5.1"><span class="toc-item-num">6.5.1&nbsp;&nbsp;</span>GridSearch</a></span></li></ul></li></ul></li><li><span><a href="#Takeaways:" data-toc-modified-id="Takeaways:-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Takeaways:</a></span></li></ul></div>

# ## Fetching Data 

# ### Import Package and Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


income_df = pd.read_csv("../input/adult.csv")
income_df.head()


# In[ ]:


income_df.describe()


# ### Data Dictionary

# ** *1. Categorical Attributes* **
#  * workclass: (categorical) Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#   -  Individual work category  
#  * education: (categorical) Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#   -  Individual's highest education degree  
#  * marital-status: (categorical) Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#   -  Individual marital status  
#  * occupation: (categorical) Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#   -  Individual's occupation  
#  * relationship: (categorical) Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#   -  Individual's relation in a family   
#  * race: (categorical) White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#   -  Race of Individual   
#  * sex: (categorical) Female, Male.
#  * native-country: (categorical) United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#   -  Individual's native country   
# 
# 
# 
# ** *2. Continuous Attributes* **
#  * age: continuous.
#   -  Age of an individual  
#  * education-num: number of education year, continuous.
#   -  Individual's year of receiving education
#  * fnlwgt: final weight, continuous. 
#  * The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US.  These are prepared monthly for us by Population Division here at the Census Bureau.
#  * capital-gain: continuous.
#  * capital-loss: continuous.
#  * hours-per-week: continuous.
#   -  Individual's working hour per week   
# 
# 
# 
# 

# ## Data Cleaning

#  ### Dealing with Missing Value

# In[ ]:


income_df.isnull().sum()


# Attributes workclass, occupation, and native-country most NAs. Let's drop these NA. 

# In[ ]:


income_df.age = income_df.age.astype(float)
income_df['hours-per-week'] = income_df['hours-per-week'].astype(float)


# In[ ]:


my_df = income_df.dropna()


# In[ ]:


my_df['predclass'] = my_df['income']
del my_df['income']
my_df['education-num'] = my_df['educational-num']
del my_df['educational-num']


# In[ ]:


my_df.info()


# In[ ]:


my_df.isnull().sum()


# ## Feature Engineering

# In[ ]:


print('workclass',my_df.workclass.unique())
print('education',my_df.education.unique())
print('marital-status',my_df['marital-status'].unique())
print('occupation',my_df.occupation.unique())
print('relationship',my_df.relationship.unique())
print('race',my_df.race.unique())
print('gender',my_df.gender.unique())
print('native-country',my_df['native-country'].unique())
print('predclass',my_df.predclass.unique())


# ### Predclass

# In[ ]:


#my_df.loc[income_df['predclass'] == ' >50K', 'predclass'] = 1
#my_df.loc[income_df['predclass'] == ' <=50K', 'predclass'] = 0


# In[ ]:


#predclass1 = my_df[my_df['predclass'] == 1]
#predclass0 = my_df[my_df['predclass'] == 0]


# In[ ]:


fig = plt.figure(figsize=(20,1))
plt.style.use('seaborn-ticks')
sns.countplot(y="predclass", data=my_df)


# Income level less than 50K is more than 3 times of those above 50K, indicating that the the dataset is somewhat skewed. However, since there is no data on the upper limit of adult's income above 50K, it's premature to conclude that the total amount of wealth are skewed towards high income group.

# ### Education

# In[ ]:


#income_df[['education', 'education-num']].groupby(['education'], as_index=False).mean().sort_values(by='education-num', ascending=False)


# In[ ]:



my_df['education'].replace('Preschool', 'dropout',inplace=True)
my_df['education'].replace('10th', 'dropout',inplace=True)
my_df['education'].replace('11th', 'dropout',inplace=True)
my_df['education'].replace('12th', 'dropout',inplace=True)
my_df['education'].replace('1st-4th', 'dropout',inplace=True)
my_df['education'].replace('5th-6th', 'dropout',inplace=True)
my_df['education'].replace('7th-8th', 'dropout',inplace=True)
my_df['education'].replace('9th', 'dropout',inplace=True)
my_df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
my_df['education'].replace('HS-grad', 'HighGrad',inplace=True)
my_df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
my_df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
my_df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
my_df['education'].replace('Bachelors', 'Bachelors',inplace=True)
my_df['education'].replace('Masters', 'Masters',inplace=True)
my_df['education'].replace('Prof-school', 'Masters',inplace=True)
my_df['education'].replace('Doctorate', 'Doctorate',inplace=True)


# In[ ]:


my_df[['education', 'education-num']].groupby(['education'], as_index=False).mean().sort_values(by='education-num', ascending=False)


# In[ ]:


fig = plt.figure(figsize=(20,3))
plt.style.use('seaborn-ticks')
sns.countplot(y="education", data=my_df)


# ### Marital-status

# In[ ]:


#df2 = my_df['marital-status'].replace(' Never-married', 'NotMarried')
my_df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
my_df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
my_df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
my_df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
my_df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)


# In[ ]:


fig = plt.figure(figsize=(20,2))
plt.style.use('seaborn-ticks')
sns.countplot(y="marital-status", data=my_df)


# ### Occupation

# In[ ]:


plt.style.use('seaborn-ticks')
plt.figure(figsize=(20,4)) 
sns.countplot(y="occupation", data=my_df)


# ### Workclass

# In[ ]:


plt.style.use('seaborn-ticks')
plt.figure(figsize=(20,3)) 
sns.countplot(y="workclass", data=my_df)


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
#grid = sns.FacetGrid(my_df, col='predclass', row='workclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'age', alpha=.5, bins=20)
#grid.add_legend()


# ### age

# In[ ]:


# make the age variable discretized 
my_df['age_bin'] = pd.cut(my_df['age'], 20)


# In[ ]:


plt.style.use('seaborn-ticks')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="age_bin", data=my_df)
plt.subplot(1, 2, 2)
sns.distplot(my_df[my_df['predclass'] == '>50K']['age'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == '<=50K']['age'], kde_kws={"label": "<=$50K"})


# In[ ]:


my_df[['predclass', 'age']].groupby(['predclass'], as_index=False).mean().sort_values(by='age', ascending=False)


# ### Race

# In[ ]:


plt.style.use('seaborn-whitegrid')
x, y, hue = "race", "prop", "gender"
#hue_order = ["Male", "Female"]
plt.figure(figsize=(20,5)) 
f, axes = plt.subplots(1, 2)
sns.countplot(x=x, hue=hue, data=my_df, ax=axes[0])

prop_df = (my_df[x]
           .groupby(my_df[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())

sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])


# ### Hours of Work 

# In[ ]:


# Let's use the Pandas Cut function to bin the data in equally sized buckets
my_df['hours-per-week_bin'] = pd.cut(my_df['hours-per-week'], 10)
my_df['hours-per-week'] = my_df['hours-per-week']


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="hours-per-week_bin", data=my_df);
plt.subplot(1, 2, 2)
sns.distplot(my_df['hours-per-week']);
sns.distplot(my_df[my_df['predclass'] == '>50K']['hours-per-week'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == '<=50K']['hours-per-week'], kde_kws={"label": "<$50K"})
plt.ylim(0, None)
plt.xlim(20, 60)


# ### Create a crossing feature: Age + hour of work

# In[ ]:


g = sns.jointplot(x = 'age', 
              y = 'hours-per-week',
              data = my_df, 
              kind = 'hex', 
              cmap= 'hot', 
              size=10)

#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn
sns.regplot(my_df.age, my_df['hours-per-week'], ax=g.ax_joint, scatter=False, color='grey')


# In[ ]:


my_df.head()


# In[ ]:


# Crossing Numerical Features
my_df['age-hours'] = my_df['age']*my_df['hours-per-week']
my_df['age-hours_bin'] = pd.cut(my_df['age-hours'], 10)


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="age-hours_bin", data=my_df);
plt.subplot(1, 2, 2)
sns.distplot(my_df[my_df['predclass'] == '>50K']['age-hours'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == '<=50K']['age-hours'], kde_kws={"label": "<$50K"})


# ## EDA

# ### Pair Plot

# In[ ]:


#pair plots of entire dataset
pp = sns.pairplot(my_df, hue = 'predclass', palette = 'deep', 
                  size=3, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=20) )
pp.set(xticklabels=[])


# ### Correlation Heatmap

# In[ ]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = "YlGn",
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(my_df)


# ### Bivariate Analysis

# In[ ]:


my_df.tail()


# In[ ]:


import math

def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            
bivariate_df = my_df.loc[:, ['workclass', 'education', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'gender','predclass']]  

plot_bivariate_bar(bivariate_df, hue='predclass', cols=2, width=20, height=15, hspace=0.4, wspace=0.5)


# The dataset was created in 1996, a large number of jobs fall into the category of mannual labor, e.g., Handlers cleaners, craft repairers, etc. Executive managerial role and some one with a professional speciality has a high level payment.

# ### Occupation vs. Income Level

# In[ ]:


from matplotlib import pyplot
a4_dims = (20, 5)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.violinplot(x="occupation", y="age", hue="predclass",
                    data=my_df, gridsize=100, palette="muted", split=True, saturation=0.75)
ax


# The general trend is in sync with common sense: more senior workers have higher salaries. Armed-forces don't have a high job salaries.
# 
# Interestingly, private house sevice has the widest range of age variation, however, the payment is no higher than 50K, indicating that senority doesn't give rise to a higher payment comparing to other jobs. 

# ### Race vs. Income Level

# ![censusrace](https://user-images.githubusercontent.com/31974451/36568899-8e25bc6c-17e0-11e8-9e85-53d0f5cc1d7f.png)
# 

# In[ ]:


from matplotlib import pyplot
a4_dims = (20, 5)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.violinplot(x="race", y="age", hue="predclass",
                    data=my_df, gridsize=100, palette="muted", split=True, saturation=0.75)
ax


# ## Building Machine Learning Models

# In[ ]:


from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import GridSearchCV


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
#train,test=train_test_split(train_df,test_size=0.2,random_state=0,stratify=abalone_data['Sex'])


# ### Feature Encoding 

# In[ ]:


# Feature Selection and Encoding
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #training and testing data split


# In[ ]:


my_df = my_df.apply(LabelEncoder().fit_transform)
my_df.head()


# ### Train-test split

# In[ ]:


drop_elements = ['education', 'native-country', 'predclass', 'age_bin', 'age-hours_bin','hours-per-week_bin']
y = my_df["predclass"]
X = my_df.drop(drop_elements, axis=1)
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# ### Principal Component Analysis (PCA)

# In[ ]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[ ]:


# PCA's components graphed in 2D and 3D
# Apply Scaling 
std_scale = preprocessing.StandardScaler().fit(my_df.drop('predclass', axis=1))
X = std_scale.transform(my_df.drop('predclass', axis=1))
y = my_df['predclass']

# Formatting
target_names = [0,1]
colors = ['blue','yellow','pink']
lw = 2
alpha = 0.3
# 2 Components PCA
plt.style.use('seaborn-whitegrid')
plt.figure(2, figsize=(20, 8))

plt.subplot(1, 2, 1)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], 
                color=color, 
                alpha=alpha, 
                lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('First two PCA directions');

# 3 Components PCA
ax = plt.subplot(1, 2, 2, projection='3d')

pca = PCA(n_components=3)
X_reduced = pca.fit(X).transform(X)
for color, i, target_name in zip(colors, [0, 1], target_names):
    ax.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], X_reduced[y == i, 2], 
               color=color,
               alpha=alpha,
               lw=lw, 
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

# rotate the axes
ax.view_init(30, 10)


# In[ ]:


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
pca = PCA(n_components=None)
x_train_pca = pca.fit_transform(X_train_std)
a = pca.explained_variance_ratio_
a_running = a.cumsum()
a_running


# ### Classification Models

# #### Perceptron Method

# In[ ]:


## Perceptron Method
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=1, random_state=1)
ppn.fit(X_train, y_train)


# In[ ]:


y_pred = ppn.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:


## cross_val_score for ppn method
from sklearn.model_selection import cross_val_score
score_ppn=cross_val_score(ppn, X,y, cv=5)
score_ppn.mean()


# #### Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
# y_pred = gaussian.predict(X_test)
score_gaussian = gaussian.score(X_test,y_test)
print('The accuracy of Gaussian Naive Bayes is', score_gaussian)


# #### Linear Support Vector Machine

# In[ ]:


# Support Vector Classifier (SVM/SVC)
from sklearn.svm import SVC
svc = SVC(gamma=0.22)
svc.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)
score_svc = svc.score(X_test,y_test)
print('The accuracy of SVC is', score_svc)


# #### Radical Support Vector Machine

# In[ ]:


svc_radical =svm.SVC(kernel='rbf',C=1,gamma=0.22)
svc_radical.fit(X_train,y_train.values.ravel())
score_svc_radical = svc_radical.score(X_test,y_test)
print('The accuracy of Radical SVC Model is', score_svc_radical)


# #### Logistic Regression

# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)
score_logreg = logreg.score(X_test,y_test)
print('The accuracy of the Logistic Regression is', score_logreg)


# #### Random Forest

# In[ ]:


# Random Forest Classifier
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
#y_pred = randomforest.predict(X_test)
score_randomforest = randomforest.score(X_test,y_test)
print('The accuracy of the Random Forest Model is', score_randomforest)


# #### K-Nearest Neighbors

# In[ ]:


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
score_knn = knn.score(X_test,y_test)
print('The accuracy of the KNN Model is',score_knn)


# ### Cross Validation

# In[ ]:


### cross validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Naive Bayes','Linear Svm','Radial Svm','Logistic Regression','Decision Tree','KNN','Random Forest']
models=[GaussianNB(), svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
models_dataframe


# Random Forest is the most accurate model.

# #### GridSearch
model = RandomForestClassifier()
parameters = {'n_estimators': [500], 
              'max_features': ['log2', 'sqrt','auto'], 
              #The number of features to consider when looking for the best split
              'max_depth': [50,80,100,150], 
              'min_samples_split': [5,7,9,11],
             }
grid_obj = GridSearchCV(model, parameters, scoring="neg_log_loss",n_jobs=4,cv = 5)
grid_obj = grid_obj.fit(X_train,y_train)
model_params = grid_obj.best_params_
model_params
# ## Takeaways: 

# ** Takeaway**
# * How to run PCA and in what situation is it useful
# * How to beautify jupyter notebook for presentation
# * Common sense strengthen the results of EDA
# 
# 
# ** Tradeoff**
# * The dataset might be suffering from selection bias, e.g., it only includes data where people have one race but omits those with two races
# * The dataset is extracted 10 years ago, some of the facts might not be applicable now
# * Since the dataset has oberseravtions, it takes a long time to run Gridsearch
# 
# ** Next Step**
# * Gridsearch
# * feature selection for the champion model

# ![life](https://user-images.githubusercontent.com/31974451/36570647-dfff6248-17e7-11e8-9d1a-3037d4897460.jpeg)
# 
