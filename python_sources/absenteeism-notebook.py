#!/usr/bin/env python
# coding: utf-8

# # Analysis of Absenteeism at work
# Gianluigi Lopardo - Tesina Data Spaces a.a 2019/2020
# 

# 
# ### Table of contents
# 1. [Introduction](#first)
# 2. [Data preprocessing](#second)
# 3. [Exploratory Data Analysis](#third)
# 4. [Data preparation](#fourth)
#   * 4.1 [Relabeling](#fourth_1)
#   * 4.2 [Principal Component Analysis](#fourth_2)
#   * 4.3 [Oversampling](#fourth_3)
# 5. [Classification Models](#fifth)
#   * 5.1 [Full classification](#fifth_1)
#   * 5.2 [Binary classification](#fifth_2)
# 6. [References](#sixth)
#   

# ## 1. <a class="anchor" id="first">Introduction</a>
# 
# The dataset used is [Absenteeism at work](http://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work) from UCI Machine learning repository. The aim of this work is to apply **classification** to **predict** absenteeism at work.
# 
# Absenteeism represents for the company the loss of productivity and quality of work. Predicting it can help companies organize tasks appropriately in order to optimize work and avoid stressful situations for both the company and its employees.
# 
# The Analysis is conducted in Python using Colab Notebook, which is a web application that allows you to create an interactive environment that contains live code, visualizations and text. 
# 
# The dataset contains 740 entries. Each entry has 21 attributes.
# It was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.
# 
# ### Attribute Information:
# 
# * Individual identification (ID)
# * Reason for absence
# 
#  Absences attested by the [International Code of Diseases (ICD)](https://www.who.int/classifications/icd/en/) stratified into 21 categories (1 to 21) as follows and 7 categories without ICD (22 to 28):
#   1. Certain infectious and parasitic diseases
#   2. Neoplasms
#   3. Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
#   4. Endocrine, nutritional and metabolic diseases
#   5. Mental and behavioural disorders
#   6. Diseases of the nervous system
#   7. Diseases of the eye and adnexa
#   8. Diseases of the ear and mastoid process
#   9. Diseases of the circulatory system
#   10. Diseases of the respiratory system
#   11. Diseases of the digestive system
#   12. Diseases of the skin and subcutaneous tissue
#   13. Diseases of the musculoskeletal system and connective tissue
#   14. Diseases of the genitourinary system
#   15. Pregnancy, childbirth and the puerperium
#   16. Certain conditions originating in the perinatal period
#   17. Congenital malformations, deformations and chromosomal abnormalities
#   18. Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
#   19. Injury, poisoning and certain other consequences of external causes
#   20. External causes of morbidity and mortality
#   21. Factors influencing health status and contact with health services.
#   22. patient follow-up 
#   23. medical consultation
#   24. blood donation 
#   25. laboratory examination 
#   26. unjustified absence 
#   27. physiotherapy 
#   28. dental consultation
# * Month of absence
# * Day of the week: Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6)
# * Seasons: summer (1), autumn (2), winter (3), spring (4)
# * Transportation expense
# * Distance from Residence to Work (kilometers)
# * Service time (years)
# * Age
# * Work load Average/day
# * Hit target
# * Disciplinary failure: yes=1, no=0
# * Education: high school (1), graduate (2), postgraduate (3), master and doctor (4))
# * Son (number of children)
# * Social drinker: yes=1, no=0
# * Social smoker yes=1, no=0
# * Pet (number of pet)
# * Weight
# * Height
# * Body mass index
# * Absenteeism time in hours (target)
# 
# 

# ## 2. <a class="anchor" id="second">Data preprocessing</a>
# After importing data and the Python libraries that we will use, the first step is to preprocess the data: we have to convert the data from one format to another, we will check the missing or invalid values and convert everything in a standardized format on all the data, so that we can easily manage and analyze them.

# In[ ]:


# IMPORTING PACKAGES

# warnings
import warnings
warnings.filterwarnings(action = 'ignore', category = FutureWarning)

# file and data
from google.colab import files
import calendar

# scientific
import pandas as pd 
import numpy as np 
from numpy import random

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# dimensionality and oversampling
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler

# models 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

SEED = 42
random.seed(SEED)


# In[ ]:


# IMPORTING DATA
url = "https://raw.githubusercontent.com/gianluigilopardo/dataspaces/master/Absenteeism_at_work.csv"
ds = pd.read_csv(url, sep=',')
print("The dataset has %d rows and %d columns." % ds.shape)


# Now that we have the data, let's take a look at it to evaluate how to proceed.

# In[ ]:


ds.head(10)


# Before entering the analysis, it is advisable to rename the columns of the dataset, in order to have more easily traceable variables.
# 
# 
# 

# In[ ]:


# renaming labels
ds = ds.rename(columns = {'Reason for absence': 'reason', 
                          'Month of absence': 'month', 
                          'Day of the week': 'day', 
                          'Transportation expense': 'trans_exp', 
                          'Distance from Residence to Work': 'distance', 
                          'Service time': 'serv_time', 
                          'Work load Average/day ': 'work_load', 
                          'Hit target': 'hit_tg', 
                          'Disciplinary failure': 'disc_fail', 
                          'Social drinker': 'drinker', 
                          'Social smoker': 'smoker', 
                          'Body mass index': 'bmi', 
                          'Absenteeism time in hours': 'abs_hours'})
ds = ds.rename(columns = lambda x: x.lower()) # using lambda function to lowercase labels
ds.head(3)


# Now I check for null values in the dataset:

# In[ ]:


print("There are " + ("some" if ds.isnull().values.any() else "no")  + " null/missing values in the dataset.")


# There are no missing values in the dataset. Now we have to check the invalid data in the dataset. 
# 
# 
# 

# In[ ]:


ds.describe() # summary of data


# Looking at the summary, we immediately see two anomalies: 
# * *month* should be from 1 to 12, but it has at least a 0 value
# * *reason* should be from 1 to 28, but it has at least a 0 value
# 

# In[ ]:


ds[ds['month'] == 0]


# These have to be considered invalid data. I ignore these records.

# In[ ]:


ds = ds[ds['month'] != 0]


# Now let's check *Reason of absence*.

# In[ ]:


len(ds[ds['reason'] == 0]) # occurences with 'reason' = 0.


# It is not a small number. It is necessary to go deeper before decide how to procede.

# In[ ]:


ds[ds['reason'] == 0].describe() # summary of the subset of data having reason = 0


# I see that all the records having *reason = 0* also have *abs_hours = 0*. 
# These records probably refers to cases of late work for less than an hour and so justification is not available. Considering these records into the analysis could be interesting and also useful for a company, so I keep them.

# In[ ]:


ds.describe()


# There is one more attribute to check: *disc_fail*. It is a boolean value.

# In[ ]:


len(ds[ds['disc_fail'] == 1]) # Check how many records have 1 


# In[ ]:


ds[ds['disc_fail'] == 1].describe()


# 40 records have *disc_fail=1*, everyone else has 0.
# We notice that all the records having *Disciplinary failure* correspond to the ones with *abs_hours* = 0. This relationship will certainly be useful in our classification models.
# 
# It could be interesesting to know the distribution of these *disciplinary failure* over the employees and understand if those who received it tending to do less delays or absences. Anyway, we don't have this information.

# I checked for the zeros and the invalid values among the dataset. 
# Now I prepare data for visualization and analysis.

# We see that there are several categorical attributes represented using integer values. In order to manage them, it is opportune to set their type as [Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html). 

# In[ ]:


# Here I will add extended values for categorical data.

# Adding Season names:
season_mapping = {1:'Summer', 2:'Autumn', 3:'Winter', 4:'Spring'}
ds['season_name'] = ds.seasons.map(season_mapping).astype('category')

# Adding Month names abbrevations:
ds['month_name'] =  ds['month'].apply(lambda x: calendar.month_abbr[x])

# Adding day names abbrevations:
ds['day_name'] =  ds['day'].apply(lambda x: calendar.day_abbr[x-2])
# calendar has 0: Monday, but I have 2: Monday

# Adding reasons value:
reason_mapping = {
    0: 'Not available',
    1: 'Certain infectious and parasitic diseases',
    2: 'Neoplasms',
    3: 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
    4: 'Endocrine, nutritional and metabolic diseases',
    5: 'Mental and behavioural disorders',
    6: 'Diseases of the nervous system',
    7: 'Diseases of the eye and adnexa',
    8: 'Diseases of the ear and mastoid process',
    9: 'Diseases of the circulatory system',
    10: 'Diseases of the respiratory system',
    11: 'Diseases of the digestive system',
    12: 'Diseases of the skin and subcutaneous tissue',
    13: 'Diseases of the musculoskeletal system and connective tissue',
    14: 'Diseases of the genitourinary system',
    15: 'Pregnancy, childbirth and the puerperium',
    16: 'Certain conditions originating in the perinatal period',
    17: 'Congenital malformations, deformations and chromosomal abnormalities',
    18: 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    19: 'Injury, poisoning and certain other consequences of external causes',
    20: 'External causes of morbidity and mortality',
    21: 'Factors influencing health status and contact with health services',
    22: 'Patient follow-up',
    23: 'Medical consultation',
    24: 'Blood donation',
    25: 'Laboratory examination',
    26: 'Unjustified absence',
    27: 'Physiotherapy',
    28: 'Dental consultation'
}
ds['reason_text'] = ds['reason'].map(reason_mapping)
ds['reason_text'] = ds['reason_text'].astype('category')

# Adding Education:
education_mapping = {
    1: '1. High School',
    2: '2. Graduate',
    3: '3. Post Graduate',
    4: '4. Master & Doctor'
}
ds['education_detail'] = ds['education'].map(education_mapping)
ds['education_detail'] = ds['education_detail'].astype('category')

# smoker and drinker are boolean
ds['smoker'] = ds['smoker'].astype('bool')
ds['drinker'] = ds['drinker'].astype('bool')
ds['disc_fail'] = ds['disc_fail'].astype('bool')

# Now ds contains categorical data twice.
# I drop the old columns, but before I keep save of everything
# in ds_explore: I use it for data exploration 

ds_explore = ds.copy()
ds = ds.drop(columns = {'reason','month','day','seasons','education'})

# I check the firsts rows:
ds_explore.head()


# ## 3.  <a class="anchor" id="third">Exploratory Data Analysis</a>
# 
# 

# Sometimes simple plotting can help understanding how to proceed. I analyze the distribution of *Absenteeism time in hours* and then we try to undestrand the relationships between the other features.

# #### Distribution of absenteeism

# In[ ]:


# Useful for visualization
def level(absh):
  if(absh < 2):
    lev = 'late'
  elif((absh >= 2) and (absh < 8)):
    lev = 'hours'
  elif(absh >= 8):
    lev = 'days'    
  return lev

ds_explore['abs_lev'] = ds_explore['abs_hours'].apply(lambda x: level(x)).astype('category')


# In[ ]:


# Plotting absenteeism hours 
bins = 40
fig, ax = plt.subplots()
ax.hist(ds_explore['abs_hours'], bins, rwidth=0.8, density = True)
ax.set_xlabel('Absenteeism time in hours')
ax.set_ylabel('Frequency')


# Absenteeism time is highly skewed due to presence of outliers. 

# In[ ]:


# boxplot of Absenteeism time. 
plt.boxplot(ds_explore['abs_hours'], widths = 0.5)
plt.ylabel('Absenteeism time in hours')
plt.xlabel('Boxplot')
plt.show()


# Clearly, in *Abseteeism time in hours* there is a great presence of outliers. It seems there are few observations particularly unusual.

# In[ ]:


# Frequency of Absenteeism time for values grater than a week (40 hours). 
bins = 20
fig, ax = plt.subplots()
out = ds_explore[ds_explore['abs_hours'] > 40]
ax.hist(out['abs_hours'], bins, rwidth = 0.8, density = True)
ax.set_xlabel('Absenteeism time in hours')
ax.set_ylabel('Frequency')


# The dataset is clearly unbalanced: the number of records of the classes can be too different. I will solve this issue with **oversampling** later.
# 
# 

# In[ ]:


# Plotting absence 
time = ds_explore['abs_hours']
hours = np.count_nonzero(time < 8)
days = np.count_nonzero(time >= 8)
x = np.array(['Less than one day', 'One day or more'])
y = np.array([hours, days])
plt.bar(x, y)
plt.show()


# #### Reason and datatime

# A very important feature is *Reason of absence*. Let's analyze which reasons are most related to each level of absence.

# In[ ]:


# Plot of Absenteeism time in hours with Reason for absence. 
sns.catplot(x = 'reason', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 2,
            jitter = '0.25',
            ).set_axis_labels("Reason of absence code", 'Absenteeism in hours')


# Recalling the coding on the reasons for absence reported at the beginning, we note for example that practically all absences whit *reason* = 0 (i.e. Not available) refer to late. We had already noticed this in the Data Preprocessing section. 
# 
# We also notice that the *Reason of absence codes* for *hours* are mainly distributed from 22 to 28, corresponding to no International Code of Diseases (ICD), i.e. not illnesses or insults, but check-ups, medical consultations and others.
# 
# On the contrary, for absences of a day or more, the reason is often a serious medical problem.
# 
# Let's see, for each *Reason*, how records are distribuited among the levels of absence.

# In[ ]:


# Main absence for abs_lev = late
reason_abs_lev = ds_explore.groupby('reason_text')['abs_lev'].value_counts().unstack().fillna(0).astype(int)
reason_abs_lev.sort_values(by = ['late'], ascending = False).head()


# In[ ]:


# Main absence for abs_lev = hours
reason_abs_lev.sort_values(by = ['hours'], ascending = False).head()


# In[ ]:


# Main absence for abs_lev = days
reason_abs_lev.sort_values(by = ['days'], ascending = False).head()


# In[ ]:


# DAY OF THE WEEK
# Distribution of days for level of absence
sns.catplot(x = 'day_name', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 2,
            jitter = '0.25',
            ).set_axis_labels("Day of the week of absence", 'Absenteeism in hours')


# In[ ]:


days_abs_lev = ds_explore.groupby('day_name')['abs_lev'].value_counts().unstack().fillna(0).astype(int)
days_abs_lev


# In[ ]:


# MONTH OF ABSENCE
# Distribution of month for level of absence
sns.catplot(x = 'month_name', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 2,
            jitter = '0.25',
            ).set_axis_labels("Month of absence", 'Absenteeism in hours')


# In[ ]:


# SEASON OF ABSENCE
# Distribution of month for level of absence
sns.catplot(x = 'season_name', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 2,
            jitter = '0.25',
            ).set_axis_labels("Season of absence", 'Absenteeism in hours')


# #### Personal and family

# In[ ]:


# EDUCATION 
sns.catplot(x = 'education_detail', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 1.5).set_axis_labels("Education detail", 'Absenteeism in hours')


# In[ ]:


# Employees per education
ds_explore.groupby('education_detail')['id'].nunique()


# In[ ]:


# SON
sns.catplot(x = 'son', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 1.5).set_axis_labels("Number of sons", 'Absenteeism in hours')


# In[ ]:


# PETS
sns.catplot(x = 'pet', y = 'abs_hours', 
            data = ds_explore,
            height = 4,
            aspect = 1.5).set_axis_labels("Number of pets", 'Absenteeism in hours')


# In[ ]:


# SMOKER
abs_smoker = ds_explore[ds_explore['smoker'] == 0].loc[:,'abs_hours']
sns.kdeplot(abs_smoker, shade = True, label = 'No smoker'
            ).set_title('Distribution of absenteeism for smoker/no smoker')
abs_smoker = ds_explore[ds_explore['smoker'] == 1].loc[:,'abs_hours']            
sns.kdeplot(abs_smoker, shade = True, label = 'Smoker'
            )


# In[ ]:


# DRINKER
abs_drinker = ds_explore[ds_explore['drinker'] == 0].loc[:,'abs_hours']
sns.kdeplot(abs_drinker, shade = True, label = 'No drinker'
            ).set_title('Distribution of absenteeism for drinker/no drinker')
abs_drinker = ds_explore[ds_explore['drinker'] == 1].loc[:,'abs_hours']            
sns.kdeplot(abs_drinker, shade = True, label = 'Drinker'
            )


# ### Correlation

# In[ ]:


# Correlation matrix 
# I drop categorical attributes and I normalize
ds_num = ds.drop(columns = ['id', 'disc_fail', 'drinker', 'smoker', 'season_name', 
                            'month_name', 'day_name', 'reason_text', 'education_detail'])
ds_norm = (ds_num-ds_num.mean())/ds_num.std()
corr = ds_norm.corr()
plt.figure(figsize = (12,12))
sns.heatmap(corr, annot = True, 
            vmin = -1, vmax = 1, center = 0,
            cmap = sns.diverging_palette(20, 220, n = 200))
plt.title('Correlation Heatmap', fontsize = 24)


# 
# Some features are being highlighted in the heatmap. Let's have a look to the more intresting. However, strong correlation among *bmi*, *height* and *weight* and between *distance* and *transport expences* may suggest to remove some feature.
# 
# 1. Strong positive correlation between *Age* and *Service time*.
# 2. Positive correlation between *Service time* and *bmi* (and so *Weight*).
# 3. Negative correlation between *Transport expences* and *Service time*.
# 4. Negative correlation between *Service time* and number of *Pet*.
# 5. Positive correlation between *Transport expense* and number of *pets* and *son*.
# 
# We notice that no feature have strong negative or positive correlation with *Absenteeism time in hours*. We have to go deeper. What is more, clearly in the heatmap they are not considered the categorical data.
# 
# 
# 

# We would like to reduce dimensionality in order to make our algorithm more agile but this has to be made by paying attention to not losing information.
# 
# We have seen that *bmi* and *weight* have a very strong correlation: 0.9. This was expected, considering how it is calculated:
# $$ bmi = \frac{weigth}{height^2} $$
# 
# We can say that we don't lose information by removing one of the two.
# Since also height is in my dataset, I decide to drop *weight*.
# 
# 

# In[ ]:


ds = ds.drop(columns = 'weight')


# ## 4. <a class="anchor" id="fourth">Data preparation</a>

# In order to make **prediction**, I would like to be as specific as possible and therefore to make a **classification** that is well suited to the application case. I want to divide the values of *Absenteeism in hours*, considering the effective time of absence. Considering a work day of 8 hours and I split the set in three labels:
# * *late*: if the absence lasts strictly less than 2 hours
# * *hours*: if the absence was for about less than a work day
# * *days*: if the absence was for one day or more
# 

# In[ ]:


# Plotting absence last
time = ds['abs_hours']
late = np.count_nonzero(time < 2)
days = np.count_nonzero(time >= 8)
hours = np.count_nonzero(time >= 2) - days
x = np.array(['Late', 'Hours', 'Days'])
y = np.array([late, hours, days])
plt.bar(x, y)
plt.show()


# On the other hand, watching at the distribution of the dataset, we can expect that applying such a division the models will not perform at their best.
# 
# We can have better performance (and therefore more safety) using only two classes. I apply **binary classification** and I try to predict absense that last one day ore more.

# In[ ]:


# Plotting absence last
time = ds['abs_hours']
less = np.count_nonzero(time < 8)
more = np.count_nonzero(time >= 8)
x = np.array(['Less than one day', 'One day or more'])
y = np.array([less, more])
plt.bar(x, y)
plt.show()


# In[ ]:


# Create label for ABS_HOURS

# Classes for full classification
def level(absh):
  if(absh < 2):
    lev = 'late'
  elif((absh >= 2) and (absh < 8)):
    lev = 'hours'
  elif(absh >= 8):
    lev = 'days'    
  return lev

# Classes for binary classification
def level_day(absh):
  if(absh < 8):
    day = 'less'
  elif(absh >= 8):
    day = 'more'    
  return day

ds['abs_lev'] = ds['abs_hours'].apply(lambda x: level(x)).astype('category')
ds['abs_day'] = ds['abs_hours'].apply(lambda x: level_day(x)).astype('category')

ds.head()


# In[ ]:


X = ds.drop(columns = ['abs_lev', 'abs_day', 'abs_hours'])
y = ds['abs_lev']
y_day = ds['abs_day']
y_int = ds['abs_hours']


# ### 4.1 <a class="anchor" id="fourth_1">Relabeling</a>
# 

# Apart from the target variabile, the dataset we use has now 10 numerical features and 8 categorical feature. Among the categorical feature, 4 of them are binary and 4 are nominal.
# 
# There is something to do in order to work with the nominal values. The original dataset applies **label encoding**: nominal values are encoded as integer positive values. 
# 
# The problem is that numerical values implies an order, but clearly we can't say that *Summer* is bigger than *Winter* or that *Diseases of the nervous system* is smaller than *Diseases of the respiratory system*. This tecnique can be useful in when applied to values that can be ordedered or have some sort of heriarchy, for example *Education* can be seen as the *Level of education* and so an order could make sense. Anyway, this is not the case for most of the nominal features here.
# 
# To overcame this problem, the most commonly technique used is **one hot encoding**. Each category value is converted into a new column: we assign 1 to the corresponding value and 0 to everything else. 
# 
# Appling it here, this implies adding about 50 new columns: one for each levels of each categorical attribute. In general, this can't be ignored and made straigthforward. 
# 
# I can try reducing dimensionality by removing some of the features. By prevision exploration, I can say that some data can be redundant. In particular, we have seen that the month of absence does not seem to be information particularly linked to the duration of absenteeism. Since this feature involves adding 12 columns and that we already have the feature on the season, I decide not to consider the month in the analysis.
# 
# However, we still have a lot of columns.
# 

# In[ ]:


# label_encoder object knows how to understand word labels. 

Z = X.copy()
label_encoder = preprocessing.LabelEncoder() 

ylabel = label_encoder.fit_transform(y)
levels = label_encoder.inverse_transform(list(set(ylabel)))
ybin = label_encoder.fit_transform(y_day)
bin_levels = label_encoder.inverse_transform(list(set(ybin)))

# ONE HOT ENCODING
# Adding the new columns
#Z = pd.concat([Z,pd.get_dummies(Z['month_name'], prefix = 'month')], axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['reason_text'], prefix = 'reason')], axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['day_name'], prefix = 'day')], axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['season_name'], prefix = 'season')], axis=1)
Z = pd.concat([Z,pd.get_dummies(Z['education_detail'], prefix = 'education')], axis=1)
# Removing the old nominal variables
Z.drop(['reason_text'],axis=1, inplace=True)
Z.drop(['month_name'],axis=1, inplace=True)
Z.drop(['day_name'],axis=1, inplace=True)
Z.drop(['season_name'],axis=1, inplace=True)
Z.drop(['education_detail'],axis=1, inplace=True)
Z.drop(['id'],axis=1, inplace=True)


# ### 4.2 <a class="anchor" id="fourth_2">Principal Component Analysis</a>
# [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) is a statistical procedure that uses an orthogonal transformation to reduce data dimensionality.
# This occurs through a linear transformation of the variables that projects the original ones into a new Cartesian system in which the new variable with the greatest variance is projected on the first axis, the new variable, the second for the size of the variance, on the second axis and so on.

# In[ ]:


X_pca = Z.copy()
# Normalize:
X_pca_norm = (X_pca-X_pca.mean())/X_pca.std()
print("There are " + ("some" if X_pca_norm.isnull().values.any() else "no")  + " null/missing values in the dataset.")


# In[ ]:


# calculate the principal components
X_pca = PCA(random_state = SEED).fit(X_pca_norm)
cumvar = np.cumsum(X_pca.explained_variance_ratio_)
#Plotting cumulative variance
plt.plot(cumvar)
plt.title('Cumulative variance')
plt.xlabel('Number of components')
plt.ylabel('Variance explained')


# In[ ]:


cumvar[36]


# The cumulative variance plot show us that we can cover about the 90% of the variance by using the first 36 principal components. I decide to use them.

# In[ ]:


n_used = 36
X_pca = np.dot(X_pca_norm.values, X_pca.components_[:n_used,:].T)
X_pca = pd.DataFrame(X_pca, columns=["PC%d" % (x + 1) for x in range(n_used)])
X_pca.head()


# Now I split the set in two subsets:
# * **Training set**: collection of labelled data objects used to learn
# the classification model
# * **Test set**: Collection of labelled data objects used to
# validate the classification model
# 
# I use 70% of data as training set and 30% for tests.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_pca, ylabel, test_size = 0.3, random_state = SEED, stratify = ylabel)
Xbin_train, Xbin_test, ybin_train, ybin_test = train_test_split(X_pca, ybin, test_size = 0.3, random_state = SEED, stratify = ybin)


# ### 4.3 <a class="anchor" id="fourth_3">Oversampling</a>
# As discussed before, in the classification setting the dataset is heavily skewed towards low level of absenteeism and our classes are strongly unbalanced. We can solve this by oversamplig. 
# 
# There are three common techniques to do this:
# 
# 1. The simplest is **Random oversampling**: it simply generates randomly new samples in the classes which are under-represented. 
# 
# 2. **SMOTE** (Synthetic Minority Oversampling Technique): take a sample from the dataset, and consider its $k$ nearest neighbors (in feature space). To create a synthetic data point, take the vector between one of those $k$ neighbors, and the current data point. Multiply this vector by a random number $x$ which lies between $0$, and $1$. Add this to the current data point to create the new, synthetic data point.
# 
# 3. **ADASYN** (ADAptive SYNthetic sampling approach) algorithm, builds on the methodology of SMOTE, by shifting the importance of the classification boundary to those minority classes which are difficult. ADASYN uses a weighted distribution for different minority class examples according to their level of difficulty in learning, where more synthetic data is generated for minority class examples that are harder to learn.

# In[ ]:


# GENERATING NEW DATASET WITH RANDOM OVERSAMPLING, SMOTE and ADASYN OVERSAMPLING
X_train_ROS, y_train_ROS = RandomOverSampler(random_state = SEED).fit_resample(X_train, y_train)
X_train_SMOTE, y_train_SMOTE = SMOTE(random_state = SEED).fit_resample(X_train, y_train)
X_train_ADASYN, y_train_ADASYN = ADASYN(random_state = SEED, sampling_strategy = 'not majority').fit_resample(X_train, y_train)


# In[ ]:


y_set = {'simple': y_train, 'ROS': y_train_ROS, 'SMOTE': y_train_SMOTE, 'ADASYN': y_train_ADASYN}
for ys in y_set.keys():
  print('The', ys, 'train set has:')
  for i in set(ylabel):    
          print('\t', np.count_nonzero(y_set[ys] == i), 'records with label',levels[i])


# ## 5. <a class="anchor" id="fifth">Classification Models</a>
# Classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known.
# 
# 

# 
# Here I apply the most commons classification algorithms and I will compare their performance using diffent metrics:
# 
# Let: $TP$ = True positives, $TN$ = True negative, $FP$ = False positive, $FN$ = False negative.
# 
# 1. **Accuracy** = $\frac{TP + TN}{TP+TN + FP + FN} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$
# 
# 2. **Precision** = $\frac{\text{TP}}{\text{TP + FP}}$
# 
# 3. **Sensitivity** = $\frac{\text{TP}}{\text{TP + FN}}$
# 
# 4. **Specificity** = $\frac{\text{TN}}{\text{TN + FP}}$
# 
# 5. **F1 Score** = $2 \cdot \frac{\text{Precision}\cdot \text{Sensitivity}}{\text{Precision + Sensitivity}} $
# 
# All of these can be easily generalized to configurations with more than two classes by using weighted versions.
# 
# Great part of the work was to identify limits and peculiarities of different metrics. I also use **confusion matrix** as a measure of performance.

# ### 5.1 <a class="anchor" id="fifth_1">Full classification</a>

# Here I try to predicts absenteeism levels using three classes: *late*, *hours*, *days*. Remember that these classes are strongly unbalanced as they have been simply chosen by considering the application.

# I have to choose which of my datasets to use for classification. I decide to evaluate all of them using the random forest classifier and then I select the one which performs better.
# 
# 

# In[ ]:


# Random forest Classifier

#Simple dataset
simple_forest = RandomForestClassifier(random_state = SEED, max_depth = 10).fit(X_train, y_train)
simple_pred = simple_forest.predict(X_test)

#Random oversampler
ros_forest = RandomForestClassifier(random_state = SEED, max_depth = 10).fit(X_train_ROS, y_train_ROS)
random_over_pred = ros_forest.predict(X_test)

#SMOTE
SMOTE_forest = RandomForestClassifier(random_state = SEED, max_depth = 10).fit(X_train_SMOTE, y_train_SMOTE)
SMOTE_pred = SMOTE_forest.predict(X_test)

#ADASYN 
ADASYN_forest = RandomForestClassifier(random_state = SEED, max_depth = 10).fit(X_train_ADASYN, y_train_ADASYN)
ADASYN_pred = ADASYN_forest.predict(X_test)

#Performance 

#Simple data_set metrics: 
simple_pred_accuracy = accuracy_score(y_test, simple_pred)
simple_pred_precision = precision_score(y_test, simple_pred, average = 'weighted')
simple_pred_sensitivity = recall_score(y_test, simple_pred, average = 'weighted')
simple_pred_f1 = f1_score(y_test, simple_pred, average = 'weighted')

#Random oversampling metrics:
rnd_sampler_accuracy = accuracy_score(y_test, random_over_pred)
rnd_sampler_precision = precision_score(y_test,random_over_pred, average = 'weighted')
rnd_sampler_sensitivity = recall_score(y_test,random_over_pred, average = 'weighted')
rnd_sampler_f1 = f1_score(y_test,random_over_pred, average = 'weighted')

#SMOTE metrics: 
SMOTE_accuracy = accuracy_score(y_test, SMOTE_pred)
SMOTE_precision = precision_score(y_test, SMOTE_pred, average = 'weighted')
SMOTE_sensitivity = recall_score(y_test, SMOTE_pred, average = 'weighted')
SMOTE_f1 = f1_score(y_test, SMOTE_pred, average = 'weighted')

#ADASYN metrics:
ADASYN_accuracy = accuracy_score(y_test, ADASYN_pred)
ADASYN_precision = precision_score(y_test, ADASYN_pred, average = 'weighted')
ADASYN_sensitivity = recall_score(y_test, ADASYN_pred, average = 'weighted')
ADASYN_f1 = f1_score(y_test, ADASYN_pred, average = 'weighted')

# metrics
metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Sensitivity", "F1 Score"])
metrics.loc["Simple Dataset"] = [simple_pred_accuracy,simple_pred_precision,simple_pred_sensitivity,simple_pred_f1]
metrics.loc["Rnd oversampling"] = [rnd_sampler_accuracy,rnd_sampler_precision,rnd_sampler_sensitivity,rnd_sampler_f1]
metrics.loc["SMOTE"] = [SMOTE_accuracy,SMOTE_precision,SMOTE_sensitivity,SMOTE_f1]
metrics.loc["ADASYN"] = [ADASYN_accuracy,ADASYN_precision,ADASYN_sensitivity,ADASYN_f1]
metrics


# In[ ]:


fig, ((simple, ros), (smote, adasyn)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))

plot_confusion_matrix(simple_forest, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true', 
                      ax = simple)
plot_confusion_matrix(ros_forest, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true',
                      ax = ros)
plot_confusion_matrix(SMOTE_forest, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true',
                      ax = smote)
plot_confusion_matrix(ADASYN_forest, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true',
                      ax = adasyn)
simple.set_title('Simple dataset')
ros.set_title('ROS dataset')
smote.set_title('SMOTE dataset')
adasyn.set_title('ADASYN dataset')


# Complexively, the best seems to be the SMOTE set: I will use it.

# In[ ]:


X_val = X_train
y_val = y_train
X_train = X_train_SMOTE
y_train = y_train_SMOTE


# In[ ]:


classification_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Sensitivity", "F1 Score"])


# #### <a class="anchor" id="fifth_1">Logitistic regression</a>
# Logistic regression is a statistical model used to model the probability of a certain class or event existing.
# 
# If classes are binary, then we can compute the probability that a certain sample belongs to the cateogy $Y = 1$: 
# 
# $p(x) = \mathbb{P}(Y=1\vert X)$
# It has distribution: $$p(x) = \frac{e^{\beta_0 + \beta_1x}}{1+e^{\beta_0 + \beta_1x}}$$
# 
# It can been generalized for $p$ variables: $$p(x) = \frac{e^{\beta_0 + \beta_1x + \dots + \beta_px_p}}{1+e^{\beta_0 + \beta_1x + \dots + \beta_px_p}}$$

# In[ ]:


logistic = LogisticRegression(max_iter = 10000, random_state = SEED, solver = 'lbfgs').fit(X_train, y_train)
logistic_predict = logistic.predict(X_test)

#metrics:
logi_acc = accuracy_score(logistic_predict, y_test)
logi_preci = precision_score(logistic_predict, y_test, average = 'weighted')
logi_sensitivity = recall_score(logistic_predict, y_test, average = 'weighted')
logi_f1 = f1_score(logistic_predict, y_test, average = 'weighted')

classification_metrics.loc["Logistic regression"] = [logi_acc,logi_preci,logi_sensitivity,logi_f1]
plot_confusion_matrix(logistic, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true')


# #### <a class="anchor" id="fifth_2">Decision Tree</a> 
# A decision tree is a very common model for classification problems (it is also called a classification tree). In this context a decision tree describes a tree structure where the leaf nodes represent the classifications and the ramifications the set of properties that lead to those classifications. Consequently, each internal node is a macro-class consisting of the union of the classes associated with its child nodes.
# 
# In many situations it is useful to define a halting criterion, or even pruning criterion in order to determine its maximum depth. This is because the growing depth of a tree (or its size) does not directly affect the goodness of the model. Indeed, an excessive growth in the size of the tree could only lead to a disproportionate increase in computational complexity compared to the benefits regarding the accuracy of the predictions/classifications.
# 
# The algorithm chooses a variable at each step that best splits the set of items, according to certain metrics.
# 
# The parameters that are mostly used to guide the construction of the tree are the Gini index and the Entropy deviance.
# 
# The Gini Index is computed as:$$ GI(t) = 1-\sum_{j=1}^k p(j|t)^2$$
# The Entropy deviance is computed as:$$ Entropy(t) = -\sum_{j=1}^k p(j|t)log_2(p(j|t))$$
# $\forall \text{ node } t$, where $p(j|t)$ is the relative frequency of class $j$ at node $t$.
# 

# In[ ]:


# Evaluating best parameters
depth = np.arange(4, 20) 
parameters = {'max_depth': depth}
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion = 'entropy', random_state = SEED), parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_depth = clf.best_params_['max_depth']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))


# In[ ]:


# Decision TREE: Entropy
tree_Entropy = tree.DecisionTreeClassifier(max_depth = opt_depth, criterion = 'entropy', random_state = SEED)
tree_Entropy = tree_Entropy.fit(X_train, y_train)
tree_Entropy_pred = tree_Entropy.predict(X_test)

#metrics
tree_acc = accuracy_score(tree_Entropy_pred, y_test)
tree_preci = precision_score(tree_Entropy_pred, y_test, average = 'weighted')
tree_sensitivity = recall_score(tree_Entropy_pred, y_test, average = 'weighted')
tree_f1 = f1_score(tree_Entropy_pred, y_test, average = 'weighted')

classification_metrics.loc["Tree Entropy"] = [tree_acc, tree_preci, tree_sensitivity, tree_f1]
plot_confusion_matrix(tree_Entropy, X_test, y_test, 
                      display_labels = levels, cmap = plt.cm.Blues,

                      values_format = '0.2%', normalize = 'true')


# In[ ]:


# Evaluating best parameters
depth = np.arange(4, 20) 
parameters = {'max_depth': depth}
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion = 'gini', random_state = SEED), parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_depth = clf.best_params_['max_depth']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))


# In[ ]:


# Decision TREE: Gini
tree_Gini = tree.DecisionTreeClassifier(max_depth = opt_depth, criterion = 'gini', random_state = SEED)
tree_Gini = tree_Gini.fit(X_train, y_train)
tree_Gini_pred = tree_Gini.predict(X_test)

#metrics
tree_acc = accuracy_score(tree_Gini_pred, y_test)
tree_preci = precision_score(tree_Gini_pred, y_test, average = 'weighted')
tree_sensitivity = recall_score(tree_Gini_pred, y_test, average = 'weighted')
tree_f1 = f1_score(tree_Gini_pred, y_test, average = 'weighted')

classification_metrics.loc["Tree Gini"] = [tree_acc, tree_preci, tree_sensitivity, tree_f1]
plot_confusion_matrix(tree_Gini, X_test, y_test, display_labels = levels, cmap = plt.cm.Blues, 
                      values_format = '0.2%',  normalize = 'true')


# #### <a class="anchor" id="fifth_3">Random forest</a> 
# 
# As the name may suggest, the Random Forest classifier is obtained by parallelly using several decision tree at the same time.
# 
# The general idea is to fit a (reasonably) large number of decision tree to different subsample and then let the majority to decide the label. Random forests are a solution that minimizes the overfitting of the training set compared to decision trees.
# 
# As for the Decision tree, we will use either the Gini and the Entropy criterion.

# In[ ]:


# Evaluating best parameters
depth = np.arange(4, 20) 
parameters = {'max_depth': depth}
clf = GridSearchCV(RandomForestClassifier(criterion = 'entropy', n_estimators = 200, random_state = SEED,
                                          ), parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_depth = clf.best_params_['max_depth']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))


# In[ ]:


# Random forest - Entropy criterion
forest_Entropy = RandomForestClassifier(max_depth = opt_depth, criterion = 'entropy', n_estimators = 200, random_state = SEED)
forest_Entropy = forest_Entropy.fit(X_train, y_train)
forest_Entropy_pred = forest_Entropy.predict(X_test)

#metrics
forest_acc = accuracy_score(forest_Entropy_pred, y_test)
forest_preci = precision_score(forest_Entropy_pred, y_test, average = 'weighted')
forest_sensitivity = recall_score(forest_Entropy_pred, y_test, average = 'weighted')
forest_f1 = f1_score(forest_Entropy_pred, y_test, average = 'weighted')

classification_metrics.loc["Forest Entropy"] = [forest_acc,forest_preci,forest_sensitivity,forest_f1]
plot_confusion_matrix(forest_Entropy, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true')


# In[ ]:


# Evaluating best parameters
depth = np.arange(4, 20) 
parameters = {'max_depth': depth}
clf = GridSearchCV(RandomForestClassifier(criterion = 'gini', n_estimators = 200, random_state = SEED,
                                          ), parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_depth = clf.best_params_['max_depth']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))


# In[ ]:


# Random forest - Gini criterion
forest_Gini = RandomForestClassifier(max_depth = opt_depth, criterion = 'gini', n_estimators = 200, random_state = SEED)
forest_Gini = forest_Gini.fit(X_train, y_train)
forest_Gini_pred = forest_Gini.predict(X_test)

#metrics
forest_acc = accuracy_score(forest_Gini_pred, y_test)
forest_preci = precision_score(forest_Gini_pred, y_test, average = 'weighted')
forest_sensitivity = recall_score(forest_Gini_pred, y_test, average = 'weighted')
forest_f1 = f1_score(forest_Gini_pred, y_test, average = 'weighted')

classification_metrics.loc["Forest Gini"] = [forest_acc,forest_preci,forest_sensitivity,forest_f1]
plot_confusion_matrix(forest_Gini, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true')


# #### <a class="anchor" id="fifth_4">K Nearest neighbours</a> 

# The K-nearest neighbors (KNN) is an algorithm used in the recognition of patterns for classifying objects based on the characteristics of the objects close to the one considered. It is one the simplest algorithms among those used in machine learning. A new point is categorized based on the similiraties of the $K$ data points that are the closest to it. The choice of the parameter $K$ depends on the characteristics of the data. Generally, as $K$ increases, the noise that compromises the classification is reduced, but the criterion of choice for the class becomes more rough. The choice can be made through heuristic techniques.

# In[ ]:


# Evaluating best parameters
neigh = np.arange(2, 20)
parameters = {'n_neighbors': neigh}
clf = GridSearchCV(KNeighborsClassifier(), parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_neigh = clf.best_params_['n_neighbors']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))


# In[ ]:


# KNN
knn = KNeighborsClassifier(opt_neigh)
knn = knn.fit(X_train, y_train)
neigh_predict = knn.predict(X_test)

#metrics:
neigh_acc = accuracy_score(neigh_predict, y_test)
neigh_preci = precision_score(neigh_predict, y_test, average = 'weighted')
neigh_sensitivity = recall_score(neigh_predict, y_test, average = 'weighted')
neigh_f1 = f1_score(neigh_predict, y_test, average = 'weighted')

classification_metrics.loc["KNN"] = [neigh_acc,neigh_preci,neigh_sensitivity,neigh_f1]
plot_confusion_matrix(knn, X_test, y_test, display_labels = levels,
                      cmap = plt.cm.Blues, values_format = '0.2%', normalize = 'true')


# #### <a class="anchor" id="fifth_5">Support vector machine</a> 

# The basic Support-Vector Machine (SVM) model is linear: it finds the optimal hyperplane between the points of two classes such that the distance of the nearest points to the decision boundary is maximized.
# 
# Clearly, it is not common to have a linear separation and SVM can be generalized.
# 
# This generalization can be made by choosing a non-linear Kernel function to apply to the support vector machine.
# 
# Linear support vector machine generally uses as linear kernel:$$ k({\vec{x_i}},{\vec{x_{j}}})={\vec{x_{i}}}\cdot{\vec{x_{j}}}$$
# 
# Probably the common non-lineare alternative is the Radial basis kernel:$${k({\vec{x_i},\vec{x_{j}}})=\exp(-\gamma \|{\vec {x_{i}}}-{\vec {x_{j}}}\|^{2})}$$where ${\displaystyle \gamma >0}$ is a parameter to chose. This is in fact the default kernel function in the library.
# 
# Basically, the kernel maps the data into another space in which the class can be linearly separated, while in the original space, the boundary will in general be non linear.

# In[ ]:


# Evaluating best parameters
Cs = np.logspace(-1, 1, 10)
parameters = {'C': Cs}
clf = GridSearchCV(svm.SVC(kernel = "rbf", gamma = 'auto', random_state = SEED), 
                       parameters, scoring = 'f1_weighted')
clf = clf.fit(X_val, y_val)
opt_C = clf.best_params_['C']
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))


# In[ ]:


# SVM
svm_mod = svm.SVC(C = opt_C, kernel = "rbf", gamma = 'auto', random_state = SEED)
svm_mod = svm_mod.fit(X_train, y_train)
svm_mod_predict = svm_mod.predict(X_test)

#metrics:
svm_acc = accuracy_score(svm_mod_predict, y_test)
svm_preci = precision_score(svm_mod_predict, y_test, average = 'weighted')
svm_sensitivity = recall_score(svm_mod_predict, y_test, average = 'weighted')
svm_f1 = f1_score(svm_mod_predict, y_test, average = 'weighted')

classification_metrics.loc["SVM"] = [svm_acc,svm_preci,svm_sensitivity,svm_f1]
plot_confusion_matrix(svm_mod, X_test, y_test, display_labels = levels, cmap = plt.cm.Blues,
                      values_format = '0.2%', normalize = 'true')


# #### <a class="anchor" id="fifth_6">Model Comparison</a> 

# In[ ]:


classification_metrics.sort_values('F1 Score')


# Overall, considering the three classes and the unbalanced sets, we can conclude that the initial work on preprocessing, labeling and oversampling of data has paid off.
# 
# Using SMOTE-based data, classification algorithms work pretty well.
# Forests have the best performances, while trees are the worsts. KNN and SVM also work in a good way, as they are over the 70% for all metrics.
# 
# We also see that Entropy criterion performs better than the Gini one for both forests and trees, even if only slightly.

# ### 5.2 <a class="anchor" id="fifth_2">Binary classification</a>

# Now we use a two-classes target, trying to predict if the absence will last a day or more.

# In[ ]:


binary_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Sensitivity", "F1 Score"])

# Logistic regression
logistic = LogisticRegression(max_iter = 10000, random_state = SEED, solver = 'lbfgs')
logistic.fit(Xbin_train, ybin_train)
logistic_predict = logistic.predict(Xbin_test)
logi_acc = accuracy_score(logistic_predict, ybin_test)
logi_preci = precision_score(logistic_predict, ybin_test)
logi_sensitivity = recall_score(logistic_predict, ybin_test)
logi_f1 = f1_score(logistic_predict, ybin_test)
binary_metrics.loc["Logistic regression"] = [logi_acc,logi_preci,logi_sensitivity,logi_f1]

depth = np.arange(4, 20) # for trees and forests
parameters = {'max_depth': depth}
# Decision Tree
## Entropy criterion
tree_Entropy = GridSearchCV(tree.DecisionTreeClassifier(criterion = 'entropy', random_state = SEED), parameters, scoring = 'f1')
tree_Entropy = tree_Entropy.fit(Xbin_train, ybin_train)
tree_Entropy_pred = tree_Entropy.predict(Xbin_test)
tree_acc = accuracy_score(tree_Entropy_pred, ybin_test)
tree_preci = precision_score(tree_Entropy_pred, ybin_test)
tree_sensitivity = recall_score(tree_Entropy_pred, ybin_test)
tree_f1 = f1_score(tree_Entropy_pred, ybin_test)
binary_metrics.loc["Tree Entropy"] = [tree_acc, tree_preci, tree_sensitivity, tree_f1]
## Gini criterion
tree_Gini = GridSearchCV(tree.DecisionTreeClassifier(criterion = 'gini', random_state = SEED), parameters, scoring = 'f1')
tree_Gini = tree_Gini.fit(Xbin_train, ybin_train)
tree_Gini_pred = tree_Gini.predict(Xbin_test)
tree_acc = accuracy_score(tree_Gini_pred, ybin_test)
tree_preci = precision_score(tree_Gini_pred, ybin_test)
tree_sensitivity = recall_score(tree_Gini_pred, ybin_test)
tree_f1 = f1_score(tree_Gini_pred, ybin_test)
binary_metrics.loc["Tree Gini"] = [tree_acc, tree_preci, tree_sensitivity, tree_f1]

# Random forest
parameters = {'max_depth': depth}
## Entropy criterion
forest_Entropy = GridSearchCV(RandomForestClassifier(n_estimators = 100, criterion = 'entropy', 
                                                     random_state = SEED), parameters, scoring = 'f1')
forest_Entropy = forest_Entropy.fit(Xbin_train, ybin_train)
forest_Entropy_pred = forest_Entropy.predict(Xbin_test)
forest_acc = accuracy_score(forest_Entropy_pred, ybin_test)
forest_preci = precision_score(forest_Entropy_pred, ybin_test)
forest_sensitivity = recall_score(forest_Entropy_pred, ybin_test)
forest_f1 = f1_score(forest_Entropy_pred, ybin_test)
binary_metrics.loc["Forest Entropy"] = [forest_acc,forest_preci,forest_sensitivity,forest_f1]
## Gini criterion
forest_Gini = GridSearchCV(RandomForestClassifier(n_estimators = 100, criterion = 'gini', 
                                                  random_state = SEED), parameters, scoring = 'f1')
forest_Gini = forest_Gini.fit(Xbin_train, ybin_train)
forest_Gini_pred = forest_Gini.predict(Xbin_test)
forest_acc = accuracy_score(forest_Gini_pred, ybin_test)
forest_preci = precision_score(forest_Gini_pred, ybin_test)
forest_sensitivity = recall_score(forest_Gini_pred, ybin_test)
forest_f1 = f1_score(forest_Gini_pred, ybin_test)
binary_metrics.loc["Forest Gini"] = [forest_acc,forest_preci,forest_sensitivity,forest_f1]

# KNN
neigh = np.arange(2, 20)
parameters = {'n_neighbors': neigh}
knn = GridSearchCV(KNeighborsClassifier(), parameters, scoring = 'f1')
knn = knn.fit(Xbin_train, ybin_train)
knn_predict = knn.predict(Xbin_test)
neigh_acc = accuracy_score(knn_predict, ybin_test)
neigh_preci = precision_score(knn_predict, ybin_test)
neigh_sensitivity = recall_score(knn_predict, ybin_test)
neigh_f1 = f1_score(knn_predict, ybin_test)
binary_metrics.loc["KNN"] = [neigh_acc,neigh_preci,neigh_sensitivity,neigh_f1]

# SVM
Cs = np.logspace(-2, 2, 8)
parameters = {'C': Cs}
svm_mod = GridSearchCV(svm.SVC(gamma = 'auto', kernel = "rbf", random_state = SEED), parameters, scoring = 'f1')
svm_mod = svm_mod.fit(Xbin_train, ybin_train)
svm_mod_predict = svm_mod.predict(Xbin_test)
svm_acc = accuracy_score(svm_mod_predict, ybin_test)
svm_preci = precision_score(svm_mod_predict, ybin_test)
svm_sensitivity = recall_score(svm_mod_predict, ybin_test)
svm_f1 = f1_score(svm_mod_predict, ybin_test)
binary_metrics.loc["SVM"] = [svm_acc,svm_preci,svm_sensitivity,svm_f1]

# MODEL COMPARISON
binary_metrics.sort_values('F1 Score')


# By using this configuration settings, all models perform very well.
# We see that with this choice of parameters we have a high level of precision and accuracy, while sensitivity remains somewhat lower. It is often appropiate to use *F1 Score* as a comparing measure, as it is a composed metrics that consider all of the other metrics. 
# 
# In general, we can manage a trade off between these metrics: we could choose the classifier parameters in order to privilege one of that, depending on whether we prefer to have false positives or false negatives. For example, a company may wish to take into account the risk of someone being absent even if this is very low. All of this can be easily performed by considering it in the validation step.

# ## 6. <a anchor='sixth'>References</a>

# 
# * Martiniano, A., Ferreira, R. P., Sassi, R. J., & Affonso, C. (2012). Application of a neuro fuzzy network in prediction of absenteeism at work. In Information Systems and Technologies (CISTI), 7th Iberian Conference on (pp. 1-4). IEEE.
# * An Introduction to Statistical Learning with application in R, Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani
# * [Performance Metrics for Classification problems in Machine Learning
# ](https://medium.com/thalus-ai/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)
# * [Credit Fraud || Dealing with Imbalanced Datasets
# ](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
# * [SMOTE and ADASYN (Handling Imbalanced Data Set)
# ](https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167)
# * [Oversampling - imbalanced-learn documentation](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html)
# 
# 
# 
