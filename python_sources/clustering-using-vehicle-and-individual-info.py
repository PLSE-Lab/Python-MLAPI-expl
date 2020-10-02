#!/usr/bin/env python
# coding: utf-8

# # In this Kernel I will conduct a rather simplified EDA at first, then cluster and predict the following aspects:
# 
# * Gender (Male or Female)
# * Injury Severity (Uninjured, Minor, Moderate...)
# * Age (in groupes of 10 years - 20-30 / 30-40 / 40-50 etc..)
# 
# Changelist commits -
# * 16.09 - First EDA & data cleaning
# * 22.09 - Clustering - First M/F, Second - Injury Severity !
# * 23.09 - Clustering second part - Injury Severity !
# * 26.10 - More complicated clustering - ADA Boost and Ensemble used !
# 
# Next - 
# * 27.10 - Age of the driver (in term of groups of 10 years)!

# First - some neccesary imports...

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder

print(os.listdir("../input/nys-motor-vehicle-crashes-and-insurance-reduction"))


# Loading the 2 relevant CSVs that could be read and compared hopefully via Vehicle ID

# In[ ]:


df_vehilce_information = pd.read_csv('../input/nys-motor-vehicle-crashes-and-insurance-reduction/motor-vehicle-crashes-vehicle-information-three-year-window.csv')
df_individual_information = pd.read_csv('../input/nys-motor-vehicle-crashes-and-insurance-reduction/motor-vehicle-crashes-individual-information-three-year-window.csv')

Dropping some useless columns..
# In[ ]:


df_vehilce_information = df_vehilce_information.drop(columns=['Type / Axles of Truck or Bus', 'Direction of Travel', 'Fuel Type', 'State of Registration', 'Engine Cylinders', 'Contributing Factor 1', 'Contributing Factor 1 Description', 'Contributing Factor 2', 'Contributing Factor 2 Description', 'Event Type', 'Partial VIN'])
df_vehilce_information.head()


# In[ ]:


df_vehilce_information.shape


# In[ ]:


df_individual_information = df_individual_information.drop(columns=['Case Individual ID', 'Victim Status', 'License State Code', 'Transported By', 'Injury Descriptor', 'Injury Location'])
df_individual_information.head()


# In[ ]:


df_individual_information.shape

Merging the two datasets using the Case Vehicle ID as the main Matching column and Year as cross Validation:
# In[ ]:


df_combined = pd.merge(df_individual_information, df_vehilce_information, on='Case Vehicle ID', how='left')
print(df_combined.shape)
df_combined.head()


# Validation of matching Case Vehicle IDs via comparing Year_x and Year_y

# In[ ]:


len(df_combined[df_combined.Year_x == df_combined.Year_y]['Case Vehicle ID']) / len(df_combined['Case Vehicle ID'])

Perfect! Almost 100% identical matching! We can drop the unmatched and the additional Year
# In[ ]:


df_combined = df_combined[df_combined.Year_x == df_combined.Year_y]
df_combined = df_combined.drop(columns=['Year_y'])
df_combined.info(null_counts=True)


# Since we have only around 2M data rows, lets not discard at this stage the existing Nans;
# Quick check for duplicates:

# In[ ]:


df_combined_no_dup = df_combined.drop_duplicates(subset='Case Vehicle ID').copy()
print(df_combined.shape)
df_combined_no_dup.shape


# We have to understand why are the duplicates! Lets inspect a few:

# In[ ]:


df_combined.sort_values(by='Case Vehicle ID').head(20)


# We see a straight and a logical corelation between the number of occupants and the amount of Case Vehicle IDs that appear.
# Now lets drop cases where all the column values are the same...

# In[ ]:


df_combined = df_combined.drop_duplicates()
df_combined.shape


# # EDA - first stage is complete ! 
# (for every question we will add more EDA)

# In[ ]:


df_combined_drivers = df_combined[df_combined['Seating Position'] == 'Driver']
df_combined_motor_drivers = df_combined_drivers[df_combined_drivers['Role Type'] == 'Driver of a Motor Vehicle in Transport']
df_combined_drivers_m_or_f = df_combined_motor_drivers[df_combined_motor_drivers.Sex != 'U']
print(df_combined_drivers.shape)
print(df_combined_motor_drivers.shape)
print(df_combined_drivers_m_or_f.shape)
df_combined_drivers_m_or_f.head()


# A bit more EDA - Lets select and remove from the data clear outliers for each column - 
# 1. Year_x - None;	
# 2. Case Vehicle ID - None;
# 3. Role Type - Driver of a Motor Vehicle in Transport - only!;
# 4. Seating Position - Driver - only!;
# 5. Ejection	- 49K Unknown - we need to decide what to do with those (3 categories - Not Ejected, Ejected, Partially Ejected)
# 6. Sex - M or F only!
# 7. Safety Equipment	- 129k Unknown - we need to decide what to do with those (15 categories..)
# 8. Injury Severity - all valid (6 categories)
# 9. Age - No outliers (boxplot)
# 10. Vehicle Body Type - 27k unknown + way to many categories...(62 categories..)
# 11. Registration Class - 177K unknown... (69 categories..)
# 12. Action Prior to Accident - 61k unknown.. (22 categories..)	
# 13. Vehicle Year - (boxplot), 11.5K outliers below 1993	
# 14. Number of Occupants	- (boxplot), lets not include anything above normal buses (60 passengers max) = 29 vehicles excluded.. 
# 15. Vehicle Make - 1904 different makes! also has duplicate makes with diff names...

# In[ ]:


# print(df_combined_drivers_m_or_f.Ejection.value_counts())

# print(df_combined_drivers_m_or_f['Safety Equipment'].value_counts())
# print(df_combined_drivers_m_or_f['Safety Equipment'].nunique())

# print(df_combined_drivers_m_or_f['Injury Severity'].value_counts())

plt.figure(figsize=(3,6))
sns.boxplot(y='Age', data=df_combined_drivers_m_or_f);

# print(df_combined_drivers_m_or_f['Vehicle Body Type'].value_counts().head(20))
# print(df_combined_drivers_m_or_f['Vehicle Body Type'].nunique())

# print(df_combined_drivers_m_or_f['Registration Class'].value_counts())
# print(df_combined_drivers_m_or_f['Registration Class'].nunique())

# print(df_combined_drivers_m_or_f['Action Prior to Accident'].value_counts())
# print(df_combined_drivers_m_or_f['Action Prior to Accident'].nunique())

plt.figure(figsize=(3,6))
sns.boxplot(y='Vehicle Year', data=df_combined_drivers_m_or_f);

# print(df_combined_drivers_m_or_f['Vehicle Year'].quantile(.01))
# print(df_combined_drivers_m_or_f[df_combined_drivers_m_or_f['Vehicle Year'] < 1993].shape)

plt.figure(figsize=(3,6))
sns.boxplot(y='Number of Occupants', data=df_combined_drivers_m_or_f);
# print(df_combined_drivers_m_or_f['Number of Occupants'].quantile(.9999))
# print(df_combined_drivers_m_or_f[df_combined_drivers_m_or_f['Number of Occupants'] > 60].shape)

# print(df_combined_drivers_m_or_f['Vehicle Make'].value_counts().head(50))
# print(df_combined_drivers_m_or_f['Vehicle Make'].nunique())


# Concluding for the F or M classification, lets choose the following features (based on simplification and logic at this stage): 
# 1. Year_x - not-relevant - not-used	
# 2. Case Vehicle ID - not-relevant - not-used	
# 3. Role Type - Driver of a Motor Vehicle in Transport - only!
# 4. Seating Position - Driver - only!
# 5. Ejection	- 49K Unknown - we need to decide what to do with those (3 categories - Not Ejected, Ejected, Partially Ejected) - use 
# 6. Sex - M or F only!
# 7. Safety Equipment	- 129k Unknown - we need to decide what to do with those (15 categories..) - not-used for simplicity for now 
# 8. Injury Severity - all valid (6 categories) - use
# 9. Age - No outliers (boxplot) - use
# 10. Vehicle Body Type - 27k unknown + way to many categories...(62 categories..) - not-used for simplicity for now
# 11. Registration Class - 177K unknown... (69 categories..) - not-used for simplicity for now
# 12. Action Prior to Accident - 61k unknown.. (22 categories..) - not-used for simplicity for now	
# 13. Vehicle Year - (boxplot), 11.5K outliers below 1993	- use
# 14. Number of Occupants	- (boxplot), not include anything above 60 passengers max => 29 vehicles excluded - use
# 15. Vehicle Make - 1904 different makes! also has duplicate makes with diff names... - to many categories  - not-used

# In[ ]:


df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f[['Ejection', 'Sex', 'Injury Severity', 'Age', 'Vehicle Year', 'Number of Occupants']]


# In[ ]:


print(df_combined_drivers_m_or_f_clustering.shape)
df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering[df_combined_drivers_m_or_f_clustering['Ejection'].isin(['Not Ejected', 'Ejected', 'Partially Ejected'])]
print(df_combined_drivers_m_or_f_clustering.shape)
df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering[df_combined_drivers_m_or_f_clustering['Number of Occupants'] < 60]
print(df_combined_drivers_m_or_f_clustering.shape)
df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering[df_combined_drivers_m_or_f_clustering['Vehicle Year'] > 1993]
print(df_combined_drivers_m_or_f_clustering.shape)


# In[ ]:


df_combined_drivers_m_or_f_clustering.head()


# In[ ]:


df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering.merge(pd.get_dummies(df_combined_drivers_m_or_f_clustering.Ejection, prefix='Ejection'), 
                                                                                    left_index=True, right_index=True)
df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering.merge(pd.get_dummies(df_combined_drivers_m_or_f_clustering['Injury Severity'], prefix='Severity'), 
                                                                                    left_index=True, right_index=True)
df_combined_drivers_m_or_f_clustering.head()


# In[ ]:


df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering.drop(columns=['Ejection', 'Injury Severity'])


# In[ ]:


df_combined_drivers_m_or_f_clustering.head()


# In[ ]:


df_combined_drivers_m_or_f_clustering = df_combined_drivers_m_or_f_clustering.dropna()


# In[ ]:


df_combined_drivers_m_or_f_clustering['Sex'] = df_combined_drivers_m_or_f_clustering.Sex.apply(lambda x: 1 if x == 'M' else 0)


# # First Question - Prediction of Driver Male / Female
# Now we will perform the clustering: Creating a df without Sex F / M:

# In[ ]:


X = df_combined_drivers_m_or_f_clustering[df_combined_drivers_m_or_f_clustering.columns[1:]]
y = df_combined_drivers_m_or_f_clustering.Sex

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(max_depth=7)
clf3 = SVC(max_iter=200)

classifiers = [('LR', clf1), ('DT', clf2), ('SVM', clf3)]


# In[ ]:


y.value_counts()


# In[ ]:


results = y_train.to_frame()
for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    results[clf_name] = clf.predict(X_train)
    print ("{:3} classifier:\n         \ttrain accuracy: {:.3f}\n         \ttest accuracy: {:.3f}"        .format(clf_name, 
                clf.score(X_train, y_train), 
                clf.score(X_test, y_test)))


# # Results 1 !!!
# 
# All models are not very suited to cluster the current features between M and F, however the best classifier in terms of consistency between the Train and the Test splits are clearly the Decision Tree classifier. SVM classifier is especially unefective since the datasample size is larger than tens of thousands and this clearly shows that!
# 
# Its accuracy is almost 0.60!

# # Second Question - Clustering of Injured / Uninjured !
# 

# In[ ]:


X_injured = df_combined_drivers_m_or_f_clustering[df_combined_drivers_m_or_f_clustering.columns[:-6]]
y_injured = df_combined_drivers_m_or_f_clustering.Severity_Uninjured

X_train_injured, X_test_injured, y_train_injured, y_test_injured = train_test_split(X_injured, y_injured)


# In[ ]:


X_injured.head()


# In[ ]:


y_injured.value_counts()


# In[ ]:


classifiers_second = [('LR', clf1), ('DT', clf2)]


# In[ ]:


results_injured = y_train_injured.to_frame()
for clf_name_injured, clf_injured in classifiers_second:
    clf_injured.fit(X_train_injured, y_train_injured)
    results_injured[clf_name_injured] = clf_injured.predict(X_train_injured)
    print ("{:3} classifier:\n         \ttrain accuracy: {:.3f}\n         \ttest accuracy: {:.3f}"        .format(clf_name_injured, 
                clf_injured.score(X_train_injured, y_train_injured), 
                clf_injured.score(X_test_injured, y_test_injured)))


# Results 2 - half_way -
# The best classifier in terms of consistency between the Train and the Test splits is again the Decision Tree classifier. 
# Its accuracy is almost 0.80!!

# Finally - ADA boosting with a sample size of the data

# In[ ]:


clf_ada_base = DecisionTreeClassifier(max_depth=3)


# In[ ]:


df_combined_drivers_m_or_f_clustering_sample_10 = df_combined_drivers_m_or_f_clustering.sample(frac = 0.1, random_state = 2)
X_injured_sample_10 = df_combined_drivers_m_or_f_clustering_sample_10[df_combined_drivers_m_or_f_clustering_sample_10.columns[:-6]]
y_injured_sample_10 = df_combined_drivers_m_or_f_clustering_sample_10.Severity_Uninjured

X_train_injured_sample_10, X_test_injured_sample_10, y_train_injured_sample_10, y_test_injured_sample_10 = train_test_split(X_injured_sample_10, y_injured_sample_10)


# In[ ]:


clf_adaboost = AdaBoostClassifier(base_estimator=clf_ada_base, n_estimators=120, learning_rate=0.01)
clf_adaboost.fit(X_train_injured_sample_10, y_train_injured_sample_10)


# In[ ]:


print(clf_adaboost.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_adaboost.score(X_test_injured_sample_10, y_test_injured_sample_10))


# Will different number of estimators or learning rate improve the score substantially?

# In[ ]:


print('n_estimators=80, learning_rate=0.01')
clf_adaboost = AdaBoostClassifier(base_estimator=clf_ada_base, n_estimators=80, learning_rate=0.01)
clf_adaboost.fit(X_train_injured_sample_10, y_train_injured_sample_10)
print(clf_adaboost.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_adaboost.score(X_test_injured_sample_10, y_test_injured_sample_10))

print('n_estimators=100, learning_rate=0.01')
clf_adaboost = AdaBoostClassifier(base_estimator=clf_ada_base, n_estimators=100, learning_rate=0.01)
clf_adaboost.fit(X_train_injured_sample_10, y_train_injured_sample_10)
print(clf_adaboost.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_adaboost.score(X_test_injured_sample_10, y_test_injured_sample_10))

print('n_estimators=140, learning_rate=0.01')
clf_adaboost = AdaBoostClassifier(base_estimator=clf_ada_base, n_estimators=140, learning_rate=0.01)
clf_adaboost.fit(X_train_injured_sample_10, y_train_injured_sample_10)
print(clf_adaboost.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_adaboost.score(X_test_injured_sample_10, y_test_injured_sample_10))

print('n_estimators=140, learning_rate=0.02')
clf_adaboost = AdaBoostClassifier(base_estimator=clf_ada_base, n_estimators=120, learning_rate=0.02)
clf_adaboost.fit(X_train_injured_sample_10, y_train_injured_sample_10)
print(clf_adaboost.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_adaboost.score(X_test_injured_sample_10, y_test_injured_sample_10))


# ADA Boosting appears to be even better than the simple DT classifier, however, in total we get a very similar performance. 
# 
# # Finally and lastly we will use ensamble voting method for the 3 methods previously chosen to find the best scores on the saple data:

# In[ ]:


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(max_depth=5)
clf3 = AdaBoostClassifier(base_estimator=clf2, n_estimators=100, learning_rate=0.01)


# In[ ]:


classifiers_sampled = [('LR', clf1), ('DT', clf2), ('ADA_DT', clf3)]
results = y_train_injured_sample_10.to_frame()


# In[ ]:


for clf_name, clf in classifiers_sampled:
    clf.fit(X_train_injured_sample_10, y_train_injured_sample_10)
    results[clf_name] = clf.predict(X_train_injured_sample_10)
    print(clf_name)
    print(clf.score(X_train_injured_sample_10, y_train_injured_sample_10))
    print(clf.score(X_test_injured_sample_10, y_test_injured_sample_10))
results.head()


# In[ ]:


clf_voting = VotingClassifier(estimators=classifiers_sampled, flatten_transform=True, voting='soft')
clf_voting.fit(X_train_injured_sample_10, y_train_injured_sample_10)
print(clf_voting.score(X_train_injured_sample_10, y_train_injured_sample_10))
print(clf_voting.score(X_test_injured_sample_10, y_test_injured_sample_10))


# # Final Results for question 2 - ADA DT boosting, DT and Ensemble methods are better than the LR model. Simplest decision here is to use the Decision Tree classifieng method because its cheaper and its results are as good as ADA DT boosting and Ensemble. 

# # Part 3 and Results for Age predictions - 
# first - lets try to increase the number of features in order to receive the best possible scores for the question

# After further EDA we see that there are still double Case Vehicle IDs for no reason. Deeper analysis showed duplicated cars for different age of the driver and a different year written - meaning a car that probably has been involved in an accident twice in different years.

# In[ ]:


df_combined_drivers_m_or_f.info()


# First lets deal with the categorical features using OneHotEncoder method -
# 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_selection
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import load_boston
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_combined_drivers_m_or_f.head()


# In[ ]:


df_combined_drivers_m_or_f = df_combined_drivers_m_or_f.dropna()


# In[ ]:


df_combined_drivers_m_or_f.info()


# In[ ]:


for name in df_combined_drivers_m_or_f.columns:
    print(df_combined_drivers_m_or_f[name].value_counts().head())


# In[ ]:


df_combined_drivers_m_or_f.columns


# In[ ]:


X_df_combined_drivers_m_or_f_rare = copy.copy(df_combined_drivers_m_or_f[['Ejection', 'Sex', 'Safety Equipment', 'Injury Severity', 'Age', 'Vehicle Body Type', 'Registration Class', 'Action Prior to Accident', 'Vehicle Year', 'Number of Occupants', 'Vehicle Make']])


# In[ ]:


X_df_combined_drivers_m_or_f_rare.head()


# In[ ]:


for i in ['Ejection', 'Sex', 'Safety Equipment', 'Injury Severity', 'Vehicle Body Type', 'Registration Class', 'Action Prior to Accident', 'Vehicle Make']:
    print(i)
    X_df_combined_drivers_m_or_f_rare.loc[X_df_combined_drivers_m_or_f_rare[i].value_counts()[X_df_combined_drivers_m_or_f_rare[i]].values < 5000, i] = "RARE_VALUE"


# In[ ]:


for name in X_df_combined_drivers_m_or_f_rare.columns:
    print(X_df_combined_drivers_m_or_f_rare[name].value_counts().head(20))


# We reduceted dramatically the number of the categorical features, and now we can label them.

# In[ ]:


X_df_combined_drivers_m_or_f_rare.info()


# In[ ]:


le = LabelEncoder()


# In[ ]:


X_df_combined_drivers_m_or_f_rare.head()


# In[ ]:


X_df_combined_drivers_m_or_f_rare['Ejection'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Ejection'])
X_df_combined_drivers_m_or_f_rare['Sex'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Sex'])
X_df_combined_drivers_m_or_f_rare['Safety Equipment'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Safety Equipment'])
X_df_combined_drivers_m_or_f_rare['Injury Severity'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Injury Severity'])
X_df_combined_drivers_m_or_f_rare['Vehicle Body Type'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Vehicle Body Type'])
X_df_combined_drivers_m_or_f_rare['Registration Class'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Registration Class'])
X_df_combined_drivers_m_or_f_rare['Action Prior to Accident'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Action Prior to Accident'])
X_df_combined_drivers_m_or_f_rare['Vehicle Make'] = le.fit_transform(X_df_combined_drivers_m_or_f_rare['Vehicle Make'])


# In[ ]:


X_df_combined_drivers_m_or_f_rare.info()


# In[ ]:


df_combined_drivers_m_or_f.Age.describe()


# 50% of the people are above 40 and 50% are below 40. Lets cluster the age and between below 40 and above 40:

# Finally lets split age in between above 40 and below 40:

# In[ ]:


X_df_combined_drivers_m_or_f_rare['Age'] = X_df_combined_drivers_m_or_f_rare['Age'].apply(lambda x: 1 if x > 40 else 0)


# Lets select the features using few feature selection methods

# In[ ]:


vt = feature_selection.VarianceThreshold(threshold=.2)


# In[ ]:


vt.fit_transform(X_df_combined_drivers_m_or_f_rare)
vt.variances_


# We can also see the correlations using a heat map

# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X_df_combined_drivers_m_or_f_rare.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["Age"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.015]
relevant_features


# Vehicle Make, Registration class and Injury severity had weak correlations to the Age of the driver.
# Reducing Vehicle Make feature will allow a removal of a substantial amount features.
# For the model we will use the following features:

# In[ ]:


X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare[['Ejection', 'Sex', 'Safety Equipment', 'Vehicle Body Type', 'Action Prior to Accident', 'Vehicle Year', 'Number of Occupants', 'Age']]


# In[ ]:


X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin[X_df_combined_drivers_m_or_f_rare_fin['Number of Occupants'] < 60]
print(X_df_combined_drivers_m_or_f_rare_fin.shape)
X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin[X_df_combined_drivers_m_or_f_rare_fin['Vehicle Year'] > 1993]
print(X_df_combined_drivers_m_or_f_rare_fin.shape)
X_df_combined_drivers_m_or_f_rare_fin.head()


# In[ ]:


X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin.merge(pd.get_dummies(X_df_combined_drivers_m_or_f_rare_fin.Ejection, drop_first = True, prefix='Ejection'), left_index=True, right_index=True)
X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin.merge(pd.get_dummies(X_df_combined_drivers_m_or_f_rare_fin['Safety Equipment'], drop_first = True, prefix='Safety_equip'), left_index=True, right_index=True)
X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin.merge(pd.get_dummies(X_df_combined_drivers_m_or_f_rare_fin['Vehicle Body Type'], drop_first = True, prefix='Vehicle_Body_Type'), left_index=True, right_index=True)
X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin.merge(pd.get_dummies(X_df_combined_drivers_m_or_f_rare_fin['Action Prior to Accident'], drop_first = True, prefix='Act_Prior_Acc'), left_index=True, right_index=True)
X_df_combined_drivers_m_or_f_rare_fin = X_df_combined_drivers_m_or_f_rare_fin.drop(columns= ['Ejection', 'Safety Equipment', 'Vehicle Body Type', 'Action Prior to Accident'])
X_df_combined_drivers_m_or_f_rare_fin.head()


# In[ ]:


X_df_combined_drivers_m_or_f_rare_fin.info()


# # Finally ready for modeling ! clustering according to Age groups - above 40 or below 40:

# In[ ]:


X_df_combined_drivers_m_or_f_rare_fin.head()


# In[ ]:


age_sample_100 = X_df_combined_drivers_m_or_f_rare_fin.sample(frac = 0.01, random_state = 222)

X_age_sample_100 = age_sample_100.drop(columns=['Age'])
y_age_sample_100 = age_sample_100.Age

X_train_age_sample_100, X_test_age_sample_100, y_train_age_sample_100, y_test_age_sample_100 = train_test_split(X_age_sample_100, y_age_sample_100)


# In[ ]:


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(max_depth=5)
clf3 = AdaBoostClassifier(base_estimator=clf2, n_estimators=100, learning_rate=0.01)


# In[ ]:


classifiers_sampled_age = [('LR', clf1), ('DT', clf2), ('ADA_DT', clf3)]
results = y_train_age_sample_100.to_frame()
for clf_name, clf in classifiers_sampled_age:
    clf.fit(X_train_age_sample_100, y_train_age_sample_100)
    results[clf_name] = clf.predict(X_train_age_sample_100)
    print(clf_name)
    print(clf.score(X_train_age_sample_100, y_train_age_sample_100))
    print(clf.score(X_test_age_sample_100, y_test_age_sample_100))
results.head()


# In[ ]:


clf_voting = VotingClassifier(estimators=classifiers_sampled_age, flatten_transform=True, voting='soft')
clf_voting.fit(X_train_age_sample_100, y_train_age_sample_100)
print(clf_voting.score(X_train_age_sample_100, y_train_age_sample_100))
print(clf_voting.score(X_test_age_sample_100, y_test_age_sample_100))


# ADA_Boost appears to be the best classifier.
# 
# Lets check it for the whole dataset (long run)

# In[ ]:


X_age = X_df_combined_drivers_m_or_f_rare_fin.drop(columns=['Age'])
y_age = X_df_combined_drivers_m_or_f_rare_fin.Age
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y_age)


# In[ ]:


classifiers_age = [('LR', clf1), ('DT', clf2), ('ADA_DT', clf3)]


# In[ ]:


results = y_train_age.to_frame()
for clf_name, clf in classifiers_age:
    clf.fit(X_train_age, y_train_age)
    results[clf_name] = clf.predict(X_train_age)
    print(clf_name)
    print(clf.score(X_train_age, y_train_age))
    print(clf.score(X_test_age, y_test_age))
results.head()


# # Final scores for the large Dataset - all classifiers are very similar with ~ 0.56-0.575, however, suprisingly, the LR classifier has the best scores for both the training and the test datasets! 
