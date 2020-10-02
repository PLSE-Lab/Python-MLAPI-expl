#!/usr/bin/env python
# coding: utf-8

# Global Terrorism Trend
# ==

# Agnes & Junming

# In the project, we would like to analyze the percent distribution of incidents and fatalities by region, the main characteristics of terrorist events,  the frequency of attacks by region ,terrorism tactics by region over time and the terrorism activity over time for select countries, etc.

# what is the distribution of terror incidents and fatalities by region? What is the frequency of attacks by region? What is terrorism tactics in different region? Which countries have the most frequency of attacked and is there some similarity between these counties? 

# 1.import libraries 
# --

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# from mpl_toolkits.basemap import Basemap
import seaborn as sns

import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')


import io
import requests
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('figure', figsize=(12,10))
get_ipython().run_line_magic('matplotlib', 'inline')

#specific to time series analysis
import scipy.stats as st
from statsmodels.tsa import stattools as stt
from statsmodels import tsa
import statsmodels.api as smapi
import datetime

from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.simplefilter(action='ignore')


# 2.Glimps of Data 
# --
# 

# ### 2.1 Read Data

# In[ ]:


try:
    terrorism = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')


# In[ ]:


# terrorism = pd.read_csv('global terrorism.csv', encoding='ISO-8859-1')
terrorism.head(10)


# In[ ]:


terrorism.info()


# ### 2.1 Statistic Overview of Data

# In[ ]:


terrorism.describe() ##describes only numeric data


# The table from above describes the information about the numeric columns of the terrorism data. Since the information is provided for only the numeric columns, and no information is provided about missing data, I created a more in-depth tool below to describe the information for all the attributes.

# 3. Preparing the Data  
# 3.1 Check missing data  
# 3.2 Data Cleaning

# ### 3.1 Rename Dataframe

# In[ ]:


rename = terrorism.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terrorism = terrorism[['Year','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terrorism['casualities']=terrorism['Killed']+terrorism['Wounded']
terrorism.head(10)


# ### 3.2 Check missing data

# In[ ]:


terrorism.dropna(how = 'all', inplace=True)
print('Size After Dropping Rows with NaN in All Columns:', terrorism.shape)


# In[ ]:


terrorism.isnull().sum()


# 4.Distribution of Terrorism by time
# --

# In[ ]:


print('Country with Highest Terrorist Attacks:',terrorism['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',terrorism['Region'].value_counts().index[0])
print('Maximum people killed in an attack are:',terrorism['Killed'].max(),'that took place in',terrorism.loc[terrorism['Killed'].idxmax()].Country)


# ### 4.1 Number of terrorist activities each year

# In[ ]:


plt.subplots(figsize=(13,6))
sns.countplot('Year',data=terrorism,palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# ### 4.2 Number of people were killed by terrorism each year.

# In[ ]:


people_killed_eachyr = terrorism[["Year","casualities"]].groupby('Year').sum()

plt.subplots(figsize = (15,6))
sns.barplot(x=people_killed_eachyr.index, 
            y=[i[0] for i in people_killed_eachyr.values], data = people_killed_eachyr, palette='RdYlGn_r',edgecolor=sns.color_palette("Set2", 10))
plt.xticks(rotation=90)
plt.title('Number Of people were killed of wouded by terrorism each year')
plt.show()


# In[ ]:


terrorism.to_csv('terr.csv', index = False)


# In[ ]:


dateparse = lambda d: pd.datetime.strptime(d, '%Y')


# In[ ]:


f='terr.csv'
terrorism_ts = pd.read_csv(f,
                   parse_dates=['Year'], 
                   index_col='Year', 
                   date_parser=dateparse,
                   )


# In[ ]:


terrorism_ts.head()


# In[ ]:


terrorism_ts = terrorism_ts.iloc[:, 0]
terrorism_ts.head()


# In[ ]:


type(terrorism_ts)


# ======================================================================================================================

# In[ ]:





# 5. Geographical Analysis of Terrorism  
# Distribution of terrorism in different regions.(groupby region on attacks)   
# Which country has the highest attacks (groupby country on attacks)  
# What kinds of methodes(weapons) were used by terrorists? (groupby methords on attacks)  
# Attack type vs region
# Regions Attacked By Terrorist Groups  
# Terrorism by country
# 
# 

# ### 5.1 Distribution of terrorism in different regions.(groupby region on attacks)

# In[ ]:


plt.subplots(figsize=(13,6))
sns.countplot('Region',data=terrorism, palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Region')
sns.set(font_scale=1)
plt.show()


# In[ ]:


terror_region=pd.crosstab(terrorism.Year,terrorism.Region)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()


# ### 5.2 Number of attacks by countries.

# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(terrorism['Country'].value_counts()[:15].index,terrorism['Country'].value_counts()[:15].values,palette='inferno')
plt.title('Top Affected Countries')
sns.set(font_scale=1)
plt.show()


# ### 5.3 What kinds of methodes(weapons) were used by terrorists? (groupby methords on attacks)

# In[ ]:


pd.crosstab(terrorism.Region,terrorism.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))
fig=plt.gcf()
fig.set_size_inches(12,8)
sns.set(font_scale=0.5)
plt.show()


# ### 5.4 What kinds of methods(weapons) were used by terrorists? (groupby methods on attacks)

# In[ ]:





# In[ ]:





# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terrorism,palette='inferno',order=terrorism['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
sns.set(font_scale=2)
plt.show()


# ### 5.5 Attack types were used by Terrorist Groups

# In[ ]:


# pd.crosstab(terrorism.AttackType,terrorism.Group).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))
# fig=plt.gcf()
# fig.set_size_inches(12,8)
# sns.set(font_scale=0.5)
# plt.show()


# 6. Act of Terrorism Analysis  
#   
#   Share of kills, wounded and property damage per attack type.

# ### 6.1 What kinds of targets were favored by terrorist?  

# In[ ]:


# plt.subplots(figsize=(13,6))
# sns.countplot('Target',data=terrorism,palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
# plt.xticks(rotation=90)
# plt.title('Number Of Terrorist Activities Each Target')
# plt.show()


# In[ ]:





# ## 7. Prediction

# As for the prediction part, we may want to figure out which factors matters most when we want to predict the outcome of an attack. To accomplish a study on this, we set casualty value (killed + injured) as our target variable and other variable as our predictors.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import itertools


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier


# Firstly, we may want to do some preprocessing. 

# In[ ]:


####7.1 Preprocessing

# Before we do the classification, we may want to delete some columns like city and motive since some of them may have
# linear relationship with other columns like killed and some of them has too many missing value like motive

terrorism_cm = terrorism.drop(columns = ['Motive','Target','Killed','Wounded','Summary','city'])


# In[ ]:


# Encode category predictors into numbers, facilitating later work
labelEncoding = LabelEncoder()
terrorism_cm['Country'] = labelEncoding.fit_transform(terrorism_cm['Country'])
terrorism_cm['AttackType'] = labelEncoding.fit_transform(terrorism_cm['AttackType'])
terrorism_cm['Target_type'] = labelEncoding.fit_transform(terrorism_cm['Target_type'])
terrorism_cm['Weapon_type'] = labelEncoding.fit_transform(terrorism_cm['Weapon_type'])
terrorism_cm['Region'] = labelEncoding.fit_transform(terrorism_cm['Region'])
terrorism_cm['Group'] = labelEncoding.fit_transform(terrorism_cm['Group'])


terrorism_cm['casualities'] = terrorism_cm['casualities'].apply(lambda x: 0 if x == 0 else 1)


# In[ ]:


terrorism_cm.head(5)


# In[ ]:


len(terrorism_cm)


# In[ ]:


# We drop na to avoid misinformation
terrorism_cm = terrorism_cm.dropna()
len(terrorism_cm)


# In[ ]:


len(terrorism_cm[terrorism_cm['casualities'] == 0])


# In[ ]:


####7.2 Cross Validation

# Split data for training data and validation data
X = terrorism_cm[['Year','Country','Region','latitude','longitude','AttackType','Group','Target_type','Weapon_type']]
valid = terrorism_cm['casualities']

X_train, X_test, valid_train, valid_test = train_test_split(X, valid, test_size=0.3)


# In[ ]:





# In[ ]:


####7.3 Compute the feature importances with random forest
forest = ExtraTreesClassifier(n_estimators=20,
                              random_state=0)

forest.fit(X, valid)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
fnames = [['Year','Country','Region','latitude','longitude','AttackType','Group','Target_type','Weapon_type'][i] for i in indices]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), fnames, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# As it can be seen from above, features like longitude, latitude, target type and year have high feature importance index, which means that these features are more influential when predicting the value of casualty. However, we may want to check whether this outcome is accurate enough so we need to calculate the accuracy from the learner to demonstrate it is a valid result.

# In[ ]:


####7.4 Train the model
X = terrorism_cm[['Year','Country','latitude','longitude','AttackType','Group','Target_type','Weapon_type']]
valid = terrorism_cm['casualities']

X_train, X_test, valid_train, valid_test = train_test_split(X, valid, test_size=0.3)


# In[ ]:


model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, valid_train)
pred = model.predict(X_test)
np.mean(pred == valid_test)


# As we can see from above, the accuracy result is about 81.9%. This proves that the result from our feature importance analysis is almost accurate.

# In[ ]:


####7.5 Confusion Matrix
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
cnf_matrix = confusion_matrix(valid_test, pred)
 

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
# Compute confusion matrix
np.set_printoptions(precision=2)
 
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      title='Confusion matrix, without normalization')
 
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')
 
plt.show()


# The result shows that about the True Positive rate reaches about 0.86 whereas the True Negative rate reaches about 0.74.

# In[ ]:


####7.6 ROC curve
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import roc_curve
score = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(pred, score, pos_label=1)
auc = np.trapz(tpr, fpr)

plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.legend()
plt.show() 


# As it can be seen from the AUC analysis that, the curve is pretty close to the upper left of the plot, demonstrating a high overall accuracy of our learner. (There exists a moment that True positive rate could be high whereas false positive rate could be maintained at a pretty low value)

# 8. Citation

# A good amount of our inspiration came from other Kaggle notebooks under this dataset. It was inevitable that we would come across the notebooks while learning about the dataset. We have referred other great kagglers' works to accomplish this project. Thanks for all of you.
