#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Preliminary note ##
# 
# For some functions and reasoning of this notebook, I had consulted the following notebooks, already available at the Kernel:
# 
# https://www.kaggle.com/brandao/setting-a-limit-for-false-negatives-covid19
# 
# https://www.kaggle.com/ossamum/exploratory-data-analysis-and-feature-importance
# 
# https://www.kaggle.com/nazeboan/null-values-exploration-logreg-67-acc
# 
# https://www.kaggle.com/endelgs/98-accuracy-at-covid-19
# 
# https://www.kaggle.com/rodrigofragoso/exploring-nans-and-the-impact-of-unbalanced-data
# 
# https://www.kaggle.com/eduardosmorgan/quick-data-exploration-and-svm-cv
# 
# https://www.kaggle.com/andrewmvd/lightgbm-baseline
# 
# https://www.kaggle.com/dmvieira/overfitting-ward-semi-intensive-or-intensive-unit
# 
# https://www.kaggle.com/dmvieira/94-precision-on-covid-19
# 
# https://www.kaggle.com/rspadim/eda-first-try-python-lgb-shap

# ## Motivation ##
# 
# After researching similar articles about the clinical characteristics of covid-positive and covid-negative pneumonia patients, I found that:
# 
# *First study*:
# 
# 105 patients, almost 50/50 positive/negative patients with a similar age and a similar division by gender. 
# 
# Eusinopenia (low number of eusinophils) were present in 78.8% of positive-patients and in only 35.8% of negative-patients.
# Both patients, however, had shown fever and respiratory symptons.
# 
# https://www.medrxiv.org/content/10.1101/2020.02.13.20022830v1.full.pdf+html
# 
# *Second study*:
# 
# 140 covid-positive patients study shows 75.4% had lymphopenia and 52.9% eusinopenia with a similar division by gender but with different ages:
# 
# https://onlinelibrary.wiley.com/doi/full/10.1111/all.14238
# 
# *Third Study*:
# 
# And this other study with 291 covid-positive patients shows that  Leukopenia, lymphopenia and eosinopenia occurred in 36.1%, 22.7% and 50.2% patients respectively. The division by gender is almost 50/50, but here, there are people with different ages.
# 
# https://www.medrxiv.org/content/10.1101/2020.03.03.20030353v1
# 
# *Fourth Study*:
# 
# This notebook only took the hemogram data and had a good performance, with a very low number of FP and FN.
# 
# https://www.kaggle.com/guidorzi93/modelagem-utilizando-dados-de-hemograma

# ## Strategy ##
# 
# First we'll use seaborn to get some visualization, for each age quantile, of the level of leukocytes, lymphocytes and eosinophils, hued by positive/negative results in the Sars-Cov-2 test.
# 
# If the different age quantiles have similar blood information/covid information correlation, we'll use an unique Decision Tree model, without separating by age.
# 
# If some quantiles present similar behaviour in the above aspect, we'll make more than one model, one for each group of similar quantiles.
# 
# The Decision Tree model, although not so good in the performance, is a very visual, intuitive model, which is good for the purpose of this work: be used by non-specialists in Data Science.

# ## Reading and Cleaning the data ##

# In[ ]:


df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head(5)


# In[ ]:


df_cleaned = pd.concat([df['SARS-Cov-2 exam result'], df['Patient age quantile'], df['Eosinophils'],
                       df['Leukocytes'], df['Lymphocytes']],
                      axis = 1)


# In[ ]:


df_cleaned.head(5)


# In[ ]:


df_cleaned['SARS-Cov-2 exam result'] = df_cleaned['SARS-Cov-2 exam result'].map(lambda r: 1 if r == 'positive' else 0 )


# From a previous study (https://www.kaggle.com/carlosasdesouza/40-nan-classes-in-features) we know that our 3 blood variables belong to Category 4, with 5042 Nan values out of 5643 entries. So we'll use only the non-Nan data, i.e: 601 patients.

# In[ ]:


def clean_nan(col, datafr):
    i = 0
    while i < len(col):
        if pd.isnull(col[i]):
            datafr.drop(i, axis = 0, inplace = True)
        i+=1
clean_nan(df_cleaned['Eosinophils'], df_cleaned)


# In[ ]:


df_cleaned.head(10)


# ## Exploratory Data Analysis ##

# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


sns.countplot(x ='Patient age quantile',data=df_cleaned,hue='SARS-Cov-2 exam result' )


# In[ ]:


sns.countplot(x ='Patient age quantile',data=df_cleaned[df_cleaned['Patient age quantile'] == 0],hue='SARS-Cov-2 exam result' )


# In[ ]:


#Here we see that the 0 age quantile doesn't have any covid-positive case, so we'll drop it
null_covid = df_cleaned[df_cleaned['Patient age quantile'] == 0]
for i in null_covid.index:
    df_cleaned.drop(i, axis = 0, inplace = True)


# In[ ]:


sns.countplot(x ='Patient age quantile',data=df_cleaned,hue='SARS-Cov-2 exam result' )


# In[ ]:


##Seting a threshold for accuracy

n_covid = len(df_cleaned[df_cleaned['SARS-Cov-2 exam result'] == 0].index)
y_covid = len(df_cleaned[df_cleaned['SARS-Cov-2 exam result'] == 1].index)
print('Min Accuracy for 0: ', n_covid/(n_covid + y_covid))
print('Min Accuracy for 1: ', y_covid/(n_covid + y_covid)) 


# In[ ]:


sns.pairplot(df_cleaned, hue = 'SARS-Cov-2 exam result')


# *By the plots above, we can have some intuition about what's going on*:
# 
# #1 Covid positive patients *never* reachs a high-medium level of eosinophils
# #2 Except for a (it seems) outlier, the same conclusion applies to the Leukocytes level
# #3 The Lymphocytes level looks pretty similar for the covid-negative and covid-positive patients. We'll drop it, since it doesn't give useful information to differentiate a covid-positive from a covid-negative patients

# In[ ]:


sns.heatmap(df_cleaned.drop('SARS-Cov-2 exam result', axis = 1).corr(),annot=True)

#We'll only see the "intra-feature" correlation in this graph


# In[ ]:


#It seems that the features(above) and the features and the target(below) are weakly pair-correlated
sns.heatmap(df_cleaned.corr(),annot=True)


# ## Training a KNN Model ##
# 
# I've made this decision because of the eosinophilis/leukocytes characteristic of concentrating the positive cases in one region

# In[ ]:


#Splitting the train/test sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned.drop(['SARS-Cov-2 exam result','Lymphocytes'], axis = 1),
                                                    df_cleaned['SARS-Cov-2 exam result'],
                                                    test_size=0.30, random_state = 42)


# In[ ]:


#Choosing a good k value
error_rate = []
for i in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    error_rate.append(np.mean(predictions != y_test))
newdf = pd.DataFrame(error_rate, columns = ['Error Rate'])
newdf['IDX'] = range(1,30)


# In[ ]:


newdf.plot.line(x = 'IDX', y = 'Error Rate', figsize = (12,6))
#From the plot above, we can see that K = 5 is the minimum K that minimizes the error rate, dropping it to 0.11


# In[ ]:


#Final KNN model

knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)


# In[ ]:


predictions = knn_model.predict(X_test)
print('WITH K = 5')
print('\n')
print(confusion_matrix(predictions, y_test))
print('\n')
print(classification_report(predictions,y_test))

# By  the data below, we can see that this isn't a good model from the 'minimize the False-Negatives' perspective.
# Out of 173 patients, we predicted 4 False-Negative and 15 False-Positives


# ## Threshold for Eosinophils and Leukocytes ##
# 
# Although the model above can be used to a quick classification of patients, it stills give a large number of False positives.
# A more safe clinical approach could be setting lower bounds for the Eosinophils and Leukocytes level - and patients with values above this level would have almost 100% of probability of being covid-negatives (we'll determine this probability below *for this subset*). We don't know yet how representative this set is compared to the whole population, making a professional statistical inference impossible with this data.

# In[ ]:


df_cleaned[df_cleaned['SARS-Cov-2 exam result']==1]['Eosinophils'].sort_values()


# In[ ]:


df_cleaned[df_cleaned['SARS-Cov-2 exam result']==1]['Leukocytes'].sort_values()


# ## Setting lower bounds/Conclusion ##
# 
# * Pick a random patient out of a population of 601. If he or she has a level of Eosinophils greater than 1.0 (in this standardized scale) and a Leukocytes level greater than 1.0, what is the probability of this patient being covid-19 positive?*
# 
# Answer: zero. The only covid-positive patient with a Eosinophils level greater than 1.0 is patient 3779. On the other side, the only covid-positive patient with a Leukocytes level greater than 1.0 is patient 5169.
# 
# * Pick a random patient out of a population of 601. If he or she has a level of Eosinophils greater than 1.0 (in this standardized scale) or a Leukocytes level greater than 1.0, what is the probability of this patient being covid-19 positive?*
# 
# Answer: 2/601 ~= 0,3%

# ## Is this conclusion useful? ##
# 
# One could ask: if the levels (of Eosinophils and Leukocytes) of ALL PATIENTS are generally below 1.0, this model isn't useful.
# Let's count how many patients had a level greater than 1.0

# In[ ]:


#The number of patients with a 'Leukocytes' level greater than 1.0 is 66. We could classify all this patients as 
#covid-negative, with a 1,5% porcentage of false-negatives
len(df_cleaned[df_cleaned['Leukocytes'] > 1.0])


# In[ ]:


#The number of patients with a 'Leukocytes' level greater than 1.0 is 71. We could classify all this patients as 
#covid-negative, with a 1,4% porcentage of false-negatives
len(df_cleaned[df_cleaned['Eosinophils'] > 1.0])


# In[ ]:


#Finally, let's count how many patients had both levels greater than 1.0
(df_cleaned[df_cleaned['Leukocytes'] > 1.0]['Eosinophils'] > 1.0).sort_values()


# For this intersection hypothesis, as we can see above, we've only reached 4 patients, however with zero rate of false negatives

# In[ ]:




