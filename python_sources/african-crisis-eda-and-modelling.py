#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing


# In[ ]:


df =  pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


# * Checking top 10 rows for of the dataset

# In[ ]:


df.head()


# According to below codes we don't have null values in our dataset

# In[ ]:


df.info()


# In[ ]:


df[pd.isnull(df).any(axis=1)]


# Now it's usefull to check for unique values within every column in order to see columns just with 0,1 values 

# In[ ]:


df.nunique().sort_values()


# With regards to values in the columns, none of the numeric columns needs to be converted into categorical only the banking_crisis.

# In[ ]:


df['banking_crisis'] = df['banking_crisis'] .apply(lambda x: 1 if x == 'crisis' else 0)


# In[ ]:


df.head()


# **Exploratory data analysis**

# First we should look on the overall crisis 'occurence' in African states. It hit Africa strongly 3 times before 1970. Since 1970 the crises have started and were occuring repeatedly, hitting hardest around 1990s.

# In[ ]:


def_palette = sns.color_palette()
cat_palette = sns.color_palette("hls", 16)
fig, ax = plt.subplots(figsize = (12,6)) 
fig = sns.lineplot(x='year', y='banking_crisis', data=df, palette=cat_palette, ax=ax).set_title('Strength of crisis in Africa')


# Distribution of continuous columns

# In[ ]:


con_columns = ['exch_usd','inflation_annual_cpi'] #columns with continuous variables


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
count = 1

for col in con_columns:
    plt.subplot(2,1,count)
    count += 1
    sns.distplot(df[col])


# We see an inclination of values towards 0 which can eventually provide incorrect information about dataset. This we can remove by taking values (outliers) within first and third quantile (removing minimum and maximum values).

# In[ ]:


q1 = df[con_columns].quantile(0.25)
q3 = df[con_columns].quantile(0.75)
iqr = q3 - q1

df[con_columns] = df[con_columns].clip(q1 - 1.5*iqr, q3 + 1.5*iqr, axis=1)


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
count = 1

for col in df[con_columns]:
    plt.subplot(2,1,count)
    count += 1
    sns.distplot(df[col])


# The data are still skewed but the range of y axis has improved. 

# First hint on number of crises in individual African countries

# In[ ]:


df.groupby(['country', 'banking_crisis']).size().sort_values(ascending=False)


# Is there any pattern in trend lines for countries and their exchange rate/inflation when banking crisis occur? There seems to be no strong evidence when looking at the visuals below.

# In[ ]:


individual_countries = list(df['country'].unique())


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(30,30))

count = 1

for country in individual_countries:
    plt.subplot(5,3,count)
    count+=1
    
    sns.lineplot(df[df.country==country]['year'],
                 df[df.country==country]['exch_usd'],
                 label=country)              
            
    plt.plot([(df[np.logical_and(df.country==country,df.banking_crisis==1)]['year'].unique()),
                  (df[np.logical_and(df.country==country,df.banking_crisis==1)]['year']).unique()],
                 [0,np.max(df[df.country==country]['exch_usd'])],
                 color='black',
                 linestyle='dotted',
                 alpha = 0.8)    
    


# Does indepedence have an effect on inflaction and exchange rate? This also doesn't seem to be the case. 

# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(30,30))
count = 1

for country in individual_countries:
    plt.subplot(5,3,count)
    count+=1
    
    sns.lineplot(df[df.country==country]['year'],
                 df[df.country==country]['exch_usd'],
                 label=country)
                 
  
    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),
              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],
             [0,
              np.max(df[df.country==country]['exch_usd'])],
             color='black',
             linestyle='dotted',
             alpha=0.8)
    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),np.max(df[df.country==country]['exch_usd'])/2,
             'Independence',
             rotation=-90)

    
   
    plt.tight_layout()


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(30,30))
count = 1

for country in individual_countries:
    plt.subplot(5,3,count)
    count+=1
    
    sns.lineplot(df[df.country==country]['year'],
                 df[df.country==country]['inflation_annual_cpi'],
                 label=country)
                 
  
    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),
              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],
             [0,
              np.max(df[df.country==country]['inflation_annual_cpi'])],
             color='black',
             linestyle='dotted',
             alpha=0.8)
    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),np.max(df[df.country==country]['inflation_annual_cpi'])/2,
             'Independence',
             rotation=-90)

    
   
    plt.tight_layout()


# Looking closer at number of events (occured vs. not-occured) in binary columns when country faced banking crisis.

# In[ ]:


sns.set(style='whitegrid')
cols_countplot=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises']
plt.figure(figsize=(20,20))
count = 1
df_bank_crisis = df.loc[df['banking_crisis'] == 1]

for col in cols_countplot:
    plt.subplot(3,2,count)    
    count+= 1
    sns.countplot(y='country', hue = col, data = df_bank_crisis).set_title(col)    
    plt.legend(loc = 0)


# The same can be done also when country didn't face a crisis.

# In[ ]:


sns.set(style='whitegrid')
cols_countplot=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises']
plt.figure(figsize=(20,20))
count = 1
df_no_bank_crisis = df.loc[df['banking_crisis'] == 0]

for col in cols_countplot:
    plt.subplot(3,2,count)    
    count+= 1
    sns.countplot(y='country', hue = col, data = df_no_bank_crisis).set_title(col)    
    plt.legend(loc = 0)


# Systemic crisis seems to be (not) occuring when banking crisis ( does not) occurs. Simoutaneosly, based just on these graphics I would assume that domestic debt in default has no effect at all on banking crisis whether it occurs or not. We should check this with a correlation matrix.

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot = True)


# Looking just on the banking crisis row we can see positive correlation between banking crisis and systemic crisis. This means when one of them appears the second one most probably follows. Investopedia also writes about relationship between systemic crisis and banking crisis (https://www.investopedia.com/terms/s/systemic-risk.asp)
# Other indicators seem to have nor strong positive or negative correlation with the crisis.

# **Modelling**

# Let's run the data against several classification models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# Keeping just relevant columns for dependent X variables and independent y variable

# In[ ]:


X = df.drop(['banking_crisis','cc3','country','year','case'], axis = 1)
y = df['banking_crisis'] 


# Splitting columns for train and test samples

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 


# **Logistic regression**

# *First I start with logistic regression as the basic technique for classifying binary data*

# In[ ]:


logistic_model = LogisticRegression()


# In[ ]:


logistic_model.fit(X_train,y_train)


# In[ ]:


y_predict_logm = logistic_model.predict(X_test)


# Coefficients of the sigmoid function for logistic model

# In[ ]:


column_label = list(X_train.columns)
model_Coeff = pd.DataFrame(logistic_model.coef_, columns = column_label)
model_Coeff['intercept'] = logistic_model.intercept_
print("Coefficient Values Of The Surface Are: ", model_Coeff)


# When we run the score of the model we see that model achieves high number and thus have high explanatory power

# In[ ]:


logmodel_score = logistic_model.score(X_test,y_test)
print('Model score:\n', logmodel_score)


# In[ ]:


print(metrics.confusion_matrix(y_test, y_predict_logm)) #22 = true positive, 291 = true negative, 2 = false positive, 3 = false negative


# We get quite good results with logistic model (we don't have a robust sample therefore results show the model accuracy high)

# In[ ]:


print(classification_report(y_test,y_predict_logm))


# **Decision tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


decision_tree = DecisionTreeClassifier()


# In[ ]:


decision_tree.fit(X_train,y_train)


# In[ ]:


y_predict_tree = decision_tree.predict(X_test)


# In[ ]:


print(metrics.confusion_matrix(y_test, y_predict_tree))


# In[ ]:


print(classification_report(y_test,y_predict_tree))


# Compared to logistic model, decision tree model show lower results.

# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list( df.drop(['banking_crisis','cc3','country','year','case'], axis = 1))


# In[ ]:


dot_data = StringIO()  
export_graphviz(decision_tree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# We can see that splitting happens with relativily low Gini coefficients (majority of values belongs to one class, if the division would be even we would 0.5 Gini index)

# **Random Forests**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# When applying Random tree on the dataset we receive very good results.

# **SVM**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC() #with predefined parameters


# In[ ]:


svc_model.fit(X_train,y_train)


# In[ ]:


svm_pred = svc_model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,svm_pred))


# In[ ]:


print(classification_report(y_test,svm_pred))


# We've got quite good results with predefined parameters

# We can search for the best parameters with the help of Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# *Best parameters*

# In[ ]:


grid.best_params_ 


# We run again the model with parameters we found as best in the grid search 

# In[ ]:


grid_pred = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,grid_pred))


# In[ ]:


print(classification_report(y_test,grid_pred))


# We can see improvement in results with new parameters and work with SVM as suitable modelling approach for this dataset

# **K Means clustering**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters=2)


# In[ ]:


kmeans.fit(df.drop(['banking_crisis','cc3','country','year','case'], axis = 1))


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


df['Cluster'] = df['banking_crisis']


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# Without any futher data manipulation applying K-means (cluster) model on the dataset results in poor numbers for precision, recall f1-score and accuracy. Model isn't suitable in this setting.

# *When I compare above models, SVM with adjusted parameters (C:100, gamma :0.01) performs the best for the dataset. Simoutaneosly, random forests and logistic regression can be also used without modifications. Remaining models would need more feature engineering in order to improve their results*
