#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


directory = "/kaggle/input/airline-passenger-satisfaction/"
feature_tables = ['train.csv', 'test.csv']

df_train = directory + feature_tables[0]
df_test = directory + feature_tables[1]

# Create dataframes
print(f'Reading csv from {df_train}...')
df = pd.read_csv(df_train)
print('...Complete')

print(f'Reading csv from {df_train}...')
df_test = pd.read_csv(df_test)
print('...Complete')


# In[ ]:




df.head(10)


# ## DATA CLEANING:
# ## We dont need id and Unnamed:0 columns. Also, since "Arrival Delay in Minutes" is 0.94 correlated with "Departure Delay in Minutes" and there are nan values in "Arrival Delay in Minutes" we dropped it too.

# In[ ]:


df=df.drop(columns=["Unnamed: 0","id","Arrival Delay in Minutes"])
df_test=df_test.drop(columns=["Unnamed: 0","id","Arrival Delay in Minutes"])


# ## We dont have any null values in dataframe.

# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr())

plt.show()


# In[ ]:


df.columns


# ## The columns below consist of categorical values so we change them into categories.

# In[ ]:


for col in ['Gender', 'Customer Type','Type of Travel', 'Class','Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness','satisfaction']:
    df[col] = df[col].astype('category')


# In[ ]:


for col in ['Gender', 'Customer Type','Type of Travel', 'Class','Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness','satisfaction']:
    df_test[col] = df_test[col].astype('category')


# ## Only correlation with non-categorical values shown below.

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr())

plt.show()


# ## OUTLIER DETECTION

# ## "Flight Distance" and "Departure Delay in Minutes" are not normal distributed as seen in below figure. 
# 

# In[ ]:


plt.hist(df["Flight Distance"])
plt.show()


# In[ ]:


plt.hist(df["Departure Delay in Minutes"])
plt.show()


# In[ ]:


def outlier_treatment(datacolumn):
 sorted(datacolumn)
 
 Q1,Q3 = np.percentile(datacolumn , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (3 * IQR)
 upper_range = Q3 + (3 * IQR)
 return (lower_range,upper_range)


# ## The IQR method says upper range for Dep_delay is 48 and there are 9318 values more than that which is 20 percent of out data.
# ## IQR method for outlier detection is used only in normal distributed data.
# ## So we will use bootstrapping. Check out this link for more detail about bootstrapping.
# ## more detail:"https://www.suplari.com/calculating-outliers-spend-data-bootstrapping/"

# In[ ]:


print(outlier_treatment(df['Departure Delay in Minutes']))
df[df['Departure Delay in Minutes']>48].count()


# ## Using bootstrapping we get upper range of 184 for dep_delay and 3888 for Flight_Distance

# In[ ]:


df['dep_delay'] = df['Departure Delay in Minutes'].apply(lambda x: 184 if x >184 else x)
df_test['dep_delay'] = df_test['Departure Delay in Minutes'].apply(lambda x: 184 if x >184 else x)


# In[ ]:


df['Flight_Distance'] = df['Flight Distance'].apply(lambda x: 3888 if x >3888 else x)
df_test['Flight_Distance'] = df_test['Flight Distance'].apply(lambda x: 3888 if x >3888 else x)


# ## DETAILED CODE ABOUT BOOTSTRAPPING

# In[ ]:


df.columns


# In[ ]:


df=df.drop(columns=["Flight Distance"])
df_test=df_test.drop(columns=["Flight Distance"])


# In[ ]:


data_bs=df["Departure Delay in Minutes"].to_numpy()
data_bs


# In[ ]:


def bootstrap_sample(amounts):
    return np.random.choice(amounts, len(amounts), replace=True)

def percentile_99(sample):
     return np.percentile(sample, 99)

def bootstrap_confidence_interval(data):
    """
    Creates list of 10000 99th percentile bootstrap replicates. 
    """
    bs_samples = np.empty(10000)
    
    for i in range(10000):
        bs_samples[i] = percentile_99(bootstrap_sample(data_bs))

    return bs_samples

transactions_ci = bootstrap_confidence_interval(data_bs)


# In[ ]:


plt.hist(transactions_ci)
plt.show()


# In[ ]:



print(np.percentile(transactions_ci, 95))


# ## The machine algorithms that we use works with integers not strings. Thats why we used get_dummies function.

# In[ ]:



df=pd.get_dummies(df,columns=["Gender","Customer Type","Type of Travel","Class","satisfaction"],drop_first=True)


# In[ ]:



df_test=pd.get_dummies(df_test,columns=["Gender","Customer Type","Type of Travel","Class","satisfaction"],drop_first=True)


# In[ ]:


df['Class_Eco Plus'] = df['Class_Eco Plus'].apply(lambda x: 2 if x==1 else x)


# In[ ]:


df_test['Class_Eco Plus'] = df_test['Class_Eco Plus'].apply(lambda x: 2 if x==1 else x)


# In[ ]:


df["class"]=df['Class_Eco Plus']  + df['Class_Eco']


# In[ ]:


df_test["class"]=df_test['Class_Eco Plus']  + df_test['Class_Eco']


# In[ ]:


df=df.drop(columns=["Class_Eco","Class_Eco Plus","Departure Delay in Minutes"])
df_test=df_test.drop(columns=["Class_Eco","Class_Eco Plus","Departure Delay in Minutes"])


# In[ ]:


y=df["satisfaction_satisfied"]
x=df.drop(columns=["satisfaction_satisfied"])


# In[ ]:


y_test=df_test["satisfaction_satisfied"]
x_test=df_test.drop(columns=["satisfaction_satisfied"])


# ## MODELING PART:

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(random_state=0)
dtc.fit(x,y)


# In[ ]:


y_pred=dtc.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)


# ## It seems Online Boarding and Inflight Wifi Service is the most important features.

# In[ ]:


feature_imp=pd.Series(dtc.feature_importances_,index=x.columns)
feature_imp.sort_values(ascending=False,inplace=True)
feature_imp.plot(kind='barh')


# In[ ]:


dtc_reduced=DecisionTreeClassifier(min_samples_leaf = 20)


# ## If we work with only best three features we still get 0.89 accuracy score. But it works a lot faster.

# In[ ]:


x_reduced=x[["Online boarding","Inflight wifi service","Type of Travel_Personal Travel"]]
dtc_reduced.fit(x_reduced,y)
x_test_reduced=x_test[["Online boarding","Inflight wifi service","Type of Travel_Personal Travel"]]

y_pred_red=dtc_reduced.predict(x_test_reduced)
accuracy_score(y_test,y_pred_red)


# ## Lets draw ROC Curve and check Area Under Curve

# In[ ]:


from sklearn.metrics import roc_auc_score , roc_curve
dtc_proba=dtc.predict_proba(x_test)
dtc_proba=dtc_proba[:,1]
auc=roc_auc_score(y_test, dtc_proba)
print('Desicion Tree: ROC AUC=%.3f' % (auc))
lr_fpr, lr_tpr, _ = roc_curve(y_test, dtc_proba)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Desicion Tree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# ## Lets try modeling with other classification algorithms ans see which is best.

# In[ ]:


import warnings
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('LGB', LGBMClassifier()))
#XGB doesnt work with categorical variables.
#models.append(('XGB', XGBClassifier()))


# In[ ]:


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ## Ploting ROC curves for different algorithms.
# ## We see that LGB is the best classifier for this data. ROC AUC is 0.995 and accuracy score is 0.9645 for LGB.

# In[ ]:


from sklearn.metrics import roc_auc_score , roc_curve

for name, model in models:
   
    model.fit(x,y)
    y_pred=model.predict(x_test)
    y_proba=model.predict_proba(x_test)
    y_proba=y_proba[:,1]
    auc=roc_auc_score(y_test, y_proba)
    print('%s: ROC AUC=%.3f' % (name,auc))
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    acc_score=accuracy_score(y_test,y_pred)
    msg = "%s:Accuracy: %f " % (name, acc_score)
    print(msg)
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()    


# In[ ]:





# In[ ]:




