#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import pearsonr
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
import itertools
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from xgboost import plot_tree,plot_importance
from sklearn import tree
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


data=pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df=data.copy()
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.describe().T


# In[ ]:


df.describe(include="object").T


# In[ ]:


df=df.drop(columns=["Over18"],axis=1)


# In[ ]:


df.isnull().sum().to_frame()


# In[ ]:


cat_columns=df.select_dtypes(include="object").columns
for i in cat_columns:
    plt.figure(figsize=(9,5));
    sns.countplot(df[i]);
    plt.xticks(rotation=90);
    plt.show();


# In[ ]:


for i in cat_columns[1:]:
    sns.catplot(x=i,y='MonthlyIncome',hue="Attrition",data=df,kind="bar",aspect=3);
    plt.xticks(rotation=90);


# In[ ]:


def diagnostic_plots(df, variable):
    
    plt.figure(figsize=(20, 9))

    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30,kde_kws={'bw': 1.5})
    plt.title('Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    
    
    plt.show()


# In[ ]:


df.hist(edgecolor='black', linewidth=1.2, figsize=(22, 22));


# In[ ]:


for i in ["Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate"]:
       diagnostic_plots(df,i)


# In[ ]:


df.dtypes.to_frame()


# In[ ]:


df["EmployeeCount"].value_counts()


# In[ ]:


df["StandardHours"].value_counts()


# In[ ]:


sns.countplot(df["PerformanceRating"],palette="coolwarm");


# In[ ]:


df=df.drop(columns=["EmployeeCount","StandardHours"],axis=1)


# In[ ]:


num_columns=df.select_dtypes(exclude="object").columns


# In[ ]:


for i in num_columns:
    sns.boxplot(df[i],color="orangered");
    plt.show();


# In[ ]:


plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(28,16))
corr=df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);


# In[ ]:


def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# In[ ]:


cat_columns


# In[ ]:


corrM = np.zeros((len(cat_columns),len(cat_columns)))
for col1, col2 in itertools.combinations(cat_columns, 2):
    idx1, idx2 = cat_columns.tolist().index(col1), cat_columns.tolist().index(col2)
    corrM[idx1, idx2] = cramers_v(pd.crosstab(df[col1], df[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]
corr = pd.DataFrame(corrM, index=cat_columns, columns=cat_columns)
fig, ax = plt.subplots(figsize=(10, 6))

ax = sns.heatmap(corr, annot=True, ax=ax,linecolor="white",linewidths=1); ax.set_title("Cramer V Correlation between Variables");


# In[ ]:


df["EmployeeNumber"].nunique()


# In[ ]:


df=df.drop("EmployeeNumber",axis=1)


# In[ ]:


df.head()


# In[ ]:


df_new=df.copy()
df_new=df_new.drop(columns=["Department","JobLevel","YearsInCurrentRole","TotalWorkingYears","PercentSalaryHike","YearsWithCurrManager"],axis=1)


# ## XGBOOST MODEL

# In[ ]:


df_new=pd.get_dummies(df_new,drop_first=True)


# In[ ]:


df_new.head()


# In[ ]:


X=df_new.drop("Attrition_Yes",axis=1)
y=df_new["Attrition_Yes"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42)


# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


cross_val_score(xgb_model,X,y).mean()


# In[ ]:


def conf_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlGnBu');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))
    plt.title(all_sample_title, size = 15);
    plt.show()
    print(classification_report(y_test,y_pred))
conf_matrix(y_test,y_pred)


# In[ ]:


def plot_roc_curve(y_test,X_test,model):
    fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_mlp, tpr_mlp)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.show()
plot_roc_curve(y_test,X_test,xgb_model)


# In[ ]:


plot_importance(xgb_model).figure.set_size_inches(10,8);


# In[ ]:


plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);
plt.show();


# ## UNDERSAMPLING-XGBOOST MODEL

# In[ ]:



rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)


# In[ ]:


X_resampled.shape,y_resampled.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.20, 
                                                    random_state=42)


# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


cross_val_score(xgb_model,X_resampled,y_resampled).mean()


# In[ ]:


conf_matrix(y_test,y_pred)


# In[ ]:


plot_roc_curve(y_test,X_test,xgb_model)


# In[ ]:


roc_auc_score(y_test,y_pred)


# In[ ]:


plot_importance(xgb_model).figure.set_size_inches(10,8);


# In[ ]:


plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);
plt.show();


# ## OVERSAMPLING-XGBOOST

# In[ ]:



ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)


# In[ ]:


X_resampled.shape,y_resampled.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.20, 
                                                    random_state=42)


# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


cross_val_score(xgb_model,X_resampled,y_resampled).mean()


# In[ ]:


conf_matrix(y_test,y_pred)


# In[ ]:


plot_roc_curve(y_test,X_test,xgb_model)


# In[ ]:


roc_auc_score(y_test,y_pred)


# In[ ]:


#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);
plt.show();


# In[ ]:



plot_importance(xgb_model).figure.set_size_inches(10,8);


# ## RANDOM FOREST MODEL

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42)


# In[ ]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[ ]:


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


cross_val_score(rf_model,X,y).mean()


# In[ ]:


conf_matrix(y_test,y_pred)


# In[ ]:


plot_roc_curve(y_test,X_test,rf_model)


# ## RANDOM FOREST-UNDERSAMPLING

# In[ ]:


rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)


# In[ ]:


X_resampled.shape,y_resampled.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.20, 
                                                    random_state=42)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[ ]:


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


cross_val_score(rf_model,X_resampled,y_resampled).mean()


# In[ ]:


conf_matrix(y_test,y_pred)


# In[ ]:


plot_roc_curve(y_test,X_test,rf_model)


# In[ ]:


roc_auc_score(y_test,y_pred)


# ## RANDOM FOREST - OVERSAMPLING

# In[ ]:


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)


# In[ ]:


X_resampled.shape, y_resampled.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.20, 
                                                    random_state=42)


# In[ ]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[ ]:


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


cross_val_score(rf_model,X_resampled,y_resampled).mean()


# In[ ]:


conf_matrix(y_test,y_pred)


# In[ ]:


plot_roc_curve(y_test,X_test,rf_model)


#  **I'm trying to improve myself so I'm open all your idea.Please write your suggestions for me in the comments! :)**
