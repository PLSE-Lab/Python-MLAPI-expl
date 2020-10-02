#!/usr/bin/env python
# coding: utf-8

# **Data Gathering and Data Preprocessing ****

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.dtypes.sort_values().to_frame('feature_type').groupby(by = 'feature_type').size().to_frame('count').reset_index()


# In[ ]:


df_dtypes = pd.merge(df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[ ]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[ ]:


df.describe().round()


# In[ ]:


def find_constant_features(dataFrame):
    const_features = []
    for column in list(dataFrame.columns):
        if dataFrame[column].unique().size < 2:
            const_features.append(column)
    return const_features


# In[ ]:


const_features = find_constant_features(df)


# In[ ]:


const_features


# **Remove Duplicate Rows**

# In[ ]:


df.drop_duplicates(inplace= True)


# In[ ]:


df.shape


# *********Remove duplicate columns*********

# In[ ]:


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


# In[ ]:


duplicate_cols = duplicate_columns(df)


# In[ ]:


duplicate_cols
df.shape


# In[ ]:


df.columns


# In[ ]:


sns.countplot('Class', data=df)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=11)


# In[ ]:



sns.catplot(x="Class",y="Amount",kind="bar",data=df);


# In[ ]:


amount = df['Amount'].values
time= df['Time'].values


# In[ ]:


sns.distplot(amount,bins=20,color='r')


# In[ ]:


sns.distplot(time,bins=50,color='r')


# ************Correlation and Correlation with Target Variable********

# In[ ]:


df.describe().T


# In[ ]:


from sklearn.utils import shuffle, class_weight
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:



corr = df.corr(method = 'spearman')


# In[ ]:


layout = cf.Layout(height=600,width=600)
corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale = 'RdBu')


# ****************Find highly correlated features************

# In[ ]:


new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)


# In[ ]:


new_corr[new_corr.correlation > 0.3]


# In[ ]:


corr_with_target = df.corrwith(df.Class).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)
unique_values = df.nunique().to_frame('unique_values').reset_index()
corr_with_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')


# In[ ]:


corr_with_unique


# In[ ]:


df_major=df[df.Class==0]


# In[ ]:


df_minor=df[df.Class==1]


# In[ ]:


df_major.shape


# In[ ]:


from sklearn.utils import resample


# In[ ]:


df_minor_upsmapled = resample(df_minor, replace = True, n_samples = 283253, random_state = 2018)


# In[ ]:


df_minor_upsmapled.shape


# In[ ]:


final_data=pd.concat([df_minor_upsmapled,df_major])


# In[ ]:


final_data.shape


# ************Data Splitation and Standerdization************

# In[ ]:


X = final_data.drop('Class', axis = 1)
Y = final_data.Class


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)


# In[ ]:


mms = StandardScaler()
mms.fit(xtrain)
xtrain_scaled = mms.transform(xtrain)
xtest_scaled = mms.transform(xtest)


# In[ ]:


def evaluate_model(ytest, ypred, ypred_proba = None):
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


logisticRegr = LogisticRegression()


# In[ ]:


logisticRegr.fit(xtrain_scaled, ytrain)


# In[ ]:


lr_pred = logisticRegr.predict(xtest_scaled)


# In[ ]:


evaluate_model(ytest, lr_pred)


# **********Random Forest**************

# In[ ]:


def random_forest(xtrain, xtest, ytrain):
    rf_params = {
        'n_estimators': 126, 
        'max_depth': 14
    }

    rf = RandomForestClassifier(**rf_params)
    rf.fit(xtrain, ytrain)
    rfpred = rf.predict(xtest)
    rfpred_proba = rf.predict_proba(xtest)
    
    return rfpred, rfpred_proba, rf


# In[ ]:


rfpred, rfpred_proba, rf = random_forest(xtrain_scaled, xtest_scaled, ytrain)


# In[ ]:


from sklearn.metrics import recall_score, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score,                             classification_report, confusion_matrix


# In[ ]:


evaluate_model(ytest, rfpred, rfpred_proba)

