#!/usr/bin/env python
# coding: utf-8

# # Vehicle Loan Default Prediction

# -------------------------------------------------------------------------
# In this kernel we will analyse and make predictions of this dataset
# 
# --------------------------------------------------------------------------

# We begin by importing the different librairies that will be needed for our analysis

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#Scikit learn librairies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# We charge the dataset and transform it into a pandas DataFrames

# In[ ]:


df= pd.read_csv('../input/train.csv')
dftmp= pd.read_csv('../input/train.csv')
dftmp2= pd.read_csv('../input/train.csv')
#dfe= pd.read_csv('test.csv')
#dftest= pd.read_csv('loan_car_short.csv')
#df=dftest
#print(dftest.shape,df.shape,df.columns)


# # Exploratory Data Analysis

# In[ ]:


df.head()


# In[ ]:


df.columns


# We plot the distribution of the target

# In[ ]:


fig11=plt.figure()
ax11=plt.axes()
the_target = dftmp['loan_default']
the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)
plt.title('Target repartition')
ax11 = ax11.set(xlabel='Default proportion', ylabel='Number of people')
the_target.value_counts().plot.bar()
plt.show()


# We want to plot the correlation matrix of the data

# In[ ]:


#Correlation Matrix calculation
corr_mat = df.corr()

fig2=plt.figure()
sns.set(rc={'figure.figsize':(20,15)})
k = 10
cols = corr_mat.nlargest(k, 'loan_default')['loan_default'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title('Correlation Matrix')
plt.show()


# An other good way to understant the data is to print the sorted correlation coefficients of the matrix

# In[ ]:


corr_mat['loan_default'].sort_values(ascending = False)


# To have a clearer view of the data we created a function to print every value for every features in function of the id of the custumer to have a better idea of the shape of the data

# In[ ]:


#A function to print every graph with the ID as 
def print_all_values():
    df1=df.drop('UniqueID',axis=1)
    cols=df1.columns
    for col in cols:
        if (df[col].dtypes !='object'):

            fig1=plt.figure()
            ax1=plt.axes()
            plt.scatter(df[[col]],df.UniqueID,alpha=1,)
            plt.title(col)
            ax1 = ax1.set(xlabel=col, ylabel='ID')
            plt.show()
            
            
print_all_values()


# Some features have to high or to low values, we want to delete them to help our model on regular values.
# 
# We decided to create a function that takes as parameters the quatile to compare the value to. We multiplie this value with a max value to fix a value limite
# 
# We are targeting columns that have disproportionated values. We are dropping to high value to concentrate more on mid range value.
# Custumers with extreme values (good or bad) are easy to predict, we want to improve our model on average custumer

# In[ ]:


#Delete to high or to low values
def delete_absurd_values(df_transformed,cols,max_value,percentage):
        
        
        for col in cols:
            if (df_transformed[col].dtypes !='object'):
                       
                q99=df_transformed[col].quantile(q=percentage)
                q01=df_transformed[col].quantile(q=(1-percentage))
                for i in df_transformed.index:
                    
                    if (df_transformed.loc[i,col]> max_value*q99 or df_transformed.loc[i,col]< q01/max_value):
                        df_transformed=df_transformed.drop(index=i)
        
        return df_transformed

cols=['disbursed_amount', 'asset_cost', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT','PRI.SANCTIONED.AMOUNT',
       'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS','SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
       'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES']
df=delete_absurd_values(df,cols,5,0.999)


# Let's plot the repartition of the target

# In[ ]:


#The repartition of the target
fig7=plt.figure()
ax7=plt.axes()
the_target = dftmp2['loan_default']
the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)
plt.title('Target repartition')
ax7 = ax7.set(xlabel='Default proportion')
the_target.value_counts().plot.pie()
plt.show()


# In[ ]:


#Printing the types of the features
df.dtypes


# Let's see the percentage of missing value for each column

# In[ ]:


def nan_count_df(df_to_print):
    
    nan_count = df_to_print.isnull().sum()

    nan_percentage = (nan_count / len(df))*100

    nan_df=pd.concat([nan_percentage], axis=1)
    nan_df=nan_df.rename(columns={0:'Percentage'})
    nan_df=nan_df[nan_df.Percentage != 0]
    nan_df = nan_df.sort_values(by='Percentage',ascending=False)
    return nan_df

nan_count_df(df)


# So there is only one column with missing values: "Employment Type"

# In[ ]:


#Number of unique values
df.nunique()


# # Feature Engineering

# In[ ]:


df=df.rename(columns={'Date.of.Birth': 'Date_of_Birth','Employment.Type': 'Employment_Type', 'PERFORM_CNS.SCORE.DESCRIPTION': 'PERFORM_CNS_SCORE_DESCRIPTION'})

df.columns


# We convert the column 'Date_of_Birth' into the age of the custumer for more clearancy

# In[ ]:


now = pd.Timestamp('now')
df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], format='%d-%m-%y')
df['Date_of_Birth'] = df['Date_of_Birth'].where(df['Date_of_Birth'] < now, df['Date_of_Birth'] -  np.timedelta64(100, 'Y'))
df['Age'] = (now - df['Date_of_Birth']).astype('<m8[Y]')
df=df.drop('Date_of_Birth',axis=1)


# # Label Encoding and One-Hot Encoding

# We want to create a function for encoding the two categories variables

# In[ ]:


#Creating a function for encoding 2 categories features
def two_cat_encoding(df_to_transf):
    le = LabelEncoder()

    for cols in df_to_transf:
        if df_to_transf[cols].dtype == 'object':
            if len(list(df_to_transf[cols].unique())) == 2:
                le.fit(df_to_transf[cols])
                df_to_transf[cols] = le.transform(df_to_transf[cols])
    return df_to_transf
df=two_cat_encoding(df)


# There are two many different value type possible for a missing value so we will merge them into one column to reduce the dimension of the probleme when applying oneHotEncoding (dummies)

# In[ ]:


df['PERFORM_CNS_SCORE_DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found', 'Not Scored: No Activity seen on the customer (Inactive)','Not Scored: No Updates available in last 36 months','Not Enough Info available on the customer','Not Scored: Only a Guarantor','Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer'], value= 'Not Scored', inplace = True)


# We will now drop the features that does not seem to improve our model

# In[ ]:


columns_to_drop = ['UniqueID','MobileNo_Avl_Flag','DisbursalDate','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','SEC.OVERDUE.ACCTS']
df=df.drop(columns=columns_to_drop)


# We finally apply the oneHotEncoding methode to the features that hasn't been encodeed yet

# In[ ]:


df = pd.get_dummies(df)
df.columns


# We want to know the new shape of the data after the encoding

# In[ ]:


df.shape


# And we also want to make sure that there are no 'object' type that hasn't been encodeed left:

# In[ ]:


df.dtypes.value_counts()


# ---------------------------------------------------------------------------------------------
# #                   Machine Learning Algorithms
# ----------------------------------------------------------------------------------------------

# #### Spliting the data between training and testing

# In[ ]:


X =df.drop('loan_default',axis=1)
y = df['loan_default']  

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#   # Logistic Regression Implementation

# In[ ]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(C=1.0, class_weight=None,fit_intercept=True,max_iter=100)
logisticRegr.fit(X_train, y_train)


# In[ ]:


#ERROR
error = (1 - logisticRegr.score(X_test, y_test))*100
print('Score  = ',logisticRegr.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')


# # Random Forest Classifier Implementation

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0)
rf.fit(X_train,y_train)
error = (1 - rf.score(X_test, y_test))*100
print('Score  = ',rf.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')


# # Linear Discriminant Analysis Implementation

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)  
X_lda_sklearn = lda.fit_transform(X_train, y_train)
error = (1 - lda.score(X_test, y_test))*100
print('Erreur: %f' % error, '%')


# # Decision Tree Classifier Implementation

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
error = (1 - clf.score(X_test, y_test))*100
print('Erreur: %f' % error, '%')


# In[ ]:





# # Cross Validation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
predictions = logisticRegr.predict(X_test)

print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator = logisticRegr , 
                         X=X_train, 
                         y=y_train, 
                         cv=3)
print('Cross-validation accuracy scores: %s' %(scores))
print('CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[1]:


print('Success rate by model:\n\nLogistic Regression:',logisticRegr.score(X_test, y_test)*100,'%','\n\nLDA:',lda.score(X_test, y_test)*100,'%','\n\nRandom Forest Classifier:',rf.score(X_test, y_test)*100,'%','\n\nDecision Tree Classifier:',clf.score(X_test, y_test)*100,'%')


# In[ ]:




