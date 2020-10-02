#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from pandas import DataFrame 
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data_train = pd.read_csv('/kaggle/input/healthcare-dataset-stroke-data/train_2v.csv',low_memory=False,skipinitialspace=True)
data_test = pd.read_csv('/kaggle/input/healthcare-dataset-stroke-data/test_2v.csv',low_memory=False,skipinitialspace=True)


# In[ ]:


print(data_test.shape)
print(data_train.shape)
print(data_test.columns)
print(data_train.columns)


# **Since there is no stroke column available in test dataset , considering only Train dataset for whole analysis**

# In[ ]:


data_stroke =data_train


# In[ ]:


data_stroke.head()


# In[ ]:


data_stroke.isnull().sum()


# **EDA on dataset**

# In[ ]:


sns.heatmap(data_stroke.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# **A countplot shows the counts of observations in each categorical bin using bars.**

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="stroke", data=data_stroke, color="c")
plt.show()


# In[ ]:


data_stroke['age'].hist(bins=100)


# In[ ]:


sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==1), 
            'age'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==0),  
            'age'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Age') 
plt.ylabel('Probability Density') 


# In[ ]:


sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==1), 
            'avg_glucose_level'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==0),  
            'avg_glucose_level'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Avg_glucose_level') 
plt.ylabel('Probability Density') 


# In[ ]:


sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==1), 
            'hypertension'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==0),  
            'hypertension'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('hypertension') 
plt.ylabel('Probability Density') 


# In[ ]:


sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==1), 
            'bmi'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(data_stroke.loc[(data_stroke['stroke']==0),  
            'bmi'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Body Mass Index') 
plt.ylabel('Probability Density') 


# In[ ]:





# **Handling Missing Data**

# In[ ]:


data_stroke['bmi'].fillna(data_stroke['bmi'].mean(),inplace=True)


# **Handling the Categorical columns**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
labelEncoder = LabelEncoder()
data_stroke['gender'] = labelEncoder.fit_transform(data_stroke['gender'])
data_stroke['ever_married'] = labelEncoder.fit_transform(data_stroke['ever_married'])
data_stroke['work_type'] = labelEncoder.fit_transform(data_stroke['work_type'])
data_stroke['Residence_type'] = labelEncoder.fit_transform(data_stroke['Residence_type'])


# In[ ]:


data_stroke.isnull().sum()


# In[ ]:


print(data_stroke.smoking_status.value_counts())
print(data_stroke[data_stroke.smoking_status.isnull()]['stroke'].value_counts())


# ** Dropping the smoking_status column , Since 30% of data is missing**

# In[ ]:


data_stroke.drop('smoking_status',axis = 1, inplace = True)


# **ID column is not required in this Logistic Regression Prediction **

# In[ ]:


data_stroke.drop('id',axis = 1, inplace = True)


# In[ ]:


data_stroke.isnull().sum()


# **UnderSampling technique**
#  
# since the data is imbalanced , the undersampling technique is processed

# In[ ]:


data_shuffled = data_stroke.sample(frac=1,random_state=4)
df_Isstroke = data_stroke.loc[data_stroke['stroke'] == 1]
df_Nostroke = data_stroke.loc[data_stroke['stroke'] == 0].sample(n= 4500,random_state= 101)


# In[ ]:


df_data_stroke = pd.concat([df_Isstroke,df_Nostroke])


# In[ ]:


df_data_stroke.stroke.value_counts()


# **Split the data as train and test**

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
df_data_stroke = shuffle(df_data_stroke)
X = df_data_stroke.drop('stroke', axis = 1)
y = df_data_stroke['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)


# In[ ]:


print('X Train dataset shapes',X_train.shape)
print('Y Train dataset shapes',y_train.shape)
print('X Test dataset shapes',X_test.shape)
print('Y Test dataset shapes',y_test.shape)


# **Applying Logistic Regression Model**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logRe = LogisticRegression()
logRe.fit(X_train,y_train)


# In[ ]:


predictions = logRe.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))
logRe.score(X_test, y_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


from sklearn import metrics
sns.heatmap(confusion_matrix(y_test,predictions), annot= True, fmt=".2f")
plt.title('Logistic Regression \n Confusion Matrix', fontsize=14)
print(roc_auc_score(y_test, predictions))


# In[ ]:


test_stroke = y_test.values
for i in range(0, len(test_stroke)):
    if predictions[i] == test_stroke[i]:
        print ('Pred: {0} Actual:{1}'.format(predictions[i], test_stroke[i]))
    else:
        print('Wrong Prediction')
        print ('Pred: {0} Actual:{1}'.format(predictions[i], test_stroke[i]))


# **StatsModel api- LogisticRegression **

# In[ ]:


import statsmodels.api as sm

logit_model = sm.Logit(y_train,sm.add_constant(X_train))


# In[ ]:


result = logit_model.fit()


# In[ ]:


print(result.summary2())


# **Get the significant variables from logit model**

# In[ ]:


def get_significant_vars(lm):
    #Store the pvalues to corresponding columns
    df_p_vals = pd.DataFrame(lm.pvalues)
    df_p_vals['vars'] = df_p_vals.index
    df_p_vals.columns=['p_values','variables']
    return list(df_p_vals[df_p_vals['p_values']<=0.05]['variables'])


# In[ ]:


significant_var= get_significant_vars(result)
print(significant_var)


# In[ ]:


if 'const' in significant_var:
    significant_var.remove('const')
    logit_model1 = sm.Logit(y_train,sm.add_constant(X_train[significant_var]))


# In[ ]:


final_logit = logit_model1.fit()


# In[ ]:


print(final_logit.summary2())


# In[ ]:


# cols_significant = significant_var.remove('const')

print(X_test[significant_var].columns)
result =final_logit.predict(sm.add_constant(X_test[significant_var]))
Y_pred_df= pd.DataFrame({"actuals":y_test,"predicted_prob":result})
Y_pred_df.sample(100,random_state=50)


# **Optimal classification cut off value**
# 
# Lets assume the OCC value us 0.3 as of now

# In[ ]:


Y_pred_df['Predicted']= Y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.3 else 0)
Y_pred_df.sample(100,random_state=42)


# In[ ]:


def create_cm(actuals, predicted):
    sns.heatmap(confusion_matrix(actuals,predicted), annot= True, fmt=".2f")
    plt.title('Logistic Regression \n Confusion Matrix', fontsize=14)
    plt.ylabel('True Lable')
    plt.xlabel('Predicted Lable')
    plt.show()


# In[ ]:


create_cm(Y_pred_df.actuals,Y_pred_df.Predicted)


# In[ ]:


print(classification_report(Y_pred_df.actuals,Y_pred_df.Predicted))


# In[ ]:


def create_roc_auc(actuals, predicted):
    logit_roc_auc = roc_auc_score(actuals, predicted)
    fpr, tpr, thresholds = roc_curve(actuals, predicted )
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


# In[ ]:


create_roc_auc(Y_pred_df.actuals,Y_pred_df.Predicted)

