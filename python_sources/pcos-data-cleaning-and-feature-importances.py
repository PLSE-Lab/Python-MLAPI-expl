#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Introduction 

# I have been working on **fastai** courses for quite sometime. This is a technique I came across in the fastai [Introduction to Machine Learning for coders](http://course18.fast.ai/ml)

# # Model Based EDA
# 
# In the course, Jeremy Howard took us through the following procedure for EDA. Rather than looking at the data and finding relationships and interactions  between the features, the course suggest fitting a model on the data and looking at the model importances and getting the intution from the model itself. 
# 
# Using this approach helps us to not all prey to any biases that we form from the features. This approach helps to find:
# * Important Features.
# * Redundant Features.
# * Feature Interactions.

# # Loading the data

# In[ ]:


PCOS_inf = pd.read_csv("../input/polycystic-ovary-syndrome-pcos/PCOS_infertility.csv")
PCOS_data = pd.read_csv("../input/polycystic-ovary-syndrome-pcos/data without infertility _final.csv")


# After loading the data, lets print the data to have a look at the data. Remember, We are not looking at the features in the data. We are just making sure that all the data has been loaded correctly

# In[ ]:


PCOS_data.head().T


# Looks like the is some discrepancy in the data. We can see the last row (as transpose of the head is displayed) has `Unnamed:42`. Lets Check the data if we can correct it

# In[ ]:


PCOS_data[~ PCOS_data['Unnamed: 42'].isna()].T


# We have extracted the two observations in the `Unnamed: 42` column. Looks like there was some mistake with entering the data. Lets look at the other columns to check if there are other mistake as well.

# In[ ]:


PCOS_data.info()


# We can see there is 1 null value in `Marraige Status (Yrs)`

# In[ ]:


PCOS_data[PCOS_data['Marraige Status (Yrs)'].isnull()].T


# In[ ]:


#lets assign the median to the missing data
PCOS_data['Marraige Status (Yrs)'].fillna(PCOS_data['Marraige Status (Yrs)'].median(),inplace=True)


# In[ ]:


PCOS_data['Fast food (Y/N)'].fillna(PCOS_data['Fast food (Y/N)'].median(),inplace=True)


# Looks like we can just drop the last erroneous column and go ahead with the analysis

# ## Dropping the column `Unnamed: 42`

# In[ ]:


PCOS_data.drop('Unnamed: 42',axis=1,inplace=True)


# In[ ]:


PCOS_inf.head()


# In[ ]:


PCOS_inf.info()


# ## Merging the two dataframes

# In[ ]:


data = pd.merge(PCOS_data,PCOS_inf, on='Patient File No.', suffixes={'','_y'},how='left')


# In[ ]:


data.columns = ['SNo', 'Patient_File_No.', 'PCOS_(Y/N)', 'Age_(yrs)', 'Weight_(Kg)',
       'Height(Cm)', 'BMI', 'Blood_Group', 'Pulse_rate(bpm)',
       'RR_(breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle_length(days)',
       'Marriage_Status_(Yrs)', 'Pregnant(Y/N)', 'No_of_aborptions',
       'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)',
       'Waist:Hip_Ratio', 'TSH_(mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
       'Vit_D3_(ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight_gain(Y/N)',
       'hair_growth(Y/N)', 'Skin_darkening (Y/N)', 'Hair_loss(Y/N)',
       'Pimples(Y/N)', 'Fast_food_(Y/N)', 'Reg_Exercise(Y/N)',
       'BP_Systolic(mmHg)', 'BP_Diastolic(mmHg)', 'Follicle_No.(L)',
       'Follicle_No.(R)', 'Avg.Fsize(L)(mm)', 'Avg.Fsize(R)(mm)',
       'Endometrium(mm)', 'Sl.No_y', 'PCOS(Y/N)_y',
       'I_beta-HCG(mIU/mL)', 'II_beta-HCG(mIU/mL)', 'AMH(ng/mL)_y']


# In[ ]:


data.drop(['Sl.No_y', 'PCOS(Y/N)_y','AMH(ng/mL)_y'],axis=1,inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# We have successfully loaded the data. 

# # Fitting a Model

# Before fitting the model, we will have to split our data into **train**, **valid** and **test** sets. We can use sklearn's train_test_split function to split our data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
target = data['PCOS_(Y/N)']
data.drop('PCOS_(Y/N)',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(8,7))
sns.countplot(target)
plt.title('Data imbalance')
plt.show()


# In[ ]:


X_train,X_test, y_train, y_test = train_test_split(data, target, test_size=0.15, random_state=1, stratify = target)
X_train,X_valid, y_train, y_valid =  train_test_split(X_train, y_train, test_size=0.3, random_state=1, stratify=y_train)


# In[ ]:


from sklearn.metrics import roc_auc_score
def print_scores(m):
    res = [roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])]
    for r in res:
        print(r)


# In[ ]:


rf = RandomForestClassifier(n_jobs=-1,n_estimators=150,max_features='sqrt',min_samples_leaf=10)
rf.fit(X_train,y_train)
print_scores(rf)


# In[ ]:


from sklearn.metrics import roc_curve
y_pred_proba = rf.predict_proba(X_valid)[:,1]
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)


# In[ ]:


plt.figure(figsize=(8,7))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# Now we are getting a high roc auc score, lets start with out Exploratory Data Analysis

# # Exploratory Data Analysis

# In[ ]:


def get_fi(m, df):
    return pd.DataFrame({'col': df.columns, 'imp': m.feature_importances_}).sort_values('imp',ascending=False)

#lets get the feature importances for training set
fi = get_fi(rf,X_train)


# In[ ]:


def plot_fi(df):
    df.plot('col','imp','barh',figsize=(10,10))
    
plot_fi(fi)


# ## Observations
# * We can see that the top features are:
#     1. Follicle_No.(R)
#     2. Follicle_No.(L)
#     3. hair_growth(Y/N)
#     4. Skin_darkening (Y/N)
#     5. Weight_gain(Y/N)
#     6. Fast_food_(Y/N)
#     7. Cycle(R/I)
#     8. AMH(ng/mL)
#     9. Cycle_length(days)
#     10. Pimples(Y/N)

# Lets plot data important features and look if we can find some interesting relationships

# # To be continued...

# In[ ]:




