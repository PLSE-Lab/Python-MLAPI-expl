#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix,classification_report


# In[ ]:


data=pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')


# In[ ]:


data.head()


# In[ ]:


data.rename({'Dataset':'Liver_disorder'},axis=1,inplace=True)


# ### EDA

# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


data.dropna().shape


# In[ ]:


data['Albumin_and_Globulin_Ratio'].unique()


# In[ ]:


data[data['Albumin_and_Globulin_Ratio'].isnull()]


# ### Dropping 4 rows

# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data['Liver_disorder'].value_counts()


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


sns.pairplot(data)


# In[ ]:


data.corr()


# In[ ]:


data['Gender']=data['Gender'].replace({'Male':0,'Female':1})


# In[ ]:


data['Gender'].value_counts()


# In[ ]:


data.info()


# In[ ]:


data['Gender']=data['Gender'].astype('category')


# ##### lets perform statistical test on Gender to see if both the means are same or it is significantly different to predict (good predictor):As gender  and Liver_disorder both are categorical we will be doing 2 sample proportion test(popularly called as Z-test).

# In[ ]:


#no_of_MF_with_disorder=data[data['Liver_disorder']==1].groupby('Gender')[['Gender']].count()


# In[ ]:


#no_of_MF_with_disorder


# #323 female have liver_disorder
# 
# #91 Male have liver_disorder
# 
# 

# In[ ]:


tab=pd.crosstab(data.Gender,data.Liver_disorder)
print(tab)


# In[ ]:


#tot=data.groupby('Gender')[['Gender']].count()


# In[ ]:


#tot


# #### HNULL= proportion of liver disorder in male = proportion of liver disorder in female
# #### Halternative= proportions are not equal

# In[ ]:


tot_male=tab.iloc[1].sum()


# In[ ]:


male_with_disorder=tab.iloc[1,0]


# In[ ]:


tot_female=tab.iloc[0].sum()


# In[ ]:


female_with_disorder=tab.iloc[0,0]


# In[ ]:


p1=male_with_disorder/tot_male


# In[ ]:


p2=female_with_disorder/tot_female


# In[ ]:


p=(male_with_disorder+female_with_disorder)/data.shape[0]


# In[ ]:


Zstats=(p1-p2)/(np.sqrt(p*(1-p)*((1/tot_male)+(1/tot_female))))


# In[ ]:


Zstats


# In[ ]:


import scipy.stats as stats
stats.norm.cdf(Zstats)


# ### pval<<0.05 we can reject the H_null i.e the proportion is not the same between male and female so it can be a one of the good predictor of Y

# In[ ]:


cor=data.corr()


# In[ ]:


cor_target=abs(cor['Liver_disorder'])


# In[ ]:


Pearson_Coeff=pd.DataFrame(cor_target)
print(Pearson_Coeff)


# ### This is the co-relationship between X's and Y

# ##### Lets drop protein beacuse it's not significant predictor of Y the co-relation co-eff is just 3%

# In[ ]:


data.info()


# In[ ]:


data.drop('Total_Protiens',axis=1,inplace=True)


# #### Now lets check if multi-colinearity exist between independent X's:Using VIF

# In[ ]:


d=data.drop('Liver_disorder',axis=1)


# In[ ]:


features = "+" .join(d)
#y, X = dmatrices('annual_inc ~' + features, df, return_type='dataframe')


# In[ ]:


features


# In[ ]:


from   patsy                                import dmatrices
from   statsmodels.stats.outliers_influence import variance_inflation_factor
y,X=dmatrices('Liver_disorder~'+features,data,return_type='dataframe')


# In[ ]:


Vif=pd.DataFrame()


# In[ ]:


Vif['features']=X.columns
Vif['VIF factor']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

print(Vif)


# ####  Multi-co-linearity: Total_Bilirubin and Direct Bilirubin have High VIF so lets Drop one of the Column.

# In[ ]:


LData=data.drop('Direct_Bilirubin',axis=1)


# In[ ]:


LData.head()


# ### Time to apply Logistic Regression:

# Lets do Grid Search: To find best Parameters:Using k-fold cross validation

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
penalty=['l2','l1']
multi_class=['ovr', 'auto']
X=LData.drop('Liver_disorder',axis=1)
y=LData[['Liver_disorder']]
model=LogisticRegression()
grid=GridSearchCV(estimator=model,cv=3,param_grid=dict(penalty=penalty,multi_class=multi_class))
grid.fit(X,y)


# In[ ]:


print(grid.best_params_)


# In[ ]:


print('Recall:',grid.best_score_)
print('Accuracy:',grid.best_score_)


# #### Let's use train lets split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=2)


# In[ ]:


y_train.shape


# In[ ]:


model1=LogisticRegression()
model1.fit(X_train, y_train)


# In[ ]:


preds= model1.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,preds))


# In[ ]:




