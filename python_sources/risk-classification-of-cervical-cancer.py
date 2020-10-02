#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# **Table Of Content**
# 
# * Importing Essential Libraries 
# * Loading CSV file
# * Univariate analysis
#     * Categorising Dat
#     * Dropping Unnecessary columns
# * Outlier Treatment
# * Exploratory Data Analysis
# * Statistica Approach
#       *Check for Normality
#       *Test of Variance
# * Model Building
#       * Decision Tree
#       * Random Forest
#       * Bagged model
# * Model evaluation
#              
#    

# **OBJECTIVE:**
# 
# A risk factor is anything that increases a person's chance of developing cancer.
# Although risk factors often influence the development of cancer, most do not directly cause cancer. 
# Some people with several risk factors never develop cancer, while others with no known risk factors do. 
# Knowing your risk factors and talking about them with your doctor may help you make more informed lifestyle 
# and health care choices.

# > **Importing essential Libraries**

# In[ ]:


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import  warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# > **Load and describe Data**

# In[ ]:


df=pd.read_csv('/kaggle/input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# **![](http://)> Categorising data **

# In[ ]:


num_cols=numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']


cat_cols=['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']


# **Dropping unneccessary columns**[](http://)

# In[ ]:


#we are dropping the STDs: column because these are of not of much use for our analysis bcz of having a lot of missing values

df.drop(['STDs: Time since last diagnosis','STDs: Time since first diagnosis'],inplace=True,axis=1)


# In[ ]:


df.head()


# **Missing value treatment**[](http://)

# In[ ]:


df=df.replace('?',np.NaN)


# In[ ]:


# now we  fill missing values in numerical columns with mean of numerical data 
for feature in num_cols:
    print(feature,'',df[feature].astype(float).mean())
    feature_mean = round(df[feature].astype(float).mean(),1)
    df[feature] = df[feature].fillna(feature_mean)


# In[ ]:


for features in cat_cols:
    df[features]=df[features].astype(float).fillna(1.0)


# **Exploaratry Data analysis**[](http://)

# In[ ]:


for feature in cat_cols:
    sns.factorplot(feature,data=df,kind='count')


# from the above graph it is clear that the harmonal contraceptive have the highest key feature that affect
# the cervical cancer .so lets concentrate more on this for our furthur analysis

# In[ ]:


g = sns.PairGrid(df,
                 y_vars=['Hormonal Contraceptives'],
                 x_vars= cat_cols,
                 aspect=2.75, size=10.5)
g.map(sns.barplot, palette="pastel");


# In[ ]:


df['Number of sexual partners'] = round(df['Number of sexual partners'].astype(float))
df['First sexual intercourse'] = df['First sexual intercourse'].astype(float)
df['Num of pregnancies']=round(df['Num of pregnancies'].astype(float))
df['Smokes'] =df['Smokes'].astype(float)
df['Smokes (years)'] =df['Smokes (years)'].astype(float)
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].astype(float)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].astype(float)
df['IUD (years)'] = df['IUD (years)'].astype(float)

print('minimum:',min(df['Hormonal Contraceptives (years)']))
print('maximum:',max(df['Hormonal Contraceptives (years)']))


# In[ ]:


plt.hist(df['Age'].mean(),bins=20)
plt.xlabel('Age')                                 # age estimation for the risk of cervical cancer 
plt.ylabel('count')
print('mean age of woman facing the cervical cancer ',df['Age'].mean())


# In[ ]:


for feature in cat_cols:
    fig=sns.FacetGrid(df,hue=feature)
    fig.map(sns.kdeplot,'Age',shade=True)
    max_age=df['Age'].max()
    fig.set(xlim=(0,max_age))
    fig.add_legend()


# the density plot shows that the woman who are having age of 26 are having a more chance of getting affected.
# and also the woman age group of age between 20 to 35 are having a more chance of having cancer.The peak of age is at 50 and the extension shows the furthur too.

# In[ ]:


for feature in cat_cols:
    plt.figure(figsize=(10,8))
    sns.factorplot(x='Number of sexual partners',y='Age',hue=feature,data=df,kind='bar')


# as the age increases the number of sexual partners increases too ,
# 
# causing to be one of the factor with getting a cancerous tumour

# In[ ]:


sns.distplot(df['First sexual intercourse'].astype(float))


# plot depicts the woman started their first sexual intercourse at the age ranging 15-20 

# In[ ]:


sns.scatterplot(x='First sexual intercourse',hue='Dx:Cancer',y='Num of pregnancies',data=df)


# the relatioal plot depicts that the women who did their first  sexual itercourse at age of 18-20 and their number of pregnancies ranging (1-3)are more prone to cancer tumour

# ### statistical approach

# #### To check  if there age is  impacting the cancer ?

# In[ ]:


df['Dx:Cancer'].value_counts()


# In[ ]:


A1=df.groupby(['Dx:Cancer'])


# In[ ]:


cancer_patients=A1.get_group(1)
cancer_free_patients=A1.get_group(0)


# ###### unpaired t test

# In[ ]:


age_affected_cancer=cancer_patients['Age']
age_notaffected=cancer_free_patients['Age']


# In[ ]:


age_affected_cancer.describe()


# In[ ]:


age_notaffected.describe()


# In[ ]:


plt.hist(age_affected_cancer)


# In[ ]:


df.boxplot(column='Age',by='Dx:Cancer')


# Hypothesis 

# H0:mu(age)cancer=mu(age)not_affected
# 
# H1:mu(age)cancer!=mu(age)not_affected

# ##### check for normality

# In[ ]:


from scipy.stats import shapiro
print(shapiro(age_affected_cancer))
print(shapiro(age_notaffected))


# hence the pvalue is greater than alpha so we will accept the null hypothesis this interprets that the data is normal

# ##### check for variance

# In[ ]:


from scipy.stats import levene


# In[ ]:


levene(age_affected_cancer,age_notaffected)


# In[ ]:


from scipy.stats import ttest_ind


# In[ ]:


ttest_ind(age_affected_cancer,age_notaffected)


# ###### so this interprets that the pvalue is less than alpha(0.05) hence from 95% cnfidence we can conclude our judgment that the mean age of poulation is not affecting the cancer occurnce

# ### check whether the fist sexual intercourse is impacting the cancerous tumour ?

# In[ ]:


df['Dx:Cancer'].value_counts()


# In[ ]:


s=df.groupby(['Dx:Cancer'])


# In[ ]:


s1=s.get_group(1)
s2=s.get_group(0)


# In[ ]:


sexual_intercourse=s1['First sexual intercourse']
no_sexual_intercourse=s2['First sexual intercourse']


# In[ ]:


sexual_intercourse.describe()


# In[ ]:


no_sexual_intercourse.describe()


# Hypothesis
# 
# H0:mean(first_sexual_intercourse)=mean(no_sexual_intercourse)
#     
# H1:mean(first_sexual_intercourse)!=mean(no_sexual_intercourse)

# ###### check for normality

# In[ ]:


print(shapiro(sexual_intercourse))   #H0:data follows normality   H1:data not following normality
shapiro(no_sexual_intercourse)


# ###### the normality test interprets that the p value is geater than alpha(0.05) hence we will accept the null hypothesis andcan conclude that the data is following normal and is parametric ,hence next we will check for the variance between the data so to satisfy the assumptions

# #### variance test

# In[ ]:


levene (sexual_intercourse,no_sexual_intercourse)  #H0:variance of (sexual_intercourse)having cancer=var(no_sexual_intercourse)having cancer
                                                    #H1:variance of (sexual_intercourse)having cancer!=var(no_sexual_intercourse)having cancer


# the variance test interprets that the pvalue is greater than alpha hence we will accept the null hypothesis and can conclude
# that the variance of (sexual_intercourse)having cancer==variance (no_sexual_intercouse)having cancer

# ###### though our both variables passed the test of normality and test of variance hence the variables are parametric so we wil go through the unpaired ttest.

# ##### Test of independency

# In[ ]:


ttest_ind(sexual_intercourse,no_sexual_intercourse)


# #### though our pvalue is less than alpha hence we will reject the null-hypothesis and with 95% confidence we can judge that the mean of first_sexua_intercourse=mean of no_sexual_intercourse so there is a impact of first sexual intercourse over thecause of cancer tumour 

# ## Building the model

# In[ ]:


from sklearn.model_selection import cross_val_score # Cross Validation Score
from sklearn.model_selection import GridSearchCV # Parameters of the Model
from sklearn.model_selection import RandomizedSearchCV # Tuning the Parameters
from sklearn.tree import DecisionTreeClassifier # Decision Tree Algo
from sklearn.ensemble import RandomForestClassifier # Random Forest Algo.
from sklearn.model_selection import train_test_split # helps in spliting the data in train and test set
from sklearn.metrics import accuracy_score # Calculating the Accuracy Score againts the Classes Predicted vs Actuals.
from sklearn.ensemble import BaggingClassifier


# ### Decision tree model

# In[ ]:


#defining my Xs and Ys
x=df.drop('Dx:Cancer',axis=1) #dropping the target
y=df['Dx:Cancer']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)


# In[ ]:


#creatig our first model called Decisiontree
tree=DecisionTreeClassifier()

#defining tree params for grid based search
tree_params={
    "criterion":["gini", "entropy"],
    "splitter":["best", "random"],
    "max_depth":[3,4,5,6],
    "max_features":["auto","sqrt","log2"],
    "random_state": [123]
}


# In[ ]:


# apply grid search algorithm

grid=GridSearchCV(tree,tree_params,cv=10)
grid


# In[ ]:


#lets fit into data so that it can giuve the best params

best_param_search=grid.fit(x_train,y_train)

best_param_search.best_params_


# In[ ]:


#creating our first model called decision tree after hypertuning
 
tree2=DecisionTreeClassifier(criterion='entropy',max_depth=5,max_features= 'auto',random_state= 123,splitter='best')

#Developiug the model
model_tree=tree.fit(x_train,y_train)
pred_tree=tree.predict(x_test)
accuracy_score(y_test,pred_tree)


# ### Random forest model

# In[ ]:


Rf_model=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0) 


# ### Bagged model

# In[ ]:


bag_model=BaggingClassifier(n_estimators=10,random_state=0)  #fully grown decision tree


# In[ ]:


bag_model2=BaggingClassifier(n_estimators=10,random_state=0,base_estimator=tree2)## Regularised decision tree


# In[ ]:


models=[]
models.append(('Decision tree',tree))
models.append(('Random Forest',Rf_model))
models.append(('Bagged_DT',bag_model))
models.append(('bagged_regularized',bag_model2))


# In[ ]:


models


# In[ ]:


from sklearn import model_selection
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=5,random_state=2)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='recall')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, np.mean(cv_results), np.var(cv_results,ddof=1))
    print(msg)


# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# #                        THE END

# In[ ]:


# #lets import adaboost and bagging classifier
# from sklearn.ensemble import BaggingClassifier
# bagg=BaggingClassifier()
# bagmodel=bagg.fit(x_train,y_train)
# #make prediction
# pred_bagged=bagg.predict(x_test)


# In[ ]:


# accuracy_score(y_test,pred_bagged)


# In[ ]:


# tree=DecisionTreeClassifier()
# cross_val_score(tree,x,y,cv=10).mean()


# In[ ]:





# In[ ]:




