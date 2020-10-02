#!/usr/bin/env python
# coding: utf-8

# ## Loading packages and data

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import warnings
warnings.simplefilter('ignore')

from IPython.display import display, HTML

from imputer import Imputer
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


data_train = pd.read_csv('../input/application_train.csv')


# In[ ]:


#print( list(data_train.columns) ) 


# ## EDA
# First of all, I'm going to deal with the main dataset "application_train.csv".  As most of descriptive staistics and visualization (e. g. distributions of single variables, number of nans, etc.) are presented on [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data#), I'll use it without reduplicating the same information in this notebook.

# In[ ]:


data_train.info()


# Firstly, I'd like to see distributions of continious variables in dataset according to their class and find variables that have significant difference between classes and can have a bigger impact inside the model.

# In[ ]:


index = 0
for i in data_train.drop(columns='SK_ID_CURR').columns:
    if np.dtype(data_train[i]) == 'float64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6:
        #if sum(data_train[i].dropna() == data_train[i].dropna().astype('int64')):
            index +=1
            plt.subplot(4, 5, index)
            #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[abs(stats.zscore(data_train[i].dropna())) < 3])
            curr_feature_1 = data_train[i][data_train.TARGET == 1].dropna()
            curr_feature_0 = data_train[i][data_train.TARGET == 0].dropna()
            sns.distplot(curr_feature_1[abs(stats.zscore(curr_feature_1)) < 3]) #to reduce most of outliers
            sns.distplot(curr_feature_0[abs(stats.zscore(curr_feature_0)) < 3])

plt.subplots_adjust(top=3, bottom=0, left=0, right=3, hspace=0.25, wspace=0.55)
plt.show()


# In[ ]:


index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'int64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6 and len(data_train[i].unique()) > 50:
        index +=1
        plt.subplot(1, 3, index)
        curr_feature_1 = data_train[i][data_train.TARGET == 1].dropna()
        curr_feature_0 = data_train[i][data_train.TARGET == 0].dropna()
        #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[data_train.TARGET == 0])
        #sns.boxplot(x = data_train.TARGET, y = data_train[i].dropna()[data_train.TARGET == 1])
        sns.distplot(curr_feature_1)
        sns.distplot(curr_feature_0)
plt.subplots_adjust(top=1, bottom=0, left=0, right=3, hspace=0.25, wspace=0.55)
plt.show()


# Well, according to graphs (taking into account that they are in the interval of 3 sd), some of them are visualised in inappropriate way, except those that are actually continious. So, some of the variables show different distribution between classes like EXT_SOURCE_2(3), DAYS_BIRTH, and probably AMT_INCOME_TOTAL which is quite right-skewed and has outliers, so I need to look a bit closer or apply some transformation. 

# In[ ]:


plt.subplot(211)
sns.distplot(np.log(data_train.AMT_INCOME_TOTAL[data_train.TARGET == 0] + 1))
plt.subplot(212)
sns.distplot(np.log(data_train.AMT_INCOME_TOTAL[data_train.TARGET == 1] + 1))
plt.show()


# In[ ]:


sum(np.log(data_train.AMT_INCOME_TOTAL + 1) > 14) #outliers


# Now, let's take a look at the varibles of type object,  I'll create crosstables that will show the percentege of TARGET = 1 and TARGET = 0 inside each class of presented variables in order to indicate class or set of them that can be associated with TARGET = 1. However, on the stage of inferential statistics it should be taken into account that there are some very unbalanced classes presented in the tables below.

# In[ ]:


index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'O' and len(data_train[i].dropna())/data_train.shape[0] > 0.6:
        tab = pd.crosstab(data_train.TARGET, data_train[i], margins=True)
        display(HTML((tab/tab.loc[tab.index[-1]]).to_html()))


# So, there are some varibles and classes that have a higher percenatge of TARGET = 1. Well, probably it would be better to redefine multinominal features into two classes, for example, "Higher risk" and "Lower risk", what will also allow us not to enlarge the dimensionality with dummy encoding, and then make sure that difference in percentages are statistically significant (e.g. apply chi-square criteria).

# The same is with binary features that are presnted below.

# In[ ]:


index = 0
for i in data_train.drop(columns=['SK_ID_CURR', 'TARGET']).columns:
    if np.dtype(data_train[i]) == 'int64' and len(data_train[i].dropna())/data_train.shape[0] > 0.6 and len(data_train[i].unique()) < 100:
        tab = pd.crosstab(data_train.TARGET, data_train[i], margins=True)
        display(HTML((tab/tab.loc[tab.index[-1]]).to_html()))


# ## Hypotheses testing

# In this block, I'll perform chi-square test for those variables that seem to have a significant association with TARGET.

# In[ ]:


def chi_test(data, feature, target = 'TARGET', group_classes=False):
    
    if sum(pd.isna(data[feature])):
        data[feature].replace(np.nan, 'Unknown', inplace=True)

    cnt_table = pd.crosstab(data[target], data[feature])#to check if there are enough observations in each class
    
    if group_classes:
        tab = pd.crosstab(data[target], data[feature], margins=True)
        tab = tab/tab.loc[tab.index[-1]]
        labels = {}
        for i in cnt_table.columns:
            if tab[i][1] > tab['All'][1]:
                labels[i] = 'High risk'
            else:
                labels[i] = 'Low risk'
        cnt_bi_table = pd.crosstab(data[target], data[feature].replace(labels))
        chi = stats.chi2_contingency(cnt_bi_table)
        display(HTML(pd.crosstab(data[target], data[feature].replace(labels), margins=True).to_html()))
        print( { 'Chi-square statisitc': chi[0],
           'p-value': chi[1], 
          'df': chi[2]} )
        return labels
    else:
        chi = stats.chi2_contingency(cnt_table)                           
        display(HTML(pd.crosstab(data[target], data[feature], margins=True).to_html()))
        print( { 'Chi-square statisitc': chi[0],
           'p-value': chi[1], 
          'df': chi[2]} )


# In[ ]:


chi_test(data_train[data_train.CODE_GENDER != 'XNA'], 'CODE_GENDER')


# In[ ]:


inc_labels = chi_test(data_train, 'NAME_INCOME_TYPE', group_classes=True)


# So, here we see that while the groups High risk and Low risk have almost the same number of observatoins (158801 and 148710) the class TARGET=1 is almost 60% frequent than TARGET=0, therefore probably such division can be informative in model. The same can be done with the next feature NAME_EDUCATION_TYPE, but as far as this feature is ordinal, I think it'd better to put them in right order while encoding. 

# In[ ]:


hsng_labels = chi_test(data_train, 'NAME_HOUSING_TYPE', group_classes=True)


# Well, although p-value is significant, the difference in TARGET = 1 between groups is about 3%, therefore I don't think that this feature should be present in model with such division.

# In[ ]:


occup_labels = chi_test(data_train, 'OCCUPATION_TYPE', group_classes=True)


# Here we see the difference in about 4%, not much, but still it can be included in the model in such division, because it has too many classes.

# In[ ]:


orgn_labels = chi_test(data_train, 'ORGANIZATION_TYPE', group_classes=True)


# In this feature there is also a significant difference in numbers between High risk TARGET=1 and Low risk TARGET = 1, moreover such grouping in this feature again reduces quite a big number of classes (58).

# ## Data preprocessing

# ### Dealing with NA

# The are quite a lot of features that have too many missing values. Let's see what can be done about it. The first feature OWN_CAR_AGE has for about 66% values missing.  As there is also FLAG_OWN_CAR feature that represents if client have a car, we will see if nans can be explained by absence of the car.

# In[ ]:


data_train.shape


# In[ ]:


print( sum(data_train.FLAG_OWN_CAR == "Y") )
print( data_train.OWN_CAR_AGE.dropna().shape )


# Well, obviously nans are produced by absence of the car, then we can just impute 0 instead of nans. 

# In[ ]:


data_train.OWN_CAR_AGE.fillna(value = 0, inplace=True)


# The next feature is OCCUPATION_TYPE and we don't know the origin of nans. I suppose that it can be two things: client refused telling it or he doesn't have a job, but still both of these tells us some information about client, therefore I will replace nans by 'Unknown'.  After cleaning data from missing values I'll encode this feature.

# In[ ]:


print( data_train.OCCUPATION_TYPE.unique() )
data_train.OCCUPATION_TYPE.fillna(value = 'Unknown', inplace=True)


# Next subset of features is about means of communications provided by client. There are very imbalanced classes inside each feature, so it should be taken into account during train-test split. Moreover some features are presented by only one class that significantly prevail and some of them can be dropped.

# In[ ]:


print( sum(data_train.FLAG_MOBIL == 0) )
print( sum(data_train.FLAG_MOBIL == 1) )


# In[ ]:


print( sum(data_train.FLAG_CONT_MOBILE == 0) )
print( sum(data_train.FLAG_CONT_MOBILE == 1) )


# In[ ]:


data_train.drop(columns=['FLAG_MOBIL', 'FLAG_CONT_MOBILE'], inplace=True)


# Another subset of features is about provided documents and almost all of them have an enormous dominance by one class, so I think it would make sense to combine them to feature which will represent the number of provided documents.

# In[ ]:


data_train.shape


# In[ ]:


data_train['DOC_COUNT'] = data_train.FLAG_DOCUMENT_2

for i in range(3, 22):
    data_train['DOC_COUNT'] = data_train['DOC_COUNT'] + data_train['FLAG_DOCUMENT_'+str(i)]


# The next subset (second plot below) of features has too many nans (>50%) and it would be better to see how these nans distributed in our data. 

# In[ ]:


msno.matrix(data_train.iloc[:,0:42])
plt.show()


# In[ ]:


msno.matrix(data_train.iloc[:,42:89])
plt.show()


# In[ ]:


msno.matrix(data_train.iloc[:,89:])
plt.show()


# Well, we see that almost all features from the second plot have too many missings. All these features are connected with the information about apartments. As it would be inaccurate to use features with such number of missings, probably it would be better to create one feature from them, that would represent the percentage of known information about client's apartments.

# In[ ]:


for i in data_train.iloc[:, 42:56].columns:
    data_train[i] = -pd.isna(data_train[i])

for i in data_train.iloc[:, 84:89].columns:
    data_train[i] = -pd.isna(data_train[i])


# In[ ]:


data_train['HOUSE_INFO'] = (data_train.iloc[:, 84:89].sum(axis=1) + data_train.iloc[:, 42:56].sum(axis=1))/19


# The last five columns shows the number of enquiries to Credit Bureau and number of enquiries of each next column excludes those that were already marked in previous one. Nans make up 14% in each column. I'll fill them later.

# In[ ]:


#for i in data_train.iloc[:, 114:120].columns:
#   data_train[i].fillna(data_train[i].median(), inplace = True)


# Now dropping all columns that have (>=50%) of values missing and those that was combined into one (DOC_COUNT).

# In[ ]:


data_train.drop(columns=data_train.iloc[:, 94:114].columns, inplace=True)
data_train.drop(columns=data_train.iloc[:, 42:89].columns, inplace=True)
data_train.drop(columns='EXT_SOURCE_1', inplace=True)


# The last column with a significant number of gaps is EXT_SOURCE_3, that has 16% of values missing, therefore I must find an appropriate way to fill them.

# In[ ]:


sns.distplot(data_train.EXT_SOURCE_3.dropna())
sns.distplot(data_train.EXT_SOURCE_3.fillna(data_train.EXT_SOURCE_3.dropna().median()))
plt.show()


# Well, as for me, imputations that made using mean/median are not suitable in that case because they violate the distribution. Probably, it would be better to use another strategy, for example, KNN to fill these gaps taking into account what values have nearest objects in a multidimensional space, but firstly, it's better to encode categorical features in order to perform imputation function faster. 

# In[ ]:


data_train.info()


# In[ ]:


to_be_scaled = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']
for i in data_train.columns:
    if data_train[i].dtype == 'float64':
        to_be_scaled.append(i)


# In[ ]:


#data_train.head()


# ### Encoding

# In[ ]:


for i in data_train.columns:
    if data_train[i].dtype == 'O':
        print( [i, data_train[i].unique()] )


# In[ ]:


sum(data_train.CODE_GENDER == 'XNA')


# In[ ]:


data_train.CODE_GENDER.replace('XNA', 'F', inplace=True)
data_train.NAME_INCOME_TYPE.replace(inc_labels, inplace=True)
data_train.OCCUPATION_TYPE.replace(occup_labels, inplace=True)
data_train.ORGANIZATION_TYPE.replace(orgn_labels, inplace=True)
data_train.WEEKDAY_APPR_PROCESS_START.replace({'WEDNESDAY': 'Week', 'MONDAY': 'Week', 'THURSDAY': 'Week', 'SUNDAY': 'Weekend',
                                               'SATURDAY': 'Weekend', 'FRIDAY': 'Week', 'TUESDAY': 'Week'}, inplace=True)


# In[ ]:


data_train.head()


# In[ ]:


binarizer = LabelBinarizer()


# In[ ]:


for i in data_train.columns:
    if data_train[i].dtype == 'O' and len(data_train[i].unique()) == 2:
        data_train[i] = binarizer.fit_transform(data_train[i])


# In[ ]:


#data_train.head()


# In[ ]:


encoder = LabelEncoder()


# In[ ]:


sum(data_train.NAME_TYPE_SUITE.isnull())


# In[ ]:


data_train.NAME_TYPE_SUITE = encoder.fit_transform(data_train.NAME_TYPE_SUITE.replace(np.nan, 'Unknown'))


# In[ ]:


print( encoder.classes_ )
sum(data_train.NAME_TYPE_SUITE == 7)


# In[ ]:


data_train.NAME_TYPE_SUITE.replace('Unknown', np.nan, inplace=True)


# In[ ]:


data_train.NAME_EDUCATION_TYPE = encoder.fit_transform(data_train.NAME_EDUCATION_TYPE)


# In[ ]:


encoder.classes_ #ordered


# In[ ]:


data_train = pd.get_dummies(data_train)


# In[ ]:


data_train.shape


# ### Imputation

# In[ ]:


impute = Imputer()


# In[ ]:


get_ipython().run_cell_magic('time', '', "data_train_cl = pd.DataFrame(impute.knn(X=data_train, column='EXT_SOURCE_3', k = 3), columns=data_train.columns)")


# In[ ]:


sns.distplot(data_train.EXT_SOURCE_3.dropna())
sns.distplot(data_train_cl.EXT_SOURCE_3)
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in data_train.columns:\n    if sum(data_train[i].isnull()):\n        data_train_cl = pd.DataFrame(impute.knn(X=data_train_cl, column=i, k = 3), columns=data_train.columns)')


# In[ ]:


data_train_cl.head()


# ### Standardization

# In[ ]:


std = RobustScaler()
for i in to_be_scaled:
    data_train_cl[i] = std.fit_transform(pd.DataFrame(data_train_cl[i], columns=[i]))


# ## Model Building

# ### Logistic regression

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_train_cl.drop(columns='TARGET'), 
                                                    data_train_cl.TARGET, test_size=0.2, random_state=23, stratify=data_train_cl.TARGET)


# In[ ]:


get_ipython().run_cell_magic('time', '', "results = {}\nfor i in [0.001, 0.01, 0.1, 1, 2]:\n    lgreg = LogisticRegression(C=i, class_weight='balanced', penalty='l2', max_iter=1000)\n    lgreg.fit(X=X_train, y=y_train)\n    pred = lgreg.predict(X_test)\n    results[i] = roc_auc_score(y_test, pred)")


# In[ ]:


results


# ### Random Forest

# In[ ]:


forest = RandomForestClassifier(max_depth=10, n_estimators=200, class_weight='balanced', n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'forest.fit(X_train, y_train)')


# In[ ]:


pred2 = forest.predict(X_test)


# In[ ]:


roc_auc_score(y_test, pred2)


# ### Gradient boosting

# In[ ]:


# for validation lgb
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_train, 
                                                    y_train, test_size=0.2, random_state=23)


# In[ ]:


train_data = lgb.Dataset(X_train_v, label=y_train_v)
test_data = lgb.Dataset(X_test_v, label=y_test_v)


# In[ ]:


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 50,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 30,
    'learning_rate': 0.05,
    'verbose': 0
}


# In[ ]:


model = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# Testing on left-off set

# In[ ]:


pred4= model.predict(X_test)


# In[ ]:


roc_auc_score(y_test, pred4)

