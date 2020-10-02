#!/usr/bin/env python
# coding: utf-8

# Thanks to Parth Sharma and Himanshu Poddar as well as other contributors. Cheers!!!

# **Import modules**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sys


# In[ ]:


data=pd.read_csv('../input/zomato.csv')


# **Read the data and see the columns which are there**

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.index


# In[ ]:


data.columns


# In[ ]:


data.info()


# **Delete uneccessary columns**

# In[ ]:


del data['url']
del data['phone']
del data['address']
del data['location']


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# Replace Bogus terms with NaN values

# In[ ]:


data['rate'] = data['rate'].replace('NEW',np.NaN)
data['rate'] = data['rate'].replace('-',np.NaN)


# In[ ]:


data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                         'listed_in(city)':'city'})


# **Convert str to float**

# In[ ]:


X=data.copy()


# In[ ]:


X.online_order=X.online_order.apply(lambda x: '1' if str(x)=='Yes' else '0')
X.book_table=X.book_table.apply(lambda x: '1' if str(x)=='Yes' else '0')


# In[ ]:


X.rate.dtype


# In[ ]:


X.rate=X.rate.astype(str)
X.rate=X.rate.apply(lambda x : x.replace('/5',''))
X.rate=X.rate.astype(float)


# In[ ]:


X.rate.dtype


# In[ ]:


X.cost.dtype


# In[ ]:


X.cost=X.cost.astype(str)
X.cost=X.cost.apply(lambda y : y.replace(',',''))
X.cost=X.cost.astype(float)


# In[ ]:


X.cost.dtype


# In[ ]:


X.online_order=X.online_order.astype(float)
X.book_table=X.book_table.astype(float)
X.votes=X.votes.astype(float)


# In[ ]:


X.info()


# Now all value related columns are float type.

# In[ ]:


X.isnull().sum()


# **Replace missing values by deleting missing values**

# In[ ]:


X_del=X.copy()
X_del.dropna(how='any',inplace=True)


# In[ ]:


X_del.isnull().sum()


# In[ ]:


X_del.info()


# **Remove duplicates**

# In[ ]:


X_del.drop_duplicates(keep='first',inplace=True)


# In[ ]:


X_del.head()


# **Data Visualization**

# In[ ]:


sns.countplot(X_del['online_order'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')


# In[ ]:


sns.countplot(X_del['book_table'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants allowing table booking or not')


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
x = pd.crosstab(X_del['rate'], X_del['online_order'])
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])
plt.title('online order vs rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# Gaussian like behaviour with outliers

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
y = pd.crosstab(X_del['rate'], X_del['book_table'])
y.div(y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])
plt.title('table booking vs rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# In[ ]:


sns.countplot(X_del['city'])
sns.countplot(X_del['city']).set_xticklabels(sns.countplot(X_del['city']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Location')


# In[ ]:


loc_plt=pd.crosstab(X_del['rate'],X_del['city'])
loc_plt.plot(kind='bar',stacked=True);
plt.title('Location - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Location',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();


# The above diagram shows that 3.8 to 4.2 rated restuarant are most preferred in various location in Bangalore

# In[ ]:


sns.countplot(X_del['rest_type'])
sns.countplot(X_del['rest_type']).set_xticklabels(sns.countplot(X_del['rest_type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Restuarant Type')


# People like Casual Dining and Quick Bite restuarants more, hence they are large in number

# In[ ]:


loc_plt=pd.crosstab(X_del['rate'],X_del['rest_type'])
loc_plt.plot(kind='bar',stacked=True);
plt.title('Rest type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Rest type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();


# Gaussian like behaviour with peak around 3.9 and 4.0

# In[ ]:


sns.countplot(X_del['type'])
sns.countplot(X_del['type']).set_xticklabels(sns.countplot(X_del['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Type of Service')


# Zomato Users prefer Delivery and Dine-out among the categories. 

# In[ ]:


type_plt=pd.crosstab(X_del['rate'],X_del['type'])
type_plt.plot(kind='bar',stacked=True);
plt.title('Type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');


# In[ ]:


sns.countplot(X_del['cost'])
sns.countplot(X_del['cost']).set_xticklabels(sns.countplot(X_del['cost']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Cost of Restuarant')


# Restuarants with around 300 to 800 rupees avergae bill(two people) are high in number

# In[ ]:



cost_for_two = pd.cut(X_del['cost'],bins = [0, 200, 500, 1000, 5000, 8000],labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000',])
cost_plt=pd.crosstab(X_del['rate'],cost_for_two)
cost_plt.plot(kind='bar',stacked=True);
plt.title('Avg cost - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Average Cost',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');


# In[ ]:


Y=X_del.copy()


# **One hot encoding**

# Reviews list could be encoded however it is a NLP task and that is saved for future work

# In[ ]:


dummy_rest_type=pd.get_dummies(Y['rest_type'])
dummy_type=pd.get_dummies(Y['type'])
dummy_city=pd.get_dummies(Y['city'])
dummy_cuisines=pd.get_dummies(Y['cuisines'])
dummy_dishliked=pd.get_dummies(Y['dish_liked'])
#dummy_reviewslist=pd.get_dummies(Y['reviews_list']) #Too much memory allocation


# Combine all the OHE categories

# In[ ]:


Y=pd.concat([Y,dummy_rest_type,dummy_type,dummy_city,dummy_cuisines,dummy_dishliked,#dummy_reviewslist
            ],axis=1)


# In[ ]:


del Y['rest_type']
del Y['type']
del Y['city']
del Y['cuisines']
del Y['dish_liked']
#del Y['reviews_list']


# Delete the original categories which are already encoded

# In[ ]:


Y.head()


# **Standardize data**

# We standardize the data because the target column(ratings) is vs other independent columns showcase Normal Distribution

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


x=Y.drop(['name',#'dish_liked',
          'reviews_list',
          'menu_item',#'cuisines'
         ],axis=1);


# In[ ]:


x_fit=scaler.fit_transform(x)


# In[ ]:


x=pd.DataFrame(x_fit,columns=x.columns)


# In[ ]:


x.info()


# In[ ]:


x.head()


# **Correlation between variables**

# In[ ]:


#corr_x=x.corr().abs()


# In[ ]:


#corr_x


# In[ ]:


from sklearn.feature_selection import SelectKBest


# In[ ]:


#col_r=list(x)
#col_r


# In[ ]:


#col_r.insert(0, col_r.pop(col_r.index('rate')))


# In[ ]:


#col_r


# In[ ]:


#x_1 = x.loc[:, col_r]


# In[ ]:


#x_1.head()


# In[ ]:


#x_1.info()


# In[ ]:


#for i,a in enumerate(x_1.columns.values[0:1809]):
    #print('%s is %d' % (a,i))


# In[ ]:


#column_names=x_1.columns.values
#column_names[34]='Delivery_A'
#column_names[35]='Delivery_B'
#column_names[81]='Delivery_remove_1'
#column_names[82]='Delivery_remove_2'


# In[ ]:


#x_1.columns=column_names


# In[ ]:


#x_1.head()


# In[ ]:


#del x_1['Delivery_remove_1']
#del x_1['Delivery_remove_2']


# In[ ]:


#x_1.info()


# Independent Columns

# In[ ]:


X_init=x.drop(['rate'],axis=1)
split_x=X_init.iloc[:,:]
split_x.info()
split_x.shape
split_x


# Target Columns/Variable

# In[ ]:


Y_init=x.drop(x.columns.difference(['rate']),axis=1)
split_y=Y_init.iloc[:,:]
split_y=split_y.astype(float)
split_y.shape
split_y


# In[ ]:


bestfeatures=SelectKBest(k='all')
fit=bestfeatures.fit(split_x,split_y)


# In[ ]:


fit


# In[ ]:


scores=pd.DataFrame(fit.scores_)
columns_=pd.DataFrame(split_x.columns)


# In[ ]:


featurescore=pd.concat([columns_,scores],axis=1)


# In[ ]:


featurescore.columns = ['Features','Score']


# In[ ]:


print(featurescore.nlargest(20,'Score'))


# The number is chosen as to decrease overfitting while keeping the accuracy intact, it's a tradeoff

# In[ ]:


col_select=featurescore.nlargest(800,'Score')


# In[ ]:


col_select.drop('Score',axis=1,inplace=True)


# In[ ]:


col_select_list=list(col_select.Features)


# In[ ]:


col_select_list


# **Feature Selection**

# In[ ]:


x_select=split_x.loc[:,col_select_list]


# In[ ]:


x_select.head()


# In[ ]:


x_select.info()


# Remove Duplicate columns

# In[ ]:


x_select = x_select.loc[:, ~x_select.columns.duplicated()]


# **Train test split**

# 5% split ratio

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_select,split_y,test_size=0.05,random_state=42)


# **Linear Regressor**

# In[ ]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[ ]:


linreg.fit(X_train,y_train)


# In[ ]:


Y_linreg_pred=linreg.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,Y_linreg_pred)


# In[ ]:


acc_len=linreg.score(X_train,y_train)


# In[ ]:


acc_len


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100,random_state=42)


# In[ ]:


rf_reg.fit(X_train,y_train)


# In[ ]:


Y_rgreg_pred=rf_reg.predict(X_test)


# In[ ]:


r2_score(y_test,Y_rgreg_pred)


# In[ ]:


acc_rfreg=rf_reg.score(X_train,y_train)


# In[ ]:


acc_rfreg


# **Ridge Regressor**

# In[ ]:


from sklearn.linear_model import RidgeCV
ridge=RidgeCV(alphas=[1e-10,1e-8,1e-6,1e-2,1e-1,1,10,20,100])


# In[ ]:


fit_ridge=ridge.fit(X_train,y_train)


# In[ ]:


fit_ridge.alpha_


# In[ ]:


Y_ridge_pred=ridge.predict(X_test)


# In[ ]:


r2_score(y_test,Y_ridge_pred)


# In[ ]:


acc_ridge=ridge.score(X_train,y_train)


# In[ ]:


acc_ridge


# **Lasso Regressor**

# In[ ]:


from sklearn.linear_model import LassoCV
lasso=LassoCV(alphas=[1e-3,1e-2,1e-1,1,10,20,100],max_iter=1e2)


# In[ ]:


fit_lasso=lasso.fit(X_train,y_train)


# In[ ]:


fit_lasso.alpha_


# In[ ]:


Y_lasso_pred=lasso.predict(X_test)


# In[ ]:


acc_lasso=lasso.score(X_train,y_train)


# In[ ]:


r2_score(y_test,Y_lasso_pred)


# In[ ]:


acc_lasso


# **MLP regressor**

# In[ ]:


from sklearn.neural_network import MLPRegressor
mlp=MLPRegressor(random_state=42)


# In[ ]:


mlp.fit(X_train,y_train)


# In[ ]:


acc_mlp=mlp.score(X_train,y_train)


# In[ ]:


Y_mlp_pred=mlp.predict(X_test)


# In[ ]:


r2_score(y_test,Y_mlp_pred)


# In[ ]:


acc_mlp


# **Extra trees regressor**

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


rf_extrareg=ExtraTreesRegressor(n_estimators=100,random_state=42)


# In[ ]:


rf_extrareg.fit(X_train,y_train)


# In[ ]:


Y_extra_rgreg_pred=rf_extrareg.predict(X_test)


# In[ ]:


r2_score(y_test,Y_extra_rgreg_pred)


# In[ ]:


acc_extra_reg_score=rf_extrareg.score(X_train,y_train)


# In[ ]:


acc_extra_reg_score


# **KNN regressor**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_jobs=-1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


Y_knn_pred=knn.predict(X_test)


# In[ ]:


r2_score(y_test,Y_knn_pred)


# In[ ]:


acc_knn_score=knn.score(X_train,y_train)


# In[ ]:


acc_knn_score


# **SVM Regressor**

# In[ ]:


from sklearn.svm import LinearSVR
svr=LinearSVR(random_state=42)


# In[ ]:


svr.fit(X_train,y_train)


# In[ ]:


Y_svr_pred=svr.predict(X_test)


# In[ ]:


r2_score(y_test,Y_svr_pred)


# In[ ]:


acc_svr=svr.score(X_train,y_train)


# In[ ]:


acc_svr


# **Gradient Boosting Regressor**

# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
gbr=HistGradientBoostingRegressor(random_state=42)


# In[ ]:


gbr.fit(X_train,y_train)


# In[ ]:


Y_gbr_pred=gbr.predict(X_test)


# In[ ]:


r2_score(y_test,Y_gbr_pred)


# In[ ]:


acc_gbr=gbr.score(X_train,y_train)


# In[ ]:


acc_gbr


# **Elastic Net regressor**

# In[ ]:


from sklearn.linear_model import ElasticNet
en=ElasticNet(random_state=42,alpha=0.0001,precompute=True)


# In[ ]:


en.fit(X_train,y_train)


# In[ ]:


Y_en_pred=en.predict(X_test)


# In[ ]:


r2_score(y_test,Y_en_pred)


# In[ ]:


acc_en=en.score(X_train,y_train)


# In[ ]:


acc_en


# **Bayesian Regressor**

# In[ ]:


from sklearn.linear_model import BayesianRidge
bay=BayesianRidge()


# In[ ]:


bay.fit(X_train,y_train)


# In[ ]:


Y_bay_pred=bay.predict(X_test)


# In[ ]:


r2_score(y_test,Y_bay_pred)


# In[ ]:


acc_bay=bay.score(X_train,y_train)


# In[ ]:


acc_bay


# **Stochastic Gradient Descent Regressor**

# In[ ]:


from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor(loss='squared_epsilon_insensitive',random_state=42,learning_rate='adaptive',max_iter=3000)


# In[ ]:


sgd.fit(X_train,y_train)


# In[ ]:


Y_sgd_pred=sgd.predict(X_test)


# In[ ]:


r2_score(y_test,Y_sgd_pred)


# In[ ]:


acc_sgd=sgd.score(X_train,y_train)


# In[ ]:


acc_sgd


# Ensemble Tree Regreesion model performs the best with R2 score of 93.5% on test set and 98.9 on train set while Gradient boost performs the worst among other models

# **Classification task**

# We will be using mulit-label classification, basically, we will be wanting to know whether a restuarant can accept online order and booking table or not based on the given data. We don't need to standardize the data in classification task.

# In[ ]:


data_class=Y.copy()
data_class.head()


# In[ ]:


del data_class['name']
del data_class['menu_item']
del data_class['reviews_list']


# In[ ]:


x_class=data_class.drop(['online_order','book_table'],axis=1)
y_class=data_class.drop(data_class.columns.difference(['online_order','book_table']),axis=1)


# In[ ]:


x_class


# **Feature Selection**

# In[ ]:


bestfeatures_class=SelectKBest(k='all')
fit_class_oo=bestfeatures_class.fit(x_class,y_class.online_order)
fit_class_bt=bestfeatures_class.fit(x_class,y_class.book_table)


# In[ ]:


fit_class_oo


# In[ ]:


fit_class_oo.scores_


# In[ ]:


fit_class_bt.scores_


# In[ ]:


class_score=pd.DataFrame(fit_class_oo.scores_)
class_columns_=pd.DataFrame(x_class.columns)


# In[ ]:


featureclass_score=pd.concat([class_columns_,class_score],axis=1)


# In[ ]:


featureclass_score.columns=['Features','Score']


# In[ ]:


print(featureclass_score.nlargest(1000,'Score'))


# In[ ]:


feature_select=featureclass_score.nlargest(90,'Score')


# Value is adjusted accordingly, feel free to experiment around

# In[ ]:


feature_select_list=list(feature_select.Features)


# In[ ]:


x_class_select=x_class.loc[:,feature_select_list]


# In[ ]:


x_class_select.info()


# In[ ]:


x_class_select = x_class_select.loc[:, ~x_class_select.columns.duplicated()]


# In[ ]:


X_class_train,X_class_test,y_class_train,y_class_test=train_test_split(x_class_select,y_class,test_size=0.05,random_state=42)


# **Random Forest Classifier (Multi Label)**

# Multi label is already included in the module

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_class=RandomForestClassifier(n_estimators=100,random_state=42)


# In[ ]:


forest_class.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_forest=forest_class.predict(X_class_test)


# In[ ]:


y_predict_class_forest


# In[ ]:


y_train_score_forest=forest_class.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_forest


# In[ ]:


y_test_score_forest=forest_class.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_forest


# **KNN classifier (Multi label)**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_class=KNeighborsClassifier(n_jobs=-1)


# In[ ]:


knn_class.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_knn=knn_class.predict(X_class_test)


# In[ ]:


y_predict_class_knn


# In[ ]:


y_train_score_knn=knn_class.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_knn


# In[ ]:


y_test_score_knn=knn_class.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_knn


# **MLP Classifier (Multi Label)**

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp_class=MLPClassifier()


# In[ ]:


mlp_class.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_mlp=mlp_class.predict(X_class_test)


# In[ ]:


y_predict_class_mlp


# In[ ]:


y_train_score_mlp=mlp_class.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_mlp


# In[ ]:


y_test_score_mlp=mlp_class.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_mlp


# **Simple Logistic Regression implementing MultiLabel classification methods**

# Although logisitc regression is a multi-class classification model we can apply transformation methods applied to transform multi-label problem into small multi-class problems

# Binary Relevance

# In[ ]:


get_ipython().system('pip install scikit-multilearn')


# In[ ]:


from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression


# In[ ]:


logit_class_bin=BinaryRelevance(LogisticRegression())


# In[ ]:


logit_class_bin.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_logit_bin=logit_class_bin.predict(X_class_test)


# In[ ]:


y_predict_class_logit_bin


# In[ ]:


y_train_score_logit_bin=logit_class_bin.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_logit_bin


# In[ ]:


y_test_score_logit_bin=logit_class_bin.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_logit_bin


# Classifier Chains

# In[ ]:


from skmultilearn.problem_transform import ClassifierChain
logit_class_chain=ClassifierChain(LogisticRegression())


# In[ ]:


logit_class_chain.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_logit_chain=logit_class_chain.predict(X_class_test)


# In[ ]:


y_predict_class_logit_chain


# In[ ]:


y_train_score_logit_chain=logit_class_chain.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_logit_chain


# In[ ]:


y_test_score_logit_chain=logit_class_chain.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_logit_chain


# Label Power Set

# In[ ]:


from skmultilearn.problem_transform import LabelPowerset
logit_class_power=LabelPowerset(LogisticRegression())


# In[ ]:


logit_class_power.fit(X_class_train,y_class_train)


# In[ ]:


y_predict_class_logit_power=logit_class_power.predict(X_class_test)


# In[ ]:


y_predict_class_logit_power


# In[ ]:


y_train_score_logit_power=logit_class_power.score(X_class_train,y_class_train)


# In[ ]:


y_train_score_logit_power


# In[ ]:


y_test_score_logit_power=logit_class_power.score(X_class_test,y_class_test)


# In[ ]:


y_test_score_logit_power


# Random forest performs the best with 99.2% mean accuracy on train test and 94.5% mean accuracy on test set, while in logistic regression performs the worst, with label powerset performing the worst due to higher number of labels.

# **Future Work**: 
# 1. Applying Cross validation instead of train_test split.
# 2. Using Deep Learning methods and unsupervised learning methods for various tasks
# 3. Tuning the hyperparamters of various models even further.
