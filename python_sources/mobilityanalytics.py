#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# **To build a predictive model, which could help cab-aggregators in predicting the '*surge_pricing_type*' pro-actively. This would in turn help them in matching the right cabs with the right customers quickly and efficiently.**

# # Data Analysis

# ## Understanding data

# In[ ]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from scipy import stats
from sklearn import preprocessing,model_selection

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report


# In[ ]:


#reading data
train = pd.read_csv("../input/train.csv")
test =  pd.read_csv("../input/test.csv")
print(train.shape,test.shape)


# In[ ]:


#combining data for preprocessing
comb = train.append(test)
comb.shape


# In[ ]:


#getting information about datatypes and null values
comb.info()
print(comb.isnull().sum())


# In[ ]:


#Converting categorical variables to type category
cat_col = ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender','Surge_Pricing_Type']
comb[cat_col]= comb[cat_col].astype('category')


# In[ ]:


comb.info()


# In[ ]:


missingno.matrix(comb)


# In[ ]:


#Checking for any duplicate data
if len(comb[comb.duplicated()]) > 0:
    print("No. of duplicated entries: ", len(comb[comb.duplicated()]))
    print(comb[comb.duplicated(keep=False)].sort_values(by=list(comb.columns)).head())
else:
    print("No duplicated entries found")


# ## Working on numerical data

# In[ ]:


comb.describe()


# In[ ]:


comb.hist(figsize=(10,10))


# **We can see there is a big variance in the values of some of the features(difference in min and max values),lets apply log transformation on them**

# In[ ]:


col_log =['Trip_Distance','Var1','Var2','Var3']
comb[col_log]=np.log(comb[col_log])


# ***Imputing values using MICE(Multivariate Imputation by Chained Equation)***

# In[ ]:


pip install impyute


# In[ ]:


col = ['Customer_Since_Months','Life_Style_Index','Var1']


# In[ ]:


from impyute.imputation.cs import mice

X = comb[col]

imputed = mice(X.values,verbose=1)

#mice_ages = imputed[:, 2]


# In[ ]:


imputed.shape


# In[ ]:


mice_csm = imputed[:, 0]
mice_lsi = imputed[:, 1]
mice_var1 = imputed[:, 2]


# In[ ]:


sns.distplot(mice_csm, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})
sns.distplot(comb['Customer_Since_Months'], hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})


# In[ ]:


sns.distplot(mice_lsi, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})
sns.distplot(comb['Life_Style_Index'], hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})


# In[ ]:


sns.distplot(mice_var1, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})
sns.distplot(comb['Var1'], hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3})


# 
# 
# > As you can see in above plots, the variance in distribution of data after imputing is done is not very large.This is the reason this method is preferred over imputing by other test statistics.
# 
# 

# In[ ]:


comb['Customer_Since_Months'] = mice_csm
comb['Life_Style_Index'] = mice_lsi
comb['Var1'] = mice_var1


# In[ ]:


comb.isnull().sum()


# ## Working with categorical features

# **We will start by imputing values in features having Null values.For this we will create a separate category 'Unknown' as the number of unknowns is greater than number of observations in some other categories and filling it with some test statistics may cause biasing.**

# In[ ]:


comb['Type_of_Cab'] = comb['Type_of_Cab'].cat.add_categories('Unknown')
comb['Type_of_Cab'].fillna("Unknown", inplace=True)


# In[ ]:


comb['Confidence_Life_Style_Index'] = comb['Confidence_Life_Style_Index'].cat.add_categories('Unknown')
comb['Confidence_Life_Style_Index'].fillna("Unknown", inplace=True)


# In[ ]:


cat_col = ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender','Surge_Pricing_Type']
for i in cat_col:
  print("=================")
  print(comb[i].value_counts())


# **We can see there are a lot of categories in destination type.This will lead to many extra columns after one-hot encoding which may impact our model accuracy**
# 
# *We will try to minimize the categories by making bins ['High_vol','Med_Vol','Low_Vol'] based on the volumes*

# In[ ]:


comb['Destination_Type'].replace({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14},inplace = True)
comb['Destination_Type']


# In[ ]:


bins= [1,4,12,14]
labels = ['High_vol','Med_vol','Low_vol']
comb['Destination_Type'] = pd.cut(comb['Destination_Type'], bins=bins, labels=labels, right=False)


# In[ ]:


comb['Destination_Type'].value_counts()


# In[ ]:


comb.info()


# **Time for One-Hot Encoding**

# In[ ]:


dummies_toc = pd.get_dummies(comb['Type_of_Cab'],prefix='toc',prefix_sep='_')
dummies_clsi = pd.get_dummies(comb['Confidence_Life_Style_Index'],prefix = 'clsi',prefix_sep='_')


# In[ ]:


dummies_dt = pd.get_dummies(comb['Destination_Type'])
dummies_gender = pd.get_dummies(comb['Gender'])


# In[ ]:


merged = pd.concat([comb,dummies_toc,dummies_clsi,dummies_dt,dummies_gender],axis = 'columns')
comb = merged.drop(['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender'],axis = 'columns')

comb.shape


# **This is how our final data looks. Data looks pretty normalized so we will not be performing nirmalization separately**

# In[ ]:


comb.head()


# In[ ]:


train_features = comb[comb['Surge_Pricing_Type'].isnull()!=True].drop(['Trip_ID','Surge_Pricing_Type'], axis=1)
train_label = comb[comb['Surge_Pricing_Type'].isnull()!=True]['Surge_Pricing_Type']

test_features = comb[comb['Surge_Pricing_Type'].isnull()==True].drop(['Trip_ID','Surge_Pricing_Type'], axis=1)

train_features.shape,train_label.shape,test_features.shape


# # Model Training

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_features, train_label, test_size=0.25,random_state=42,stratify = train_label)


# In[ ]:


train = x_train.append(x_val)
val = y_train.append(y_val)


# 
# 
# > The approach for this problem will be to choose some models which will provide stable and good accuracies and finally apply hard voting on these to get the results
# 
# **Following models have been chosen for this purpose:**
# 
# *   XGBoost
# *   Light Gradient Boost
# *   Multi Layer Perceptron
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


#xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.6,learning_rate=0.05,objective='multi:softmax',n_estimators=300,max_depth=8,n_jobs=-1)
xgb.fit(x_train,y_train)


# In[ ]:


print(xgb.score(x_val,y_val))


# In[ ]:


# light gradient boost
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=425, learning_rate=0.08,objective= 'multiclass', reg_lambda=3,max_depth=8,min_child_weight=0.1)
lgbc.fit(x_train,y_train)


# In[ ]:


lgbc.score(x_val,y_val)


# In[ ]:


#best parameter selection for Perceptron
from sklearn.model_selection import GridSearchCV


mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.01,0.1],
    'learning_rate': ['constant','adaptive'],
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train, y_train)

print('Best parameters found:\n', clf.best_params_)


# In[ ]:


from sklearn.neural_network import MLPClassifier
p = MLPClassifier(random_state=42,
              max_iter=200,activation='relu',solver ='adam',learning_rate='adaptive')
p.fit(x_train,y_train)


# In[ ]:


p.score(x_val,y_val)


# *Fitting our models to entire dataset*

# In[ ]:


xgb.fit(train,val)
lgbc.fit(train,val)
p.fit(train,val)


# **Hard-Voting Ensemble**

# ***In hard voting (also known as majority voting), every individual classifier votes for a class, and the majority wins. In statistical terms, the predicted target label of the ensemble is the mode of the distribution of individually predicted labels.***

# In[ ]:


from sklearn.ensemble import VotingClassifier
estimator = []
estimator.append(('MLP', p))
estimator.append(('XGB', xgb )) 
estimator.append(('LGBM', lgbc)) 
 
  
# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(train, val) 


# In[ ]:


print(vot_hard.score(x_val,y_val))


# 
# 
# > The model performs quite well on the validation dataset. Let us now use it for predictions on test dataset.
# 
# 
# 
# 
# 
# 
# 

# # Making Predictions

# In[ ]:


pred_vot_hard = vot_hard.predict(test_features)
unique_elements, counts_elements = np.unique(pred_vot_hard, return_counts=True)
print(unique_elements,counts_elements)


# In[ ]:


pd.DataFrame(pred_vot_hard, columns=['Surge_Pricing_Type']).to_csv('/prediction_vot_final.csv')


# **The final score recieved on the scoreboard was 0.7012.It is a general agreement that ensembles are more stable and accurate than stand-alone models**

# # Conclusion

# 
# 
# *In this notebook we successfully predicted the Surge_Pricing_Type for cab aggregators.*
# 
# **Some important concepts learned:**
# 
# *   MICE helps in imputation of missing data for numerical features with minimal effect on variance of the distribution.
# *   If number of NaN's are very large(larger than observations in one of the category) it is better to make a separate category out of them.
# 
# 
# *   To remove skewness/outliers in features we can apply log transformation or normalization.
# *   Building an ensemble is an effective way to get a stable model.
# 
# 
# 
# 
# 
# 
