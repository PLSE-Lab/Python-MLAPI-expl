#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>Introduction</h3>
# <p>Banks play a crucial role in market economies. They decide who can get financing and on what terms and can make or stop investment decisions. For markets and society to function, individuals and companies need access to credit. Credit scoring algorithms, which predict the probability of default, are the method used by banks to determine whether or not a loan should be granted..</p>
# <h3>objective</h3>
# <p>Creation of a model in which he can try to predict the probability of the customer being able to repay the requested loan to the bank</p>
# <h3>About Dataset </h3>
# <p>History of approx. 250,000 customers in which it was divided between training and test dataset</p>
# <h5 style='text-align: center'>Variable Name Description Type</h5>
# 
# <table>
# <tr>
#   <th>Variable</th>
#   <th>Description</th>
# </tr>
# <tr>
#   <td>SeriousDlqin2yrs</td>
#   <td>Person experienced 90 days past due delinquency or worse Y/N</td>
# </tr>
#     <tr>
#     <td>RevolvingUtilizationOfUnsecuredLines</td>
#      <td>Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits percentage</td>
#     </tr>
#      <tr>
#      <td>Age</td>
#      <td>Age of borrower in years integer</td>
#     </tr>
#       <tr>
#      <td>NumberOfTime3059DaysPastDueNotWorse</td>
#      <td>Number of times borrower has been 30-59 days past due but no worse in the last 2 years. integer</td>
#     </tr>
#      <tr>
#      <td>DebtRatio</td>
#      <td>Monthly debt payments, alimony,living costs divided by monthy gross income percentage</td>
#     </tr>
#      <tr>
#      <td>MonthlyIncome</td>
#      <td>Monthly income real</td>
#     </tr>
#       <tr>
#      <td>NumberOfOpenCreditLinesAndLoans</td>
#      <td>Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards) integer</td>
#     </tr>
#      <tr>
#      <td>NumberOfTimes90DaysLate</td>
#      <td>Number of times borrower has been 90 days or more past due. integer</td>
#     </tr>
#       <tr>
#      <td>NumberRealEstateLoansOrLines</td>
#      <td>Number of mortgage and real estate loans including home equity lines of credit integer</td>
#     </tr>
#           <tr>
#      <td>NumberOfTime60-89DaysPastDueNotWorse</td>
#      <td>Number of times borrower has been 60-89 days past due but no worse in the last 2 years. integer</td>
#     </tr>
#       <tr>
#      <td>NumberOfDependents</td>
#      <td> Number of dependents in family excluding themselves (spouse, children etc.) integer</td>
#     </tr>
#     
# </table>

# In[ ]:


df_train = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')
df_test = pd.read_csv('../input/GiveMeSomeCredit/cs-test.csv')


# In[ ]:


print('Train Shape is :',df_train.shape,'\nTest shape is :',df_test.shape)


# In[ ]:


df_train.head(2)


# In[ ]:


(df_train.isna().sum()/len(df_train)) * 100


# In[ ]:


mask = df_train.isnull()
sns.heatmap(df_train, mask=mask,cmap="YlGnBu");


# <p>Devido a quantidade menor que 5% de missing na coluna NumberOfDependents eu preferi optar pelo drop da mesma</p>

# In[ ]:


df_train.dropna(subset=['NumberOfDependents'],inplace=True)


# In[ ]:


df_train.shape


# In[ ]:


df_append = df_train.append(df_test)


# In[ ]:


(df_train.SeriousDlqin2yrs.value_counts() / len(df_train) ) * 100


# In[ ]:


sns.countplot(x="SeriousDlqin2yrs", data=df_train);


# In[ ]:


sns.distplot((df_train.age));


# <p>I ended up choosing to carry out the test by grouping the age of the customers where in the final model I would test which of the two forms would have the best performance, with only the age column or it grouped</p>

# In[ ]:


bins= [20,60,80,120]
labels_age = ['Adult','Young Senior','Senior']
df_append['AgeGroup'] = pd.cut(df_append['age'], bins=bins, labels=labels_age, right=False)
mask_2 = {
         'Adult':0,
         'Young Senior':1,
         'Senior':2}
df_append['AgeGroup'].replace(mask_2,inplace=True)


# In[ ]:


df_append['AgeGroup'].value_counts()


# In[ ]:


df_append['MonthlyIncome'].fillna(df_append['MonthlyIncome'].median(),inplace=True)
df_append['NumberOfDependents'].fillna(df_append['NumberOfDependents'].median(),inplace=True)


# <p>
# I chose to separate the ages from 18 to 60 as "adults" because above that I will already consider you as a gentleman (retired), another detail that I made a classification above 80 years old due to some rules that at least exist in Brazil even though the original dataset is not Brazilian</p>
# 
# <p>"The profile with the highest approval rate is that of the private sector employee, a graduate and an average income close to 3.2 thousand reais. This type of consumer corresponds to only 9% of those who completed the registration to apply for credit, but 37% were approved." </p>
# <img src='exame.png'>
# <p>"Of the requests made to pay debts, 25% were approved; for investments, 26%, and to renovate the house 28%. The highest approval rate was for purchases, trips and parties, with 32%."
# <a href="https://exame.com/seu-dinheiro/os-perfis-com-mais-chances-de-conseguir-um-emprestimo-segundo-a-finanzero/">Fonte Exame</a>
# </p>
# <p> with that we try to pull to the reality of brazil to see if the profile that corresponds here can be similar to the same that the dataset represents</p>

# In[ ]:


df_train = df_append[0:146076]
df_test = df_append[146076:]


# In[ ]:


#df_append = df_append[df_append != 5400]
df_adult = df_train[df_train['AgeGroup'] == 0]


# In[ ]:


sns.countplot(x='NumberOfDependents',data=df_adult);


# <p>With that we realized that for our current dataset we have that our group of adults tend to have up to 4 dependents, with a low rate above that</p>

# In[ ]:


sns.countplot(df_adult.SeriousDlqin2yrs);


# In[ ]:


sns.countplot(x="AgeGroup", data=df_train);


# In[ ]:


g = sns.jointplot("age", "NumberOfDependents", data=df_train, ylim=(0, 12),
                  color="m", height=7)


# <p> even though in the main dataset we have dropped the numberOfDependents column in our test dataset we still have them as nulls so we will choose to fill it in with values of our median </p>

# In[ ]:


df_train['AgeGroup'].fillna(df_train['AgeGroup'].median(),inplace=True)


# In[ ]:


X = df_train.drop(columns={'Unnamed: 0','age','SeriousDlqin2yrs'})
y = df_train['SeriousDlqin2yrs']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMModel,LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


# In[ ]:


from sklearn.metrics import auc,roc_curve
def plot_roc(pred):
    fpr,tpr,_ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic')
    sns.lineplot(fpr, tpr, label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


print('Train Shapes',X_train.shape,' and ',y_train.shape,'\nTest Shapes',X_test.shape,' and ' ,y_test.shape,'\nOutput Values\n',y_train.value_counts())


# In[ ]:


smote = SMOTE(sampling_strategy = 'minority',k_neighbors = 2,random_state=0)
X_train_smote,y_train_smote = smote.fit_sample(X_train,y_train)

#Realizar Teste Smote + kfold(com cv)


# In[ ]:


print('Train Shapes',X_train_smote.shape,' and ',y_train_smote.shape,'\nTest Shapes',X_test.shape,' and ' ,y_test.shape,'\nOutput Values\n',y_train_smote.value_counts())


# A test was performed using Smote for upsample but the results were worse than with the data as it was (unbalanced), due to this it was chosen to continue with the data unbalanced for the models

# Choosing to make a base model with random forest just for testing and visual selection of features

# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train,y_train)


# In[ ]:


feat_names = X.columns.values
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="y", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[ ]:


pred = clf.predict_proba(X_test)[:,1]
#Predict Primeiro modelo basico
print(roc_auc_score(y_test, pred))


# In[ ]:


plot_roc(pred)


# In[ ]:


Lgb = LGBMClassifier(objective='binary',metrics ='auc')


# In[ ]:


cv = StratifiedKFold(5)
visualizer = RFECV(Lgb, cv=cv, scoring='roc_auc')
visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
visualizer.show();           # Finalize and render the figure


# In[ ]:


pred_rfe = visualizer.predict_proba(X_test)[:,1]


# In[ ]:


plot_roc(pred_rfe)


# In[ ]:


from hyperopt import hp, tpe, fmin
from sklearn.model_selection import cross_val_score


# In[ ]:


#Usando Hypteropt
space = {'n_estimators':hp.quniform('n_estimators', 10, 4000, 10),
        'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),
         'max_depth':hp.quniform('max_depth', 3,7,1),
         'subsample':hp.uniform('subsample', 0.60, 0.95),
         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.95),
         'reg_lambda': hp.uniform('reg_lambda', 1, 20),
        }

def objective(params):
    params = {'n_estimators': int(params['n_estimators']),
             'learning_rate': params['learning_rate'],
             'max_depth': int(params['max_depth']),
             'subsample': params['subsample'],
             'colsample_bytree': params['colsample_bytree'],
             'reg_lambda': params['reg_lambda'],
             }
    
    lgbm= LGBMClassifier(**params)
    cv = StratifiedKFold(5)
    score = cross_val_score(lgbm, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
    return -score


# In[ ]:


best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)


# In[ ]:


lgbm = LGBMClassifier(random_state=0,
                        n_estimators=int(best['n_estimators']), 
                        colsample_bytree= best['colsample_bytree'],
                        learning_rate= best['learning_rate'],
                        max_depth= int(best['max_depth']),
                        subsample= best['subsample'],
                        reg_lambda= best['reg_lambda']
                       )

lgbm.fit(X_train, y_train)


# In[ ]:


lgbm_hype = lgbm.predict_proba(X_test)[:,1]


# In[ ]:


plot_roc(lgbm_hype)


# In[ ]:


#prediction
df_x_test = df_test.drop(columns={'Unnamed: 0','age','SeriousDlqin2yrs'})
pred = lgbm.predict_proba(df_x_test)[:,1]
#output
output = pd.DataFrame({'Id': df_test['Unnamed: 0'],'Probability': pred})
output.to_csv('submission.csv', index=False)


# In[ ]:




