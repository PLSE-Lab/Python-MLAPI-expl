#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes

from sklearn.svm import SVC,LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report 
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc

from scipy.stats import skew, kurtosis

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#plt.style.use('dark_background')
current_palette = sns.color_palette('colorblind')
sns.palplot(current_palette)


# ### import data

# In[ ]:


data_set = pd.read_csv('../input/loan_data_set.csv')


# In[ ]:


data_set.info()


# In[ ]:


data_set.LoanAmount = data_set.LoanAmount*1000


# In[ ]:


data_set.Loan_Status.value_counts(normalize = True).reset_index()


# In[ ]:


sns.barplot(data = data_set.Loan_Status.value_counts(normalize = True).reset_index(),
            x = 'index',
            y = 'Loan_Status',
            palette=current_palette)


# - we can see the data is not balanced, 69% approved for load while 31% where not
# - lets check the proportion in each feature between Y and N

# In[ ]:


def prop_check(data):
    f, axes = plt.subplots(6,2,figsize= (12,20))
    plt.suptitle('Train data, count vs proportion of each object feature vs Loan_Status', size =16, y = 0.9)
    col = data.columns[1:data.shape[1]-1]
    r = 0
    for i in col:
        if (data.dtypes == 'object')[i]:        
            data_prop = (data['Loan_Status']
                          .groupby(data[i])
                          .value_counts(normalize = True)
                          .rename('prop')
                          .reset_index())
            sns.countplot(data = data, 
                          x ='Loan_Status', 
                          hue = i, 
                          ax = axes[r,0], 
                          hue_order=data_prop[i].unique(), 
                          palette=current_palette)
            sns.barplot(data = data_prop, 
                        x = 'Loan_Status', 
                        y = 'prop',
                        hue = i,
                        ax = axes[r,1],
                        palette=current_palette)
            r = r+1
prop_check(data_set)


# - as the graphs indicate, they are mostly distributed evenly between the features
# - we can also learn there is no ovious correlation jumping out at this stage
# - lets make the Loan_ID column to be an index for further analysis (we could also delete it)

# In[ ]:


def make_index(df):
    df.set_index('Loan_ID', inplace=True)
    return df
data_set = make_index(data_set)


# - drop all NAs
# - previous attempts to fill the gaps showed that most frequent dummy classifier bring the best results, although increases the bias within the data

# In[ ]:


data_set.dropna(inplace=True)


# In[ ]:


sns.pairplot(data_set, hue = 'Loan_Status', palette=current_palette)


# - we can see here first sign to some correlation with both the incomes features and the loan amount
# - lets work them out to be more models friendly (normaly distributed and with 0 mean)
# 

# ## Categorizing numerically object columns by column name
# - each column translated to binary value
# - multi variables columns slpitted with dummy columns
# - drop the duplications in the data

# In[ ]:


def categorize(df):
    df.Gender.replace({'Male': 1, 'Female': 0}, inplace = True)
    df.Married.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Education.replace({'Graduate': 1, 'Not Graduate': 0}, inplace = True)
    df.Self_Employed.replace({'Yes': 1, 'No': 0}, inplace = True)
    df = df.join(pd.get_dummies(df.Dependents, prefix='Dependents'))
    df.drop(columns= ['Dependents', 'Dependents_3+'], inplace=True)
    df = df.join(pd.get_dummies(df.Property_Area, prefix='Property_Area'))
    df.drop(columns= ['Property_Area', 'Property_Area_Rural'], inplace=True)
    return df


# In[ ]:


data_set = categorize(data_set)


# In[ ]:


data_set.Loan_Status.replace({'Y': 1, 'N':0}, inplace=True)


# ## Model selection

# > ### Adding new features:
# - naive estimation of monthly loan return (LoanAmount/Loan_Amount_Term) normalized and with ln() let us get the distribution as closer as can be to normal distribution
# - total income (ApplicantIncome + CoaplicantIncome) normlized and with ln() let us get the distribution as closer as can be to normal distribution

# In[ ]:


def add_feat(df):
    ln_monthly_return = np.log(df.LoanAmount/df.Loan_Amount_Term)
    df['ln_monthly_return'] = (ln_monthly_return - np.mean(ln_monthly_return))/(np.std(ln_monthly_return)/np.sqrt(len(ln_monthly_return)))
    
    ln_total_monthly_income = np.log(df.ApplicantIncome + df.CoapplicantIncome)
    df['ln_total_income'] = (ln_total_monthly_income - np.mean(ln_total_monthly_income))/(np.std(ln_total_monthly_income)/np.sqrt(len(ln_total_monthly_income)))
    
    ln_LoanAmount = np.log(1000*df.LoanAmount)
    df['ln_LoanAmount'] = (ln_LoanAmount - np.mean(ln_LoanAmount))/(np.std(ln_LoanAmount)/np.sqrt(len(ln_LoanAmount)))
    
    
    return df


# In[ ]:


data_set = add_feat(data_set)


# - lets check the new features distributions

# In[ ]:



def norm_plt(df):
    f, axes = plt.subplots(3,2,figsize= (12,15),squeeze=False)

    ######total income########
    sns.distplot(df.ln_total_income
                 ,ax=axes[0,0]).set_title('ln(total_income) norm distribution')
    #axes[0,0].set_xlim(-100,100)
    axes[0,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_total_income),
                                          kurtosis(df.ln_total_income)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[0,0].transAxes,
                   bbox={'facecolor': 'white'})
    sns.distplot((df.ApplicantIncome+df.CoapplicantIncome),
                 ax=axes[0,1]).set_title('total_income distribution')
    axes[0,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ApplicantIncome+df.CoapplicantIncome),
                           kurtosis(df.ApplicantIncome+df.CoapplicantIncome)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[0,1].transAxes,
                   bbox={'facecolor': 'white'})

    #######monthly return###########
    sns.distplot(df.ln_monthly_return,
                 ax=axes[1,0]).set_title('ln(monthly_return) norm distribution')
    #axes[1,0].set_xlim(-100,100)
    axes[1,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_monthly_return),
                           kurtosis(df.ln_monthly_return)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[1,0].transAxes,
                   bbox={'facecolor': 'white'})

    sns.distplot((1000*df.LoanAmount/df.Loan_Amount_Term),
                 ax=axes[1,1]).set_title('monthly_return distribution')
    axes[1,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.LoanAmount/df.Loan_Amount_Term),
                           kurtosis(df.LoanAmount/df.Loan_Amount_Term)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[1,1].transAxes,
                   bbox={'facecolor': 'white'})

    ######norm ln_LoanAmount########
    sns.distplot(df.ln_LoanAmount
                 ,ax=axes[2,0]).set_title('ln(LoanAmount) norm distribution')
    #axes[2,0].set_xlim(-100,100)
    axes[2,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_LoanAmount),
                                          kurtosis(df.ln_LoanAmount)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[2,0].transAxes,
                   bbox={'facecolor': 'white'})
    sns.distplot((df.LoanAmount),
                 ax=axes[2,1]).set_title('LoanAmount distribution')
    axes[2,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.LoanAmount),
                           kurtosis(df.LoanAmount)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[2,1].transAxes,
                   bbox={'facecolor': 'white'})
    
    
    ####### adding grid to the graph#########
    for i in range(3):
        for j in range(2):
            axes[i,j].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')


# In[ ]:


norm_plt(data_set)


# - the skew and the kurtosis are much closer to 0, as expected from normalized normal distribution
# - now lets drop the neglectable features those are dependeds of the new features created above
# - i found in previous runs the Married and Dependents are neglectable too

# In[ ]:


dropit=['LoanAmount', 
        'Loan_Amount_Term', 
        'ApplicantIncome',
        'CoapplicantIncome',
        'Married',
        'Dependents_0',
        'Dependents_1',
        'Dependents_2']
data_set.drop(columns=dropit, 
           inplace=True)


# - the following is a sanity check for the distribution between Y and N in the target column
# - the distribution between the values didn't change much (less than 0.5%)

# In[ ]:


data_set['Loan_Status'].value_counts(normalize=True)
sns.barplot(data = data_set.Loan_Status.value_counts(normalize = True).reset_index(),
            x = 'index',
            y = 'Loan_Status',
            palette=current_palette)
plt.grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')


# ### Classifiers optimization

# - In our case: binary classification, the "cross_val_score" function uses StratifiedKFold cross validation to reduce the bias effect in imbalanced data.
# - it is a good approach for the cases the target column distribution is biased

# In[ ]:



def cv_check(X,y, CV):
    models = [
        RandomForestClassifier(criterion='gini',
                               n_estimators=50,
                               max_depth=11,
                               max_features=6,
                               random_state=42,
                               class_weight='balanced_subsample',
                               n_jobs=4),
        SVC(C=1, kernel='rbf', gamma='auto',random_state=42,class_weight='balanced'),
        LogisticRegression(solver='lbfgs',
                           multi_class='ovr',
                           max_iter=500,
                           C=1,
                           random_state=42,
                           class_weight='balanced'),
        GaussianNB(),
        #LinearSVC(C=1, 
        #         max_iter=500,
        #          random_state=0),
        DummyClassifier(strategy='most_frequent',random_state=42)
    ]

    entries = []
    
    for model in models:
        model_name = model.__class__.__name__
        print ("Currently fitting: {}".format(model_name))
        accuracies = cross_val_score(model,
                                     X,
                                     y, 
                                     scoring='roc_auc', cv=CV, n_jobs=4)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'roc_auc'])
        
    return cv_df


# In[ ]:


def cv_bp(cv_df, title, axes):
    axes.grid(b=True, 
              which='both', 
              axis='both', 
              color='grey', 
              linestyle = '--', 
              linewidth = '0.3')    
    sns.boxplot(x='model_name', 
                y='roc_auc', 
                data=cv_df, 
                width = 0.5, 
                ax=axes,
                palette=current_palette).set_title(title)
    sns.stripplot(x='model_name', 
                  y='roc_auc',
                  data=cv_df, 
                  size=5, jitter=True, 
                  edgecolor="grey", 
                  linewidth=1, 
                  ax=axes)
    plt.ylim(0.2,1)
    plt.savefig('{}.png'.format(title), format='png')
    #plt.show()


# In[ ]:


f, axes = plt.subplots(1,1,figsize= (20,8),squeeze=False, sharey=True)
cv_bp(cv_check(data_set.drop(['Loan_Status'],axis=1),
               data_set.Loan_Status,10), '{} without NAs'.format('train'),axes[0,0])


# - using grid search cross validation we scanned to find the optimal values for each model variables
# - the values chosen after several runs to get the optimum variables within the optimal range for the roc_auc as closer we can get to 1

# ### RandomForest

# In[ ]:


def model_score(train, model, grid_values, scorers_list):
    X_train = train.drop(columns=['Loan_Status'])
    y_train = train['Loan_Status']
    
    clf_dict = {}
    
    for i, scorer in enumerate(scorers_list):
        clf_eval = GridSearchCV(model, param_grid=grid_values, scoring=scorer, cv=5, iid=False)
        clf_eval.fit(X_train,y_train)
        print('Grid best parameters for {0}: {1} scoring: {2}'
              .format(scorer, clf_eval.best_params_, round(clf_eval.best_score_,3)))
        clf_dict[scorer] = clf_eval
    return clf_dict


# In[ ]:


grid_values = {'max_features': [4, 5, 6, 7],
              'max_depth': [3, 7, 11, 13]}
scorers_list = ['accuracy','roc_auc','precision','recall', 'f1']

rf_cv = model_score(data_set,
            RandomForestClassifier(random_state=42, 
                                   n_jobs=4, 
                                   class_weight='balanced_subsample', 
                                   n_estimators=50), 
            grid_values, 
            scorers_list)

temp_df1 = pd.DataFrame()
for i in scorers_list:
      temp_df1[i]=rf_cv[i].cv_results_['mean_test_score'][rf_cv[i].cv_results_['param_max_features']==4]
temp_df1['max_depth'] = rf_cv['roc_auc'].cv_results_['param_max_depth'][rf_cv['roc_auc'].cv_results_['param_max_features']==4]
temp_df1.set_index('max_depth', inplace=True)
print('4:\n')
temp_df1

temp_df2 = pd.DataFrame()
for i in scorers_list:
      temp_df2[i]=rf_cv[i].cv_results_['mean_test_score'][rf_cv[i].cv_results_['param_max_features']==6]
temp_df2['max_depth'] = rf_cv['roc_auc'].cv_results_['param_max_depth'][rf_cv['roc_auc'].cv_results_['param_max_features']==6]
temp_df2.set_index('max_depth', inplace=True)
print('6:\n')
temp_df2


# ### LogisticRegression

# In[ ]:


grid_values = {'C': [0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}
scorers_list = ['accuracy','roc_auc','precision','recall', 'f1']


lr_cv = model_score(data_set,
                    LogisticRegression(solver='liblinear',random_state=42, max_iter=500,
                                      class_weight='balanced'),
                    grid_values,
                    scorers_list)


temp_df1 = pd.DataFrame()
for i in scorers_list:
      temp_df1[i]=lr_cv[i].cv_results_['mean_test_score'][lr_cv[i].cv_results_['param_penalty']=='l1']
temp_df1['C'] = lr_cv['roc_auc'].cv_results_['param_C'][lr_cv['roc_auc'].cv_results_['param_penalty']=='l1']
temp_df1.set_index('C', inplace=True)
print('l1:\n')
temp_df1

temp_df2 = pd.DataFrame()
for i in scorers_list:
      temp_df2[i]=lr_cv[i].cv_results_['mean_test_score'][lr_cv[i].cv_results_['param_penalty']=='l2']
temp_df2['C'] = lr_cv['roc_auc'].cv_results_['param_C'][lr_cv['roc_auc'].cv_results_['param_penalty']=='l2']
temp_df2.set_index('C', inplace=True)
print('l2:\n')
temp_df2


# ### SVC

# In[ ]:


grid_values = {'C': [1, 10],
              'gamma': [0.5, 0.7, 0.9, 0.95]}
scorers_list = ['accuracy','roc_auc','precision','recall', 'f1']


svc_cv = model_score(data_set,
                    SVC(random_state=42, class_weight='balanced',kernel='rbf'),
                    grid_values,
                    scorers_list)


temp_df1 = pd.DataFrame()
for i in scorers_list:
      temp_df1[i]=svc_cv[i].cv_results_['mean_test_score'][svc_cv[i].cv_results_['param_C']==1]
temp_df1['gamma'] = svc_cv['roc_auc'].cv_results_['param_gamma'][svc_cv['roc_auc'].cv_results_['param_C']==1]
temp_df1.set_index('gamma', inplace=True)
print('C=1:\n')
temp_df1

temp_df2 = pd.DataFrame()
for i in scorers_list:
      temp_df2[i]=svc_cv[i].cv_results_['mean_test_score'][svc_cv[i].cv_results_['param_C']==10]
temp_df2['gamma'] = svc_cv['roc_auc'].cv_results_['param_gamma'][svc_cv['roc_auc'].cv_results_['param_C']==10]
temp_df2.set_index('gamma', inplace=True)
print('C=10:\n')
temp_df2


# ### The following are the main functions to run each model evaluation in the chosen best values

# In[ ]:


def mod_eval(df,predictions, predprob, y_test, title):
    # prints confusion matrix heatmap    
    cm = confusion_matrix(df.Loan_Status[y_test.index], predictions)
    sns.heatmap(cm, annot=True, fmt='.3g', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']).set_title(title)
    plt.xlabel('Real')
    plt.ylabel('Predict')
    
    print(classification_report(df.Loan_Status[y_test.index], predictions))
    
    f, axes = plt.subplots(1,2,figsize= (20,6),squeeze=False)

    fpr, tpr, _ = roc_curve(df.Loan_Status[y_test.index], predprob[:,1])
    roc_auc = auc(fpr,tpr)
    axes[0,0].plot(fpr, tpr, lw=3)
    axes[0,0].set_title('{} ROC curve (area = {:0.2f})'.format(title, roc_auc))
    axes[0,0].set(xlabel='False Positive Rate',ylabel='True Positive Rate')
    axes[0,0].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')

    precision, recall, thresholds = precision_recall_curve(y_test, predprob[:,1])
    best_index = np.argmin(np.abs(precision-recall)) # set the best index to be the minimum delta between precision and recall
    axes[0,1].plot(precision,recall)
    axes[0,1].set_title('{} Precision-Recall Curve'.format(title))
    axes[0,1].set(xlabel='Precision', ylabel='Recall', xlim=(0.4,1.05))
    axes[0,1].plot(precision[best_index],recall[best_index],'o',color='r')
    axes[0,1].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')


# In[ ]:


def model_training(classifier,df):
    clf = classifier
    t=df.drop(columns=['Loan_Status'])
    X_train, X_test, y_train, y_tests = train_test_split(t,
                                                         df['Loan_Status'],
                                                         test_size=ts,
                                                         stratify=df['Loan_Status'])
    clf.fit(X_train, y_train)
    return clf


# ### Chosen values per model

# In[ ]:


#RandomForest
max_depth=11
max_features=6


# In[ ]:


#LogisticRegression
lr_C=0.1
penalty='l1'


# In[ ]:


#SVC
svc_C=1
gamma=0.9


# In[ ]:


#Test Size
ts = 0.333


# ## RandomForest

# In[ ]:


rf = model_training(RandomForestClassifier(random_state=42, 
                                           n_jobs=4, 
                                           n_estimators=50, 
                                           max_depth=max_depth,
                                           max_features=max_features),data_set)

t=data_set.drop(columns=['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(t,
                                                     data_set['Loan_Status'],
                                                     test_size=ts,
                                                     stratify=data_set['Loan_Status'])

mod_eval(data_set, rf.predict(X_test), rf.predict_proba(X_test), y_test, 'RandomForest')
fi_df = pd.DataFrame({'fi': rf.feature_importances_},index=t.columns).sort_values(by='fi', ascending=False)
fi_df
plt.show()
plt.figure(figsize=(12,5))
plt.xticks(rotation='vertical')
sns.barplot(x=fi_df.index, y=fi_df['fi'], palette=current_palette)
plt.grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')
plt.show()


# ## LogisticRegression

# In[ ]:


lr = model_training(LogisticRegression(C=lr_C, 
                                       penalty=penalty,
                                       solver='liblinear',
                                       max_iter=1000),data_set)

t=data_set.drop(columns=['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(t,
                                                    data_set['Loan_Status'],
                                                    test_size=ts,
                                                    random_state = 42, stratify=data_set['Loan_Status'])

t = 0.71
predprob = lr.predict_proba(X_test)

pred_y = [np.ceil(x) if x>=t else np.floor(x) for x in predprob[:,1]]

#pred_y = lr.predict(X_test)
mod_eval(data_set, pred_y, lr.predict_proba(X_test), y_test, 'LogisticRegressin') 
plt.show()


# ## NaiveBayes

# In[ ]:


gnb = model_training(GaussianNB(),data_set)

t=data_set.drop(columns=['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(t,
                                                    data_set['Loan_Status'],
                                                    test_size=ts,
                                                    random_state = 42, stratify=data_set['Loan_Status'])


t = 0.75
predprob = gnb.predict_proba(X_test)

pred_y = [np.ceil(x) if x>=t else np.floor(x) for x in predprob[:,1]]
#pred_y = gnb.predict(X_test)
mod_eval(data_set,pred_y, gnb.predict_proba(X_test), y_test, 'GaussianNB')
plt.show()


# ## SVC

# In[ ]:


svc = model_training(SVC(kernel='linear',
                         C=1, 
                         gamma='auto',
                         class_weight='balanced',
                         probability=True),data_set)


t=data_set.drop(columns=['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(t,
                                                    data_set['Loan_Status'],
                                                    test_size=ts,
                                                    random_state = 42, stratify=data_set['Loan_Status'])

t=0.75
print('t={}'.format(t))
predprob = svc.predict_proba(X_test)

pred_y = [np.ceil(x) if x>=t else np.floor(x) for x in predprob[:,1]]
#pred_y = svc.predict(X_test)

mod_eval(data_set,pred_y, svc.predict_proba(X_test), y_test, 'SVC')
plt.show()


# ## Dummy

# In[ ]:


dummy = model_training(DummyClassifier(strategy='stratified'),data_set)

t=data_set.drop(columns=['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(t,
                                                    data_set['Loan_Status'],
                                                    test_size=ts,
                                                    random_state = 42, stratify=data_set['Loan_Status'])

t = 0.5
predprob = dummy.predict_proba(X_test)

pred_y = [np.ceil(x) if x>=t else np.floor(x) for x in predprob[:,1]]
#pred_y = dummy.predict(X_test)

mod_eval(data_set, pred_y, dummy.predict_proba(X_test), y_test, 'Dummy')
plt.show()


# ## Summary
# - The RandomForest classifier brings the best results
# - Although, it might seems a bit overfitted according the CrossValidation session before the last fitting
# - Dummy classifier is essensial for sanity check along the analysis
