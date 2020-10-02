#!/usr/bin/env python
# coding: utf-8

# # Imported Library
# 
# Load all libraries for prepocessing data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pd.set_option('Display.max_columns',500)
pd.set_option('Display.max_rows',500)


# # Imported Dataset
# 
# Load data for classification. In this moment, we just used application_train and application_test data to build the model.

# In[ ]:


app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')


# In[ ]:


app_train.head()


# <hr>

# # Null value analysis
# 
# In this stage, we try to handle the missing data in application_train and application_test.

# In[ ]:


app_train.isnull().sum()


# In[ ]:


app_train[app_train.isnull().any(axis=1)].shape


# In[ ]:


app_train[~app_train.isnull().any(axis=1)].shape


# In[ ]:


app_train.select_dtypes(include=object).dtypes


# In[ ]:


app_test.select_dtypes(include=object).isnull().sum()


# In[ ]:


app_train.select_dtypes(include=object).isnull().sum()


# There are 298909 rows contain missing data in application_train and 8602 in application_test.

# In[ ]:


app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna('Unaccompanied')
app_test.NAME_TYPE_SUITE = app_test.NAME_TYPE_SUITE.fillna('Unaccompanied')


# In[ ]:


app_train.OCCUPATION_TYPE = app_train.OCCUPATION_TYPE.fillna('Others')
app_test.OCCUPATION_TYPE = app_test.OCCUPATION_TYPE.fillna('Others')


# In[ ]:


app_train.FONDKAPREMONT_MODE = app_train.FONDKAPREMONT_MODE.fillna('not specified')
app_test.FONDKAPREMONT_MODE = app_test.FONDKAPREMONT_MODE.fillna('not specified')


# In[ ]:


app_train.HOUSETYPE_MODE = app_train.HOUSETYPE_MODE.fillna('Others')
app_test.HOUSETYPE_MODE = app_test.HOUSETYPE_MODE.fillna('Others')


# In[ ]:


app_train.WALLSMATERIAL_MODE = app_train.WALLSMATERIAL_MODE.fillna('Others')
app_test.WALLSMATERIAL_MODE = app_test.WALLSMATERIAL_MODE.fillna('Others')


# In[ ]:


app_train = app_train.drop(columns=['EMERGENCYSTATE_MODE'])
app_test = app_test.drop(columns=['EMERGENCYSTATE_MODE'])


# In[ ]:


min_ext_1 = app_train.EXT_SOURCE_1.min()
min_ext_2 = app_train.EXT_SOURCE_2.min()
min_ext_3 = app_train.EXT_SOURCE_3.min()


# In[ ]:


app_train.EXT_SOURCE_1 = app_train.EXT_SOURCE_1.fillna(min_ext_1)
app_train.EXT_SOURCE_2 = app_train.EXT_SOURCE_2.fillna(min_ext_2)
app_train.EXT_SOURCE_3 = app_train.EXT_SOURCE_3.fillna(min_ext_3)

app_test.EXT_SOURCE_1 = app_test.EXT_SOURCE_1.fillna(min_ext_1)
app_test.EXT_SOURCE_2 = app_test.EXT_SOURCE_2.fillna(min_ext_2)
app_test.EXT_SOURCE_3 = app_test.EXT_SOURCE_3.fillna(min_ext_3)

app_train = app_train.fillna(0)
app_test = app_test.fillna(0)


# In[ ]:


print(app_train.isnull().sum().sum())
print(app_test.isnull().sum().sum())


# In this stage, we made some assumptions and filled missing data with the assumptions. For column NAME_TYPE_SUITE, we decided to fill missing data with 'Unaccompanied'; OCCUPATION_TYPE, HOUSE_TYPE, and WALLSMATERIAL_MODE with 'Others'; FONDKAPREMONT_MODE with 'not specified'; EXT_SOURCE with the minimum value of each column; and others with zero.

# # Exploratory Data Analysis

# ## Distribution of application_train data

# In[ ]:


x_cat = app_train[app_train.select_dtypes(include=object).columns].columns
x_num = app_train[app_train.select_dtypes(exclude=object).columns].columns.drop(['SK_ID_CURR','TARGET'])


# In[ ]:


def plot_hist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.countplot(x=x,data=app_train)
    plt.xlabel(str(x))
    plt.title('Histogram of '+str(x))
    plt.xticks(rotation=70)
    plt.show()


# In[ ]:


plot_hist('TARGET')


# We can see that, the dataset is imbalanced.

# In[ ]:


for x in x_cat:
    plot_hist(x)


# In[ ]:


def plot_dist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.distplot(app_train[x])
    plt.xlabel(str(x))
    plt.title('Distribution of '+str(x))
    plt.show()


# In[ ]:


for x in x_num:
    plot_dist(x)


# ## Distribution of application_test data

# In[ ]:


def plot_hist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.countplot(x=x,data=app_test)
    plt.xlabel(str(x))
    plt.title('Histogram of '+str(x))
    plt.xticks(rotation=70)
    plt.show()


# In[ ]:


for x in x_cat:
    plot_hist(x)


# In[ ]:


def plot_dist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.distplot(app_test[x])
    plt.xlabel(str(x))
    plt.title('Distribution of '+str(x))
    plt.show()


# In[ ]:


for x in x_num:
    plot_dist(x)


# <hr>

# # Data Preprocessing

# ## Categorical data, numerical data, and target data separation
# 
# In this stage, we separated catagerical data, numerical data, and target data to do different treatment.

# In[ ]:


categorical = app_train[app_train.select_dtypes(include=object).columns]
x_cat = categorical.columns
categorical.head()


# In[ ]:


numerical = app_train[app_train.select_dtypes(exclude=object).columns]
numerical = numerical.drop(columns=['SK_ID_CURR','TARGET'])
x_num = numerical.columns
numerical.head()


# In[ ]:


target = app_train.TARGET
target.head()


# ## Numerical data normalization
# 
# For numerical columns, we normalized using MinMaxScaller to remove outlier.

# In[ ]:


scaller = MinMaxScaler()
app_train[x_num] = scaller.fit_transform(app_train[x_num])
app_test[x_num] = scaller.transform(app_test[x_num])
app_train[x_num].head()


# ## Categorical data encoding
# 
# For Categorical data, we converted to numerical using Label Encoder. By this method, all categorical data are sorted by alphabetically.

# In[ ]:


for x in x_cat:
    lb = LabelEncoder()
    app_train[x] = lb.fit_transform(app_train[x])
    app_test[x] = lb.transform(app_test[x])


# In[ ]:


app_train[x_cat].head()


# ## Target data preprocessing
# 
# We just make sure the target is binarizer using Label Binarizer method.

# In[ ]:


lb = LabelBinarizer()
app_train['TARGET'] = lb.fit_transform(app_train.TARGET)


# In[ ]:


app_train.TARGET.head()


# # Fix Data

# In[ ]:


app_train.head()


# In[ ]:


x_call = app_train.columns[2:]


# In[ ]:


app_test.head()


# <hr>

# ## Feature correlation analysis
# 
# In this stage, we try to analyze the correlation among features.

# ### 1. Heatmap correlation

# In[ ]:


corr = app_train[x_num].corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[ ]:


fig,ax = plt.subplots(figsize=(12,10))
corr = app_train[x_call].corr()
hm = sns.heatmap(corr,ax=ax,vmin=-1,vmax=1,annot=False,cmap='coolwarm',square=True,fmt='.2f',linewidths=.05)


# In[ ]:


for x in x_call:
    msg = "%s : %.3f" % (x,np.corrcoef(app_train[x],app_train.TARGET)[0,1])
    print(msg)


# We can see that, there is no correlation between each features and the target. However, some features have high correlation among them. By this condition, we decided to use tree model rather than Ordinary Least Squared Model.

# <hr>

# # Train Test Split
# 
# In this stage, we separate data to be 67% train_df and 33% test_df.

# In[ ]:


train_df, test_df = train_test_split(app_train,test_size=0.33,shuffle=True,stratify=app_train.TARGET,
                                     random_state=217)


# # Choosing Model
# 
# In this stage, we try to build model using several kinds of model. To choose the best model, we using several metrics, those are accuracy_score to see the accuracy, roc_auc_acore, average_precision or PR_AUC, and f1_score. However, we give more attention to roc_auc.

# In[ ]:


from sklearn.model_selection import StratifiedKFold,cross_validate,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,log_loss,roc_curve
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.utils import compute_sample_weight,compute_class_weight
from sklearn.calibration import calibration_curve,CalibratedClassifierCV
from sklearn import model_selection


# In[ ]:


x_calls = train_df.columns[2:]


# In[ ]:


scorer = ('accuracy','roc_auc','f1_weighted','average_precision')


# In[ ]:


models = []
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(class_weight='balanced')))
models.append(('ETC', ExtraTreesClassifier(class_weight='balanced')))
models.append(('XGBC', XGBClassifier(scale_pos_weight=189399/16633)))
models.append(('GBM', LGBMClassifier(class_weight='balanced')))


# In[ ]:


for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=217, shuffle=True)
    cv_results = cross_validate(model, train_df[x_calls], train_df.TARGET,cv=kfold, scoring=scorer)
    cv_results1=cv_results['test_accuracy']
    cv_results2=cv_results['test_roc_auc']
    cv_results3=cv_results['test_f1_weighted']
    cv_results4=cv_results['test_average_precision']
    msg = "%s by Accuracy: %f(%f), by ROC_AUC: %f(%f), by F1-score: %f(%f), PR_AUC: %f(%f)" % (name, np.mean(cv_results1),
        np.std(cv_results1),np.mean(cv_results2),np.std(cv_results2),np.mean(cv_results3),np.std(cv_results3),
        np.mean(cv_results4),np.std(cv_results4))
    print(msg)


# By the test above, we choose LightGBM to build the model.

# # Evaluating the best model
# 
# Before evaluating, we used RandomSearchCV to choose the better parameters for the model. However, we don't put the method in this script to handle the long running time.

# In[ ]:


model_gbm = LGBMClassifier(boosting_type='gbdt', class_weight='balanced',
        colsample_bytree=1.0, importance_type='split',
        learning_rate=0.09275695087706179, max_depth=3669,
        min_child_samples=60, min_child_weight=0.001, min_data=6,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=64,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, sub_feature=0.7757070409332384, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
model_gbm.fit(train_df[x_calls], train_df.TARGET)


# In[ ]:


predictions = model_gbm.predict(test_df[x_calls])
prob = model_gbm.predict_proba(test_df[x_calls])[:,1]
test_df['TARGET_hat']=predictions
test_df['TARGET_prob']=prob
Y_validation = test_df.TARGET
print("Accuracy Score: %f" % accuracy_score(Y_validation, predictions))
print("ROC_AUC: %f" % roc_auc_score(Y_validation, prob,average='weighted'))
print("PR_AUC: %f" % average_precision_score(Y_validation, prob,average='weighted'))
print("F1: %f" % f1_score(Y_validation, predictions,average='weighted'))
print("Recall: %f" % recall_score(Y_validation, predictions,average='weighted'))
print("Precision: %f" % precision_score(Y_validation, predictions,average='weighted'))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# By the model, we got 71.83% accuracy score, 75.93% roc_auc, and 78.07% F1_score. The roc_auc curve can be seen below.

# In[ ]:


fpr_rf_lm, tpr_rf_lm, _ = roc_curve(test_df.TARGET, test_df.TARGET_prob)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# # Calibration to get probability
# 
# In this stage we convert the score of model to probability using CalibrationClassifierCV. There are two functions that can be used to calibrate the model, those are Sigmoid Function and Isotonic function. To choose the best calibration, we used Brier_score_loss and log_loss as metrics. The lower the both score, the better model.

# In[ ]:


def plot_calibration_curve(est, name, X_train, y_train, X_test, y_test):
    isotonic = CalibratedClassifierCV(est, cv='prefit', method='isotonic')
    sigmoid = CalibratedClassifierCV(est, cv='prefit', method='sigmoid')
    lr = LogisticRegression(C=1., solver='lbfgs',class_weight='balanced')
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),(est, name),(isotonic, name + ' + Isotonic'),(sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())
        L2_score = log_loss(y_test, prob_pos)
        print("%s:" % name)
        print("\tBrier: %.3f" % (clf_score))
        print("\tLog Loss: %.3f" % (L2_score))
        print("\tAUC: %.3f" % roc_auc_score(y_test, prob_pos,average='weighted'))
        print("\tF1: %.3f\n" % f1_score(y_test, y_pred,average='weighted'))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()


# In[ ]:


plot_calibration_curve(model_gbm,'LGBM',train_df[x_calls], train_df['TARGET'],
                       test_df[x_calls], test_df['TARGET'])


# As we can see above, the calibration result using isotonic funstion got the same score with that of sigmoid function in Brier Loss Score. Nevertheless, the loss score of sigmoid function is slighly better than that of Isotonic function. Therefore, we decided to use Sigmoid function to calibrate the score of model to be probability of chossing the positif target.

# # Building Fix Model

# In[ ]:


model_fix = LGBMClassifier(boosting_type='gbdt', class_weight='balanced',
        colsample_bytree=1.0, importance_type='split',
        learning_rate=0.09275695087706179, max_depth=3669,
        min_child_samples=60, min_child_weight=0.001, min_data=6,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=64,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, sub_feature=0.7757070409332384, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
model_fix.fit(app_train[x_calls],app_train.TARGET)


# ## Feature impotances of fix model

# In[ ]:


importances = model_fix.feature_importances_
indices = np.argsort(importances)[::-1]


# In[ ]:


def variable_importance(importance, indices,x):
    print("Feature ranking:")
    importances = []
    for f in range(len(x)):
        i = f
        t=0
        print("%d. The feature '%s' has a Mean Decrease in Gini of %f" % (f + 1,x[indices[i]],importance[indices[f]]))
        importances.append([x[indices[i]],importance[indices[f]]])
    importances = pd.DataFrame(importances,columns=['Features','Gini'])
    return importances

importance = variable_importance(importances, indices,x_calls)


# As we can see above, according to Gini Score the most impotant feature is EXT_SOURCE_3 followed by AMT_CREDIT, EXT_SOURCE_1, AMT_ANNUITY, EXT_SOURCE_2, and so on. However, there are nine features that got zero gini score.

# ## Calculating the probability of application_test data

# In[ ]:


model_fix = CalibratedClassifierCV(model_fix, cv='prefit', method='sigmoid')
model_fix.fit(app_train[x_calls],app_train.TARGET)


# In[ ]:


TARGET = model_fix.predict_proba(app_test[x_calls])[:,1]
submission = pd.DataFrame({'SK_ID_CURR':app_test['SK_ID_CURR'],'TARGET':TARGET})


# In[ ]:


submission.to_csv('submission.csv', index=False)
submission.head()

