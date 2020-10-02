#!/usr/bin/env python
# coding: utf-8

# #  Part 1: Initial Data Exploration

# In[ ]:


# !pip install featexp


# In[ ]:


# General use libraries
import numpy as np
import os
import pandas as pd
import pickle 
from scipy.stats import mode

# Set numpy and pandas options for viewing
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('max_colwidth',1000)
pd.options.display.max_info_columns = 999
pd.set_option('expand_frame_repr',True)

# Data Visualization Libaries
import seaborn as sns
import featexp
from matplotlib import pyplot as plt
from matplotlib import style

# keras and sklearn for machine learning 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


# In[ ]:


# import data
train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
meta = pd.read_csv('../input/metadata/Metadata.csv',index_col='variable', header=0)
print(f'Train data:{train.shape}, Test data: {test.shape}, Metadata: {meta.shape}')


# In[ ]:


# check how data is structured + datatypes
print('Train')
print(train.info(),'\n')
print('Test')
print(test.info())


# In[ ]:


# check for duplicates
print(f'Train duplicates: {len(train[train.duplicated()])}\nTest Duplicates: {len(test[test.duplicated()])}')


# In[ ]:


# define function to change labels columns and convert to numeric
def chg_lb_num(df, col_ls):
    '''
    ====================================
    df: dataframe
    col_ls: a list of column names

    >>> df
        |   a   |  b  |  c  |
    ---------------------------
    0   | "yes" |  2  |  1  |
    1   | "no"  |  4  |  1  |
    2   | "yes" |  6  |  1  |

    >>> chg_lb_num(df,[a])
        |  a   |  b  |  c  |
    ---------------------------    
    0   |  1   |  2  |  1  |
    1   |  0   |  4  |  1  |
    2   |  1   |  6  |  1  |
    a:
    [1 0 1]
    ====================================
    '''
    for item in col_ls:
        df[item] = pd.to_numeric(df[item].replace(['yes','no'],['1','0']), errors='coerce')
        print(f'{item}:\n{df[item].unique()}\n')
    return df


# In[ ]:


label_chg_ls = ['dependency','edjefe','edjefa']
train = chg_lb_num(train, label_chg_ls)
test = chg_lb_num(test, label_chg_ls)


# In[ ]:


# show metadata
print(meta.info())
meta


# In[ ]:


# summarize train data
train.describe()


# In[ ]:


# check for train data outliers
train.describe().loc['max'][(train.describe().loc['max'] > (train.describe().loc['mean'] + 10 * train.describe().loc['std'])) == True][train.describe().loc['max']>1]


# In[ ]:


# check for test data outliers
test.describe().loc['max'][(test.describe().loc['max'] > (test.describe().loc['mean'] + 10 * test.describe().loc['std'])) == True][test.describe().loc['max']>1]


# In[ ]:


#outlier in test set which rez_esc is 99.0
test.loc[test['rez_esc'] == 99.0 , 'rez_esc'] = 5


# In[ ]:


# print out columns with null values
print(f'Train:\n{train.isnull().sum()[train.isnull().sum()>0]}\n')
print(f'Test:\n{test.isnull().sum()[test.isnull().sum()>0]}\n')

# check if columns are the same 
pd.testing.assert_series_equal(train.isnull().any()[train.isnull().any()==True],test.isnull().any()[test.isnull().any()==True])

# put columns with null values in a list
null_list = list(train.isnull().any()[train.isnull().any()==True].index)
print(f'Columns with Null values: {null_list}\n')

# show which columns have null values
print('Definitions:')
for item in null_list:
    print(f'{item}\n{meta.loc[item]}\n')


# # Part 2: Feature engineering

#  **Treating NA values** 
# 
# There are five variables with null values in the column: 
# 1. Monthly Rent Payment
# 2. Number of tablets owned by household
# 3. Years behind in school
# 4. Average years of education for adults (18+)
# 5. Square of the mean years of education of adults (>=18) in the household
# 
# We also note that variable 5 is non-linearly dependent variable 5, with the relationship y = x^2. 
# 
# Due to the high occurence of missing data in the first three variables, we fill them with 0 directly. 

# In[ ]:


# create new train and test dataframes, with missing columns encoded
train_n = train
test_n = test

name = [item + '_null' for item in null_list[:3]]
i = 0
for item in null_list[:3]:
    # train data
    train_n[item].fillna(0,inplace=True)
    # test data
    test_n[item].fillna(0,inplace=True)
    i+=1
    
pd.concat([train_n.head(), test_n.head()], sort=False)


# In[ ]:


# define function for shifting cols to the end train and test 
def shift_cols_last(df, col_index):
    '''
    ====================================
    df: dataframe
    col_ls: a list of column names

    >>> df
        |  a  |  b  |  c  |
    ---------------------------
    0   |  0  |  2  |  1  |
    1   |  0  |  4  |  1  |
    2   |  0  |  6  |  1  |

    >>> shift_cols_last(df,[1])
        |  a   |  c  |  b  |
    ---------------------------    
    0   |  0   |  1  |  2  |
    1   |  0   |  1  |  4  |
    2   |  0   |  1  |  6  |
    ====================================
    '''
    cols = df.columns.tolist()
    i=0
    for item in col_index:
        if i==0:
            new_cols = cols[:item] + cols[item+1:] + [cols[item]] 
        elif i>=1:
            new_cols = new_cols[:item] + new_cols[item+1:] + [cols[item]] 
        i+=1
    return df[new_cols]


# In[ ]:


# show dataframes
pd.concat([train_n.head(), test_n.head()], sort=False)


# In[ ]:


# change datatype for entire dataframe for train and test
train_n = train_n.astype('float64', errors= 'ignore')
test_n = test_n.astype('float64', errors= 'ignore')


# In[ ]:


# check how data is structured + datatypes
print('Train')
print(train_n.info(),'\n')
print('Test')
print(test_n.info())


# In[ ]:


# confirming y =x^2 relationship between meaneduc and SQBmeaned
train_n.plot(kind='scatter', x='meaneduc', y='SQBmeaned')
plt.title('SQBmeaned VS meaneduc')
plt.show()


# In[ ]:


# Fill by median values for meaneduc in both train and test set
tr_median = train_n.meaneduc.median()
print(f'Train Median = {tr_median}')
train_n.meaneduc = train_n.meaneduc.fillna(tr_median)

test_median = test_n.meaneduc.median()
print(f'Test Median = {test_median}')
test_n.meaneduc = test_n.meaneduc.fillna(test_median)


# In[ ]:


# store index of rows with missing SQBmeaned values to list
sqb_null_trls = train_n[train_n['SQBmeaned'].isnull()].index.tolist()
sqb_null_tels = test_n[test_n['SQBmeaned'].isnull()].index.tolist()
print(f'Train SQBmeaned Null rows\n{sqb_null_trls}\nTest SQBmeaned Null rows\n{sqb_null_tels}')


# In[ ]:


# Substitute values in SQBmeaned with the squared values of meanduc for train and test
for item in sqb_null_trls:
    train_n.loc[item,'SQBmeaned']=train_n.loc[item,'meaneduc']**2 
for item in sqb_null_tels:
    test_n.loc[item,'SQBmeaned']=test_n.loc[item,'meaneduc']**2 

# show result
pd.concat([train_n.loc[sqb_null_trls, ['Id','meaneduc', 'SQBmeaned', 'Target']],
           test_n.loc[sqb_null_tels, ['Id','meaneduc', 'SQBmeaned']]], sort=False)


# In[ ]:


# check if there are any remaining nas

print(f'Train data remaining NAs: {len(train_n[train_n.isnull().any(axis=1)])}')
print(f'Test data remaining NAs: {len(test_n[test_n.isnull().any(axis=1)])}')


# In[ ]:


# prepare for label encoding
pared = []
piso = []
techo = []
abasta = []
sanitario = [] 
energcocinar = []
elimbasu = []
epared = []
etecho = []
eviv = []
gender = []
estadocivil = []
parentesco = []
instlevel = []
tipovivi = []
lugar = [] 
area = []
elec = []

# append items into corresponding lists
for item in train_n.columns:
    if item.startswith('pared'):
        # print(item)
        pared.append(item)
    elif item.startswith('piso'):
        # print(item)
        piso.append(item)
    elif item.startswith('techo'):
        # print(item)
        techo.append(item)
    elif item.startswith('abasta'):
        # print(item)
        abasta.append(item)
    elif item.startswith('sanitario'):
        # print(item)
        sanitario.append(item)
    elif item.startswith('energcocinar'):
        # print(item)
        energcocinar.append(item)
    elif item.startswith('elimbasu'):
        # print(item)
        elimbasu.append(item)
    elif item.startswith('epared'):
        # print(item)
        epared.append(item)
    elif item.startswith('etecho'):
        # print(item)
        etecho.append(item)
    elif item.startswith('eviv'):
        # print(item)
        eviv.append(item)
    elif item== 'male' or item =='female':
        # print(item)
        gender.append(item)
    elif item.startswith('estadocivil'):
        # print(item)
        estadocivil.append(item)
    elif item.startswith('parentesco'):
        # print(item)
        parentesco.append(item)
    elif item.startswith('instlevel'):
        # print(item)
        instlevel.append(item)
    elif item.startswith('tipovivi'):
        # print(item)
        tipovivi.append(item)
    elif item.startswith('lugar'):
        # print(item)
        lugar.append(item)
    elif item.startswith('area'):
        # print(item)
        area.append(item)
    elif item == 'noelec'or item =='coopele' or item == 'planpri' or item =='public':
        elec.append(item)


# In[ ]:


# store lists into a list for later for loops
lbl_ls = [pared, piso, techo, abasta, sanitario, energcocinar, elimbasu, epared, etecho, eviv, gender, estadocivil, parentesco, 
          instlevel, tipovivi, lugar, area, elec]
lbl_strls = ['pared', 'piso', 'techo', 'abasta', 'sanitario', 'energcocinar', 'elimbasu', 'epared', 'etecho', 'eviv', 'gender', 'estadocivil', 
             'parentesco', 'instlevel', 'tipovivi', 'lugar', 'area', 'elec']
print(len(lbl_ls), len(lbl_strls))


# In[ ]:


# reverse one-hot encoding for train and test
i = 0
for item in lbl_ls:
    trdf = train_n[item]
    cols = trdf.columns.to_series().values
    if i==0:
        lbl_trdf = pd.DataFrame(np.repeat(cols[None, :], len(trdf), 0)[trdf.astype(bool).values], trdf.index[trdf.any(1)], columns = [lbl_strls[i]])
    else:
        lbl_trdf = pd.concat([lbl_trdf, pd.DataFrame(np.repeat(cols[None, :], len(trdf), 0)[trdf.astype(bool).values], 
                                                 trdf.index[trdf.any(1)], columns = [lbl_strls[i]])], axis = 1)
    i+=1

j = 0
for item in lbl_ls:
    tedf = test_n[item]
    cols = tedf.columns.to_series().values
    if j==0:
        lbl_tedf = pd.DataFrame(np.repeat(cols[None, :], len(tedf), 0)[tedf.astype(bool).values], 
                                tedf.index[tedf.any(1)], columns = [lbl_strls[j]])
    else:
        lbl_tedf = pd.concat([lbl_tedf, pd.DataFrame(np.repeat(cols[None, :], len(tedf), 0)[tedf.astype(bool).values], 
                                                 tedf.index[tedf.any(1)], columns = [lbl_strls[j]])], axis = 1)
    j+=1

#show resulting dataframe
print(lbl_trdf.shape, lbl_tedf.shape)
pd.concat([lbl_trdf.head(), lbl_tedf.head()], sort=False)


# In[ ]:


# label encode data 
for i in range(0, len(lbl_strls)):
    if i%10==0:
        print(i)
    tr_ls = lbl_trdf.iloc[:,i].tolist()
    lbl_trdf.iloc[:,i] = LabelEncoder().fit_transform(tr_ls)
    te_ls = lbl_tedf.iloc[:,i].tolist()
    lbl_tedf.iloc[:,i] = LabelEncoder().fit_transform(te_ls)


# In[ ]:


# show encoded dataframe
pd.concat([lbl_trdf.head(), lbl_tedf.head()], sort=False)


# In[ ]:


# concat lbl_df for train and test datasets
train_n = pd.concat([train_n, lbl_trdf], axis = 1)
test_n = pd.concat([test_n, lbl_tedf], axis = 1)


# In[ ]:


# delete one-hot encoded columns
for item in lbl_ls:
    train_n = train_n.drop(labels=item, axis=1)
    test_n = test_n.drop(labels=item, axis=1)


# In[ ]:


# show new dataframe
pd.concat([train_n.head(), test_n.head()], sort = False)


# In[ ]:


# find the idhogarcolumn for train and test (they share the same column index in both datasets)
col_index = []
for c, value in enumerate(train_n.columns):
    if value == 'Target':
        col_index.append(c)
print(f'Column indexes for Target: {col_index}')


# In[ ]:


# execute function to shift Target column the last in train
train_n = shift_cols_last(train_n, col_index)
train_n.head()


# # Part 3: Preparing data for training

# In[ ]:


# find the idhogar column for train and test (they share the same column indexes in both datasets)
col_index = []
for c, value in enumerate(train_n.columns):
    if value == 'idhogar':
        col_index.append(c)

print(f'Column index for idhogar: {col_index}')


# In[ ]:


# define function for shifting cols in front for train and test 
def shift_cols_front(df, col_index):
    '''
    ====================================
    df: dataframe
    col_ls: a list of column names

    >>> df
        |  a  |  b  |  c  |
    ---------------------------
    0   |  0  |  2  |  1  |
    1   |  0  |  4  |  1  |
    2   |  0  |  6  |  1  |

    >>> shift_cols_front(df,[2])
        |  c   |  b  |  a  |
    ---------------------------    
    0   |  1   |  2  |  0  |
    1   |  1   |  4  |  0  |
    2   |  1   |  6  |  0  |
    ====================================
    '''
    cols = df.columns.tolist()
    i=0
    for item in col_index:
        if i==0:
            new_cols = [cols[0]] + [cols[item]] + cols[1:item] + cols[item+1:]
        elif item < col_index[i-1]:
            new_cols = new_cols[0:i+1] + [new_cols[item+1]] + new_cols[1+i:item+1] + new_cols[item+2:]
        elif item > col_index[i-1]:
            new_cols = new_cols[0:i+1] + [new_cols[item]] + new_cols[1+i:item] + new_cols[item+1:]
        i+=1
    return df[new_cols]


# In[ ]:


# shift idhogar col in front for train and test
train_n = shift_cols_front(train_n, col_index)
test_n = shift_cols_front(test_n, col_index)


# In[ ]:


# shuffle the data
df = train_n.sample(frac=1).reset_index(drop=True)


# In[ ]:


# check correlation of features
tr_corr = df.corr()

sorted_cor = tr_corr['Target'][tr_corr['Target'].isna()==False].sort_values(ascending=False)
print(f'Most Positive Correlations:\n{sorted_cor.head(20)}\n\nMost Negative Correlations:\n{sorted_cor.tail(20)}')


# In[ ]:


# plot distribution of Target in train dataset
style.use('seaborn-white')
sns.distplot(train_n.Target, kde=False, norm_hist = True)
plt.ylabel('Density')
plt.title('Distribution of poverty levels in Train Data')
plt.savefig('Initial Distribution of Target Class.png')
plt.show()


# In[ ]:


# increase samples where target = 1 - 3 
train_class1_3 = train_n[train_n.Target!=4]
train_n.drop(index = train_class1_3.index, inplace=True)

train_class1_3 = pd.concat([train_class1_3, train_class1_3, train_class1_3], sort=False, ignore_index = True)
train_n = pd.concat([train_n, train_class1_3], sort=False, ignore_index = True)
print(len(train_n))


# In[ ]:


# plot distribution of Target in train dataset
style.use('seaborn-white')
sns.distplot(train_n.Target, kde=False, norm_hist = True)
plt.ylabel('Density')
plt.title('Distribution of poverty levels in Train Data')
plt.show()


# In[ ]:


# turn X and y into numpy arrays
X_arr = train_n.iloc[:,2:-1].values
y = train_n.iloc[:,-1].values
te_X = test_n.iloc[:,2:].values


# In[ ]:


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_arr = scaler.fit_transform(X_arr)
te_X = scaler.fit_transform(te_X)


# In[ ]:


# Check shape of data (X)
input_dim = X_arr.shape[1]
print(f'Train features\n{input_dim}')


# In[ ]:


assert not np.any(np.isnan(X_arr))


# # Part 4: Training Data

# In[ ]:


# define function to plot learning curve of estimators
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'f1_macro')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


# set up model based on previously tuned features
model3 = LogisticRegression(penalty='l2', solver = 'newton-cg', multi_class='multinomial', verbose = 1)
scores3 = cross_validate(model3, X_arr, y, cv=10, n_jobs = 4, return_train_score = True, verbose=1, scoring = 'f1_macro')
model3.fit(X_arr,y)


# In[ ]:


# show scores of model 3
scores3


# In[ ]:


# Make prediction
pred3 = model3.predict(te_X)


# In[ ]:


# store prediction from logistic regression in dataframe
pred3s = pd.DataFrame(pred3, columns = ['Target'])
pred3s = pred3s.assign(Id=test_n.Id)
pred3s = pred3s[['Id','Target']]
pred3s.Target = pd.to_numeric(pred3s.Target, downcast='signed')
pred3s.to_csv('log_reg.csv', header = True, index=False)
pred3s.head()


# In[ ]:


# plot training results
model3_graph = plot_learning_curve(model3, 'Logistic_Regression', X_arr, y, cv=10,
                        n_jobs=4, train_sizes = np.linspace(0.5, 1.0, 5))
model3_graph.savefig('Log_Reg_train.png')
model3_graph.show()


# In[ ]:


# store calculated data into a dictionary
model3_dict = {}
for i in range(len(model3.classes_)):
    model3_dict.update({' Class ' + str(i+1): model3.classes_[i],'Intercept '+ str(i+1): model3.intercept_[i], 'Coefficients '+ str(i+1):model3.coef_[i]})


# In[ ]:


# convert dictionary into dataframe
pd.DataFrame.from_dict(model3_dict, orient = 'Index', columns=['Value'])


# In[ ]:


# show correlation matrix
pd.crosstab(y, model3.predict(X_arr), rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:


# show classification report in dataframe
pd.DataFrame.from_dict(classification_report(y, model3.predict(X_arr), output_dict=True), orient='index')


# In[ ]:


model7 = BaggingClassifier()
scores7 = cross_validate(model7, X_arr, y, cv=10, verbose=1, return_train_score = True, n_jobs=4, scoring = 'f1_macro')
model7.fit(X_arr,y)


# In[ ]:


scores7


# In[ ]:


pred7 = model7.predict(te_X)


# In[ ]:


pred7s = pd.DataFrame(pred7, columns = ['Target'])
pred7s = pred7s.assign(Id=test_n.Id)
pred7s = pred7s[['Id','Target']]
pred7s.Target = pd.to_numeric(pred7s.Target, downcast='signed')
pred7s.to_csv('bagging.csv', header = True, index=False)
pred7s.head()


# In[ ]:


model7_graph = plot_learning_curve(model7, 'Bagging', X_arr, y, cv=10,
                        n_jobs=4, train_sizes = np.linspace(0.5, 1.0, 5))
model7_graph.savefig('Bagging_train.png')
model7_graph.show()


# In[ ]:


model10 = GradientBoostingClassifier()
scores10 = cross_validate(model10, X_arr, y, cv=10, verbose=1, return_train_score = True, n_jobs=4, scoring = 'f1_macro')
model10.fit(X_arr,y)


# In[ ]:


scores10


# In[ ]:


pred10 = model10.predict(te_X)


# In[ ]:


pred10s = pd.DataFrame(pred10, columns = ['Target'])
pred10s = pred10s.assign(Id=test_n.Id)
pred10s = pred10s[['Id','Target']]
pred10s.Target = pd.to_numeric(pred10s.Target, downcast='signed')
pred10s.to_csv('grad_boosting.csv', header = True, index=False)
pred10s.head()


# In[ ]:


model10_graph = plot_learning_curve(model10, 'Gradient_Boosting', X_arr, y, cv=10,
                        n_jobs=4, train_sizes = np.linspace(0.5, 1.0, 5))
model10_graph.savefig('GB_train.png')
model10_graph.show()


# In[ ]:


# attach it to a numpy array
final_pred = np.array([])
for i in range(0,len(te_X)):
    final_pred = np.append(final_pred, mode([pred3[i], pred7[i], pred10[i]])[0])


# In[ ]:


pred = pd.DataFrame(final_pred, columns = ['Target'])
pred = pred.assign(Id=test_n.Id)
pred = pred[['Id','Target']]
pred.Target = pd.to_numeric(pred.Target, downcast='signed')
pred.to_csv('prediction.csv', header = True, index=False)
pred.head()


# In[ ]:


# show distribution of predictions
sns.distplot(a=pred.Target, kde=False)
plt.show()

