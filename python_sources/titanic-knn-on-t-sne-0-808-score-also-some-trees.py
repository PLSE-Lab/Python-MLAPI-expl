#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle project and I am new to ML<br>
# I have not taken proper python courses yet, so the code is not beautiful nor it is well documented.<br>
# My score on leaderboard shows 0.82, this happened occasionaly when I run someones notebook and submitted. I have never achieved this score. My highest score is 0.808.<br>
# I did not group passengers by their family name and ticket in order to count group survival rates and create a feature based on that.<br>
# Some people say it increaces score by 0.3, but I just don't feel like it.<br>
# Interesting things to be found here are enhanced model metrics including accuracy vs threshold graphs and a funny t-SNE gif.<br>
# Ok, let's go!

# **Import libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import string
import random as rd

#%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import math
import statistics
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import tree as treepl

seed = 45
rd.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


# **Big and ugly function to show model metrics on cross validation, is called "cv_results"**<br>
# It returns accuracy, ROC curves and so on, will see it in action later

# In[ ]:


def cv_results(X_4_cv, y_4_cv, estimator_4_cv, n_splits_4_cv, seed_4_cv):

    _X_cv = X_4_cv
    _y_cv = y_4_cv
    _estimator = sklearn.base.clone(estimator_4_cv)
    _n_splits_4_cv = n_splits_4_cv
    _shuff_seed = seed_4_cv
    
    _features = 'No names'
    
    if isinstance(_X_cv, pd.DataFrame):
        _features = list(_X_cv.columns)
        _X_cv = _X_cv.to_numpy()
            
    elif isinstance(_X_cv, pd.Series):
        _features = _X_cv.name
        _X_cv = _X_cv.to_numpy()
      
        
    if isinstance(_y_cv, pd.DataFrame) or isinstance(_y_cv, pd.Series):
        _y_cv = _y_cv.to_numpy()   
    
    print('cross validating with ', _n_splits_4_cv, ' folds\n',
          'model: ', _estimator.__class__.__name__, '\n',
          'with parameters:\n', _estimator.get_params(), '\n',
          'dataset size is:\n', _X_cv.shape[0], ' data points\n',
          _X_cv.shape[1], 'features\n',
          'feature names: \n', _features)

    kf = StratifiedKFold(n_splits=_n_splits_4_cv, random_state = _shuff_seed, shuffle = True)
    
    _probs = []
    _mean_c_matr = np.zeros((2,2))
    _mean_b_acc = []
    _mean_b_prec = []
    _mean_b_recall = []
    _mean_b_f1 = []

    
    _mean_acc = []
    _mean_prec = []
    _mean_recall = []
    _mean_f1 = []
    _tprs = []
    _aucs = []
    _mean_fpr = np.linspace(0,1,100)   
    
    _i = 1

    _T_space = np.linspace(0,1,100)
    _acc_scores_curve_b = []
    _acc_scores_curve = []
    
    fig, ax = plt.subplots(1,3, figsize = (20,5))
    
    for _train_ind, _test_ind in kf.split(_X_cv, _y_cv):
        
        _X_cv_train = _X_cv[_train_ind]
        _X_cv_test = _X_cv[_test_ind]
        _y_cv_train = _y_cv[_train_ind]
        _y_cv_test = _y_cv[_test_ind]

        _y_cv_test_weights = compute_sample_weight(class_weight='balanced', y=_y_cv_test)

        _estimator.fit(_X_cv_train, _y_cv_train)
        _y_cv_pred = _estimator.predict(_X_cv_test)
        _y_cv_pred_proba = _estimator.predict_proba(_X_cv_test)
        
        _probs = _probs + list(_y_cv_pred_proba[:,1])
        
        _c_matr = 10*confusion_matrix(_y_cv_test, _y_cv_pred, labels = [1,0], normalize = 'all')
                                   #sample_weight = _y_cv_test_weights)#)
        _c_matr = _c_matr.T
        
        #c_matr = np.round((c_matr)*100)

        #_mean_c_matr = np.round( _mean_c_matr + _c_matr/kf.n_splits , 0)

        _mean_c_matr = np.round( _mean_c_matr + _c_matr , 1)
        
        _mean_acc.append(accuracy_score(_y_cv_test, _y_cv_pred))
        _mean_b_acc.append(accuracy_score(_y_cv_test, _y_cv_pred, 
                                        sample_weight = _y_cv_test_weights))
        
        _mean_prec.append(precision_score(_y_cv_test, _y_cv_pred))    
        _mean_b_prec.append(precision_score(_y_cv_test, _y_cv_pred, 
                                          sample_weight = _y_cv_test_weights, labels = [1,0]))

        _mean_recall.append(recall_score(_y_cv_test, _y_cv_pred))        
        _mean_b_recall.append(recall_score(_y_cv_test, _y_cv_pred, 
                                         sample_weight = _y_cv_test_weights, labels = [1,0]))

        _mean_f1.append(f1_score(_y_cv_test, _y_cv_pred))        
        _mean_b_f1.append(f1_score(_y_cv_test, _y_cv_pred, 
                                 sample_weight = _y_cv_test_weights, labels = [1,0]))
 
        _fpr, _tpr, _t = roc_curve(_y_cv_test, _y_cv_pred_proba[:, 1])

        _tprs.append(np.interp(_mean_fpr, _fpr, _tpr))        

        _roc_auc = auc(_fpr, _tpr)
        _aucs.append(_roc_auc)
        
        ax[0].plot(_fpr, _tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (_i, _roc_auc))
        
        #print('fold ', _i, 'of ', _n_splits_4_cv)
        _i= _i+1
        
        _acc_scores_b = []
        _acc_scores = []
        
        for _T in _T_space:
            
            
            _acc_scores.append(accuracy_score(_y_cv_test, 
                                              [1 if _m > _T else 0 for _m in _y_cv_pred_proba[:, 1]]))
            _acc_scores_b.append(accuracy_score(_y_cv_test, 
                                              [1 if _m > _T else 0 for _m in _y_cv_pred_proba[:, 1]],
                                              sample_weight = _y_cv_test_weights))

        _acc_scores_curve.append(_acc_scores)    
        _acc_scores_curve_b.append(_acc_scores_b)
        
        
    _mean_acc = round(statistics.mean(_mean_acc), 3)
    _mean_prec = round(statistics.mean(_mean_prec), 3)
    _mean_recall = round(statistics.mean(_mean_recall), 3)
    _mean_f1 = round(statistics.mean(_mean_f1), 3)
    
    _mean_b_acc = round(statistics.mean(_mean_b_acc), 3)
    _mean_b_prec = round(statistics.mean(_mean_b_prec), 3)
    _mean_b_recall = round(statistics.mean(_mean_b_recall), 3)
    _mean_b_f1 = round(statistics.mean(_mean_b_f1), 3)
 
    _mean_tpr = np.mean(_tprs, axis=0)
    _mean_auc = auc(_mean_fpr, _mean_tpr)

    
    print()
    print('cv mean confusion matrix (in % of all, not balanced):')
    print(_mean_c_matr)
    print()
    print('cv balanced accuracy: ', _mean_b_acc )
    print('cv balanced precision: ', _mean_b_prec )
    print('cv balanced recall: ', _mean_b_recall )
    print('cv balanced f1: ', _mean_b_f1 )
    print()

    print('cv accuracy: ', _mean_acc )
    print('cv precision: ', _mean_prec )
    print('cv recall: ', _mean_recall )
    print('cv f1: ', _mean_f1 )
    print('cv AUC: ', round(_mean_auc,3) )

      
    ax[0].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')

    ax[0].plot(_mean_fpr, _mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (_mean_auc),lw=2, alpha=1)

    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC')
    ax[0].legend(loc="lower right")
    ax[0].text(0.32,0.7,'More accurate area',fontsize = 12)
    ax[0].text(0.63,0.4,'Less accurate area',fontsize = 12)
    
    ax[1].set_title('Balanced accuracy vs threshold')
    ax[1].plot(_T_space, np.mean(_acc_scores_curve_b, axis=0))
    
    ax[2].grid(False)
    ax[2].set_title('Accuracy vs threshold')
    ax[2].hist(_probs, bins = 20, color='pink')
    ax[2].set_ylabel('predicted probabilities', color = 'pink')
    
    ax2_t = ax[2].twinx()
    ax2_t.plot(_T_space, np.mean(_acc_scores_curve, axis=0))
     
    
    plt.show()
    return None


# **Load data**

# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

print('train_data shape = ', train_data.shape)
print('test_data shape = ', test_data.shape)
train_data.head()


# **Join train and test data**

# In[ ]:


#data = all data, y = train targets. Will divide X on train and test after preproccessing data
y = train_data['Survived']

data = pd.concat( [train_data.drop(columns = ['Survived']), test_data], axis = 0 )

print('all data shape = ', data.shape)
print('y shape = ', y.shape)
print('Fraction of passangers survived: ',
      round(train_data[train_data['Survived'] == 1]['Survived'].count()/
            train_data['Survived'].count(),2))


# **Correlation matrices. Pclass, Age and Fare are correlated as well as Sibsp and Parch**<br>
# The older - the richer

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2 , figsize = (20,7))
sns.heatmap(data.corr(method = 'spearman'), annot = True, ax = ax1)
ax1.set_title('spearman')
sns.heatmap(data.corr(method = 'pearson'), annot = True, ax = ax2)
ax2.set_title('pearson')
plt.show()


# **Survival rate by class, sex and embarkation**<br>

# In[ ]:


train_length = train_data.shape[0]
#dm = data modified
#concat train data with y in order to perform some plotting
dm_train = pd.concat( [data.iloc[:train_length], y], axis = 1 )
dm_test = data.iloc[train_length:]


# In[ ]:


round(dm_train.groupby(['Pclass'])['Survived'].mean(),2)


# In[ ]:


round(dm_train.groupby(['Pclass', 'Sex'])['Survived'].mean(),2)


# In[ ]:


round(dm_train.groupby(['Embarked', 'Pclass'])['Survived'].mean(),2)


# ## Fill NaN

# **Missing values**

# In[ ]:


data.isna().sum()


# **First create feature AgeEstRec. 0 if age was known, and 1 if age was NaN or estimated (=xx.5 from data description)**

# In[ ]:


def age_est_rec(age):
    if (age - 0.5).is_integer():
        return 1
    elif pd.isna(age):
        return 1
    else:
        return 0
    
data['AgeEstRec'] = data['Age'].map(age_est_rec)


# In[ ]:


sns.set_style("whitegrid")
sns.pointplot(x=data['AgeEstRec'].iloc[:train_data.shape[0]], y = y, palette="muted")
plt.show()


# Passengers with unknown age are less likely to survive

# **Need to extract title. It is needed to reconstruct missing Age**

# In[ ]:


data['Title'] = data['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]


# In[ ]:


pd.crosstab(data['Title'], data['Sex'])


# One Doctor was female, just curious who she was

# In[ ]:


data[(data['Title'] == 'Dr') & (data['Sex'] == 'female')]


# **join small title groups with big ones**

# In[ ]:


data.loc[(data['Title'] == 'Dr') & (data['Sex'] == 'female'), ['Title']] = 'Mrs'

data['Title'] = data['Title'].replace(['Capt', 'Col', 'Major', 'Rev', 'Don', 'Jonkheer','Sir', 'Dr'], 'Mr')
data['Title'] = data['Title'].replace(['Mlle'], 'Miss')
data['Title'] = data['Title'].replace(['Mme',  'Ms', 'Dona',  'Lady',  'the Countess'], 'Mrs')

pd.crosstab(data['Title'], data['Sex'])


# In[ ]:


sns.pointplot(x=data['Title'].iloc[:train_data.shape[0]], y = y, palette="muted")
plt.show()


# Misters obviously suck at surviving the Titanic.

# **Fill one empty fare cell with average fare of that class**

# In[ ]:


data[data.Fare.isna()]


# In[ ]:


data.index = range(data.shape[0])
data.at[1043, 'Fare'] = data[data['Pclass']==3].median()['Fare']


# Filled two missing Embarked values with info from EncyclopediaTitanica. Yes, it is a cheat.

# In[ ]:


data['Embarked'].fillna('S', inplace = True)
sns.countplot(x = 'Embarked', data = data, hue = 'Pclass')
plt.show()


# # Ticket

# **Extract ticket numbers. Somehow descigion trees find this feature useful**

# In[ ]:


def ticket_split(s):
    
    s = s.strip()
    split = s.split()
    
    if len(split) == 1:
        tnum = split[0]
        #there are 4 strange ticket numbers
        #that state 'LINE'. Assign them to 0
        if tnum == 'LINE':
            tnum = 0
        tstr = 'NA'

            
    elif len(split) == 2:
        tstr = split[0]
        tnum = split[1]
    else:
        tstr = split[0] + split[1]
        tnum = split[2]
        
    
    tnum = int(tnum)

    return tstr, tnum


# In[ ]:


data['TicketStr'], data['TicketNum'] = zip(*data['Ticket'].map(ticket_split))


# In[ ]:


plt.figure(figsize = (8,3))
sns.distplot(data["TicketNum"])
plt.show()


# Let's explore those greater than 2 500 000. They all embarked in Southampton and were 3 class.

# In[ ]:


data[data['TicketNum']>2500000].head()


# Ticket numbers have huge variance. Let's logarithm them for the sake of visualisation.

# In[ ]:


data["LogTicketNum"] = data["TicketNum"].map(lambda i: np.log(i) if i > 0 else 0)
plt.figure(figsize = (5,3))
sns.distplot(data["LogTicketNum"])
plt.show()


# In[ ]:


print('correlation of LogTicketNum feature with others:')
round(data.corr().iloc[:,-1],2).sort_values(ascending = False)


# It is slightly correlated with Pclass meaning 1 class passengers had smaller ticket numbers then the others, as well as more expencive tickets had smaller numbers. Well, maybe ticket numbers do contain some information.

# **Some ticket numbers are duplicated. It means passengers with same ticket numbers form a family or a group.**<br>
# I create a feature named TicketFreq based on that<br>
# There are passengers with same Family names and close ticket numbers, but counting their family size based on this information did not improve model performance.<br>

# In[ ]:


print('tickets count: ', (data['TicketNum']).count())
print('unique ticket numbers: ', np.unique(data['TicketNum']).shape[0])

#create ticket frequency coloumn
data['TicketFreq'] = data.groupby(['TicketNum', 'TicketStr', 'Fare'])['TicketNum'].transform('count')
print('Amount of tickets with frequency > 1:', data[data.TicketFreq>1].TicketNum.count())


# **Create feature "Family size" by adding SibSp and Parch**

# In[ ]:


data['FamSize'] = data['SibSp'] + data['Parch']


# In[ ]:


fig, ax = plt.subplots(2,2, figsize = (15,8))

plt.subplot(2,2,1)
sns.countplot(data["FamSize"], color="g", label="Famsize").legend(loc="best")
plt.subplot(2,2,2)
sns.countplot(data["SibSp"], color="lightblue",  label="SibSp").legend(loc="best")
plt.subplot(2,2,4)
sns.countplot(data["Parch"], color="pink",  label="Parch").legend(loc="best")
plt.subplot(2,2,3)
sns.countplot(data["TicketFreq"], color="y",  label="TicketFreq").legend(loc="best")
plt.show()


# In[ ]:


train_length = train_data.shape[0]
fig, ax = plt.subplots(2,2, figsize = (15,8))
plt.subplot(2,2,4)
sns.pointplot(x = data[:train_length]['Parch'], y = y, palette = 'muted')
plt.subplot(2,2,2)
sns.pointplot(x = data[:train_length]['SibSp'], y = y, palette = 'muted')
plt.subplot(2,2,3)
sns.pointplot(x = data[:train_length]['FamSize'], y = y, palette = 'muted')
plt.subplot(2,2,1)
sns.pointplot(x = data[:train_length]['TicketFreq'], y = y, palette = 'muted')
plt.show()


# **Group together passengers with 4 or more family members in order to cut off those long tails.**

# In[ ]:


def family_size(fam):
    if fam < 1:
        return 0
    elif fam < 2:
        return 1
    elif fam < 3:
        return 2
    elif fam < 4:
        return 3
    else:
        return 4   
    
    
data['ModFamSize'] = data['FamSize'].map(family_size)
data['ModTicketFreq'] = data['TicketFreq'] - 1
data['ModTicketFreq'] = data['ModTicketFreq'].map(family_size)
data['ModSibSp'] = data['SibSp'].map(family_size)
data['ModParch'] = data['Parch'].map(family_size)


# Now it looks nice and neat

# In[ ]:


fig, ax = plt.subplots(2,2, figsize = (15,8))
plt.subplot(2,2,1)
sns.countplot(data["ModFamSize"], color="g", label="Famsize").legend(loc="best")
plt.subplot(2,2,2)
sns.countplot(data["ModSibSp"], color="lightblue",  label="SibSp").legend(loc="best")
plt.subplot(2,2,4)
sns.countplot(data["ModParch"], color="pink",  label="Parch").legend(loc="best")
plt.subplot(2,2,3)
sns.countplot(data["ModTicketFreq"], color="y",  label="TicketFreq").legend(loc="best")
plt.show()


# In[ ]:


train_length = train_data.shape[0]
fig, ax = plt.subplots(2,2, figsize = (15,8))
plt.subplot(2,2,4)
sns.pointplot(x = data[:train_length]['ModParch'], y = y, palette = 'muted')
plt.subplot(2,2,2)
sns.pointplot(x = data[:train_length]['ModSibSp'], y = y, palette = 'muted')
plt.subplot(2,2,3)
sns.pointplot(x = data[:train_length]['ModFamSize'], y = y, palette = 'muted')
plt.subplot(2,2,1)
sns.pointplot(x = data[:train_length]['ModTicketFreq'], y = y, palette = 'muted')
plt.show()


# # Floor
# There were several floors on the Titanic, each with its own letter.

# In[ ]:


data['CabinNA'] = np.where(data['Cabin'].isna(), 1, 0)
#fill all missing cabin with M
data.Cabin.fillna('NA', inplace = True)
#extract cabin letters
data['Floor'] = data['Cabin'].str.extract('([A-Za-z]+)')
print('Floor letters:')
print(pd.unique(data['Floor']))


# In[ ]:


data.groupby('Floor').count()['PassengerId']


# In[ ]:


#join T with A because they vere close, and there are too few T passengers
T_index = data[data.Floor == 'T'].index
data.at[T_index, 'Floor'] = 'A'


# In[ ]:


plt.figure(figsize = (20,5))
sns.countplot(x='Floor', hue='Pclass', data = data, palette="muted")
plt.show()


# In[ ]:


plt.figure(figsize = (20,5))
sns.countplot(x='Floor', hue='Pclass', data = data[data['Floor']!='NA'], palette="muted")
plt.show()


# Floor is known for mostly 1 class passengers. But if it is known for 2 and 3 class it can be valuable information.

# # Age NaN Filled here
# Age is correlated with Title (Miss-Mrs, Mr-Master), Pclass and Age (the older the wealthier), and SibSp and Parch.<br>
# Thus I group by those features and count median. For groups with less than 5 members I do it again, but group only by Title and Pclass.

# In[ ]:


print('we have ', data['Age'].isna().sum(), 'NaN entries in Age')
grouped = data.groupby(['Title', 'Pclass', 'ModSibSp', 
                            'ModParch'])['Age']
for name, group in grouped:
    if (group.shape[0] > 5) & (group.isna().sum()>0):
        data['Age'].iloc[list(group.index)] = group.fillna(group.median())
print('grouped by Title, Pclass, ModSibSp, ModParch, replaced NaN with group median for groups > 5')
print('we have ', data['Age'].isna().sum(), 'NaN entries in Age now')


# In[ ]:


data['Age'] = data.groupby(['Title', 'Pclass',])['Age'].apply(lambda x: x.fillna(x.median()))
print('grouped by Title, Pclass replaced NaN with group median')
print('we have ', data['Age'].isna().sum(), 'NaN entries in Age now')


# In[ ]:


g = sns.FacetGrid(data, col='Title', margin_titles=True, sharey = False, sharex = False)
g.map(plt.hist, "Age", color="steelblue")
plt.show()


# There are Misters younger than 19 and even 15. Let's have a look at them.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize = (12,3))
ax[0].hist(data[ (data['Title'] == 'Mr') & (data['Age'] < 20) & (data['AgeEstRec']==0) ]['Age'])
ax[1].hist(data[ (data['Title'] == 'Mr') & (data['Age'] < 20) & (data['AgeEstRec']==1) ]['Age'])
ax[0].set_title('Mr, Age < 19, age not reconstructed')
ax[1].set_title('Mr, Age < 19, age reconstructed')
plt.show()


# # Fare

# Divide Fare by TicketFrequency in order to get fare per passenger (create new ModFare feature)<br>
# Then create LogFare feature by logarythming ModFare

# In[ ]:


fig, ax = plt.subplots(1,3, figsize = (15,3))
plt.subplot(1,3,1)
sns.distplot(data["Fare"])
plt.subplot(1,3,2)
data['ModFare'] = data['Fare']/data['TicketFreq']
sns.distplot(data["ModFare"])
plt.subplot(1,3,3)
data["LogFare"] = data["ModFare"].map(lambda i: np.log(i) if i > 0 else 0)
sns.distplot(data["LogFare"])
plt.show()


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15,3))

ax1.scatter(data[:train_length]['LogFare'], data[:train_length]['Pclass'])
ax1.set_title('Train data')
ax1.set_xlabel('Log Fare')
ax1.set_ylabel('Pclass')

ax2.scatter(data[train_length:]['LogFare'], data[train_length:]['Pclass'])
ax2.set_title('Test data')
ax2.set_xlabel('Log Fare')
ax2.set_ylabel('Pclass')
plt.show()


# **There are outliers with Fare = 0 and 3 class passengers with high fare. Let's replace it with median**

# In[ ]:


data.loc[ ((data['LogFare']>2.5) | (data['LogFare']<1)) & (data['Pclass']==3), ['ModFare'] ] = data[data['Pclass']==3]['ModFare'].median()

data.loc[ (data['LogFare']<1) & (data['Pclass']==2), ['ModFare'] ] = data[data['Pclass']==2]['ModFare'].median()

data.loc[ (data['LogFare']<2) & (data['Pclass']==1), ['ModFare'] ] = data[data['Pclass']==1]['ModFare'].median()

data["LogFare"] = data["ModFare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15,3))

ax1.scatter(data[:train_length]['LogFare'], data[:train_length]['Pclass'])
ax1.set_title('Train data')
ax1.set_xlabel('Log Fare')
ax1.set_ylabel('Pclass')

ax2.scatter(data[train_length:]['LogFare'], data[train_length:]['Pclass'])
ax2.set_title('Test data')
ax2.set_xlabel('Log Fare')
ax2.set_ylabel('Pclass')
plt.show()


# **Now let's explore fare and age survival distribution by cutting into quantiles**

# In[ ]:


plt.figure(figsize = (5,3))
sns.distplot(data["Age"])
plt.show()


# In[ ]:


data['ModFareCut'] = pd.qcut(data['ModFare'], 7)
data['AgeCut'] = pd.qcut(data['Age'], 9)


# In[ ]:


train_length = train_data.shape[0]
#dm = data modified
#concat train data with y in order to perform some plotting
dm_train = pd.concat( [data.iloc[:train_length], y], axis = 1 )
dm_test = data.iloc[train_length:]


# In[ ]:


fig, ax = plt.subplots(4,1, figsize = (10,14))
fig.subplots_adjust(hspace = 0.4)
plt.subplot(4,1,1)
sns.pointplot(x='ModFareCut', y = 'Survived', data = dm_train, palette="muted").set_title('Log fare per passenger')
plt.subplot(4,1,2)
sns.pointplot(x='AgeCut', y = 'Survived', data = dm_train, palette="muted").set_title('all Age')
plt.subplot(4,1,3)
sns.pointplot(x='AgeCut', y = 'Survived',
              data = dm_train[dm_train['AgeEstRec'] == 1], palette="muted").set_title('Age reconstructed')
plt.subplot(4,1,4)
sns.pointplot(x='AgeCut', y = 'Survived',
              data = dm_train[dm_train['AgeEstRec'] == 0], palette="muted").set_title('Age known')
plt.show()


# It seems like all misters with missing Age got their reconstructed Age equal to 25

# In[ ]:


fig, ax = plt.subplots(2,1, figsize = (10,7))
fig.subplots_adjust(hspace = 0.4)
plt.subplot(2,1,1)
sns.pointplot(x='AgeCut', y = 'Survived', 
              data = dm_train[ (dm_train['AgeEstRec']==0) &
                               (dm_train['Title']!='Mr')], palette="muted").set_title('Mr, Age known')
plt.subplot(2,1,2)
sns.countplot(x='AgeCut', hue = 'Survived', 
              data = dm_train[ (dm_train['AgeEstRec']==1) &
                               (dm_train['Title']=='Mr')], palette="muted").set_title('Mr, Age reconstructed')

plt.show()


# **Make an overall plot for all features**

# In[ ]:


features = ['Pclass', 'Sex', 'Embarked', 'AgeEstRec', 
            'Title', 'CabinNA', 'ModFamSize', 'ModTicketFreq', 'ModFareCut', 'AgeCut', 'Floor']
fig, ax = plt.subplots(4,3, figsize = (20,15))
fig.subplots_adjust(hspace = 0.3,wspace = 0.3)
fig.suptitle('train data survived distribution', fontsize=16)

for i, feature in enumerate(features):

    i += 1
    plt.subplot(4,3,i)
    sns.countplot(x=feature, hue='Survived', data = dm_train, palette="muted")
    plt.legend(loc='upper right')
    
plt.show()


# In[ ]:


features = ['Pclass', 'Sex', 'Embarked', 'AgeEstRec', 
            'Title', 'CabinNA', 'ModFamSize', 'ModTicketFreq', 'ModFareCut', 'AgeCut', 'Floor']
fig, ax = plt.subplots(4,3, figsize = (20,15))
fig.subplots_adjust(hspace = 0.3,wspace = 0.3)
fig.suptitle('train data survived percentage with bootstrapped confidence interval', fontsize=16)

for i, feature in enumerate(features):

    i += 1
    plt.subplot(4,3,i)
    sns.pointplot(x=feature, y = 'Survived', data = dm_train, palette="muted")
    
plt.show()


# **Encode Title and Floor with numbers**

# In[ ]:


#encode Title by survival rate
data['TitleDigit'] = data['Title'].map({'Mr': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4})


# In[ ]:


data['FloorDigit'] = data['Floor'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7, 'NA':8})


# In[ ]:


train_length = train_data.shape[0]
#dm = data modified
#concat train data with y in order to perform some plotting
dm_train = pd.concat( [data.iloc[:train_length], y], axis = 1 )
dm_test = data.iloc[train_length:]


# In[ ]:


fig, ax = plt.subplots(1,2,figsize = (13,4))
plt.subplot(1,2,1)
sns.pointplot('TitleDigit', 'Survived', data = dm_train, palette = 'muted')
plt.subplot(1,2,2)
sns.pointplot('FloorDigit', 'Survived', data = dm_train, palette = 'muted')
plt.show()


# **One-hot encoding for categorial features**

# In[ ]:


features_to_onehot = ['Pclass', 'Embarked', 'Title', 'Floor']

enc = OneHotEncoder(sparse = False)

for feature in features_to_onehot:
    
    encoded = pd.DataFrame( enc.fit_transform(data[feature].to_numpy().reshape(-1,1)) )
    encoded.columns = feature + ' ' + enc.get_feature_names()
    
    data = pd.concat([data, encoded], axis = 1)


# **drop features not to be used**

# In[ ]:


feat_to_drop = ['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title',
       'TicketStr','Floor','ModFareCut', 'AgeCut']
data.drop(columns = feat_to_drop, inplace=True)
print('we have ', len(data.columns), ' features now')


# In[ ]:


data.columns


# **Cut train data into train and test, X_submit is data with unknown labels that we have to find out**

# In[ ]:


train_length = train_data.shape[0]
X = data.iloc[:train_length]
X_submit = data.iloc[train_length:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = seed)


# # Done with features

# **Let's see if passengers can be separated by Age and Fare**

# In[ ]:


cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","blue"])

fig, ax = plt.subplots(1,2, figsize = (20,7))
fig.suptitle('Survival vs LogFare/Age')
scatter_fem = ax[0].scatter(dm_train[dm_train['Sex'] == 'female']['Age'], 
                          dm_train[dm_train['Sex'] == 'female']['LogFare'], 
                          c = y[dm_train['Sex'] == 'female'], cmap = cmap1)
legend1 = ax[0].legend(*scatter_fem.legend_elements(), loc="upper left", title="Survived women")

ax[0].set_ylabel('LogFare')
ax[0].set_xlabel('Age')
scatter_men = ax[1].scatter(dm_train[dm_train['Sex'] == 'male']['Age'], 
                          dm_train[dm_train['Sex'] == 'male']['LogFare'], 
                          c = y[dm_train['Sex'] == 'male'], cmap = cmap1)
legend2 = ax[1].legend(*scatter_men.legend_elements(), loc="upper left", title="Survived men")
ax[1].set_ylabel('LogFare')
ax[1].set_xlabel('Age')

plt.show()


# **We need something better to separate them. t-SNE!**<br>
# I select some features for tsne and look at their correlation first.<br>
# Some of them are heavily correlatad, but removing correlated features did not improve performance, so I kept them.

# In[ ]:


tsne_features = ['Pclass', 'Age',  'AgeEstRec',
       'LogTicketNum', 'TicketFreq',
       'SibSp', 'Parch', 'ModFare', 'FloorDigit',
       'Embarked x0_C', 'Embarked x0_Q', 'Embarked x0_S', 
       'Title x0_Master','Title x0_Miss', 'Title x0_Mr', 'Title x0_Mrs']

fig, (ax1,ax2) = plt.subplots(1,2 , figsize = (20,7))
sns.heatmap(data[tsne_features].corr(method = 'spearman'), annot = False, ax = ax1)
ax1.set_title('spearman')
sns.heatmap(data[tsne_features].corr(method = 'pearson'), annot = False, ax = ax2)
ax2.set_title('pearson')
plt.show()


# In[ ]:


data_4tsne = data[tsne_features]

#now scale

scaler = StandardScaler()
scaler.fit(data_4tsne)
data_4tsne_scaled_np = scaler.transform(data_4tsne)
data_4tsne_scaled = pd.DataFrame(data = data_4tsne_scaled_np, 
                                 index = data_4tsne.index, columns = data_4tsne.columns) 

#tSNE


tsne2 = manifold.TSNE( **{'n_components': 2, 'perplexity': 50,
                          'learning_rate': 230, 'init': 'pca',
                          'random_state': seed, 'n_jobs': -1} )


data_2d_tsne = tsne2.fit_transform(data_4tsne_scaled)

train_length = train_data.shape[0]
X_train_2d_tsne = data_2d_tsne[:train_length,:]
X_test_2d_tsne = data_2d_tsne[train_length:,:]


# In[ ]:


cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","blue"])

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1, 1, 1)

tr = ax.scatter(X_train_2d_tsne[:, 0], X_train_2d_tsne[:, 1], c = y, cmap = cmap1)
te = ax.scatter(X_test_2d_tsne[:, 0], X_test_2d_tsne[:, 1], c = 'g', alpha = 0.4)


legend_elements = [Line2D([0], [0], marker='o', color='w', label='Dead', markerfacecolor='r'),
                   Line2D([0], [0], marker='o', color='w', label='Survived', markerfacecolor='b'),
                   Line2D([0], [0], marker='o', color='w', label='test data', markerfacecolor='g')]

ax.legend(handles=legend_elements)

plt.show()


# **Below is a link to a gif for 2D t-SNE with different perplexity values. It was recorded for a different feature set, so it is different from the above picture**<br>
# The gif is 12Mb, so kaggle won't allow me to include it in the kernel directly.
# https://ibb.co/2jXkyyw

# # kNN on t-SNE
# **Search parameters grid for best number of neighbours on cross validation**<br>
# I search only for odd NN because we have a two-class classification problem

# In[ ]:


knn_data = data_2d_tsne

knn_train_data = knn_data[:train_length]
knn_test_data = knn_data[train_length:]

KNN = KNeighborsClassifier()

kNN_params = {'n_neighbors' : range(1,30,2),
              'weights' : ['uniform'],
              'p' : [2],
              'n_jobs' : [-1]}

grid_cv = GridSearchCV(KNN, kNN_params, scoring = 'accuracy', cv = 10, verbose = 1, n_jobs = -1)

grid_cv.fit(knn_train_data, y)

print(grid_cv.best_params_, grid_cv.best_score_)


# **11 NN shows best performance on cross validation, but 19 NN gets best public score**<br>
# Now cv_results function defined at the very top of this notebook takes action

# In[ ]:


knn_best_pars = {'n_jobs': -1, 'n_neighbors': 19, 'p': 2, 'weights': 'uniform'}

#KNN1 = KNeighborsClassifier(**grid_cv.best_params_)

KNN1 = KNeighborsClassifier(**knn_best_pars)

cv_results(knn_train_data, y, KNN1, 10, 15)


# What is Accuracy vs threshold?<br>
# I take estimator predicted probabilities (predict_proba) and make predictions in the following way:<br>
# I predict 1 if estimator predicted probability is greater than the theshold an 0 if less<br>
# This way I can maximize accuracy by choosing a proper threshold<br>
# In case of kNN I choose 0.6 threshold<br>
# Pink bars are counts of estimator predictions with certain probabilities, in other words - predicted probabilities distribution

# In[ ]:


KNN1.fit(knn_train_data, y)
y_knn_sub_pr = KNN1.predict_proba(knn_test_data)
y_knn_sub = [1 if m > 0.6 else 0  for m in y_knn_sub_pr[:,1] ]

test_data2 = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': y_knn_sub})
output.to_csv('Titanic_submission_KNN.csv', index=False)


# **kNN on t-SNE gets 0.80861 public score**

# # Simple tree
# I selected features based on their importance (see below)

# In[ ]:


tree_features = ['Title x0_Mr', 'Pclass x0_3', 'TicketFreq', 'CabinNA', 'LogTicketNum','LogFare', 'Floor x0_E']

X_tree = X[tree_features]
X_submit_tree = X_submit[tree_features]

X_train_tree = X_train[tree_features]
X_test_tree = X_test[tree_features] 


# **Some correlation matrices again.**<br>
# I don't know if those make any sense

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2 , figsize = (20,7))
sns.heatmap(data[tree_features].corr(method = 'spearman'), annot = True, ax = ax1)
ax1.set_title('spearman')
sns.heatmap(data[tree_features].corr(method = 'pearson'), annot = True, ax = ax2)
ax2.set_title('pearson')
plt.show()


# **I do a cv parameters search based on min_impurity_decrease**<br>
# A curious fact:<br>
# Though it is not documented, if max_leaf_nodes is not set, 
# a DepthFirstTreeBuilder will be used to fit the underlying
# tree object; if it is, then a BestFirstTreeBuilder will be used;
# this difference results in trees of different depths being generated.<br>
# https://stackoverflow.com/questions/56132387/sklearn-tree-

# In[ ]:


tree = DecisionTreeClassifier(random_state = seed)
    
parameters = {'criterion': ['entropy', 'gini'],
              'splitter' : ['best'],
              'min_samples_leaf': list(range(3,9,2)),
              'min_impurity_decrease' : list(np.linspace(0.00001,0.01,100)),             
              'max_features': [None],
              'class_weight' : ['balanced'],
              'random_state' : [seed]}

grid_cv = GridSearchCV(tree, parameters, scoring = 'accuracy', cv = 5, verbose = 1, n_jobs = -1)

grid_cv.fit(X_train_tree, y_train)
print('best accuracy: ', grid_cv.best_score_)                          
print(grid_cv.best_params_)

print('\n depth: ', grid_cv.best_estimator_.get_depth())
best_tree_params = grid_cv.best_params_


# **Best results on cv are shown by 'min_impurity_decrease' = 0.00223<br>**
# But in order to make the model less overfitted I have chosen a higher value to make my tree smaller<br>
# This way I get a tree with depth 5 instead of 7<br>
# The value is chosen almost by guessing

# In[ ]:


best_tree_params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_features': None,
                    'min_impurity_decrease': 0.004853636363636364, 'min_samples_leaf': 3, 
                    'random_state': 45, 'splitter': 'best'}


# In[ ]:


best_tree_impurity = best_tree_params['min_impurity_decrease']
best_tree_min_samples_leaf = best_tree_params['min_samples_leaf']
best_tree_crit = best_tree_params['criterion']

best_tree = DecisionTreeClassifier(**best_tree_params)
best_tree.fit(X_train_tree, y_train)
print('depth ',best_tree.get_depth())


# **Here I obtain feature importances for my tree**

# In[ ]:


tree_importances = pd.Series(best_tree.feature_importances_,X_train_tree.columns).sort_values(ascending=False)
print(tree_importances.loc[tree_importances > 0])
tree_importances.loc[tree_importances > 0].index


# In[ ]:


plt.figure(figsize=(6,6))
pd.Series(best_tree.feature_importances_,X_train_tree.columns).sort_values(ascending=True).plot.barh(width=0.8)
plt.show()


# **The tree itself**

# In[ ]:


plt.figure(figsize = (50,50))
plt.style.use('default')
a = treepl.plot_tree(best_tree, filled = True, rounded = True,
                     feature_names = tree_features, class_names = ['Dead', 'Survived'])
sns.set_style("whitegrid")


# **Here are some graphs of model performance depending on parameter values**

# In[ ]:


imp_list = np.linspace(0.00001,0.01,100)

imp_train_acc_list = []
imp_test_acc_list = []

depth_list = []

for imp in imp_list:

    parameters = {'criterion': best_tree_crit,
                  'splitter' : 'best',
                  'min_impurity_decrease' : imp,
                  #'max_depth': list(range(1,12)),
                  #'min_samples_split': list(range(2,20,2)),
                  'min_samples_leaf': best_tree_min_samples_leaf,
                  'max_features': None,
                  #'max_leaf_nodes': (list(range(2,60,2)) + [None]),
                  'class_weight' : 'balanced',
                  'random_state' : seed}

    imp_tree = DecisionTreeClassifier(**parameters)

    imp_tree.fit(X_train_tree,y_train)
    
    acc_sc_train = accuracy_score(y_train, imp_tree.predict(X_train_tree))
    acc_sc_test = accuracy_score(y_test, imp_tree.predict(X_test_tree))

    imp_train_acc_list.append(acc_sc_train)
    imp_test_acc_list.append(acc_sc_test)
    
    depth_list.append(imp_tree.get_depth())
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(imp_list,imp_train_acc_list, label = 'accuracy train')
ax.plot(imp_list,imp_test_acc_list, label = 'accuracy test')

ax2 = ax.twinx()
ax2.plot(imp_list,depth_list, color = 'pink', label = 'tree depth')

ax.legend()
ax.set_title('accuracy vs min_impurity_decrease')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
plt.show()


# It can seem that I should have chosen min_impurity_decrease based on best performance on test, somewhere close to 0.0028, where I get almost 0.88 accuracy on test, but it is overfitting for test data and will show poor results on public.

# In[ ]:


leaf_list = range(1,10)

leaf_train_acc_list = []
leaf_test_acc_list = []

depth_list = []

for leaf in leaf_list:

    parameters = {'criterion': best_tree_crit,
                  'splitter' : 'best',
                  'min_impurity_decrease' : best_tree_impurity,
                  #'max_depth': list(range(1,12)),
                  #'min_samples_split': list(range(2,20,2)),
                  'min_samples_leaf': leaf,
                  'max_features': None,
                  #'max_leaf_nodes': (list(range(2,60,2)) + [None]),
                  'class_weight' : 'balanced',
                  'random_state' : seed}


    
    leaf_tree = DecisionTreeClassifier(**parameters)
    
    leaf_tree.fit(X_train_tree,y_train)

    
    leaf_train_acc_list.append(accuracy_score(y_train, leaf_tree.predict(X_train_tree)))
    
    leaf_test_acc_list.append(accuracy_score(y_test, leaf_tree.predict(X_test_tree)))
    
    #depth_list.append(imp_tree.get_depth())
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('accuracy vs min_samples_leaf')

plt.plot(leaf_list,leaf_train_acc_list, label = 'train')
plt.plot(leaf_list,leaf_test_acc_list, label = 'test')
plt.legend()
plt.show()


# **my metrics function again**<br>
# we can see that tree is more certain than kNN, it has less probabilities in range (0.2 - 0.7)

# In[ ]:


cv_results(X_train_tree, y_train, best_tree, 10, 15)


# **Here are results for the chosen tree on the test set**

# In[ ]:


best_tree.fit(X_train_tree, y_train)

y_tree_test_pred_prob = best_tree.predict_proba(X_test_tree)[:,1]

tree_acc_scores = []

for thr in np.linspace(0,1,100):
    y_tree_test_pred = [1 if m > thr else 0 for m in y_tree_test_pred_prob]
    tree_acc_score = accuracy_score(y_test, y_tree_test_pred)
    tree_acc_scores.append(tree_acc_score)

print('accuracy with 0.5 threshold', 
      accuracy_score(y_test, [1 if m > 0.5 else 0 for m in best_tree.predict_proba(X_test_tree)[:,1]]))

fig = plt.figure(figsize = (4,3))
ax = fig.add_subplot(1,1,1)
ax.set_title('accuracy vs threshold on test')
plt.plot(np.linspace(0,1,100), tree_acc_scores)
plt.show()


# **best results were shown with threshold = 0.7**

# In[ ]:


best_tree.fit(X_tree, y)

y_tree_sub = best_tree.predict_proba(X_submit_tree)
y_tree_sub_tr = [1 if m > 0.7 else 0 for m in y_tree_sub[:, 1]]

test_data2 = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': y_tree_sub_tr})
output.to_csv('Titanic_submission_tree.csv', index=False)


# **the tree gets 0.80382 public score**

# # Forest here
# **features were selected based on estimator feature importance**

# In[ ]:


forest_features = [
    
        'Age',

        'ModSibSp',
        'ModTicketFreq',

        'LogFare',

        'LogTicketNum',

        'AgeEstRec',

        'TitleDigit',

        'FloorDigit', 

        'Pclass']


#split data to train and test
X_forest = X[forest_features]
X_submit_forest = X_submit[forest_features]

X_train_forest = X_train[forest_features]
X_test_forest = X_test[forest_features] 


# **Feature correlation matrices**<br>
# Again, lots of correlated features. Deal with it.

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2 , figsize = (20,7))
sns.heatmap(data[forest_features].corr(method = 'spearman'), annot = True, ax = ax1)
ax1.set_title('spearman')
sns.heatmap(data[forest_features].corr(method = 'pearson'), annot = True, ax = ax2)
ax2.set_title('pearson')
plt.show()


# **Forest parameters search**<br>
# As with tree, my stopping criteria for the forest is min_impurity_decrease. It seems more precise than max_depth.<br>
# I have chosen max_features = 6 out of 9 to my forest more random and less overfitted<br>
# Also, I choose min_samples_leaf = 1, because in the forest case I don't care if a single tree will turn out to be overfitted. The whole forest will be ok.<br>
# The search takes a lot of time due to 2700 trees in the forest. Go smoke a cigarette.

# In[ ]:


imp_list = np.linspace(0.00001,0.01,30)

imp_train_acc_list = []
imp_train_oob_list = []
imp_test_acc_list = []

for imp in imp_list:
    
    #print('min_impurity_decrease: ', round(imp,4))
    parameters = {'random_state': 45,
                  'n_estimators': 2700,
                  'min_samples_leaf': 1, 
                  'min_impurity_decrease': imp,
                  'max_samples': None,
                  'max_features': 6,
                  'criterion': 'entropy',
                  'class_weight': 'balanced',
                  'oob_score' : True,
                  'n_jobs' : -1
                 }
    
    imp_forest = RandomForestClassifier(**parameters)
    
    imp_forest.fit(X_train_forest,y_train)

    f_depth = 0
    for f_tree in imp_forest.estimators_:
        f_depth = f_depth + f_tree.get_depth()
        
    #print('average forest depth: ', round(f_depth/len(imp_forest.estimators_),1))
    
    
    imp_train_acc_list.append(accuracy_score(y_train, imp_forest.predict(X_train_forest)))
    
    imp_train_oob_list.append(imp_forest.oob_score_)
                              
    imp_test_acc_list.append(accuracy_score(y_test, imp_forest.predict(X_test_forest)))
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(imp_list,imp_train_acc_list, label = 'accuracy on train')
ax.plot(imp_list,imp_train_oob_list, label = 'oob accuracy on train')
ax.plot(imp_list,imp_test_acc_list, label = 'accuracy on test')


ax.legend()
ax.set_title('accuracy vs min_impurity_decrease')
plt.show()


# In[ ]:



n_feat = len(forest_features)
n_feat_start = round(math.sqrt(n_feat),0)
#print('max_features start: ', n_feat_start)

feat_list = range(int(n_feat_start),n_feat)

feat_train_acc_list = []
feat_train_oob_list = []
feat_test_acc_list = []

for feat in feat_list:
    
    #print('max_features: ', feat)
    parameters = {'random_state': 45,
                  'n_estimators': 2700,
                  'min_samples_leaf': 1, 
                  'min_impurity_decrease': 0.0045,
                  'max_samples': None,
                  'max_features': feat,
                  'criterion': 'entropy',
                  'class_weight': 'balanced',
                  'oob_score' : True,
                  'n_jobs' : -1
                 }
    
    feat_forest = RandomForestClassifier(**parameters)

    feat_forest.fit(X_train_forest,y_train)

    
    feat_train_acc_list.append(accuracy_score(y_train, feat_forest.predict(X_train_forest)))
    
    feat_train_oob_list.append(feat_forest.oob_score_)
                              
    feat_test_acc_list.append(accuracy_score(y_test, feat_forest.predict(X_test_forest)))
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(feat_list,feat_train_acc_list, label = 'accuracy train')
ax.plot(feat_list,feat_train_oob_list, label = 'oob train')
ax.plot(feat_list,feat_test_acc_list, label = 'accuracy test')


ax.legend()
ax.set_title('accuracy vs number of features')
plt.show()


# In[ ]:



best_params = {'random_state': 45, 'n_estimators': 2700, 'min_samples_leaf': 1, 
 'min_impurity_decrease': 0.0045, 'max_samples': None,
 'max_features': 6, 'criterion': 'entropy', 'class_weight': 'balanced', 'n_jobs' : -1}

best_params['oob_score'] = True
print(best_params)


# In[ ]:


best_forest = RandomForestClassifier(**best_params)
                              
best_forest.fit(X_train_forest, y_train)
print('oob accuracy score for chosen parameters: ', best_forest.oob_score_)


# In[ ]:


f_depth = 0
for f_tree in best_forest.estimators_:
    f_depth = f_depth + f_tree.get_depth()
print('average forest depth: ', round(f_depth/len(best_forest.estimators_),1))


# In[ ]:


cv_results(X_train_forest, y_train, best_forest, 10, 15)


# **Forest feature importances**

# In[ ]:


forest_importn = pd.Series(best_forest.feature_importances_,X_train_forest.columns).sort_values(ascending=False)
print(forest_importn.loc[forest_importn > 0])
forest_importn.loc[forest_importn > 0].index


# In[ ]:


plt.figure(figsize = (5,5))
pd.Series(best_forest.feature_importances_,X_train_forest.columns).sort_values(ascending=True).plot.barh(width=0.8)
plt.show()


# **accuracy score on test data depending on threshold**

# In[ ]:


best_forest.fit(X_train_forest, y_train)

y_forest_test_pred_prob = best_forest.predict_proba(X_test_forest)[:,1]

forest_acc_scores = []

for thr in np.linspace(0,1,100):
    y_forest_test_pred = [1 if m > thr else 0 for m in y_forest_test_pred_prob]
    forest_acc_score = accuracy_score(y_test, y_forest_test_pred)
    forest_acc_scores.append(forest_acc_score)

print('accuracy score with 0.5 threshold: ',
      accuracy_score(y_test, [1 if m > 0.5 else 0 for m in best_forest.predict_proba(X_test_forest)[:,1]]))

fig = plt.figure(figsize = (3,3))
ax = fig.add_subplot(1,1,1)
ax.set_title('accuracy vs threshold on test')
plt.plot(np.linspace(0,1,100), forest_acc_scores)

plt.show()


# **I choose 0.7 threshold for submission**

# In[ ]:


best_forest.fit(X_forest, y)


# In[ ]:


y_forest_sub = best_forest.predict_proba(X_submit_forest)
y_forest_sub_tr = [1 if m > 0.7 else 0 for m in y_forest_sub[:, 1]]

test_data2 = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': y_forest_sub_tr})
output.to_csv('Titanic_submission_forest.csv', index=False)


# **Forest scores 0.80382 on public, the same as tree**

# # The overall table for kNN, Tree and Forest
# **and pictures too, all obtained on 10 fold cross validation**<br>
# I just collected all the data in one table manually, too lazy to make changes in the code
# 

# In[ ]:


a = pd.DataFrame(columns=['kNN', 'Tree', 'Forest'], 
                 index = ['accuracy','precision','recall', 'f1', 'ROC AUC', 'public score'])

a.loc['accuracy'] = [0.817,0.795,0.828]
a.loc['precision'] = [0.786,0.725,0.787]
a.loc['recall'] = [0.722,0.762,0.758]
a.loc['f1'] = [0.749,0.74,0.771]
a.loc['ROC AUC'] = [0.871,0.85,0.885]
a.loc['public score'] = [0.80861,0.80382,0.80382]

a


# **kNN**

# ![knn.png](attachment:knn.png)

# **Tree**

# ![tree.png](attachment:tree.png)

# **Forest**

# ![forest.png](attachment:forest.png)

# # Conclusion
# As we can see, random forest gets best scores on cross validation, but kNN gets lucky on public data.<br>
# 
