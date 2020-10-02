# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 01:44:06 2018

@author: Hager
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
import re
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#################
training_data = '../input/AdultIncome/adults-training.csv'
test_data = '../input/AdultIncome/adults-test.csv'
columns = ['Age','Workclass','fnlgwt','Education','Education Num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Country','Above/Below 50K']
train=pd.read_csv(training_data, names=columns)
test=pd.read_csv(test_data, names=columns)
##################
def missing_value(df):
    miss=[]
    col_list=df.columns
    for i in col_list:
        missing=df[i].isnull().sum()
        miss.append(missing)
        list_of_missing=pd.DataFrame(list(zip(col_list,miss)))
    return list_of_missing
###################
  missing_value(test)
  missing_value(train)
##################
  train.Relationship.value_counts()
##################
  test.Occupation.value_counts()
  ##################
  print(test.shape)
print(train.shape)
#################
test.drop(test.index[0]).head()
train.drop(train.index[0]).head()
################
all_data=[train, test]
str_list=[]

for data in all_data:
    for colname, colvalue in data.iteritems(): 
        if type(colvalue[1]) == str:
            str_list.append(colname) 
num_list = data.columns.difference(str_list)
################
print(test.isnull().sum())
###############
for data in all_data:
    for i in data.columns:
        data[i].replace(' ?', np.nan, inplace=True)
    data.dropna(inplace=True)
###############
     test.isnull().sum()
     ############
    # defining the target variable
for data in all_data:
    data['target']=data['Above/Below 50K'].apply(lambda x: x.replace('.', ''))
    data['target']=data['target'].apply(lambda x: x.strip())
    data['target']=data['target'].apply(lambda x: 1 if x=='>50K' else 0)
    data.drop(['Above/Below 50K'], axis=1, inplace=True)
###############
     train.target.sum()/len(train)
###############
     def bin_var(data, var, bins, group_names):
    bin_value = bins
    group = group_names
    data[var+'Cat'] = pd.cut(train[var], bin_value, labels=group)
###############
     bin_var(train, 'Education Num', [0,6,11,16], ['Low', 'Medium', 'High'])
     bin_var(test, 'Education Num', [0,6,11,16], ['Low', 'Medium', 'High'])
###############
   pd.crosstab(train['Education NumCat'],train['target'] )
##############
   bin_var(train, 'Hours/Week', [0,35,40,60,100], ['Low', 'Medium', 'High','VeryHigh'])
   bin_var(test, 'Hours/Week', [0,35,40,60,100], ['Low', 'Medium', 'High','VeryHigh'])
#############
   pd.crosstab(train['Hours/WeekCat'],train['target'], margins=True)
############
   occu=pd.crosstab(train['Occupation'],train['target'], margins=True).reset_index()
############
   def occup(x):
    if re.search('managerial', x):
        return 'Highskill'
    elif re.search('specialty',x):
        return 'Highskill'
    else:
        return 'Lowskill'
##############
        train['Occupa_cat']=train.Occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))
test['Occupa_cat']=test.Occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))

########
train['Occupa_cat'].value_counts()

###########
bin_var(test, 'Age', [17,30,55,100], ['Young', 'Middle_aged', 'Old'])

#########
bin_var(train, 'Age', [17,30,55,100], ['Young', 'Middle_aged', 'Old'])

#########
train['Marital Status_cat']=train['Marital Status'].apply(lambda x: 'married' if x.startswith('Married',1) else 'Single')
test['Marital Status_cat']=test['Marital Status'].apply(lambda x: 'married' if x.startswith('Married',1) else 'Single')

###########
pd.crosstab(train['Race'],train['target'], margins=True)

############
train['Race_cat']=train['Race'].apply(lambda x: x.strip())
train['Race_cat']=train['Race_cat'].apply(lambda x: 'White' if x=='White' else 'Other')
test['Race_cat']=test['Race'].apply(lambda x: x.strip())
test['Race_cat']=test['Race_cat'].apply(lambda x: 'White' if x=='White' else 'Other')

#########28
train.Workclass.value_counts()

############29
def workclas(x):
    if re.search('Private', x):
        return 'Private'
    elif re.search('Self', x):
        return 'selfempl'
    elif re.search('gov', x):
        return 'gov'
    else:
        return 'others'
    
    ###########
train['WorfClass_cat']=train.Workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
test['WorfClass_cat']=test.Workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
##########31
train['WorfClass_cat'].value_counts()

############
# assigning the target to Y variable
Y_tr=train['target']
Y_te=test['target']

##########
train.drop(['Education','Occupation','Race','Education Num','Age', 'Hours/Week', 'Marital Status','target','fnlgwt','Workclass', 'Capital Gain','Capital Loss', 'Country'], axis=1, inplace=True)
test.drop(['Education','Occupation','Race','Education Num','Age', 'Hours/Week', 'Marital Status','Workclass','target','fnlgwt', 'Capital Gain','Capital Loss', 'Country'], axis=1, inplace=True)

##########
str_list=['WorfClass_cat','Education NumCat', 'AgeCat', 'Race_cat',
'Hours/WeekCat',
 'Marital Status_cat',
 'Occupa_cat',
 'Relationship',
 'Sex']

train_set=pd.get_dummies(train, columns=str_list)
test_set=pd.get_dummies(test, columns=str_list)

###########
train_set.columns

###########
from sklearn.feature_selection import VarianceThreshold
def variance_threshold_select(df, thresh=0.0, na_replacement=-999):
    df1 = df.copy(deep=True) # Make a deep copy of the dataframe
    selector = VarianceThreshold(thresh) # passing Threshold
    selector.fit(df1.fillna(na_replacement)) # Fill NA values as VarianceThreshold cannot deal with those
    df2 = df.loc[:,selector.get_support(indices=False)] # Get new dataframe with columns deleted that have NA values
    return df2

###########
    df2=variance_threshold_select(train_set, thresh=.8* (1 - .8))
    
    ###########
print(df2.columns)

##########
col_tr=df2.columns # creates list of columns
col_te=test_set.columns # creates list of columns for test
X_tr=df2.values # creates array of values of features
X_te=test_set[col_tr].values#subseting the test dataset to get the same variable as train and

###########
len(col_tr)

###########
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')

#    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    if print_res == 1:
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)


###########
    lrn = LogisticRegression(penalty = 'l1', C = .001, class_weight='balanced')

lrn.fit(X_tr, Y_tr)
y_pred = lrn.predict(X_te)
cm = confusion_matrix(Y_te, y_pred)
if lrn.classes_[0] == 1:
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])

plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1);


##########

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
    
print ('Accuracy:', accuracy_score(Y_te, y_pred))
print ('F1 score:', f1_score(Y_te,y_pred))

##########
# Understanding the coefficients
coff=pd.DataFrame(lrn.coef_).T 
col=pd.DataFrame(col_tr).T 
print(coff)
print(col)

#########
from sklearn.feature_selection import RFE, f_regression
#stop the search when only the last feature is left
rfe = RFE(lrn, n_features_to_select=10, verbose =3 )
rfe.fit(X_tr,Y_tr)
list(zip(map(lambda x: round(x, 4), rfe.ranking_), col_tr))