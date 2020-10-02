#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train.head()
#Survived is the target variable


# **Missing Value Analysis and Data Clean Up**
# 

# In[ ]:


#dropping PassengerId as it does not contain information
train.drop('PassengerId', axis=1,inplace=True)

#Missing Value Analysis
def missing_check(column):
    check = column.isna().value_counts()
    try:
        count =  check[True]
    except KeyError:
        print('{} does not have any missing'.format(column.name))
    else:
        percentage_missing = count/column.shape[0]
        print('{} has {} of its data missing'.format(column.name, str(percentage_missing)))

for col in train:
    missing_check(train[col])


# You can see that Age, Cabin and Embarked have missing data. Age and Embarked have relatively insignificant number of missing data while Cabin has most of its data missing.
# 
# There are two ways to deal with missing data.
# 1. Discard the feature entirely
# 2. Fill in the missing data
#     
# Discarding method is usually employed on features with majority of its data missing (e.g. Cabin in our case), because the resulting features will be uninformative  that are no better than randomly generated ones.
# 
# For now, we will fill the values only for Age as LGBM model is able to handle null values. I have chosen to fill with the value 100 in order make them outliers.

# In[ ]:


train.Age.fillna(value = 100, inplace = True)


# In[ ]:


sns.countplot(x='Survived', data = train)
plt.show()


#  The data is imbalanced. Even though the imbalance is not significant, we will see that this imbalance will influence our final decision in making predictions. Dealing with imbalanced data is one of the hottest topic in machine learning and data mining at the moment. Using a naive model will run into Accuracy Paradox. There are different approaches to this. Sampling to artificially balance the data, different learning methods, calibrating probability outputs,  etc. Currently sampling method is the most efficient way to tackle this problem, but it is an expedient approach as the sampled data does not represent the original distribution.. In this kernel, I will try t
# 
# 
# **Analysis of features by graphs**
# 
# **2.1 Distribution of Features**
# 
# It is important to get a good idea of each feature's distribution, which will be important for feature extraction and model selection. The very first thing one should do is to make histrogram plots for each feature. For example, linear regression works on the assumption that predictor variables are approximately normally distributed (which can be checked by Q-Q plots). 
# 
# We will use count plots for categoricals and histograms for other ordinal or nominal variables

# In[ ]:


numerical = ['Age', 'Fare']

fig, axs = plt.subplots(1,len(numerical), figsize=(30, 15))
for k, col in enumerate(numerical):
        sns.distplot(train[col], kde=False, ax=axs[k])
plt.show()


# In[ ]:


fig, axs = plt.subplots(2,3, figsize=(30, 30))
axs = axs.flatten()
c = 0
for col in train.columns:
    if col not in (numerical + ['Survived','Name', 'Ticket', 'Cabin']):
        sns.countplot(x=col, data=train, ax=axs[c])
        c += 1
plt.show()


# **2.2 Distribution of the variable with respect to the target variable**
# 
# Graphing features with respect to the target variable will show each feature's general predictive power. Large separation between the disbritutions will suggest that it can be a powerful predictive tool whereas small separation will suggest the otherwise. 

# In[ ]:


def v_plot(violin_d, train):
    #standardize for plotting
    violin_d = (violin_d - violin_d.mean())/ violin_d.std() 
    violin_d = pd.concat([violin_d, train.Survived], axis=1)
    violin_d = pd.melt(violin_d, id_vars = 'Survived')

    plt.figure(figsize=(10,10))
    ax = sns.violinplot(x = 'variable', y='value', hue = 'Survived', data=violin_d
                  ,palette="muted", split=True)
    plt.show()
    plt.figure(figsize=(10,10))
    ax1 = sns.swarmplot(x = 'variable', y='value', hue = 'Survived', data=violin_d
                  ,palette="muted", split=True) 

    plt.figure(figsize=(10,10))
    ax2 = sns.boxplot(x = 'variable', y='value', hue = 'Survived', data=violin_d )
    plt.show()
    
violin_d = train[['Age', 'Fare']].copy()
v_plot(violin_d, train)


# It seems like Age might not a be a very powerful feature, as the two distributions are very similar. For Fare, distributions quite differ. It is evident that people with more expensive tickets were more likely to survive.

# In[ ]:


not_these = ['Name', 'Ticket', 'Cabin']
for col in train.columns:
    if col not in (['Survived','Age', 'Fare'] + not_these ) :
        sns.catplot(x= col, hue='Survived', kind="count", data=train)
        plt.show()


# **2.3 Feature Importance Analysis**
# 
# If the given variables were real not nominal, these graphs can also be summarized numerically by calculating each feature's correlation with the target variable. A predictor's Pearson correlation coefficient being close to 0 will suggest that it is indiscriminate. 
# 
# With ordinal or nominal variables,, one can make use of the Chi-Squared test of independence. However, it also assumes indepedence. 
# 
# The reason why the above graphs, the Pearson's R, or the Chi-Square  only suggest one feature's GENERAL predictive power is because of its independence assumptions which is made by any univariate methods. **Individually irrelevant features maybe become relevant in the context of others. Also, relevant features maybe not be useful due to possible redundancies. **
# 
# Thus, along with the above univariate methods, we will also make use of a multi-variate method to find the relative importance of each feature. In other words, we will look at the importance of a single feature in the context of others. The method that we will employ is called Random Forrest. It is a greedy method that adds one feature at a time to the subset of features measuring the improvment of a performance index. We will use information gain (entropy) as our performance index.

# In[ ]:


from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb

train[['Sex', 'Embarked']] = train[['Sex', 'Embarked']].astype('category')

Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis = 1)

#this choice of parameters is rather arbitrary (standard choice)


kfold = model_selection.StratifiedKFold(n_splits = 6, random_state=50)


def rf_v_importance(X, Y):
    vimp = pd.DataFrame()
    vimp['feature'] = X.columns
    AUC_t = []
    AUC_v = []
    
    params = { 'boosting_type': 'rf',
            'num_leaves' : 50,
            'colsample_bytree' : 0.6,
            'n_estimators' : 300,
            'min_child_weight' : 5,
            'min_child_samples' : 10,
            'subsample' : 0.632, 
            'subsample_freq' : 1,
            'metric' : 'auc'}
    
    for i, (t_idx, v_idx) in enumerate(kfold.split(X.values, Y.values)):

        print("Cross validation {0:d}-------------".format(i+1))
        lgb_t = lgb.Dataset(X.iloc[t_idx], Y.iloc[t_idx])
        lgb_v = lgb.Dataset(X.iloc[v_idx], Y.iloc[v_idx])
        t_ = lgb.train(params, lgb_t, valid_sets=lgb_v, verbose_eval=False, early_stopping_rounds = 15)

        fpr, tpr, _ = roc_curve(Y.iloc[t_idx], t_.predict(X.iloc[t_idx]))
        AUC_t.append(auc(fpr, tpr))
        print("Training AUC score: {0:.3f}".format(AUC_t[i]))

        fpr, tpr, _ = roc_curve(Y.iloc[v_idx], t_.predict(X.iloc[v_idx]))
        AUC_v.append(auc(fpr, tpr))
        print("Validation AUC score: {0:.3f}".format(AUC_v[i]))

        vimp["CV_"+ str(i)] = t_.feature_importance(importance_type='gain') 

    print("Final RF training AUC score {} | Final RF validation AUC score {}".format(
        sum(AUC_t)/len(AUC_t) , sum(AUC_v)/len(AUC_v)))

    vimp['S'] = pd.Series(vimp.iloc[:,1:-1].mean(axis=1).values)
    plt.figure(figsize = (20,10))
    sns.barplot(x='S',y='feature',data=vimp)
    plt.xlabel('Score')
    plt.ylabel('Feature Names')
    plt.title("Feature Importance by Information Gain")
    plt.show()

rf_v_importance(X, Y)


# **Baseline Model Performance**
# 
# We will use a LGBM gradient bossting machine model to set a baseline of our untouched dataset. (Random forrest model's baseline has already been determined above). Parameters used for training was quite arbitrary, mostly default setting give by LightGBM.. 

# In[ ]:


gbm_cv = lgb.Dataset(X,Y)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.01,
        'n_estimators' : 300,
        'metric' : 'auc'
    }

print("Cross Validation Score")
CV_baseline = lgb.cv(params, train_set=gbm_cv)
print(sum(CV_baseline['auc-mean'])/len(CV_baseline['auc-mean']))


# **ROC-AUC as the choice of a performance index**
# 
# It can be noted that ROC-AUC was used to measure the above model's performance rather than accuracy. The following section will illustrate the downside of using accruacy or other static metric to measure a model's performance.
# 
# 
# 

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, recall_score, average_precision_score, precision_score, precision_recall_curve
def auc_plot(true, pred):
    fpr, tpr, threshold = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %0.2f' % roc_auc)
    plt.plot([0,1], [0,1])
    plt.title("Receiver Operating Characteristic")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.show()

def precision_recall_plot(true, pred):
    precision, recall, threshold = precision_recall_curve(true, pred)
    AP = average_precision_score(true, pred)
    plt.plot(recall, precision, label='AP = %0.2f' % AP)
    plt.title("Precision-Recall Curve")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.show()

X_train, X_val, Y_train, Y_val,= model_selection.train_test_split(X, Y, test_size = 0.1, random_state = 20, shuffle =True)
gbm_train = lgb.Dataset(X_train, Y_train)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'n_estimators' : 100,
        #'num_leaves': 31,
        'max_depth': 3,
        #'min_child_samples': 2,
    }

model = lgb.train(params, gbm_train) #model for illustration
prediction = model.predict(X_val)
auc_plot(Y_val, prediction)
precision_recall_plot(Y_val, prediction)


# In[ ]:


def plot_confusion_matrix(true, pred, classifier, classes):
    decision = pred.copy()
    decision[decision <=  classifier] = 0 #make decisions
    decision[decision != 0] = 1
    
    cm = confusion_matrix(true, decision)
    
    accuracy = accuracy_score(true, decision)
    fscore = f1_score(true, decision)
    R = recall_score(true, decision)
    P = precision_score(true, decision)
    print("With {} decision threshold, the model's accuracy is {}".format(str(classifier),str(accuracy)))
    print("With {} decision threshold, the model's f1-score is {}".format(str(classifier),str(fscore)))
    print("With {} decision threshold, the model's recall is {}".format(str(classifier),str(R)))
    print("With {} decision threshold, the model's precision is {}".format(str(classifier),str(P)))
    plt.imshow(cm)
    plt.title('Confusion Matrix')
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(i,j, cm[i,j])

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.ylabel
    ('True label')
    plt.xlabel('Predicted label')
    plt.show()

classes = ['Died', 'Survived']
plot_confusion_matrix(Y_val, prediction, 0.5, classes)
plot_confusion_matrix(Y_val, prediction, 0.4, classes)
plot_confusion_matrix(Y_val, prediction, 0.6, classes)
plot_confusion_matrix(Y_val, prediction, 0.3838, classes) #prior probability: p(Survived = 1)


# Using different decision thresholds led to different accuracy scores. Using 0.5 as the decision threshold resulted in lower accuracy than using the prior probability of the positive class. 
# 
# There are two problems in using 0.5 as the decision threshold to make predictions (thus using it to measure performance);
# 
# 1. Distribution of classes are imbalanced
# 2. Decision Tree classifier's resulting probabilities are not proper posterior probabilities. (LGBM does a good job of "calibrating" probabilities as it does leaf-wise splitting)
# 
# ROC-AUC performance measure is indifferent to class-distribution (in the scope of binary classification). In this case, the imbalance is not great and carefully training the model can render using 0.5 as the threshold appropriate. (It is usually positive to negative ratio of 1:10 or worse that will require special treatment)
# 
# The important thing to take away from this is that once a model has ouputted baseline probabilities, statistical analysis has ended. Choosing the right decision threshold will depend on one's the objective. For example, in the context of cancer prediction, reducing the rate of false negative is far more of importance than the false positive rate (as telling cancer baring patient that he/she is okay would be disasterous). Thus, one would take this into account to make decisions.
# 

# **3. Feature Selection and Feature Engineering**
# 
# **When you are adding new features, you are providing more informaiton to the model but also you are adding noise to the model**. Thus, the act of adding more features will not necessarily improve the model's effectiveness. 
# 
# Also, decision trees work by making horizontal or vertical cuts on a domain, which means altaring an existing non-categorical feature into a ordinal or nominal feature will not be effective (i.e. Altering Age features into an ordinal feature by age category.)
# 
# First we will look at yet to be used features and see if we can extract valuable features from them.
# 
# 
# **3.1 Cabin**
# 
# We will extract Deck and Cabin Number

# In[ ]:


#Cabin
train.Cabin.fillna(value = 'Non', inplace = True)
def get_deck(x):
    if x == "Non":
        return "N"
    else:
        return x[0]
       
def get_cnumber(x):
    if x == 'Non':
        return 0
    else:
        c = x.split(" ")
        n = []
        for i in c:
            if i[1:] != '':
                n.append(int(i[1:]))
            else:
                n.append(200)
        
        return sum(n)/len(n)
    
train['Deck'] = train.Cabin.apply(lambda x: get_deck(x))
train.Deck = train.Deck.astype('category')
train['CNumber'] = train.Cabin.apply(lambda x: get_cnumber(x))    

Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis = 1)

sns.countplot(x='Deck', hue='Survived', data=train)
v_plot(train['CNumber'].copy(), train)
rf_v_importance(X, Y)


# Adding Cabin features added more noise than information. This makes sense as 77 percent of its data was missing from the column.
# 
# **3.2 Ticket**
# 
# We will extract the number only.

# In[ ]:


def clean_ticket(x):
    t = x.split(' ')
    if len(t) != 1:
        return int(t[-1])
    else:
        if t[0].isnumeric():
            return int(t[0])
        else:
            return 0
train['ticket'] = train.Ticket.apply(lambda x: clean_ticket(x))
train.ticket = train.ticket.astype('int32')

Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'Deck', 'CNumber'], axis = 1)

v_plot(train['ticket'].copy(), train)
rf_v_importance(X, Y)


# Adding Cabin features added more noise than information. 

# In[ ]:


a = ['Pclass', 'Age','Fare', 'CNumber', 'ticket', 'Deck']
for i in range(len(a)):
    for j in ['Parch','SibSp']:
         sns.catplot(x=j, y= a[i], hue='Survived', data=train)


# **3.3 Title**
# 
# We will estract title from each name.

# In[ ]:


import re
def get_title(x):
    title = re.search('\s(.*)\.', x).group(1)
    t = title.split(' ')
    if len(t) != 1:
        return t[-1]
    else:
        return t[0]

train['Title'] = train.Name.apply(lambda x: get_title(x))
train.Title = train.Title.astype('category')

Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'Deck', 'CNumber', 'ticket'], axis = 1)

plt.figure(figsize=(20,10))
sns.countplot(x='Title', hue='Survived', data=train)
plt.show()
rf_v_importance(X, Y)


# This feature seems to be useful!

# **3.4 Has Other Name**
# Observing "Name", it can be seen that some people will have an additional name attached to it. We will make a feature out of this.

# In[ ]:


def has_other_name(x):
    try:
        re.search('\(', x).group(0)
    except AttributeError:
        return 0
    else:
        return 1
train['Other_Name'] = train.Name.apply(lambda x: has_other_name(x))
train.Other_Name = train.Other_Name.astype('uint8')

Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'Deck', 'CNumber', 'ticket', 'Title'], axis = 1)

sns.countplot(x='Other_Name', hue='Survived', data=train)
plt.show()
rf_v_importance(X, Y)


# Now we will look at all the features together and compare it with the previous subset of features..

# In[ ]:


Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'Deck'], axis = 1)
rf_v_importance(X, Y)


# This result asserts the point made above: 
# **Individually irrelevant features maybe become relevant in the context of others.** 
# 
# Also, note that "Deck" has little to no importance  to the model. Leaving out this feature results in a slight decrease in our validation score (0.01), thus we will keep the column. However. this is the case only because we are not working with a big dataset with hundreds of columns and a lot of records where having one more columns can be much more costly to train and maintain in the future, This tradeoff between cost and accuracy is an important one.

# **4. Hyperparameter Tuning**
# 
# Instead of using grid/random search we will make use of hyperopt for hyperparameter tuning. 

# In[ ]:


from hyperopt import hp, tpe
from hyperopt.fmin import fmin
X_cv = lgb.Dataset(X, Y)
def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'n_estimators': int(params['n_estimators']),
        'learning_rate': '{:.5f}'.format(params['learning_rate'])
    }
    
    clf = lgb.LGBMClassifier(params)

    
    model_cv = lgb.cv(params, X_cv, nfold = 10,metrics = 'auc',seed = 50)
    score = 1 - max(model_cv['auc-mean'])
    print("AUC {:.6f} params {}".format(1 - score, params))
    return score

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'n_estimators': hp.quniform('n_estimator',50, 500, 50),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5))
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30)


# In[ ]:


best


# In[ ]:


best['boosting'] = 'gbdt'
best['metric'] = 'auc'
best['n_estimator'] = int(best['n_estimator'])
best['num_leaves'] = int(best['num_leaves'] )


# In[ ]:


Y = train.Survived
X = train.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis = 1)

X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size = 0.1, random_state = 50, shuffle = True)
gbm_train = lgb.Dataset(X_train, Y_train)
gbm_val = lgb.Dataset(X_val, Y_val)

final_model = lgb.train(best, gbm_train, valid_sets = gbm_val, early_stopping_rounds = 15)

def predict(t, final_model, threshold):
    t.Age.fillna(value = 200, inplace = True)
    t.Cabin.fillna(value = 'Non', inplace = True)
    t.Embarked.fillna(value = 'Non', inplace = True)
    t['Deck'] = t.Cabin.apply(lambda x: get_deck(x))
    t['Title'] = t.Name.apply(lambda x: get_title(x))
    t['ticket'] = train.Ticket.apply(lambda x: clean_ticket(x))
    t['Other_Name'] = t.Name.apply(lambda x: has_other_name(x))
    t['CNumber'] = train.Cabin.apply(lambda x: get_cnumber(x))   
    t[['Sex', 'Embarked', 'Title', 'Deck']] = t[['Sex', 'Embarked','Title', 'Deck']].astype('category')
    t.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis = 1, inplace = True)
    
    prediction = final_model.predict(t)
    prediction[prediction <= threshold] = 0
    prediction[prediction != 0] = 1
    
    return prediction

prediction = predict(test.copy(),final_model, 0.5)
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':prediction})
submission.to_csv('Submission',index=False)
    


# **5.Ensembling**
