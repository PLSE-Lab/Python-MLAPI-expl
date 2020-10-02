#!/usr/bin/env python
# coding: utf-8

# ## Import libs

# In[251]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics, model_selection, tree, neural_network, ensemble, linear_model, svm, neighbors, preprocessing

import matplotlib.pyplot as plt
import seaborn as sns


# ## Helper class and methos

# The learning class is just a class that helps me to test different models and data manipulations. It can automatically split the data into train and test, drop things, train, generate the output file, etc.

# In[252]:


class Learning():
    model = None
    
    data = []
    prediction_data = []
    train = {}
    test = {}
    
    to_drop = []
    to_dummy = []
    to_binarize = []
    
    def split(self):
        data_temp = self.data.copy()
        
        data_temp = self.preprocess_data_temp(data_temp)
        
        X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']
        
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = model_selection.train_test_split(X, y)
        
    def train_model(self, complete=False):
        if complete:
            data_temp = self.data.copy()
            data_temp = self.preprocess_data_temp(data_temp)
            
            X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']
            
            self.model.fit(X, y)
            
        else:
            self.model.fit(self.train['X'], self.train['y'])
        
    def score(self):
        probs = self.model.predict_proba(self.train['X'])
        probs = probs[:, 1]
        auc = metrics.roc_auc_score(self.train['y'], probs)
        
        p = self.model.predict(self.train['X'])
        
        print('Train score:')
        print(f'Precision: {metrics.precision_score(self.train["y"], p)}')
        print(f'Accurracy: {metrics.accuracy_score(self.train["y"], p)}')
        print(f'Recall: {metrics.recall_score(self.train["y"], p)}')
        print(f'ROC_AUC: {auc}')
        
        p = self.model.predict(self.test['X'])
        
        probs = self.model.predict_proba(self.test['X'])
        probs = probs[:, 1]
        auc = metrics.roc_auc_score(self.test['y'], probs)
        
        print('Test score:')
        print(f'Precision: {metrics.precision_score(self.test["y"], p)}')
        print(f'Accurracy: {metrics.accuracy_score(self.test["y"], p)}')
        print(f'Recall: {metrics.recall_score(self.test["y"], p)}')
        print(f'ROC_AUC: {auc}')
        
    def cross_validate(self):
        data_temp = self.data.copy()
        
        data_temp = self.preprocess_data_temp(data_temp)
        
        X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']
        
        cv = model_selection.cross_validate(self.model,
            X, y, cv=5, scoring=['precision', 'accuracy', 'recall', 'roc_auc'],
            return_train_score=True)

            
        print('Train score:')
        print(f'Precision: {cv["train_precision"]}')
        print(f'Accurracy: {cv["train_accuracy"]}')
        print(f'Recall: {cv["train_recall"]}')
        print(f'ROC_AUC: {cv["train_roc_auc"]}')
        
        print('Test score:')
        print(f'Precision: {cv["test_precision"]}')
        print(f'Accurracy: {cv["test_accuracy"]}')
        print(f'Recall: {cv["test_recall"]}')
        print(f'ROC_AUC: {cv["test_roc_auc"]}')
        
    def output(self):
        data_temp = self.prediction_data.copy()
        
        data_temp = self.preprocess_data_temp(data_temp)
        
        p = self.model.predict(data_temp)
        
        csv_dict = {'ID': self.prediction_data.ID, 'Default': p}
        
        csv = pd.DataFrame(csv_dict)
        
        csv.to_csv('output.csv', index=False, encoding='utf8')
        
    def preprocess_data_temp(self, data_temp):
        if self.to_dummy:
            data_temp = pd.get_dummies(columns=self.to_dummy, data=data_temp)
            
        if self.to_drop:
            data_temp = data_temp.drop(columns=self.to_drop)
        
        if self.to_binarize:
            binarizer = preprocessing.Binarizer()
            data_temp[self.to_binarize] = binarizer.fit_transform(data_temp[self.to_binarize])
        
        return data_temp


# In[253]:


def correlation_plot(dataframe):
    correlations = dataframe.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();


# ## Read data

# In[254]:


data = pd.read_csv('../input/train.csv')
prediction_data = pd.concat([pd.read_csv('../input/valid.csv'), pd.read_csv('../input/test.csv')])


# ## Visual Analysis

# In[255]:


data.describe()


# For some reason we have PAY_0 instead of PAY_1. I don't actually know if it should mean that we have data from the month before month 1, but I have no reason to believe so. Also, I can't really work considering that I don't have one month of data, so let just change the column name.

# In[256]:


data = data.rename(columns={'PAY_0': 'PAY_1'})
prediction_data = prediction_data.rename(columns={'PAY_0': 'PAY_1'})


# Let's check the amount of individuals that each class have.

# In[257]:


plt.figure(figsize=(10, 6))

sns.countplot(x='default payment next month', data=data)


# In[258]:


non_default = len(data[data['default payment next month'] == 0])
default = len(data[data['default payment next month'] == 1])

print(f'Percentage of non-defaulters: {non_default/(non_default+default)}')
print(f'Percentage of defaulters: {default/(non_default+default)}')


# We have alot more non-defaulters than defaulters. This might pose a problem while training.

# Let's analyse our categorial columns.

# In[259]:


plt.figure(figsize=(10, 6))

sns.countplot(x='EDUCATION', hue='default payment next month', data=data)


# In[260]:


data.groupby('EDUCATION').count()


# The documentation doesn't say what education levels 0, 5 and 6 mean. They also have low amounts of data. They could easily mean other kind of education levels, though.

# In[261]:


plt.figure(figsize=(10, 6))

sns.countplot(x='SEX', hue='default payment next month', data=data)


# More woman than men , that's ok.

# In[262]:


plt.figure(figsize=(10, 6))

sns.countplot(x='MARRIAGE', hue='default payment next month', data=data)


# In[263]:


data.groupby('MARRIAGE').count()


# The docs also doesn't say what marriage status 0 and 3 mean. They could easily mean divorce, stable union, or other things.

# In[264]:


pay_x = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

data[pay_x].describe()


# In[265]:


fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(14, 10))

sns.countplot(x='PAY_1', hue='default payment next month', data=data, ax=axs[0][0])
sns.countplot(x='PAY_2', hue='default payment next month', data=data, ax=axs[0][1])
sns.countplot(x='PAY_3', hue='default payment next month', data=data, ax=axs[0][2])
sns.countplot(x='PAY_4', hue='default payment next month', data=data, ax=axs[1][0])
sns.countplot(x='PAY_5', hue='default payment next month', data=data, ax=axs[1][1])
sns.countplot(x='PAY_6', hue='default payment next month', data=data, ax=axs[1][2])


# The docs say that PAY_X denotes the amount of delay in the payment at that month. It doesn't really explain what values -2 and 0 mean. Theoretically, these columns should be somewhat related to bill columns (I would assume an increased bill if someone delayed the payment). Let's check.

# In[266]:


f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

data[f].head(20)


# That's strange. There are columns with -2 in PAY_X that don't have any BILL_AMTX value. 

# In[267]:


data.loc[data.index == 18][f]


# This would lead me to think that -2 means no credit card use. But, at the same time, we have this:

# In[268]:


data.loc[data.index == 20989][f]


# In[269]:


data.loc[data.index == 17][f]


# So this means that -2 is still somewhat related to credit card use. Same with zero. The fact that there are no negative values in BILL when the PAY_X values are 0 or -2 makes me think that they aren't related to some kind of antecipated payment. So, I have three options:
# 
# 1. Leave these columns as they are.
# 2. Bring every value that is less than 0 to zero, imagining that they are the same thing.
# 3. Assume that -2 means "no credit card use", and then drop every column where this isn't true.

# In[270]:


data.columns


# In[271]:


correlation_plot(data)


# In[272]:


f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'default payment next month', 'LIMIT_BAL']

correlation_plot(data[f])


# As expected, the LIMIT_BAL values, the BILL_AMT and the PAY_X values are highly correlated. BILL seems to increase if PAY_X increases. I wonder if this relation becomes stronger if I drop every PAY_X < 0.

# In[273]:


temp = data.copy()


# In[274]:


temp = data.copy()

pay_x = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for pay in pay_x:
    mask = (temp[pay] >= 0)
    temp = temp.loc[mask]
    
correlation_plot(temp[f])


# The relation becomes inverse... This means that the BILL values in the columns where the PAY_X is negative are very, very high. One interesting thing to note is that the relation to the default payment next month column becomes higher for all pay levels.

# ## Testing models

# In[275]:


learning = Learning()
learning.data = data
learning.prediction_data = prediction_data

learning.split()


# ### Random Forest

# In[276]:


learning.model = ensemble.RandomForestClassifier(n_estimators=10)
learning.train_model()
learning.score()


# ### K-Nearest Neighbors

# In[277]:


learning.model = neighbors.KNeighborsClassifier()
learning.train_model()
learning.score()


# ### Neural Network

# In[278]:


learning.model = neural_network.MLPClassifier()
learning.train_model()
learning.score()


# All of these algorithms where tried with different parameters multiple times. RandomForest, without any parameter tuning, performed better than all of them. It is probably a result of my lack of knowledge about these other algorithms.

# ## Data Manipulation

# First, let's try to resample the data to balance the proportion of defaulters and non-defaulters.

# In[279]:


mask = (data['default payment next month'] == 1)

defaulters_data = data[mask]

defaulters_amount = len(defaulters_data)

mask = (data['default payment next month'] == 0)

non_defaulters_data = data.loc[mask]

non_defaulters_data = non_defaulters_data.sample(defaulters_amount)


# In[280]:


new_data = pd.concat([defaulters_data, non_defaulters_data])


# Since the docs doesn't tell us what the parameters -2 and 0 mean, let's try to bring everything to zero based on our previous assumption.

# In[281]:


def modify_data(dataframe):
    f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    for column in f:
        mask = (dataframe[column] < 0)
        dataframe.loc[mask, column] = 0
        
    return dataframe
        
new_data = modify_data(new_data)


# Now, let's dummy our categories (with the exception of PAY_X, since it would generate alot of columns).

# In[282]:


learning = Learning()

learning.data = new_data
learning.prediction_data = prediction_data

learning.to_drop = ['ID']
learning.to_dummy = ['EDUCATION', 'SEX', 'MARRIAGE']
learning.split()


# In[283]:


learning.model = ensemble.RandomForestClassifier(n_estimators=10)
learning.train_model()
learning.score()


# Okay. Let's check our model with cross validation and full training.

# In[284]:


learning.train_model(complete=True)
learning.cross_validate()


# ## Hyperparametrization

# In[285]:


learning.model = ensemble.RandomForestClassifier(n_estimators=200, max_leaf_nodes=200, min_samples_split=42,
                                                 n_jobs=-1)
learning.train_model()
learning.score()


# The parameters where chosen manually, running the algorithm multiple times. A better strategy would use something like ```GridSearch``` to find the best parameters.

# Let's check our performance with cross-validation.

# In[286]:


learning.cross_validate()


# Seems okay. Let's see our performance with full training.

# In[287]:


learning.train_model(complete=True)
learning.score()


# ## ROC and AUC

# The ROC curve is the measure of the true positive rate (the probability of the model being right when it says that a value pertains to a class), also called sensitivity, versus the false positive rate (the probability of the model being wrong when it says that a value pertains to a class), also called the inverse specitivity (specitivity is the probability of a model being right when it says that a value does not pertains to a class). 
# 
# The curve is created changing the threshold. The threshold is the limit where we define that every output above it is from a class and every output below is from another class.

# In[288]:


learning.train_model(complete=False)
learning.score()


# In[321]:


probs = learning.model.predict_proba(learning.test['X'])
probs = probs[:, 1]

fpr, tpr, thresholds = metrics.roc_curve(learning.test['y'], probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')


# The area below the curve is a way to compare different ROC curves. It gives us the probability of our model giving a higher score to a random X value that is from class 1, and a lesser score to a random X value that is from class 0.

# Alright. Now we can generate our output.

# In[290]:


learning.prediction_data = modify_data(prediction_data)
learning.output()

