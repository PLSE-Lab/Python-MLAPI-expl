#!/usr/bin/env python
# coding: utf-8

# ***Introduction***
# Christian R. Lynn M.S. Data Science
# 07/13/2019

# This analysis will follow the CRISP-DM industry standard practices, following the basic format outlined below:
# * 1.) Business Understanding
# * 2.) Data Understanding
# * 3.) Data Preparation
# * 4.) Modeling
# * 5.) Deployment
# 
# This dataset obtained from Kaggle, is presented with the goal of identifying and creating a model to identify the risk factors associated with cervical cancer. However, due to the nature of the data, it is rather difficult to identify a superiorly performing model, and the understanding gained from any modeling technique is somewhat thin. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")


# In order to take a quick look at the data:

# In[ ]:


df.head()


# **PARTS 1 AND 2: BUISNESS AND DATA UNDERSTANDING**

# This code is heavily populated with a "?" value. This value is either due to informal sampling, a failure in data input, or a combination of both. However, we can replace these values with something our computer can understand more easily, a NaN value

# In[ ]:


df_nan = df.replace("?", np.nan)


# In[ ]:


df_nan.head()


# As shown in person number 2: all "?" have been converted to a NAN

# In[ ]:


df1 = df_nan.convert_objects(convert_numeric=True)


# Let's take a deeper dive into what type of data we have. Graphically represented, we can understand all the data quite easily all at once

# In[ ]:


df1.hist()
fig = plt.gcf()
fig.set_size_inches(25,17)


# As we can see there are two distinct shapes of data in this project, a right-tailed distribution of a continuous variable, and many binominal variables that seem to take the shape (1/0) with the zeros heavily outweighing the ones. Looking initially I see three problem features:
# Age, there are a few very large numbers that may influence the modeling process heavily
# STDs:AIDS looks like there are no positives, and may be a great variable to trim out of modeling dataset
# STDs:Vaginal Condylomatosis looks as if there are no positive instances as well. 
# 
# What is also very important to note is the lack of a target function. In this case, we are going to manufacture a target variable using some of the features above. 
# 
# 'Hinselmann', 'Schiller','Citology', and 'Biopsy' are all tests for cancer
# 
# Hinselmann's test is also known as a colposcopy test, is an early diagnostic/preventative test for cancer, it involves detecting and treating precancerous lesions early via colposcope.
# 
# Schiller's test is a preliminary test for cancer in which the cervix is painted with a solution of iodine and potassium iodide and which highlights cancerous tissues white or yellow - this test is not known for being patricularly accurate, but will highlight most tissue that is unhealthy (inflammation, unceration...)
# 
# A Cytology is a test preformed on body fluid that will help indicate the presence of canceroud tissue, it is not as accurate as a full bioposy, but is much easier to obtain a sample.
# 
# These features, along with biopsy will be used as the target variable for cancer. Concidering their lack of presence in the dataset as shown below:

# In[ ]:


H = df1['Hinselmann'].T.sum()
S = df1['Schiller'].T.sum()
C = df1['Citology'].T.sum()
B = df1['Biopsy'].T.sum()
H+S+C+B


# To visualize the lack of certain values in the dataset, we will first graphically inspect the data. Wherever a bar is shown there is a mising value. 

# In[ ]:


sns.heatmap(df1.isnull(), cbar=False)


# **PART 3: DATA MANIPULATION**
# 
# There are plenty of missing/NaN values in this dataset, we need to efficiently filter and create a good way to delete them. Firstly, let's get rid of Time Since First Diagnosis and Time Since Last Diagnosis

# In[ ]:


df1.columns = df1.columns.str.replace(' ', '')  #deleting spaces for ease of use


# In[ ]:


df1.drop(['STDs:Timesincefirstdiagnosis','STDs:Timesincelastdiagnosis','STDs:cervicalcondylomatosis','STDs:AIDS'],inplace=True,axis=1)


# In[ ]:


df1.isnull().T.any().T.sum()


# There are still over 190 distinct measurements that have missing values. Due to the size of our dataset, instead of simply deleting them, lets come up with a better way of handling them. Let's see if there are some entries that are particularly bad (missing many values)
# 
# After a few tests, the best limiting number is 10, with entries missing more than 10 variables are worthy of deletion

# In[ ]:


sns.heatmap(df1.isnull(), cbar=False)
df.shape


# In[ ]:


df = df1[df1.isnull().sum(axis=1) < 10]


# In[ ]:


df.shape


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)


# GREAT! Now we can fill values more efficiently with reasonable replacement values. This dataset contains two types of values, numerical, and binominal values indicating the presence of a certain diagnosis or risk factor in that patient. 

# In[ ]:


numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse','Numofpregnancies', 'Smokes(years)',
                'Smokes(packs/year)','HormonalContraceptives(years)','IUD(years)','STDs(number)']
categorical_df = ['Smokes','HormonalContraceptives','IUD','STDs','STDs:condylomatosis',
                  'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis','STDs:pelvicinflammatorydisease', 'STDs:genitalherpes',
                  'STDs:molluscumcontagiosum','STDs:HIV','STDs:HepatitisB', 'STDs:HPV', 'STDs:Numberofdiagnosis',
                  'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']


# We will fill all of the numerical value types with average values. After exhaustive investigating into the data, there was no need to more carefully fix the data.

# In[ ]:


for feature in numerical_df:
    print(feature,'',df[feature].convert_objects(convert_numeric=True).mean())
    feature_mean = round(df[feature].convert_objects(convert_numeric=True).mean(),1)
    df[feature] = df[feature].fillna(feature_mean)


# In[ ]:


(df['Age'] == 0).astype(int).sum() # checking if any 0 values in a column that could not contain such values


# In[ ]:


for feature in categorical_df:
    
    df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)
    
#Filling binominal values with 0, with the assumption that if present, feature would have been recorded


# In[ ]:


df5 = df.copy()


# Creating a whole new set of features, mostly creating features with values using AGE, and denoting the length of the condition. A few of the features are common sense and add to the models. Smokes(packs/year) and Smokes(years) will be deleted in favor of "TSP"
# 
# a few of the variables are Z-scores of other numberical variables such as Numofpregnancies, Numberofsexualpartners, HormonalContraceptives(years), and Firstsexualintercourse

# In[ ]:


df5['YAFSI'] = df5['Age'] - df5['Firstsexualintercourse']
df5['CNT'] = df.astype(bool).sum(axis=1)
df5['SEX'] = (df5['Numofpregnancies']+1) * (df5['Numberofsexualpartners']+1)
df5['FirstSexZ'] = (((df5['Firstsexualintercourse']+1) - (df5.loc[:,'Firstsexualintercourse'].mean())+1) / (df5.loc[:,'Firstsexualintercourse'].var()+1)*100)
df5['SexZ'] = (((df5['Numberofsexualpartners']+1) - (df5.loc[:,'Numberofsexualpartners'].mean())+1) / (df5.loc[:,'Numberofsexualpartners'].var()+1)*100)
df5['PILL'] = (((df5['HormonalContraceptives(years)']+1) - (df5.loc[:,'HormonalContraceptives(years)'].mean())+1) / (df5.loc[:,'HormonalContraceptives(years)'].var()+1)*100)
df5['SSY'] = df5['Age'] - df5['Smokes(years)']
df5['SPYP'] = df5['Numberofsexualpartners'] / df5['YAFSI']
df5['SP'] = df5['Smokes(years)'] / df5['Age']
df5['HCP'] = df5['HormonalContraceptives(years)'] / df5['Age']
df5['STDP'] = df5['STDs(number)'] / df5['Age']
df5['IUDP'] = df5['IUD(years)'] / df5['Age']
df5['TSP'] = df5['Smokes(packs/year)'] * df5['Smokes(years)']
df5['NPP'] = df5['Numofpregnancies'] / df5['Age']
df5['NSPP'] = df5['Numberofsexualpartners'] / df5['Age']
df5['NDP'] = df5['STDs:Numberofdiagnosis'] / df5['Age']
df5['YAHC'] = df5['Age'] - df5['HormonalContraceptives(years)']
df5['YAIUD'] = df5['Age'] - df5['IUD(years)']
df5['NPSP'] = df5['Numofpregnancies'] / df5['Numberofsexualpartners']
df5['IUDSY'] = df5['IUD(years)'] / df5['YAFSI']
df5['HCSY'] = df5['HormonalContraceptives(years)'] / df5['YAFSI']


# In[ ]:


df5.replace([np.inf, -np.inf], np.nan, inplace = True) #deleting extreme values caused by calculations


# In[ ]:


df = df5.copy()


# In[ ]:


numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse','Numofpregnancies', 'Smokes(years)',
                'Smokes(packs/year)','HormonalContraceptives(years)','IUD(years)','STDs(number)', 'YAFSI', 'CNT',
                'FirstSexZ', 'SexZ', 'PILL','SSY','SPYP', 'SP', 'HCP', 'STDP', 'IUDP', 'TSP', 'NPP', 'NSPP', 'NDP',
                'YAHC', 'YAIUD', 'NPSP', 'IUDSY', 'HCSY']


# In[ ]:


#Adding in our newly created values to the NA filter
for feature in numerical_df:
    print(feature,'',df[feature].convert_objects(convert_numeric=True).mean())
    feature_mean = round(df[feature].convert_objects(convert_numeric=True).mean(),1)
    df[feature] = df[feature].fillna(feature_mean)


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)


# In[ ]:


df.columns[df.isna().any()].tolist()


# All double checked, we have no null values and have corrected missing values to mean values for categorical variables, and to 0 for binominal variables. Time to start considering Modeling! 
# 
# The only feature that worries me is the age column. As shown below the distribution is highly right-tailed, with a few very "extreme" values. Will the few very extreme values have an impact on the modeling?

# In[ ]:


figure = plt.figure(figsize=(6,9), dpi=100);    
graph = figure.add_subplot(111);

dfN = df['Age']
freq = pd.value_counts(dfN)
bins = freq.index
x=graph.bar(bins, freq.values) #gives the graph without NaN

plt.ylabel('Frequency')
plt.xlabel('Age')
figure.show()


# In[ ]:


dfN.eq(0).any().any()


# In[ ]:


df[df['Age'] > 58]


# With only 1 positive in the set, we will take the risk and delete these extreme values, they can skew many of our features too easily

# In[ ]:


df['Age'] = np.clip(df['Age'], a_max=58, a_min=None)


# In[ ]:


figure = plt.figure(figsize=(6,9), dpi=100);    
graph = figure.add_subplot(111);

dfN = df['Age']
freq = pd.value_counts(dfN)
bins = freq.index
x=graph.bar(bins, freq.values) #gives the graph without NaN

plt.ylabel('Frequency')
plt.xlabel('Age')
figure.show()


# In[ ]:


category_df = ['Hinselmann', 'Schiller','Citology', 'Biopsy']


# In[ ]:


for feature in categorical_df:
   sns.factorplot(feature,data=df,size=3,kind='count')


# We see some very sparse features in the STDs feature group, we will leave them in for now and investigate later

# In[ ]:


corrmat = df.corr()


# In[ ]:


k = 30 #number of variables for heatmap
cols = corrmat.nlargest(k, 'HormonalContraceptives')['HormonalContraceptives'].index

cm = df[cols].corr()

plt.figure(figsize=(20,20))

sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 12},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# In[ ]:


df = df.round()


# **PART 4: MODELING**

# First the creation of a target variable, one that combines all the positive features of all cancer tests. In reality [target=1] = cancer test performed
# In doing so all modeling can be of a categorical and binary nature, without the confusion of multiple classes.

# In[ ]:


target = df['Hinselmann'] | df['Schiller'] | df['Citology'] | df['Biopsy'] 


# Dropping the specific features that were used to create the target to avoid data leak

# In[ ]:


df = df.drop(columns=['Hinselmann', 'Schiller', 'Citology', 'Biopsy', 'Dx', 'Dx:Cancer', 'Smokes(years)', 'Smokes(packs/year)'])


# Splitting the data into two dataset splits, one is the more traditional split, with the other being a split that is specifically for SMOTE oversampling

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.4, random_state=1) # 60% training and 40% test
X_tr, X_te, y_tr, y_te = train_test_split(df, target, test_size=0.2, random_state=1) # 60% training 20% test for SMOTE()


# In[ ]:


figure = plt.figure(figsize=(2,4), dpi=100);    
graph = figure.add_subplot(111);

df5 = y_train
freq = pd.value_counts(y_train)
bins = freq.index
x=graph.bar(bins, freq.values) #gives the graph without NaN

plt.ylabel('Frequency')
plt.xlabel('Level of Cancer Test Output in Training Set')
figure.show()


# In[ ]:


## oversampling
from imblearn.over_sampling import SMOTE, ADASYN
X_trOVR, y_trOVR = SMOTE(random_state=2).fit_sample(X_tr, y_tr)


# In[ ]:


figure = plt.figure(figsize=(2,4), dpi=100);    
graph = figure.add_subplot(111);

df5 = y_trOVR
freq = pd.value_counts(y_trOVR)
bins = freq.index
x=graph.bar(bins, freq.values) #gives the graph without NaN

plt.ylabel('Frequency')
plt.xlabel('Level of Cancer Test Output in Oversampled Set')
figure.show()


# SMOTE is an oversampling method that creates its own synthetic positive samples from the given data in order to overrepresent positive data examples. SMOTE uses a k-nearest-neighbors algorithm to synthetically create the sample class and has been shown to be especially effective in handwritten character recognition. 

# In[ ]:


#starting with a simple decision tree, attempting to classify the origional data into cancer test or no
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# Then to test the Acccuracy:

# In[ ]:


y_pred = clf.predict(X_test)
print("Accuracy of Decision Tree",metrics.accuracy_score(y_test, y_pred))


# Although we show a fairly good accuracy, let us see how well the model did at picking up "true" cancer cases:

# In[ ]:


import scikitplot as skplt
preds = clf.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=preds)
plt.show()


# Very good, this particular model only missed 10 positive values and was able to effectively "catch" 27 risky patients, we can also visualize the image to inspect the relationships
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Tree") 
dot_data = tree.export_graphviz(clf, out_file=None, filled=True,
                                feature_names=X_test.columns, class_names=["0", "1"])  
graph = graphviz.Source(dot_data)  
graph


# In[ ]:


importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.figure(figsize=(15,20), dpi=100); 
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns)
plt.xlabel('Relative Importance')


# Overall, this method looks rather robust and simple in explanation. How does this model compare when oversamples with the SMOTE method?

# In[ ]:


clf = tree.DecisionTreeClassifier()
clf2 = clf.fit(X_trOVR, y_trOVR)


# In[ ]:


y_pred2 = clf2.predict(X_te)


# In[ ]:


print("Accuracy of Oversampled Decision Tree:",metrics.accuracy_score(y_te, y_pred2))


# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf2, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Tree") 
dot_data = tree.export_graphviz(clf2, out_file=None, filled=True,
                                feature_names=X_test.columns, class_names=["0", "1"])  
graph = graphviz.Source(dot_data)  
graph 


# 

# In[ ]:


preds2 = clf2.predict(X_te)
skplt.metrics.plot_confusion_matrix(y_true=y_te, y_pred=preds2)
plt.show()


# According to our results so far SMOTE() is producing a better model.
# In theory this should not be the case, as SMOTE upsampling this much data should be very difficult and would simply overfit the model

# In[ ]:


#print('Value counts of each target variable:',target.value_counts())
#cancer_df_label = target.astype(int)
#cancer_df_label = cancer_df_label.values.ravel()

#print('Final feature vector shape:',df.shape)
#print('Final target vector shape',cancer_df_label.shape)


# In[ ]:


importances = clf2.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.figure(figsize=(13,17), dpi=100); 
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns)
plt.xlabel('Relative Importance')


# The numerically created variables seem to have a much larger impact on our models thus far. However, there is some multicollinearity here as shown in our heatmaps, and eliminating as many variables as possible should be helpful to our model. Therefore, in order to attempt at a smaller learning sample, a Random Forest feature selection tool is applied, attempting to trim down the variables.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
rand = sel.fit(X_train, y_train)
#rand2 = sel.fit(X_trainOVR, y_trainOVR)


# In[ ]:


selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)


# There are 16 features that are chosen as important by the Random forest model, they are listed below. Limiting the inputs of models allows us to eliminate confusion within the model, as well as trimming the about of variables that the model has to compute.  

# In[ ]:


print(selected_feat)


# In[ ]:


k = 16 #number of variables for heatmap
cols = corrmat.nlargest(k, 'HCSY')['HCSY'].index

cm = df[selected_feat].corr()

plt.figure(figsize=(16,16))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# In[ ]:





# Immediately there are signs for concern, as most important features in the Decision tree are not included
# HCSC, IUDSY, (NPSP is included)
# 
# So we will create a new dataframe that can hold this new trimmed dataset

# In[ ]:


dfFOR = df[selected_feat]


# In[ ]:


dfFOR.shape


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(dfFOR, target, test_size=0.4, random_state=1) # 60% training and 40% test
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(dfFOR, target, test_size=0.2, random_state=1) # 80% training and 20% test


# In[ ]:


## oversampling
X_trOVR2, y_trOVR2 = SMOTE(random_state=2).fit_sample(X_tr2, y_tr2)


# In[ ]:


clf3 = clf.fit(X_train2, y_train2)
clf4 = clf.fit(X_trOVR2, y_trOVR2)


# FOR THE REMAINDER OF THIS KERNAL:
# * X_test, X_train, ect... is the basic data
# * X_test2, X_train2... is the data that used a RF feature selector to trim the data
# * X_testOVR2, trainOVR2 is data that has used SMOTE() oversampling
# * and X_trOVR2, y_trOVR2 is data that has used SMOTE() oversampling and has been trimmed

# In[ ]:


y_pred3 = clf3.predict(X_test2)
y_pred4 = clf4.predict(X_te2)
print("Accuracy of Trimmed Decision Tree:",metrics.accuracy_score(y_test2, y_pred3))
print("Accuracy of Trimmed Oversampled Decision Tree:",metrics.accuracy_score(y_te2, y_pred4))
print("Recall of Trimmed Decision Tree:",metrics.recall_score(y_test2, y_pred3))
print("Recall of Trimmed Oversampled Decision Tree:",metrics.recall_score(y_te2, y_pred4))


# Initial testing shows that the Trimmed Decision tree performed really quite well, the rest of the modeling portion will be to verify that this is the correct model for us to use during deployment or to tune to a further degree.

# In[ ]:


preds3 = clf3.predict(X_test2)
skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=preds3)
plt.title('Trimmed DT')
plt.show()
preds4 = clf4.predict(X_te2)
skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=preds4)
plt.title('Trimmed and Oversampled DT')
plt.show()


# In theory, an ensemble method should be our best bet for data like this (small and difficult), but we don't see a hike in preformance

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clfAB = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clfAB, X_train2, y_train2, cv=5)
scores.mean()  


# In[ ]:


clfABA = clfAB.fit(X_train, y_train)
y_predAB = clfAB.predict(X_test)
print("Accuracy of Origional AdaBoost:",metrics.accuracy_score(y_test, y_predAB))

clfAB2 = clfAB.fit(X_train2, y_train2)
y_predAB2 = clfAB2.predict(X_test2)
print("Accuracy of Trimmed AdaBoost:",metrics.accuracy_score(y_test2, y_predAB2))

clfAB3 = clfAB.fit(X_trOVR2, y_trOVR2)
y_predAB2 = clfAB2.predict(X_te2)
print("Accuracy of Oversampled and Trimmed AdaBoost:",metrics.accuracy_score(y_te2, y_predAB2))


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_predAB)
plt.title('Origional Adaboost')
plt.show()

predAB2 = clfAB2.predict(X_test2)
skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=predAB2)
plt.title('Trimmed Adaboost')
plt.show()

#predAB3 = clfAB3.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=y_predAB2)
plt.title('Adaboost Oversampled and Trimmed')
plt.show()


# AdaBoost did rather well with the raw dataset, and faltered the more data we took from the model. 
# It may an interesting model to choose if we need a more nuanced effort to train for final deployment

# In[ ]:


from sklearn.svm import SVC
clfs = SVC(gamma='auto')
svm = clfs.fit(X_train, y_train) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
sv = svm.predict(X_test)


# In[ ]:


from sklearn.svm import SVC
clfs = SVC(gamma='auto')
svm = clfs.fit(X_train2, y_train2) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svv = svm.predict(X_test2)


# In[ ]:


clfs = SVC(gamma='auto')
svm = clfs.fit(X_trOVR2, y_trOVR2) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svvv = svm.predict(X_te2)


# In[ ]:


print("Accuracy of SVM",metrics.accuracy_score(y_test, sv))
print("Accuracy of SVM Trimmed",metrics.accuracy_score(y_test2, svv))
print("Accuracy of SVM Trimmed",metrics.accuracy_score(y_te2, svvv))


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=sv)
plt.title('SVM Origional')
plt.show()

skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=svv)
plt.title('SVM Trimmed')
plt.show()

skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=svvv)
plt.title('SVM Trimmed and Oversampled')
plt.show()


# Although SVM is a powerful model, it certainly does not seem to fit with the data and was unable to pick up on positive cancer tests. This model, however, does a great job of showing how much accuracy can fool, this model adds no value, but has great accuracy.  Moving on to some other methods

# To determine a true benchmark for performance, cross-validation is going to be the best option. Cross-validations allows the model to run multiple times, fitting to new data, then testing on other data. The preferred method for cross-validation is k-folds, however, but to the size of this dataset and the need to randomly include as many positive class members as possible, a bootstrapping method is used. Bootstrapping allows the model to go get a completely new sample each and every time WITH replacement. So sample 356 that is positive has an equal likelihood of appearing in every training and test sample. After this method has been performed some number of time (I will use 100) all of the models are averaged to gain a true understanding of the robustness of each model. 
# 
# I perform bootstrapped cross-validation with both the trimmed and original dataset for a number of models. 

# In[ ]:


from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import statistics

# load dataset
data = df
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerDT = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperDT = min(1.0, np.percentile(stats, p))
meanDT = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerDT2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperDT2 = min(1.0, np.percentile(stats2, p))
meanDT2 = statistics.mean(stats2)
print('%.3f average recall, confidence interval %.2f%% and %.2f%%' % (meanDT2, lowerDT2*100, upperDT2*100))


# In[ ]:


# load dataset
data = df
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = RandomForestClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerRF = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperRF = min(1.0, np.percentile(stats, p))
meanRF = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanRF, lowerRF*100, upperRF*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerRF2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperRF2 = min(1.0, np.percentile(stats2, p))
meanRF2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanRF2, lowerRF2*100, upperRF2*100))


# In[ ]:


# load dataset
data = dfFOR
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerDTT = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperDTT = min(1.0, np.percentile(stats, p))
meanDTT = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDTT, lowerDTT*100, upperDTT*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerDTT2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperDTT2 = min(1.0, np.percentile(stats2, p))
meanDTT2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDTT2, lowerDTT2*100, upperDTT2*100))


# In[ ]:


from sklearn.linear_model import LogisticRegression
# load dataset
data = dfFOR
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = LogisticRegression(random_state=0, solver='lbfgs',class_weight='balanced')
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerLRT = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperLRT = min(1.0, np.percentile(stats, p))
meanLRT = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLRT, lowerLRT*100, upperLRT*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerLRT2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperLRT2 = min(1.0, np.percentile(stats2, p))
meanLRT2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLRT2, lowerLRT2*100, upperLRT2*100))


# In[ ]:


from sklearn.linear_model import LogisticRegression
# load dataset
data = df
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = LogisticRegression(random_state=0, solver='lbfgs',class_weight='balanced')
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerLR = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperLR = min(1.0, np.percentile(stats, p))
meanLR = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLR, lowerLR*100, upperLR*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerLR2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperLR2 = min(1.0, np.percentile(stats2, p))
meanLR2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLR2, lowerLR2*100, upperLR2*100))


# In[ ]:


from sklearn.svm import LinearSVC
# load dataset
data = df
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerS = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperS = min(1.0, np.percentile(stats, p))
meanSVC = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVC, lowerS*100, upperS*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerS2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperS2 = min(1.0, np.percentile(stats2, p))
meanSVC2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVC2, lowerS2*100, upperS2*100))


# In[ ]:


from sklearn.svm import LinearSVC
# load dataset
data = dfFOR
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerST = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperST = min(1.0, np.percentile(stats, p))
meanSVCT = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVCT, lowerST*100, upperST*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerST2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperST2 = min(1.0, np.percentile(stats2, p))
meanSVCT2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVCT2, lowerST2*100, upperST2*100))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
# load dataset
data = df
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerG = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperG = min(1.0, np.percentile(stats, p))
meanG = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanG, lowerG*100, upperG*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerG2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperG2 = min(1.0, np.percentile(stats2, p))
meanG2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanG2, lowerG2*100, upperG2*100))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
# load dataset
data = dfFOR
values = data.values
# configure bootstrap
n_iterations = 100
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
stats2 = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = np.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	score2 = recall_score(test[:,-1], predictions, average = 'macro')
	#print(score)
	stats.append(score)
	#print(score2)
	stats2.append(score2)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerGT = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperGT = min(1.0, np.percentile(stats, p))
meanGT = statistics.mean(stats)
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanGT, lowerGT*100, upperGT*100))

# plot scores
pyplot.hist(stats2)
pyplot.show()
# confidence intervals
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lowerGT2 = max(0.0, np.percentile(stats2, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperGT2 = min(1.0, np.percentile(stats2, p))
meanGT2 = statistics.mean(stats2)
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanGT2, lowerGT2*100, upperGT2*100))


# In[ ]:


print('Decision Tree')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDT2, lowerDT2*100, upperDT2*100))
print('Decision Tree Trimmed')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDTT, lowerDTT*100, upperDTT*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDTT2, lowerDTT2*100, upperDTT2*100))
print('Random Forest Classifier')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanRF, lowerRF*100, upperRF*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanRF2, lowerRF2*100, upperRF2*100))
print('Random Forest Classifier Trimmed')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDT2, lowerDT2*100, upperDT2*100))
print('Logistic Regression')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLR, lowerLR*100, upperLR*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLR2, lowerLR2*100, upperLR2*100))
print('Logistic Regression Trimmed')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLRT, lowerLRT*100, upperLRT*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLRT2, lowerLRT2*100, upperLRT2*100))
print('SVC')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVC, lowerS*100, upperS*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVC2, lowerS2*100, upperS2*100))
print('SVC Trimmed')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVCT, lowerST*100, upperST*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVCT2, lowerST2*100, upperST2*100))
print('Gradient Boosting')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanG, lowerG*100, upperG*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanG2, lowerG2*100, upperG2*100))
print('Gradient Boosting Trimmed')
print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanGT, lowerGT*100, upperGT*100))
print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanGT2, lowerGT2*100, upperGT2*100))


# Our best performing model is.......
# 
# 
# 
# a simple decision tree.....
# While rather unclimactic, it is good to have validation that this modeling method is both able to stand up to some other pretty advanced methods, as well as being able to fit a variety of samples and still perform. Gradient boosting did offer some good numbers, but the great benifit of choosing a simple model over a more complex one will make life easier within deployment.

# **PART 5: DEPLOYMENT**

# In order to get at the truth behind this model, we are going to examine the strength of the features against each other and hopefully glean what our business objective is:
# a better understanding of what causes cervical cancer and how strong each risk factor is.

# In[ ]:


clf = tree.DecisionTreeClassifier()
clf2 = clf.fit(X_train2, y_train2)

importances = clf2.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), dfFOR.columns)
plt.xlabel('Relative Importance')


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

permumtation_impor = PermutationImportance(clf2, random_state=2019).fit(X_test2, y_test2)
eli5.show_weights(permumtation_impor, feature_names = X_test2.columns.tolist())


# As we begin to inspect feature importance, it is easiest to simply look at our decision tree. The beauty of such a simple model is contained in the simplicity of understanding. However, for other models something involving partial dependancy plots (each variable moving alone and its impact on the model output) may be more insightful. 
# We will keep the ranking of feature importances, but use a better model type to examine partial depandancies. 

# In[ ]:


from pdpbox import pdp, get_dataset, info_plots
random_forest = RandomForestClassifier(n_estimators=500, random_state=2019).fit(X_test2, y_test2)
def pdpplot( feature_to_plot, pdp_model = clf2, pdp_dataset = X_test2, pdp_model_features = X_test2.columns):
    pdp_cancer = pdp.pdp_isolate(model=pdp_model, dataset=pdp_dataset, model_features=pdp_model_features, feature=feature_to_plot)
    fig, axes = pdp.pdp_plot(pdp_cancer, feature_to_plot, figsize = (13,6),plot_params={})
     #_ = axes['pdp_ax'].set_ylabel('Probability of Cancer')
    
pdpplot('CNT')
pdpplot('PILL')
pdpplot('TSP')
pdpplot('YAFSI')
pdpplot('YAHC')
pdpplot('SSY')
pdpplot('SEX')
pdpplot('SexZ')
pdpplot('HormonalContraceptives(years)')
pdpplot('YAIUD')
pdpplot('FirstSexZ')
pdpplot('Numofpregnancies')
pdpplot('YAIUD')

plt.show()


# Features such as CNT, PILL, YAFSI, YAHC, and SEX all show that there is something to be learned from each feature itself. 
# As the CNT of features goes up, we see an increaced risk of cancer. PILL, suprisingly, seems to trend in the opposite direction. 
# As more data becomes available, features may be able to be studied like this. 

# For our final deployment we look at our decision tree with trimmed features:

# In[ ]:


#Remember the trimmed data set
#X_train2, X_test2, y_train2, y_test2 = train_test_split(dfFOR, target, test_size=0.4, random_state=1) # 60% training and 40% test

clf = tree.DecisionTreeClassifier()

clfFINAL = clf.fit(X_train2, y_train2)
y_pred = clfFINAL.predict(X_test2)

import graphviz 
dot_data = tree.export_graphviz(clfFINAL, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Tree") 
dot_data = tree.export_graphviz(clfFINAL, out_file=None, filled=True,
                                feature_names=X_test2.columns, class_names=["0", "1"])  
graph = graphviz.Source(dot_data)  
graph


# In[ ]:


predicted_probas = clfFINAL.predict_proba(X_te2)
skplt.metrics.plot_cumulative_gain(y_te2, predicted_probas)
plt.show()
print("Accuracy of Decision Tree",metrics.accuracy_score(y_test2, y_pred))
print("Recall of Decision Tree",metrics.recall_score(y_test2, y_pred))


# This is the final decision tree that will be left to production. Following a relatively simple set of questions or steps, this model is able to correctly identify cases where a cancer diagnosis is highly likely or times where a patient should be tested. Although not groundbreaking, this simple tool could be deployed to an API, able to be used by healthcare professionals using records they are already keeping. With some integration, a warning score could be displayed at the top of a patient's chart, giving statistical odds of danger. With more work and more data, there could be a steady improvement of this algorithm over time after deployment. Overall the lack of positive data is a concern. One that may be addressed at a later point with more data. 
