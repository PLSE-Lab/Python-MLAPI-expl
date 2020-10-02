#!/usr/bin/env python
# coding: utf-8

# # A classification analysis on suicide data
# 
#  For this exploration, I used the suicide rate data set from Kaggle https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016. 
# 
#  "This compiled dataset pulled from four other datasets linked by time and place, and was built to find signals correlated to increased suicide rates among different cohorts globally, across the socio-economic spectrum."
# 
#  This data set (I call it the "originaldata") contains a combination of numerical and categorical features:
# 
#  **categorical**
#  * Country
#  * Year: 1985 to 2016
#  * Sex: Male/female 
#  * Age: Five age groups 
#  * Generation
# 
#  **Numerical**
#  * Population
#  * Number of Suicides: Suicide incidences 
#  * Suicides per 100k people:normalized version of suicide incidences
#  * GDP for year: Gross Domestic Product(a measure of economic development)
#  * GDP per capita for year: GDP/population
#  * HDI for year : Human Development Index
# 
#  I was interested in exploring the possible causes that might increase the risk of suicide in societies, using different machine learning algorithms. Therefore, I decided to enrich the data set by adding more features from the World Development Indicator database to the original suicide data ("extra_data")
# 

# In[ ]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1- Importing the original suicide data set and renaming the columns

# In[ ]:


originaldata = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

originaldata.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population','suicidesper100k',
                      'country-year', 'yearlyHDI', 'GDPpyear', 'GDPpcapita', 'generation']

originaldata.head()


# In[ ]:


#fixing and cleaning the original data  
originaldata['GDPpyear'] = originaldata.apply(lambda x: float(x['GDPpyear'].replace(',', '')), axis=1)
originaldata.sex.astype('category')


# 2- Importing the suicide data set with extra features added from WDI database

# In[ ]:


extra_data = pd.read_csv('/kaggle/input/widandsuicide/suicidedataextrafestures.csv')

extra_data.columns = [
    'country', 'year', 'sex', 'age', 'suicides_no', 'population','suicidesper100k', 'country-year', 'yearlyHDI',
    'GDPpyear', 'GDPpcapita', 'generation', 'suicide%', 'Internetusers', 'Expenses', 'employeecompensation',
    'Unemployment', 'Physiciansp1000', 'Legalrights', 'Laborforcetotal', 'Lifeexpectancy', 'Mobilesubscriptionsp100',
    'Refugees', 'Selfemployed', 'electricityacess', 'secondarycompletion']


# In[ ]:


extra_data.head()


# # Countries
#  The original data set contains data from more than 100 countries during 1985-2016. I've decided to limit the number of countries to make the analysis more insightful. I've chosen 40 countries from different parts of the world that I believe are a good sample of the different regions.

# In[ ]:


countrynames = [
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Belgium',
    'Brazil',
    'Bulgaria',
    'Canada',
    'Chile',
    'Colombia',
    'Croatia',
    'Cuba',
    'Czech Republic',
    'Denmark',
    'Finland',
    'France',
    'Germany',
    'Greece',
    'Hungary',
    'Iceland',
    'Ireland',
    'Israel',
    'Italy',
    'Japan',
    'Mexico',
    'Netherlands',
    'New Zealand',
    'Norway',
    'Poland',
    'Portugal',
    'Romania',
    'Russian Federation',
    'South Africa',
    'Spain',
    'Sweden',
    'Switzerland',
    'Thailand', 
    'Turkmenistan',
    'Ukraine',
    'United Kingdom', 
    'United States']


# Also, the extra data from the WDI database was only available for 1995-2013, so I decided to limit the data to this time period.

# In[ ]:


df1 = extra_data.copy()
df = df1.iloc[np.where(df1.country == countrynames[0])]
for i, x in enumerate(countrynames[1:]):
    df = df.append(df1.iloc[np.where(df1.country == x)])

df = df[df.year >= 1995]
df = df[df.year <= 2013]


# # Exploratory Data Analysis
# 

# In[ ]:


col = plt.cm.Spectral(np.linspace(0, 1, 20))

plt.figure(figsize=(8, 6))

agedistf = pd.DataFrame(df.groupby('sex').get_group('female').groupby('age').suicides_no.sum())

agedistm = pd.DataFrame(df.groupby('sex').get_group('male').groupby('age').suicides_no.sum())

plt.bar(agedistm.index, agedistm.suicides_no, color=col[18])
plt.bar(agedistf.index, agedistf.suicides_no, color=col[7])
plt.legend(['male', 'female'], fontsize=16)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Suicides per 100K', fontsize=14)


# In[ ]:


plt.figure(figsize=(12, 15))


plt.subplot(211)
df.groupby(['country']).suicidesper100k.mean().nlargest(10).plot(kind='barh', color=col)
plt.xlabel('Average Suicides/100k', size=20)
plt.ylabel('Country', fontsize=20)
plt.title('Top 10 countries', fontsize=30)

plt.subplot(212)
df.groupby(['country']).suicides_no.mean().nlargest(10).plot(kind='barh', color=col)
plt.xlabel('Average Suicides_no', size=20)
plt.ylabel('Country', fontsize=20);


# In[ ]:


plt.figure(figsize=(10, 16))

plt.subplot(311)

sns.barplot(x='sex', y='population', hue='age', data=df, palette="Blues")
plt.xticks(ha='right', fontsize=20)
plt.ylabel('Population', fontsize=20)
plt.xlabel('Sex', fontsize=20)
plt.legend(fontsize=14, loc='best')

plt.subplot(313)

sns.barplot(x='sex', y='suicidesper100k', hue='age', data=df,palette="Blues")
plt.xticks(ha='right', fontsize=20);
plt.ylabel('suicidesper100k',fontsize=20);
plt.xlabel('Sex',fontsize=20);
plt.legend(fontsize=14);

plt.subplot(312)
sns.barplot(x='sex', y='suicides_no', hue='age', data=df, palette="Blues")
plt.xticks(ha='right', fontsize=20)
plt.ylabel('suicides incidences', fontsize=20)
plt.xlabel('Sex', fontsize=20)
plt.legend(fontsize=14)


# In[ ]:


year = originaldata.groupby('year').year.unique()

plt.figure(figsize=(6, 5))

totalpyear = pd.DataFrame(originaldata.groupby('year').suicides_no.sum())

plt.plot(year.index[0:31], totalpyear[0:31], color=col[18])
plt.xlabel('year', fontsize=14)
plt.ylabel('Total number of suicides in the world', fontsize=14)


# In[ ]:


plt.figure(figsize=(20, 8))
plt.subplot(121)
plt.hist(df.suicidesper100k, bins=30, color=col[18])
plt.xlabel('Suicides per 100K of population', fontsize=14)
plt.ylabel('count', fontsize=14)

plt.subplot(122)
plt.hist(df.GDPpcapita, bins=30, color=col[7])
plt.xlabel('GDP', fontsize=14)
plt.ylabel('count', fontsize=14)


# From all the extra features added from the WDI database, I'm using 11 features for exploring. These are basically the features with fewer null values.
# 
# totalfeatures: 
# 'country', 'year','GDPpyear','GDPpcapita','employeecompensation','Unemployment',
#  'Lifeexpectancy','Refugees','Selfemployed','Internetusers','population'.
#       
#  I'm aslo lifting the fine subsample level of sex and age group. I'm creating a new pandas data frame "total", with a new column "suicides" as the total value of Suicides per 100k of population per country per year, calculated by summing over both sexes and age groups.

# In[ ]:


features = ['country', 'year', 'GDPpyear', 'GDPpcapita', 'employeecompensation', 'Unemployment',
            'Lifeexpectancy', 'Refugees', 'Selfemployed', 'Internetusers']

total = df[features].groupby('country').get_group(countrynames[0]).groupby('year').mean()
total['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('country').get_group(countrynames[0]).groupby('year').sum()
total['population'] = df[['country', 'year', 'population']].groupby('country').get_group(countrynames[0]).groupby('year').sum()

total['country'] = countrynames[0]

for i, x in enumerate(countrynames[1:]):
    suicides = df[features].groupby('country').get_group(x).groupby('year').mean()
    suicides['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('country').get_group(x).groupby('year').sum()
    total['population'] = df[['country', 'year', 'population']].groupby('country').get_group(x).groupby('year').sum()
  
    suicides['country'] = x
    total = total.append(suicides)

total.reset_index(inplace=True)
sort = True


# In[ ]:


totalfeatures = ['country', 'year', 'GDPpyear', 'GDPpcapita', 'employeecompensation', 'Unemployment',
                 'Lifeexpectancy', 'Refugees', 'Selfemployed', 'Internetusers', 'population']


# In[ ]:


plt.figure(figsize=(20, 8))
plt.subplot(121)
sns.distplot(total.Suicides, bins=15)
plt.xlabel('total Suicides (summed over sex and age group) per 100K of population', fontsize=14)

plt.subplot(122)
plt.hist(total.GDPpcapita, bins=30, color=col[7])
plt.xlabel('GDP', fontsize=14)


# In[ ]:


plt.figure(figsize=(8, 5))

suicides = df[['year', 'GDPpyear', 'Selfemployed', 'Unemployment', 'Lifeexpectancy']].groupby('year').mean()
suicides['Suicides'] = df[['country', 'year', 'suicidesper100k']].groupby('year').sum()

plt.plot(suicides.index, suicides.GDPpyear/suicides.GDPpyear.max(), color=col[1])
plt.plot(suicides.index, suicides.Unemployment/suicides.Unemployment.max(), color=col[7])
plt.plot(suicides.index, suicides.Lifeexpectancy/suicides.Lifeexpectancy.max(), color=col[15])
plt.plot(suicides.index, suicides.Suicides/suicides.Suicides.max(), color=col[17])
plt.legend(['global average GDPpyear', 'global average Unemployment', 'global average Life expectancy', 'Total suicides per 100k'], fontsize=14, loc='best')
plt.ylabel('Normalized', fontsize=14)
plt.xlabel('year', fontsize=14)


# # Correlations
# # Calculating the correlation matrix for all chosen features:

# In[ ]:


corr = total.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=0.2, cbar_kws={"shrink": 0.8});


# The feature "Internetusers" is strongly correlated with GDP features, as expected. "Lifeexpectancy" also shows strong corrlaiton with GDP per capital. This should be noted when evaluating feature importances from model outputs. Other features do not show strong correlations.

# # Data preprocessing 
#  Cleaning the data, replacing null values with appropriate replacements
# 

# In[ ]:


# Cleaning the data, replacing null values with appropriate replacements

total.Internetusers.fillna(total.Internetusers. min(), inplace=True)
total.Refugees.fillna(8, inplace=True)
total.employeecompensation.fillna(total.employeecompensation.mean(), inplace=True)
total.population.fillna(total.population.mean(), inplace=True)


# # Binary classification for suicide risk
#  I decided to perform a binary classification on the suicide data, by assigning risk classes based on the suicide incidences per 100k of the population as high/low suicide risk. Adding an extra column to the "total" data frame as 'risk'. 
#  * Suicides<mean(Suicides) --> low risk --> class 0
#  *   Suicides>mean(Suicides) --> high risk --> class 1

# In[ ]:


total['risk'] = total.Suicides.copy()

total['risk'] = np.where(total.risk < total.Suicides.mean(), 0, 1)


# In[ ]:


plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.hist(total.risk, color=col[6])
plt.ylabel('counts', fontsize=14)
plt.xlabel('Suicide risk', fontsize=14)

plt.subplot(122)
sns.distplot(total.Suicides[total.risk == 0], bins=10)
sns.distplot(total.Suicides[total.risk == 1], bins=20)  
plt.xlabel('Suicides', fontsize=14)


# In[ ]:


# Label encoding countries

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

total.country = le.fit_transform(total.country)  # Alphabetic order [0:40]
total.country.unique()


# In[ ]:


# Preparing data for modeling

X = np.asarray(total[totalfeatures])
y = np.asarray(total['risk'])


# Applying standard scaler on data, since ML algorithms work with the assumption that the data is normally distributed

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[ ]:


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# In[ ]:


ax1 = total[total['risk'] == 1][0:200].plot(kind='scatter', x='GDPpcapita', y='employeecompensation', color='DarkRed',
                                            label='high risk', figsize=(6, 5), fontsize=12)
total[total['risk'] == 0][0:200].plot(kind='scatter', x='GDPpcapita', y='employeecompensation', color='LightGreen',
                                      label='low risk', ax=ax1)

plt.ylabel('employeecompensation', fontsize=16)
plt.xlabel('GDP per capita', fontsize=16)
plt.legend(fontsize=14)


ax1 = total[total['risk'] == 1][0:200].plot(kind='scatter', x='Lifeexpectancy', y='Selfemployed', color='DarkRed',
                                            label='high risk', figsize=(6, 5), fontsize=12)
total[total['risk'] == 0][0:200].plot(kind='scatter', x='Lifeexpectancy', y='Selfemployed', color='LightGreen',
                                      label='low risk', ax=ax1);

plt.ylabel('Selfemployed', fontsize=16)
plt.xlabel('Lifeexpectancy', fontsize=16)
plt.legend(fontsize=14)


ax1 = total[total['risk'] == 1][0:200].plot(kind='scatter', x='GDPpcapita', y='Unemployment', color='DarkRed',
                                            label='high risk', figsize=(6, 5), fontsize=12);
total[total['risk'] == 0][0:200].plot(kind='scatter', x='GDPpcapita', y='Unemployment', color='LightGreen',
                                     label='low risk', ax=ax1);

plt.ylabel('Unemployment', fontsize=16)
plt.xlabel('GDP per capita', fontsize=16);
plt.legend(fontsize=14);


# In[ ]:


fig = plt.figure(figsize=(30, 30))

j = 0
for i, x in enumerate(total.columns[0:11]):
    plt.subplot(4, 3, j+1)
    j += 1
    sns.distplot(total[x][total.risk == 0], label='low risk')
    sns.distplot(total[x][total.risk == 1], label='high risk')       
    plt.legend(loc='best', fontsize=18)  
    plt.xlabel(x, fontsize=18)


# # Model development 
#  I'm using four different classification methods, optimize each, compare the performance of models and extract the feature importance for each model. Based on the EDA results, I expect nonlinear classification to perform better on this data. 
# 
# models explored in this work are:
#  1. Logistic regression
#  2. Decision tree
#  3. Random forest
#  4. MLP classifiere (Neural Network)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report


LR = LogisticRegression(C=0.001, solver='liblinear').fit(X_train, y_train)

yLRhat = LR.predict(X_test)

yLRhat_prob = LR.predict_proba(X_test)


print('precision_recall_fscore_support', precision_recall_fscore_support(y_test, yLRhat, average='weighted'))

cm = confusion_matrix(y_test, yLRhat)
print('\n confusion matrix \n', cm)

print('classification report for Logistic Regression\n', classification_report(y_test, yLRhat))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


DT = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_leaf_nodes=30)
DT = DT.fit(X_train, y_train)
ydthat = DT.predict(X_test)

print('******************Decision Tree classifier**************')

print('Accuracy =', DT.score(X_test, y_test))
print('Train Accuracy=', DT.score(X_train, y_train))
print('CM\n', confusion_matrix(y_test, ydthat))
print('classification report for decision tree\n', classification_report(y_test, ydthat))
print('# of leaves', DT.get_n_leaves(), '\n Depth', DT.get_depth())


DTfeat_importance = DT.feature_importances_
DTfeat_importance = pd.DataFrame([totalfeatures, DT.feature_importances_]).T


print(DTfeat_importance.sort_values(by=1, ascending=False))
print('\n# of features= ', DT.n_features_)


# # Model performance optimization
# # shown for the decision tree classifier only.

# In[ ]:


# USing Area under curve of ROC curve as the metric. This shows how much our classification is better than just
# randomly chosen classes

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ydthat)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []

for max_depth in max_depths: 
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    
plt.figure(figsize=(6, 5))
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'DarkRed', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'DarkBlue', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, fontsize=14)
plt.ylabel('AUC score', fontsize=14)
plt.xlabel('Tree depth', fontsize=14)
plt.show()


# In[ ]:


max_leaf_nodes = np.linspace(3, 33, 31, endpoint=True).astype(int)
train_results = []
test_results = []

for max_leaf_nodes in max_leaf_nodes: 
    dt2 = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, max_depth=7)
    dt2.fit(X_train, y_train)
    train_pred = dt2.predict(X_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results
    train_results.append(roc_auc)

    y_pred = dt2.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    
plt.figure(figsize=(6, 5))
    
line1, = plt.plot(np.linspace(3, 33, 31, endpoint=True), train_results, 'DarkRed', label='Train AUC')
line2, = plt.plot(np.linspace(3, 33, 31, endpoint=True), test_results, 'DarkBlue', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, fontsize=14)
plt.ylabel('AUC score', fontsize=14)
plt.xlabel('Max leaf nodes', fontsize=14)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=2, min_samples_leaf=5,
                                       max_leaf_nodes=20, max_features=len(totalfeatures)) 

random_forest.fit(X_train, y_train)

yrfhat = random_forest.predict(X_test)
feat_importance = random_forest.feature_importances_
rffeat_importance = pd.DataFrame([totalfeatures, random_forest.feature_importances_]).T

print('******************Random forest classifier**************')
print('Accuracy on training data', random_forest.score(X_train, y_train))
print('Accuracy on test data', random_forest.score(X_test, y_test))
print('CM\n', confusion_matrix(y_test, yrfhat))
print('Classification report for random forest\n', classification_report(y_test, yrfhat))
print(rffeat_importance.sort_values(by=1, ascending=False))


# In[ ]:


from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-4,
                   hidden_layer_sizes=(4, 4), random_state=4)

NN.fit(X_train, y_train)
y_predict = NN.predict(X_test)
cmMLP = confusion_matrix(y_test, y_predict)


print("Training set score:", NN.score(X_train, y_train))
print("Test set score:", NN.score(X_test, y_test))
print('minimum loss achived=', NN.loss_)
print('confusion matrix for MLPclassifier from scikit learn\n', cmMLP)
print('classification reportfor MLPclassifier from scikit learn\n', classification_report(y_test, y_predict))


# In[ ]:


import sklearn
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
sorted(sklearn.metrics.SCORERS.keys())


def score(X, y):
    
    y_pred = NN.predict(X)
 
    return accuracy_score(y, y_pred)

base_score, score_decreases = get_score_importances(score, X_test, y_test)
feature_importances = np.mean(score_decreases, axis=0)

NNfeatureimportance = pd.DataFrame(totalfeatures, feature_importances)
NNfeatureimportance.reset_index(inplace=True)
NNfeatureimportance.columns = ['importance', 'feature']
NNfeatureimportance.sort_values(by='importance', ascending=False)


# # Classification performance comparison/ROC plot

# In[ ]:


models = [LR, NN, DT, random_forest]
modelnames = ['Logistic regression', 'MLP classifier NN', 'Random Forest', 'Decison tree']


for i, x in enumerate(models):
    
    y_true = y_test
    y_probas = x.predict_proba(X_test)
    ax1 = skplt.metrics.plot_roc(y_true, y_probas, plot_micro=False, plot_macro=True, classes_to_plot=[], figsize=(5, 5))
    plt.axis((-0.01, 1, 0, 1.1))
    plt.legend([modelnames[i]], loc='best')


# # k-fold cross validation to minimize overfitting

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
rfscores = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy')

rfpredictions = cross_val_predict(random_forest, X_train, y_train)

print('CM for random forest with cross validation\n', confusion_matrix(y_train, rfpredictions))
print('classification report for random forest with CV \n', classification_report(y_train, rfpredictions))

dtscores = cross_val_score(DT, X_train, y_train, cv=5, scoring='accuracy')
DTpredictions = cross_val_predict(DT, X_train, y_train)

print('CM for Decision tree with cross validation\n', confusion_matrix(y_train, DTpredictions))
print('classification report for Decision tree with CV \n', classification_report(y_train, DTpredictions))

nnscores = cross_val_score(NN, X_train, y_train, cv=5, scoring='accuracy')
NNpredictions = cross_val_predict(NN, X_train, y_train)

print('CM for MLP classifier  with cross validation\n', confusion_matrix(y_train, NNpredictions))
print('classification report for MLP classifier with CV \n', classification_report(y_train, NNpredictions))


# # Feature importance results summary
# The bar plot summarizes the feature importances from  different classification results. But what are the true causes? There are some overlaps in feature importances between different classification methods, and there are also some disagreements. This is why one can not trust the results of a single model. As scientists, we should learn not to panic when encountering controversies. Controversies are indeed built in to the scientific method. Instead, we should try to explore other resources and feed more insight into the picture we are exploring. The truth is all models are wrong, but some are useful!

# In[ ]:


print('Feature importance results for the three best models')
print('random forest accuracy score (5-fold cross validation)=', rfscores.mean(), '+/-', rfscores.std()*2)

plt.figure(figsize=(23, 4))
plt.bar(rffeat_importance[0], rffeat_importance[1], color=col[2], width=0.4,)
plt.xticks(ha='right', rotation=30, fontsize=15)
plt.legend(['Random Forest classifier acc = % f'% rfscores.mean()], fontsize=14)

print('Decison Tree accuracy score (5-fold cross validation)=', dtscores.mean(), '+/-', dtscores.std()*2)
plt.figure(figsize=(23, 4))
plt.bar(DTfeat_importance[0], DTfeat_importance[1], color=col[14], width=0.4)
plt.legend(['Desicion Tree cassifier acc = % f'% dtscores.mean()], fontsize=14)
plt.xticks(ha='right', rotation=30, fontsize=15)

print('MLP classifier accuracy score (5-fold cross validation)=', nnscores.mean(), '+/-', nnscores.std()*2)
plt.figure(figsize=(23, 4))
plt.bar(NNfeatureimportance['feature'], NNfeatureimportance['importance'], color=col[18], width=0.4)
plt.legend(['MLP classifier Neural net acc = % f'% nnscores.mean()], fontsize=14);
plt.xticks(ha='right', rotation=30, fontsize=15);


# In[ ]:




